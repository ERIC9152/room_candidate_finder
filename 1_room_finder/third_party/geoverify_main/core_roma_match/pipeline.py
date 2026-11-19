import os
import cv2
import numpy as np
import tempfile
from PIL import Image

from core_roma_match.matchers import MatcherFactory
from core_roma_match.clustering import DBSCANClusterManager, KMeansClusterManager
from core_roma_match.proection import FOVOptimizer
from core_roma_match.refiner import LocalRefiner
from core_roma_match.visualization import visualize_slice_matches

class ImageHandler:
    """處理影像載入與中文字元路徑相容性 (SRP: Image I/O)"""
    @staticmethod
    def load_cv2(path):
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

    @staticmethod
    def get_info(path):
        with Image.open(path) as img:
            return {"w": img.size[0], "h": img.size[1]}

# =======================================================================
# Main Pipeline Orchestrator
# =======================================================================

class RoMaScanPipeline:
    def __init__(self, model_type="roma_v1", batch_bidir=False):
        self.model_type = model_type
        self.batch_bidir = batch_bidir
        self.matcher = None

    def execute(self, img_a, img_b, threshold=0.1, max_clusters=5, export_dir=None, enable_poles_match=False, local_conf_threshold=0.65, cluster_method="kmeans", fov_mode="adaptive", fixed_fov=90, draw_lines=True, line_percent=0.05, line_alpha=0.5):
        if export_dir and not os.path.exists(export_dir): os.makedirs(export_dir)
        info = ImageHandler.get_info(img_a)

        # 延遲初始化 Matcher 以節省資源並支援 model_type 選擇
        if self.matcher is None:
            self.matcher = MatcherFactory.create(model_type=self.model_type, conf_threshold=0.1, batch_bidir=self.batch_bidir)

        # 1. 全局匹配 (恢復 sample_count=5000 以確保連通性評分的精確度)
        # 注意：全局匹配是直接吃 ERP 全景圖，所以必須關閉 MAGSAC
        g_matches, _ = self.matcher.match(img_a, img_b, conf_threshold=0.1, sample_count=5000, use_magsac=False)
        g_matches = np.array([m for m in g_matches if m[4] >= threshold])

        # [DEBUG] 診斷座標順序 (x,y vs y,x)
        # if len(g_matches) > 0:
        #     print(f"[DEBUG] Sample match: xa={g_matches[0][0]:.1f}, ya={g_matches[0][1]:.1f} | "
        #           f"Expected ranges: x∈[0,{info['w']}], y∈[0,{info['h']}]")

        # 將原始全局匹配點存入 info，供下游連通性模型使用
        info["global_matches"] = g_matches.tolist() if len(g_matches) > 0 else []

        if len(g_matches) == 0: return [], info

        # 2. 分群策略選擇
        #    對 conf 施加 cos(lat) 緯度權重，降低極區假匹配對分群的吸引力。
        #    原始 g_matches 保持不動 (已存入 info["global_matches"])，只影響分群輸入。
        g_for_cluster = g_matches.copy()
        H = info["h"]
        lat_w_a = np.sin(np.pi * g_for_cluster[:, 1] / H)  # ya 的緯度權重: 赤道=1, 極區→0
        lat_w_b = np.sin(np.pi * g_for_cluster[:, 3] / H)  # yb 的緯度權重
        g_for_cluster[:, 4] *= (lat_w_a * lat_w_b)          # 兩端都離極區才保留高權重

        if cluster_method == "kmeans":
            cm = KMeansClusterManager(info["w"], n_clusters=max_clusters)
            clusters = cm.run(g_for_cluster)
        else:
            cm = DBSCANClusterManager(info["w"])
            clusters = cm.merge(cm.run(g_for_cluster))

        if not clusters: return [], info

        # 3. 排序與限額
        for c in clusters: c['score'] = len(c['points']) * np.mean(c['points'][:, 4])
        clusters = sorted(clusters, key=lambda x: x['score'], reverse=True)[:max_clusters]

        # 將群中心資訊存入 info
        info["cluster_centers"] = [
            {"center_a": c['center_a'].tolist(), "center_b": c['center_b'].tolist(), "id": c['id']}
            for c in clusters
        ]

        # 4. 局部細化：批次 GPU 推論
        img_a_cv, img_b_cv = ImageHandler.load_cv2(img_a), ImageHandler.load_cv2(img_b)
        refiner = LocalRefiner(self.matcher, export_dir,
                               draw_lines=draw_lines, line_percent=line_percent, line_alpha=line_alpha)
        all_matches = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            # ── Phase A: CPU 裁切（生成所有透視圖 + map_back）──
            crop_pairs = []   # [(pil_a, pil_b), ...]
            map_backs = []    # [(mb_a, mb_b), ...]
            cv_crops = []     # [(p_a_cv, p_b_cv, cluster, fa, fb), ...] for export

            for c in clusters:
                if fov_mode == "fixed":
                    fa, fb = fixed_fov, fixed_fov
                else:
                    fa, fb = FOVOptimizer.calculate(c, info["w"], info["h"])
                pil_a, pil_b, mb_a, mb_b, p_a_cv, p_b_cv = refiner.prepare_crops(
                    img_a_cv, img_b_cv, c, tmp_dir, fa, fb)
                crop_pairs.append((pil_a, pil_b))
                map_backs.append((mb_a, mb_b))
                cv_crops.append((p_a_cv, p_b_cv, c, fa, fb))

            # Poles（如啟用）
            if enable_poles_match:
                for pole, y in [('top', 0), ('bot', info['h'])]:
                    c = {'id': pole, 'center_a': (info['w']/2, y), 'center_b': (info['w']/2, y)}
                    pil_a, pil_b, mb_a, mb_b, p_a_cv, p_b_cv = refiner.prepare_crops(
                        img_a_cv, img_b_cv, c, tmp_dir, 120, 120)
                    crop_pairs.append((pil_a, pil_b))
                    map_backs.append((mb_a, mb_b))
                    cv_crops.append((p_a_cv, p_b_cv, c, 120, 120))

            # ── Phase B: GPU 批次匹配（1 次 forward 取代 N 次）──
            if crop_pairs and hasattr(self.matcher, 'match_batch'):
                batch_results = self.matcher.match_batch(
                    crop_pairs, conf_threshold=local_conf_threshold)
            else:
                # Fallback: 逐一匹配（V1 或無 match_batch 時）
                batch_results = []
                for pil_a, pil_b in crop_pairs:
                    m, _ = self.matcher.match(pil_a, pil_b, conf_threshold=local_conf_threshold)
                    batch_results.append((m, _))

            # ── Phase C: CPU 座標反投影 + export ──
            for k, ((matches, _), (mb_a, mb_b)) in enumerate(zip(batch_results, map_backs)):
                # Export 視覺化
                if export_dir and matches:
                    p_a_cv, p_b_cv, c, fa, fb = cv_crops[k]
                    vis = visualize_slice_matches(p_a_cv, p_b_cv, matches,
                                                  draw_lines=draw_lines,
                                                  line_percent=line_percent,
                                                  line_alpha=line_alpha)
                    cv2.imwrite(os.path.join(export_dir, f"c_{c['id']}_f{int(fa)}_{int(fb)}.jpg"), vis)

                all_matches.extend(LocalRefiner.apply_map_back(matches, mb_a, mb_b))

        # 5. 去重 (Duplicate Filter)
        final, v_a, v_b = [], set(), set()
        for m in sorted(all_matches, key=lambda x: x[4], reverse=True):
            ka, kb = (int(round(m[0])), int(round(m[1]))), (int(round(m[2])), int(round(m[3])))
            if ka not in v_a and kb not in v_b:
                v_a.add(ka); v_b.add(kb); final.append(m)

        return final, info

    def execute_batch(self, pairs_list, threshold=0.1, max_clusters=5, local_conf_threshold=0.65, cluster_method="kmeans", fov_mode="adaptive", fixed_fov=90, export_dir=None, enable_poles_match=False, draw_lines=True, line_percent=0.05, line_alpha=0.5):
        """批次匹配模式：逐對呼叫 execute()，確保每一對的隨機狀態消耗順序
        與 batch_size=1 完全一致。

        批次加速的價值在外層的 cache-aware 設計 (match_core_batch)，
        呼叫順序維持原樣。RoMa V2 的 model.sample() 使用 KDE 隨機取樣，
        若改變 global_match 與 refine 的交錯順序，會導致相同 pair
        在不同 batch_size 下產出不同結果。
        """
        if not pairs_list: return []

        results = []
        for img_a, img_b in pairs_list:
            matches, info = self.execute(
                img_a, img_b,
                threshold=threshold,
                max_clusters=max_clusters,
                export_dir=export_dir,
                enable_poles_match=enable_poles_match,
                local_conf_threshold=local_conf_threshold,
                cluster_method=cluster_method,
                fov_mode=fov_mode,
                fixed_fov=fixed_fov,
                draw_lines=draw_lines,
                line_percent=line_percent,
                line_alpha=line_alpha,
            )
            results.append((matches, info))
        return results

def RomaScan(*args, **kwargs):
    # 支援傳入 model_type 和 batch_bidir
    model_type = kwargs.pop('model_type', 'roma_v1')
    batch_bidir = kwargs.pop('batch_bidir', False)
    kwargs.pop('depth_variance', None)
    kwargs.pop('depth_variance_filter', None)
    return RoMaScanPipeline(model_type=model_type, batch_bidir=batch_bidir).execute(*args, **kwargs)
