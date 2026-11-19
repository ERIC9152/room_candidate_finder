"""
[影像匹配模組] 基於 RoMa 的全景圖匹配系統 (支援 Cubemap 與 Direct ERP 模式)
"""
import os
import torch
import random
import numpy as np

# ==========================================
# Reproducibility Configuration
# ==========================================
def fix_all_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"  [Reproducibility] Global random seed fixed to: {seed}")

import hashlib
from PIL import Image
import numpy as np
try:
    from utils_cubemap import CubemapProjector
except ImportError:
    from roma_match.utils_cubemap import CubemapProjector

# Model imports are handled by MatcherFactory (lazy, only loads V1 or V2 as needed).

# core_roma_match 位於本專案根目錄底下。
from core_roma_match.interface import run_roma_match

def batch_cubemap_to_erp(face_name, x_arr, y_arr, erp_w, erp_h, face_size):
    u = 2 * (x_arr + 0.5) / face_size - 1
    v = -(2 * (y_arr + 0.5) / face_size - 1)
    zeros, ones = np.zeros_like(u), np.ones_like(u)
    if face_name == 'front':   vx, vy, vz = u, v, ones
    elif face_name == 'right': vx, vy, vz = ones, v, -u
    elif face_name == 'back':  vx, vy, vz = -u, v, -ones
    elif face_name == 'left':  vx, vy, vz = -ones, v, u
    elif face_name == 'top':   vx, vy, vz = u, ones, -v
    elif face_name == 'bottom': vx, vy, vz = u, -ones, v
    else: return zeros, zeros
    norm = np.sqrt(vx**2 + vy**2 + vz**2)
    vx, vy, vz = vx/norm, vy/norm, vz/norm
    lat, lon = np.arcsin(vy), np.arctan2(vx, vz)
    px = ((lon / (2 * np.pi)) + 0.5) * erp_w
    py = (0.5 - (lat / np.pi)) * erp_h
    return px, py

def resolve_path(path):
    """處理路徑轉換，支援 Kaggle 與本地環境"""
    if not path: return path
    if os.path.exists(path): return path
    # 處理 Windows/Linux 路徑差異
    path = path.replace("\\", "/")
    if "datasets" in path:
        parts = path.split("/")
        try:
            # 找到最後一個 "datasets" 的位置
            indices = [i for i, x in enumerate(parts) if x == "datasets"]
            if indices:
                idx = indices[-1]
                rel_parts = parts[idx:]
                # 環境變數 ROMA_GEO_DATASETS(同 config.datasets_root;此處直接讀變數
                # 以維持 roma_match 可獨立匯入)。rel_parts[0] 為 "datasets" 那一節。
                env_root = os.environ.get("ROMA_GEO_DATASETS")
                if env_root and len(rel_parts) > 1:
                    env_path = os.path.normpath(os.path.join(
                        os.path.abspath(os.path.expanduser(env_root)), *rel_parts[1:]))
                    if os.path.exists(env_path): return env_path
                # 退回:專案內的 datasets/
                script_dir = os.path.dirname(os.path.abspath(__file__))
                local_path = os.path.normpath(os.path.join(os.path.dirname(script_dir), *rel_parts))
                if os.path.exists(local_path): return local_path
                
                # 嘗試在根目錄找 datasets
                root_path = os.path.normpath(os.path.join(os.getcwd(), *rel_parts))
                if os.path.exists(root_path): return root_path
        except ValueError: pass
    return path

class RoMaMatcher:
    def __init__(self, device=None, face_size=512, conf_threshold=0.5, use_cache=True, cache_dir=None, model_type="roma_v2", workers=0, batch_bidir=False):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.workers = workers
        self.face_size = face_size
        self.conf_threshold = conf_threshold
        self.use_cache = use_cache
        self.batch_bidir = batch_bidir
        # 快取位置：明確參數 > 環境變數 ROMA_GEO_CACHE（同 config.cache_root）> 模組同層的 cache/。
        # 此處直接讀取環境變數而不 import config，以維持 roma_match 可獨立匯入。
        if cache_dir:
            self.cache_dir = cache_dir
        elif os.environ.get("ROMA_GEO_CACHE"):
            self.cache_dir = os.path.abspath(os.path.expanduser(os.environ["ROMA_GEO_CACHE"]))
        else:
            self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

        # 核心改動：只有在不使用 Worker 的情況下，才在主進程初始化模型
        self.matcher = None
        if self.workers == 0:
            fix_all_seeds(42)
            from core_roma_match.RoMaScan import MatcherFactory
            self.matcher = MatcherFactory.create(model_type=model_type, device=self.device, conf_threshold=conf_threshold, batch_bidir=batch_bidir)

        if CubemapProjector:
            self.projector = CubemapProjector(face_size=face_size)
        else:
            self.projector = None

        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_info(self, path_a, path_b, mode):        
        """獲取快取路徑與對稱性資訊 (統一 0.1 門檻)"""
        try:
            from roma_geo.names import house_of, image_stem
        except ImportError:
            def image_stem(path):
                return os.path.splitext(os.path.basename(str(path)))[0]

            def house_of(path):
                parts = [p for p in str(path).replace("\\", "/").split("/")
                         if p not in ("", ".", "..")]
                return parts[-2] if len(parts) >= 2 else "unknown"

        ha, na = house_of(path_a), image_stem(path_a)
        hb, nb = house_of(path_b), image_stem(path_b)
        h_prefix = ha if ha == hb else f"{ha}_{hb}"
        v_key = f"{mode}_{self.model_type}_std0.1_face{self.face_size}"
        
        f_ab = f"{h_prefix}_{na}_{nb}_{v_key}.npz"
        f_ba = f"{h_prefix}_{nb}_{na}_{v_key}.npz"
        p_ab, p_ba = os.path.join(self.cache_dir, f_ab), os.path.join(self.cache_dir, f_ba)
        
        if os.path.exists(p_ab): return p_ab, False
        if os.path.exists(p_ba): return p_ba, True
        return p_ab, False

    def _ensure_matcher(self):
        """確保模型已載入 (用於串行模式)"""
        if self.matcher is None:
            fix_all_seeds(42)
            from core_roma_match.RoMaScan import MatcherFactory
            self.matcher = MatcherFactory.create(model_type=self.model_type, device=self.device, conf_threshold=self.conf_threshold)

    def batch_process(self, tasks, mode="direct", exp_name=None):
        """
        批次處理匹配與評分任務 (向後相容介面)。
        現在統一重定向至階段式邏輯。
        """
        # 串行模式 (workers=0) 直接在此執行評分階段的邏輯
        if self.workers == 0:
            from depth.depth_estimator import DepthAnythingEstimator
            from roi_prediction.inference.predictor import ROIPredictor
            from segmentation.segmentor import SemanticSegmentor
            from scoring.scorer import (
                PointConfidenceScorer, AggregateConfidenceScorer, 
                AggregateROIScorer, GridDepthVarianceScorer,
                AggregateWeightedDepthScorer, AggregateSemanticWeightedDepthScorer,
                ModelFoldDispatcher
            )
            
            d_est = DepthAnythingEstimator(device=self.device, use_cache=self.use_cache)
            r_pred = ROIPredictor(device=self.device)
            s_seg = SemanticSegmentor(device=self.device, use_cache=self.use_cache)
            p_conf = PointConfidenceScorer()
            agg_conf = AggregateConfidenceScorer()
            agg_roi = AggregateROIScorer()
            grid_scorer = GridDepthVarianceScorer()
            agg_depth = AggregateWeightedDepthScorer()
            agg_sem = AggregateSemanticWeightedDepthScorer()
            
            model_dispatcher = None
            if exp_name:
                model_dispatcher = ModelFoldDispatcher(exp_name=exp_name)

            results = []
            from tqdm import tqdm
            for img_a, img_b, label, target_id, house_name, q_id in tqdm(tasks, desc="Sequential Scoring", leave=False):
                # 匹配 (會讀快取)
                if mode == "cubemap": matches, info = self.match_cubemap(img_a, img_b)
                elif mode == "core": matches, info = self.match_core(img_a, img_b)
                else: matches, info = self.match_direct(img_a, img_b)

                if not matches:
                    results.append({
                        "House": house_name, "Query": q_id, "Gallery": os.path.basename(img_b), "Is_Connected": label, 
                        "RawScore_RoMa": 0.0, "GridScore_Depth": 0.0, "ROIScore": 0.0, 
                        "WeightedDepthScore": 0.0, "SemanticScore": 0.0, "ModelBasedScore": 0.0,
                        "Is_Target": 0, "MatchPoints": 0
                    })
                else:
                    w, h = info["w"], info["h"]
                    depth_maps = {"A": d_est.predict(img_a), "B": d_est.predict(img_b)}
                    roi_maps = {"A": r_pred.predict(img_a), "B": r_pred.predict(img_b)}
                    lmap_a, lbl_a, _ = s_seg.predict(img_a, mode="direct")
                    lmap_b, lbl_b, _ = s_seg.predict(img_b, mode="direct")

                    s_raw = agg_conf.compute(matches)
                    s_roi = agg_roi.compute(matches, roi_maps, w, h)
                    s_grid = grid_scorer.compute(matches, p_conf.compute(matches), depth_maps, w, h)
                    s_depth = agg_depth.compute(matches, depth_maps, w, h)
                    s_sem = agg_sem.compute(matches, depth_maps, lmap_a, lbl_a, lmap_b, lbl_b, w, h)
                    
                    s_model = 0.0
                    if model_dispatcher:
                        scorer = model_dispatcher.get_scorer(house_name, self.device)
                        global_matches = info.get("global_matches", matches)
                        s_model = scorer.compute(img_a, global_matches, depth_maps["A"], depth_maps["B"])

                    results.append({
                        "House": house_name, "Query": q_id, "Gallery": os.path.basename(img_b), "Is_Connected": label, 
                        "RawScore_RoMa": round(float(s_raw), 4), 
                        "GridScore_Depth": round(float(s_grid), 4), 
                        "ROIScore": round(float(s_roi), 4),
                        "WeightedDepthScore": round(float(s_depth), 4),
                        "SemanticScore": round(float(s_sem), 4),
                        "ModelBasedScore": round(float(s_model), 4),
                        "Is_Target": 1 if target_id and target_id in os.path.basename(img_b) else 0, 
                        "MatchPoints": len(matches)
                    })
            return results
        else:
            return self.parallel_engine.run_stage(tasks, stage="score")

    def match_cubemap(self, img_a_path, img_b_path, tmp_dir="tmp_faces"):
        img_a_path = resolve_path(img_a_path)
        img_b_path = resolve_path(img_b_path)
        img_a, img_b = Image.open(img_a_path), Image.open(img_b_path)
        orig_w, orig_h = img_a.size
        info = {"w": orig_w, "h": orig_h}
        
        cache_path, is_swapped = self._get_cache_info(img_a_path, img_b_path, "cubemap")
        if self.use_cache and os.path.exists(cache_path):
            m = np.load(cache_path)['matches']
            if is_swapped and len(m) > 0: m[:, [0, 1, 2, 3]] = m[:, [2, 3, 0, 1]]
            return [p.tolist() for p in m if p[4] >= self.conf_threshold], info

        faces_a, faces_b = self.projector.erp_to_cubemap(img_a), self.projector.erp_to_cubemap(img_b)
        
        import tempfile
        if not os.path.exists(tmp_dir): os.makedirs(tmp_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_dir) as actual_tmp_dir:
            pa = {n: os.path.join(actual_tmp_dir, f"A_{n}.png") for n in faces_a}
            pb = {n: os.path.join(actual_tmp_dir, f"B_{n}.png") for n in faces_b}
            for n in faces_a: faces_a[n].save(pa[n])
            for n in faces_b: faces_b[n].save(pb[n])

            horiz, vert = ['front', 'right', 'back', 'left'], ['top', 'bottom']
            match_pairs = [(fa, fb) for fa in horiz for fb in horiz] + [(v, v) for v in vert]
            all_matches_raw = []
            self._ensure_matcher()
            for na, nb in match_pairs:
                # 這裡固定用 0.1 以確保快取是完整的
                face_matches, _ = self.matcher.match(pa[na], pb[nb], conf_threshold=0.1)
                if not face_matches: continue
                m_np = np.array(face_matches)
                xa_f, ya_f = m_np[:, 0], m_np[:, 1]
                xb_f, yb_f = m_np[:, 2], m_np[:, 3]
                m_confs = m_np[:, 4]
                xae, yae = batch_cubemap_to_erp(na, xa_f, ya_f, orig_w, orig_h, self.face_size)
                xbe, ybe = batch_cubemap_to_erp(nb, xb_f, yb_f, orig_w, orig_h, self.face_size)
                all_matches_raw.extend(list(zip(xae.tolist(), yae.tolist(), xbe.tolist(), ybe.tolist(), m_confs.tolist())))

        # 儲存完整的 0.1 結果到快取
        if self.use_cache: np.savez_compressed(cache_path, matches=np.array(all_matches_raw))
        
        # 執行演算法核心：僅針對符合當前 conf_threshold 的點進行處理
        all_matches = [m for m in all_matches_raw if m[4] >= self.conf_threshold]

        # 1-to-1 Filtering & Consensus Boosting: 
        # 1. 物理對應聚合 (Physical Correspondence Aggregation)
        # 針對 (xA, yA, xB, yB) 四元組進行聚合。若同一個對應出現在多個 Cubemap 面組中，代表幾何穩定性高。
        physical_matches = {} 
        for m in all_matches:
            xa, ya, xb, yb, conf = m
            # 使用 A 與 B 的座標作為 Key，取小數 1 位以容許投影微小漂移
            p_key = (round(xa, 1), round(ya, 1), round(xb, 1), round(yb, 1))
            if p_key not in physical_matches:
                physical_matches[p_key] = {"data": list(m), "count": 1, "max_conf": conf}
            else:
                physical_matches[p_key]["count"] += 1
                if conf > physical_matches[p_key]["max_conf"]:
                    physical_matches[p_key]["max_conf"] = conf
                    physical_matches[p_key]["data"] = list(m)

        # 2. 1-to-1 過濾 (確保 A 圖每個像素只匹配到一個 B 點)
        # 同時套用共識加成 (Consensus Boost): 每多一個面組證實，信心分數加成 20%
        best_unique_matches = {}
        for p_key, p_info in physical_matches.items():
            xa, ya = p_key[0], p_key[1]
            # 獎勵共識: 每多一個面組證實，信心分數加成 20% (上限 1.0)
            boosted_conf = min(p_info["max_conf"] * (1.0 + 0.2 * (p_info["count"] - 1)), 1.0)

            a_key = (xa, ya)
            if a_key not in best_unique_matches or boosted_conf > best_unique_matches[a_key][4]:
                final_m = p_info["data"]
                final_m[4] = boosted_conf
                best_unique_matches[a_key] = final_m

        all_matches = list(best_unique_matches.values())


        for p in list(pa.values()) + list(pb.values()):
            if os.path.exists(p): os.remove(p)
        return all_matches, info

    def match_direct(self, img_a_path, img_b_path):
        img_a_path, img_b_path = resolve_path(img_a_path), resolve_path(img_b_path)
        with Image.open(img_a_path) as img_a: info = {"w": img_a.size[0], "h": img_a.size[1]}
        
        cache_path, is_swapped = self._get_cache_info(img_a_path, img_b_path, "direct")
        if self.use_cache and os.path.exists(cache_path):
            m = np.load(cache_path)['matches']
            if is_swapped and len(m) > 0: m[:, [0, 1, 2, 3]] = m[:, [2, 3, 0, 1]]
            return [p.tolist() for p in m if p[4] >= self.conf_threshold], info

        self._ensure_matcher()
        # Direct mode uses ERP images, so we must disable MAGSAC geometric filtering
        all_m, _ = self.matcher.match(img_a_path, img_b_path, conf_threshold=0.1, use_magsac=False)
        if self.use_cache: np.savez_compressed(cache_path, matches=np.array(all_m))
        return [m for m in all_m if m[4] >= self.conf_threshold], info
    
    def match_core(self, img_a_path, img_b_path):
        self._ensure_matcher()
        # 使用 resolve_path 處理影像路徑
        img_a_path = resolve_path(img_a_path)
        img_b_path = resolve_path(img_b_path)

        # 預先取得圖片尺寸資訊以供回傳
        with Image.open(img_a_path) as img_a:
            info = {"w": img_a.size[0], "h": img_a.size[1]}

        if self.use_cache:
            cache_path, is_swapped = self._get_cache_info(img_a_path, img_b_path, "romascan")
            if os.path.exists(cache_path):
                cached = np.load(cache_path)
                m = cached['matches']
                if is_swapped and len(m) > 0:
                    # 交換座標：(xa, ya, xb, yb, conf) -> (xb, yb, xa, ya, conf)
                    m[:, [0, 1, 2, 3]] = m[:, [2, 3, 0, 1]]
                # 還原 global_matches 供 ModelBasedScorer 使用
                if 'global_matches' in cached and len(cached['global_matches']) > 0:
                    gm = cached['global_matches']
                    if is_swapped and len(gm) > 0:
                        gm = gm.copy()
                        gm[:, [0, 1, 2, 3]] = gm[:, [2, 3, 0, 1]]
                    info["global_matches"] = gm.tolist()
                # 執行時過濾：從 0.1 門檻的快取中提取符合當前要求的點
                return [p.tolist() for p in m if p[4] >= self.conf_threshold], info

        # 注意：run_roma_match 現在回傳的是 (matches, info)
        # 這裡強制使用 threshold=0.1 進行標準化快取，以便後續不同門檻共享
        matches, core_info = run_roma_match(img_a_path, img_b_path, threshold=0.1, model_type=self.model_type)
        info.update(core_info)

        if self.use_cache:
            # 使用統一的路徑生成器儲存（同時保存 global_matches 供 ModelBasedScorer 使用）
            cache_path, _ = self._get_cache_info(img_a_path, img_b_path, "romascan")
            gm = np.array(info.get("global_matches", []))
            np.savez_compressed(cache_path, matches=np.array(matches), global_matches=gm)

        return [m for m in matches if m[4] >= self.conf_threshold], info

    def match_core_batch(self, pairs_list):
        """[新增] 批次核心匹配：支援一次處理多對影像，提升 GPU 利用率"""
        self._ensure_matcher()
        results = [None] * len(pairs_list)
        to_compute_indices = []
        to_compute_pairs = []

        # 1. 快速檢查所有配對的快取
        for i, (img_a, img_b) in enumerate(pairs_list):
            img_a, img_b = resolve_path(img_a), resolve_path(img_b)
            cache_path, is_swapped = self._get_cache_info(img_a, img_b, "romascan")

            if self.use_cache and os.path.exists(cache_path):
                cached = np.load(cache_path)
                m = cached['matches']
                if is_swapped and len(m) > 0: m[:, [0, 1, 2, 3]] = m[:, [2, 3, 0, 1]]
                # 取得圖片尺寸供回傳
                with Image.open(img_a) as pil_a: info = {"w": pil_a.size[0], "h": pil_a.size[1]}
                # 還原 global_matches
                if 'global_matches' in cached and len(cached['global_matches']) > 0:
                    gm = cached['global_matches']
                    if is_swapped and len(gm) > 0:
                        gm = gm.copy()
                        gm[:, [0, 1, 2, 3]] = gm[:, [2, 3, 0, 1]]
                    info["global_matches"] = gm.tolist()
                results[i] = ([p.tolist() for p in m if p[4] >= self.conf_threshold], info)
            else:
                to_compute_indices.append(i)
                to_compute_pairs.append((img_a, img_b))

        # 2. 只有沒命中的部分才送入 Batch Pipeline
        if to_compute_pairs:
            from core_roma_match.RoMaScan import RoMaScanPipeline
            pipeline = RoMaScanPipeline(model_type=self.model_type)
            pipeline.matcher = self.matcher # 重用 Matcher

            # 參數必須與 match_core → run_roma_match → RomaScan 一致
            # interface.py 中指定: cluster_method="kmeans", fov_mode="fixed"
            batch_outputs = pipeline.execute_batch(
                to_compute_pairs, threshold=0.1,
                cluster_method="kmeans", fov_mode="fixed"
            )

            for i, (res, info) in enumerate(batch_outputs):
                idx = to_compute_indices[i]
                results[idx] = ([p for p in res if p[4] >= self.conf_threshold], info)

                # 寫回快取（同時保存 global_matches）
                if self.use_cache:
                    c_path, _ = self._get_cache_info(to_compute_pairs[i][0], to_compute_pairs[i][1], "romascan")
                    gm = np.array(info.get("global_matches", []))
                    np.savez_compressed(c_path, matches=np.array(res), global_matches=gm)

        return results