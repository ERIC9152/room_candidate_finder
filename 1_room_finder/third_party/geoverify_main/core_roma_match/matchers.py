import torch
import cv2
import numpy as np
from PIL import Image

class MatcherBase:
    """匹配模型介面 (SRP: Model Interface)"""
    def __init__(self, device=None, conf_threshold=0.01):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.conf_threshold = conf_threshold
        self.model = None

    def match(self, img_a_path, img_b_path, conf_threshold=None, sample_count=None):
        """應返回 (matches, info), matches 格式: [[xa, ya, xb, yb, conf], ...]
        conf_threshold: 過濾門檻，None 則使用建構時的預設值
        sample_count:   取樣數量 (僅 V2 有效)，None 則使用預設值 5000
        """
        raise NotImplementedError

class RomaV1Matcher(MatcherBase):
    """RoMa v1 實作"""
    def __init__(self, device=None, conf_threshold=0.01, **kwargs):
        super().__init__(device, conf_threshold)
        try:
            from romatch.models import roma_indoor
            self.model = roma_indoor(device=self.device, use_custom_corr=False)
        except ImportError:
            print("錯誤: 請先安裝 'romatch' 以使用 Roma v1")

    def match(self, img_a_path, img_b_path, conf_threshold=None, sample_count=None, **kwargs):
        from core_roma_match.pipeline import ImageHandler
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        warp, cert = self.model.match(img_a_path, img_b_path, device=self.device)
        mask = (cert > threshold).squeeze()
        
        # 處理 img_a_path 可能是 Image 實例的情況 (In-Memory)
        if isinstance(img_a_path, Image.Image):
            w, h = img_a_path.size
            info_a = {"w": w, "h": h}
        else:
            if mask.sum() == 0: return [], ImageHandler.get_info(img_a_path)
            with Image.open(img_a_path) as img:
                w, h = img.size
                info_a = {"w": w, "h": h}

        if mask.sum() == 0: return [], info_a

        idx = torch.nonzero(mask, as_tuple=False)
        m_coords = warp[0, idx[:, 0], idx[:, 1]].cpu().numpy()
        m_confs = cert[0, idx[:, 0], idx[:, 1]].cpu().numpy()

        xa, ya = (m_coords[:, 0] + 1) / 2 * w, (m_coords[:, 1] + 1) / 2 * h
        xb, yb = (m_coords[:, 2] + 1) / 2 * w, (m_coords[:, 3] + 1) / 2 * h
        return list(zip(xa.tolist(), ya.tolist(), xb.tolist(), yb.tolist(), m_confs.tolist())), info_a

    def match_batch(self, pairs_list, conf_threshold=None, sample_count=None):
        """[新增] 為 Roma V1 提供 Batch 介面支援"""
        results = []
        for img_a, img_b in pairs_list:
            if img_a is None or img_b is None:
                results.append(([], {}))
                continue
            res, info = self.match(img_a, img_b, conf_threshold, sample_count)
            results.append((res, info))
        return results

class RomaV2Matcher(MatcherBase):
    """
    RoMa v2 實作 (Harder Better Faster Denser)

    ─────────────────────────────────────────────────────────────────────────
    【實驗結論 2026-04-29：雙向匹配合併 (Bidirectional Match & Merge)】

    RoMa V2 的匹配具有不對稱性：match(A, B) 與 match(B, A) 會產生不同的匹配點集合。
    原因：官方 Multiview ViT 使用 alternating attention mode，
    將 (f_A, f_B) 堆疊在 position 0/1 進行交互注意力。
    當輸入順序交換時，position 0/1 的角色互換，導致注意力分佈不同。
    此外 DPT Head 以 position 0 的特徵為基底預測 warp field，
    因此「誰在 position 0」直接決定哪些區域被重點關注。

    實驗發現：單方向匹配容易遺漏某些連通區域的關鍵匹配點（尤其是門框、走廊等
    在某一方向信心較低但另一方向信心較高的區域）。

    注意：這與官方 bidirectional=True 是不同的概念。
    bidirectional=True 是在「同一次 forward」中同時計算 warp_AB 和 warp_BA，
    但兩者共用相同的 backbone 特徵（A 固定在 position 0）。
    本實驗證明需要「真正交換輸入順序」重新跑一次 forward，
    讓 B 也當過 position 0，才能獲得互補的匹配結果。

    解法：收到 match(A, B) 要求時，分別跑 forward(A, B) 和 forward(B, A) 兩次，
    再將結果合併為嚴格 1-to-1 mapping（衝突取較高 confidence）。

    加速策略：將 (A,B) 和 (B,A) 打包成 batch=2 送入同一次 forward()，
    GPU 只需一次 kernel launch 即可同時完成兩個方向的推論。
    ─────────────────────────────────────────────────────────────────────────
    """
    def __init__(self, device=None, conf_threshold=0.01, batch_bidir=False):
        """
        Args:
            device:          推論裝置
            conf_threshold:  信心度門檻
            batch_bidir:     是否使用 batch=2 打包加速雙向匹配。
                             True：將 (A,B) 和 (B,A) 打包成 batch=2 一次 forward（VRAM 需求翻倍）
                             False：串行跑兩次 forward（VRAM 友善，速度稍慢）[預設]
        """
        super().__init__(device, conf_threshold)
        self.batch_bidir = batch_bidir
        try:
            from romav2 import RoMaV2
            self.model = RoMaV2()
            if hasattr(self.model, 'to'):
                self.model.to(self.device)

            # [核心優化] 步驟二與步驟三：動態推論解析度調優與雙向均衡採樣策略
            if hasattr(self.model, 'apply_setting'):
                # 由於新版 romav2 的 apply_setting 只接受 Preset 字串，因此直接複寫屬性
                self.model.H_lr = 512
                self.model.W_lr = 512
                self.model.H_hr = 800
                self.model.W_hr = 800
                self.model.bidirectional = True      # 獲取雙向匹配並估計雙向協方差矩陣
                self.model.balanced_sampling = True  # 確保空間中匹配點的多樣性分佈
                # 拿掉 threshold 設定，讓 model.sample 回傳連續的信心度 (0~1)
                self.model.threshold = None

        except ImportError:
            print("錯誤: 請先安裝 'romav2' 以使用 Roma v2")

    # =====================================================================
    # 共用後處理模組 (match / match_batch 共用)
    # =====================================================================

    @staticmethod
    def _postprocess_matches(m_a, m_b, m_conf, use_magsac=True):
        """
        統一後處理：MAGSAC 幾何過濾 (可選) + 1-to-1 約束過濾。

        Args:
            m_a:        np.ndarray, shape (N, 2)：A 圖匹配點像素座標
            m_b:        np.ndarray, shape (N, 2)：B 圖匹配點像素座標
            m_conf:     np.ndarray, shape (N,)：信心度
            use_magsac: bool, 是否啟用基於針孔相機的幾何過濾 (對 ERP 全景圖應設為 False)

        Returns:
            list[tuple]：[(xA, yA, xB, yB, conf), ...], 嚴格 1-to-1
        """
        if len(m_a) == 0:
            return []

        # ── Step 1: MAGSAC 幾何過濾 ──
        # 使用 OpenCV USAC_MAGSAC 剔除異常點，確保內點具備良好的幾何一致性
        if use_magsac and len(m_a) >= 8:
            try:
                F, inlier_mask = cv2.findFundamentalMat(
                    m_a, m_b,
                    ransacReprojThreshold=0.2,
                    method=cv2.USAC_MAGSAC,
                    confidence=0.999999,
                    maxIters=10000
                )
                if inlier_mask is not None:
                    valid = inlier_mask.ravel().astype(bool)
                    m_a = m_a[valid]
                    m_b = m_b[valid]
                    m_conf = m_conf[valid]
            except Exception:
                pass  # 若求解失敗則保留原始點集

        if len(m_a) == 0:
            return []

        # ── Step 2: 1-to-1 約束過濾 ──
        # 兩方向合併後同一個 A/B 點可能對到多個對手，需做去重。
        # 按 confidence 降序排列，貪心保留最高者。
        all_matches = list(zip(
            m_a[:, 0].tolist(), m_a[:, 1].tolist(),
            m_b[:, 0].tolist(), m_b[:, 1].tolist(),
            m_conf.tolist()
        ))
        all_matches.sort(key=lambda x: x[4], reverse=True)

        final_matches = []
        used_a = set()
        used_b = set()
        for m in all_matches:
            ka = (round(m[0]), round(m[1]))
            kb = (round(m[2]), round(m[3]))
            if ka not in used_a and kb not in used_b:
                used_a.add(ka)
                used_b.add(kb)
                final_matches.append(m)

        return final_matches

    # =====================================================================
    # 共用推論核心：batch forward + per-pair sample + merge
    # =====================================================================

    def _batch_forward_and_sample(self, tensors_A, tensors_B, orig_sizes, threshold, sc, use_magsac=True):
        """
        Batch 模式：將 N 對圖片打包成 2N 對（正向+反向），
        一次 batch forward，再逐對 sample 並合併雙向結果。

        ⚠ VRAM 需求約為單向的 2 倍，需要足夠的 GPU 記憶體。
        若 VRAM 不足請使用 _sequential_forward_and_sample()。

        Args:
            tensors_A:   list[Tensor], 每個 shape (1, 3, H, W)：A 圖 raw tensors
            tensors_B:   list[Tensor], 每個 shape (1, 3, H, W)：B 圖 raw tensors
            orig_sizes:  list[(wa, ha, wb, hb)]：原始像素尺寸
            threshold:   float：信心度門檻
            sc:          int：sample_count
            use_magsac:  bool：是否使用幾何過濾

        Returns:
            list[(matches, info)]：長度 = N
        """
        import torch.nn.functional as TF
        from romav2.romav2 import _map_confidence

        N = len(tensors_A)
        self.model.eval()

        H_lr, W_lr = self.model.H_lr, self.model.W_lr
        H_hr, W_hr = self.model.H_hr, self.model.W_hr

        # ── 1. Resize 每張圖片（只做一次，正向和反向共用）──
        resized_lr = []
        resized_hr = []
        for t in tensors_A + tensors_B:
            resized_lr.append(TF.interpolate(t, size=(H_lr, W_lr), mode="bicubic",
                                             align_corners=False, antialias=True))
            if H_hr is not None and W_hr is not None:
                resized_hr.append(TF.interpolate(t, size=(H_hr, W_hr), mode="bicubic",
                                                 align_corners=False, antialias=True))

        # resized_lr / resized_hr 排列：[A0, A1, ..., A_{N-1}, B0, B1, ..., B_{N-1}]
        # 正向 pair i: A_lr[i] vs B_lr[i]   → index i vs index N+i
        # 反向 pair i: B_lr[i] vs A_lr[i]   → index N+i vs index i

        # ── 2. 組裝 batch：2N 對 (交錯排列：fwd_0, rev_0, fwd_1, rev_1, ...) ──
        left_lr_list = []
        right_lr_list = []
        left_hr_list = []
        right_hr_list = []
        for i in range(N):
            # 正向: (A, B)
            left_lr_list.append(resized_lr[i])
            right_lr_list.append(resized_lr[N + i])
            if resized_hr:
                left_hr_list.append(resized_hr[i])
                right_hr_list.append(resized_hr[N + i])
            # 反向: (B, A)
            left_lr_list.append(resized_lr[N + i])
            right_lr_list.append(resized_lr[i])
            if resized_hr:
                left_hr_list.append(resized_hr[N + i])
                right_hr_list.append(resized_hr[i])

        batch_left_lr = torch.cat(left_lr_list)   # (2N, 3, H_lr, W_lr)
        batch_right_lr = torch.cat(right_lr_list)
        batch_left_hr = torch.cat(left_hr_list) if left_hr_list else None
        batch_right_hr = torch.cat(right_hr_list) if right_hr_list else None

        # 釋放中間 tensors
        del resized_lr, resized_hr, left_lr_list, right_lr_list, left_hr_list, right_hr_list

        # ── 3. 單次 Batch Forward（2N 對一起通過 GPU）──
        with torch.inference_mode():
            raw_preds = self.model(
                batch_left_lr, batch_right_lr,
                img_A_hr=batch_left_hr, img_B_hr=batch_right_hr
            )
        del batch_left_lr, batch_right_lr, batch_left_hr, batch_right_hr

        # ── 4. Confidence Mapping ──
        warp_AB = raw_preds["warp_AB"]              # (2N, H, W, 2)
        confidence_AB = raw_preds["confidence_AB"]  # (2N, H, W, 4)
        overlap_AB, precision_AB = _map_confidence(
            confidence=confidence_AB, threshold=self.model.threshold
        )

        warp_BA = raw_preds.get("warp_BA")
        confidence_BA = raw_preds.get("confidence_BA")
        if self.model.bidirectional and confidence_BA is not None:
            overlap_BA, precision_BA = _map_confidence(
                confidence=confidence_BA, threshold=self.model.threshold
            )
        else:
            overlap_BA = precision_BA = None

        # ── 5. Per-pair Sample + 雙向合併 ──
        # batch 中的排列：[fwd_0, rev_0, fwd_1, rev_1, ..., fwd_{N-1}, rev_{N-1}]
        results = []
        for i in range(N):
            wa, ha, wb, hb = orig_sizes[i]
            info = {"w": wa, "h": ha}

            idx_fwd = 2 * i       # 正向 (A, B)
            idx_rev = 2 * i + 1   # 反向 (B, A)

            all_m_a_list = []
            all_m_b_list = []
            all_conf_list = []

            # ─ 正向 sample: forward(A, B) → kptsA 對應 A, kptsB 對應 B ─
            pair_preds_fwd = {
                "warp_AB": warp_AB[idx_fwd:idx_fwd+1],
                "overlap_AB": overlap_AB[idx_fwd:idx_fwd+1],
                "precision_AB": precision_AB[idx_fwd:idx_fwd+1],
                "warp_BA": warp_BA[idx_fwd:idx_fwd+1] if warp_BA is not None else None,
                "overlap_BA": overlap_BA[idx_fwd:idx_fwd+1] if overlap_BA is not None else None,
                "precision_BA": precision_BA[idx_fwd:idx_fwd+1] if precision_BA is not None else None,
            }
            matches_fwd, overlaps_fwd, _, _ = self.model.sample(pair_preds_fwd, sc)
            kptsA_fwd, kptsB_fwd = self.model.to_pixel_coordinates(matches_fwd, ha, wa, hb, wb)
            conf_fwd = overlaps_fwd.cpu().numpy()

            mask_fwd = conf_fwd > threshold
            if mask_fwd.sum() > 0:
                all_m_a_list.append(kptsA_fwd[mask_fwd].cpu().numpy())
                all_m_b_list.append(kptsB_fwd[mask_fwd].cpu().numpy())
                all_conf_list.append(conf_fwd[mask_fwd])

            # ─ 反向 sample: forward(B, A) → kptsA 對應 B, kptsB 對應 A ─
            # 注意：反向的 "A" 是原始 B，"B" 是原始 A
            # to_pixel_coordinates 的參數順序要對應 forward 的輸入順序
            pair_preds_rev = {
                "warp_AB": warp_AB[idx_rev:idx_rev+1],
                "overlap_AB": overlap_AB[idx_rev:idx_rev+1],
                "precision_AB": precision_AB[idx_rev:idx_rev+1],
                "warp_BA": warp_BA[idx_rev:idx_rev+1] if warp_BA is not None else None,
                "overlap_BA": overlap_BA[idx_rev:idx_rev+1] if overlap_BA is not None else None,
                "precision_BA": precision_BA[idx_rev:idx_rev+1] if precision_BA is not None else None,
            }
            matches_rev, overlaps_rev, _, _ = self.model.sample(pair_preds_rev, sc)
            # forward(B, A) 中 H_A=hb, W_A=wb, H_B=ha, W_B=wa
            # 回傳的 kptsA 對應 B 的像素座標, kptsB 對應 A 的像素座標
            kptsB_rev, kptsA_rev = self.model.to_pixel_coordinates(matches_rev, hb, wb, ha, wa)
            conf_rev = overlaps_rev.cpu().numpy()

            mask_rev = conf_rev > threshold
            if mask_rev.sum() > 0:
                all_m_a_list.append(kptsA_rev[mask_rev].cpu().numpy())
                all_m_b_list.append(kptsB_rev[mask_rev].cpu().numpy())
                all_conf_list.append(conf_rev[mask_rev])

            # ─ 合併雙向結果 + 後處理 ─
            if all_m_a_list:
                m_a = np.vstack(all_m_a_list)
                m_b = np.vstack(all_m_b_list)
                m_conf = np.concatenate(all_conf_list)
                final_matches = self._postprocess_matches(m_a, m_b, m_conf, use_magsac=use_magsac)
            else:
                final_matches = []

            results.append((final_matches, info))

        return results

    # =====================================================================
    # 串行推論核心：兩次獨立 forward + per-pair sample + merge
    # =====================================================================

    def _sequential_forward_and_sample(self, tensors_A, tensors_B, orig_sizes, threshold, sc, use_magsac=True):
        """
        串行模式：對每對圖片分別跑兩次 forward（正向 + 反向），
        再合併雙向結果。VRAM 需求與單向相同，適合記憶體不足的 GPU。

        Args:
            tensors_A:   list[Tensor], 每個 shape (1, 3, H, W)：A 圖 raw tensors
            tensors_B:   list[Tensor], 每個 shape (1, 3, H, W)：B 圖 raw tensors
            orig_sizes:  list[(wa, ha, wb, hb)]：原始像素尺寸
            threshold:   float：信心度門檻
            sc:          int：sample_count
            use_magsac:  bool：是否使用幾何過濾

        Returns:
            list[(matches, info)]：長度 = N
        """
        import torch.nn.functional as TF
        from romav2.romav2 import _map_confidence

        N = len(tensors_A)
        self.model.eval()

        H_lr, W_lr = self.model.H_lr, self.model.W_lr
        H_hr, W_hr = self.model.H_hr, self.model.W_hr

        results = []
        for i in range(N):
            wa, ha, wb, hb = orig_sizes[i]
            info = {"w": wa, "h": ha}

            # ── Resize（每張圖片只做一次，正向/反向共用）──
            t_a_lr = TF.interpolate(tensors_A[i], size=(H_lr, W_lr), mode="bicubic",
                                    align_corners=False, antialias=True)
            t_b_lr = TF.interpolate(tensors_B[i], size=(H_lr, W_lr), mode="bicubic",
                                    align_corners=False, antialias=True)
            t_a_hr = TF.interpolate(tensors_A[i], size=(H_hr, W_hr), mode="bicubic",
                                    align_corners=False, antialias=True) if H_hr and W_hr else None
            t_b_hr = TF.interpolate(tensors_B[i], size=(H_hr, W_hr), mode="bicubic",
                                    align_corners=False, antialias=True) if H_hr and W_hr else None

            all_m_a_list = []
            all_m_b_list = []
            all_conf_list = []

            # ── 正向 forward: (A, B) ──
            with torch.inference_mode():
                raw_fwd = self.model(t_a_lr, t_b_lr, img_A_hr=t_a_hr, img_B_hr=t_b_hr)

            overlap_fwd, precision_fwd = _map_confidence(
                confidence=raw_fwd["confidence_AB"], threshold=self.model.threshold)
            preds_fwd = {
                "warp_AB": raw_fwd["warp_AB"],
                "overlap_AB": overlap_fwd,
                "precision_AB": precision_fwd,
            }
            if self.model.bidirectional and raw_fwd.get("confidence_BA") is not None:
                overlap_ba_fwd, precision_ba_fwd = _map_confidence(
                    confidence=raw_fwd["confidence_BA"], threshold=self.model.threshold)
                preds_fwd["warp_BA"] = raw_fwd.get("warp_BA")
                preds_fwd["overlap_BA"] = overlap_ba_fwd
                preds_fwd["precision_BA"] = precision_ba_fwd
            else:
                preds_fwd["warp_BA"] = None
                preds_fwd["overlap_BA"] = None
                preds_fwd["precision_BA"] = None
            del raw_fwd

            matches_fwd, overlaps_fwd, _, _ = self.model.sample(preds_fwd, sc)
            kptsA_fwd, kptsB_fwd = self.model.to_pixel_coordinates(matches_fwd, ha, wa, hb, wb)
            conf_fwd = overlaps_fwd.cpu().numpy()
            del preds_fwd

            mask_fwd = conf_fwd > threshold
            if mask_fwd.sum() > 0:
                all_m_a_list.append(kptsA_fwd[mask_fwd].cpu().numpy())
                all_m_b_list.append(kptsB_fwd[mask_fwd].cpu().numpy())
                all_conf_list.append(conf_fwd[mask_fwd])

            # ── 反向 forward: (B, A)：真正交換輸入順序 ──
            with torch.inference_mode():
                raw_rev = self.model(t_b_lr, t_a_lr, img_A_hr=t_b_hr, img_B_hr=t_a_hr)

            overlap_rev, precision_rev = _map_confidence(
                confidence=raw_rev["confidence_AB"], threshold=self.model.threshold)
            preds_rev = {
                "warp_AB": raw_rev["warp_AB"],
                "overlap_AB": overlap_rev,
                "precision_AB": precision_rev,
            }
            if self.model.bidirectional and raw_rev.get("confidence_BA") is not None:
                overlap_ba_rev, precision_ba_rev = _map_confidence(
                    confidence=raw_rev["confidence_BA"], threshold=self.model.threshold)
                preds_rev["warp_BA"] = raw_rev.get("warp_BA")
                preds_rev["overlap_BA"] = overlap_ba_rev
                preds_rev["precision_BA"] = precision_ba_rev
            else:
                preds_rev["warp_BA"] = None
                preds_rev["overlap_BA"] = None
                preds_rev["precision_BA"] = None
            del raw_rev

            matches_rev, overlaps_rev, _, _ = self.model.sample(preds_rev, sc)
            # forward(B, A) 中 H_A=hb, W_A=wb, H_B=ha, W_B=wa
            # 回傳的 kptsA 對應 B 的像素座標, kptsB 對應 A 的像素座標
            kptsB_rev, kptsA_rev = self.model.to_pixel_coordinates(matches_rev, hb, wb, ha, wa)
            conf_rev = overlaps_rev.cpu().numpy()
            del preds_rev

            mask_rev = conf_rev > threshold
            if mask_rev.sum() > 0:
                all_m_a_list.append(kptsA_rev[mask_rev].cpu().numpy())
                all_m_b_list.append(kptsB_rev[mask_rev].cpu().numpy())
                all_conf_list.append(conf_rev[mask_rev])

            # ── 合併雙向結果 + 後處理 ──
            if all_m_a_list:
                m_a = np.vstack(all_m_a_list)
                m_b = np.vstack(all_m_b_list)
                m_conf = np.concatenate(all_conf_list)
                final_matches = self._postprocess_matches(m_a, m_b, m_conf, use_magsac=use_magsac)
            else:
                final_matches = []

            results.append((final_matches, info))

        return results

    # =====================================================================
    # 單對匹配
    # =====================================================================

    def match(self, img_a_path, img_b_path, conf_threshold=None, sample_count=None, use_magsac=True):
        from core_roma_match.pipeline import ImageHandler
        if self.model is None:
            return [], ImageHandler.get_info(img_a_path) if isinstance(img_a_path, str) else {"w": img_a_path.size[0], "h": img_a_path.size[1]}

        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        sc = sample_count if sample_count is not None else 5000

        # 處理 img_a_path/img_b_path 可能是 Image 實例的情況 (In-Memory)
        if isinstance(img_a_path, Image.Image):
            info_a = {"w": img_a_path.size[0], "h": img_a_path.size[1]}
        else:
            info_a = ImageHandler.get_info(img_a_path)

        if isinstance(img_b_path, Image.Image):
            info_b = {"w": img_b_path.size[0], "h": img_b_path.size[1]}
        else:
            info_b = ImageHandler.get_info(img_b_path)

        wa, ha = info_a["w"], info_a["h"]
        wb, hb = info_b["w"], info_b["h"]

        # 載入圖片 tensor（只載一次，正向/反向共用）
        tensor_A = self.model._load_image(img_a_path)
        tensor_B = self.model._load_image(img_b_path)

        # 根據 batch_bidir 選擇推論策略
        if self.batch_bidir:
            # Batch 模式：(A,B) + (B,A) 打包成 batch=2，一次 forward（VRAM 需求 ×2）
            batch_results = self._batch_forward_and_sample(
                [tensor_A], [tensor_B], [(wa, ha, wb, hb)], threshold, sc, use_magsac=use_magsac
            )
        else:
            # 串行模式：兩次獨立 forward，VRAM 友善
            batch_results = self._sequential_forward_and_sample(
                [tensor_A], [tensor_B], [(wa, ha, wb, hb)], threshold, sc, use_magsac=use_magsac
            )

        final_matches, _ = batch_results[0]

        # [新增] 提取密集信心圖 (Direct Map) 供連通性模型訓練使用
        # 注意：certainty map 目前僅從正向 preds 中提取（如需要可另外處理）

        return final_matches, info_a

    # =====================================================================
    # 批次匹配
    # =====================================================================

    def match_batch(self, pairs_list, conf_threshold=None, sample_count=None):
        """GPU 批次匹配 + 雙向合併。

        根據 self.batch_bidir 選擇推論策略：
          - batch_bidir=True:  將 N 對展開為 2N 對，一次 batch forward（VRAM ×2）
          - batch_bidir=False: 逐對串行跑兩次 forward（VRAM 友善）[預設]

        OOM 時自動 fallback 至逐對 match() 模式。
        """
        if self.model is None:
            return [([], {}) for _ in pairs_list]

        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        sc = sample_count if sample_count is not None else 5000

        # 過濾 None pairs，記錄有效索引
        valid_indices = []
        valid_pairs = []
        results = [([], {}) for _ in pairs_list]

        for i, (img_a, img_b) in enumerate(pairs_list):
            if img_a is None or img_b is None:
                continue
            valid_indices.append(i)
            valid_pairs.append((img_a, img_b))

        if not valid_pairs:
            return results

        try:
            # 載入圖片 tensors
            tensors_A = []
            tensors_B = []
            orig_sizes = []

            for img_a, img_b in valid_pairs:
                tensor_A = self.model._load_image(img_a)
                tensor_B = self.model._load_image(img_b)
                ha, wa = tensor_A.shape[2], tensor_A.shape[3]
                hb, wb = tensor_B.shape[2], tensor_B.shape[3]
                orig_sizes.append((wa, ha, wb, hb))
                tensors_A.append(tensor_A)
                tensors_B.append(tensor_B)

            # 根據 batch_bidir 選擇推論策略
            if self.batch_bidir:
                # Batch 模式：N 對展開為 2N 對，一次 forward（VRAM 需求 ×2）
                batch_results = self._batch_forward_and_sample(
                    tensors_A, tensors_B, orig_sizes, threshold, sc
                )
            else:
                # 串行模式：逐對跑兩次 forward，VRAM 友善
                batch_results = self._sequential_forward_and_sample(
                    tensors_A, tensors_B, orig_sizes, threshold, sc
                )

            for j, (final_matches, info) in enumerate(batch_results):
                results[valid_indices[j]] = (final_matches, info)

            return results

        except torch.cuda.OutOfMemoryError:
            # OOM fallback：清理顯存後退回逐對模式
            torch.cuda.empty_cache()
            print(f"  [match_batch] OOM with batch={len(valid_pairs)}, falling back to sequential")
            fallback = []
            for img_a, img_b in pairs_list:
                if img_a is None or img_b is None:
                    fallback.append(([], {}))
                    continue
                res, info = self.match(img_a, img_b, conf_threshold, sample_count)
                fallback.append((res, info))
            return fallback

class MatcherFactory:
    """工廠類別：根據名稱創建對應的 Matcher，並快取實例以避免重複載入 (Singleton Pattern)
    使用 registry dict 註冊 Matcher 類型 (OCP: 新增版本不需修改 Factory)"""
    _instances = {}
    _registry = {
        "roma_v1": RomaV1Matcher,
        "roma_v2": RomaV2Matcher,
    }

    @classmethod
    def register(cls, name, matcher_class):
        """註冊新的 Matcher 類型 (OCP: 擴展不需修改 Factory)"""
        cls._registry[name] = matcher_class

    @staticmethod
    def create(model_type="roma_v1", device=None, conf_threshold=0.01, **kwargs):
        """
        建立或取得快取的 Matcher 實例。

        Args:
            model_type:      "roma_v1" 或 "roma_v2"
            device:          推論裝置
            conf_threshold:  信心度門檻
            **kwargs:        傳遞給 Matcher 的額外參數，例如 batch_bidir=True
        """
        target_device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        key = (model_type, str(target_device))

        if key not in MatcherFactory._instances:
            if model_type not in MatcherFactory._registry:
                raise ValueError(f"未知的 model_type: '{model_type}'，可用: {list(MatcherFactory._registry.keys())}")
            print(f"[MatcherFactory] Creating new {model_type} instance on {target_device}...")
            matcher_cls = MatcherFactory._registry[model_type]
            MatcherFactory._instances[key] = matcher_cls(target_device, conf_threshold, **kwargs)

        # 不再修改快取實例的 conf_threshold：由呼叫端透過 match() 參數傳入
        return MatcherFactory._instances[key]