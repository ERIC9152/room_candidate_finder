import cv2
import numpy as np

class GeometryFilter:
    """
    全景圖幾何驗證插件 (SRP: Geometric Verification)
    使用極線幾何 (Epipolar Geometry) 過濾不符合物理規律的匹配點。
    """
    def __init__(self, W, H):
        self.W = W
        self.H = H

    def _erp_to_unit_vectors(self, pts):
        """將 ERP 像素座標轉換為球面 3D 向量 (x-右, y-下, z-前)"""
        pts = np.atleast_2d(pts)
        # 經度 (Longitude): -pi to pi
        lon = (pts[:, 0] / self.W - 0.5) * 2 * np.pi
        # 緯度 (Latitude): -pi/2 to pi/2
        lat = (pts[:, 1] / self.H - 0.5) * np.pi
        
        vx = np.cos(lat) * np.sin(lon)
        vy = np.sin(lat)
        vz = np.cos(lat) * np.cos(lon)
        return np.stack([vx, vy, vz], axis=1)

    def filter(self, matches, ransac_threshold=0.01):
        """
        輸入: matches [N, 5] (xA, yA, xB, yB, conf)
        輸出: (filtered_matches, pose_info)
        """
        if len(matches) < 8:
            return matches, {"status": "too_few_points", "inlier_ratio": 0.0}

        m_np = np.array(matches)
        # 1. 轉換為 3D 向量
        pts1_3d = self._erp_to_unit_vectors(m_np[:, 0:2])
        pts2_3d = self._erp_to_unit_vectors(m_np[:, 2:4])

        # 2. 準備 2D 歸一化座標 (僅用於計算 E 矩陣)
        # 將 3D 向量投影到 Z=1 的平面。為了穩定性，過濾掉 Z <= 0 的點
        # 注意：這僅用於求解 E，最後的 recoverPose 會檢驗所有點。
        mask_front = (pts1_3d[:, 2] > 0.1) & (pts2_3d[:, 2] > 0.1)
        if np.sum(mask_front) < 8:
            # 如果前方點太少，嘗試放寬條件或直接使用前兩個分量 (Orthographic approx)
            pts1_2d = pts1_3d[:, :2].astype(np.float32)
            pts2_2d = pts2_3d[:, :2].astype(np.float32)
        else:
            pts1_2d = (pts1_3d[:, :2] / pts1_3d[:, 2:3]).astype(np.float32)
            pts2_2d = (pts2_3d[:, :2] / pts2_3d[:, 2:3]).astype(np.float32)

        # 估計本質矩陣 (Essential Matrix)
        E, mask = cv2.findEssentialMat(
            pts1_2d, pts2_2d, focal=1.0, pp=(0., 0.),
            method=cv2.RANSAC, prob=0.999, threshold=ransac_threshold
        )

        if E is None or E.shape != (3, 3):
            return matches, {"status": "failed_to_find_E", "inlier_ratio": 0.0}

        # 3. 執行手性檢測 (Cheirality Check)
        # 使用原始的 2D 投影點與 E 矩陣來恢復位姿
        _, R, t, mask_pose = cv2.recoverPose(E, pts1_2d, pts2_2d, mask=mask)
        # 4. 根據 RANSAC 內點 (Inliers) 進行過濾
        # mask_pose 包含了 Essential Matrix 的 RANSAC 結果與 Cheirality 測試結果
        inlier_mask = (mask_pose.ravel() > 0)
        filtered_matches = m_np[inlier_mask].tolist()
        
        pose_info = {
            "status": "success",
            "R": R.tolist(),
            "t": t.tolist(),
            "inlier_ratio": len(filtered_matches) / len(matches),
            "original_count": len(matches),
            "filtered_count": len(filtered_matches)
        }
        
        return filtered_matches, pose_info
