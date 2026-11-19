import numpy as np

class ProjectionEngine:
    """處理 ERP 與透視投影間的數學映射 (SRP: Math Only)"""
    def __init__(self, W, H, out_size=512):
        self.W = W
        self.H = H
        self.out_size = out_size

    def get_mapping(self, cx, cy, fov_deg):
        """
        將 ERP 中心點 (cx, cy) 轉換為透視投影映射。

        座標約定:
          - cx: ERP 水平像素 (0=最左, W=最右) → yaw ∈ [-π, π]
          - cy: ERP 垂直像素 (0=頂部/天空, H=底部/地板) → pitch ∈ [+π/2, -π/2]
          - cy < H/2 → pitch > 0 → 相機朝上看 (天空)
          - cy > H/2 → pitch < 0 → 相機朝下看 (地板)
        """
        yaw = (cx / self.W - 0.5) * 2 * np.pi
        pitch = -(cy / self.H - 0.5) * np.pi
        f = (self.out_size / 2) / np.tan(np.radians(fov_deg) / 2)

        u, v = np.meshgrid(np.arange(self.out_size), np.arange(self.out_size))
        x_c = u - self.out_size / 2
        # 座標系說明 (y-down camera convention):
        #   v=0 (圖片頂部) → y_c = -out_size/2 (負值)
        #   v=out_size (圖片底部) → y_c = +out_size/2 (正值)
        # 搭配 lat = arcsin(-y_r / rho)，y_c < 0 → lat > 0 → ERP 上方 (天空)
        # 此方向保證輸出影像不會上下翻轉。請勿對 y_c 取反。
        y_c = v - self.out_size / 2
        z_c = np.full_like(x_c, f)

        # 旋轉矩陣：定義相機相對於世界的姿態
        # 此處 R 是 Cam -> World 的轉換
        R_pitch = np.array([[1, 0, 0], 
                           [0, np.cos(pitch), -np.sin(pitch)], 
                           [0, np.sin(pitch), np.cos(pitch)]])
        R_yaw = np.array([[np.cos(yaw), 0, np.sin(yaw)], 
                         [0, 1, 0], 
                         [-np.sin(yaw), 0, np.cos(yaw)]])
        R = R_yaw @ R_pitch

        vecs = np.stack([x_c, y_c, z_c], axis=-1)
        vecs_rot = np.einsum('ij,hwj->hwi', R, vecs)
        rho = np.linalg.norm(vecs_rot, axis=-1)
        x_r, y_r, z_r = vecs_rot[..., 0], vecs_rot[..., 1], vecs_rot[..., 2]

        lon = np.arctan2(x_r, z_r)
        # 在 R_pitch 定義下，y_r 負值代表向上。緯度 lat 正值代表天空。
        lat = np.arcsin(np.clip(-y_r / rho, -1.0, 1.0))
        
        map_x = (lon / (2 * np.pi) + 0.5) * self.W
        map_y = (0.5 - lat / np.pi) * self.H

        def map_back(pts_persp):
            if len(pts_persp) == 0: return np.array([])
            px, py = pts_persp[:, 0] - self.out_size / 2, pts_persp[:, 1] - self.out_size / 2
            pz = np.full_like(px, f)
            # 反投影：Cam 點透過 R 轉回 World 空間
            pts_3d_rot = np.stack([px, py, pz], axis=1) @ R.T
            r = np.linalg.norm(pts_3d_rot, axis=1)
            lat_back = np.arcsin(np.clip(-pts_3d_rot[:, 1] / r, -1.0, 1.0))
            lon_back = np.arctan2(pts_3d_rot[:, 0], pts_3d_rot[:, 2])
            return np.stack([(lon_back / (2 * np.pi) + 0.5) * self.W, (0.5 - lat_back / np.pi) * self.H], axis=1)

        return map_x.astype(np.float32), map_y.astype(np.float32), map_back

class FOVOptimizer:
    """計算適合群組大小的動態 FOV (SRP: Optimization Logic)"""
    @staticmethod
    def calculate(cluster, W, H, min_fov=60, max_fov=120, margin=20):
        def _get_fov(pts, cx, center_W, total_W, total_H):
            shifted_x = (pts[:, 0] - cx + total_W / 2) % total_W - total_W / 2
            span_x, span_y = np.max(shifted_x) - np.min(shifted_x), np.max(pts[:, 1]) - np.min(pts[:, 1])
            return np.clip(max((span_x / total_W) * 360, (span_y / total_H) * 180) + margin, min_fov, max_fov)

        fov_a = _get_fov(cluster['points'][:, 0:2], cluster['center_a'][0], cluster['center_a'][0], W, H)
        fov_b = _get_fov(cluster['points'][:, 2:4], cluster['center_b'][0], cluster['center_b'][0], W, H)
        return fov_a, fov_b
