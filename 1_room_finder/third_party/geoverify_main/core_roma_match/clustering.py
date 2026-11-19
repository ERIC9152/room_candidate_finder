import numpy as np
from sklearn.cluster import DBSCAN, KMeans

class DBSCANClusterManager:
    """處理 DBSCAN 分群與合併 (SRP: Clustering Logic)"""
    def __init__(self, W, eps=150, min_samples=20):
        self.W = W
        self.eps = eps
        self.min_samples = min_samples

    def _get_circular_center(self, pts_x, pts_y):
        theta = (pts_x / self.W) * 2 * np.pi
        avg_theta = np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta)))
        return (avg_theta / (2 * np.pi)) * self.W % self.W, np.mean(pts_y)

    def run(self, matches):
        if len(matches) == 0: return []
        pts, R = matches[:, 0:4], self.W / (2 * np.pi)
        theta_a, theta_b = (pts[:, 0] / self.W) * 2 * np.pi, (pts[:, 2] / self.W) * 2 * np.pi
        feat = np.column_stack([R*np.cos(theta_a), R*np.sin(theta_a), pts[:, 1], R*np.cos(theta_b), R*np.sin(theta_b), pts[:, 3]])

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(feat)
        clusters = []
        for cid in set(db.labels_):
            if cid == -1: continue
            c_pts = matches[db.labels_ == cid]
            ca_x, ca_y = self._get_circular_center(c_pts[:, 0], c_pts[:, 1])
            cb_x, cb_y = self._get_circular_center(c_pts[:, 2], c_pts[:, 3])
            clusters.append({'id': cid, 'points': c_pts, 'center_a': np.array([ca_x, ca_y]), 'center_b': np.array([cb_x, cb_y])})
        return clusters

    def merge(self, clusters, threshold=800):
        if not clusters: return []
        while True:
            merged, new_clusters, used = False, [], set()
            for i in range(len(clusters)):
                if i in used: continue
                curr = clusters[i].copy()
                used.add(i)
                for j in range(i + 1, len(clusters)):
                    if j in used: continue
                    da = np.hypot(min(abs(curr['center_a'][0]-clusters[j]['center_a'][0]), self.W-abs(curr['center_a'][0]-clusters[j]['center_a'][0])), abs(curr['center_a'][1]-clusters[j]['center_a'][1]))
                    db = np.hypot(min(abs(curr['center_b'][0]-clusters[j]['center_b'][0]), self.W-abs(curr['center_b'][0]-clusters[j]['center_b'][0])), abs(curr['center_b'][1]-clusters[j]['center_b'][1]))
                    if da < threshold and db < threshold:
                        curr['points'] = np.vstack([curr['points'], clusters[j]['points']])
                        curr['center_a'] = np.array(self._get_circular_center(curr['points'][:, 0], curr['points'][:, 1]))
                        curr['center_b'] = np.array(self._get_circular_center(curr['points'][:, 2], curr['points'][:, 3]))
                        used.add(j)
                        merged = True
                new_clusters.append(curr)
            clusters = new_clusters
            if not merged: break
        return clusters

class KMeansClusterManager:
    """處理 K-Means 固定群數分群 (SRP: KMeans Clustering Logic)"""
    def __init__(self, W, n_clusters=5):
        self.W = W
        self.n_clusters = n_clusters

    def _get_circular_center(self, pts_x, pts_y, weights=None):
        if len(pts_x) == 0: return 0, 0
        if weights is None: weights = np.ones_like(pts_x)
        
        # [優化] 密度峰值選擇 (Density Peak Selection)
        # 目的：避免質心落在空曠處。尋找「局部點雲最密集」的地方。
        # sigma 代表感應半徑，約 100 像素
        sigma_sq = 100.0**2
        
        # 考慮 ERP X 軸環形距離
        dx = np.abs(pts_x[:, None] - pts_x[None, :])
        dx = np.minimum(dx, self.W - dx)
        dy = pts_y[:, None] - pts_y[None, :]
        dist_sq = dx**2 + dy**2
        
        # 計算每個點的局部密度得分 (Confidence-weighted RBF Kernel)
        density_scores = np.sum(weights[None, :] * np.exp(-dist_sq / (2 * sigma_sq)), axis=1)
        best_idx = np.argmax(density_scores)
        
        return pts_x[best_idx], pts_y[best_idx]

    def run(self, matches):
        if len(matches) == 0: return []
        pts, R = matches[:, 0:4], self.W / (2 * np.pi)
        confs = matches[:, 4]
        theta_a, theta_b = (pts[:, 0] / self.W) * 2 * np.pi, (pts[:, 2] / self.W) * 2 * np.pi
        feat = np.column_stack([R*np.cos(theta_a), R*np.sin(theta_a), pts[:, 1], R*np.cos(theta_b), R*np.sin(theta_b), pts[:, 3]])

        # 確保 K 不大於總點數
        n_c = min(self.n_clusters, len(matches))
        # 傳入 sample_weight，讓中心點會受到高信心度點的「萬有引力」吸引
        kmeans = KMeans(n_clusters=n_c, random_state=42, n_init='auto').fit(feat, sample_weight=confs)

        clusters = []
        for cid in range(n_c):
            mask = (kmeans.labels_ == cid)
            c_pts = matches[mask]
            if len(c_pts) == 0: continue
            c_confs = confs[mask]
            ca_x, ca_y = self._get_circular_center(c_pts[:, 0], c_pts[:, 1], weights=c_confs)
            cb_x, cb_y = self._get_circular_center(c_pts[:, 2], c_pts[:, 3], weights=c_confs)
            # 安全性斷言：確保中心點在合理的像素範圍內
            assert 0 <= ca_x <= self.W and 0 <= cb_x <= self.W, \
                f"[KMeans] center x out of bounds: ca_x={ca_x}, cb_x={cb_x}, W={self.W}"
            clusters.append({'id': cid, 'points': c_pts, 'center_a': np.array([ca_x, ca_y]), 'center_b': np.array([cb_x, cb_y])})
        return clusters
