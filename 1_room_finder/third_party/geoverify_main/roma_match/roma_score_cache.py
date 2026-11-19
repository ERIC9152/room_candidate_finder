"""
[模組用途]
`RomaScoreCache`：以「無向配對」為鍵的稀疏聚合分數表。

批次匹配跑完後，逐點的對應只留在 `roma_match/cache/` 的 `.npz` 內，記憶體中的匹配即丟棄。
本模組另存一份「每對一筆」的聚合分數（匹配點數與信心度的總和／平均），讓下游工具
（`stitch_tool`）能依「RoMa 為這條連線提供了多少證據」排序，不必重跑 RoMa，
也不必逐檔讀取原始對應。

[鍵]
`(house, min(imgA, imgB), max(imgA, imgB))`：無向。`match_direct(A, B)` 與
`match_direct(B, A)` 共用同一份快取檔（見 `RoMaMatcher._get_cache_info` 的 `is_swapped`
處理），描述的是同一組實體對應。

[格式]
`.npz` 內為數個等長的平行陣列，查不到的配對回傳 None（代表「未驗證」，不是 0）。
"""
import os
import numpy as np


class RomaScoreCache:
    """Accumulable, npz-backed cache of per-pair aggregated RoMa scores."""

    def __init__(self):
        self._keys = {}   # (house, a, b) -> row index
        self._match_count = []   # list[int]
        self._conf_sum = []      # list[float]
        self._conf_mean = []     # list[float]

    @staticmethod
    def _basename(path_or_name):
        return path_or_name.replace("\\", "/").rsplit("/", 1)[-1]

    @staticmethod
    def make_key(house, img_a, img_b):
        a = RomaScoreCache._basename(img_a)
        b = RomaScoreCache._basename(img_b)
        lo, hi = (a, b) if a <= b else (b, a)
        return (house, lo, hi)

    def has(self, house, img_a, img_b):
        return self.make_key(house, img_a, img_b) in self._keys

    def put(self, house, img_a, img_b, match_count, conf_sum):
        """Stores/overwrites one pair's aggregated score. conf_mean is derived, not stored twice."""
        key = self.make_key(house, img_a, img_b)
        conf_mean = float(conf_sum) / match_count if match_count > 0 else 0.0
        if key in self._keys:
            idx = self._keys[key]
            self._match_count[idx] = int(match_count)
            self._conf_sum[idx] = float(conf_sum)
            self._conf_mean[idx] = conf_mean
        else:
            self._keys[key] = len(self._match_count)
            self._match_count.append(int(match_count))
            self._conf_sum.append(float(conf_sum))
            self._conf_mean.append(conf_mean)

    def get(self, house, img_a, img_b):
        """Returns {'match_count', 'conf_sum', 'conf_mean'} or None if this pair was never scored."""
        idx = self._keys.get(self.make_key(house, img_a, img_b))
        if idx is None:
            return None
        return {
            "match_count": self._match_count[idx],
            "conf_sum": self._conf_sum[idx],
            "conf_mean": self._conf_mean[idx],
        }

    def num_edges(self):
        return len(self._match_count)

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        houses = np.array([k[0] for k in self._keys], dtype="U")
        a_names = np.array([k[1] for k in self._keys], dtype="U")
        b_names = np.array([k[2] for k in self._keys], dtype="U")
        rows = np.array([self._keys[k] for k in self._keys], dtype=np.int64)
        order = np.argsort(rows)
        np.savez_compressed(
            path,
            houses=houses[order], a_names=a_names[order], b_names=b_names[order],
            match_count=np.asarray(self._match_count, dtype=np.int32)[rows[order]],
            conf_sum=np.asarray(self._conf_sum, dtype=np.float32)[rows[order]],
            conf_mean=np.asarray(self._conf_mean, dtype=np.float32)[rows[order]],
        )

    @classmethod
    def load(cls, path):
        cache = cls()
        if not os.path.exists(path):
            return cache
        d = np.load(path, allow_pickle=False)
        houses, a_names, b_names = d["houses"], d["a_names"], d["b_names"]
        match_count, conf_sum, conf_mean = d["match_count"], d["conf_sum"], d["conf_mean"]
        for idx in range(len(houses)):
            key = (str(houses[idx]), str(a_names[idx]), str(b_names[idx]))
            cache._keys[key] = len(cache._match_count)
            cache._match_count.append(int(match_count[idx]))
            cache._conf_sum.append(float(conf_sum[idx]))
            cache._conf_mean.append(float(conf_mean[idx]))
        return cache
