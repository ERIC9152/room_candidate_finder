"""
幾何純函式測試：方位統計、凸包、單側區域幾何、DBSCAN。

這些函式共同的風險是**跨接縫（±180°）**與**退化輸入**：出錯時不會拋出例外，只會
安靜地給出偏掉的數值。因此本檔的期望值盡量以獨立算法求得。`wedge_width` 以
另一種列舉方式暴力算一遍對照，凸包以「所有輸入點都在包內」與面積公式驗證，
期望值由規約或獨立算法推得。
"""
import unittest

import numpy as np

from _support import W_DEF, H_DEF  # noqa: F401（匯入時補 sys.path）

from pose_regions.batch_pose_from_cache import (
    _convex_hull, _dbscan, _side_geom, az_coverage, extract_regions, wedge_width,
    REGION_EPS_DEG,
)


# ---------- 獨立實作的對照 ----------
def wedge_width_bruteforce(az_deg, frac=0.8):
    """以另一種列舉方式求最窄圓弧：對每個起點，取繞行方向上第 k 近的點的距離。

    與受測實作的「排序後滑動視窗」是不同的推導路徑，可作為交叉驗證。
    """
    a = np.mod(np.asarray(az_deg, dtype=float), 360.0)
    m = len(a)
    if m == 0:
        return None
    k = max(1, int(np.ceil(frac * m)))
    best = 360.0
    for i in range(m):
        d = np.sort(np.mod(a - a[i], 360.0))
        best = min(best, float(d[k - 1]))
    return best


def polygon_area(poly):
    """鞋帶公式。"""
    p = np.asarray(poly, dtype=float)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def point_in_hull(pt, hull, tol=1e-6):
    """凸包（逆時針）內外判定。`tol` 的單位是與輸入相同的長度，故外積須除以邊長。"""
    h = np.asarray(hull, dtype=float)
    if len(h) < 3:
        return True
    for i in range(len(h)):
        e = h[(i + 1) % len(h)] - h[i]
        L = float(np.linalg.norm(e))
        if L < 1e-12:
            continue
        v = np.asarray(pt, dtype=float) - h[i]
        if float(np.cross(e, v)) / L < -tol:
            return False
    return True


class TestWedgeWidth(unittest.TestCase):
    def test_known_cases(self):
        self.assertAlmostEqual(wedge_width(np.array([10.0] * 20), 1.0), 0.0, places=6)
        # 均勻散在四個象限，要全包住需要 270°。
        self.assertAlmostEqual(wedge_width(np.array([0.0, 90.0, 180.0, 270.0]), 1.0),
                               270.0, places=6)

    def test_crossing_seam(self):
        """跨 0°/360° 的一叢點，寬度應是實際跨幅。"""
        az = np.array([350.0, 355.0, 0.0, 5.0, 10.0])
        self.assertAlmostEqual(wedge_width(az, 1.0), 20.0, places=6)
        # 以 ±180 表示同一組點應得到相同答案。
        self.assertAlmostEqual(wedge_width(np.array([-10.0, -5.0, 0.0, 5.0, 10.0]), 1.0),
                               20.0, places=6)

    def test_matches_bruteforce(self):
        rng = np.random.default_rng(0)
        for trial in range(30):
            n = int(rng.integers(3, 60))
            style = trial % 3
            if style == 0:
                az = rng.uniform(0, 360, n)                       # 均勻散佈
            elif style == 1:
                az = rng.normal(0, 8, n)                          # 跨接縫的窄叢
            else:
                az = np.concatenate([rng.normal(30, 5, n // 2),
                                     rng.normal(210, 5, n - n // 2)])   # 雙叢
            for frac in (0.5, 0.8, 1.0):
                self.assertAlmostEqual(wedge_width(az, frac),
                                       wedge_width_bruteforce(az, frac), places=6,
                                       msg="trial=%d frac=%s" % (trial, frac))

    def test_empty_returns_none(self):
        self.assertIsNone(wedge_width(np.array([])))

    def test_frac_monotonic(self):
        """包住的比例越高，所需圓弧不會變窄。"""
        rng = np.random.default_rng(3)
        az = rng.uniform(0, 360, 50)
        widths = [wedge_width(az, f) for f in (0.2, 0.5, 0.8, 1.0)]
        for a, b in zip(widths, widths[1:]):
            self.assertLessEqual(a, b + 1e-9)


class TestAzCoverage(unittest.TestCase):
    def test_single_bin(self):
        self.assertAlmostEqual(az_coverage(np.array([0.0, 5.0, 9.9])), 1.0 / 36.0, places=9)

    def test_full_circle(self):
        self.assertAlmostEqual(az_coverage(np.arange(0.0, 360.0, 10.0)), 1.0, places=9)

    def test_negative_angles_normalised(self):
        """−5° 與 355° 是同一個 bin，不應算成兩個。"""
        self.assertAlmostEqual(az_coverage(np.array([-5.0, 355.0])), 1.0 / 36.0, places=9)

    def test_empty_returns_none(self):
        self.assertIsNone(az_coverage(np.array([])))


class TestConvexHull(unittest.TestCase):
    def test_square_with_interior_points(self):
        rng = np.random.default_rng(0)
        inner = rng.uniform(1, 9, (200, 2))
        pts = np.vstack([[[0, 0], [10, 0], [10, 10], [0, 10]], inner])
        hull = _convex_hull(pts)
        self.assertEqual(len(hull), 4)
        self.assertAlmostEqual(polygon_area(hull), 100.0, places=6)

    def test_all_points_inside_hull(self):
        rng = np.random.default_rng(1)
        pts = rng.normal(size=(300, 2)) * [50, 20] + [400, 100]
        hull = _convex_hull(pts)
        for p in pts:
            self.assertTrue(point_in_hull(p, hull), msg="point %s outside hull" % (p,))

    def test_counter_clockwise(self):
        """輸出須為逆時針，`_side_geom` 的面積計算與 `point_in_hull` 都依賴此方向。"""
        hull = _convex_hull(np.array([[0, 0], [4, 0], [4, 3], [0, 3], [2, 1]], dtype=float))
        p = np.asarray(hull, dtype=float)
        x, y = p[:, 0], p[:, 1]
        signed = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        self.assertGreater(signed, 0.0)

    def test_collinear_degenerates_to_zero_area(self):
        pts = np.array([[float(i), float(i)] for i in range(10)])
        self.assertAlmostEqual(polygon_area(_convex_hull(pts)), 0.0, places=9)

    def test_duplicate_and_few_points(self):
        self.assertEqual(len(_convex_hull(np.array([[1.0, 1.0]] * 5))), 1)
        self.assertEqual(len(_convex_hull(np.array([[0.0, 0.0], [1.0, 1.0]]))), 2)


class TestSideGeom(unittest.TestCase):
    def test_normal_region(self):
        rng = np.random.default_rng(0)
        px = rng.uniform(W_DEF * 0.55, W_DEF * 0.65, 200)
        py = rng.uniform(H_DEF * 0.4, H_DEF * 0.6, 200)
        g = _side_geom(px, py, W_DEF, H_DEF)
        self.assertLess(g["az_width_deg"], 60.0)
        lo, hi = g["az_range"]
        self.assertLess(lo, hi)
        self.assertGreater(lo, 0.0)          # 影像右半，方位為正
        self.assertLess(hi, 180.0)
        self.assertGreater(g["area_frac"], 0.0)
        self.assertLess(g["area_frac"], 1.0)

    def test_region_crossing_seam_is_continuous(self):
        """橫跨接縫的區域：方位範圍應連續跨過 180°，寬度仍是實際跨幅。"""
        rng = np.random.default_rng(1)
        px = np.concatenate([rng.uniform(0, W_DEF * 0.02, 100),
                             rng.uniform(W_DEF * 0.98, W_DEF, 100)])
        py = rng.uniform(H_DEF * 0.45, H_DEF * 0.55, 200)
        g = _side_geom(px, py, W_DEF, H_DEF)
        lo, hi = g["az_range"]
        self.assertLess(g["az_width_deg"], 30.0, "跨接縫的窄區域不應被算成寬區域")
        self.assertLess(lo, 180.0)
        self.assertGreater(hi, 180.0)
        # az_range 兩端與 az_width_deg 各自四捨五入到 0.1°，兩者相差可達 0.1。
        self.assertAlmostEqual(hi - lo, g["az_width_deg"], delta=0.11)
        self.assertLess(g["area_frac"], 0.05, "跨接縫時面積不應被拉成整張圖寬")

    def test_polygon_encloses_input(self):
        """多邊形須包住全部輸入點（頂點四捨五入到 0.1 像素，故容差取 0.5）。"""
        rng = np.random.default_rng(2)
        px = rng.uniform(W_DEF * 0.3, W_DEF * 0.4, 300)
        py = rng.uniform(H_DEF * 0.3, H_DEF * 0.7, 300)
        g = _side_geom(px, py, W_DEF, H_DEF)
        for x, y in zip(px, py):
            self.assertTrue(point_in_hull((x, y), g["polygon"], tol=0.5),
                            msg="(%.1f, %.1f) 落在多邊形外" % (x, y))

    def test_output_is_json_safe(self):
        import json
        rng = np.random.default_rng(3)
        g = _side_geom(rng.uniform(0, W_DEF, 50), rng.uniform(0, H_DEF, 50), W_DEF, H_DEF)
        json.dumps(g)                        # numpy 型別漏出時此處會拋出


class TestDbscan(unittest.TestCase):
    def test_two_separated_clusters(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(60, 2)) * 0.05
        b = rng.normal(size=(60, 2)) * 0.05 + 5.0
        labels = _dbscan(np.vstack([a, b]), eps=0.5, min_pts=5)
        self.assertEqual(len(set(labels[labels >= 0])), 2)
        self.assertEqual(len(set(labels[:60])), 1)
        self.assertEqual(len(set(labels[60:])), 1)
        self.assertNotEqual(labels[0], labels[60])

    def test_isolated_points_are_noise(self):
        rng = np.random.default_rng(1)
        core = rng.normal(size=(40, 2)) * 0.05
        far = np.array([[100.0, 100.0], [-100.0, -100.0]])
        labels = _dbscan(np.vstack([core, far]), eps=0.5, min_pts=5)
        self.assertTrue(np.all(labels[-2:] == -1))
        self.assertTrue(np.all(labels[:40] >= 0))

    def test_deterministic(self):
        rng = np.random.default_rng(2)
        X = np.vstack([rng.normal(size=(80, 3)) * 0.1,
                       rng.normal(size=(80, 3)) * 0.1 + 3.0])
        np.testing.assert_array_equal(_dbscan(X, 0.5, 8), _dbscan(X, 0.5, 8))

    def test_crosses_internal_chunk_boundary(self):
        """實作以 512 列為一塊計算距離；點數跨過該邊界時結果必須相同。"""
        rng = np.random.default_rng(3)
        X = np.vstack([rng.normal(size=(400, 2)) * 0.05,
                       rng.normal(size=(400, 2)) * 0.05 + 8.0])
        labels = _dbscan(X, 0.4, 10)
        self.assertEqual(len(set(labels[labels >= 0])), 2)
        self.assertEqual(len(set(labels[:400])), 1)
        self.assertEqual(len(set(labels[400:])), 1)

    def test_min_pts_suppresses_small_cluster(self):
        rng = np.random.default_rng(4)
        big = rng.normal(size=(50, 2)) * 0.05
        small = rng.normal(size=(3, 2)) * 0.05 + 10.0
        labels = _dbscan(np.vstack([big, small]), eps=0.5, min_pts=10)
        self.assertTrue(np.all(labels[-3:] == -1))


class TestExtractRegions(unittest.TestCase):
    """區域萃取：兩側必須同時連續才算同一區。"""

    @staticmethod
    def _bearings(az_deg, el_deg):
        az = np.radians(az_deg)
        el = np.radians(el_deg)
        return np.stack([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)], axis=1)

    def _make(self, az_a, az_b, n, rng, spread=2.0):
        aa = rng.normal(az_a, spread, n)
        bb = rng.normal(az_b, spread, n)
        el = rng.normal(0.0, spread, n)
        da = self._bearings(aa, el)
        db = self._bearings(bb, el)
        px_a = np.mod((aa + 180.0) / 360.0 * W_DEF, W_DEF)
        px_b = np.mod((bb + 180.0) / 360.0 * W_DEF, W_DEF)
        py = np.full(n, H_DEF / 2.0)
        mi = np.column_stack([px_a, py, px_b, py, rng.uniform(0.7, 1.0, n)])
        return mi, da, db

    def test_two_clusters_become_two_regions(self):
        rng = np.random.default_rng(0)
        m1, a1, b1 = self._make(-60.0, 20.0, 80, rng)
        m2, a2, b2 = self._make(100.0, 150.0, 80, rng)
        regions = extract_regions(np.vstack([m1, m2]), np.vstack([a1, a2]),
                                  np.vstack([b1, b2]), W_DEF, H_DEF, W_DEF, H_DEF)
        self.assertEqual(len(regions), 2)
        self.assertAlmostEqual(sum(r["frac_of_inliers"] for r in regions), 1.0, delta=0.05)

    def test_split_on_b_side_only(self):
        """A 側同一叢、B 側分成兩處 ⇒ 必須拆成兩區（6D 聯合聚類的目的）。"""
        rng = np.random.default_rng(1)
        m1, a1, b1 = self._make(0.0, 30.0, 80, rng)
        m2, a2, b2 = self._make(0.0, 160.0, 80, rng)
        regions = extract_regions(np.vstack([m1, m2]), np.vstack([a1, a2]),
                                  np.vstack([b1, b2]), W_DEF, H_DEF, W_DEF, H_DEF)
        self.assertEqual(len(regions), 2)

    def test_sorted_by_size_descending(self):
        rng = np.random.default_rng(2)
        m1, a1, b1 = self._make(-90.0, -30.0, 120, rng)
        m2, a2, b2 = self._make(60.0, 120.0, 40, rng)
        regions = extract_regions(np.vstack([m1, m2]), np.vstack([a1, a2]),
                                  np.vstack([b1, b2]), W_DEF, H_DEF, W_DEF, H_DEF)
        self.assertEqual([r["n"] for r in regions], sorted((r["n"] for r in regions), reverse=True))

    def test_too_few_points_returns_empty(self):
        rng = np.random.default_rng(3)
        mi, da, db = self._make(0.0, 0.0, 5, rng)
        self.assertEqual(extract_regions(mi, da, db, W_DEF, H_DEF, W_DEF, H_DEF), [])

    def test_eps_constant_is_declared_in_degrees(self):
        """鄰接半徑以角度宣告，換算成弦距後應落在合理範圍。"""
        self.assertGreater(REGION_EPS_DEG, 0.0)
        self.assertLess(REGION_EPS_DEG, 90.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
