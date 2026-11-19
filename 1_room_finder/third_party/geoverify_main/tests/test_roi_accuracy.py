"""廠商 ROI 精確度：GT 通道方位是否落在交付的共視區域內。

GT hotspot 是 ERP 上的一點（方位），不是多邊形。能客觀量的是：
  1. 通道方位 az_ab 與 GT 的圓角誤差
  2. GT 是否落在主共視區 az_range（交付檔的 ROI）
  3. 以 az_ab 為中心、固定窗寬裁切時能否包住 GT

期望值由圓弧規約獨立推得，不從評估腳本反推。
"""
import unittest

from _support import temp_dir, write_json  # noqa: F401（匯入時補 sys.path）

from roma_geo.semantic import az_contains, circ_err
from pose_regions.roi_accuracy_eval import score_pair, WINDOW_DEGS


class TestCircErr(unittest.TestCase):
    def test_identical_is_zero(self):
        self.assertAlmostEqual(circ_err(10.0, 10.0), 0.0)
        self.assertAlmostEqual(circ_err(-10.0, 350.0), 0.0)

    def test_shortest_arc_not_the_long_way(self):
        self.assertAlmostEqual(circ_err(10.0, 350.0), 20.0)
        self.assertAlmostEqual(circ_err(179.0, -179.0), 2.0)


class TestAzContains(unittest.TestCase):
    def test_interior_and_endpoints_of_a_simple_span(self):
        self.assertTrue(az_contains(30.0, 10.0, 50.0))
        self.assertTrue(az_contains(10.0, 10.0, 50.0))
        self.assertTrue(az_contains(50.0, 10.0, 50.0))
        self.assertFalse(az_contains(0.0, 10.0, 50.0))
        self.assertFalse(az_contains(90.0, 10.0, 50.0))

    def test_span_crossing_the_seam(self):
        # az_range 規約：正向走到 a1，可過縫。[-20, 10] 寬 30°。
        self.assertTrue(az_contains(355.0, -20.0, 10.0))
        self.assertTrue(az_contains(0.0, -20.0, 10.0))
        self.assertTrue(az_contains(5.0, -20.0, 10.0))
        self.assertFalse(az_contains(180.0, -20.0, 10.0))

    def test_signed_range_from_real_json(self):
        # 58553 一對主區 az_range=[-150.2, -71.7]，位姿 az_ab=-91.31。
        self.assertTrue(az_contains(-91.31, -150.2, -71.7))
        self.assertFalse(az_contains(0.0, -150.2, -71.7))

    def test_empty_span_contains_nothing(self):
        self.assertFalse(az_contains(10.0, 10.0, 10.0))

    def test_full_circle_contains_every_heading(self):
        self.assertTrue(az_contains(0.0, 0.0, 360.0))
        self.assertTrue(az_contains(179.0, 0.0, 360.0))


class TestScorePair(unittest.TestCase):
    def _pair(self, **kw):
        rec = {
            "pose": {"az_ab_deg": -91.31, "az_ba_deg": 177.65},
            "regions": [{
                "a": {"az_range": [-150.2, -71.7], "az_width_deg": 78.5},
                "b": {"az_range": [-130.3, 60.5], "az_width_deg": 190.8},
            }],
            "gt": {"label": 1, "same_room": False,
                   "az_ab_deg": -95.0, "az_ba_deg": 0.0},
            "flags": {"doorway_wedge": False, "no_covisibility": False},
            "conf_used": 0.65,
        }
        rec.update(kw)
        return rec

    def test_gt_inside_main_region_counts_as_hit(self):
        s = score_pair(self._pair())
        self.assertTrue(s["usable"])
        self.assertTrue(s["region_hit_a"])
        self.assertTrue(s["region_hit_b"])
        self.assertAlmostEqual(s["err_az_ab"], 3.69, places=2)
        self.assertAlmostEqual(s["region_width_a"], 78.5)

    def test_gt_outside_region_is_a_miss(self):
        rec = self._pair(gt={"label": 1, "same_room": False,
                             "az_ab_deg": 0.0, "az_ba_deg": 0.0})
        s = score_pair(rec)
        self.assertTrue(s["usable"])
        self.assertFalse(s["region_hit_a"])

    def test_no_gt_azimuth_is_not_usable(self):
        rec = self._pair(gt={"label": 0, "same_room": False,
                             "az_ab_deg": None, "az_ba_deg": None})
        self.assertFalse(score_pair(rec)["usable"])

    def test_no_pose_is_not_usable(self):
        rec = self._pair(pose=None, regions=[])
        self.assertFalse(score_pair(rec)["usable"])

    def test_narrow_window_around_az_ab_misses_far_gt(self):
        rec = self._pair(gt={"label": 1, "same_room": False,
                             "az_ab_deg": 0.0, "az_ba_deg": 170.0})
        s = score_pair(rec)
        self.assertIn(15, WINDOW_DEGS)
        self.assertFalse(s["window_hit"][15])
        self.assertTrue(s["window_hit"][180])
