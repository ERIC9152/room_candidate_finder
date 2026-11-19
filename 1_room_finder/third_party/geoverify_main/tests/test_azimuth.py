"""
方位角規約的一致性測試。

專案內有四處各自實作方位角換算：`upstream_direction`（上游輸入）、
`batch_pose_from_cache.px_to_az`（幾何驗證）、`estimate_relative_pose.erp_to_bearing`
（球面射線）、以及 `stitch_tool` 前端的反解。四者若不一致，方向會整批偏移或反向，
而且不會有任何例外或警告。本檔把規約本身寫成斷言，鎖住這條界線。

規約：az(x) = ((x / W) × 360 − 180) mod 360，0° = 影像中央，右為正。
"""
import math
import unittest

import numpy as np

from _support import W_DEF, H_DEF, bearing_to_erp, circ_diff  # noqa: F401（匯入時補 sys.path）

import upstream_direction as ud
from pose_regions import erp_to_bearing
from pose_regions.batch_pose_from_cache import px_to_az, circ_err


class TestConvention(unittest.TestCase):
    """規約的定點：由定義推得，不參照實作。"""

    def test_image_center_is_zero(self):
        self.assertAlmostEqual(ud.from_x(W_DEF / 2, W_DEF), 0.0, places=9)
        self.assertAlmostEqual(px_to_az(W_DEF / 2, W_DEF), 0.0, places=9)

    def test_seam_is_180(self):
        # 影像左右緣皆為接縫：x=0 對應 −180°（正規化後 180°），x=W 同一點。
        self.assertAlmostEqual(ud.from_x(0, W_DEF), 180.0, places=9)
        self.assertAlmostEqual(px_to_az(0, W_DEF), -180.0, places=9)
        self.assertAlmostEqual(circ_diff(ud.from_x(0, W_DEF), px_to_az(0, W_DEF)), 0.0, places=9)

    def test_right_of_center_is_positive(self):
        # 右為正：中央偏右四分之一處應為 +90°。
        self.assertAlmostEqual(px_to_az(W_DEF * 0.75, W_DEF), 90.0, places=9)
        self.assertAlmostEqual(ud.from_x(W_DEF * 0.75, W_DEF), 90.0, places=9)

    def test_two_implementations_agree(self):
        """`upstream_direction.from_x` 與 `px_to_az` 僅差在輸出區間。"""
        for x in np.linspace(0, W_DEF, 97):
            self.assertAlmostEqual(circ_diff(ud.from_x(x, W_DEF), px_to_az(x, W_DEF)),
                                   0.0, places=6, msg="x=%.1f" % x)

    def test_output_range(self):
        for x in (0, 1, W_DEF / 3, W_DEF - 1):
            self.assertGreaterEqual(ud.from_x(x, W_DEF), 0.0)
            self.assertLess(ud.from_x(x, W_DEF), 360.0)
            self.assertGreaterEqual(px_to_az(x, W_DEF), -180.0)
            self.assertLess(px_to_az(x, W_DEF), 180.0)

    def test_width_is_per_house(self):
        """寬度不得寫死：同一比例位置在不同寬度下應得到相同方位。"""
        for W in (5376, 6144, 8192):
            self.assertAlmostEqual(ud.from_x(W * 0.25, W), 270.0, places=6)


class TestBearingConsistency(unittest.TestCase):
    """球面射線與像素方位必須指向同一個方向。"""

    def test_bearing_azimuth_matches_pixel_azimuth(self):
        xs = np.linspace(0, W_DEF, 37)[:-1]
        pts = np.stack([xs, np.full_like(xs, H_DEF / 2)], axis=1)
        d = erp_to_bearing(pts, W_DEF, H_DEF)
        az_from_bearing = np.degrees(np.arctan2(d[:, 0], d[:, 2]))
        for x, azb in zip(xs, az_from_bearing):
            self.assertAlmostEqual(circ_diff(azb, px_to_az(x, W_DEF)), 0.0, places=6,
                                   msg="x=%.1f" % x)

    def test_bearing_is_unit_length(self):
        rng = np.random.default_rng(0)
        pts = np.column_stack([rng.uniform(0, W_DEF, 500), rng.uniform(0, H_DEF, 500)])
        d = erp_to_bearing(pts, W_DEF, H_DEF)
        np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-12)

    def test_round_trip_pixel_bearing_pixel(self):
        rng = np.random.default_rng(1)
        # 避開極點附近（|lat| > 89°）：該處經度的數值條件差，非規約問題。
        pts = np.column_stack([rng.uniform(0, W_DEF, 500),
                               rng.uniform(H_DEF * 0.02, H_DEF * 0.98, 500)])
        back = bearing_to_erp(erp_to_bearing(pts, W_DEF, H_DEF), W_DEF, H_DEF)
        np.testing.assert_allclose(np.mod(back[:, 0], W_DEF), np.mod(pts[:, 0], W_DEF), atol=1e-6)
        np.testing.assert_allclose(back[:, 1], pts[:, 1], atol=1e-6)

    def test_vertical_axis_orientation(self):
        """影像上緣為天頂：y 小 → lat 大 → 方向向量的 Y 分量為正。"""
        d = erp_to_bearing(np.array([[W_DEF / 2, H_DEF * 0.1]]), W_DEF, H_DEF)
        self.assertGreater(d[0, 1], 0.0)
        d = erp_to_bearing(np.array([[W_DEF / 2, H_DEF * 0.9]]), W_DEF, H_DEF)
        self.assertLess(d[0, 1], 0.0)


class TestFromSinCos(unittest.TestCase):
    """矩陣 ch1/ch2 的解碼（上游交界）。"""

    def test_round_trip(self):
        for az in (0.0, 45.0, 90.0, 179.9, 180.0, 270.0, 359.9):
            s, c = math.sin(math.radians(az)), math.cos(math.radians(az))
            self.assertAlmostEqual(circ_diff(ud.from_sincos(s, c), az), 0.0, places=9,
                                   msg="az=%s" % az)

    def test_accepts_numpy_scalars(self):
        m = np.zeros((1, 1, 3), dtype=np.float32)
        m[0, 0, 1], m[0, 0, 2] = np.float32(1.0), np.float32(0.0)
        self.assertAlmostEqual(ud.from_sincos(m[0, 0, 1], m[0, 0, 2]), 90.0, places=4)

    def test_unwritten_entry_decodes_to_zero(self):
        """矩陣未填方位時 ch1=ch2=0，解出 0°。呼叫端須自行判斷有效性。"""
        self.assertAlmostEqual(ud.from_sincos(0.0, 0.0), 0.0, places=9)


class TestFromBox(unittest.TestCase):
    """框 → 方位角，含跨接縫。"""

    def test_normal_box_uses_center(self):
        self.assertAlmostEqual(ud.from_box((3000, 100, 5000, 900), W_DEF),
                               ud.from_x(4000, W_DEF), places=9)

    def test_box_crossing_seam(self):
        """x0 > x1 表示框橫跨接縫；中心必須繞過接縫。"""
        W = 8192
        got = ud.from_box((8000, 0, 200, 100), W)
        # 由定義獨立算：8000 → 8192 → 0 → 200，弧長 392，中心在 8000+196 = 8196 ≡ 4。
        self.assertAlmostEqual(circ_diff(got, ud.from_x(4, W)), 0.0, places=6)
        # 算術平均會落在影像正中央（4100 ≈ 0°），此為必須避免的錯誤答案。
        self.assertGreater(circ_diff(got, ud.from_x(4100, W)), 170.0)

    def test_seam_box_stays_near_seam(self):
        self.assertLess(circ_diff(ud.from_box((8100, 0, 100, 10), W_DEF), 180.0), 5.0)

    def test_convention_string_is_declared(self):
        """規約字串是對外契約的一部分。"""
        self.assertIn("image center", ud.CONVENTION)
        self.assertIn("360", ud.CONVENTION)


class TestConventionStringAgreement(unittest.TestCase):
    """house JSON 寫出的 az_convention 須與 `upstream_direction.CONVENTION` 同義。

    寫出處目前直接引用該常數，故此處驗證的是「確實引用了同一個來源」：
    若改回自行複製一份字面值，則退回比對分號之前的公式。兩者若分歧，前端的貼圖座標
    與後端的方位角就會整批偏移。
    """

    #: 寫出處應引用的常數名稱。
    EXPECTED_REF = "upstream_direction.CONVENTION"

    @staticmethod
    def _formula(s):
        return (s or "").split(";")[0].strip().replace(" ", "")

    def test_formula_matches_house_json_writer(self):
        import os
        import re
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "stitch_tool", "prepare_house_data.py")
        with open(path, encoding="utf-8") as f:
            src = f.read()
        m = re.search(r'"az_convention":\s*(?:"([^"]+)"|([A-Za-z0-9_.]+))', src)
        self.assertIsNotNone(m, "prepare_house_data.py 未寫出 az_convention 欄位")
        literal, ref = m.group(1), m.group(2)
        if literal is not None:
            self.assertEqual(self._formula(literal), self._formula(ud.CONVENTION))
        else:
            # 引用其他常數時比對會變成同義反覆，故直接檢查引用的是哪一個。
            self.assertEqual(ref, self.EXPECTED_REF,
                             "az_convention 應引用 %s，實得 %s" % (self.EXPECTED_REF, ref))

    def test_formula_is_the_implemented_one(self):
        """公式字串須與 from_x 的實際行為相符，而不只是註解。"""
        self.assertEqual(self._formula(ud.CONVENTION), "az(x)=((x/W)*360-180)%360")
        for x, W in ((0, 8192), (2048, 8192), (4096, 8192), (6144, 8192), (1344, 5376)):
            self.assertAlmostEqual(ud.from_x(x, W), ((x / W) * 360.0 - 180.0) % 360.0, places=9)


class TestCircErr(unittest.TestCase):
    def test_wraps_around_seam(self):
        self.assertAlmostEqual(circ_err(1.0, 359.0), 2.0, places=9)
        self.assertAlmostEqual(circ_err(-179.0, 179.0), 2.0, places=9)

    def test_maximum_is_180(self):
        for a, b in ((0.0, 180.0), (90.0, 270.0), (-90.0, 90.0)):
            self.assertLessEqual(circ_err(a, b), 180.0 + 1e-9)
            self.assertAlmostEqual(circ_err(a, b), 180.0, places=9)

    def test_symmetric(self):
        rng = np.random.default_rng(2)
        for a, b in rng.uniform(-360, 360, (200, 2)):
            self.assertAlmostEqual(circ_err(a, b), circ_err(b, a), places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
