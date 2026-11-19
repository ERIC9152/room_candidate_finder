"""
`analyze_pair` 的契約測試：schema 完整性、自適應 conf 階梯、旗標定義、決定性。

`analyze_pair` 是幾何驗證層的單一進入點，其回傳值即 per-pair JSON 的主體，下游按
schema 讀取。本檔驗證的是**契約**：欄位齊全、型別可序列化、旗標符合
文件定義、相同輸入得到相同輸出。位姿數值本身的正確性在 `test_pose_estimation.py`。

全部以合成匹配進行，不需要 GPU、資料集或快取。
"""
import json
import unittest

import numpy as np

from _support import (W_DEF, H_DEF, az_ab_of, circ_diff, cone_matches,  # noqa: F401
                      rot_matrix, synth_matches)

from pose_regions import analyze_pair
from pose_regions.batch_pose_from_cache import SCHEMA_VERSION

#: 與 `pose_regions/batch_pose_from_cache.py` 的 argparse 預設一致，
#: 亦即 `roma_geo.DEFAULTS`。此處刻意重列，兩邊不一致時測試會發現。
P = dict(conf_min=0.65, max_pts=4000, thresh_deg=0.5, iters=2000,
         seed=0, sampler="stratified", retry_conf=0.3, lat_max=60.0)

TOP_KEYS = {"schema_version", "img_size_a", "img_size_b", "params", "n", "conf_used",
            "inlier_rate", "pose", "quality", "flags", "regions", "inliers"}
POSE_KEYS = {"R", "t", "cam_b_in_a", "az_ab_deg", "az_ba_deg", "rot_angle_deg", "baseline_txz"}
QUALITY_KEYS = {"sampson_med_deg", "sampson_p90_deg", "rot_only_med_deg",
                "az_wedge_a_deg", "az_wedge_b_deg", "az_cov_a", "az_cov_b"}
FLAG_KEYS = {"low_matches", "no_covisibility", "degenerate_rot_only", "doorway_wedge"}


def run(m, **kw):
    p = dict(P, **kw)
    return analyze_pair(m, W_DEF, H_DEF, W_DEF, H_DEF, **p)


class TestSchema(unittest.TestCase):
    def setUp(self):
        m, self.R_gt, self.t_gt = synth_matches(n=300, outlier_frac=0.15, seed=0)
        self.res = run(m, iters=500)

    def test_top_level_keys(self):
        self.assertEqual(set(self.res) & TOP_KEYS, TOP_KEYS)
        self.assertEqual(self.res["schema_version"], SCHEMA_VERSION)

    def test_pose_and_quality_keys(self):
        self.assertEqual(set(self.res["pose"]), POSE_KEYS)
        self.assertEqual(set(self.res["quality"]), QUALITY_KEYS)
        self.assertEqual(set(self.res["flags"]), FLAG_KEYS)

    def test_params_echo_inputs(self):
        """params 必須回報實際採用的參數，下游據此判斷結果可否互相比較。"""
        self.assertEqual(self.res["params"]["conf_min"], P["conf_min"])
        self.assertEqual(self.res["params"]["retry_conf"], P["retry_conf"])
        self.assertEqual(self.res["params"]["thresh_deg"], P["thresh_deg"])
        self.assertEqual(self.res["params"]["sampler"], P["sampler"])
        self.assertEqual(self.res["params"]["lat_max"], P["lat_max"])
        self.assertEqual(self.res["params"]["ransac_iters"], 500)

    def test_json_serialisable(self):
        """numpy 純量若漏進結果，批次寫檔會在執行到一半時失敗。"""
        json.dumps(self.res)

    def test_image_sizes_recorded(self):
        self.assertEqual(self.res["img_size_a"], [W_DEF, H_DEF])
        self.assertEqual(self.res["img_size_b"], [W_DEF, H_DEF])

    def test_counts_are_consistent(self):
        n = self.res["n"]
        self.assertEqual(n["raw"], 300)
        self.assertLessEqual(n["inlier"], n["used"])
        self.assertLessEqual(n["used"], n["conf"])
        self.assertAlmostEqual(self.res["inlier_rate"], n["inlier"] / n["used"], places=4)

    def test_lat_drop_field_present_when_filtering(self):
        self.assertIn("lat_drop", self.res["n"])
        self.assertNotIn("lat_max", run(np.empty((0, 5)), lat_max=None)["params"])


class TestPoseRecovery(unittest.TestCase):
    def test_azimuth_matches_ground_truth(self):
        R_gt = rot_matrix([0.0, 1.0, 0.0], 25.0)
        t_gt = np.array([1.0, 0.0, 0.4])
        m, _, _ = synth_matches(n=400, R=R_gt, t=t_gt, outlier_frac=0.2, seed=1)
        res = run(m)
        self.assertIsNotNone(res["pose"])
        self.assertLess(circ_diff(res["pose"]["az_ab_deg"], az_ab_of(R_gt, t_gt)), 2.0)

    def test_reverse_azimuth_is_from_translation(self):
        """az_ba 取自 t 本身（A 相機在 B 幀的方向），與 az_ab 是不同的量。"""
        R_gt = rot_matrix([0.0, 1.0, 0.0], 40.0)
        t_gt = np.array([1.0, 0.0, 0.0])
        m, _, _ = synth_matches(n=300, R=R_gt, t=t_gt, seed=2)
        p = run(m, iters=500)["pose"]
        expect = float(np.degrees(np.arctan2(t_gt[0], t_gt[2])))
        self.assertLess(circ_diff(p["az_ba_deg"], expect), 2.0)

    def test_rotation_angle_reported(self):
        m, _, _ = synth_matches(n=300, R=rot_matrix([0, 1, 0], 30.0), seed=3)
        self.assertAlmostEqual(run(m, iters=500)["pose"]["rot_angle_deg"], 30.0, delta=1.0)

    def test_regions_extracted(self):
        m, _, _ = synth_matches(n=400, seed=4)
        res = run(m, iters=500)
        self.assertGreaterEqual(len(res["regions"]), 1)
        self.assertLessEqual(sum(r["frac_of_inliers"] for r in res["regions"]), 1.001)
        for r in res["regions"]:
            self.assertGreaterEqual(r["n"], 8)


class TestAdaptiveConfidence(unittest.TestCase):
    """自適應 conf 階梯：0.65 級點數不足時降到 retry_conf 重試。"""

    @staticmethod
    def _tiered(n_high, n_low, seed=0):
        m, R, t = synth_matches(n=n_high + n_low, seed=seed)
        m[:n_high, 4] = 0.9          # 高 conf 段
        m[n_high:, 4] = 0.4          # 低於 0.65、高於 0.3
        return m, R, t

    def test_uses_conf_min_when_enough_points(self):
        m, _, _ = self._tiered(200, 100)
        res = run(m, iters=500)
        self.assertEqual(res["conf_used"], 0.65)
        self.assertEqual(res["n"]["used"], 200)
        self.assertFalse(res["flags"]["low_matches"])

    def test_falls_back_to_retry_conf(self):
        """0.65 級只有 20 點（<30）⇒ 降到 0.3，使用全部 220 點。"""
        m, _, _ = self._tiered(20, 200)
        res = run(m, iters=500)
        self.assertEqual(res["conf_used"], 0.3)
        self.assertEqual(res["n"]["used"], 220)
        self.assertTrue(res["flags"]["low_matches"])
        self.assertFalse(res["flags"]["no_covisibility"])
        self.assertIsNotNone(res["pose"])

    def test_n_conf_counts_the_tier_actually_used(self):
        """⚠ `n.conf` 記的是 `conf_used` 那一層的點數，`low_matches` 看的是 `conf_min` 層。

        兩個欄位用不同門檻，因此撈回級會出現「low_matches 為真但 n.conf 很大」的組合，
        這是預期行為。下游要判斷證據強度應該看 `conf_used` 與 `n.inlier`，不要看 `n.conf`。
        """
        m, _, _ = self._tiered(20, 200)
        res = run(m, iters=500)
        self.assertEqual(res["n"]["conf"], 220)
        self.assertTrue(res["flags"]["low_matches"])

    def test_no_covisibility_when_both_tiers_short(self):
        m, _, _ = synth_matches(n=5, seed=0)
        res = run(m)
        self.assertTrue(res["flags"]["no_covisibility"])
        self.assertIsNone(res["pose"])
        self.assertIsNone(res["conf_used"])
        self.assertIsNone(res["inlier_rate"])
        self.assertEqual(res["regions"], [])
        self.assertEqual(res["inliers"], [])

    def test_empty_input(self):
        res = run(np.empty((0, 5)))
        self.assertTrue(res["flags"]["no_covisibility"])
        self.assertEqual(res["n"]["raw"], 0)
        json.dumps(res)

    def test_no_covisibility_means_no_evidence_not_disconnected(self):
        """語意檢核：無位姿時仍須留下計數，下游才能分辨「沒證據」與「沒跑」。"""
        m, _, _ = synth_matches(n=5, seed=0)
        res = run(m)
        self.assertEqual(res["n"]["raw"], 5)
        self.assertIn("lat_drop", res["n"])


class TestLatitudeFilter(unittest.TestCase):
    """天底（腳架帶）與天頂前濾。"""

    def test_drops_points_beyond_lat_max(self):
        m, _, _ = synth_matches(n=200, seed=0)
        # 追加 40 個天底附近的點（|lat| > 60° ⇒ y 超出中間 2/3 的範圍）。
        bad = m[:40].copy()
        bad[:, 1] = H_DEF * 0.97
        bad[:, 3] = H_DEF * 0.97
        res = run(np.vstack([m, bad]), iters=500)
        self.assertEqual(res["n"]["raw"], 240)
        self.assertEqual(res["n"]["lat_drop"], 40)
        self.assertLessEqual(res["n"]["used"], 200)

    def test_either_side_beyond_limit_is_dropped(self):
        """只要任一側超出即丟棄，不必兩側都超。"""
        m, _, _ = synth_matches(n=200, seed=1)
        bad = m[:30].copy()
        bad[:, 3] = H_DEF * 0.02          # 只有 B 側在天頂
        res = run(np.vstack([m, bad]), iters=500)
        self.assertEqual(res["n"]["lat_drop"], 30)

    def test_disabled_when_lat_max_is_none(self):
        m, _, _ = synth_matches(n=200, seed=2)
        bad = m[:40].copy()
        bad[:, 1] = H_DEF * 0.97
        res = run(np.vstack([m, bad]), lat_max=None, iters=500)
        self.assertNotIn("lat_drop", res["n"])
        self.assertNotIn("lat_max", res["params"])


class TestFlags(unittest.TestCase):
    def test_degenerate_when_near_zero_baseline(self):
        """近零基線：觀測可由純旋轉解釋 ⇒ t 方向不可靠。"""
        m, _, _ = synth_matches(n=300, R=rot_matrix([0, 1, 0], 30.0),
                                t=np.array([1e-4, 0.0, 0.0]), seed=0)
        res = run(m, iters=500)
        self.assertTrue(res["flags"]["degenerate_rot_only"])
        self.assertFalse(res["flags"]["doorway_wedge"], "退化的 pair 不得同時標為門洞")

    def test_not_degenerate_with_real_baseline(self):
        m, _, _ = synth_matches(n=300, t=np.array([1.5, 0.0, 0.0]), seed=1)
        res = run(m, iters=500)
        self.assertFalse(res["flags"]["degenerate_rot_only"])

    def test_doorway_wedge_on_narrow_support(self):
        """窄楔形但證據充分 ⇒ 門洞旗標。這是穿門 pair 與退化垃圾的分界。"""
        m, _, _ = cone_matches(n=300, half_angle_deg=14.0, seed=0)
        res = run(m, iters=1000)
        q, f = res["quality"], res["flags"]
        self.assertFalse(f["degenerate_rot_only"])
        self.assertGreaterEqual(res["inlier_rate"], 0.3)
        self.assertGreaterEqual(res["n"]["inlier"], 30)
        self.assertLessEqual(q["az_wedge_a_deg"], 75.0)
        self.assertLessEqual(q["az_wedge_b_deg"], 75.0)
        self.assertTrue(f["doorway_wedge"])

    def test_wide_support_is_not_doorway(self):
        m, _, _ = synth_matches(n=400, spread=(12.0, 3.0, 4.0), seed=2)
        res = run(m, iters=500)
        if res["quality"]["az_wedge_a_deg"] > 75.0:
            self.assertFalse(res["flags"]["doorway_wedge"])

    def test_low_matches_threshold_is_thirty(self):
        m, _, _ = synth_matches(n=40, seed=3)
        m[:11, 4] = 0.2                    # 只剩 29 點達到 0.65
        self.assertTrue(run(m, iters=500)["flags"]["low_matches"])
        m[10, 4] = 0.9                     # 補到 30 點
        self.assertFalse(run(m, iters=500)["flags"]["low_matches"])


class TestDeterminism(unittest.TestCase):
    def test_same_input_same_output(self):
        m, _, _ = synth_matches(n=400, outlier_frac=0.25, seed=0)
        a = json.dumps(run(m, iters=500), sort_keys=True)
        b = json.dumps(run(m, iters=500), sort_keys=True)
        self.assertEqual(a, b, "相同輸入必須得到逐位元相同的結果")

    def test_seed_changes_sampling_only_slightly(self):
        """不同 seed 之下位姿結論須一致（方位差在容許範圍內）。"""
        m, R_gt, t_gt = synth_matches(n=400, outlier_frac=0.25, seed=1)
        a = run(m, seed=0, iters=1000)["pose"]["az_ab_deg"]
        b = run(m, seed=12345, iters=1000)["pose"]["az_ab_deg"]
        self.assertLess(circ_diff(a, b), 3.0)

    def test_inliers_capped_at_two_thousand(self):
        m, _, _ = synth_matches(n=2400, seed=2)
        res = run(m, iters=300)
        self.assertGreater(res["n"]["inlier"], 2000)
        self.assertEqual(len(res["inliers"]), 2000, "inliers 欄位須抽樣至 2000 點上限")
        for row in res["inliers"][:20]:
            self.assertEqual(len(row), 5)


class TestAsymmetricImageSizes(unittest.TestCase):
    """兩張圖尺寸不同時，各側須用各自的寬高換算。"""

    def test_different_sizes(self):
        R_gt = rot_matrix([0, 1, 0], 20.0)
        t_gt = np.array([1.0, 0.0, 0.2])
        Wb, Hb = 5376, 2688
        m, _, _ = synth_matches(n=300, R=R_gt, t=t_gt, seed=0)
        # B 側改用另一組尺寸重新投影。
        from _support import bearing_to_erp
        from pose_regions import erp_to_bearing
        db = erp_to_bearing(m[:, 2:4], W_DEF, H_DEF)
        m[:, 2:4] = bearing_to_erp(db, Wb, Hb)
        res = analyze_pair(m, W_DEF, H_DEF, Wb, Hb, **dict(P, iters=500))
        self.assertEqual(res["img_size_b"], [Wb, Hb])
        self.assertLess(circ_diff(res["pose"]["az_ab_deg"], az_ab_of(R_gt, t_gt)), 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
