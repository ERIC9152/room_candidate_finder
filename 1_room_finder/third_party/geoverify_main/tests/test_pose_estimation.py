"""
位姿估計核心測試：RANSAC 本質矩陣、位姿分解、退化偵測。

期望值來自合成資料的**已知真值**：以指定的 R、t 正向投影出兩側觀測，受測程式碼必須
解回同一組 R、t。這是唯一能區分「演算法正確」與「演算法穩定地錯」的作法。

`estimate_relative_pose.selftest()` 原本即以此方式驗證，本檔一併納入，
使其進入單一指令的涵蓋範圍。該函式同時是 `environments/check_env.py` 的三項實跑之一，
故保留在原模組，不搬移。
"""
import unittest

import numpy as np

from _support import W_DEF, H_DEF, rot_matrix, synth_matches, circ_diff, az_ab_of  # noqa: F401

from pose_regions import erp_to_bearing, ransac_E, select_pose, rotation_only_fit, sampson_residual
from pose_regions.estimate_relative_pose import decompose_E, eight_point_E, selftest, triangulate

THRESH = np.sin(np.radians(0.5))          # 正式參數：對極內點門檻 0.5°


def bearings_of(m, W=W_DEF, H=H_DEF):
    return erp_to_bearing(m[:, :2], W, H), erp_to_bearing(m[:, 2:4], W, H)


def rot_err_deg(R, R_gt):
    c = (np.trace(np.asarray(R).T @ np.asarray(R_gt)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def dir_err_deg(t, t_gt):
    a = np.asarray(t, dtype=float) / np.linalg.norm(t)
    b = np.asarray(t_gt, dtype=float) / np.linalg.norm(t_gt)
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), -1.0, 1.0))))


class TestSelfTest(unittest.TestCase):
    def test_geometry_selftest_passes(self):
        """`--selftest` 的合成資料驗證（uniform 與 stratified 兩種取樣皆須通過）。

        該函式設計成 CLI 使用，會印出逐項結果；此處收走它的輸出，失敗時才附進訊息。
        """
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ok = selftest()
        self.assertTrue(ok, buf.getvalue())


class TestEightPoint(unittest.TestCase):
    def test_exact_fit_on_clean_data(self):
        m, R_gt, t_gt = synth_matches(n=200, seed=0)
        da, db = bearings_of(m)
        E = eight_point_E(da, db)
        res = sampson_residual(E, da, db)
        self.assertLess(float(np.max(res)), 1e-6, "無雜訊資料的對極殘差應接近零")

    def test_essential_matrix_has_two_equal_singular_values(self):
        """本質矩陣的奇異值須為 (σ, σ, 0)，實作以投影回流形保證此性質。"""
        m, _, _ = synth_matches(n=120, seed=1)
        da, db = bearings_of(m)
        s = np.linalg.svd(eight_point_E(da, db), compute_uv=False)
        self.assertAlmostEqual(s[0], s[1], delta=1e-9 + 1e-6 * s[0])
        self.assertLess(s[2], 1e-9 + 1e-6 * s[0])


class TestRansac(unittest.TestCase):
    def test_recovers_pose_without_outliers(self):
        R_gt = rot_matrix([0.1, 1.0, 0.2], 25.0)
        t_gt = np.array([1.0, 0.1, 0.4])
        m, _, _ = synth_matches(n=300, R=R_gt, t=t_gt, seed=0)
        da, db = bearings_of(m)
        E, inl = ransac_E(da, db, THRESH, iters=500, seed=0, conf=m[:, 4])
        R, t = select_pose(E, da, db, inl)
        self.assertGreater(int(inl.sum()), 290)
        self.assertLess(rot_err_deg(R, R_gt), 1.0)
        self.assertLess(dir_err_deg(t, t_gt), 2.0)

    def test_rejects_outliers(self):
        m, R_gt, t_gt = synth_matches(n=400, outlier_frac=0.35, seed=2)
        n_out = int(round(0.35 * 400))
        da, db = bearings_of(m)
        E, inl = ransac_E(da, db, THRESH, iters=2000, seed=0, conf=m[:, 4])
        # 前 n_out 列為外點：內點集合應幾乎完全避開它們。
        self.assertLess(int(inl[:n_out].sum()), 0.05 * n_out, "外點被大量誤收為內點")
        self.assertGreater(int(inl[n_out:].sum()), 0.9 * (400 - n_out), "內點被大量漏收")
        R, t = select_pose(E, da, db, inl)
        self.assertLess(rot_err_deg(R, R_gt), 1.5)
        self.assertLess(dir_err_deg(t, t_gt), 3.0)

    def test_both_samplers_recover_pose(self):
        m, R_gt, t_gt = synth_matches(n=300, outlier_frac=0.2, seed=3)
        da, db = bearings_of(m)
        for sampler, conf in (("uniform", None), ("stratified", m[:, 4])):
            E, inl = ransac_E(da, db, THRESH, iters=2000, seed=0, conf=conf, sampler=sampler)
            R, t = select_pose(E, da, db, inl)
            self.assertLess(rot_err_deg(R, R_gt), 1.5, msg=sampler)
            self.assertLess(dir_err_deg(t, t_gt), 3.0, msg=sampler)

    def test_deterministic_for_same_seed(self):
        m, _, _ = synth_matches(n=250, outlier_frac=0.3, seed=4)
        da, db = bearings_of(m)
        a = ransac_E(da, db, THRESH, iters=500, seed=7, conf=m[:, 4])
        b = ransac_E(da, db, THRESH, iters=500, seed=7, conf=m[:, 4])
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_allclose(a[0], b[0], atol=1e-12)

    def test_requires_eight_points(self):
        m, _, _ = synth_matches(n=7, seed=5)
        da, db = bearings_of(m)
        with self.assertRaises(ValueError):
            ransac_E(da, db, THRESH, iters=100, seed=0)

    def test_confidence_does_not_bias_inlier_decision(self):
        """conf 只影響取樣順序；內點判定須對全部點計算殘差，不得被 conf 篩掉。

        把真內點全部給低 conf、外點給高 conf，正確的實作仍應找回那些真內點。
        """
        m, _, _ = synth_matches(n=300, outlier_frac=0.25, seed=6)
        n_out = 75
        conf = np.empty(len(m))
        conf[:n_out] = 0.99          # 外點給高 conf
        conf[n_out:] = 0.31          # 內點給低 conf
        da, db = bearings_of(m)
        _, inl = ransac_E(da, db, THRESH, iters=2000, seed=0, conf=conf)
        self.assertGreater(int(inl[n_out:].sum()), 0.85 * (300 - n_out))


class TestSelectPose(unittest.TestCase):
    def test_four_candidates_from_decomposition(self):
        m, _, _ = synth_matches(n=100, seed=0)
        da, db = bearings_of(m)
        cands = decompose_E(eight_point_E(da, db))
        self.assertEqual(len(cands), 4)
        for R, _ in cands:
            self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=6,
                                   msg="旋轉矩陣的行列式必須為 +1（不得含鏡射）")

    def test_cheirality_picks_the_physical_solution(self):
        """四組候選中只有一組讓全部點落在兩台相機前方。"""
        R_gt = rot_matrix([0.0, 1.0, 0.0], 20.0)
        t_gt = np.array([1.0, 0.0, 0.3])
        m, _, _ = synth_matches(n=200, R=R_gt, t=t_gt, seed=1)
        da, db = bearings_of(m)
        E = eight_point_E(da, db)
        inl = np.ones(len(da), dtype=bool)
        R, t = select_pose(E, da, db, inl)
        la, lb = triangulate(da, db, R, t)
        self.assertGreater(float(np.mean((la > 0) & (lb > 0))), 0.98)
        self.assertLess(rot_err_deg(R, R_gt), 0.5)

    def test_azimuth_matches_ground_truth(self):
        """交付物的核心數值：az_ab 必須等於真值 −Rᵀt 的方位。"""
        for axis, ang, t_gt in (([0, 1, 0], 30.0, [1.0, 0.0, 0.0]),
                                ([0, 1, 0], -45.0, [0.3, 0.0, 1.0]),
                                ([0.2, 1, 0.1], 15.0, [-1.0, 0.05, 0.5])):
            R_gt = rot_matrix(axis, ang)
            m, _, _ = synth_matches(n=250, R=R_gt, t=np.array(t_gt), seed=2)
            da, db = bearings_of(m)
            E, inl = ransac_E(da, db, THRESH, iters=500, seed=0, conf=m[:, 4])
            R, t = select_pose(E, da, db, inl)
            Cb = -R.T @ t
            az = float(np.degrees(np.arctan2(Cb[0], Cb[2])))
            self.assertLess(circ_diff(az, az_ab_of(R_gt, np.array(t_gt))), 2.0,
                            msg="axis=%s ang=%s" % (axis, ang))


class TestRotationOnly(unittest.TestCase):
    def test_pure_rotation_has_near_zero_residual(self):
        R_gt = rot_matrix([0.0, 1.0, 0.0], 35.0)
        m, _, _ = synth_matches(n=200, R=R_gt, t=np.array([0.0, 0.0, 0.0]), seed=0)
        da, db = bearings_of(m)
        R_fit, err = rotation_only_fit(da, db)
        self.assertLess(float(np.median(err)), 0.01)
        self.assertLess(rot_err_deg(R_fit, R_gt), 0.05)

    def test_translation_produces_measurable_residual(self):
        R_gt = rot_matrix([0.0, 1.0, 0.0], 20.0)
        m, _, _ = synth_matches(n=200, R=R_gt, t=np.array([2.0, 0.0, 0.0]), seed=1)
        da, db = bearings_of(m)
        _, err = rotation_only_fit(da, db)
        self.assertGreater(float(np.median(err)), 1.5,
                           "有明顯基線時純旋轉不應解釋得了觀測")


class TestSampsonResidual(unittest.TestCase):
    def test_separates_inliers_from_outliers(self):
        m, _, _ = synth_matches(n=300, outlier_frac=0.3, seed=0)
        n_out = 90
        da, db = bearings_of(m)
        E, _ = ransac_E(da, db, THRESH, iters=2000, seed=0, conf=m[:, 4])
        res = sampson_residual(E, da, db)
        self.assertLess(float(np.median(res[n_out:])), THRESH)
        self.assertGreater(float(np.median(res[:n_out])), THRESH)

    def test_symmetric_in_argument_order(self):
        """殘差取兩個方向的平均，交換兩側並轉置 E 應得到同值。"""
        m, _, _ = synth_matches(n=150, seed=1)
        da, db = bearings_of(m)
        E = eight_point_E(da, db)
        np.testing.assert_allclose(sampson_residual(E, da, db),
                                   sampson_residual(E.T, db, da), atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
