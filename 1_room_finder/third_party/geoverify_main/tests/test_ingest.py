"""
上游交界測試：candidate-connections JSON → 連通性矩陣。

這是本專案唯一讀取外部模型輸出的地方，兩邊只透過檔案往來。三項不變量：

  1. 機率無損：sigmoid(ch0) 須逐位還原上游給的 probability。
  2. 連通分數無方向：ch0 對稱。
  3. 方位有方向且不得翻面：M[i,j] 存「在 i 圖看向 j」的方位，即 yaw_a；M[j,i] 存 yaw_b。

第三項最需要鎖住。上游與本專案的方位規約經實測一致（0° = 影像中央、右為正），
但若任一方改動規約或此處寫反，全部邊方向會整批反向，而數值本身看不出異常。
"""
import json
import math
import os
import unittest

import numpy as np

from _support import temp_dir, circ_diff, write_json  # noqa: F401（匯入時補 sys.path）

import ingest_connections as ic
import upstream_direction as ud


def upstream_json(pairs, house="DollhouseTask_00000"):
    return {"house_id": house, "threshold": 0.5, "notes": "test",
            "view_convention": "image-center-zero", "pairs": pairs}


def pair(a, b, prob, yaw_a=None, yaw_b=None):
    p = {"pano_a": a, "pano_b": b, "probability": prob}
    if yaw_a is not None or yaw_b is not None:
        p["corresponding_views"] = {"view_a": 0, "yaw_a_degrees": yaw_a,
                                    "view_b": 3, "yaw_b_degrees": yaw_b}
    return p


class TestProbToLogit(unittest.TestCase):
    def test_round_trip(self):
        for p in (0.001, 0.1, 0.5, 0.9, 0.9999):
            back = 1.0 / (1.0 + math.exp(-ic.prob_to_logit(p)))
            self.assertAlmostEqual(back, p, places=9)

    def test_half_maps_to_zero(self):
        """門檻約定：prob ≥ 0.5 ⇔ logit ≥ 0.0，`select_pairs` 的門檻據此設為 0。"""
        self.assertAlmostEqual(ic.prob_to_logit(0.5), 0.0, places=12)

    def test_extremes_are_clamped(self):
        for p in (0.0, 1.0, -0.5, 1.5):
            v = ic.prob_to_logit(p)
            self.assertTrue(math.isfinite(v), "prob=%s 產生非有限的 logit" % p)

    def test_monotonic(self):
        vals = [ic.prob_to_logit(p) for p in (0.01, 0.2, 0.5, 0.8, 0.99)]
        for a, b in zip(vals, vals[1:]):
            self.assertLess(a, b)


class TestBuildMatrix(unittest.TestCase):
    def setUp(self):
        self.cl = upstream_json([
            pair("x.jpg", "y.jpg", 0.9, yaw_a=30.0, yaw_b=200.0),
            pair("y.jpg", "z.jpg", 0.2, yaw_a=100.0, yaw_b=280.0),
            pair("x.jpg", "z.jpg", 0.7, yaw_a=350.0, yaw_b=170.0),
        ])
        self.M, self.names, self.N = ic.build_matrix(self.cl)
        self.idx = {n: k for k, n in enumerate(self.names)}

    def test_shape_and_names(self):
        self.assertEqual(self.N, 3)
        self.assertEqual(self.M.shape, (3, 3, 3))
        self.assertEqual(self.names, ["x.jpg", "y.jpg", "z.jpg"])
        self.assertEqual(self.M.dtype, np.float32)

    def test_names_are_sorted_deterministically(self):
        shuffled = upstream_json(list(reversed(self.cl["pairs"])))
        _, names2, _ = ic.build_matrix(shuffled)
        self.assertEqual(self.names, names2, "影像順序須與 pair 出現順序無關")

    def test_probability_is_recoverable(self):
        for p in self.cl["pairs"]:
            i, j = self.idx[p["pano_a"]], self.idx[p["pano_b"]]
            back = 1.0 / (1.0 + math.exp(-float(self.M[i, j, 0])))
            self.assertAlmostEqual(back, p["probability"], places=5)

    def test_connection_channel_is_symmetric(self):
        np.testing.assert_allclose(self.M[:, :, 0], self.M[:, :, 0].T, atol=1e-6)

    def test_azimuth_direction_is_not_flipped(self):
        """M[i,j] 必須是 yaw_a（在 A 圖看向 B），M[j,i] 是 yaw_b。寫反即整批 180° 誤差。"""
        for p in self.cl["pairs"]:
            i, j = self.idx[p["pano_a"]], self.idx[p["pano_b"]]
            cv = p["corresponding_views"]
            self.assertLess(circ_diff(ud.from_sincos(self.M[i, j, 1], self.M[i, j, 2]),
                                      cv["yaw_a_degrees"]), 1e-3)
            self.assertLess(circ_diff(ud.from_sincos(self.M[j, i, 1], self.M[j, i, 2]),
                                      cv["yaw_b_degrees"]), 1e-3)

    def test_azimuth_channels_are_unit_norm_where_written(self):
        for p in self.cl["pairs"]:
            i, j = self.idx[p["pano_a"]], self.idx[p["pano_b"]]
            for a, b in ((i, j), (j, i)):
                r = math.hypot(float(self.M[a, b, 1]), float(self.M[a, b, 2]))
                self.assertAlmostEqual(r, 1.0, places=5)

    def test_diagonal_is_untouched(self):
        for k in range(self.N):
            np.testing.assert_allclose(self.M[k, k], np.zeros(3), atol=0)

    def test_missing_yaw_leaves_channels_zero(self):
        cl = upstream_json([pair("a.jpg", "b.jpg", 0.8)])
        M, names, _ = ic.build_matrix(cl)
        np.testing.assert_allclose(M[0, 1, 1:], np.zeros(2), atol=0)
        self.assertNotEqual(float(M[0, 1, 0]), 0.0, "缺方位不影響連通分數")

    def test_partial_yaw(self):
        """只給單向 yaw 時，另一向維持未填。"""
        cl = upstream_json([pair("a.jpg", "b.jpg", 0.8, yaw_a=45.0)])
        M, _, _ = ic.build_matrix(cl)
        self.assertAlmostEqual(ud.from_sincos(M[0, 1, 1], M[0, 1, 2]), 45.0, places=3)
        np.testing.assert_allclose(M[1, 0, 1:], np.zeros(2), atol=0)


class TestSelfCheck(unittest.TestCase):
    def test_reports_near_zero_error(self):
        cl = upstream_json([pair("a.jpg", "b.jpg", 0.93, yaw_a=12.5, yaw_b=195.0),
                            pair("b.jpg", "c.jpg", 0.05, yaw_a=300.0, yaw_b=120.0)])
        M, names, _ = ic.build_matrix(cl)
        chk = ic.selfcheck(cl, M, names)
        self.assertLess(chk["max_prob_err"], 1e-5)
        self.assertLess(chk["max_az_err_deg"], 1e-3)

    def test_detects_corrupted_matrix(self):
        """自檢本身要能失敗。"""
        cl = upstream_json([pair("a.jpg", "b.jpg", 0.9, yaw_a=30.0, yaw_b=210.0)])
        M, names, _ = ic.build_matrix(cl)
        M[0, 1, 1] *= -1.0                    # 破壞方位
        M[0, 1, 0] += 3.0                     # 破壞機率
        chk = ic.selfcheck(cl, M, names)
        self.assertGreater(chk["max_prob_err"], 0.01)
        self.assertGreater(chk["max_az_err_deg"], 1.0)

    def test_wraps_around_seam(self):
        """yaw 359° 與 1° 的圓角差是 2°，不得算成 358°。"""
        cl = upstream_json([pair("a.jpg", "b.jpg", 0.5, yaw_a=359.9, yaw_b=0.1)])
        M, names, _ = ic.build_matrix(cl)
        self.assertLess(ic.selfcheck(cl, M, names)["max_az_err_deg"], 1e-3)


class TestMatrixContract(unittest.TestCase):
    """輸出的檔案組合須符合 `matrices/README.md` 的契約。"""

    def test_images_json_aligns_with_matrix(self):
        cl = upstream_json([pair("a.jpg", "b.jpg", 0.9, yaw_a=10.0, yaw_b=190.0),
                            pair("b.jpg", "c.jpg", 0.4, yaw_a=80.0, yaw_b=260.0)])
        M, names, N = ic.build_matrix(cl)
        with temp_dir() as d:
            np.save(os.path.join(d, "h.npy"), M)
            write_json(os.path.join(d, "h_images.json"), names)
            M2 = np.load(os.path.join(d, "h.npy"))
            with open(os.path.join(d, "h_images.json"), encoding="utf-8") as f:
                names2 = json.load(f)
        self.assertEqual(M2.shape, (len(names2), len(names2), 3))
        self.assertEqual(names2, names)
        self.assertTrue(all(n.endswith(".jpg") for n in names2), "影像清單須含副檔名")

    def test_complete_graph_detection(self):
        """完整圖判定：N·(N−1)/2 對。非完整時 main() 會提示，此處驗算式本身。"""
        cl = upstream_json([pair("a.jpg", "b.jpg", 0.9), pair("b.jpg", "c.jpg", 0.9),
                            pair("a.jpg", "c.jpg", 0.9)])
        _, _, N = ic.build_matrix(cl)
        self.assertEqual(N * (N - 1) // 2, len(cl["pairs"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
