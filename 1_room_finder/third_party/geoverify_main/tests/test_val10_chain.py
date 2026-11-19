"""10 房完整鏈：有位姿用幾何方位，否則用同學方向，再對 GT hotspot 量誤差。"""
import json
import os
import unittest

from _support import temp_dir  # noqa: F401（匯入時補 sys.path）

from roma_geo.semantic import circ_err
from pose_regions.val10_chain_eval import (
    iter_gt_directions,
    list_matrix_houses,
    summarize,
    evaluate_houses,
)


class TestIterGtDirections(unittest.TestCase):
    def test_geo_pair_scores_pose_azimuth_against_hotspot(self):
        pair = {
            "az_source": "geo",
            "az_ij": 10.0, "az_ji": 190.0,
            "az_coarse_ij": 0.0, "az_coarse_ji": 180.0,
            "gt_az_ij": 12.0, "gt_az_ji": 188.0,
            "gt_connected": True,
            "geo_posed": 1,
            "prob": 0.9,
        }
        rows = list(iter_gt_directions(pair))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["source"], "geo")
        self.assertAlmostEqual(rows[0]["err_combined"], 2.0)
        self.assertAlmostEqual(rows[0]["err_geo"], 2.0)
        self.assertAlmostEqual(rows[0]["err_coarse"], 12.0)
        self.assertAlmostEqual(rows[1]["err_combined"], 2.0)

    def test_coarse_fallback_scores_classmate_azimuth(self):
        pair = {
            "az_source": "coarse",
            "az_ij": 60.0, "az_ji": 240.0,
            "az_coarse_ij": 60.0, "az_coarse_ji": 240.0,
            "gt_az_ij": 70.0, "gt_az_ji": None,
            "gt_connected": True,
            "geo_posed": 0,
            "prob": 0.8,
        }
        rows = list(iter_gt_directions(pair))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "coarse")
        self.assertAlmostEqual(rows[0]["err_combined"], 10.0)
        self.assertIsNone(rows[0]["err_geo"])
        self.assertAlmostEqual(rows[0]["err_coarse"], 10.0)

    def test_side_without_hotspot_is_skipped(self):
        pair = {
            "az_source": "geo",
            "az_ij": 10.0, "az_ji": 190.0,
            "az_coarse_ij": 0.0, "az_coarse_ji": 180.0,
            "gt_az_ij": None, "gt_az_ji": None,
            "gt_connected": False,
            "geo_posed": 1,
            "prob": 0.2,
        }
        self.assertEqual(list(iter_gt_directions(pair)), [])

    def test_seam_error_uses_shortest_arc(self):
        pair = {
            "az_source": "geo",
            "az_ij": 350.0, "az_ji": 10.0,
            "az_coarse_ij": 0.0, "az_coarse_ji": 0.0,
            "gt_az_ij": 10.0, "gt_az_ji": None,
            "gt_connected": True,
            "geo_posed": 1,
            "prob": 0.99,
        }
        row = next(iter_gt_directions(pair))
        self.assertAlmostEqual(row["err_combined"], circ_err(350.0, 10.0))
        self.assertAlmostEqual(row["err_combined"], 20.0)


class TestSummarize(unittest.TestCase):
    def test_empty_is_none(self):
        self.assertIsNone(summarize([]))

    def test_median_and_hit_rates_on_known_list(self):
        rows = [{"err_combined": x} for x in (1.0, 10.0, 20.0, 40.0, 90.0)]
        s = summarize(rows, "err_combined")
        self.assertEqual(s["n"], 5)
        self.assertAlmostEqual(s["median"], 20.0)
        self.assertAlmostEqual(s["le22"], 60.0)
        self.assertAlmostEqual(s["le45"], 80.0)


class TestEvaluateHouses(unittest.TestCase):
    def test_reads_only_houses_listed_in_the_matrix_dir(self):
        with temp_dir() as td:
            mdir = os.path.join(td, "mat")
            ddir = os.path.join(td, "data")
            os.makedirs(mdir)
            os.makedirs(ddir)
            open(os.path.join(mdir, "HouseA.npy"), "wb").close()
            open(os.path.join(mdir, "HouseB.npy"), "wb").close()
            json.dump({
                "house": "HouseA",
                "pairs": [{
                    "az_source": "geo",
                    "az_ij": 10.0, "az_ji": 190.0,
                    "az_coarse_ij": 60.0, "az_coarse_ji": 240.0,
                    "gt_az_ij": 10.0, "gt_az_ji": 190.0,
                    "gt_connected": True,
                    "geo_posed": 1,
                    "prob": 0.9,
                }],
            }, open(os.path.join(ddir, "HouseA.json"), "w", encoding="utf-8"))
            json.dump({
                "house": "HouseB",
                "pairs": [{
                    "az_source": "coarse",
                    "az_ij": 60.0, "az_ji": 240.0,
                    "az_coarse_ij": 60.0, "az_coarse_ji": 240.0,
                    "gt_az_ij": 90.0, "gt_az_ji": None,
                    "gt_connected": True,
                    "geo_posed": 0,
                    "prob": 0.7,
                }],
            }, open(os.path.join(ddir, "HouseB.json"), "w", encoding="utf-8"))
            json.dump({
                "house": "Other",
                "pairs": [{
                    "az_source": "geo",
                    "az_ij": 0.0, "az_ji": 0.0,
                    "az_coarse_ij": 0.0, "az_coarse_ji": 0.0,
                    "gt_az_ij": 0.0, "gt_az_ji": 0.0,
                    "gt_connected": True,
                    "geo_posed": 1,
                    "prob": 0.9,
                }],
            }, open(os.path.join(ddir, "Other.json"), "w", encoding="utf-8"))
            self.assertEqual(list_matrix_houses(mdir), ["HouseA", "HouseB"])
            out = evaluate_houses(mdir, ddir)
            self.assertEqual(out["n_houses"], 2)
            self.assertEqual(out["combined"]["n"], 3)
            self.assertEqual(out["geo"]["n"], 2)
            self.assertEqual(out["coarse"]["n"], 1)
            self.assertAlmostEqual(out["combined"]["median"], 0.0)
            self.assertAlmostEqual(out["coarse"]["median"], 30.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
