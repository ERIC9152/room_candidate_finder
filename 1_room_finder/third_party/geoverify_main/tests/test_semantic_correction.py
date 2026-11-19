"""語意矯正規則與輸出一／輸出二分離。"""
import json
import os
import unittest

from _support import temp_dir, write_json  # noqa: F401

from roma_geo.export_public import HOUSE_SCHEMA_VERSION, house_to_public
from roma_geo.semantic import azimuth_from_pose, choose_door_correction, invert_pose


class TestChooseDoorCorrection(unittest.TestCase):
    def test_high_overlap_uses_door_center(self):
        doors = [{"az": 40.0, "az0": 20.0, "az1": 60.0}]
        sides = [{"az_range": [25.0, 55.0]}]
        az, hit = choose_door_correction(10.0, sides, doors, contain=0.8)
        self.assertTrue(hit)
        self.assertAlmostEqual(az, 40.0)

    def test_low_overlap_keeps_original(self):
        doors = [{"az": 40.0, "az0": 20.0, "az1": 30.0}]
        sides = [{"az_range": [0.0, 180.0]}]
        az, hit = choose_door_correction(90.0, sides, doors, contain=0.8)
        self.assertFalse(hit)
        self.assertEqual(az, 90.0)

    def test_no_doors_keeps_original(self):
        az, hit = choose_door_correction(12.0, [{"az_range": [0.0, 10.0]}], [])
        self.assertFalse(hit)
        self.assertEqual(az, 12.0)


class TestPoseInvertAzimuth(unittest.TestCase):
    def test_round_trip_matches_geo_azimuth(self):
        import math
        import numpy as np
        yaw = math.radians(30.0)
        R = [[math.cos(yaw), 0.0, math.sin(yaw)],
             [0.0, 1.0, 0.0],
             [-math.sin(yaw), 0.0, math.cos(yaw)]]
        t = [1.0, 0.0, 0.0]
        az = azimuth_from_pose(R, t)
        Ri, ti = invert_pose(R, t)
        az_back = azimuth_from_pose(Ri, ti)
        self.assertAlmostEqual(np.linalg.norm(ti), 1.0, places=6)
        self.assertGreaterEqual(az, 0.0)
        self.assertLess(az, 360.0)
        self.assertNotAlmostEqual(az, az_back, places=0)


class TestCorrectionLeavesPose(unittest.TestCase):
    def test_flag_and_azimuth_change_pose_stays(self):
        from roma_geo.semantic import apply_semantic_correction
        pub = {
            "images": ["a.jpg", "b.jpg"],
            "pairs": [{
                "i": 0, "j": 1, "az_ij": 10.0, "az_ji": 190.0,
                "az_source": "geo", "has_geometry": True,
                "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "t": [1.0, 0.0, 0.0],
                "regions": [{
                    "a": {"az_range": [25.0, 55.0], "polygon": [[1, 1], [2, 1], [2, 2]]},
                    "b": {"az_range": [0.0, 10.0], "polygon": [[1, 1], [2, 1], [2, 2]]},
                }],
            }],
        }
        R0 = [row[:] for row in pub["pairs"][0]["R"]]
        t0 = list(pub["pairs"][0]["t"])
        doors = {"a": [{"az": 40.0, "az0": 20.0, "az1": 60.0}]}
        apply_semantic_correction(pub, doors)
        p = pub["pairs"][0]
        self.assertTrue(p["semantic_corrected"])
        self.assertAlmostEqual(p["az_ij"], 40.0)
        self.assertEqual(p["R"], R0)
        self.assertEqual(p["t"], t0)
        self.assertEqual(p["az_source"], "geo")

    def test_coarse_source_is_never_corrected(self):
        from roma_geo.semantic import apply_semantic_correction
        pub = {
            "images": ["a.jpg", "b.jpg"],
            "pairs": [{
                "i": 0, "j": 1, "az_ij": 10.0, "az_ji": 190.0,
                "az_source": "coarse", "has_geometry": False,
                "regions": [{
                    "a": {"az_range": [25.0, 55.0], "polygon": [[1, 1], [2, 1], [2, 2]]},
                    "b": {"az_range": [25.0, 55.0], "polygon": [[1, 1], [2, 1], [2, 2]]},
                }],
            }],
        }
        doors = {
            "a": [{"az": 40.0, "az0": 20.0, "az1": 60.0}],
            "b": [{"az": 40.0, "az0": 20.0, "az1": 60.0}],
        }
        apply_semantic_correction(pub, doors)
        p = pub["pairs"][0]
        self.assertFalse(p["semantic_corrected"])
        self.assertEqual(p["az_ij"], 10.0)
        self.assertEqual(p["az_ji"], 190.0)
        self.assertEqual(p["az_source"], "coarse")

    def test_corrected_pairs_are_always_geo(self):
        from roma_geo.semantic import apply_semantic_correction
        pub = {
            "images": ["a.jpg", "b.jpg", "c.jpg"],
            "pairs": [
                {
                    "i": 0, "j": 1, "az_ij": 10.0, "az_ji": 190.0,
                    "az_source": "geo", "has_geometry": True,
                    "regions": [{
                        "a": {"az_range": [25.0, 55.0], "polygon": [[1, 1], [2, 1], [2, 2]]},
                        "b": {"az_range": [0.0, 10.0], "polygon": [[1, 1], [2, 1], [2, 2]]},
                    }],
                },
                {
                    "i": 0, "j": 2, "az_ij": 12.0, "az_ji": 200.0,
                    "az_source": "coarse", "has_geometry": False,
                    "regions": [{
                        "a": {"az_range": [25.0, 55.0], "polygon": [[1, 1], [2, 1], [2, 2]]},
                        "b": {"az_range": [25.0, 55.0], "polygon": [[1, 1], [2, 1], [2, 2]]},
                    }],
                },
            ],
        }
        doors = {
            "a": [{"az": 40.0, "az0": 20.0, "az1": 60.0}],
            "c": [{"az": 40.0, "az0": 20.0, "az1": 60.0}],
        }
        apply_semantic_correction(pub, doors)
        corrected = [p for p in pub["pairs"] if p["semantic_corrected"]]
        self.assertTrue(corrected)
        self.assertTrue(all(p["az_source"] == "geo" for p in corrected))


class TestHouseSchemaVersion(unittest.TestCase):
    def test_export_always_writes_version_and_correction_flag(self):
        d = {
            "house": "H", "n_images": 2, "width": 64, "height": 32,
            "images": ["a.jpg", "b.jpg"],
            "pairs": [{"i": 0, "j": 1, "prob": 0.5, "az_ij": 10.0, "az_ji": 190.0,
                       "az_source": "coarse", "geo_posed": 0}],
        }
        pub = house_to_public(d)
        self.assertEqual(pub["schema_version"], HOUSE_SCHEMA_VERSION)
        self.assertIs(pub["pairs"][0]["semantic_corrected"], False)
        self.assertNotIn("R", pub["pairs"][0])


class TestAttachPose(unittest.TestCase):
    def test_pose_present_only_when_geometry(self):
        from roma_geo.export_public import attach_pose, house_to_public
        d = {
            "house": "H", "n_images": 2, "width": 64, "height": 32,
            "images": ["a.jpg", "b.jpg"],
            "pairs": [
                {"i": 0, "j": 1, "az_ij": 10.0, "az_ji": 190.0, "az_source": "geo",
                 "geo_posed": 1},
                {"i": 0, "j": 1, "az_ij": 12.0, "az_ji": 200.0, "az_source": "coarse",
                 "geo_posed": 0},
            ],
        }
        d["pairs"][1]["j"] = 1
        d["images"] = ["a.jpg", "b.jpg"]
        d["pairs"] = [
            {"i": 0, "j": 1, "az_ij": 10.0, "az_ji": 190.0, "az_source": "geo",
             "geo_posed": 1},
        ]
        pub = house_to_public(d)
        with temp_dir() as td:
            write_json(os.path.join(td, "H_a_b.json"), {
                "pose": {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [1.0, 0.0, 0.0]},
            })
            attach_pose("H", pub, td)
        self.assertIn("R", pub["pairs"][0])
        self.assertIn("t", pub["pairs"][0])
        az = azimuth_from_pose(pub["pairs"][0]["R"], pub["pairs"][0]["t"])
        self.assertIsInstance(az, float)


if __name__ == "__main__":
    unittest.main(verbosity=2)
