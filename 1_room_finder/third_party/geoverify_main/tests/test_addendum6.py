"""追加6：降級時保留全部配對；無方位時欄位為空值、來源缺席。"""
import json
import math
import os
import unittest

import numpy as np

from _support import env_override, quiet, synth_matches, temp_dir, write_json

from roma_geo import pipeline
from roma_geo.export_public import house_to_public

from test_pipeline import HOUSE, SIZE, make_image_house, settings_for, write_cache
import check_deliverables as cd


def _matrix(n, az=40.0):
    m = np.zeros((n, n, 3), dtype=np.float32)
    rad = math.radians(az)
    s, c = math.sin(rad), math.cos(rad)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            m[i, j, 0] = 1.0
            m[i, j, 1] = s
            m[i, j, 2] = c
    return m


def _public_payload(has_matrix, az_ij, az_ji, az_source=None):
    pair = {"i": 0, "j": 1, "az_ij": az_ij, "az_ji": az_ji,
            "has_geometry": False, "semantic_corrected": False}
    if az_source is not None:
        pair["az_source"] = az_source
    d = {
        "house": "H", "n_images": 2, "width": 64, "height": 32,
        "images": ["a.jpg", "b.jpg"],
        "az_convention": "az(x)=((x/W)*360-180)%360 ; 0deg = image center",
        "schema_version": "house-v2",
        "pairs": [pair],
    }
    if has_matrix is not None:
        d["has_connectivity_matrix"] = has_matrix
    return d


class TestVDegradedKeepsAllPairs(unittest.TestCase):
    def test_three_images_one_posed_still_three_pairs(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, stems=("lobby", "hall", "yard"))
            s = settings_for(d)
            m, _, _ = synth_matches(n=80, W=SIZE[0], H=SIZE[1], seed=4)
            write_cache(s.cache_root(), HOUSE, "lobby", "hall", m)
            with quiet():
                pipeline.estimate_poses(settings=s, pairs=[(HOUSE, "lobby", "hall")])
                result = pipeline.prepare_house(settings=s, houses=[HOUSE])
            data = json.load(open(result.outputs["house_json"], encoding="utf-8"))
            self.assertEqual(len(data["pairs"]), 3)
            posed = [p for p in data["pairs"] if p.get("az_source") == "geo"]
            empty = [p for p in data["pairs"] if p.get("az_ij") is None]
            self.assertEqual(len(posed), 1)
            self.assertEqual(len(empty), 2)
            for p in empty:
                self.assertIsNone(p["az_ij"])
                self.assertIsNone(p["az_ji"])
                self.assertNotIn("az_source", p)
            with quiet():
                exp = pipeline.export_public_houses(settings=s, houses=[HOUSE])
            pub = json.load(open(exp.outputs["public_json"], encoding="utf-8"))
            self.assertEqual(len(pub["pairs"]), 3)
            self.assertIs(pub["has_connectivity_matrix"], False)
            for p in pub["pairs"]:
                self.assertIn("az_ij", p)
                self.assertIn("az_ji", p)
                if p["az_ij"] is None:
                    self.assertIsNone(p["az_ji"])
                    self.assertNotIn("az_source", p)
                else:
                    self.assertEqual(p["az_source"], "geo")
            r = cd.check_public_houses(s.public_dir(), datasets_dir=s.datasets_root())
            self.assertEqual(r.status, cd.OK, r.problems)

    def test_second_export_has_the_same_pair_set(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, stems=("lobby", "hall", "yard"))
            s = settings_for(d)
            with quiet():
                pipeline.prepare_house(settings=s, houses=[HOUSE])
                a = pipeline.export_public_houses(settings=s, houses=[HOUSE])
                b = pipeline.export_public_houses(settings=s, houses=[HOUSE])
            pa = json.load(open(a.outputs["public_json"], encoding="utf-8"))
            pb = json.load(open(b.outputs["public_json"], encoding="utf-8"))
            keys = lambda pub: {(p["i"], p["j"]) for p in pub["pairs"]}
            self.assertEqual(keys(pa), keys(pb))
            self.assertEqual(len(pa["pairs"]), 3)


class TestVMatrixPathUnchanged(unittest.TestCase):
    def test_matrix_house_keeps_all_pairs_with_numeric_azimuth(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, stems=("lobby", "hall", "yard"))
            s = settings_for(d)
            mdir = s.matrices_root()
            os.makedirs(mdir, exist_ok=True)
            np.save(os.path.join(mdir, HOUSE + ".npy"), _matrix(3, az=40.0))
            write_json(os.path.join(mdir, HOUSE + "_images.json"),
                       ["lobby.jpg", "hall.jpg", "yard.jpg"])
            with quiet():
                result = pipeline.prepare_house(settings=s, houses=[HOUSE])
            data = json.load(open(result.outputs["house_json"], encoding="utf-8"))
            self.assertTrue(data["has_connectivity_matrix"])
            self.assertEqual(len(data["pairs"]), 3)
            for p in data["pairs"]:
                self.assertIsInstance(p["az_ij"], float)
                self.assertIsInstance(p["az_ji"], float)
                self.assertEqual(p["az_source"], "coarse")
                self.assertIn("prob", p)
            pub = house_to_public(data)
            for p in pub["pairs"]:
                self.assertIsNotNone(p["az_ij"])
                self.assertIsNotNone(p["az_ji"])
                self.assertEqual(p["az_source"], "coarse")


class TestVPublicAzimuthGuard(unittest.TestCase):
    def _check(self, payload):
        with temp_dir() as d:
            write_json(os.path.join(d, "H.json"), payload)
            ds = os.path.join(d, "datasets")
            os.makedirs(ds, exist_ok=True)
            return cd.check_public_houses(d, datasets_dir=ds)

    def test_degraded_null_azimuth_passes(self):
        r = self._check(_public_payload(False, None, None))
        self.assertEqual(r.status, cd.OK, r.problems)

    def test_matrix_null_azimuth_fails(self):
        r = self._check(_public_payload(True, None, None, "coarse"))
        self.assertEqual(r.status, cd.FAIL)
        self.assertTrue(any("空值" in p or "缺席" in p for p in r.problems))

    def test_undeclared_house_null_azimuth_fails(self):
        r = self._check(_public_payload(None, None, None))
        self.assertEqual(r.status, cd.FAIL)

    def test_degraded_missing_keys_fail(self):
        d = _public_payload(False, None, None)
        del d["pairs"][0]["az_ij"]
        del d["pairs"][0]["az_ji"]
        r = self._check(d)
        self.assertEqual(r.status, cd.FAIL)
        self.assertTrue(any("缺席" in p for p in r.problems))

    def test_matrix_numeric_azimuth_passes(self):
        r = self._check(_public_payload(True, 10.0, 190.0, "coarse"))
        self.assertEqual(r.status, cd.OK, r.problems)


if __name__ == "__main__":
    unittest.main(verbosity=2)
