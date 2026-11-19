"""az_source 缺席只在連通矩陣缺席的降級下才容許。"""
import json
import os
import unittest

from _support import temp_dir, write_json  # noqa: F401

import check_deliverables as cd


def _house(has_matrix, az_source):
    pair = {"i": 0, "j": 1, "az_ij": 10.0, "az_ji": 190.0, "prob": 0.5}
    if az_source is not None:
        pair["az_source"] = az_source
    d = {
        "house": "H", "n_images": 2, "width": 64, "height": 32,
        "images": ["a.jpg", "b.jpg"],
        "az_convention": "az(x)=((x/W)*360-180)%360 ; 0deg = image center",
        "pairs": [pair],
    }
    if has_matrix is not None:
        d["has_connectivity_matrix"] = has_matrix
    return d


class TestAzSourceAbsenceGuard(unittest.TestCase):
    def _check(self, payload):
        with temp_dir() as d:
            write_json(os.path.join(d, "H.json"), payload)
            return cd.check_house_json(d)

    def test_degraded_missing_az_source_passes(self):
        r = self._check(_house(False, None))
        self.assertEqual(r.status, cd.OK, r.problems)

    def test_matrix_house_missing_az_source_fails(self):
        r = self._check(_house(True, None))
        self.assertEqual(r.status, cd.FAIL)
        self.assertTrue(any("az_source 缺席" in p for p in r.problems))

    def test_legacy_house_missing_az_source_fails(self):
        r = self._check(_house(None, None))
        self.assertEqual(r.status, cd.FAIL)

    def test_matrix_house_with_az_source_passes(self):
        r = self._check(_house(True, "coarse"))
        self.assertEqual(r.status, cd.OK, r.problems)


if __name__ == "__main__":
    unittest.main(verbosity=2)
