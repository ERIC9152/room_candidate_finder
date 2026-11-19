"""受限來源的守門條件。

某些資料來源的授權只及於研究評測，不得出現在對外交付內容中。判定方式是資料集裡
該房屋的目錄底下放一個標記檔，`export_public` 見到就整批停下。只跳過該房不夠：
交付內容少一房不會有人察覺，整批停下才會。

判定只看標記檔，不認任何特定來源的名稱，因此本測試用的房屋名不指涉實際來源。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from _support import ROOT  # noqa: F401（匯入時補 sys.path）

import export_public

HOUSE = "RestrictedFake"
REASON = "僅供評測，不得對外交付。"


def _house_json():
    return {"house": HOUSE, "n_images": 2, "width": 4096, "height": 2048,
            "images": ["a", "b"], "az_convention": "erp", "doors": [],
            "pairs": [{"i": 0, "j": 1, "prob": 0.9, "az_ij": 10.0, "az_ji": -170.0,
                       "az_source": "geometry", "geo_posed": False,
                       "n_inlier": 100, "inlier_rate": 0.5, "conf_used": 0.65}]}


class TestRestrictedGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="restricted_")
        self.datasets = os.path.join(self.tmp, "datasets")
        self.in_dir = os.path.join(self.tmp, "in")
        self.out_dir = os.path.join(self.tmp, "out")
        os.makedirs(os.path.join(self.datasets, HOUSE))
        os.makedirs(self.in_dir)
        with open(os.path.join(self.in_dir, HOUSE + ".json"), "w", encoding="utf-8") as f:
            json.dump(_house_json(), f)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _mark(self, *sub):
        d = os.path.join(self.datasets, HOUSE, *sub)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, export_public.RESTRICTED_MARKER), "w", encoding="utf-8") as f:
            f.write(REASON + "\n")

    def _export(self):
        env = dict(os.environ, ROMA_GEO_DATASETS=self.datasets)
        return subprocess.run(
            [sys.executable, os.path.join(ROOT, "export_public.py"),
             "--in_dir", self.in_dir, "--out_dir", self.out_dir, "--no_regions"],
            env=env, capture_output=True, text=True)

    def test_no_marker_exports_normally(self):
        r = self._export()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, HOUSE + ".json")))

    def test_marker_stops_the_whole_batch(self):
        self._mark()
        r = self._export()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("受限來源", r.stdout + r.stderr)
        # 停下時不得留下半套輸出
        self.assertFalse(os.path.exists(os.path.join(self.out_dir, HOUSE + ".json")))

    def test_marker_found_in_nested_house_dir(self):
        # 房屋目錄可能同名巢狀，標記放在深處也要擋得住
        self._mark(HOUSE)
        r = self._export()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("受限來源", r.stdout + r.stderr)

    def test_reason_text_is_reported(self):
        self._mark()
        self.assertEqual(export_public.restricted_reason(HOUSE, self.datasets), REASON)

    def test_unmarked_house_returns_none(self):
        self.assertIsNone(export_public.restricted_reason(HOUSE, self.datasets))

    def test_missing_house_dir_returns_none(self):
        self.assertIsNone(export_public.restricted_reason("NoSuchHouse", self.datasets))


if __name__ == "__main__":
    unittest.main()
