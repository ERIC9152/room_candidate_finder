# -*- coding: utf-8 -*-
"""Core tests (no GPU, no UI): python -m unittest discover -s tests"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image

from finder import dataset
from finder.model import ReviewModel
from finder.scores import Calibration, active_wedge, load_scores, passes, summarize

try:
    import numpy  # noqa: F401  (scan environment; the review environment has only Pillow)
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


def geom(inl, wa, wb, pose=True):
    return {"n": {"inlier": inl}, "inlier_rate": 0.5, "flags": {"no_covisibility": not pose},
            "pose": {"az_ab_deg": 10.0} if pose else None, "quality": {"az_wedge_a_deg": wa, "az_wedge_b_deg": wb},
            "regions": [{}]}


class TestCore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _img(self, path, size=(200, 100), color=(10, 20, 30)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.new("RGB", size, color).save(path)

    def test_scan_houses(self):
        r = os.path.join(self.tmp, "root")
        self._img(os.path.join(r, "H1", "a.jpg"), color=(1, 1, 1))
        self._img(os.path.join(r, "H1", "sub", "b.jpg"), color=(2, 2, 2))
        self._img(os.path.join(r, "H1", "copy_of_a.jpg"), color=(1, 1, 1))          # duplicate content
        self._img(os.path.join(r, "H1", "wide.jpg"), size=(300, 100))               # not 2:1
        self._img(os.path.join(r, "H1", "bi_layout_predictions", "x.jpg"))          # excluded folder
        self._img(os.path.join(r, "H2", "only.jpg"))                                # < 2 panoramas
        houses, skipped = dataset.scan_houses(r)
        self.assertEqual(list(houses), ["H1"])
        self.assertEqual(sorted(p["id"] for p in houses["H1"]), ["a", "b"])
        reasons = " ".join(s["reason"] for s in skipped)
        self.assertIn("duplicate", reasons); self.assertIn("not 2:1", reasons); self.assertIn("fewer than 2", reasons)
        self.assertEqual(len(list(dataset.all_pairs(houses["H1"]))), 1)

    def test_classifier(self):
        s = summarize(geom(100, 50, 70))
        self.assertEqual(s["wedge_min"], 50)
        self.assertTrue(passes(s, 45, 30)); self.assertFalse(passes(s, 55, 30)); self.assertFalse(passes(s, 45, 200))
        self.assertFalse(passes(summarize(geom(100, 80, 80, pose=False)), 45, 30))
        self.assertFalse(passes(summarize({"n": {"inlier": 0}, "flags": {"no_covisibility": True}}), 0, 0))

    def test_calibration_lookup(self):
        q = Calibration().expected(44)
        self.assertEqual(q["at"], 45.0); self.assertTrue(0 < q["P"] < 1)
        o = Calibration().expected(41, "open")
        self.assertEqual(o["at"], 40.0); self.assertGreater(o["P"], Calibration().expected(40)["P"])

    def test_open_classifier(self):
        s = summarize(geom(100, 50, 70))
        s["open"] = dict(n_keep=80, door_frac=0.2, wedge_a=35, wedge_b=60, wedge_min=35)
        self.assertTrue(passes(s, 45, 30, "plain")); self.assertFalse(passes(s, 45, 30, "open"))
        self.assertTrue(passes(s, 30, 30, "open")); self.assertFalse(passes(s, 30, 90, "open"))   # n_keep < 90
        self.assertEqual(active_wedge(s, "open"), 35); self.assertEqual(active_wedge(s, "plain"), 50)
        p = summarize(geom(100, 50, 70))                     # no open scores -> plain rule in open mode
        self.assertTrue(passes(p, 45, 30, "open"))

    def _scan(self):
        """A portable scan: data/<ds>/H/*.jpg and scans/<ds>/ with relative paths."""
        data = os.path.join(self.tmp, "data", "ds", "H")
        d = os.path.join(self.tmp, "scans", "ds"); os.makedirs(d)
        ids = ["p1", "p2", "p3", "p4"]
        for i in ids:
            self._img(os.path.join(data, i + ".jpg"))
        man = {"geoverify_commit": "abc", "root": os.path.join("..", "..", "data", "ds"), "root_abs": "",
               "houses": {"H": [{"id": i, "path": os.path.join("H", i + ".jpg"), "w": 200, "h": 100} for i in ids]}}
        with open(os.path.join(d, "manifest.json"), "w") as fh:
            json.dump(man, fh)
        rows = [("p1", "p2", geom(300, 80, 90)), ("p2", "p3", geom(100, 50, 50)), ("p1", "p3", geom(40, 20, 30)),
                ("p3", "p4", geom(60, 46, 47)), ("p1", "p4", geom(10, 90, 90))]
        with open(os.path.join(d, "scores.jsonl"), "w") as fh:
            for a, b, g in rows:
                fh.write(json.dumps({"house": "H", "a": a, "b": b, "score": summarize(g)}) + "\n")
        return d

    def test_model_filter_groups_export_resume(self):
        d = self._scan()
        m = ReviewModel(d, "tester")
        self.assertTrue(os.path.exists(m.image_path("H", "p1")))                     # relative path resolves
        c = m.candidates("H")
        self.assertEqual([(x["a"], x["b"]) for x in c], [("p1", "p2"), ("p2", "p3"), ("p3", "p4")])   # sorted by wedge
        m.wedge_deg = 15
        self.assertEqual(len(m.candidates("H")), 4)          # p1-p3 now in; p1-p4 still out (10 inliers)
        m.wedge_deg = 45
        self.assertEqual(m.panos_without_candidates("H"), [])
        m.decide("H", "p2", "p1", "same_room")                # reversed order is the same pair
        m.decide("H", "p2", "p3", "same_room")
        m.decide("H", "p1", "p3", "different_room")
        groups, conflicts = m.groups("H")
        self.assertEqual(groups, [["p1", "p2", "p3"]]); self.assertEqual(conflicts, [("p1", "p3")])
        m.decide("H", "p1", "p3", None)                       # clear
        self.assertEqual(m.groups("H")[1], [])
        out = os.path.join(d, "groups.json"); data = m.export(out)
        self.assertEqual(data["groups"][0]["panorama_uuids"], ["p1", "p2", "p3"])
        self.assertEqual(data["groups"][0]["status"], "human_verified")
        self.assertEqual(data["classifier"]["wedge_deg"], 45)
        m2 = ReviewModel(d, "tester")                         # reopen -> decisions restored
        self.assertEqual(m2.decisions.get("H", "p1", "p2"), "same_room")
        self.assertIsNone(m2.decisions.get("H", "p1", "p3"))
        self.assertEqual(m2.progress("H"), {"candidates": 3, "decided": 2})

    def test_open_mode_in_model(self):
        d = self._scan()
        self.assertEqual(ReviewModel(d).mode, "plain")                       # no open_scores.jsonl yet
        with open(os.path.join(d, "open_scores.jsonl"), "w") as fh:
            for a, b, w in (("p1", "p2", 20), ("p2", "p3", 48), ("p3", "p4", 41)):
                fh.write(json.dumps({"house": "H", "a": a, "b": b,
                                     "open": dict(n_keep=40, door_frac=0.1, wedge_a=w, wedge_b=w, wedge_min=w)}) + "\n")
        m = ReviewModel(d, "tester")
        self.assertTrue(m.has_open); self.assertEqual((m.mode, m.wedge_deg), ("open", 40.0))
        self.assertEqual([(x["a"], x["b"]) for x in m.candidates("H")], [("p2", "p3"), ("p3", "p4")])
        self.assertEqual(m.panos_without_candidates("H"), ["p1"])
        data = m.export(os.path.join(d, "g.json"))
        self.assertEqual(data["classifier"]["mode"], "open"); self.assertIn("door-aware", data["classifier"]["rule"])
        m.set_mode("plain")
        self.assertEqual(m.wedge_deg, 45.0); self.assertEqual(len(m.candidates("H")), 3)
        self.assertIn("plain", m.decide("H", "p1", "p2", "same_room")["thresholds"]["mode"])

    @unittest.skipUnless(HAVE_NUMPY, "needs numpy (scan environment)")
    def test_open_wedge_and_compute(self):
        import numpy as np
        from finder import openings
        # inliers spread over 0..180 deg in A; the half at 90..180 deg lies inside A's opening
        xs = np.linspace(0, 2048 * 0.5, 200)
        pts = [[x, 512, x, 512] for x in xs]
        none = openings.open_wedge(pts, (2048, 1024, 2048, 1024), None, None)
        self.assertEqual(none["n_keep"], 200); self.assertAlmostEqual(none["wedge_a"], 144, delta=2)
        ma = np.zeros((1024, 2048), bool); ma[:, 512:1100] = True
        o = openings.open_wedge(pts, (2048, 1024, 2048, 1024), ma, None, n_inlier=400)
        self.assertAlmostEqual(o["door_frac"], 0.5, delta=0.02); self.assertAlmostEqual(o["n_keep"], 200, delta=10)
        self.assertLess(o["wedge_min"], 80)
        dm = openings.dilate(np.pad(np.ones((1, 1), bool), 20))
        self.assertEqual(dm.sum(axis=1).max(), 9); self.assertEqual(dm.sum(axis=0).max(), 21)   # 4 px wide, 10 px tall
        # end to end on a scan: masks in openings/<house>/, inliers in pairs/
        d = self._scan()
        os.makedirs(os.path.join(d, "pairs", "H")); os.makedirs(os.path.join(d, "openings", "H"))
        g = dict(geom(300, 80, 90), img_size_a=[2048, 1024], img_size_b=[2048, 1024], inliers=pts)
        with open(os.path.join(d, "pairs", "H", "p1__p2.json"), "w") as fh:
            json.dump(g, fh)
        for pid in ("p1", "p2", "p3", "p4"):
            m = Image.new("L", (400, 200), 0)
            if pid == "p1":
                m.paste(255, (100, 0, 215, 200))
            m.save(os.path.join(d, "openings", "H", pid + "_mask.png"))
        cnt = openings.compute(d, say=lambda *a: None)
        self.assertEqual(cnt["written"], 1); self.assertEqual(cnt["no_inliers"], 3)
        s = load_scores(d)["H"][("p1", "p2")]
        self.assertGreater(s["open"]["door_frac"], 0.3)
        todo = os.path.join(d, "todo.txt"); self.assertEqual(openings.list_todo(d, todo), [])

    def test_moved_folder_still_works(self):
        d = self._scan()
        moved = os.path.join(self.tmp, "moved"); shutil.move(self.tmp + os.sep + "data", moved + "_data")
        os.makedirs(moved); shutil.move(os.path.join(self.tmp, "scans"), os.path.join(moved, "scans"))
        shutil.move(moved + "_data", os.path.join(moved, "data"))
        m = ReviewModel(os.path.join(moved, "scans", "ds"))
        self.assertTrue(os.path.exists(m.image_path("H", "p4")))


if __name__ == "__main__":
    unittest.main()
