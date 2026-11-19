"""追加7：寫入原子性、授權把關、路徑優先序、失敗契約、回推順序。"""
import csv
import json
import os
import subprocess
import sys
import textwrap
import unittest

import numpy as np

from _support import ROOT, env_override, quiet, synth_matches, temp_dir, write_json

from roma_geo.config import PROJECT_ROOT, Settings
from roma_geo.errors import RomaGeoError
from roma_geo import pipeline
from roma_geo.export_public import export_houses, RESTRICTED_MARKER

from test_pipeline import HOUSE, SIZE, make_image_house, settings_for, write_cache
import check_deliverables as cd


class TestX1SummaryAtomicAndSchema(unittest.TestCase):
    def test_extra_column_leaves_file_intact_and_raises_roma_geo_error(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, stems=("lobby", "hall"))
            s = settings_for(d)
            os.makedirs(s.regions_root(), exist_ok=True)
            summary = os.path.join(s.regions_root(), "_summary.csv")
            with open(summary, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["house", "img_a", "img_b", "legacy_col"])
                w.writerow(["OldH", "a", "b", "keep"])
            before = open(summary, encoding="utf-8").read()
            m, _, _ = synth_matches(n=40, W=SIZE[0], H=SIZE[1], seed=1)
            write_cache(s.cache_root(), HOUSE, "lobby", "hall", m)
            with self.assertRaises(RomaGeoError) as ctx:
                with quiet():
                    pipeline.estimate_poses(settings=s, pairs=[(HOUSE, "lobby", "hall")])
            self.assertIn("legacy_col", str(ctx.exception))
            self.assertEqual(open(summary, encoding="utf-8").read(), before)

    def test_missing_column_leaves_file_intact(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, stems=("lobby", "hall"))
            s = settings_for(d)
            os.makedirs(s.regions_root(), exist_ok=True)
            summary = os.path.join(s.regions_root(), "_summary.csv")
            with open(summary, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["house", "img_a", "img_b"])
                w.writerow(["OldH", "a", "b"])
            before = open(summary, encoding="utf-8").read()
            m, _, _ = synth_matches(n=40, W=SIZE[0], H=SIZE[1], seed=2)
            write_cache(s.cache_root(), HOUSE, "lobby", "hall", m)
            with self.assertRaises(RomaGeoError):
                with quiet():
                    pipeline.estimate_poses(settings=s, pairs=[(HOUSE, "lobby", "hall")])
            self.assertEqual(open(summary, encoding="utf-8").read(), before)


class TestX2PairsCsvThisCallOnly(unittest.TestCase):
    def test_second_house_overwrites_shared_pairs_csv(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, house="H1", stems=("a", "b"))
            make_image_house(ds, house="H2", stems=("c", "d", "e"))
            s = settings_for(d)
            with quiet():
                r1 = pipeline.select_pairs(settings=s, houses=["H1"])
                r2 = pipeline.select_pairs(settings=s, houses=["H2"])
            path = r1.outputs["pairs_csv"]
            self.assertEqual(path, r2.outputs["pairs_csv"])
            rows = list(csv.DictReader(open(path, encoding="utf-8")))
            houses = {r["house"] for r in rows}
            self.assertEqual(houses, {"H2"})
            self.assertEqual(len(rows), 3)


class TestYRestrictedGuard(unittest.TestCase):
    def test_existing_ok_file_survives_abort(self):
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            ds = os.path.join(d, "datasets")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)
            ok = {
                "house": "ZZZ_ok", "n_images": 2, "width": 64, "height": 32,
                "images": ["a.jpg", "b.jpg"],
                "pairs": [{"i": 0, "j": 1, "az_ij": 10.0, "az_ji": 190.0,
                           "az_source": "coarse", "geo_posed": 0}],
            }
            bad = dict(ok, house="AAA_restricted")
            json.dump(bad, open(os.path.join(in_dir, "AAA_restricted.json"), "w",
                                encoding="utf-8"))
            json.dump(ok, open(os.path.join(in_dir, "ZZZ_ok.json"), "w",
                               encoding="utf-8"))
            write_json(os.path.join(out_dir, "ZZZ_ok.json"), ok)
            write_json(os.path.join(out_dir, "AAA_restricted.json"), bad)
            os.makedirs(os.path.join(ds, "AAA_restricted"), exist_ok=True)
            open(os.path.join(ds, "AAA_restricted", RESTRICTED_MARKER), "w",
                 encoding="utf-8").write("restricted\n")
            with self.assertRaises(RomaGeoError):
                with quiet():
                    export_houses(in_dir, out_dir, with_regions=False, datasets_dir=ds)
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "ZZZ_ok.json")))
            self.assertFalse(os.path.isfile(os.path.join(out_dir, "AAA_restricted.json")))

    def test_deliverable_check_reports_when_datasets_missing(self):
        with temp_dir() as d:
            write_json(os.path.join(d, "H.json"), {
                "house": "H", "n_images": 2, "width": 64, "height": 32,
                "images": ["a.jpg", "b.jpg"],
                "schema_version": "house-v2",
                "az_convention": "az(x)=((x/W)*360-180)%360 ; 0deg = image center",
                "pairs": [{"i": 0, "j": 1, "az_ij": 10.0, "az_ji": 190.0,
                           "az_source": "coarse", "has_geometry": False,
                           "semantic_corrected": False}],
            })
            r = cd.check_public_houses(d, datasets_dir=os.path.join(d, "no_such_ds"))
            self.assertEqual(r.status, cd.SKIP)
            self.assertIn("未驗到", r.detail)


class TestZPathPriority(unittest.TestCase):
    def test_output_does_not_move_matrices_or_override_env(self):
        with temp_dir() as d, env_override(ROMA_GEO_MATRICES=os.path.join(d, "envmat"),
                                           ROMA_GEO_DATASETS=None, ROMA_GEO_OUTPUT=None,
                                           ROMA_GEO_CACHE=None, ROMA_GEO_CONNECTIONS=None):
            s = Settings(output=os.path.join(d, "myout"))
            self.assertEqual(os.path.normcase(s.matrices_root()),
                             os.path.normcase(os.path.join(d, "envmat")))
            self.assertFalse(s.matrices_root().startswith(s.output_root()))

    def test_connections_reads_environment(self):
        with temp_dir() as d, env_override(ROMA_GEO_CONNECTIONS=os.path.join(d, "conn")):
            s = Settings()
            self.assertEqual(os.path.normcase(s.connections_dir()),
                             os.path.normcase(os.path.join(d, "conn")))

    def test_empty_semantic_dir_does_not_read_cwd_file(self):
        from roma_geo.export_public import export_houses as exp
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            os.makedirs(in_dir, exist_ok=True)
            house = {
                "house": "H", "n_images": 2, "width": 64, "height": 32,
                "images": ["a.jpg", "b.jpg"],
                "pairs": [{"i": 0, "j": 1, "az_ij": 10.0, "az_ji": 190.0,
                           "az_source": "geo", "geo_posed": 0}],
            }
            json.dump(house, open(os.path.join(in_dir, "H.json"), "w", encoding="utf-8"))
            bait = os.path.join(d, "H.json")
            json.dump({"doors": {"a": [{"az": 40.0, "az0": 20.0, "az1": 60.0}]}},
                      open(bait, "w", encoding="utf-8"))
            old = os.getcwd()
            try:
                os.chdir(d)
                with quiet():
                    exp(in_dir, out_dir, with_regions=False, semantic_correct=True,
                        semantic_dir="")
            finally:
                os.chdir(old)
            pub = json.load(open(os.path.join(out_dir, "H.json"), encoding="utf-8"))
            self.assertFalse(pub["pairs"][0]["semantic_corrected"])


class TestAAFailures(unittest.TestCase):
    def test_matrix_length_mismatch_is_roma_geo_error(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, stems=("lobby", "hall", "yard"))
            s = settings_for(d)
            mdir = s.matrices_root()
            os.makedirs(mdir, exist_ok=True)
            np.save(os.path.join(mdir, HOUSE + ".npy"),
                    np.zeros((3, 3, 3), dtype=np.float32))
            write_json(os.path.join(mdir, HOUSE + "_images.json"),
                       ["lobby.jpg", "hall.jpg"])
            with self.assertRaises(RomaGeoError) as ctx:
                with quiet():
                    pipeline.select_pairs(settings=s, houses=[HOUSE])
            self.assertNotIsInstance(ctx.exception, SystemExit)

    def test_geometry_without_pose_degrades_when_regions_empty(self):
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(os.path.join(d, "regions"), exist_ok=True)
            json.dump({
                "house": "H", "n_images": 2, "width": 64, "height": 32,
                "images": ["a.jpg", "b.jpg"],
                "pairs": [{"i": 0, "j": 1, "az_ij": 10.0, "az_ji": 190.0,
                           "az_source": "geo", "geo_posed": 1}],
            }, open(os.path.join(in_dir, "H.json"), "w", encoding="utf-8"))
            with quiet():
                export_houses(in_dir, out_dir, regions_dir=os.path.join(d, "regions"),
                              with_regions=False)
            pub = json.load(open(os.path.join(out_dir, "H.json"), encoding="utf-8"))
            self.assertTrue(pub["pairs"][0]["has_geometry"])
            self.assertNotIn("R", pub["pairs"][0])
            self.assertNotIn("t", pub["pairs"][0])


class TestABCacheNameUsesDataset(unittest.TestCase):
    def test_underscored_stems_resolved_without_pair_list(self):
        from pose_regions.batch_pose_from_cache import run_pose_batch
        from pose_regions.estimate_relative_pose import CACHE_SUFFIX
        house, na, nb = "Office_A", "lobby_1", "hall_2"
        with temp_dir() as d:
            ds = os.path.join(d, "datasets", house)
            os.makedirs(ds, exist_ok=True)
            open(os.path.join(ds, na + ".jpg"), "wb").close()
            open(os.path.join(ds, nb + ".jpg"), "wb").close()
            from PIL import Image
            Image.new("RGB", SIZE, (1, 2, 3)).save(os.path.join(ds, na + ".jpg"), "JPEG")
            Image.new("RGB", SIZE, (1, 2, 3)).save(os.path.join(ds, nb + ".jpg"), "JPEG")
            cache = os.path.join(d, "cache")
            out = os.path.join(d, "out")
            m, _, _ = synth_matches(n=40, W=SIZE[0], H=SIZE[1], seed=8)
            os.makedirs(cache, exist_ok=True)
            np.savez(os.path.join(cache, "%s_%s_%s%s" % (house, na, nb, CACHE_SUFFIX)),
                     matches=m)
            with quiet():
                n = run_pose_batch(cache, out, datasets_root=os.path.join(d, "datasets"),
                                   pairs=None, on_missing="skip")
            self.assertEqual(n, 1)
            self.assertTrue(os.path.isfile(os.path.join(out, "%s_%s_%s.json" % (house, na, nb))))


class TestACPlaceholdersRestored(unittest.TestCase):
    def test_js_missing_value_marker_is_not_ideographic_period(self):
        path = os.path.join(ROOT, "stitch_tool", "js", "viewmodes", "mode-regions.js")
        text = open(path, encoding="utf-8").read()
        self.assertIn("rj.conf_used != null ? rj.conf_used : '—'", text)
        self.assertNotIn("rj.conf_used != null ? rj.conf_used : '。'", text)


class TestADAzOverlapShared(unittest.TestCase):
    def test_eval_script_matches_delivery_path(self):
        from roma_geo.semantic import az_overlap as delivery
        from semantic_doors.semantic_door_eval import az_overlap as eval_fn
        cases = [(0.0, 200.0, 180.0, 20.0), (350.0, 10.0, 0.0, 340.0),
                 (350.0, 10.0, 355.0, 15.0)]
        for args in cases:
            self.assertAlmostEqual(delivery(*args), eval_fn(*args), places=6, msg=args)


class TestAEHouseDirsAndTests(unittest.TestCase):
    def test_house_dirs_imports_alone(self):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        r = subprocess.run(
            [sys.executable, "-c", "import house_dirs; print(house_dirs.resolve_task_dir.__name__)"],
            cwd=os.path.join(ROOT, "stitch_tool"), env=env,
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_semantic_fallback_path_runs(self):
        code = textwrap.dedent("""
            import sys
            class Block:
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "semantic_doors" or fullname.startswith("semantic_doors."):
                        raise ImportError("blocked")
                    return None
            sys.meta_path.insert(0, Block())
            from roma_geo.semantic import _load_extract_doors
            fn = _load_extract_doors()
            print("FALLBACK", callable(fn))
        """)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        r = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("FALLBACK True", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
