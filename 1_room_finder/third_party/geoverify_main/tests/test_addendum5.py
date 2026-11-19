"""追加5：修正前會失敗的行為測試。每一項對應 N–U 的一則事實。"""
import csv
import json
import os
import subprocess
import sys
import textwrap
import unittest

import numpy as np
from PIL import Image

from _support import ROOT, SRC, env_override, quiet, synth_matches, temp_dir, write_json  # noqa: F401

from roma_geo.config import PROJECT_ROOT, Settings
from roma_geo.errors import RomaGeoError
from roma_geo.names import house_of
from roma_geo import pipeline
from roma_geo.semantic import az_overlap, invert_pose, region_az_span

from test_pipeline import (
    ENV_CLEAR, HOUSE, SIZE, StubMatcher, make_image_house, settings_for, write_cache,
)


def _save_jpg(path, size=SIZE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, (12, 24, 36)).save(path, "JPEG")
    return path


class TestN1MatcherDoesNotStealSysPath(unittest.TestCase):
    def test_get_cache_info_leaves_user_config_in_place(self):
        with temp_dir() as d:
            with open(os.path.join(d, "config.py"), "w", encoding="utf-8") as f:
                f.write("marker = 'user-config'\n")
            code = textwrap.dedent("""
                import os, sys
                from roma_match.matcher import RoMaMatcher
                m = RoMaMatcher(workers=1, use_cache=False)
                m._get_cache_info(os.path.join(%r, "a.jpg"),
                                  os.path.join(%r, "b.jpg"), "direct")
                sys.modules.pop("config", None)
                import config
                print(getattr(config, "marker", "NO_MARKER"))
            """ % (d, d))
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            r = subprocess.run(
                [sys.executable, "-c", code],
                cwd=d, env=env, capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("user-config", r.stdout)


class TestN2SemanticDoorsWithoutProjectOnPath(unittest.TestCase):
    def test_extract_house_doors_outside_project(self):
        with temp_dir() as d:
            code = textwrap.dedent("""
                import os, sys
                sys.path.insert(0, %r)
                os.chdir(%r)
                # 專案根不在搜尋路徑上
                sys.path[:] = [p for p in sys.path
                               if os.path.normcase(os.path.abspath(p) if p else p)
                               != os.path.normcase(%r)]
                from roma_geo.semantic import extract_house_doors
                from roma_geo.errors import RomaGeoError
                try:
                    doors = extract_house_doors(%r, "NoHouse")
                except RomaGeoError as e:
                    print("READABLE", e)
                else:
                    print("OK", len(doors))
            """ % (SRC, d, ROOT, d))
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            r = subprocess.run(
                [sys.executable, "-c", code],
                cwd=d, env=env, capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertTrue(r.stdout.strip().startswith("OK"), r.stdout + r.stderr)
            self.assertNotIn("ModuleNotFoundError", r.stdout + r.stderr)


class TestN3SettingsWritePathsStayUnderOutput(unittest.TestCase):
    def test_only_datasets_output_cache_never_touch_install_source(self):
        with temp_dir() as d, env_override(
                ROMA_GEO_DATASETS=None, ROMA_GEO_MATRICES=None,
                ROMA_GEO_OUTPUT=None, ROMA_GEO_CACHE=None):
            ds = os.path.join(d, "datasets")
            out = os.path.join(d, "out")
            cache = os.path.join(d, "cache")
            s = Settings(datasets=ds, output=out, cache=cache)
            root = os.path.normcase(os.path.abspath(PROJECT_ROOT))
            write_paths = [
                s.output_root(), s.cache_root(),
                s.house_data_dir(), s.semantic_out_dir(),
                s.public_dir(), s.regions_root(),
            ]
            for p in write_paths:
                npth = os.path.normcase(os.path.abspath(p))
                self.assertFalse(
                    npth == root or npth.startswith(root + os.sep),
                    "%s 落在安裝來源 %s" % (p, PROJECT_ROOT))
            self.assertFalse(os.path.normcase(s.matrices_root()).startswith(
                os.path.normcase(s.output_root()) + os.sep))


class TestN4CovisRankingEvalUsesPackageConfig(unittest.TestCase):
    def test_module_does_not_insert_project_root(self):
        import inspect
        import pose_regions.covis_ranking_eval as m
        src = inspect.getsource(m)
        self.assertNotIn("sys.path.insert", src)
        self.assertIn("roma_geo.config", src)

    def test_helpers_import_without_project_root_on_path(self):
        code = textwrap.dedent("""
            import os, sys
            sys.path[:] = [p for p in sys.path
                           if os.path.normcase(os.path.abspath(p) if p else p)
                           != os.path.normcase(%r)]
            from pose_regions.covis_ranking_eval import _datasets_root
            print(_datasets_root())
        """ % (ROOT,))
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        r = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(ROOT), env=env, capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


class TestO1PartialMatchFailuresAreReported(unittest.TestCase):
    def test_match_pairs_reports_n_fail_and_reasons(self):
        class Partial(StubMatcher):
            def match_direct(self, path_a, path_b):
                if "yard" in path_a or "yard" in path_b:
                    raise RuntimeError("simulated OOM")
                return super().match_direct(path_a, path_b)

        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, stems=("lobby", "hall", "yard"))
            s = settings_for(d)
            with quiet():
                sel = pipeline.select_pairs(settings=s, houses=[HOUSE])
            stub = Partial(s.cache_root())
            with env_override(**ENV_CLEAR), quiet():
                result = pipeline.match_pairs(
                    settings=s, pairs_csv=sel.outputs["pairs_csv"], matcher=stub)
            self.assertGreaterEqual(result.outputs["n_done"], 1)
            self.assertGreaterEqual(result.outputs["n_fail"], 1)
            self.assertTrue(result.outputs.get("failures"))
            self.assertTrue(any("OOM" in x or "simulated" in x
                                for x in result.outputs["failures"]))
            self.assertIn("ok_pairs", result.outputs)
            self.assertEqual(len(result.outputs["ok_pairs"]), result.outputs["n_done"])


class TestO2EvaluateRankingWritesCsv(unittest.TestCase):
    def test_writes_ranking_eval_when_gt_present(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            house_dir, _ = make_image_house(ds, stems=("lobby", "hall"))
            write_json(os.path.join(house_dir, "floor_HOTSPOT.json"), {
                "HOTSPOTOFROOM": [{
                    "IDName": "lobby.jpg", "Roomidx": 0,
                    "ToIDName": {"IDName": ["hall.jpg"]},
                    "hotspotx": [0.5], "hotspoty": [0.5],
                }],
            })
            s = settings_for(d)
            os.makedirs(s.regions_root(), exist_ok=True)
            summary = os.path.join(s.regions_root(), "_summary.csv")
            with open(summary, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "house", "img_a", "img_b", "same_room", "conf_used",
                    "flag_no_covisibility", "n_raw", "n_conf", "n_inlier",
                    "inlier_rate", "n_regions", "region1_frac",
                ])
                w.writeheader()
                w.writerow({
                    "house": HOUSE, "img_a": "lobby", "img_b": "hall",
                    "same_room": "1", "conf_used": "0.65",
                    "flag_no_covisibility": "0", "n_raw": "10", "n_conf": "8",
                    "n_inlier": "7", "inlier_rate": "0.7", "n_regions": "1",
                    "region1_frac": "0.9",
                })
            with env_override(**ENV_CLEAR), quiet():
                result = pipeline.evaluate_ranking(settings=s, summary_csv=summary)
            self.assertFalse(result.skipped, result.message)
            csv_path = result.outputs.get("ranking_eval_csv")
            self.assertTrue(csv_path and os.path.isfile(csv_path), result.outputs)


class TestP1DegradedPairsAllHaveAzimuth(unittest.TestCase):
    def test_prepare_house_keeps_all_pairs_null_azimuth_when_missing(self):
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
            for p in data["pairs"]:
                self.assertIn("az_ij", p)
                self.assertIn("az_ji", p)
            with quiet():
                exp = pipeline.export_public_houses(settings=s, houses=[HOUSE])
            pub = json.load(open(exp.outputs["public_json"], encoding="utf-8"))
            self.assertEqual(len(pub["pairs"]), 3)
            import check_deliverables as cd
            r = cd.check_public_houses(s.public_dir(), datasets_dir=s.datasets_root())
            self.assertEqual(r.status, cd.OK, r.problems)


class TestP2ConnectivityFlagOnPublicHouse(unittest.TestCase):
    def test_public_house_carries_has_connectivity_matrix(self):
        from roma_geo.export_public import house_to_public
        d = {
            "house": HOUSE, "n_images": 2, "width": 64, "height": 32,
            "images": ["lobby.jpg", "hall.jpg"],
            "az_convention": "az(x)=((x/W)*360-180)%360 ; 0deg = image center",
            "has_connectivity_matrix": False,
            "pairs": [{"i": 0, "j": 1, "az_ij": 10.0, "az_ji": 190.0,
                       "az_source": "geo", "geo_posed": 1}],
        }
        pub = house_to_public(d)
        self.assertIn("has_connectivity_matrix", pub)
        self.assertIs(pub["has_connectivity_matrix"], False)


class TestP3DegradeWhenImagesJsonMissing(unittest.TestCase):
    def test_npy_without_images_json_is_degraded(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, stems=("lobby", "hall"))
            s = settings_for(d)
            os.makedirs(s.matrices_root(), exist_ok=True)
            np.save(os.path.join(s.matrices_root(), HOUSE + ".npy"),
                    np.zeros((2, 2, 3), dtype=np.float32))
            m, _, _ = synth_matches(n=40, W=SIZE[0], H=SIZE[1], seed=5)
            write_cache(s.cache_root(), HOUSE, "lobby", "hall", m)
            with quiet():
                pipeline.estimate_poses(settings=s, pairs=[(HOUSE, "lobby", "hall")])
                result = pipeline.prepare_house(settings=s, houses=[HOUSE])
            names = [x.input_name for x in result.degradations]
            self.assertIn("upstream_connectivity", names)
            data = json.load(open(result.outputs["house_json"], encoding="utf-8"))
            self.assertFalse(data["has_connectivity_matrix"])


class TestQ1RegionsSwitchDoesNotDropPose(unittest.TestCase):
    def test_no_regions_still_attaches_pose_and_can_correct(self):
        from roma_geo.export_public import export_houses
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            regions = os.path.join(d, "regions")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(regions, exist_ok=True)
            internal = {
                "house": HOUSE, "n_images": 2, "width": 64, "height": 32,
                "images": ["lobby.jpg", "hall.jpg"],
                "az_convention": "az(x)=((x/W)*360-180)%360 ; 0deg = image center",
                "pairs": [{"i": 0, "j": 1, "az_ij": 10.0, "az_ji": 190.0,
                           "az_source": "geo", "geo_posed": 1}],
            }
            json.dump(internal, open(os.path.join(in_dir, HOUSE + ".json"), "w",
                                     encoding="utf-8"))
            write_json(os.path.join(regions, "%s_lobby_hall.json" % HOUSE), {
                "pose": {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                         "t": [1.0, 0.0, 0.0]},
                "regions": [{
                    "n": 10, "frac_of_inliers": 0.9,
                    "a": {"polygon": [[1, 1], [2, 1], [2, 2]],
                          "area_frac": 0.1, "az_range": [25.0, 55.0]},
                    "b": {"polygon": [[1, 1], [2, 1], [2, 2]],
                          "area_frac": 0.1, "az_range": [0.0, 10.0]},
                }],
            })
            sem = os.path.join(d, "sem")
            write_json(os.path.join(sem, HOUSE + ".json"), {
                "house": HOUSE,
                "doors": {"lobby": [{"az": 40.0, "az0": 20.0, "az1": 60.0}]},
            })
            with quiet():
                export_houses(in_dir, out_dir, regions_dir=regions,
                              with_regions=False, semantic_correct=True,
                              semantic_dir=sem)
            pub = json.load(open(os.path.join(out_dir, HOUSE + ".json"),
                                 encoding="utf-8"))
            p = pub["pairs"][0]
            self.assertIn("R", p)
            self.assertIn("t", p)
            self.assertNotIn("regions", p)
            self.assertTrue(p["semantic_corrected"])
            self.assertAlmostEqual(p["az_ij"], 40.0)


class TestQ2MatchPairsRejectsBadMode(unittest.TestCase):
    def test_illegal_mode_is_rejected_at_match_pairs(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, stems=("lobby", "hall"))
            s = settings_for(d)
            pairs_csv = os.path.join(d, "pairs.csv")
            with open(pairs_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["house", "img_a", "img_b"])
                w.writerow([HOUSE, "lobby", "hall"])
            stub = StubMatcher(s.cache_root())
            with self.assertRaises(ValueError) as ctx:
                pipeline.match_pairs(settings=s, pairs_csv=pairs_csv,
                                     matcher=stub, mode="direcct")
            self.assertIn("direct", str(ctx.exception))
            self.assertIn("core", str(ctx.exception))


class TestR1CacheIdentityFromPairs(unittest.TestCase):
    def test_underscored_names_keep_caller_identity(self):
        from pose_regions.batch_pose_from_cache import run_pose_batch
        from pose_regions.estimate_relative_pose import CACHE_SUFFIX
        house, na, nb = "Office_A", "lobby_1", "hall_2"
        with temp_dir() as d:
            ds = os.path.join(d, "datasets", house)
            os.makedirs(ds, exist_ok=True)
            _save_jpg(os.path.join(ds, na + ".jpg"))
            _save_jpg(os.path.join(ds, nb + ".jpg"))
            cache = os.path.join(d, "cache")
            out = os.path.join(d, "out")
            m, _, _ = synth_matches(n=40, W=SIZE[0], H=SIZE[1], seed=6)
            os.makedirs(cache, exist_ok=True)
            np.savez(os.path.join(cache, "%s_%s_%s%s" % (house, na, nb, CACHE_SUFFIX)),
                     matches=m)
            with quiet():
                n = run_pose_batch(
                    cache, out, datasets_root=os.path.join(d, "datasets"),
                    pairs=[(house, na, nb)], on_missing="error")
            self.assertEqual(n, 1)
            self.assertTrue(os.path.isfile(os.path.join(out, "%s_%s_%s.json"
                                                        % (house, na, nb))))
            row = list(csv.DictReader(open(os.path.join(out, "_summary.csv"),
                                           encoding="utf-8")))[0]
            self.assertEqual(row["house"], house)
            self.assertEqual(row["img_a"], na)
            self.assertEqual(row["img_b"], nb)


class TestR2HouseOfIgnoresDatasetRootMarkers(unittest.TestCase):
    def test_dataset_root_named_direct_match_does_not_steal_house(self):
        self.assertEqual(
            house_of(os.path.join("D:", "direct_match_data", "HouseA", "x.jpg")),
            "HouseA")
        self.assertEqual(
            house_of(os.path.join("D:", "DollhouseTask_root", "HouseB", "x.jpg")),
            "HouseB")


class TestR3ImageExtensionsAreShared(unittest.TestCase):
    def test_jpeg_uppercase_is_found(self):
        from pose_regions import find_house_image
        tool = os.path.join(ROOT, "stitch_tool")
        if tool not in sys.path:
            sys.path.append(tool)
        from house_dirs import resolve_task_dir
        with temp_dir() as d:
            house = "Hjpeg"
            cur = os.path.join(d, house)
            os.makedirs(cur, exist_ok=True)
            path = os.path.join(cur, "img.JPEG")
            open(path, "wb").close()
            found = find_house_image(house, "img", datasets_root=d)
            self.assertEqual(os.path.normcase(found or ""), os.path.normcase(path))
            task = resolve_task_dir(d, house)
            self.assertEqual(os.path.normcase(task or ""), os.path.normcase(cur))


class TestSSummaryAccumulatesAcrossHouses(unittest.TestCase):
    def test_second_house_does_not_erase_first_summary(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, house="H1", stems=("a", "b"))
            make_image_house(ds, house="H2", stems=("c", "d"))
            s = settings_for(d)
            m, _, _ = synth_matches(n=40, W=SIZE[0], H=SIZE[1], seed=7)
            write_cache(s.cache_root(), "H1", "a", "b", m)
            write_cache(s.cache_root(), "H2", "c", "d", m)
            with quiet():
                pipeline.estimate_poses(settings=s, pairs=[("H1", "a", "b")])
                pipeline.estimate_poses(settings=s, pairs=[("H2", "c", "d")])
            rows = list(csv.DictReader(open(
                os.path.join(s.regions_root(), "_summary.csv"), encoding="utf-8")))
            houses = {r["house"] for r in rows}
            self.assertEqual(houses, {"H1", "H2"})


class TestU1InvertedPoseRounded(unittest.TestCase):
    def test_invert_pose_json_has_six_decimals(self):
        import math
        import re
        yaw = math.radians(33.333)
        R = [[math.cos(yaw), 0.0, math.sin(yaw)],
             [0.0, 1.0, 0.0],
             [-math.sin(yaw), 0.0, math.cos(yaw)]]
        t = [0.3, 0.4, 0.86602540378]
        Ri, ti = invert_pose(R, t)
        blob = json.dumps({"R": Ri, "t": ti})
        fracs = re.findall(r"\.\d+", blob)
        self.assertTrue(fracs)
        self.assertTrue(all(len(f) - 1 <= 6 for f in fracs), fracs)


class TestU2RestrictedAbortDeletesWritten(unittest.TestCase):
    def test_prior_house_is_removed_when_later_house_is_restricted(self):
        from roma_geo.export_public import export_houses, RESTRICTED_MARKER
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            ds = os.path.join(d, "datasets")
            os.makedirs(in_dir, exist_ok=True)
            ok = {
                "house": "OKHouse", "n_images": 2, "width": 64, "height": 32,
                "images": ["a.jpg", "b.jpg"],
                "pairs": [{"i": 0, "j": 1, "az_ij": 10.0, "az_ji": 190.0,
                           "az_source": "coarse", "geo_posed": 0}],
            }
            bad = dict(ok, house="BadHouse")
            json.dump(ok, open(os.path.join(in_dir, "OKHouse.json"), "w",
                               encoding="utf-8"))
            json.dump(bad, open(os.path.join(in_dir, "BadHouse.json"), "w",
                                encoding="utf-8"))
            os.makedirs(os.path.join(ds, "BadHouse"), exist_ok=True)
            open(os.path.join(ds, "BadHouse", RESTRICTED_MARKER), "w",
                 encoding="utf-8").write("restricted\n")
            with self.assertRaises(RomaGeoError):
                with quiet():
                    export_houses(in_dir, out_dir, with_regions=False,
                                  datasets_dir=ds)
            self.assertFalse(os.path.exists(os.path.join(out_dir, "OKHouse.json")))
            self.assertFalse(os.path.exists(os.path.join(out_dir, "BadHouse.json")))


class TestU3SafeIdRejectsDotDot(unittest.TestCase):
    def test_dot_and_dotdot_are_rejected(self):
        tool = os.path.join(ROOT, "stitch_tool")
        if tool not in sys.path:
            sys.path.append(tool)
        import server
        self.assertFalse(server._safe_id("."))
        self.assertFalse(server._safe_id(".."))
        self.assertTrue(server._safe_id("DollhouseTask_1"))


class TestU4ScoreCacheDoesNotUnpickle(unittest.TestCase):
    def test_roundtrip_without_pickle_and_object_array_rejected(self):
        from roma_match.roma_score_cache import RomaScoreCache
        with temp_dir() as d:
            path = os.path.join(d, "scores.npz")
            c = RomaScoreCache()
            c.put("H", "a.jpg", "b.jpg", match_count=3, conf_sum=1.5)
            c.save(path)
            loaded = RomaScoreCache.load(path)
            self.assertEqual(loaded.num_edges(), 1)
            bad = os.path.join(d, "bad.npz")
            np.savez(bad, houses=np.array(["H"], dtype=object),
                     a_names=np.array(["a"], dtype=object),
                     b_names=np.array(["b"], dtype=object),
                     match_count=np.array([1], dtype=np.int32),
                     conf_sum=np.array([1.0], dtype=np.float32),
                     conf_mean=np.array([1.0], dtype=np.float32))
            with self.assertRaises(Exception):
                RomaScoreCache.load(bad)


class TestU5DeadPolygonBranchRemoved(unittest.TestCase):
    def test_polygon_without_az_range_is_unusable(self):
        self.assertIsNone(region_az_span({"polygon": [[0, 0], [1, 0], [1, 1]]}))
        src = open(os.path.join(ROOT, "src", "roma_geo", "semantic.py"),
                   encoding="utf-8").read()
        self.assertNotIn("xs = [p[0] for p in poly]", src)


class TestU6WrapOverlapCountsBothSegments(unittest.TestCase):
    def test_two_long_wrapping_arcs_sum_both_overlaps(self):
        # [0,200] 與 [180,20]（寬 200°）交集為 [180,200] 與 [0,20]，共 40°。
        self.assertAlmostEqual(az_overlap(0.0, 200.0, 180.0, 20.0), 40.0)
        self.assertAlmostEqual(az_overlap(350.0, 10.0, 355.0, 15.0), 15.0)


class TestU7VerifyPairsAcceptsGenerator(unittest.TestCase):
    def test_progress_with_generator_does_not_ask_len(self):
        import roma_geo as rg
        orig = rg.verify_pair

        def fake(a, b, **kw):
            return {"img_a": "a", "img_b": "b", "pose": None,
                    "regions": [], "timing": {"total_s": 0.0}}

        rg.verify_pair = fake
        try:
            gen = (x for x in [("a.jpg", "b.jpg")])
            out = list(rg.verify_pairs(gen, progress=True, matcher=object()))
            self.assertEqual(len(out), 1)
        finally:
            rg.verify_pair = orig


if __name__ == "__main__":
    unittest.main(verbosity=2)
