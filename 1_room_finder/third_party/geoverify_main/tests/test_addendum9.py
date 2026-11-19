"""追加9：名稱守衛位置、配對來源分開、檢查工具路徑、逐對產物降級。"""
import csv
import json
import os
import unittest

import numpy as np

from _support import env_override, quiet, synth_matches, temp_dir, write_json

from roma_geo.config import PROJECT_ROOT, Settings
from roma_geo.errors import RomaGeoError
from roma_geo import pipeline
from roma_geo.export_public import export_houses
from roma_geo.ingest_connections import convert_connections
from roma_geo.names import is_path_component

from pose_regions.estimate_relative_pose import CACHE_SUFFIX
from test_pipeline import HOUSE, SIZE, StubMatcher, make_image_house, settings_for, write_cache
from test_addendum8 import _empty_datasets, _internal_house, _public_house, public_doc_field_errors
import check_deliverables as cd


class TestAK1ValidateAfterFilter(unittest.TestCase):
    def test_named_export_ignores_illegal_neighbor(self):
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            os.makedirs(in_dir, exist_ok=True)
            write_json(os.path.join(in_dir, "Good.json"), _internal_house("Good"))
            write_json(os.path.join(in_dir, "weird.json"), _internal_house("A/B"))
            with quiet():
                written = export_houses(
                    in_dir, out_dir, with_regions=False,
                    datasets_dir=_empty_datasets(d), houses=["Good"],
                )
            self.assertEqual(len(written), 1)
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "Good.json")))
            self.assertFalse(os.path.isfile(os.path.join(out_dir, "A.json")))
            self.assertFalse(os.path.isfile(os.path.join(os.path.dirname(out_dir), "B.json")))

    def test_unnamed_export_stops_on_illegal_name(self):
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            os.makedirs(in_dir, exist_ok=True)
            write_json(os.path.join(in_dir, "Good.json"), _internal_house("Good"))
            write_json(os.path.join(in_dir, "weird.json"), _internal_house("A/B"))
            with self.assertRaises(RomaGeoError) as ctx:
                with quiet():
                    export_houses(in_dir, out_dir, with_regions=False,
                                  datasets_dir=_empty_datasets(d))
            self.assertIn("A/B", str(ctx.exception))
            self.assertIn("房子 A/B", str(ctx.exception))
            self.assertFalse(os.path.isdir(out_dir) and os.listdir(out_dir))


class TestAK2PrepareValidatesNames(unittest.TestCase):
    def test_prepare_house_rejects_parent_ref(self):
        with temp_dir() as d:
            s = settings_for(d)
            parent = os.path.dirname(s.house_data_dir())
            os.makedirs(parent, exist_ok=True)
            before = set(os.listdir(parent))
            with self.assertRaises(RomaGeoError) as ctx:
                with quiet():
                    pipeline.prepare_house(settings=s, houses=["../escaped"])
            self.assertIn("../escaped", str(ctx.exception))
            self.assertFalse(os.path.isfile(os.path.join(parent, "escaped.json")))
            self.assertEqual(set(os.listdir(parent)), before)

    def test_prepare_semantic_rejects_parent_ref(self):
        with temp_dir() as d:
            s = settings_for(d)
            parent = os.path.dirname(s.semantic_out_dir())
            os.makedirs(parent, exist_ok=True)
            before = set(os.listdir(parent))
            with self.assertRaises(RomaGeoError):
                with quiet():
                    pipeline.prepare_semantic(settings=s, houses=["../escaped"])
            self.assertFalse(os.path.isfile(os.path.join(parent, "escaped.json")))
            self.assertEqual(set(os.listdir(parent)), before)

    def test_select_pairs_rejects_parent_ref_even_if_images_exist(self):
        with temp_dir() as d:
            s = settings_for(d)
            make_image_house(d, house="escaped", stems=("a", "b"))
            os.makedirs(s.regions_root(), exist_ok=True)
            before = set(os.listdir(d))
            with self.assertRaises(RomaGeoError) as ctx:
                with quiet():
                    pipeline.select_pairs(settings=s, houses=["../escaped"])
            self.assertIn("../escaped", str(ctx.exception))
            self.assertFalse(os.path.isfile(
                os.path.join(s.regions_root(), pipeline.STEP_PAIRS_FILENAME)))
            self.assertEqual(set(os.listdir(d)), before)

    def test_match_pairs_rejects_parent_ref_in_csv(self):
        with temp_dir() as d:
            s = settings_for(d)
            make_image_house(d, house="escaped", stems=("a", "b"))
            os.makedirs(s.regions_root(), exist_ok=True)
            csv_path = os.path.join(s.regions_root(), pipeline.STEP_PAIRS_FILENAME)
            with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["house", "img_a", "img_b", "score"])
                w.writerow(["../escaped", "a", "b", ""])
            matcher = StubMatcher(s.cache_root())
            with self.assertRaises(RomaGeoError) as ctx:
                with quiet():
                    pipeline.match_pairs(settings=s, pairs_csv=csv_path, matcher=matcher)
            self.assertIn("../escaped", str(ctx.exception))
            self.assertFalse(os.path.isdir(s.cache_root()) and os.listdir(s.cache_root()))

    def test_estimate_poses_rejects_parent_ref_in_pairs(self):
        with temp_dir() as d:
            s = settings_for(d)
            make_image_house(d, house="escaped", stems=("a", "b"))
            os.makedirs(s.regions_root(), exist_ok=True)
            os.makedirs(s.cache_root(), exist_ok=True)
            m, _, _ = synth_matches(n=40, W=SIZE[0], H=SIZE[1], seed=1)
            leaked_cache = os.path.join(
                s.cache_root(), "..", "escaped_a_b%s" % CACHE_SUFFIX)
            os.makedirs(os.path.dirname(os.path.abspath(leaked_cache)), exist_ok=True)
            np.savez(leaked_cache, matches=np.asarray(m, dtype=float))
            parent = os.path.dirname(s.regions_root())
            with self.assertRaises(RomaGeoError) as ctx:
                with quiet():
                    pipeline.estimate_poses(settings=s, pairs=[("../escaped", "a", "b")])
            self.assertIn("../escaped", str(ctx.exception))
            self.assertFalse(os.path.isfile(os.path.join(parent, "escaped_a_b.json")))


class TestAK4MissingMessageNamesHouseField(unittest.TestCase):
    def test_filename_match_but_house_field_differs(self):
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            os.makedirs(in_dir, exist_ok=True)
            write_json(os.path.join(in_dir, "Outer.json"), _internal_house("Inner"))
            with self.assertRaises(RomaGeoError) as ctx:
                with quiet():
                    export_houses(in_dir, out_dir, with_regions=False,
                                  datasets_dir=_empty_datasets(d), houses=["Outer"])
            msg = str(ctx.exception)
            self.assertIn("house 欄位", msg)
            self.assertTrue(os.path.isfile(os.path.join(in_dir, "Outer.json")))


class TestAK5RealHouseNames(unittest.TestCase):
    def test_sample_house_names_are_path_components(self):
        names = (
            "DollhouseTask_58553", "DollhouseTask_58582_NoOutdoor",
            "DollhouseTask_58472_NoOutdoor", "Office_A", "RestrictedFake",
        )
        for name in names:
            self.assertTrue(is_path_component(name), name)
        self.assertFalse(is_path_component("A/B"))
        self.assertFalse(is_path_component("../escaped"))


class TestALStepPairsSeparateFromBatch(unittest.TestCase):
    def test_select_pairs_does_not_clobber_batch_source(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, stems=("lobby", "hall"))
            s = settings_for(d)
            os.makedirs(s.regions_root(), exist_ok=True)
            batch = os.path.join(s.regions_root(), pipeline.BATCH_PAIRS_FILENAME)
            with open(batch, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["house", "img_a", "img_b", "score"])
                for i in range(6):
                    w.writerow(["OldH", "a%d" % i, "b%d" % i, 0.1])
            before = open(batch, encoding="utf-8").read()
            with quiet():
                r = pipeline.select_pairs(settings=s, houses=[HOUSE])
            self.assertEqual(open(batch, encoding="utf-8").read(), before)
            self.assertTrue(r.outputs["pairs_csv"].endswith(pipeline.STEP_PAIRS_FILENAME))
            rows = list(csv.DictReader(open(r.outputs["pairs_csv"], encoding="utf-8")))
            self.assertEqual({row["house"] for row in rows}, {HOUSE})

    def test_three_houses_each_step_only_that_house(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, house="H1", stems=("a", "b"))
            make_image_house(ds, house="H2", stems=("c", "d", "e"))
            make_image_house(ds, house="H3", stems=("f", "g"))
            s = settings_for(d)
            os.makedirs(s.regions_root(), exist_ok=True)
            batch = os.path.join(s.regions_root(), pipeline.BATCH_PAIRS_FILENAME)
            open(batch, "w", encoding="utf-8").write("house,img_a,img_b,score\nKeep,x,y,1\n")
            with quiet():
                r1 = pipeline.select_pairs(settings=s, houses=["H1"])
                r2 = pipeline.select_pairs(settings=s, houses=["H2"])
                r3 = pipeline.select_pairs(settings=s, houses=["H3"])
            self.assertEqual(len(list(csv.DictReader(open(r3.outputs["pairs_csv"], encoding="utf-8")))), 1)
            self.assertEqual(r2.outputs["n_pairs"], 3)
            self.assertEqual(open(batch, encoding="utf-8").read(),
                             "house,img_a,img_b,score\nKeep,x,y,1\n")


class TestAM1CheckPathsFollowSettings(unittest.TestCase):
    def test_output_env_moves_write_check_targets(self):
        with temp_dir() as d, env_override(
                ROMA_GEO_OUTPUT=d,
                ROMA_GEO_HOUSE_DATA=None,
                ROMA_GEO_PUBLIC_HOUSES=None,
                ROMA_GEO_DATASETS=None,
                ROMA_GEO_MATRICES=None):
            s = Settings()
            captured = []

            def stub(name):
                def inner(path, *a, **k):
                    captured.append((name, path))
                    return cd.Result(name, cd.SKIP, path)
                return inner

            orig_pair = cd.check_pair_jsons
            orig_sum = cd.check_summary_csv
            orig_mat = cd.check_matrices
            orig_house = cd.check_house_json
            orig_pub = cd.check_public_houses
            try:
                cd.check_pair_jsons = stub("check_pair_jsons")
                cd.check_summary_csv = stub("check_summary_csv")
                cd.check_matrices = stub("check_matrices")
                cd.check_house_json = stub("check_house_json")
                cd.check_public_houses = stub("check_public_houses")
                cd.run_checks()
            finally:
                cd.check_pair_jsons = orig_pair
                cd.check_summary_csv = orig_sum
                cd.check_matrices = orig_mat
                cd.check_house_json = orig_house
                cd.check_public_houses = orig_pub
            by = dict(captured)
            out = os.path.normcase(os.path.abspath(s.output_root()))
            for key in ("check_pair_jsons", "check_summary_csv", "check_house_json",
                        "check_public_houses"):
                path = os.path.normcase(os.path.abspath(by[key]))
                self.assertTrue(path == out or path.startswith(out + os.sep), key)
            self.assertEqual(os.path.normcase(os.path.abspath(by["check_matrices"])),
                             os.path.normcase(os.path.abspath(s.matrices_root())))

    def test_no_env_keeps_install_source_house_dir(self):
        with env_override(ROMA_GEO_OUTPUT=None, ROMA_GEO_HOUSE_DATA=None,
                          ROMA_GEO_PUBLIC_HOUSES=None):
            s = Settings()
            self.assertEqual(os.path.normcase(s.house_data_dir()),
                             os.path.normcase(os.path.join(PROJECT_ROOT, "stitch_tool", "data")))
            self.assertEqual(os.path.normcase(s.public_dir()),
                             os.path.normcase(os.path.join(s.output_root(), "public_houses")))


class TestAM2MissingDatasetsIsSkip(unittest.TestCase):
    def test_public_check_skips_when_datasets_missing(self):
        with temp_dir() as d:
            write_json(os.path.join(d, "H.json"), _public_house("H"))
            r = cd.check_public_houses(d, datasets_dir=os.path.join(d, "no_ds"))
            self.assertEqual(r.status, cd.SKIP)
            self.assertIn("未驗到", r.detail)

    def test_report_counts_skip_not_fail(self):
        import io
        with temp_dir() as d:
            write_json(os.path.join(d, "H.json"), _public_house("H"))
            r = cd.check_public_houses(d, datasets_dir=os.path.join(d, "no_ds"))
            buf = io.StringIO()
            n_fail = cd.report([r], stream=buf)
            self.assertEqual(n_fail, 0)
            self.assertIn("1 項略過", buf.getvalue())


class TestAM3DecodeErrorsAreProblems(unittest.TestCase):
    def test_truncated_and_undecodable_are_reported(self):
        with temp_dir() as d:
            pub = os.path.join(d, "pub")
            houses = os.path.join(d, "houses")
            os.makedirs(pub, exist_ok=True)
            os.makedirs(houses, exist_ok=True)
            write_json(os.path.join(pub, "Good.json"), _public_house("Good"))
            open(os.path.join(pub, "Trunc.json"), "w", encoding="utf-8").write('{"house":')
            open(os.path.join(pub, "BadEnc.json"), "wb").write(b"\xff\xfe{")
            write_json(os.path.join(houses, "Good.json"), _internal_house("Good"))
            open(os.path.join(houses, "Trunc.json"), "w", encoding="utf-8").write("{")
            open(os.path.join(houses, "BadEnc.json"), "wb").write(b"\xff\xfe{")
            ds = _empty_datasets(d)
            r_pub = cd.check_public_houses(pub, datasets_dir=ds)
            r_house = cd.check_house_json(houses)
            self.assertEqual(r_pub.status, cd.FAIL)
            self.assertTrue(any("Trunc" in p and "無法解析" in p for p in r_pub.problems), r_pub.problems)
            self.assertTrue(any("BadEnc" in p and "無法解析" in p for p in r_pub.problems), r_pub.problems)
            self.assertEqual(r_house.status, cd.FAIL)
            self.assertTrue(any("Trunc" in p for p in r_house.problems), r_house.problems)
            self.assertTrue(any("BadEnc" in p for p in r_house.problems), r_house.problems)


class TestANIngestReadErrors(unittest.TestCase):
    def test_truncated_json_is_roma_geo_error_and_writes_nothing(self):
        with temp_dir() as d:
            src = os.path.join(d, "src")
            dest = os.path.join(d, "mat")
            os.makedirs(src, exist_ok=True)
            write_json(os.path.join(src, "Good.json"), {
                "house_id": "Good",
                "pairs": [{"pano_a": "a.jpg", "pano_b": "b.jpg", "probability": 0.9}],
            })
            open(os.path.join(src, "Bad.json"), "w", encoding="utf-8").write('{"house_id":')
            with self.assertRaises(RomaGeoError) as ctx:
                with quiet():
                    convert_connections(src, dest)
            self.assertIn("Bad.json", str(ctx.exception))
            self.assertNotIsInstance(ctx.exception, json.JSONDecodeError)
            self.assertFalse(os.path.isdir(dest) and os.listdir(dest))


class TestAOPairJsonsDegrade(unittest.TestCase):
    def test_empty_regions_exports_without_rt_and_records_degradation(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, stems=("lobby", "hall"))
            s = settings_for(d)
            m, _, _ = synth_matches(n=40, W=SIZE[0], H=SIZE[1], seed=2)
            write_cache(s.cache_root(), HOUSE, "lobby", "hall", m)
            with quiet():
                pipeline.estimate_poses(settings=s, pairs=[(HOUSE, "lobby", "hall")])
                pipeline.prepare_house(settings=s, houses=[HOUSE])
            s_empty = Settings(
                datasets=s.datasets_root(), matrices=s.matrices_root(),
                output=os.path.join(d, "empty_out"), cache=s.cache_root(),
                house_data=s.house_data_dir(), public_houses=os.path.join(d, "pub_empty"),
            )
            os.makedirs(s_empty.regions_root(), exist_ok=True)
            with quiet():
                r = pipeline.export_public_houses(settings=s_empty, houses=[HOUSE],
                                                  with_regions=False)
            self.assertTrue(any(x.input_name == "pair_jsons" for x in r.degradations))
            pub = json.load(open(r.outputs["public_json"], encoding="utf-8"))
            self.assertTrue(pub["pairs"][0]["has_geometry"])
            self.assertNotIn("R", pub["pairs"][0])
            self.assertNotIn("t", pub["pairs"][0])

    def test_missing_one_of_two_pair_jsons_aborts(self):
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            regions = os.path.join(d, "regions")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(regions, exist_ok=True)
            write_json(os.path.join(in_dir, "H.json"), {
                "house": "H", "n_images": 3, "width": 64, "height": 32,
                "images": ["a.jpg", "b.jpg", "c.jpg"],
                "pairs": [
                    {"i": 0, "j": 1, "az_ij": 10.0, "az_ji": 190.0,
                     "az_source": "geo", "geo_posed": 1},
                    {"i": 0, "j": 2, "az_ij": 20.0, "az_ji": 200.0,
                     "az_source": "geo", "geo_posed": 1},
                ],
            })
            write_json(os.path.join(regions, "H_a_b.json"), {
                "pose": {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [1.0, 0.0, 0.0]},
            })
            with self.assertRaises(RomaGeoError) as ctx:
                with quiet():
                    export_houses(in_dir, out_dir, regions_dir=regions, with_regions=False)
            self.assertIn("部分缺漏", str(ctx.exception))
            self.assertIn("a/c", str(ctx.exception))
            self.assertFalse(os.path.isfile(os.path.join(out_dir, "H.json")))


class TestAPAssertionsDistinguish(unittest.TestCase):
    def test_unrelated_doc_fails_field_check(self):
        import roma_geo.export_public as ep
        self.assertTrue(public_doc_field_errors("i j R t schema_version n_images"))
        self.assertEqual(public_doc_field_errors(ep.__doc__), [])
