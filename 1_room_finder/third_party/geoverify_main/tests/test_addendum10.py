"""追加10：相對位姿逐對判定、測試不讀未入庫目錄、搜尋路徑敘述。"""
import io
import json
import os
import subprocess
import sys
import unittest

import numpy as np

from _support import ROOT, env_override, quiet, synth_matches, temp_dir, write_json

from roma_geo.config import PROJECT_ROOT
from roma_geo.errors import RomaGeoError
from roma_geo.export_public import export_houses
from roma_geo import pipeline
from roma_geo.ingest_connections import convert_connections, build_matrix

from pose_regions.batch_pose_from_cache import run_pose_batch
from pose_regions.estimate_relative_pose import CACHE_SUFFIX
from test_pipeline import HOUSE, SIZE, make_image_house, settings_for
from test_addendum8 import _empty_datasets, _internal_house, _public_house, public_doc_field_errors
import check_deliverables as cd


def _geom_house(name="H", n=2):
    images = ["%s.jpg" % chr(ord("a") + i) for i in range(n)]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                "i": i, "j": j, "az_ij": 10.0 + i, "az_ji": 190.0 + j,
                "az_source": "geo", "geo_posed": 1,
            })
    return {
        "house": name, "n_images": n, "width": 64, "height": 32,
        "images": images, "pairs": pairs,
    }


def _pose_blob():
    return {"pose": {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [1.0, 0.0, 0.0]}}


def _geom_public(name="H"):
    d = _public_house(name)
    d["pairs"][0]["has_geometry"] = True
    d["pairs"][0]["az_source"] = "geo"
    d["pairs"][0].pop("R", None)
    d["pairs"][0].pop("t", None)
    return d


class TestAQPerPairPose(unittest.TestCase):
    def test_empty_regions_degrades_and_writes(self):
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            regions = os.path.join(d, "regions")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(regions, exist_ok=True)
            write_json(os.path.join(in_dir, "H.json"), _geom_house("H"))
            with quiet():
                result = export_houses(in_dir, out_dir, regions_dir=regions,
                                       with_regions=False,
                                       datasets_dir=_empty_datasets(d))
            self.assertEqual(result.pose_status, "absent")
            pub = json.load(open(os.path.join(out_dir, "H.json"), encoding="utf-8"))
            self.assertTrue(pub["pairs"][0]["has_geometry"])
            self.assertNotIn("R", pub["pairs"][0])
            r = cd.check_public_houses(out_dir, datasets_dir=_empty_datasets(d))
            self.assertEqual(r.status, cd.FAIL, r.detail)
            self.assertIn("整批缺席", r.detail)
            self.assertTrue(any("整批缺席" in p for p in r.problems), r.problems)
            buf = io.StringIO()
            self.assertEqual(cd.report([r], stream=buf), 1)
            self.assertNotIn("來源不存在", buf.getvalue())

    def test_unrelated_pair_json_same_as_empty(self):
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            regions = os.path.join(d, "regions")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(regions, exist_ok=True)
            write_json(os.path.join(in_dir, "H.json"), _geom_house("H"))
            write_json(os.path.join(regions, "Other_x_y.json"), _pose_blob())
            with quiet():
                result = export_houses(in_dir, out_dir, regions_dir=regions,
                                       with_regions=False,
                                       datasets_dir=_empty_datasets(d))
            self.assertEqual(result.pose_status, "absent")
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "H.json")))

    def test_partial_missing_names_house_and_pair(self):
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            regions = os.path.join(d, "regions")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(regions, exist_ok=True)
            write_json(os.path.join(in_dir, "H.json"), _geom_house("H", n=3))
            write_json(os.path.join(regions, "H_a_b.json"), _pose_blob())
            with self.assertRaises(RomaGeoError) as ctx:
                with quiet():
                    export_houses(in_dir, out_dir, regions_dir=regions,
                                  with_regions=False,
                                  datasets_dir=_empty_datasets(d))
            msg = str(ctx.exception)
            self.assertIn("部分缺漏", msg)
            self.assertIn("房子 H", msg)
            self.assertFalse(os.path.isfile(os.path.join(out_dir, "H.json")))

    def test_complete_has_rt_and_no_degrade(self):
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            regions = os.path.join(d, "regions")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(regions, exist_ok=True)
            write_json(os.path.join(in_dir, "H.json"), _geom_house("H"))
            write_json(os.path.join(regions, "H_a_b.json"), _pose_blob())
            with quiet():
                result = export_houses(in_dir, out_dir, regions_dir=regions,
                                       with_regions=False,
                                       datasets_dir=_empty_datasets(d))
            self.assertEqual(result.pose_status, "complete")
            pub = json.load(open(os.path.join(out_dir, "H.json"), encoding="utf-8"))
            self.assertIn("R", pub["pairs"][0])
            self.assertIn("t", pub["pairs"][0])
            r = cd.check_public_houses(out_dir, datasets_dir=_empty_datasets(d))
            self.assertEqual(r.status, cd.OK, r.problems)

    def test_complete_plus_unrelated_file_still_complete(self):
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            regions = os.path.join(d, "regions")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(regions, exist_ok=True)
            write_json(os.path.join(in_dir, "H.json"), _geom_house("H"))
            write_json(os.path.join(regions, "H_a_b.json"), _pose_blob())
            write_json(os.path.join(regions, "Other_x_y.json"), _pose_blob())
            with quiet():
                result = export_houses(in_dir, out_dir, regions_dir=regions,
                                       with_regions=False,
                                       datasets_dir=_empty_datasets(d))
            self.assertEqual(result.pose_status, "complete")

    def test_check_distinguishes_absent_and_partial(self):
        with temp_dir() as d:
            ds = _empty_datasets(d)
            absent = os.path.join(d, "absent")
            partial = os.path.join(d, "partial")
            os.makedirs(absent, exist_ok=True)
            os.makedirs(partial, exist_ok=True)
            write_json(os.path.join(absent, "H.json"), _geom_public("H"))
            mixed = _geom_public("H")
            mixed["pairs"][0]["R"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            mixed["pairs"][0]["t"] = [1.0, 0.0, 0.0]
            mixed["n_images"] = 3
            mixed["images"] = ["a.jpg", "b.jpg", "c.jpg"]
            mixed["pairs"].append({
                "i": 0, "j": 2, "az_ij": 20.0, "az_ji": 200.0,
                "az_source": "geo", "has_geometry": True,
                "semantic_corrected": False,
            })
            write_json(os.path.join(partial, "H.json"), mixed)
            r_abs = cd.check_public_houses(absent, datasets_dir=ds)
            r_part = cd.check_public_houses(partial, datasets_dir=ds)
            self.assertEqual(r_abs.status, cd.FAIL, r_abs.detail)
            self.assertIn("整批缺席", r_abs.detail)
            self.assertTrue(any("整批缺席" in p for p in r_abs.problems), r_abs.problems)
            self.assertFalse(any("部分缺漏" in p for p in r_abs.problems), r_abs.problems)
            self.assertEqual(r_part.status, cd.FAIL)
            self.assertTrue(any("部分缺漏" in p for p in r_part.problems), r_part.problems)
            self.assertFalse(any("整批缺席" in p for p in r_part.problems), r_part.problems)

    def test_absent_pose_does_not_drop_other_problems(self):
        with temp_dir() as d:
            pub = os.path.join(d, "pub")
            os.makedirs(pub, exist_ok=True)
            write_json(os.path.join(pub, "H.json"), _geom_public("H"))
            open(os.path.join(pub, "Trunc.json"), "w", encoding="utf-8").write('{"house":')
            r = cd.check_public_houses(pub, datasets_dir=_empty_datasets(d))
            self.assertEqual(r.status, cd.FAIL, r.detail)
            self.assertTrue(any("Trunc" in p and "無法解析" in p for p in r.problems),
                            r.problems)
            self.assertTrue(any("整批缺席" in p for p in r.problems), r.problems)

    def test_cli_exit_status_is_two_when_pose_absent(self):
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            regions = os.path.join(d, "regions")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(regions, exist_ok=True)
            write_json(os.path.join(in_dir, "H.json"), _geom_house("H"))
            r = subprocess.run(
                [sys.executable, os.path.join(ROOT, "export_public.py"),
                 "--in_dir", in_dir, "--out_dir", out_dir,
                 "--regions_dir", regions, "--no_regions"],
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertIn("整批缺席", r.stdout)
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "H.json")))

    def test_pair_json_without_pose_aborts(self):
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            regions = os.path.join(d, "regions")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(regions, exist_ok=True)
            write_json(os.path.join(in_dir, "H.json"), _geom_house("H"))
            write_json(os.path.join(regions, "H_a_b.json"),
                       {"pose": None, "flags": {"no_covisibility": True}})
            with self.assertRaises(RomaGeoError) as ctx:
                with quiet():
                    export_houses(in_dir, out_dir, regions_dir=regions,
                                  with_regions=False,
                                  datasets_dir=_empty_datasets(d))
            self.assertIn("a/b", str(ctx.exception))
            self.assertIn("沒有相對位姿", str(ctx.exception))
            self.assertIn("房子 H", str(ctx.exception))
            self.assertFalse(os.path.isfile(os.path.join(out_dir, "H.json")))


class TestBacklogCheapFixes(unittest.TestCase):
    def test_truncated_house_json_is_roma_geo_error(self):
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            os.makedirs(in_dir, exist_ok=True)
            open(os.path.join(in_dir, "H.json"), "w", encoding="utf-8").write('{"house":')
            with self.assertRaises(RomaGeoError) as ctx:
                with quiet():
                    export_houses(in_dir, out_dir, with_regions=False,
                                  datasets_dir=_empty_datasets(d))
            self.assertNotIsInstance(ctx.exception, json.JSONDecodeError)
            self.assertIn("H.json", str(ctx.exception))

    def test_truncated_pair_json_is_roma_geo_error(self):
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            regions = os.path.join(d, "regions")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(regions, exist_ok=True)
            write_json(os.path.join(in_dir, "H.json"), _geom_house("H"))
            open(os.path.join(regions, "H_a_b.json"), "w", encoding="utf-8").write("{")
            with self.assertRaises(RomaGeoError) as ctx:
                with quiet():
                    export_houses(in_dir, out_dir, regions_dir=regions,
                                  with_regions=False,
                                  datasets_dir=_empty_datasets(d))
            self.assertIn("H_a_b.json", str(ctx.exception))

    def test_windows_illegal_chars_rejected(self):
        from roma_geo.names import is_path_component, require_path_component
        for name in ("foo*bar", "foo?x", "foo|y", 'foo"z', "foo<w", "foo>v"):
            self.assertFalse(is_path_component(name), name)
            with self.assertRaises(RomaGeoError):
                require_path_component(name)
        self.assertTrue(is_path_component("DollhouseTask_58553"))

    def test_empty_pairs_is_roma_geo_error(self):
        with self.assertRaises(RomaGeoError) as ctx:
            build_matrix({"house_id": "H", "pairs": []})
        self.assertIn("兩張", str(ctx.exception))
        with temp_dir() as d:
            src = os.path.join(d, "src")
            dest = os.path.join(d, "mat")
            os.makedirs(src, exist_ok=True)
            write_json(os.path.join(src, "H.json"), {"house_id": "H", "pairs": []})
            with self.assertRaises(RomaGeoError):
                with quiet():
                    convert_connections(src, dest)
            self.assertFalse(os.path.isdir(dest) and os.listdir(dest))

    def test_empty_house_does_not_block_sibling(self):
        with temp_dir() as d:
            src = os.path.join(d, "src")
            dest = os.path.join(d, "mat")
            os.makedirs(src, exist_ok=True)
            write_json(os.path.join(src, "Good.json"), {
                "house_id": "Good",
                "pairs": [{"pano_a": "a.jpg", "pano_b": "b.jpg", "probability": 0.9}],
            })
            write_json(os.path.join(src, "Empty.json"), {"house_id": "Empty", "pairs": []})
            with quiet():
                houses = convert_connections(src, dest)
            self.assertEqual(houses, ["Good"])
            self.assertTrue(os.path.isfile(os.path.join(dest, "Good.npy")))
            self.assertFalse(os.path.isfile(os.path.join(dest, "Empty.npy")))

    def test_non_string_pano_name_is_roma_geo_error(self):
        from roma_geo.ingest_connections import build_matrix
        with self.assertRaises(RomaGeoError) as ctx:
            build_matrix({"pairs": [{"pano_a": 1, "pano_b": "b.jpg", "probability": 0.9}]})
        self.assertIn("字串", str(ctx.exception))

    def test_probability_none_is_roma_geo_error(self):
        from roma_geo.ingest_connections import build_matrix
        with self.assertRaises(RomaGeoError) as ctx:
            build_matrix({"pairs": [{"pano_a": "a.jpg", "pano_b": "b.jpg",
                                     "probability": None}]})
        self.assertNotIsInstance(ctx.exception, TypeError)
        self.assertIn("probability", str(ctx.exception))

    def test_probability_missing_is_roma_geo_error(self):
        from roma_geo.ingest_connections import build_matrix
        with self.assertRaises(RomaGeoError) as ctx:
            build_matrix({"pairs": [{"pano_a": "a.jpg", "pano_b": "b.jpg"}]})
        self.assertNotIsInstance(ctx.exception, KeyError)
        self.assertIn("probability", str(ctx.exception))

    def test_pano_parent_ref_is_roma_geo_error(self):
        from roma_geo.ingest_connections import build_matrix
        with self.assertRaises(RomaGeoError) as ctx:
            build_matrix({"pairs": [{"pano_a": "../x.jpg", "pano_b": "b.jpg",
                                     "probability": 0.9}]})
        self.assertIn("../x.jpg", str(ctx.exception))

    def test_pairs_csv_missing_column_is_roma_geo_error(self):
        with temp_dir() as d:
            s = settings_for(d)
            os.makedirs(s.regions_root(), exist_ok=True)
            path = os.path.join(s.regions_root(), pipeline.STEP_PAIRS_FILENAME)
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write("house,img_a\nH,a\n")
            with self.assertRaises(RomaGeoError) as ctx:
                with quiet():
                    pipeline.match_pairs(settings=s, pairs_csv=path,
                                         matcher=object())
            self.assertNotIsInstance(ctx.exception, KeyError)
            self.assertIn("img_b", str(ctx.exception))

    def test_pairs_csv_non_numeric_score_is_roma_geo_error(self):
        with temp_dir() as d:
            path = os.path.join(d, "pairs.csv")
            with self.assertRaises(RomaGeoError) as ctx:
                pipeline._write_pairs_csv(path, [("H", "a", "b", "N/A")])
            self.assertNotIsInstance(ctx.exception, ValueError)
            self.assertIn("N/A", str(ctx.exception))
            self.assertFalse(os.path.isfile(path))

    def test_readme_lists_pair_jsons_degradation(self):
        text = open(os.path.join(PROJECT_ROOT, "README.md"), encoding="utf-8").read()
        section = text.split("## 可選輸入", 1)[1].split("## 對外契約", 1)[0]
        self.assertIn("pair_jsons", section)

    def test_check_public_houses_when_dir_has_brackets(self):
        with temp_dir() as d:
            pub = os.path.join(d, "out[x]")
            os.makedirs(pub, exist_ok=True)
            write_json(os.path.join(pub, "H.json"), _public_house("H"))
            r = cd.check_public_houses(pub, datasets_dir=_empty_datasets(d))
            self.assertEqual(r.status, cd.OK, r.problems)

    def test_prepare_house_interrupt_leaves_no_partial_file(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, stems=("lobby", "hall"))
            s = settings_for(d)
            orig = os.replace

            def boom(src, dst):
                raise RuntimeError("interrupt")

            try:
                os.replace = boom
                with self.assertRaises(RuntimeError):
                    with quiet():
                        pipeline.prepare_house(settings=s, houses=[HOUSE])
            finally:
                os.replace = orig
            out_dir = s.house_data_dir()
            if os.path.isdir(out_dir):
                self.assertFalse(os.path.isfile(os.path.join(out_dir, HOUSE + ".json")))
                leftover = [n for n in os.listdir(out_dir) if n.endswith(".tmp")]
                self.assertEqual(leftover, [])

    def test_top_level_list_is_roma_geo_error(self):
        from roma_geo.ingest_connections import build_matrix
        with self.assertRaises(RomaGeoError) as ctx:
            build_matrix([{"pano_a": "a.jpg", "pano_b": "b.jpg", "probability": 0.9}])
        self.assertIn("物件", str(ctx.exception))

    def test_pairs_item_not_object_is_roma_geo_error(self):
        from roma_geo.ingest_connections import build_matrix
        with self.assertRaises(RomaGeoError) as ctx:
            build_matrix({"pairs": ["nope"]})
        self.assertIn("物件", str(ctx.exception))

    def test_restricted_reason_finds_marker_when_path_has_brackets(self):
        from roma_geo.export_public import RESTRICTED_MARKER, restricted_reason
        with temp_dir() as d:
            ds = os.path.join(d, "ds[prod]")
            house = "House[1]"
            nested = os.path.join(ds, house, house)
            os.makedirs(nested, exist_ok=True)
            open(os.path.join(nested, RESTRICTED_MARKER), "w", encoding="utf-8").write("no\n")
            self.assertEqual(restricted_reason(house, ds), "no")

    def test_export_finds_json_when_in_dir_has_brackets(self):
        with temp_dir() as d:
            in_dir = os.path.join(d, "in[x]")
            out_dir = os.path.join(d, "out")
            os.makedirs(in_dir, exist_ok=True)
            write_json(os.path.join(in_dir, "H.json"), _internal_house("H"))
            with quiet():
                written = export_houses(in_dir, out_dir, with_regions=False,
                                        datasets_dir=_empty_datasets(d))
            self.assertEqual(len(written), 1)

    def test_skip_detail_keeps_house_pair_counts(self):
        with temp_dir() as d:
            write_json(os.path.join(d, "H.json"), _geom_public("H"))
            r = cd.check_public_houses(d, datasets_dir=_empty_datasets(d))
            self.assertEqual(r.status, cd.FAIL, r.detail)
            self.assertIn("1 房", r.detail)
            self.assertIn("整批缺席", r.detail)

    def test_pose_batch_lists_cache_when_dir_has_brackets(self):
        house, na, nb = "Office_A", "lobby_1", "hall_2"
        with temp_dir() as d:
            ds = os.path.join(d, "datasets", house)
            os.makedirs(ds, exist_ok=True)
            from PIL import Image
            Image.new("RGB", SIZE, (1, 2, 3)).save(os.path.join(ds, na + ".jpg"), "JPEG")
            Image.new("RGB", SIZE, (1, 2, 3)).save(os.path.join(ds, nb + ".jpg"), "JPEG")
            cache = os.path.join(d, "cache[x]")
            out = os.path.join(d, "out")
            os.makedirs(cache, exist_ok=True)
            m, _, _ = synth_matches(n=40, W=SIZE[0], H=SIZE[1], seed=8)
            np.savez(os.path.join(cache, "%s_%s_%s%s" % (house, na, nb, CACHE_SUFFIX)),
                     matches=m)
            with quiet():
                n = run_pose_batch(cache, out, datasets_root=os.path.join(d, "datasets"),
                                   pairs=None, on_missing="skip")
            self.assertEqual(n, 1)
            self.assertTrue(os.path.isfile(os.path.join(out, "%s_%s_%s.json" % (house, na, nb))))

    def test_cli_defaults_follow_settings_output(self):
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            os.makedirs(in_dir, exist_ok=True)
            write_json(os.path.join(in_dir, "H.json"), _internal_house("H"))
            env = os.environ.copy()
            env["ROMA_GEO_HOUSE_DATA"] = in_dir
            env["ROMA_GEO_OUTPUT"] = os.path.join(d, "outroot")
            env["ROMA_GEO_DATASETS"] = _empty_datasets(d)
            env["ROMA_GEO_PUBLIC_HOUSES"] = os.path.join(d, "pub")
            r = subprocess.run(
                [sys.executable, os.path.join(ROOT, "export_public.py"), "--no_regions"],
                env=env, capture_output=True, text=True, cwd=d,
            )
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertTrue(os.path.isfile(os.path.join(d, "pub", "H.json")))

    def test_default_input_dir_reads_env_at_call_time(self):
        from roma_geo.ingest_connections import default_input_dir
        with env_override(ROMA_GEO_CONNECTIONS="/tmp/conn_now"):
            self.assertEqual(default_input_dir(), "/tmp/conn_now")


class TestAPAssertionsDistinguish(unittest.TestCase):
    def test_unrelated_doc_fails_field_check(self):
        import roma_geo.export_public as ep
        self.assertTrue(public_doc_field_errors("i j R t schema_version n_images"))
        self.assertEqual(public_doc_field_errors(ep.__doc__), [])


class TestASSearchPathDocs(unittest.TestCase):
    """README、API.md、pipeline 模組說明不得再承諾會改搜尋路徑。

    實際行為由 TestATImportBoundary 與 test_package_isolation 實測。
    """

    def _docs(self):
        readme = open(os.path.join(PROJECT_ROOT, "README.md"), encoding="utf-8").read()
        api = open(os.path.join(PROJECT_ROOT, "API.md"), encoding="utf-8").read()
        pipe = open(os.path.join(PROJECT_ROOT, "src", "roma_geo", "pipeline.py"),
                    encoding="utf-8").read().split('"""')[1]
        return readme, api, pipe

    def test_docs_promise_untouched_caller_search_path(self):
        for text in self._docs():
            self.assertIn("不修改", text)
            self.assertIn("專案根目錄", text)
            self.assertIn("src", text)
            self.assertIn("檔案路徑", text)

    def test_docs_drop_obsolete_insert_and_append_promises(self):
        for text in self._docs():
            self.assertNotIn("最前", text)
            self.assertNotIn("末端", text)

    def test_steps_load_project_scripts_by_file_path(self):
        before = list(sys.path)
        mod = pipeline._import_project("environments", "select_pairs")
        self.assertEqual(sys.path, before)
        self.assertTrue(mod.__name__.startswith("roma_geo._project."), mod.__name__)
        self.assertEqual(
            os.path.normcase(os.path.abspath(mod.__file__)),
            os.path.normcase(os.path.join(PROJECT_ROOT, "environments", "select_pairs.py")))
