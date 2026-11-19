"""步驟可獨立呼叫、可選輸入缺席、未見過的房子、設定由呼叫端指定。"""
import csv
import json
import os
import unittest

import numpy as np
from PIL import Image

from _support import ROOT, env_override, quiet, synth_matches, temp_dir  # noqa: F401

from roma_geo.config import Settings
from roma_geo.errors import RomaGeoError
from roma_geo.names import house_of
from pose_regions.estimate_relative_pose import CACHE_SUFFIX, parse_cache_name
from roma_geo import pipeline


HOUSE = "Office_A"
STEMS = ("lobby", "hall", "yard")
SIZE = (64, 32)

ENV_CLEAR = {
    "ROMA_GEO_DATASETS": None,
    "ROMA_GEO_MATRICES": None,
    "ROMA_GEO_OUTPUT": None,
    "ROMA_GEO_CACHE": None,
    "ROMA_GEO_PAIR_THRESHOLD": None,
    "ROMA_GEO_CONNECTIONS": None,
    "ROMA_GEO_HOUSE_DATA": None,
    "ROMA_GEO_SEMANTIC_OUT": None,
    "ROMA_GEO_PUBLIC_HOUSES": None,
}


def _save_jpg(path, size=SIZE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, (12, 24, 36)).save(path, "JPEG")
    return path


def make_image_house(root, house=HOUSE, stems=STEMS, size=SIZE, nested=False):
    """只含全景影像的房子：沒有連通矩陣、沒有 HOTSPOT、沒有語意圖。"""
    d = os.path.join(root, house, house) if nested else os.path.join(root, house)
    os.makedirs(d, exist_ok=True)
    paths = []
    for s in stems:
        paths.append(_save_jpg(os.path.join(d, s + ".jpg"), size))
    return d, paths


def settings_for(root, **extra):
    kw = dict(
        datasets=os.path.join(root, "datasets"),
        matrices=os.path.join(root, "matrices"),
        output=os.path.join(root, "out"),
        cache=os.path.join(root, "cache"),
        house_data=os.path.join(root, "house_data"),
        semantic=os.path.join(root, "semantic"),
        public_houses=os.path.join(root, "public"),
        connections=os.path.join(root, "connections"),
    )
    kw.update(extra)
    return Settings(**kw)


def write_cache(cache_dir, house, a, b, matches):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "%s_%s_%s%s" % (house, a, b, CACHE_SUFFIX))
    np.savez(path, matches=np.asarray(matches, dtype=float))
    return path


class StubMatcher:
    """不載入模型：寫入合成匹配快取，供 match_pairs 在無 GPU 時可測。"""

    def __init__(self, cache_dir, size=SIZE):
        self.cache_dir = cache_dir
        self.model_type = "roma_v2"
        self.face_size = 512
        self.size = size

    def _get_cache_info(self, path_a, path_b, mode):
        ha, na = house_of(path_a), os.path.splitext(os.path.basename(path_a))[0]
        hb, nb = house_of(path_b), os.path.splitext(os.path.basename(path_b))[0]
        prefix = ha if ha == hb else "%s_%s" % (ha, hb)
        name = "%s_%s_%s_%s_%s_std0.1_face%d.npz" % (
            prefix, na, nb, mode, self.model_type, self.face_size)
        return os.path.join(self.cache_dir, name), False

    def match_direct(self, path_a, path_b):
        w, h = self.size
        m, _, _ = synth_matches(n=80, W=w, H=h, seed=1)
        path, _ = self._get_cache_info(path_a, path_b, "direct")
        os.makedirs(self.cache_dir, exist_ok=True)
        np.savez(path, matches=m)
        return m.tolist(), {"w": w, "h": h}


class TestHouseOfUnseen(unittest.TestCase):
    def test_parent_directory_is_the_house_name(self):
        self.assertEqual(house_of(os.path.join("data", "Office_A", "lobby.jpg")), "Office_A")

    def test_nested_house_folder(self):
        p = os.path.join("datasets", "Office_A", "Office_A", "lobby.jpg")
        self.assertEqual(house_of(p), "Office_A")

    def test_dollhouse_marker_still_wins(self):
        p = os.path.join("datasets", "DollhouseTask_58553", "DollhouseTask_58553", "a.jpg")
        self.assertEqual(house_of(p), "DollhouseTask_58553")

    def test_bare_filename_is_unknown(self):
        self.assertEqual(house_of("a.jpg"), "unknown")


class TestParseCacheNameUnseen(unittest.TestCase):
    def test_plain_stems(self):
        name = "Office_A_lobby_hall%s" % CACHE_SUFFIX
        self.assertEqual(parse_cache_name(name), ("Office_A", "lobby", "hall"))

    def test_uuid_names_still_round_trip(self):
        house = "DollhouseTask_58472_NoOutdoor"
        a, b = "08ddb590-a6df-409d-850f-e45f59efa7a0", "08ddb590-ad9f-4647-8dab-ccfd0b14c8da"
        name = "%s_%s_%s%s" % (house, a, b, CACHE_SUFFIX)
        self.assertEqual(parse_cache_name(name), (house, a, b))


class TestSettingsExplicit(unittest.TestCase):
    def test_explicit_paths_ignore_environment(self):
        with temp_dir() as d:
            s = settings_for(d)
            with env_override(ROMA_GEO_DATASETS="/from/env", ROMA_GEO_OUTPUT="/env/out",
                              ROMA_GEO_CACHE="/env/cache", ROMA_GEO_MATRICES="/env/mat"):
                self.assertEqual(s.datasets_root(), os.path.abspath(os.path.join(d, "datasets")))
                self.assertEqual(s.output_root(), os.path.abspath(os.path.join(d, "out")))
                self.assertEqual(s.cache_root(), os.path.abspath(os.path.join(d, "cache")))
                self.assertEqual(s.matrices_root(), os.path.abspath(os.path.join(d, "matrices")))

    def test_relative_explicit_path_becomes_absolute(self):
        s = Settings(output="rel_out")
        self.assertTrue(os.path.isabs(s.output_root()))


class TestSelectPairsFromImages(unittest.TestCase):
    def test_all_pairs_when_matrix_absent(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds)
            s = settings_for(d)
            with quiet():
                result = pipeline.select_pairs(settings=s, houses=[HOUSE])
            self.assertTrue(result.degraded)
            names = [x.input_name for x in result.degradations]
            self.assertIn("upstream_connectivity", names)
            csv_path = result.outputs["pairs_csv"]
            rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
            self.assertEqual(len(rows), 3)
            keys = {frozenset((r["img_a"], r["img_b"])) for r in rows}
            self.assertEqual(keys, {frozenset(("hall", "yard")),
                                    frozenset(("lobby", "hall")),
                                    frozenset(("lobby", "yard"))})
            self.assertEqual(result.outputs["pair_source"], "images")

    def test_independent_of_cwd_and_env(self):
        with temp_dir() as d, temp_dir() as elsewhere:
            ds = os.path.join(d, "datasets")
            make_image_house(ds)
            s = settings_for(d)
            old = os.getcwd()
            try:
                os.chdir(elsewhere)
                with env_override(**ENV_CLEAR), quiet():
                    result = pipeline.select_pairs(settings=s, houses=[HOUSE])
            finally:
                os.chdir(old)
            self.assertEqual(len(list(csv.DictReader(open(result.outputs["pairs_csv"], encoding="utf-8")))), 3)


class TestEstimatePosesIndependent(unittest.TestCase):
    def test_runs_from_explicit_cache_without_prior_steps(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, stems=("lobby", "hall"))
            s = settings_for(d)
            m, _, _ = synth_matches(n=80, W=SIZE[0], H=SIZE[1], seed=2)
            write_cache(s.cache_root(), HOUSE, "lobby", "hall", m)
            with env_override(**ENV_CLEAR), quiet():
                result = pipeline.estimate_poses(
                    settings=s,
                    pairs=[(HOUSE, "lobby", "hall")],
                )
            out = os.path.join(s.regions_root(), "%s_lobby_hall.json" % HOUSE)
            self.assertTrue(os.path.isfile(out), result.message)
            data = json.load(open(out, encoding="utf-8"))
            self.assertEqual(data["house"], HOUSE)
            self.assertEqual(data["img_a"], "lobby")
            self.assertIn("schema_version", data)
            summary = os.path.join(s.regions_root(), "_summary.csv")
            self.assertTrue(os.path.isfile(summary))


class TestMatchPairsIndependent(unittest.TestCase):
    def test_stub_matcher_writes_cache(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, stems=("lobby", "hall"))
            s = settings_for(d)
            pairs_csv = os.path.join(d, "pairs.csv")
            os.makedirs(s.regions_root(), exist_ok=True)
            with open(pairs_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["house", "img_a", "img_b"])
                w.writerow([HOUSE, "lobby", "hall"])
            stub = StubMatcher(s.cache_root())
            with env_override(**ENV_CLEAR), quiet():
                result = pipeline.match_pairs(settings=s, pairs_csv=pairs_csv, matcher=stub)
            self.assertGreaterEqual(result.outputs["n_done"], 1)
            found = [fn for fn in os.listdir(s.cache_root()) if fn.endswith(".npz")]
            self.assertTrue(found)


class TestPrepareHouseWithoutMatrix(unittest.TestCase):
    def test_images_and_summary_suffice(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, stems=("lobby", "hall"))
            s = settings_for(d)
            m, _, _ = synth_matches(n=80, W=SIZE[0], H=SIZE[1], seed=3)
            write_cache(s.cache_root(), HOUSE, "lobby", "hall", m)
            with quiet():
                pipeline.estimate_poses(settings=s, pairs=[(HOUSE, "lobby", "hall")])
                result = pipeline.prepare_house(settings=s, houses=[HOUSE])
            self.assertTrue(result.degraded)
            names = [x.input_name for x in result.degradations]
            self.assertIn("upstream_connectivity", names)
            self.assertIn("ground_truth", names)
            data = json.load(open(result.outputs["house_json"], encoding="utf-8"))
            self.assertEqual(data["house"], HOUSE)
            self.assertEqual(data["n_images"], 2)
            self.assertFalse(data["has_gt"])
            self.assertFalse(data["has_connectivity_matrix"])
            pair = data["pairs"][0]
            self.assertIsNone(pair["gt_connected"])
            self.assertEqual(pair["az_source"], "geo")
            self.assertIsNone(pair.get("prob"))


class TestExportIndependent(unittest.TestCase):
    def test_exports_from_house_json_alone(self):
        with temp_dir() as d:
            s = settings_for(d)
            os.makedirs(s.house_data_dir(), exist_ok=True)
            internal = {
                "house": HOUSE, "n_images": 2, "width": 64, "height": 32,
                "images": ["lobby.jpg", "hall.jpg"],
                "az_convention": "az(x)=((x/W)*360-180)%360 ; 0deg = image center",
                "has_gt": False, "has_connectivity_matrix": False,
                "pairs": [{
                    "i": 0, "j": 1, "az_ij": 10.0, "az_ji": 190.0,
                    "az_source": "geo", "geo_posed": 0,
                    "geo_n_inlier": 40, "geo_inlier_rate": 0.8, "geo_conf_used": 0.65,
                    "gt_connected": None,
                }],
            }
            src = os.path.join(s.house_data_dir(), HOUSE + ".json")
            json.dump(internal, open(src, "w", encoding="utf-8"))
            with env_override(**ENV_CLEAR), quiet():
                result = pipeline.export_public_houses(settings=s, houses=[HOUSE], with_regions=False)
            pub = json.load(open(result.outputs["public_json"], encoding="utf-8"))
            self.assertEqual(pub["house"], HOUSE)
            self.assertNotIn("gt_connected", pub["pairs"][0])
            self.assertFalse(pub["pairs"][0]["has_geometry"])
            self.assertNotIn("prob", pub["pairs"][0])


class TestImagesOnlyPipeline(unittest.TestCase):
    def test_run_house_reports_every_absent_optional_input(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, stems=("lobby", "hall"))
            s = settings_for(d)
            stub = StubMatcher(s.cache_root())
            old = os.getcwd()
            try:
                os.chdir(d)
                with env_override(**ENV_CLEAR), quiet():
                    result = pipeline.run_house(HOUSE, settings=s, matcher=stub)
            finally:
                os.chdir(old)
            names = [x.input_name for x in result.degradations]
            self.assertIn("upstream_connectivity", names)
            self.assertIn("ground_truth", names)
            self.assertIn("semantic_maps", names)
            self.assertTrue(os.path.isfile(os.path.join(s.house_data_dir(), HOUSE + ".json")))
            self.assertTrue(os.path.isfile(os.path.join(s.public_dir(), HOUSE + ".json")))
            pub = json.load(open(os.path.join(s.public_dir(), HOUSE + ".json"), encoding="utf-8"))
            self.assertEqual(len(pub["pairs"]), 1)
            self.assertTrue(pub["pairs"][0]["has_geometry"])


class TestReadableErrors(unittest.TestCase):
    def test_missing_house_names_the_input(self):
        with temp_dir() as d:
            s = settings_for(d)
            os.makedirs(s.datasets_root(), exist_ok=True)
            with self.assertRaises(RomaGeoError) as ctx:
                pipeline.select_pairs(settings=s, houses=["NoSuchHouse"])
            msg = str(ctx.exception)
            self.assertIn("全景影像", msg)
            self.assertIn("NoSuchHouse", msg)

    def test_estimate_poses_names_missing_cache(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, stems=("lobby", "hall"))
            s = settings_for(d)
            with self.assertRaises(RomaGeoError) as ctx:
                pipeline.estimate_poses(settings=s, pairs=[(HOUSE, "lobby", "hall")])
            msg = str(ctx.exception)
            self.assertIn("匹配快取", msg)
            self.assertIn("lobby", msg)


class TestGtAbsentIsNotDisconnected(unittest.TestCase):
    def test_empty_gt_yields_none(self):
        import sys
        tool = os.path.join(ROOT, "stitch_tool")
        if tool not in sys.path:
            sys.path.insert(0, tool)
        import gt_hotspot
        fields = gt_hotspot.gt_pair_fields({}, "a.jpg", "b.jpg")
        self.assertIsNone(fields["gt_connected"])
        self.assertIsNone(fields["gt_az_ij"])
        self.assertIsNone(fields["gt_az_ji"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
