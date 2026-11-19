"""追加8：交付檔安全、配對清單覆寫、安裝路徑、失敗契約、測試不依賴開發機資料集。"""
import csv
import json
import os
import unittest

from _support import env_override, quiet, synth_matches, temp_dir, write_json

from roma_geo.config import PROJECT_ROOT, Settings
from roma_geo.errors import RomaGeoError
from roma_geo import pipeline
from roma_geo.export_public import export_houses, house_to_public
from roma_geo.ingest_connections import convert_connections
from roma_geo.names import is_path_component, require_path_component

from test_pipeline import HOUSE, SIZE, make_image_house, settings_for, write_cache
import check_deliverables as cd


def _internal_house(name="H"):
    return {
        "house": name, "n_images": 2, "width": 64, "height": 32,
        "images": ["a.jpg", "b.jpg"],
        "az_convention": "az(x)=((x/W)*360-180)%360 ; 0deg = image center",
        "pairs": [{"i": 0, "j": 1, "az_ij": 10.0, "az_ji": 190.0,
                   "az_source": "coarse", "geo_posed": 0}],
    }


def _public_house(name="H"):
    d = _internal_house(name)
    d["schema_version"] = "house-v2"
    d["has_connectivity_matrix"] = False
    pair = d["pairs"][0]
    pair.pop("geo_posed", None)
    pair["has_geometry"] = False
    pair["semantic_corrected"] = False
    return d


def _empty_datasets(root):
    path = os.path.join(root, "datasets")
    os.makedirs(path, exist_ok=True)
    return path


class TestAF1PathComponentGuard(unittest.TestCase):
    def test_legal_names_still_accepted(self):
        for name in ("DollhouseTask_58553", "DollhouseTask_58582_NoOutdoor",
                     "Office_A", "RestrictedFake"):
            self.assertTrue(is_path_component(name), name)
            self.assertEqual(require_path_component(name), name)

    def test_house_with_parent_ref_aborts_and_writes_nothing(self):
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            parent = d
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)
            ds = _empty_datasets(d)
            before_parent = set(os.listdir(parent))
            write_json(os.path.join(in_dir, "bad.json"), _internal_house("../escaped"))
            with self.assertRaises(RomaGeoError) as ctx:
                with quiet():
                    export_houses(in_dir, out_dir, with_regions=False,
                                  datasets_dir=ds)
            self.assertIn("../escaped", str(ctx.exception))
            self.assertEqual(os.listdir(out_dir), [])
            self.assertFalse(os.path.isfile(os.path.join(parent, "escaped.json")))
            self.assertEqual(set(os.listdir(parent)), before_parent)

    def test_upstream_house_id_with_parent_ref_writes_nothing(self):
        with temp_dir() as d:
            src = os.path.join(d, "src")
            dest = os.path.join(d, "mat")
            os.makedirs(src, exist_ok=True)
            write_json(os.path.join(src, "x.json"), {
                "house_id": "../escaped",
                "pairs": [{"pano_a": "a.jpg", "pano_b": "b.jpg", "probability": 0.9}],
            })
            with self.assertRaises(RomaGeoError):
                with quiet():
                    convert_connections(src, dest)
            self.assertFalse(os.path.isdir(dest) and os.listdir(dest))
            self.assertFalse(os.path.isfile(os.path.join(d, "escaped.npy")))


class TestAF2AtomicWriteAndParseGuard(unittest.TestCase):
    def test_interrupt_leaves_no_partial_public_file(self):
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            os.makedirs(in_dir, exist_ok=True)
            write_json(os.path.join(in_dir, "H.json"), _internal_house("H"))
            import roma_geo.export_public as ep

            def boom(src, dst):
                raise RuntimeError("interrupt")

            orig = os.replace
            try:
                os.replace = boom
                with self.assertRaises(RuntimeError):
                    with quiet():
                        export_houses(in_dir, out_dir, with_regions=False,
                                      datasets_dir=_empty_datasets(d))
            finally:
                os.replace = orig
            self.assertFalse(os.path.isfile(os.path.join(out_dir, "H.json")))
            leftover = [n for n in os.listdir(out_dir) if n.endswith(".tmp")]
            self.assertEqual(leftover, [])

    def test_atomic_write_keeps_compact_json_and_key_order(self):
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            os.makedirs(in_dir, exist_ok=True)
            src = _internal_house("H")
            write_json(os.path.join(in_dir, "H.json"), src)
            with quiet():
                export_houses(in_dir, out_dir, with_regions=False,
                              datasets_dir=_empty_datasets(d))
            text = open(os.path.join(out_dir, "H.json"), encoding="utf-8").read()
            pub = house_to_public(src)
            self.assertEqual(text, json.dumps(pub, ensure_ascii=False))

    def test_unreadable_json_is_a_problem_and_other_checks_continue(self):
        with temp_dir() as d:
            pub = os.path.join(d, "pub")
            houses = os.path.join(d, "houses")
            os.makedirs(pub, exist_ok=True)
            os.makedirs(houses, exist_ok=True)
            write_json(os.path.join(pub, "Good.json"), _public_house("Good"))
            open(os.path.join(pub, "Bad.json"), "w", encoding="utf-8").write('{"house":')
            write_json(os.path.join(houses, "Good.json"), _internal_house("Good"))
            open(os.path.join(houses, "Broken.json"), "w", encoding="utf-8").write("{")
            ds = _empty_datasets(d)
            r_pub = cd.check_public_houses(pub, datasets_dir=ds)
            r_house = cd.check_house_json(houses)
            self.assertEqual(r_pub.status, cd.FAIL)
            self.assertTrue(any("Bad" in p and "無法解析" in p for p in r_pub.problems),
                            r_pub.problems)
            self.assertEqual(r_house.status, cd.FAIL)
            self.assertTrue(any("Broken" in p and "無法解析" in p for p in r_house.problems),
                            r_house.problems)
            results = cd.run_checks(
                limit=1,
                regions_dir=os.path.join(d, "no_regions"),
                matrices_dir=os.path.join(d, "no_mat"),
                house_dir=houses,
                public_dir=pub,
                datasets_dir=ds,
            )
            self.assertEqual(len(results), 5)
            names = [r.name for r in results]
            self.assertIn("house JSON", names)
            self.assertIn("對外交付子集", names)


class TestAF3DuplicateHouseId(unittest.TestCase):
    def test_duplicate_house_field_aborts_and_names_both_files(self):
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            os.makedirs(in_dir, exist_ok=True)
            write_json(os.path.join(in_dir, "A.json"), _internal_house("Same"))
            write_json(os.path.join(in_dir, "B.json"), _internal_house("Same"))
            with self.assertRaises(RomaGeoError) as ctx:
                with quiet():
                    export_houses(in_dir, out_dir, with_regions=False,
                                  datasets_dir=_empty_datasets(d))
            msg = str(ctx.exception)
            self.assertIn("A.json", msg)
            self.assertIn("B.json", msg)
            self.assertFalse(os.path.isdir(out_dir) and os.listdir(out_dir))

    def test_houses_filter_uses_house_field_not_filename(self):
        with temp_dir() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            os.makedirs(in_dir, exist_ok=True)
            write_json(os.path.join(in_dir, "file_a.json"), _internal_house("Wanted"))
            write_json(os.path.join(in_dir, "file_b.json"), _internal_house("Other"))
            with quiet():
                written = export_houses(
                    in_dir, out_dir, with_regions=False,
                    datasets_dir=_empty_datasets(d), houses=["Wanted"],
                )
            self.assertEqual(len(written), 1)
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "Wanted.json")))
            self.assertFalse(os.path.isfile(os.path.join(out_dir, "Other.json")))
            self.assertFalse(os.path.isfile(os.path.join(out_dir, "file_a.json")))

    def test_upstream_duplicate_house_id_writes_nothing(self):
        with temp_dir() as d:
            src = os.path.join(d, "src")
            dest = os.path.join(d, "mat")
            os.makedirs(src, exist_ok=True)
            write_json(os.path.join(src, "A.json"), {
                "house_id": "Same",
                "pairs": [{"pano_a": "a.jpg", "pano_b": "b.jpg", "probability": 0.9}],
            })
            write_json(os.path.join(src, "B.json"), {
                "house_id": "Same",
                "pairs": [{"pano_a": "c.jpg", "pano_b": "d.jpg", "probability": 0.1}],
            })
            with self.assertRaises(RomaGeoError) as ctx:
                with quiet():
                    convert_connections(src, dest)
            msg = str(ctx.exception)
            self.assertIn("A.json", msg)
            self.assertIn("B.json", msg)
            self.assertIn("Same", msg)
            self.assertFalse(os.path.isdir(dest) and os.listdir(dest))


def public_doc_field_errors(doc):
    """模組說明的對外保留段缺少哪些契約欄位。無關文字應回傳非空。"""
    import re
    from roma_geo.export_public import HOUSE_FIELDS, PAIR_FIELDS
    if not doc or "[對外保留]" not in doc or "[刻意不對外]" not in doc:
        return ["missing sections"]
    reserved = doc.split("[對外保留]", 1)[1].split("[刻意不對外]", 1)[0]
    names, seen = [], set()
    for name in list(PAIR_FIELDS.values()) + list(HOUSE_FIELDS) + ["schema_version"]:
        if name not in seen:
            seen.add(name)
            names.append(name)
    missing = []
    for name in names:
        if not re.search(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(name),
                         reserved):
            missing.append(name)
    if "本來就不在 house JSON" in doc:
        missing.append("stale R/t sentence")
    return missing


class TestAF4ModuleDocMatchesFields(unittest.TestCase):
    def test_docstring_lists_public_fields(self):
        import roma_geo.export_public as ep
        self.assertEqual(public_doc_field_errors(ep.__doc__), [])

    def test_unrelated_text_is_rejected(self):
        fake = "The R and t of i and j appear in ordinary English prose."
        self.assertTrue(public_doc_field_errors(fake))


class TestAG1PairsCsvIsThisCallOnly(unittest.TestCase):
    def test_third_call_only_contains_third_house(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, house="H1", stems=("a", "b"))
            make_image_house(ds, house="H2", stems=("c", "d", "e"))
            make_image_house(ds, house="H3", stems=("f", "g"))
            s = settings_for(d)
            with quiet():
                r1 = pipeline.select_pairs(settings=s, houses=["H1"])
                r2 = pipeline.select_pairs(settings=s, houses=["H2"])
                r3 = pipeline.select_pairs(settings=s, houses=["H3"])
            path = r3.outputs["pairs_csv"]
            self.assertEqual(path, r1.outputs["pairs_csv"])
            rows = list(csv.DictReader(open(path, encoding="utf-8")))
            self.assertEqual({r["house"] for r in rows}, {"H3"})
            self.assertEqual(len(rows), 1)
            self.assertEqual(r3.outputs["n_pairs"], 1)
            self.assertEqual(r2.outputs["n_pairs"], 3)

    def test_missing_prior_cache_does_not_affect_new_call(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, house="H1", stems=("a", "b"))
            make_image_house(ds, house="H2", stems=("c", "d"))
            s = settings_for(d)
            m, _, _ = synth_matches(n=40, W=SIZE[0], H=SIZE[1], seed=3)
            write_cache(s.cache_root(), "H2", "c", "d", m)
            with quiet():
                pipeline.select_pairs(settings=s, houses=["H1"])
                r2 = pipeline.select_pairs(settings=s, houses=["H2"])
                pipeline.estimate_poses(settings=s, pairs_csv=r2.outputs["pairs_csv"])
            rows = list(csv.DictReader(open(
                os.path.join(s.regions_root(), "_summary.csv"), encoding="utf-8")))
            self.assertEqual({r["house"] for r in rows}, {"H2"})

    def test_summary_still_accumulates_after_pairs_overwrite(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, house="H1", stems=("a", "b"))
            make_image_house(ds, house="H2", stems=("c", "d"))
            s = settings_for(d)
            m, _, _ = synth_matches(n=40, W=SIZE[0], H=SIZE[1], seed=4)
            write_cache(s.cache_root(), "H1", "a", "b", m)
            write_cache(s.cache_root(), "H2", "c", "d", m)
            with quiet():
                r1 = pipeline.select_pairs(settings=s, houses=["H1"])
                pipeline.estimate_poses(settings=s, pairs_csv=r1.outputs["pairs_csv"])
                r2 = pipeline.select_pairs(settings=s, houses=["H2"])
                pipeline.estimate_poses(settings=s, pairs_csv=r2.outputs["pairs_csv"])
            pairs = list(csv.DictReader(open(r2.outputs["pairs_csv"], encoding="utf-8")))
            summary = list(csv.DictReader(open(
                os.path.join(s.regions_root(), "_summary.csv"), encoding="utf-8")))
            self.assertEqual({r["house"] for r in pairs}, {"H2"})
            self.assertEqual({r["house"] for r in summary}, {"H1", "H2"})


class TestAH1NonEditableInstallError(unittest.TestCase):
    def test_missing_project_tree_is_roma_geo_error(self):
        with self.assertRaises(RomaGeoError) as ctx:
            pipeline._import_project("no_such_tree_dir", "no_such_module_xyz")
        self.assertNotIsInstance(ctx.exception, ModuleNotFoundError)
        msg = str(ctx.exception)
        self.assertIn("編輯模式", msg)
        self.assertIn("no_such_module_xyz", msg)


class TestAH2WritePathsFollowResolvedOutput(unittest.TestCase):
    def test_output_env_moves_write_targets(self):
        with temp_dir() as d, env_override(
                ROMA_GEO_OUTPUT=d,
                ROMA_GEO_HOUSE_DATA=None,
                ROMA_GEO_SEMANTIC_OUT=None,
                ROMA_GEO_PUBLIC_HOUSES=None,
                ROMA_GEO_DATASETS=None,
                ROMA_GEO_MATRICES=None,
                ROMA_GEO_CACHE=None,
                ROMA_GEO_CONNECTIONS=None):
            s = Settings()
            out = os.path.normcase(os.path.abspath(s.output_root()))
            writes = [s.regions_root(), s.public_dir(),
                      s.house_data_dir(), s.semantic_out_dir()]
            for p in writes:
                self.assertTrue(
                    os.path.normcase(os.path.abspath(p)).startswith(out + os.sep),
                    p)
            self.assertFalse(
                os.path.normcase(s.matrices_root()).startswith(out + os.sep))
            self.assertFalse(
                os.path.normcase(s.datasets_root()).startswith(out + os.sep))

    def test_explicit_house_data_is_unchanged(self):
        with temp_dir() as d:
            hd = os.path.join(d, "keep_hd")
            out = os.path.join(d, "out")
            s = Settings(output=out, house_data=hd)
            with env_override(ROMA_GEO_OUTPUT=os.path.join(d, "envout"),
                              ROMA_GEO_HOUSE_DATA=os.path.join(d, "envhd")):
                self.assertEqual(os.path.normcase(s.house_data_dir()),
                                 os.path.normcase(os.path.abspath(hd)))


class TestAH3DescribeCoversAllEnvAndPaths(unittest.TestCase):
    def test_describe_mentions_every_env_constant_and_path(self):
        import roma_geo.config as cfg
        text = cfg.describe()
        env_names = [
            getattr(cfg, n) for n in dir(cfg)
            if n.startswith("ENV_") and isinstance(getattr(cfg, n), str)
        ]
        self.assertEqual(len(env_names), len(set(env_names)))
        for name in env_names:
            self.assertIn(name, text, name)
        for key in ("datasets_root", "semantic_root", "matrices_root",
                    "pair_threshold", "output_root", "regions_root", "cache_root",
                    "house_data_dir", "semantic_out_dir", "public_dir",
                    "connections_dir"):
            self.assertIn(key, text)


class TestAIFailureContract(unittest.TestCase):
    def test_existing_batch_pairs_are_left_intact(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, stems=("lobby", "hall"))
            s = settings_for(d)
            path = os.path.join(s.regions_root(), pipeline.BATCH_PAIRS_FILENAME)
            os.makedirs(s.regions_root(), exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write("house,img_a,img_b,score\nH,a,b,N/A\n")
            before = open(path, encoding="utf-8").read()
            with quiet():
                r = pipeline.select_pairs(settings=s, houses=[HOUSE])
            self.assertEqual(open(path, encoding="utf-8").read(), before)
            self.assertTrue(r.outputs["pairs_csv"].endswith(pipeline.STEP_PAIRS_FILENAME))
            rows = list(csv.DictReader(open(r.outputs["pairs_csv"], encoding="utf-8")))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["house"], HOUSE)

    def test_missing_pairs_field_is_roma_geo_error(self):
        with temp_dir() as d:
            src = os.path.join(d, "src")
            dest = os.path.join(d, "mat")
            os.makedirs(src, exist_ok=True)
            write_json(os.path.join(src, "H.json"), {"house_id": "H", "notes": "x"})
            with self.assertRaises(RomaGeoError) as ctx:
                with quiet():
                    convert_connections(src, dest)
            self.assertNotIsInstance(ctx.exception, KeyError)
            self.assertIn("pairs", str(ctx.exception))
            self.assertFalse(os.path.isdir(dest) and os.listdir(dest))

    def test_empty_summary_is_overwritten(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, stems=("lobby", "hall"))
            s = settings_for(d)
            os.makedirs(s.regions_root(), exist_ok=True)
            summary = os.path.join(s.regions_root(), "_summary.csv")
            open(summary, "wb").close()
            self.assertEqual(os.path.getsize(summary), 0)
            m, _, _ = synth_matches(n=40, W=SIZE[0], H=SIZE[1], seed=9)
            write_cache(s.cache_root(), HOUSE, "lobby", "hall", m)
            with quiet():
                pipeline.estimate_poses(settings=s, pairs=[(HOUSE, "lobby", "hall")])
            rows = list(csv.DictReader(open(summary, encoding="utf-8")))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["house"], HOUSE)

    def test_empty_pairs_csv_is_overwritten(self):
        with temp_dir() as d:
            ds = os.path.join(d, "datasets")
            make_image_house(ds, stems=("lobby", "hall"))
            s = settings_for(d)
            path = os.path.join(s.regions_root(), pipeline.STEP_PAIRS_FILENAME)
            os.makedirs(s.regions_root(), exist_ok=True)
            open(path, "wb").close()
            with quiet():
                r = pipeline.select_pairs(settings=s, houses=[HOUSE])
            rows = list(csv.DictReader(open(r.outputs["pairs_csv"], encoding="utf-8")))
            self.assertEqual(len(rows), 1)


class TestAJTestsSupplyDatasetsDir(unittest.TestCase):
    def test_check_public_houses_with_missing_env_datasets_still_passes(self):
        with temp_dir() as d, env_override(ROMA_GEO_DATASETS=os.path.join(d, "no_ds")):
            write_json(os.path.join(d, "H.json"), _public_house("H"))
            ds = _empty_datasets(d)
            r = cd.check_public_houses(d, datasets_dir=ds)
            self.assertEqual(r.status, cd.OK, r.problems)


#: 呼叫端行程的探針：跑一個流程步驟，回報搜尋路徑清單、哪些泛用檔名被當成
#: 頂層模組載入、以及使用端自己的 config 是否還在。
#: 路徑比的是完整清單不是集合：目錄已在 path 上再被插到最前，集合差集看不出來。
_BOUNDARY_PROBE = r'''
import json, os, sys

GENERIC = ("config", "names", "errors", "pipeline", "export_public",
           "house_dirs", "gt_hotspot", "select_pairs", "prepare_house_data")


def search_path():
    return [os.path.normcase(os.path.abspath(p)) for p in sys.path if p]


from roma_geo.config import Settings
from roma_geo import pipeline

src = os.path.normcase(os.path.abspath(%(src)r))
# 先讓 src 出現在清單中段，模擬「目錄已在 path 上」；若步驟把它插到最前，清單會變。
if src in search_path():
    sys.path = [p for p in sys.path
                if os.path.normcase(os.path.abspath(p or ".")) != src]
sys.path.append(%(src)r)
before = search_path()
settings = Settings(**%(kwargs)r)
%(call)s
after = search_path()
loaded = dict((n, getattr(sys.modules[n], "__file__", "") or "")
              for n in GENERIC if n in sys.modules)
sys.modules.pop("config", None)
import config
try:
    import house_dirs
    house_dirs_file = getattr(house_dirs, "__file__", "") or ""
except ImportError:
    house_dirs_file = ""
print("PROBE " + json.dumps({
    "before": before,
    "after": after,
    "loaded": loaded,
    "marker": getattr(config, "marker", "STOLEN"),
    "config_file": getattr(config, "__file__", "") or "",
    "house_dirs_file": house_dirs_file,
}))
'''


class TestATImportBoundary(unittest.TestCase):
    """流程步驟以檔案路徑載入專案腳本：呼叫端的搜尋路徑與泛用模組名都不受影響。"""

    def _user_workdir(self, d):
        """使用端工作目錄：自己的 config.py 與 pipeline.py，加上一組指向本目錄的設定。"""
        open(os.path.join(d, "config.py"), "w", encoding="utf-8").write(
            "marker = 'user-config'\n")
        open(os.path.join(d, "pipeline.py"), "w", encoding="utf-8").write(
            "marker = 'user-pipeline'\n")
        return dict(
            datasets=os.path.join(d, "datasets"), matrices=os.path.join(d, "matrices"),
            output=os.path.join(d, "out"), cache=os.path.join(d, "cache"),
            house_data=os.path.join(d, "house_data"), semantic=os.path.join(d, "semantic"),
            public_houses=os.path.join(d, "public"), connections=os.path.join(d, "connections"),
        )

    def _probe(self, d, kwargs, call):
        import subprocess
        import sys
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        for name in [k for k in env if k.startswith("ROMA_GEO_")]:
            env.pop(name)
        r = subprocess.run(
            [sys.executable, "-c", _BOUNDARY_PROBE % {
                "kwargs": kwargs, "call": call,
                "src": os.path.join(PROJECT_ROOT, "src"),
            }],
            cwd=d, env=env, capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith("PROBE ")]
        self.assertTrue(lines, r.stdout + r.stderr)
        return json.loads(lines[-1][len("PROBE "):])

    def _assert_inside_project(self, path, message):
        root = os.path.normcase(os.path.realpath(PROJECT_ROOT))
        actual = os.path.normcase(os.path.realpath(path or "."))
        self.assertFalse(actual == root or actual.startswith(root + os.sep), message)

    def _assert_boundary(self, info, d):
        self.assertEqual(info["after"], info["before"], info)
        for name, path in info["loaded"].items():
            self._assert_inside_project(path, "%s 變成本專案的 %s" % (name, path))
        self.assertEqual(info["marker"], "user-config")
        self.assertEqual(os.path.normcase(os.path.realpath(info["config_file"])),
                         os.path.normcase(os.path.realpath(os.path.join(d, "config.py"))))

    def test_select_pairs_keeps_caller_search_path_and_modules(self):
        with temp_dir() as d:
            kwargs = self._user_workdir(d)
            mdir = Settings(**kwargs).matrices_root()
            os.makedirs(mdir, exist_ok=True)
            import numpy as np
            np.save(os.path.join(mdir, HOUSE + ".npy"), np.zeros((2, 2, 3), dtype=np.float32))
            write_json(os.path.join(mdir, HOUSE + "_images.json"), ["lobby.jpg", "hall.jpg"])
            info = self._probe(
                d, kwargs,
                "pipeline.select_pairs(settings=settings, houses=[%r])" % HOUSE)
            self._assert_boundary(info, d)

    def test_prepare_house_keeps_caller_search_path_and_modules(self):
        with temp_dir() as d:
            kwargs = self._user_workdir(d)
            make_image_house(os.path.join(d, "datasets"), stems=("lobby", "hall"))
            info = self._probe(
                d, kwargs,
                "pipeline.prepare_house(settings=settings, houses=[%r])" % HOUSE)
            self._assert_boundary(info, d)
            if info["house_dirs_file"]:
                self._assert_inside_project(
                    info["house_dirs_file"],
                    "import house_dirs 拿到本專案的 %s" % info["house_dirs_file"])

    def test_house_dirs_import_does_not_insert_src_at_front(self):
        """標註工具把 stitch_tool 放進搜尋路徑時，house_dirs 自己不再插入 src。

        套件已可匯入（先 import roma_geo，再把 src 移出搜尋路徑），此時 house_dirs
        不需要也不應該再動搜尋路徑。
        """
        import subprocess
        import sys
        import textwrap
        tool = os.path.join(PROJECT_ROOT, "stitch_tool")
        src = os.path.join(PROJECT_ROOT, "src")
        code = textwrap.dedent("""
            import os, sys
            tool = %r
            src = %r
            import roma_geo
            sys.path = [p for p in sys.path
                        if os.path.normcase(os.path.abspath(p or "."))
                        != os.path.normcase(os.path.abspath(src))]
            sys.path.insert(0, tool)
            before = list(sys.path)
            import house_dirs
            print(os.path.normcase(os.path.abspath(sys.path[0])))
            print("SAME" if sys.path == before else "CHANGED")
            print(house_dirs.resolve_task_dir.__name__)
        """ % (tool, src))
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        with temp_dir() as d:
            r = subprocess.run([sys.executable, "-c", code], cwd=d, env=env,
                               capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        out = r.stdout.splitlines()
        self.assertEqual(out[0], os.path.normcase(os.path.abspath(tool)))
        self.assertEqual(out[1], "SAME")
        self.assertEqual(out[2], "resolve_task_dir")


class TestDeferredItemsUntouched(unittest.TestCase):
    def test_select_pairs_cli_refuses_existing_without_force(self):
        import subprocess
        import sys
        with temp_dir() as d:
            mdir = os.path.join(d, "mat")
            os.makedirs(mdir, exist_ok=True)
            import numpy as np
            np.save(os.path.join(mdir, "H.npy"), np.zeros((2, 2, 3), dtype=np.float32))
            write_json(os.path.join(mdir, "H_images.json"), ["a.jpg", "b.jpg"])
            out = os.path.join(d, "_pairs.csv")
            body = "house,img_a,img_b,score\nH,a,b,0.1\n"
            with open(out, "w", encoding="utf-8", newline="") as f:
                f.write(body)
            r = subprocess.run(
                [sys.executable, os.path.join(PROJECT_ROOT, "environments", "select_pairs.py"),
                 "--matrices", mdir, "--out", out],
                capture_output=True, text=True,
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("已存在", r.stdout + r.stderr)
            self.assertEqual(open(out, encoding="utf-8").read(), body)
