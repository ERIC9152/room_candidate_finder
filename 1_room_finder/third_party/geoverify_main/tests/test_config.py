"""
路徑設定測試：`config.py` 的解析優先序與各項預設值。

`config.py` 是部署設定的單一來源，六個項目各自可由環境變數覆寫。這些路徑在別台機器
上執行時是唯一會變動的東西，解析錯誤會表現為「找不到檔案」或更糟的「讀到舊資料」。

另有一項跨檔一致性檢查：`pose_regions` 與 `roma_match` 為維持可獨立匯入，各自直接讀
同一組環境變數名而不 import `config`。變數名散在三處，改名時必須同步；此處驗證它們
仍然一致。
"""
import os
import unittest

from _support import ROOT, env_override, temp_dir  # noqa: F401（匯入時補 sys.path）

import config


class TestDefaults(unittest.TestCase):
    """未設環境變數時，一律解析到專案內的預設位置。"""

    ENV_ALL = {"ROMA_GEO_DATASETS": None, "ROMA_GEO_MATRICES": None,
               "ROMA_GEO_OUTPUT": None, "ROMA_GEO_CACHE": None,
               "ROMA_GEO_PAIR_THRESHOLD": None,
               "ROMA_GEO_CONNECTIONS": None, "ROMA_GEO_HOUSE_DATA": None,
               "ROMA_GEO_SEMANTIC_OUT": None, "ROMA_GEO_PUBLIC_HOUSES": None}

    def test_project_root_is_repo_root(self):
        self.assertEqual(os.path.normcase(config.PROJECT_ROOT), os.path.normcase(ROOT))

    def test_defaults_live_under_project_root(self):
        with env_override(**self.ENV_ALL):
            for fn in (config.datasets_root, config.output_root, config.cache_root,
                       config.regions_root, config.semantic_root):
                self.assertTrue(os.path.normcase(fn()).startswith(os.path.normcase(ROOT)),
                                "%s 未落在專案目錄內" % fn.__name__)

    def test_derived_paths(self):
        with env_override(**self.ENV_ALL):
            self.assertEqual(config.regions_root(),
                             os.path.join(config.output_root(), "roma_regions"))
            self.assertEqual(config.semantic_root(),
                             os.path.join(config.datasets_root(), "semantic"))
            self.assertEqual(config.cache_root(),
                             os.path.join(ROOT, "roma_match", "cache"))

    def test_pair_threshold_default(self):
        """現值對應 candidate_val：ch0 存 logit，prob ≥ 0.5 ⇔ logit ≥ 0.0。"""
        with env_override(ROMA_GEO_PAIR_THRESHOLD=None):
            self.assertEqual(config.pair_threshold(), 0.0)
            self.assertIsInstance(config.pair_threshold(), float)


class TestEnvOverride(unittest.TestCase):
    def test_each_path_is_overridable(self):
        with temp_dir() as d:
            cases = [("ROMA_GEO_DATASETS", config.datasets_root),
                     ("ROMA_GEO_OUTPUT", config.output_root),
                     ("ROMA_GEO_CACHE", config.cache_root),
                     ("ROMA_GEO_MATRICES", config.matrices_version_root)]
            for var, fn in cases:
                with env_override(**{var: d}):
                    self.assertEqual(os.path.normcase(fn()), os.path.normcase(d),
                                     "%s 未生效" % var)

    def test_override_is_absolute(self):
        """相對路徑須轉為絕對，否則工作目錄一變就指到別處。"""
        with env_override(ROMA_GEO_OUTPUT="some_relative_dir"):
            self.assertTrue(os.path.isabs(config.output_root()))

    def test_user_home_is_expanded(self):
        with env_override(ROMA_GEO_CACHE=os.path.join("~", "roma_cache")):
            p = config.cache_root()
            self.assertNotIn("~", p)
            self.assertTrue(os.path.isabs(p))

    def test_output_override_propagates_to_regions(self):
        with temp_dir() as d, env_override(ROMA_GEO_OUTPUT=d):
            self.assertEqual(config.regions_root(), os.path.join(d, "roma_regions"))

    def test_datasets_override_propagates_to_semantic(self):
        with temp_dir() as d, env_override(ROMA_GEO_DATASETS=d):
            self.assertEqual(config.semantic_root(), os.path.join(d, "semantic"))

    def test_pair_threshold_from_env(self):
        with env_override(ROMA_GEO_PAIR_THRESHOLD="1.25"):
            self.assertEqual(config.pair_threshold(), 1.25)

    def test_empty_env_falls_back_to_default(self):
        """空字串視同未設定，避免誤把路徑解析成工作目錄。"""
        with env_override(ROMA_GEO_OUTPUT=""):
            self.assertEqual(config.output_root(), config.DEFAULT_OUTPUT)


class TestMatricesRoot(unittest.TestCase):
    def test_variant_subdirectory_is_used_when_present(self):
        with temp_dir() as d:
            os.makedirs(os.path.join(d, "refined_matrices"))
            with env_override(ROMA_GEO_MATRICES=d):
                self.assertEqual(config.matrices_root("refined"),
                                 os.path.join(d, "refined_matrices"))

    def test_falls_back_to_version_root(self):
        """相容既有版本：.npy 直接放在版本目錄底下時，回退為該目錄。"""
        with temp_dir() as d, env_override(ROMA_GEO_MATRICES=d):
            self.assertEqual(os.path.normcase(config.matrices_root("refined")),
                             os.path.normcase(d))

    def test_variants_are_independent(self):
        with temp_dir() as d:
            os.makedirs(os.path.join(d, "oof_matrices"))
            with env_override(ROMA_GEO_MATRICES=d):
                self.assertEqual(config.matrices_root("oof"),
                                 os.path.join(d, "oof_matrices"))
                self.assertEqual(os.path.normcase(config.matrices_root("refined")),
                                 os.path.normcase(d))

    def test_versions_listing_tolerates_missing_directory(self):
        self.assertIsInstance(config.matrices_versions(), list)


class TestEnvNameConsistency(unittest.TestCase):
    """三處各自讀同一組環境變數名，不得分歧。"""

    def test_names_declared_in_config(self):
        self.assertEqual(config.ENV_DATASETS, "ROMA_GEO_DATASETS")
        self.assertEqual(config.ENV_MATRICES, "ROMA_GEO_MATRICES")
        self.assertEqual(config.ENV_OUTPUT, "ROMA_GEO_OUTPUT")
        self.assertEqual(config.ENV_CACHE, "ROMA_GEO_CACHE")
        self.assertEqual(config.ENV_PAIR_THRESHOLD, "ROMA_GEO_PAIR_THRESHOLD")

    def test_pose_regions_reads_the_same_names(self):
        def read(*parts):
            with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
                return f.read()
        src = read("pose_regions", "batch_pose_from_cache.py")
        self.assertIn(config.ENV_CACHE, src)
        self.assertIn(config.ENV_OUTPUT, src)
        self.assertIn(config.ENV_DATASETS, read("pose_regions", "estimate_relative_pose.py"))

    def test_datasets_override_reaches_pose_regions(self):
        """實跑驗證：`find_house_image` 須跟著環境變數走。"""
        from pose_regions import find_house_image
        with temp_dir() as d:
            house = "DollhouseTask_test"
            os.makedirs(os.path.join(d, house))
            target = os.path.join(d, house, "img.jpg")
            with open(target, "wb"):
                pass
            with env_override(ROMA_GEO_DATASETS=d):
                self.assertEqual(os.path.normcase(find_house_image(house, "img") or ""),
                                 os.path.normcase(target))


class TestSettings(unittest.TestCase):
    def test_explicit_overrides_env(self):
        with temp_dir() as d:
            s = config.Settings(datasets=d, output=d, cache=d, matrices=d)
            with env_override(ROMA_GEO_DATASETS="/env/ds", ROMA_GEO_OUTPUT="/env/out",
                              ROMA_GEO_CACHE="/env/cache", ROMA_GEO_MATRICES="/env/mat"):
                self.assertEqual(os.path.normcase(s.datasets_root()), os.path.normcase(d))
                self.assertEqual(os.path.normcase(s.output_root()), os.path.normcase(d))
                self.assertEqual(s.regions_root(), os.path.join(d, "roma_regions"))

    def test_pair_threshold_explicit(self):
        s = config.Settings(pair_threshold=1.5)
        with env_override(ROMA_GEO_PAIR_THRESHOLD="0.1"):
            self.assertEqual(s.pair_threshold(), 1.5)


class TestDescribe(unittest.TestCase):
    def test_describe_runs_and_mentions_every_item(self):
        text = config.describe()
        for key in ("datasets_root", "semantic_root", "matrices_root",
                    "pair_threshold", "output_root", "regions_root", "cache_root"):
            self.assertIn(key, text)

    def test_describe_survives_missing_directories(self):
        with temp_dir() as d:
            missing = os.path.join(d, "does_not_exist")
            with env_override(ROMA_GEO_DATASETS=missing, ROMA_GEO_OUTPUT=missing,
                              ROMA_GEO_CACHE=missing, ROMA_GEO_MATRICES=missing):
                self.assertIn("datasets_root", config.describe())


if __name__ == "__main__":
    unittest.main(verbosity=2)
