"""
對外交付契約測試。

專案有兩條對外輸出路徑，各自負責不同層級：

    roma_geo.to_public()          單對結果 → 對外欄位子集
    export_public.house_to_public()   整間房的連接圖 → 對外欄位子集

兩者都以白名單裁切，因此新增內部欄位不會自動外洩；真正的風險在反方向。
schema 改動時把下游依賴的欄位裁掉，或是側別、旗標的語意悄悄反轉。本檔對兩個方向
都設斷言，並檢查兩條路徑對「一個共視區域包含什麼」的定義是否一致。

另有一項跨檔一致性檢查：`roma_geo.DEFAULTS` 必須逐項等於正式批次
`batch_pose_from_cache` 的 CLI 預設，否則對外入口與批次產出將不可互相比較。
"""
import ast
import inspect
import json
import os
import textwrap
import unittest

from _support import ROOT, temp_dir, write_json  # noqa: F401（匯入時補 sys.path）

import export_public
import roma_geo
from pose_regions import batch_pose_from_cache


def sample_result():
    """一份具代表性的 per-pair 結果（欄位取自實際產出）。"""
    return {
        "schema_version": "wp1-step3-v2",
        "img_size_a": [8192, 4096], "img_size_b": [8192, 4096],
        "params": {"conf_min": 0.65, "retry_conf": 0.3, "max_pts": 4000,
                   "thresh_deg": 0.5, "ransac_iters": 2000, "seed": 0,
                   "sampler": "stratified", "lat_max": 60.0},
        "n": {"raw": 5136, "conf": 869, "used": 869, "inlier": 821, "lat_drop": 134},
        "conf_used": 0.65, "inlier_rate": 0.9448,
        "pose": {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [1.0, 0.0, 0.0],
                 "cam_b_in_a": [-1.0, 0.0, 0.0], "az_ab_deg": -90.0, "az_ba_deg": 90.0,
                 "rot_angle_deg": 0.0, "baseline_txz": 1.0},
        "quality": {"sampson_med_deg": 0.1, "sampson_p90_deg": 0.3, "rot_only_med_deg": 4.2,
                    "az_wedge_a_deg": 40.0, "az_wedge_b_deg": 44.0,
                    "az_cov_a": 0.42, "az_cov_b": 0.39},
        "flags": {"low_matches": False, "no_covisibility": False,
                  "degenerate_rot_only": False, "doorway_wedge": True},
        "regions": [{"n": 700, "frac_of_inliers": 0.853, "mean_conf": 0.872,
                     "a": {"polygon": [[1.0, 2.0], [3.0, 4.0], [1.0, 4.0]],
                           "area_frac": 0.21, "az_range": [10.0, 50.0], "az_width_deg": 40.0},
                     "b": {"polygon": [[5.0, 6.0], [7.0, 8.0], [5.0, 8.0]],
                           "area_frac": 0.18, "az_range": [-20.0, 24.0], "az_width_deg": 44.0}}],
        "inliers": [[1.0, 2.0, 3.0, 4.0, 0.9]] * 5,
        "house": "DollhouseTask_00000", "img_a": "aaa", "img_b": "bbb",
        "gt": {"label": 1, "same_room": True, "az_ab_deg": -88.0, "az_ba_deg": 91.0},
        "time_ms": 123.4,
    }


class TestToPublic(unittest.TestCase):
    """單對契約：共視區域 + 通道方位角 + 可信度。"""

    def setUp(self):
        self.pub = roma_geo.to_public(sample_result())

    def test_contains_contracted_fields(self):
        for k in ("schema_version", "house", "img_a", "img_b", "img_size_a", "img_size_b",
                  "conf_used", "inlier_rate", "az_ab_deg", "az_ba_deg",
                  "no_covisibility", "regions"):
            self.assertIn(k, self.pub)

    def test_excludes_internal_fields(self):
        for k in ("pose", "quality", "flags", "inliers", "params", "n", "gt", "time_ms",
                  "R", "t", "cam_b_in_a"):
            self.assertNotIn(k, self.pub, "%s 不屬於對外契約" % k)

    def test_azimuth_lifted_out_of_pose(self):
        """方位角是交付物本身，須提升到頂層而不是留在 pose 裡。"""
        self.assertEqual(self.pub["az_ab_deg"], -90.0)
        self.assertEqual(self.pub["az_ba_deg"], 90.0)

    def test_region_fields_are_trimmed(self):
        r = self.pub["regions"][0]
        self.assertEqual(set(r), set(roma_geo.PUBLIC_REGION_FIELDS) | {"a", "b"})
        for side in ("a", "b"):
            self.assertEqual(set(r[side]), set(roma_geo.PUBLIC_REGION_SIDE_FIELDS))

    def test_region_public_constants(self):
        """常數即契約，改動須是有意識的。"""
        self.assertEqual(roma_geo.PUBLIC_REGION_FIELDS, ("n", "frac_of_inliers"))
        self.assertEqual(roma_geo.PUBLIC_REGION_SIDE_FIELDS,
                         ("polygon", "area_frac", "az_range"))

    def test_region_to_public_tolerates_missing_side(self):
        """區域缺一側時不應拋出例外，該側以空物件表示。"""
        r = roma_geo.region_to_public({"n": 5, "frac_of_inliers": 1.0,
                                       "a": {"polygon": [[0.0, 0.0]]}})
        self.assertEqual(r["b"], {})
        self.assertEqual(r["a"], {"polygon": [[0.0, 0.0]]})

    def test_no_covisibility_case(self):
        res = sample_result()
        res["pose"] = None
        res["regions"] = []
        res["conf_used"] = None
        res["flags"]["no_covisibility"] = True
        pub = roma_geo.to_public(res)
        self.assertTrue(pub["no_covisibility"])
        self.assertIsNone(pub["az_ab_deg"])
        self.assertIsNone(pub["az_ba_deg"])
        self.assertEqual(pub["regions"], [])

    def test_no_covisibility_is_always_boolean(self):
        res = sample_result()
        del res["flags"]
        self.assertIs(roma_geo.to_public(res)["no_covisibility"], False)

    def test_json_serialisable(self):
        json.dumps(self.pub)

    def test_public_fields_constant_matches_output(self):
        for k in roma_geo.PUBLIC_FIELDS:
            self.assertIn(k, self.pub)


class TestHouseOf(unittest.TestCase):
    def test_infers_house_from_path(self):
        p = os.path.join("datasets", "DollhouseTask_58553", "DollhouseTask_58553", "a.jpg")
        self.assertEqual(roma_geo.house_of(p), "DollhouseTask_58553")

    def test_accepts_windows_separators(self):
        self.assertEqual(roma_geo.house_of(r"D:\data\DollhouseTask_1\x.jpg"), "DollhouseTask_1")

    def test_parent_directory_when_no_dataset_marker(self):
        self.assertEqual(roma_geo.house_of("/tmp/a.jpg"), "tmp")

    def test_bare_filename_is_unknown(self):
        self.assertEqual(roma_geo.house_of("a.jpg"), "unknown")


class TestDefaultsConsistency(unittest.TestCase):
    """`roma_geo.DEFAULTS` 與正式批次的 CLI 預設必須逐項相同。

    兩者分處不同檔案，靠人工同步；不一致時對外入口與 `Output/roma_regions/` 的既有
    產出就不再是同參數的結果，而這種偏移不會有任何錯誤訊息。此處以語法樹讀出批次的
    argparse 預設值直接比對。
    """

    @staticmethod
    def _batch_cli_defaults():
        src = textwrap.dedent(inspect.getsource(batch_pose_from_cache.main))
        out = {}
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)):
                for kw in node.keywords:
                    if kw.arg == "default":
                        try:
                            out[node.args[0].value] = ast.literal_eval(kw.value)
                        except ValueError:
                            pass
        return out

    def test_defaults_match_batch_cli(self):
        cli = self._batch_cli_defaults()
        self.assertTrue(cli, "未能從批次腳本讀出 argparse 預設值")
        for flag, key in (("--conf", "conf_min"), ("--retry_conf", "retry_conf"),
                          ("--max_pts", "max_pts"), ("--thresh_deg", "thresh_deg"),
                          ("--iters", "iters"), ("--sampler", "sampler"),
                          ("--lat_filter", "lat_max")):
            self.assertIn(flag, cli)
            self.assertEqual(roma_geo.DEFAULTS[key], cli[flag],
                             "%s 與 roma_geo.DEFAULTS[%r] 不一致" % (flag, key))

    def test_schema_version_shared(self):
        self.assertEqual(roma_geo.SCHEMA_VERSION, batch_pose_from_cache.SCHEMA_VERSION)


# ---------- 整間房的對外子集 ----------
def sample_house():
    """一份內部 house JSON（欄位取自 stitch_tool/data 的實際產出）。"""
    return {
        "house": "DollhouseTask_00000", "n_images": 3, "width": 8192, "height": 4096,
        "images": ["a.jpg", "b.jpg", "c.jpg"],
        "az_convention": "az(x)=((x/W)*360-180)%360 ; 0deg = image center",
        "doors": [], "n_gt_connected_pairs": 2,
        "pairs": [
            {"i": 0, "j": 1, "logit": 2.89, "prob": 0.9476,
             "az_ij": 116.39, "az_ji": 209.37, "az_source": "geo",
             "az_coarse_ij": 134.23, "az_coarse_ji": 84.38,
             "gt_connected": True, "gt_az_ij": 129.29, "gt_az_ji": 202.24,
             "roma_match_count": 2094, "roma_conf_sum": 1678.54, "roma_conf_mean": 0.8016,
             "geo_n_inlier": 1508, "geo_inlier_rate": 0.7264, "geo_n_regions": 3,
             "geo_conf_used": 0.65, "geo_posed": 1},
            {"i": 0, "j": 2, "logit": -1.2, "prob": 0.2315,
             "az_ij": 12.0, "az_ji": 190.0, "az_source": "coarse",
             "az_coarse_ij": 12.0, "az_coarse_ji": 190.0,
             "gt_connected": False, "gt_az_ij": None, "gt_az_ji": None,
             "roma_match_count": None, "roma_conf_sum": None, "roma_conf_mean": None,
             "geo_n_inlier": None, "geo_inlier_rate": None, "geo_n_regions": None,
             "geo_conf_used": None, "geo_posed": 0},
        ],
    }


class TestHouseToPublic(unittest.TestCase):
    def setUp(self):
        self.pub = export_public.house_to_public(sample_house())

    def test_house_level_fields_kept(self):
        for k in ("house", "n_images", "width", "height", "images", "az_convention",
                  "schema_version"):
            self.assertIn(k, self.pub)

    def test_doors_not_exported(self):
        """doors 是上游輔助頭的單圖輸出,不屬於本模組的交付契約。"""
        d = sample_house()
        d["doors"] = [{"i": 0, "dirs": [{"az": 0.0, "score": 1.0}]}]
        self.assertNotIn("doors", export_public.house_to_public(d))

    def test_internal_house_fields_dropped(self):
        self.assertNotIn("n_gt_connected_pairs", self.pub)

    def test_pair_fields_are_whitelisted(self):
        expected = {"i", "j", "prob", "az_ij", "az_ji", "az_source",
                    "has_geometry", "n_inlier", "inlier_rate", "conf_used",
                    "semantic_corrected"}
        for p in self.pub["pairs"]:
            self.assertEqual(set(p), expected)
            self.assertIs(p["semantic_corrected"], False)

    def test_internal_pair_fields_dropped(self):
        for p in self.pub["pairs"]:
            for k in ("gt_connected", "gt_az_ij", "az_coarse_ij", "az_coarse_ji",
                      "roma_match_count", "roma_conf_mean", "logit", "geo_posed"):
                self.assertNotIn(k, p)

    def test_leaked_fields_is_empty(self):
        self.assertEqual(export_public.leaked_fields(self.pub), set())

    def test_leaked_fields_detects_injection(self):
        """自檢本身要能失敗，否則等於沒檢查。"""
        bad = export_public.house_to_public(sample_house())
        bad["pairs"][0]["gt_connected"] = True
        self.assertEqual(export_public.leaked_fields(bad), {"gt_connected"})

    def test_has_geometry_is_boolean(self):
        """內部存 0/1，對外必須是布林。下游若做嚴格型別檢查，整數會被判為格式錯誤。"""
        vals = [p["has_geometry"] for p in self.pub["pairs"]]
        self.assertEqual(vals, [True, False])
        for v in vals:
            self.assertIsInstance(v, bool)

    def test_az_source_is_preserved(self):
        """降級必須看得出來：方位來自幾何驗證或上游粗篩，不可混為一談。"""
        self.assertEqual([p["az_source"] for p in self.pub["pairs"]], ["geo", "coarse"])

    def test_indices_stay_within_images(self):
        n = len(self.pub["images"])
        for p in self.pub["pairs"]:
            self.assertLess(p["i"], n)
            self.assertLess(p["j"], n)

    def test_missing_optional_fields_are_omitted_not_none(self):
        d = sample_house()
        del d["az_convention"]
        del d["pairs"][0]["az_source"]
        pub = export_public.house_to_public(d)
        self.assertNotIn("az_convention", pub)
        self.assertNotIn("az_source", pub["pairs"][0])

    def test_json_serialisable(self):
        json.dumps(self.pub)


class TestAttachRegions(unittest.TestCase):
    """共視區域併入與側別正規化。

    per-pair 檔名的兩張圖順序由匹配那一輪決定，未必等於 (i, j)。輸出中區域的 a 側
    必須永遠對應 i 圖：漏做這件事的症狀是左右對調而畫面仍然合理，不會有錯誤訊息，
    因此這裡用「兩側內容可區分」的測資，反序時真的去比對內容有沒有換位置。
    """

    HOUSE = "DollhouseTask_00000"
    SIDE_A = {"polygon": [[0.0, 0.0]], "area_frac": 0.1,
              "az_range": [0.0, 10.0], "az_width_deg": 10.0}
    SIDE_B = {"polygon": [[1.0, 1.0]], "area_frac": 0.2,
              "az_range": [5.0, 20.0], "az_width_deg": 15.0}
    REGION = {"n": 100, "frac_of_inliers": 1.0, "mean_conf": 0.9,
              "a": SIDE_A, "b": SIDE_B}

    def _run(self, filenames):
        pub = export_public.house_to_public(sample_house())
        with temp_dir() as d:
            for fn in filenames:
                write_json(os.path.join(d, fn), {"regions": [self.REGION]})
            n = export_public.attach_regions(self.HOUSE, pub, d)
        return pub, n

    def test_forward_order_keeps_sides(self):
        pub, n = self._run(["%s_a_b.json" % self.HOUSE])
        self.assertEqual(n, 1)
        r = pub["pairs"][0]["regions"][0]
        self.assertEqual(r["a"]["area_frac"], self.SIDE_A["area_frac"])
        self.assertEqual(r["b"]["area_frac"], self.SIDE_B["area_frac"])

    def test_reverse_order_is_normalised(self):
        """只有反向檔存在時，兩側必須對調，使 a 側仍對應 i 圖。"""
        pub, n = self._run(["%s_b_a.json" % self.HOUSE])
        self.assertEqual(n, 1)
        r = pub["pairs"][0]["regions"][0]
        self.assertEqual(r["a"]["area_frac"], self.SIDE_B["area_frac"],
                         "反序檔的 b 側應成為輸出的 a 側")
        self.assertEqual(r["b"]["area_frac"], self.SIDE_A["area_frac"])
        self.assertEqual(r["a"]["polygon"], self.SIDE_B["polygon"])

    def test_side_order_is_indistinguishable_downstream(self):
        """正序與反序的來源檔，經正規化後應得到相同的輸出。"""
        fwd, _ = self._run(["%s_a_b.json" % self.HOUSE])
        # 反向檔的內容本身也是反的（該檔的 a 側是 b 圖），故先自行對調再寫入。
        pub = export_public.house_to_public(sample_house())
        with temp_dir() as d:
            write_json(os.path.join(d, "%s_b_a.json" % self.HOUSE),
                       {"regions": [export_public.swap_sides(self.REGION)]})
            export_public.attach_regions(self.HOUSE, pub, d)
        self.assertEqual(pub["pairs"][0]["regions"], fwd["pairs"][0]["regions"])

    def test_swap_sides_is_involutive(self):
        self.assertEqual(export_public.swap_sides(export_public.swap_sides(self.REGION)),
                         self.REGION)

    def test_swap_sides_keeps_other_fields(self):
        s = export_public.swap_sides(self.REGION)
        self.assertEqual(s["n"], self.REGION["n"])
        self.assertEqual(s["frac_of_inliers"], self.REGION["frac_of_inliers"])
        self.assertIsNot(s, self.REGION, "不得就地修改來源")

    def test_forward_wins_when_both_exist(self):
        pub, _ = self._run(["%s_a_b.json" % self.HOUSE, "%s_b_a.json" % self.HOUSE])
        r = pub["pairs"][0]["regions"][0]
        self.assertEqual(r["a"]["area_frac"], self.SIDE_A["area_frac"])

    def test_order_marker_is_not_emitted(self):
        """側別已正規化，不再輸出 regions_order：留著會誘使下游再判斷一次。"""
        pub, _ = self._run(["%s_b_a.json" % self.HOUSE])
        self.assertNotIn("regions_order", pub["pairs"][0])

    def test_absent_file_leaves_pair_untouched(self):
        pub, n = self._run([])
        self.assertEqual(n, 0)
        for p in pub["pairs"]:
            self.assertNotIn("regions", p)

    def test_empty_regions_are_not_attached(self):
        pub = export_public.house_to_public(sample_house())
        with temp_dir() as d:
            write_json(os.path.join(d, "%s_a_b.json" % self.HOUSE), {"regions": []})
            n = export_public.attach_regions(self.HOUSE, pub, d)
        self.assertEqual(n, 0)
        self.assertNotIn("regions", pub["pairs"][0])

    def test_attached_regions_do_not_leak_internal_prefixes(self):
        pub, _ = self._run(["%s_a_b.json" % self.HOUSE])
        self.assertEqual(export_public.leaked_fields(pub), set())

    def test_region_field_set_matches_single_pair_contract(self):
        """兩條對外路徑的區域欄位必須完全相同（共用 `roma_geo.region_to_public()`）。"""
        pub, _ = self._run(["%s_a_b.json" % self.HOUSE])
        attached = pub["pairs"][0]["regions"][0]
        single = roma_geo.to_public(sample_result())["regions"][0]
        self.assertEqual(set(attached), set(single))
        for side in ("a", "b"):
            self.assertEqual(set(attached[side]), set(single[side]))

    def test_internal_region_fields_are_trimmed(self):
        """`mean_conf` 屬匹配內部品質、`az_width_deg` 可由 az_range 推得，皆不外流。"""
        pub, _ = self._run(["%s_a_b.json" % self.HOUSE])
        r = pub["pairs"][0]["regions"][0]
        self.assertNotIn("mean_conf", r)
        self.assertEqual(set(r), {"n", "frac_of_inliers", "a", "b"})
        for side in ("a", "b"):
            self.assertEqual(set(r[side]), {"polygon", "area_frac", "az_range"})
        # 來源確實帶著這兩個欄位，否則本測試只是在驗空集合。
        self.assertIn("mean_conf", self.REGION)
        self.assertIn("az_width_deg", self.REGION["a"])

    def test_az_width_is_recoverable_from_az_range(self):
        """裁掉 az_width_deg 的前提：它可由 az_range 兩端相減得到（誤差限於四捨五入）。"""
        pub, _ = self._run(["%s_a_b.json" % self.HOUSE])
        for side, src in (("a", self.SIDE_A), ("b", self.SIDE_B)):
            lo, hi = pub["pairs"][0]["regions"][0][side]["az_range"]
            self.assertAlmostEqual(hi - lo, src["az_width_deg"], delta=0.11)


if __name__ == "__main__":
    unittest.main(verbosity=2)
