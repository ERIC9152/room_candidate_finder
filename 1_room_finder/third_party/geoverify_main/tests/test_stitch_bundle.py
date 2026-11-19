"""
工具載入對外交付檔的相容層測試，以及 HOTSPOT 真值（GT）即時補上的測試。

`export_public.py` 的交付檔可以直接放進產出包給工具載入。工具渲染得出來，即為
「交付內容足夠」的直接檢驗。要成立有三個前提，本檔逐一驗證：

  1. 包內的 `<house>.json` 是交付檔本身，未經改寫（改寫過就不算檢驗了）。
  2. 伺服器讀取時把幾何證據欄位對映回工具沿用的內部名，並標明無 GT 真值。
  3. 每對內嵌的共視區域能經 `/api/regions` 供應，且側別（swap）判定正確。

第 3 點最需要測：交付檔的區域側別已正規化為「a 側 = 該對的 i 圖」，查詢時哪張圖當
上半部才決定 swap。判反了的症狀是遮罩上下顛倒而畫面仍然合理，不會有錯誤訊息。

本檔另涵蓋 `gt_hotspot.py`（HOTSPOT 真值的載入／對映／比對回報，`prepare_house_data.py`
建檔時與 `server.py` 即時補 GT 共用同一份實作）：第 2 點說的「標明無 GT 真值」只是
交付檔轉換當下的狀態，bundle 內若帶了 HOTSPOT 檔，伺服器讀取時會嘗試即時補上。
真的對得上才補，房號對不上時必須在回報訊息裡明說，不能安靜地假裝有 GT
（使用者曾把別棟房子的 HOTSPOT 放錯資料夾，因為安靜不顯示而誤以為功能沒接上）。
"""
import json
import os
import sys
import unittest

from _support import ROOT, W_DEF, H_DEF, quiet, temp_dir, write_json  # noqa: F401（匯入時補 sys.path）

# stitch_tool 內部彼此以同層模組名互相匯入（house_dirs 等），需補進 sys.path。
_TOOL = os.path.join(ROOT, "stitch_tool")
if _TOOL not in sys.path:
    sys.path.insert(0, _TOOL)

import gt_hotspot                                     # noqa: E402
import pack_bundle                                    # noqa: E402
import server                                         # noqa: E402（匯入會建立 state/，見模組說明）

HOUSE = "DollhouseTask_00000"
IMG = ["aaa.jpg", "bbb.jpg", "ccc.jpg"]

REGION = {"n": 120, "frac_of_inliers": 0.9,
          "a": {"polygon": [[1.0, 2.0], [3.0, 4.0], [1.0, 4.0]],
                "area_frac": 0.11, "az_range": [0.0, 30.0]},
          "b": {"polygon": [[5.0, 6.0], [7.0, 8.0], [5.0, 8.0]],
                "area_frac": 0.22, "az_range": [10.0, 50.0]}}


def public_house():
    """一份對外交付檔（欄位取自 export_public.py 的實際輸出）。"""
    return {
        "house": HOUSE, "n_images": 3, "width": 8192, "height": 4096,
        "images": list(IMG),
        "az_convention": "az(x)=((x/W)*360-180)%360 ; 0deg = image center",
        "doors": [],
        "pairs": [
            {"i": 0, "j": 1, "prob": 0.95, "az_ij": 116.39, "az_ji": 209.37,
             "az_source": "geo", "has_geometry": True, "n_inlier": 1508,
             "inlier_rate": 0.7264, "conf_used": 0.65, "regions": [REGION]},
            {"i": 0, "j": 2, "prob": 0.23, "az_ij": 12.0, "az_ji": 190.0,
             "az_source": "coarse", "has_geometry": False},
        ],
    }


def internal_house():
    """對照組：流程內部的房子資料（工具原本讀的那一種）。"""
    return {
        "house": HOUSE, "n_images": 3, "width": 8192, "height": 4096,
        "images": list(IMG), "doors": [], "n_gt_connected_pairs": 1,
        "pairs": [{"i": 0, "j": 1, "logit": 2.89, "prob": 0.95,
                   "az_ij": 116.39, "az_ji": 209.37, "az_source": "geo",
                   "gt_connected": True, "geo_posed": 1, "geo_n_inlier": 1508}],
    }


class TestIsPublicHouse(unittest.TestCase):
    def test_detects_public_file(self):
        self.assertTrue(server.is_public_house(public_house()))

    def test_rejects_internal_file(self):
        self.assertFalse(server.is_public_house(internal_house()))

    def test_no_pairs_is_not_public(self):
        self.assertFalse(server.is_public_house({"house": HOUSE, "pairs": []}))
        self.assertFalse(server.is_public_house({}))


class TestPublicHouseToTool(unittest.TestCase):
    def setUp(self):
        self.src = public_house()
        self.out = server.public_house_to_tool(self.src)

    def test_geometry_fields_are_renamed(self):
        p = self.out["pairs"][0]
        self.assertEqual(p["geo_n_inlier"], 1508)
        self.assertEqual(p["geo_inlier_rate"], 0.7264)
        self.assertEqual(p["geo_conf_used"], 0.65)
        self.assertEqual(p["geo_posed"], 1)
        for k in ("n_inlier", "inlier_rate", "conf_used", "has_geometry"):
            self.assertNotIn(k, p)

    def test_mapping_covers_every_public_geometry_field(self):
        """對映表須涵蓋交付檔實際會出現的欄位，漏一個就會在工具上默默消失。"""
        import export_public
        renamed = set(server.PUBLIC_TO_TOOL)
        geo_fields = {v for k, v in export_public.PAIR_FIELDS.items() if k.startswith("geo_")}
        self.assertEqual(geo_fields, renamed)

    def test_has_geometry_becomes_integer_flag(self):
        """前端把 geo_posed 當 0/1 用（排序與徽章），布林要轉成整數。"""
        vals = [p.get("geo_posed") for p in self.out["pairs"]]
        self.assertEqual(vals, [1, 0])
        for v in vals:
            self.assertIsInstance(v, int)

    def test_passthrough_fields_are_untouched(self):
        p = self.out["pairs"][0]
        for k in ("i", "j", "prob", "az_ij", "az_ji", "az_source"):
            self.assertEqual(p[k], self.src["pairs"][0][k])

    def test_marks_absence_of_ground_truth(self):
        """交付檔不含真值；工具據 has_gt=false 顯示「無 GT 真值」。"""
        self.assertIs(self.out["has_gt"], False)
        self.assertEqual(self.out["n_gt_connected_pairs"], 0)

    def test_regions_are_moved_out_of_house_payload(self):
        """區域改由 /api/regions 供應，維持前端既有取用方式，也避免一次回傳全部多邊形。"""
        for p in self.out["pairs"]:
            self.assertNotIn("regions", p)

    def test_input_is_not_mutated(self):
        self.assertIn("regions", self.src["pairs"][0])
        self.assertIn("has_geometry", self.src["pairs"][0])
        self.assertNotIn("has_gt", self.src)

    def test_house_level_fields_survive(self):
        for k in ("house", "n_images", "width", "height", "images", "az_convention"):
            self.assertEqual(self.out[k], self.src[k])


class TestEmbeddedRegions(unittest.TestCase):
    """`/api/regions` 從交付檔取區域，含側別判定。"""

    def setUp(self):
        self.d = public_house()

    def test_forward_query_does_not_swap(self):
        r = server.embedded_regions(self.d, "aaa", "bbb")
        self.assertIsNotNone(r)
        self.assertFalse(r["swap"])
        self.assertEqual(r["regions"], [REGION])

    def test_reverse_query_sets_swap(self):
        """查詢把 j 圖當上半部時必須回 swap=true，前端據此對調兩側。"""
        r = server.embedded_regions(self.d, "bbb", "aaa")
        self.assertTrue(r["swap"])
        self.assertEqual(r["regions"], [REGION], "regions 本身不對調，只回旗標")

    def test_image_sizes_come_from_house_level(self):
        """前端用 img_size_* 換算多邊形座標；同房影像尺寸相同，交付檔只記一次。"""
        r = server.embedded_regions(self.d, "aaa", "bbb")
        self.assertEqual(r["img_size_a"], [8192, 4096])
        self.assertEqual(r["img_size_b"], [8192, 4096])

    def test_pair_without_regions_returns_none(self):
        self.assertIsNone(server.embedded_regions(self.d, "aaa", "ccc"))

    def test_unknown_images_return_none(self):
        self.assertIsNone(server.embedded_regions(self.d, "aaa", "zzz"))

    def test_tolerates_extensions_in_image_list(self):
        """images 帶副檔名、查詢參數不帶，比對前須先去除。"""
        self.assertIsNotNone(server.embedded_regions(self.d, "aaa", "bbb"))

    def test_empty_house(self):
        self.assertIsNone(server.embedded_regions({}, "aaa", "bbb"))


class TestPackHousePublic(unittest.TestCase):
    """交付檔模式的打包：包內房子資料必須是交付檔本身。"""

    def _pack(self, with_images=True):
        d = temp_dir().__enter__()
        pub_dir = os.path.join(d, "public")
        ds_dir = os.path.join(d, "datasets", HOUSE)
        out_dir = os.path.join(d, "bundles")
        os.makedirs(ds_dir, exist_ok=True)
        write_json(os.path.join(pub_dir, HOUSE + ".json"), public_house())
        if with_images:
            for fn in IMG:
                with open(os.path.join(ds_dir, fn), "wb") as f:
                    f.write(b"jpeg-bytes-" + fn.encode())
        with quiet():
            ok = pack_bundle.pack_house_public(HOUSE, pub_dir, os.path.join(d, "datasets"),
                                               out_dir, copy=True)
        return d, pub_dir, out_dir, ok

    def test_house_json_is_byte_identical(self):
        """未經改寫是這條路徑的前提：改寫過就不再是「下游收到的那一份」。"""
        d, pub_dir, out_dir, ok = self._pack()
        try:
            self.assertTrue(ok)
            src = os.path.join(pub_dir, HOUSE + ".json")
            dst = os.path.join(out_dir, HOUSE, HOUSE + ".json")
            with open(src, "rb") as f1, open(dst, "rb") as f2:
                self.assertEqual(f1.read(), f2.read())
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_images_are_packed_and_no_regions_dir(self):
        d, _, out_dir, _ = self._pack()
        try:
            hd = os.path.join(out_dir, HOUSE)
            self.assertEqual(sorted(os.listdir(os.path.join(hd, "images"))), sorted(IMG))
            self.assertFalse(os.path.isdir(os.path.join(hd, "regions")),
                             "交付檔的區域已內嵌，不應另建 regions/")
            self.assertFalse(os.path.exists(os.path.join(hd, "semantic.json")))
            self.assertFalse([f for f in os.listdir(hd) if "HOTSPOT" in f],
                             "交付檔模式不得帶入真值")
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_missing_house_is_reported_not_raised(self):
        with temp_dir() as d, quiet():
            self.assertFalse(pack_bundle.pack_house_public(
                "NoSuchHouse", d, d, os.path.join(d, "out"), copy=True))


class TestRoundTrip(unittest.TestCase):
    """打包 → 伺服器讀取，兩步串起來的結果須與前端預期一致。"""

    def test_packed_file_loads_as_tool_data(self):
        with temp_dir() as d:
            pub_dir = os.path.join(d, "public")
            out_dir = os.path.join(d, "bundles")
            write_json(os.path.join(pub_dir, HOUSE + ".json"), public_house())
            with quiet():
                pack_bundle.pack_house_public(HOUSE, pub_dir, os.path.join(d, "ds"),
                                              out_dir, copy=True)
            with open(os.path.join(out_dir, HOUSE, HOUSE + ".json"), encoding="utf-8") as f:
                packed = json.load(f)
            self.assertTrue(server.is_public_house(packed))
            tool = server.public_house_to_tool(packed)
            self.assertIs(tool["has_gt"], False)
            self.assertEqual(tool["pairs"][0]["geo_n_inlier"], 1508)
            r = server.embedded_regions(packed, "aaa", "bbb")
            self.assertEqual(r["regions"][0]["n"], 120)

    def test_internal_house_is_served_unchanged(self):
        """既有的內部資料包不受影響：不是交付檔就原樣回傳。"""
        d = internal_house()
        self.assertFalse(server.is_public_house(d))


# ---------------------------------------------------------------------------
# gt_hotspot.py：HOTSPOT 真值的載入／對映／比對回報，prepare_house_data.py 建檔與
# server.py 即時補 GT 共用同一份實作。
# ---------------------------------------------------------------------------

def hotspot_room(owner, roomidx, targets_with_x):
    """建一筆 HOTSPOTOFROOM 項目。`targets_with_x` 為 `[(檔名, x座標), ...]`，依序對映
    `ToIDName.IDName` 與 `HSLoc.Coordinate`。兩者逐項對應，是規約明定的格式（見
    gt_hotspot.py 模組說明），不是巧合。"""
    return {
        "IDName": owner,
        "Roomidx": roomidx,
        "ToIDName": {"IDName": [t for t, _ in targets_with_x]},
        "HSLoc": {"Coordinate": [{"x": x, "y": 0.0} for _, x in targets_with_x]},
        "HSLoc_3D": {"Coordinate_3D": []},   # 未使用，見 gt_hotspot.py
        "hscount": len(targets_with_x),
    }


def az_of(x, w=W_DEF):
    """由規約獨立算方位，供測試當 oracle，不呼叫受測程式碼：
    az = ((x / W) * 360 - 180) mod 360。"""
    return ((x / w) * 360.0 - 180.0) % 360.0


def house_without_gt(house, images):
    """『還沒有 GT 欄位』的房子資料。對外交付檔轉換後的形狀：每對沒有 gt_connected
    鍵，是 has_gt_fields() 判定「還沒算過」、augment_house_with_gt 因此該接手的前提。"""
    pairs = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            pairs.append({"i": i, "j": j, "prob": 0.5, "az_ij": 0.0, "az_ji": 180.0,
                          "az_source": "coarse"})
    return {"house": house, "n_images": len(images), "width": W_DEF, "height": H_DEF,
            "images": list(images), "doors": [], "pairs": pairs}


class TestLoadGtFormula(unittest.TestCase):
    """`load_gt` 的方位換算須與規約公式一致；用已知座標獨立驗證，不從實作反推。"""

    def test_known_x_yields_expected_azimuth(self):
        # 規約範例：x = 0.75W 應得 90 度。
        x = 0.75 * W_DEF
        with temp_dir() as d:
            hs = {"HOTSPOTOFROOM": [hotspot_room("a.jpg", 0, [("b.jpg", x)])]}
            p = write_json(os.path.join(d, "x_HOTSPOT.json"), hs)
            gt = gt_hotspot.load_gt([p], W_DEF)
        self.assertAlmostEqual(gt["a.jpg"]["b.jpg"], 90.0, places=6)
        self.assertAlmostEqual(gt["a.jpg"]["b.jpg"], az_of(x), places=6)

    def test_multiple_targets_map_positionally_to_coordinates(self):
        """ToIDName.IDName[k] 與 HSLoc.Coordinate[k] 逐項對應，不是集合關係。"""
        xs = [0.0, W_DEF / 2.0, W_DEF * 0.75]
        with temp_dir() as d:
            hs = {"HOTSPOTOFROOM": [hotspot_room(
                "a.jpg", 0, [("b.jpg", xs[0]), ("c.jpg", xs[1]), ("d.jpg", xs[2])])]}
            p = write_json(os.path.join(d, "m_HOTSPOT.json"), hs)
            gt = gt_hotspot.load_gt([p], W_DEF)
        self.assertAlmostEqual(gt["a.jpg"]["b.jpg"], az_of(xs[0]))
        self.assertAlmostEqual(gt["a.jpg"]["c.jpg"], az_of(xs[1]))
        self.assertAlmostEqual(gt["a.jpg"]["d.jpg"], az_of(xs[2]))

    def test_raw_dict_direction_is_not_implicitly_symmetric(self):
        """只有 a 說連到 b：raw dict 不該自動生出 b→a 的反向項，對稱化是
        gt_pair_fields 的責任，不是 load_gt 的。"""
        with temp_dir() as d:
            hs = {"HOTSPOTOFROOM": [hotspot_room("a.jpg", 0, [("b.jpg", 0.0)])]}
            p = write_json(os.path.join(d, "d_HOTSPOT.json"), hs)
            gt = gt_hotspot.load_gt([p], W_DEF)
        self.assertNotIn("b.jpg", gt)

    def test_missing_width_yields_none_azimuth_but_keeps_connectivity(self):
        with temp_dir() as d:
            hs = {"HOTSPOTOFROOM": [hotspot_room("a.jpg", 0, [("b.jpg", 100.0)])]}
            p = write_json(os.path.join(d, "w_HOTSPOT.json"), hs)
            gt = gt_hotspot.load_gt([p], None)
        self.assertIn("b.jpg", gt["a.jpg"])
        self.assertIsNone(gt["a.jpg"]["b.jpg"])


class TestGtPairFields(unittest.TestCase):
    """GT 對映到單一對的規則：任一方向出現在 ToIDName 就算連通；方位維持方向性，
    沒登記的那個方向就是 None，不會用另一個方向的值頂替。"""

    def test_connected_forward_only(self):
        gt = {"a.jpg": {"b.jpg": 90.0}}
        out = gt_hotspot.gt_pair_fields(gt, "a.jpg", "b.jpg")
        self.assertTrue(out["gt_connected"])
        self.assertEqual(out["gt_az_ij"], 90.0)
        self.assertIsNone(out["gt_az_ji"])

    def test_connected_reverse_only(self):
        gt = {"b.jpg": {"a.jpg": 270.0}}
        out = gt_hotspot.gt_pair_fields(gt, "a.jpg", "b.jpg")
        self.assertTrue(out["gt_connected"])
        self.assertIsNone(out["gt_az_ij"])
        self.assertEqual(out["gt_az_ji"], 270.0)

    def test_connected_both_directions_keep_independent_values(self):
        gt = {"a.jpg": {"b.jpg": 90.0}, "b.jpg": {"a.jpg": 265.0}}
        out = gt_hotspot.gt_pair_fields(gt, "a.jpg", "b.jpg")
        self.assertTrue(out["gt_connected"])
        self.assertEqual(out["gt_az_ij"], 90.0)
        self.assertEqual(out["gt_az_ji"], 265.0)

    def test_not_connected(self):
        gt = {"a.jpg": {"c.jpg": 90.0}}
        out = gt_hotspot.gt_pair_fields(gt, "a.jpg", "b.jpg")
        self.assertFalse(out["gt_connected"])
        self.assertIsNone(out["gt_az_ij"])
        self.assertIsNone(out["gt_az_ji"])

    def test_unrelated_house_has_no_entries_at_all(self):
        """沒有任何標註時 gt_connected 為 None，與「已確認不連通」的 False 分開。"""
        self.assertEqual(gt_hotspot.gt_pair_fields({}, "a.jpg", "b.jpg"),
                         {"gt_connected": None, "gt_az_ij": None, "gt_az_ji": None})


class TestFindHotspotFiles(unittest.TestCase):
    """尋找 HOTSPOT 檔案：一律排序；recursive=True 才往子資料夾找。bundle 資料夾裡
    HOTSPOT 檔不一定乖乖放在最上層，這是實際發生過的真實佈局，不是假設性邊界情況。"""

    def test_sorted_output_regardless_of_glob_order(self):
        with temp_dir() as d:
            write_json(os.path.join(d, "zzz_HOTSPOT.json"), {"HOTSPOTOFROOM": []})
            write_json(os.path.join(d, "aaa_HOTSPOT.json"), {"HOTSPOTOFROOM": []})
            found = gt_hotspot.find_hotspot_files(d)
        self.assertEqual([os.path.basename(p) for p in found],
                         ["aaa_HOTSPOT.json", "zzz_HOTSPOT.json"])

    def test_flat_search_ignores_nested_files(self):
        with temp_dir() as d:
            write_json(os.path.join(d, "images", "nested_HOTSPOT.json"), {"HOTSPOTOFROOM": []})
            self.assertEqual(gt_hotspot.find_hotspot_files(d, recursive=False), [])

    def test_recursive_search_finds_nested_files(self):
        """實際發生過的佈局：HOTSPOT 檔混在 images/ 資料夾裡一起出現在 bundle 中。"""
        with temp_dir() as d:
            nested = write_json(os.path.join(d, "images", "nested_HOTSPOT.json"),
                                {"HOTSPOTOFROOM": []})
            found = gt_hotspot.find_hotspot_files(d, recursive=True)
        self.assertEqual(found, [nested])

    def test_recursive_search_still_finds_top_level_files(self):
        """recursive=True 是 recursive=False 的超集，不能因為往下找就漏掉最上層的檔案。"""
        with temp_dir() as d:
            top = write_json(os.path.join(d, "top_HOTSPOT.json"), {"HOTSPOTOFROOM": []})
            found = gt_hotspot.find_hotspot_files(d, recursive=True)
        self.assertEqual(found, [top])

    def test_no_files_returns_empty_list(self):
        with temp_dir() as d:
            self.assertEqual(gt_hotspot.find_hotspot_files(d), [])
            self.assertEqual(gt_hotspot.find_hotspot_files(d, recursive=True), [])


class TestLoadGtOrderIndependence(unittest.TestCase):
    """多樓層房子每層一個 HOTSPOT 檔，樓梯配對在相鄰兩層都有標註但座標不同；規約明定
    「後寫覆蓋前寫」且必須先排序檔名，結果不能受建檔（或檔案系統列舉）順序影響。"""

    X_FLOOR1, X_FLOOR2 = 100.0, 5000.0   # 差距遠大於實測下限 5~120px，斷言更明確

    def _fixtures(self):
        floor1 = {"HOTSPOTOFROOM": [hotspot_room("stair.jpg", 0, [("up.jpg", self.X_FLOOR1)])]}
        floor2 = {"HOTSPOTOFROOM": [hotspot_room("stair.jpg", 1, [("up.jpg", self.X_FLOOR2)])]}
        return floor1, floor2

    def test_result_independent_of_creation_order(self):
        floor1, floor2 = self._fixtures()

        with temp_dir() as d1:
            write_json(os.path.join(d1, "floor1_HOTSPOT.json"), floor1)
            write_json(os.path.join(d1, "floor2_HOTSPOT.json"), floor2)
            gt_1_then_2 = gt_hotspot.load_gt(gt_hotspot.find_hotspot_files(d1), W_DEF)

        with temp_dir() as d2:
            write_json(os.path.join(d2, "floor2_HOTSPOT.json"), floor2)
            write_json(os.path.join(d2, "floor1_HOTSPOT.json"), floor1)
            gt_2_then_1 = gt_hotspot.load_gt(gt_hotspot.find_hotspot_files(d2), W_DEF)

        self.assertEqual(gt_1_then_2, gt_2_then_1)
        # 排序後 floor2 在 floor1 之後處理，後寫覆蓋前寫，最終值須是 floor2 那筆。
        self.assertAlmostEqual(gt_1_then_2["stair.jpg"]["up.jpg"], az_of(self.X_FLOOR2))

    def test_load_gt_sorts_its_own_input(self):
        """直接傳入未排序的路徑清單也要得到同一個結果。

        上一個測試經由 `find_hotspot_files` 取得清單，而該函式本身已排序，因此驗不到
        `load_gt` 自己那層排序。但 `load_gt` 的契約是「呼叫端不必自行排序」。若這層
        排序被當成冗餘而移除，直接以 `glob` 結果呼叫的路徑就會退回依賴檔案系統列舉
        順序，Linux 與 Windows 取到不同樓層的座標，且不會有任何錯誤徵兆。
        """
        floor1, floor2 = self._fixtures()
        with temp_dir() as d:
            p1 = write_json(os.path.join(d, "floor1_HOTSPOT.json"), floor1)
            p2 = write_json(os.path.join(d, "floor2_HOTSPOT.json"), floor2)
            asc = gt_hotspot.load_gt([p1, p2], W_DEF)
            desc = gt_hotspot.load_gt([p2, p1], W_DEF)
        self.assertEqual(asc, desc, "load_gt 未對傳入的路徑排序")
        self.assertAlmostEqual(desc["stair.jpg"]["up.jpg"], az_of(self.X_FLOOR2))

    def test_augment_house_with_gt_also_independent_of_creation_order(self):
        """同一個不變量在整合層（augment_house_with_gt）再驗一次，確保上層沒有另外
        引入順序依賴（例如改用 os.listdir 蒐集檔案）。"""
        floor1, floor2 = self._fixtures()
        images = ["stair.jpg", "up.jpg"]

        with temp_dir() as d1:
            write_json(os.path.join(d1, "floor1_HOTSPOT.json"), floor1)
            write_json(os.path.join(d1, "floor2_HOTSPOT.json"), floor2)
            out1, _ = gt_hotspot.augment_house_with_gt(house_without_gt("M1", images), d1)

        with temp_dir() as d2:
            write_json(os.path.join(d2, "floor2_HOTSPOT.json"), floor2)
            write_json(os.path.join(d2, "floor1_HOTSPOT.json"), floor1)
            out2, _ = gt_hotspot.augment_house_with_gt(house_without_gt("M1", images), d2)

        self.assertEqual(out1["pairs"], out2["pairs"])
        conn = [p for p in out1["pairs"] if p["gt_connected"]]
        self.assertEqual(len(conn), 1)
        self.assertAlmostEqual(conn[0]["gt_az_ij"], round(az_of(self.X_FLOOR2), 2) % 360.0, places=2)


class TestHasGtFields(unittest.TestCase):
    """判斷房子資料是否『已經算過』GT（不論算出的結論是有沒有），決定
    augment_house_with_gt 該不該接手。"""

    def test_true_when_has_gt_true(self):
        self.assertTrue(gt_hotspot.has_gt_fields({"has_gt": True, "pairs": []}))

    def test_true_when_pairs_carry_key_even_if_has_gt_false(self):
        """has_gt=False 但每對都有 gt_connected 鍵：代表建檔當時真的找過但沒找到，
        是「已有答案」，不是「還沒算過」，不該被覆寫。"""
        d = {"has_gt": False, "pairs": [{"i": 0, "j": 1, "gt_connected": False,
                                         "gt_az_ij": None, "gt_az_ji": None}]}
        self.assertTrue(gt_hotspot.has_gt_fields(d))

    def test_false_when_schema_lacks_gt_entirely(self):
        """對外交付檔轉換後的形狀：沒有 has_gt、每對也沒有 gt_connected 鍵。"""
        d = {"pairs": [{"i": 0, "j": 1, "prob": 0.5}]}
        self.assertFalse(gt_hotspot.has_gt_fields(d))

    def test_false_when_no_pairs_and_no_has_gt(self):
        self.assertFalse(gt_hotspot.has_gt_fields({"pairs": []}))
        self.assertFalse(gt_hotspot.has_gt_fields({}))


class TestMatchReport(unittest.TestCase):
    """比對回報：無論全對不上、部分對上、或全數相符都要出聲，不能只在「正常」時才
    輸出。安靜地不顯示 GT 正是使用者曾經誤以為功能沒接上的原因。"""

    def test_zero_overlap_reports_recognizable_warning(self):
        gt = {"x1.jpg": {"x2.jpg": 0.0}, "x3.jpg": {"x2.jpg": 10.0}}   # 提到 x1,x2,x3 三張
        msg = gt_hotspot.match_report(gt, ["y1.jpg", "y2.jpg"])        # 本房是完全不同的兩張
        self.assertIn("3", msg)          # HOTSPOT 提到的張數
        self.assertIn("2", msg)          # 本房張數
        self.assertIn("無一相符", msg)
        self.assertIn("放錯房", msg, "訊息須包含可辨識的警告字樣")

    def test_partial_overlap_reports_actual_match_count(self):
        gt = {"p1.jpg": {"p2.jpg": 0.0}, "q1.jpg": {"q2.jpg": 0.0}}     # 提到 p1,p2,q1,q2 四張
        msg = gt_hotspot.match_report(gt, ["p1.jpg", "p2.jpg", "p3.jpg"])   # 只有 p1,p2 對上
        self.assertIn("4", msg)   # HOTSPOT 提到的張數
        self.assertIn("2", msg)   # 不屬於本房的張數
        self.assertIn("不屬於本房", msg)

    def test_house_images_not_covered_are_reported(self):
        """兩層樓只放一層：被提到的影像確實全屬本房，但本房有影像沒被標註涵蓋。

        這是實測發現會誤導的情況。舊判斷式只比「HOTSPOT 提到的有幾張落在本房」，
        缺樓層時該式不成立而回報「全數相符」，使用者會誤以為真值完整，未涵蓋影像
        的配對卻一律顯示 GT 不連通，看起來像模型誤報。
        """
        gt = {"f1a.jpg": {"f1b.jpg": 0.0}}                       # 只有一樓的兩張
        house = ["f1a.jpg", "f1b.jpg", "f2a.jpg", "f2b.jpg"]     # 房子實際有四張
        msg = gt_hotspot.match_report(gt, house)
        self.assertNotIn("全數相符", msg, "缺樓層不可回報為完整")
        self.assertIn("未被標註涵蓋", msg)
        self.assertIn("4", msg)   # 本房張數
        self.assertIn("2", msg)   # 未涵蓋張數

    def test_both_directions_reported_together(self):
        """兩種不齊全同時發生時，兩件事都要講，不能只報其中一個。"""
        gt = {"p1.jpg": {"p2.jpg": 0.0}, "other.jpg": {"alien.jpg": 0.0}}
        msg = gt_hotspot.match_report(gt, ["p1.jpg", "p2.jpg", "p3.jpg"])
        self.assertIn("不屬於本房", msg)
        self.assertIn("未被標註涵蓋", msg)

    def test_full_overlap_reports_success(self):
        gt = {"a.jpg": {"b.jpg": 0.0}}
        msg = gt_hotspot.match_report(gt, ["a.jpg", "b.jpg"])
        self.assertIn("全數相符", msg)
        self.assertNotIn("未被標註涵蓋", msg)

    def test_empty_gt_is_reported_distinctly_from_mismatch(self):
        """HOTSPOT 檔案存在但內容空／格式不符：不該被誤報成「放錯房」。"""
        msg = gt_hotspot.match_report({}, ["a.jpg", "b.jpg"])
        self.assertNotIn("放錯房", msg)


class TestAugmentHouseWithGt(unittest.TestCase):
    """`/api/house/<house>` 即時補 GT 的整合測試，逐一對應驗收要求的每個情境。"""

    def test_matching_hotspot_computes_gt_fields(self):
        """HOTSPOT 與本房影像正常對上：GT 欄位正確算出，方位用已知座標驗證。"""
        images = ["a.jpg", "b.jpg", "c.jpg"]
        x = 0.75 * W_DEF   # 規約範例：應得 90 度
        hs = {"HOTSPOTOFROOM": [hotspot_room("a.jpg", 0, [("b.jpg", x)])]}
        house = house_without_gt("H1", images)
        with temp_dir() as d:
            write_json(os.path.join(d, "H1_HOTSPOT.json"), hs)
            out, msg = gt_hotspot.augment_house_with_gt(house, d)

        self.assertIsNotNone(msg)
        self.assertTrue(out["has_gt"])
        self.assertEqual(out["n_gt_connected_pairs"], 1)
        pair_ab = next(p for p in out["pairs"] if p["i"] == 0 and p["j"] == 1)
        self.assertTrue(pair_ab["gt_connected"])
        self.assertAlmostEqual(pair_ab["gt_az_ij"], 90.0, places=2)
        self.assertIsNone(pair_ab["gt_az_ji"])   # b 沒有登記回連
        pair_ac = next(p for p in out["pairs"] if p["i"] == 0 and p["j"] == 2)
        self.assertFalse(pair_ac["gt_connected"])

    def test_completely_mismatched_uuids_warn_and_produce_no_fake_gt(self):
        """UUID 完全不符（實際發生過：房號差一位數字，兩邊影像 UUID 前綴完全不同）：
        必須在回報訊息明說，且不能讓任何一對顯示看似真實的 GT。"""
        images = ["08ddb590-aaa.jpg", "08ddb590-bbb.jpg", "08ddb590-ccc.jpg"]
        hs = {"HOTSPOTOFROOM": [hotspot_room("08ddb5bb-xxx.jpg", 0,
                                             [("08ddb5bb-yyy.jpg", 100.0)])]}
        house = house_without_gt("H2", images)
        with temp_dir() as d:
            write_json(os.path.join(d, "wrong_house_HOTSPOT.json"), hs)
            out, msg = gt_hotspot.augment_house_with_gt(house, d)

        self.assertIsNotNone(msg)
        self.assertIn("放錯房", msg, "訊息須包含可辨識的警告字樣，msg=%r" % msg)
        self.assertFalse(out["has_gt"], "完全對不上不能讓 has_gt 變成 True（會顯示假的 GT連通 0）")
        self.assertEqual(out["n_gt_connected_pairs"], 0)
        self.assertTrue(all(not p["gt_connected"] for p in out["pairs"]))
        self.assertTrue(all(p["gt_az_ij"] is None and p["gt_az_ji"] is None for p in out["pairs"]))

    def test_partial_match_reports_actual_matched_count(self):
        """部分相符（例如只帶了一層樓的標註）：回報要講出實際對上幾張，不能只說「不完整」。"""
        images = ["p1.jpg", "p2.jpg", "p3.jpg"]
        hs = {"HOTSPOTOFROOM": [
            hotspot_room("p1.jpg", 0, [("p2.jpg", 0.0)]),      # p1,p2 屬於本房
            hotspot_room("q1.jpg", 1, [("q2.jpg", 0.0)]),      # q1,q2 不屬於本房
        ]}
        house = house_without_gt("H3", images)
        with temp_dir() as d:
            write_json(os.path.join(d, "partial_HOTSPOT.json"), hs)
            out, msg = gt_hotspot.augment_house_with_gt(house, d)

        self.assertIn("4", msg)   # HOTSPOT 共提到 4 張 (p1,p2,q1,q2)
        self.assertIn("2", msg)   # 對上 2 張 (p1,p2)
        self.assertTrue(out["has_gt"], "有部分有效，不同於完全對不上")
        self.assertEqual(out["n_gt_connected_pairs"], 1)

    def test_existing_gt_fields_are_not_overwritten(self):
        """房子資料已有 GT 欄位：原樣回傳，即使 bundle 內帶著會算出不同結果的 HOTSPOT。"""
        images = ["a.jpg", "b.jpg"]
        house = {
            "house": "H4", "n_images": 2, "width": W_DEF, "height": H_DEF,
            "images": images, "doors": [], "has_gt": True, "n_gt_connected_pairs": 1,
            "pairs": [{"i": 0, "j": 1, "prob": 0.9, "gt_connected": True,
                      "gt_az_ij": 12.34, "gt_az_ji": 199.99}],
        }
        hs = {"HOTSPOTOFROOM": [hotspot_room("a.jpg", 0, [("b.jpg", 999.0)])]}  # 會算出不同的值
        with temp_dir() as d:
            write_json(os.path.join(d, "H4_HOTSPOT.json"), hs)
            out, msg = gt_hotspot.augment_house_with_gt(house, d)

        self.assertIsNone(msg)
        self.assertIs(out, house, "已有 GT 欄位時必須原樣回傳同一個物件，不覆寫、不複製")
        self.assertEqual(out["pairs"][0]["gt_az_ij"], 12.34)

    def test_no_hotspot_in_bundle_is_unchanged(self):
        """包內沒有 HOTSPOT：行為與本功能實作之前相同。原樣回傳，不印訊息。"""
        images = ["a.jpg", "b.jpg"]
        house = house_without_gt("H5", images)
        with temp_dir() as d:
            out, msg = gt_hotspot.augment_house_with_gt(house, d)
        self.assertIsNone(msg)
        self.assertIs(out, house)

    def test_multi_floor_files_are_order_independent(self):
        """多樓層兩個檔：建立順序相反的兩份，斷言結果相同。"""
        floor1 = {"HOTSPOTOFROOM": [hotspot_room("stair.jpg", 0, [("up.jpg", 100.0)])]}
        floor2 = {"HOTSPOTOFROOM": [hotspot_room("stair.jpg", 1, [("up.jpg", 5000.0)])]}
        images = ["stair.jpg", "up.jpg"]

        with temp_dir() as d1:
            write_json(os.path.join(d1, "floor1_HOTSPOT.json"), floor1)
            write_json(os.path.join(d1, "floor2_HOTSPOT.json"), floor2)
            out1, _ = gt_hotspot.augment_house_with_gt(house_without_gt("H6", images), d1)

        with temp_dir() as d2:
            write_json(os.path.join(d2, "floor2_HOTSPOT.json"), floor2)
            write_json(os.path.join(d2, "floor1_HOTSPOT.json"), floor1)
            out2, _ = gt_hotspot.augment_house_with_gt(house_without_gt("H6", images), d2)

        self.assertEqual(out1["pairs"], out2["pairs"])

    def test_same_room_images_keep_independent_connections(self):
        """Roomidx 相同表示同房；但連通判定完全看各自的 ToIDName，同房不代表連通關係
        會互相取代或合併。每張圖各自的鄰接清單必須維持獨立。"""
        images = ["ra.jpg", "rb.jpg", "rc.jpg", "rd.jpg"]
        hs = {"HOTSPOTOFROOM": [
            hotspot_room("ra.jpg", 0, [("rc.jpg", 0.0)]),   # ra、rb 同房 (Roomidx=0)
            hotspot_room("rb.jpg", 0, [("rd.jpg", 0.0)]),   # 但各自連到不同的第三張圖
        ]}
        house = house_without_gt("H7", images)
        with temp_dir() as d:
            write_json(os.path.join(d, "room_HOTSPOT.json"), hs)
            out, _ = gt_hotspot.augment_house_with_gt(house, d)

        def connected(a, b):
            ia, ib = images.index(a), images.index(b)
            i, j = min(ia, ib), max(ia, ib)
            return next(p for p in out["pairs"] if p["i"] == i and p["j"] == j)["gt_connected"]

        self.assertTrue(connected("ra.jpg", "rc.jpg"))
        self.assertTrue(connected("rb.jpg", "rd.jpg"))
        self.assertFalse(connected("ra.jpg", "rd.jpg"), "ra 從未連到 rd，同房不該憑空產生連通")
        self.assertFalse(connected("rb.jpg", "rc.jpg"), "rb 從未連到 rc，同房不該憑空產生連通")


if __name__ == "__main__":
    unittest.main(verbosity=2)
