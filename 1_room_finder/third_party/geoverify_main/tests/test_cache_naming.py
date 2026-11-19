"""
檔名約定與資料集探索測試。

RoMa 匹配快取以檔名攜帶 house 與兩張圖的識別碼，批次流程完全依賴此約定回推來源；
影像則以「房子資料夾可能同名巢狀任意層」的規則尋找。兩者都不是計算，出錯時的症狀
是整批略過或配到錯誤的影像。

另含 `HouseGT` 的載入順序：多樓層房子每層一個 HOTSPOT 檔，樓梯配對在相鄰兩層皆有
標註且座標不同。若不先排序，採用哪一筆取決於檔案系統的列舉順序，Linux 與 Windows
會給出不同的 GT。此處驗證其結果與建檔順序無關。
"""
import json
import os
import unittest

from _support import env_override, temp_dir, write_json  # noqa: F401（匯入時補 sys.path）

from pose_regions import find_house_image, parse_cache_name
from pose_regions.batch_pose_from_cache import HouseGT
from pose_regions.estimate_relative_pose import CACHE_SUFFIX

UUID_A = "08ddb590-a6df-409d-850f-e45f59efa7a0"
UUID_B = "08ddb590-ad9f-4647-8dab-ccfd0b14c8da"
HOUSE = "DollhouseTask_58553"


class TestParseCacheName(unittest.TestCase):
    def test_round_trip(self):
        name = "%s_%s_%s%s" % (HOUSE, UUID_A, UUID_B, CACHE_SUFFIX)
        self.assertEqual(parse_cache_name(name), (HOUSE, UUID_A, UUID_B))

    def test_accepts_full_path(self):
        p = os.path.join("some", "dir", "%s_%s_%s%s" % (HOUSE, UUID_A, UUID_B, CACHE_SUFFIX))
        self.assertEqual(parse_cache_name(p), (HOUSE, UUID_A, UUID_B))

    def test_house_name_may_contain_underscores(self):
        house = "DollhouseTask_58472_NoOutdoor"
        name = "%s_%s_%s%s" % (house, UUID_A, UUID_B, CACHE_SUFFIX)
        self.assertEqual(parse_cache_name(name), (house, UUID_A, UUID_B))

    def test_rejects_wrong_suffix(self):
        with self.assertRaises(SystemExit):
            parse_cache_name("%s_%s_%s_other_mode.npz" % (HOUSE, UUID_A, UUID_B))

    def test_parses_plain_image_stems(self):
        name = "%s_imgA_imgB%s" % (HOUSE, CACHE_SUFFIX)
        self.assertEqual(parse_cache_name(name), (HOUSE, "imgA", "imgB"))

    def test_suffix_encodes_the_matching_parameters(self):
        """檔名後綴帶著模式與參數；換參數即換後綴，舊快取不會被誤讀為新結果。"""
        for token in ("direct", "roma_v2", "std0.1", "face512", ".npz"):
            self.assertIn(token, CACHE_SUFFIX)


class TestFindHouseImage(unittest.TestCase):
    def _make(self, root, depth, name="img"):
        cur = root
        for _ in range(depth):
            cur = os.path.join(cur, HOUSE)
        os.makedirs(cur, exist_ok=True)
        p = os.path.join(cur, name + ".jpg")
        with open(p, "wb"):
            pass
        return p

    def test_finds_at_each_nesting_depth(self):
        for depth in (1, 2, 3):
            with temp_dir() as d:
                target = self._make(d, depth)
                with env_override(ROMA_GEO_DATASETS=d):
                    self.assertEqual(os.path.normcase(find_house_image(HOUSE, "img") or ""),
                                     os.path.normcase(target), "depth=%d" % depth)

    def test_returns_none_when_absent(self):
        with temp_dir() as d:
            self._make(d, 2)
            with env_override(ROMA_GEO_DATASETS=d):
                self.assertIsNone(find_house_image(HOUSE, "missing"))

    def test_stops_descending_once_a_level_has_images(self):
        """該層已有影像卻沒有這一張 ⇒ 不再往下找，避免撈到別層的同名檔。"""
        with temp_dir() as d:
            self._make(d, 1, name="other")
            deeper = self._make(d, 2, name="img")
            with env_override(ROMA_GEO_DATASETS=d):
                self.assertIsNone(find_house_image(HOUSE, "img"))
            self.assertTrue(os.path.exists(deeper))

    def test_unknown_house_returns_none(self):
        with temp_dir() as d, env_override(ROMA_GEO_DATASETS=d):
            self.assertIsNone(find_house_image("NoSuchHouse", "img"))


def hotspot(owner, targets, coords, roomidx):
    return {"HOTSPOTOFROOM": [{
        "IDName": owner + ".jpg",
        "Roomidx": roomidx,
        "ToIDName": {"IDName": [t + ".jpg" for t in targets]},
        "HSLoc": {"Coordinate": [{"x": c, "y": 100} for c in coords]},
    }]}


class TestHouseGT(unittest.TestCase):
    """GT 載入：連通標籤、方位、同房判定。"""

    def _house_dir(self, root):
        d = os.path.join(root, HOUSE)
        os.makedirs(d, exist_ok=True)
        return d

    def _fake_image(self, d, name, width=8192):
        """建立可被 PIL 讀出尺寸的小圖；寬度用於方位換算，以等比例位置表示。"""
        from PIL import Image
        p = os.path.join(d, name + ".jpg")
        Image.new("RGB", (width, width // 2), (0, 0, 0)).save(p)
        return p

    def test_label_and_azimuth(self):
        with temp_dir() as root:
            d = self._house_dir(root)
            img_a = self._fake_image(d, "A")
            self._fake_image(d, "B")
            # x = 3/4 寬 ⇒ 方位 +90°（規約：0° = 影像中央）。
            write_json(os.path.join(d, "floor1_HOTSPOT.json"),
                       hotspot("A", ["B"], [8192 * 0.75], 1))
            gt = HouseGT()
            lab, az_ab, az_ba, same = gt.get(HOUSE, "A", "B", img_a)
            self.assertEqual(lab, 1)
            self.assertAlmostEqual(az_ab, 90.0, places=3)
            self.assertIsNone(az_ba, "B 未標註回 A ⇒ 反向方位不存在")

    def test_unlisted_pair_is_explicitly_disconnected(self):
        with temp_dir() as root:
            d = self._house_dir(root)
            img_a = self._fake_image(d, "A")
            self._fake_image(d, "C")
            write_json(os.path.join(d, "floor1_HOTSPOT.json"),
                       hotspot("A", ["B"], [4096.0], 1))
            gt = HouseGT()
            self.assertEqual(gt.get(HOUSE, "A", "C", img_a)[0], 0)

    def test_house_without_hotspot_returns_unknown(self):
        """完全沒有 HOTSPOT 檔的房子回 None，不可把「不知道」報成「不連通」。"""
        with temp_dir() as root:
            d = self._house_dir(root)
            img_a = self._fake_image(d, "A")
            gt = HouseGT()
            lab, az_ab, az_ba, same = gt.get(HOUSE, "A", "B", img_a)
            self.assertIsNone(lab)
            self.assertIsNone(az_ab)
            self.assertIsNone(same)

    def test_label_is_symmetric(self):
        with temp_dir() as root:
            d = self._house_dir(root)
            img_a = self._fake_image(d, "A")
            self._fake_image(d, "B")
            write_json(os.path.join(d, "f_HOTSPOT.json"), hotspot("A", ["B"], [4096.0], 1))
            gt = HouseGT()
            self.assertEqual(gt.get(HOUSE, "B", "A", img_a)[0], 1)

    def test_same_room_detection(self):
        with temp_dir() as root:
            d = self._house_dir(root)
            img_a = self._fake_image(d, "A")
            self._fake_image(d, "B")
            data = hotspot("A", ["B"], [4096.0], 7)
            data["HOTSPOTOFROOM"].append({
                "IDName": "B.jpg", "Roomidx": 7,
                "ToIDName": {"IDName": ["A.jpg"]},
                "HSLoc": {"Coordinate": [{"x": 2048.0, "y": 100}]}})
            write_json(os.path.join(d, "f_HOTSPOT.json"), data)
            self.assertTrue(HouseGT().get(HOUSE, "A", "B", img_a)[3])

    def test_multi_floor_result_is_order_independent(self):
        """多樓層：同一對在兩層都有標註時，結果不得取決於檔案建立順序。

        `_load` 以排序後的檔名逐一讀取，因此兩種建檔順序須得到相同的 GT。
        """
        def build(order):
            root = temp_dir().__enter__()
            d = self._house_dir(root)
            img_a = self._fake_image(d, "A")
            self._fake_image(d, "B")
            files = {"floor1_HOTSPOT.json": hotspot("A", ["B"], [8192 * 0.25], 1),
                     "floor2_HOTSPOT.json": hotspot("A", ["B"], [8192 * 0.75], 2)}
            for fn in order:
                write_json(os.path.join(d, fn), files[fn])
            return HouseGT().get(HOUSE, "A", "B", img_a), root

        forward, r1 = build(["floor1_HOTSPOT.json", "floor2_HOTSPOT.json"])
        reverse, r2 = build(["floor2_HOTSPOT.json", "floor1_HOTSPOT.json"])
        self.assertEqual(forward[0], reverse[0])
        self.assertAlmostEqual(forward[1], reverse[1], places=6,
                               msg="多樓層 GT 方位隨建檔順序改變")
        self.assertEqual(forward[3], reverse[3])
        import shutil
        for r in (r1, r2):
            shutil.rmtree(r, ignore_errors=True)

    def test_corrupt_hotspot_file_is_skipped(self):
        with temp_dir() as root:
            d = self._house_dir(root)
            img_a = self._fake_image(d, "A")
            self._fake_image(d, "B")
            with open(os.path.join(d, "broken_HOTSPOT.json"), "w", encoding="utf-8") as f:
                f.write("{ not json")
            write_json(os.path.join(d, "good_HOTSPOT.json"), hotspot("A", ["B"], [4096.0], 1))
            self.assertEqual(HouseGT().get(HOUSE, "A", "B", img_a)[0], 1)

    def test_image_size_is_cached(self):
        with temp_dir() as root:
            d = self._house_dir(root)
            p = self._fake_image(d, "A", width=5376)
            gt = HouseGT()
            self.assertEqual(gt.img_size(p), (5376, 2688))
            self.assertIn(p, gt.sizes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
