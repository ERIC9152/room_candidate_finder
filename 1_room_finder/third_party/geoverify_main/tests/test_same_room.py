# -*- coding: utf-8 -*-
"""同房的兩種判法，以及幾何結果先找產出、沒有再跑。"""
import json
import os
import unittest

from _support import env_override, temp_dir  # noqa: F401

from roma_geo.same_room import by_door, by_wedge
from roma_geo.same_room.arcs import covered_length
from roma_geo.same_room.door import decides_same_room as door_decides, kept_regions
from roma_geo.same_room.draw import cross_room_door_hits
from roma_geo.same_room.geometry import geometry_for, load_named, load_saved
from roma_geo.same_room.kept import dump_groups, is_hallway, is_outdoor, load_groups, pair_keys
from roma_geo.same_room.mengtian import (
    doors_for_geometry, doors_of, has_semantic, load_door_maps, render,
    score_geometry, slice_stats, split_ring,
)
from roma_geo.same_room.wedge import decides_same_room as wedge_decides


def _region(a_range, b_range):
    return {
        "a": {"az_range": list(a_range), "az_width_deg": a_range[1] - a_range[0]},
        "b": {"az_range": list(b_range), "az_width_deg": b_range[1] - b_range[0]},
    }


def _geometry(inlier=40, wedge_a=70.0, wedge_b=80.0, regions=None, pose=True):
    return {
        "house": "house",
        "img_a": "a",
        "img_b": "b",
        "n": {"inlier": inlier},
        "pose": {"az_ab_deg": 12.0} if pose else None,
        "quality": {"az_wedge_a_deg": wedge_a, "az_wedge_b_deg": wedge_b},
        "flags": {"no_covisibility": not pose},
        "regions": regions or [],
    }


class TestWedge(unittest.TestCase):
    def test_three_bars_all_met(self):
        self.assertTrue(wedge_decides(_geometry()))

    def test_narrow_side_fails(self):
        self.assertFalse(wedge_decides(_geometry(wedge_b=20.0)))

    def test_no_pose_fails(self):
        self.assertFalse(wedge_decides(_geometry(pose=False)))


class TestDoor(unittest.TestCase):
    def test_arc_across_the_seam(self):
        self.assertAlmostEqual(covered_length(170, 30, [(160, 40)]), 30.0, places=4)

    def test_one_side_mostly_inside_a_door_is_dropped(self):
        regions = [_region((0, 20), (100, 140))]
        doors_a = [{"az0": 0.0, "width_deg": 40.0}]
        self.assertFalse(door_decides(regions, doors_a, []))

    def test_a_region_left_after_filtering_counts(self):
        inside = _region((0, 20), (100, 140))
        outside = _region((80, 150), (10, 90))
        doors_a = [(0.0, 40.0)]
        self.assertTrue(door_decides([inside, outside], doors_a, []))
        self.assertEqual(kept_regions([inside, outside], doors_a, []), [outside])

    def test_no_region_left_is_not_same_room(self):
        self.assertFalse(door_decides([], [], []))

    def test_overlapping_doors_are_not_counted_twice(self):
        # 0–50 與 20–70 的聯集是 70，不是 100。
        self.assertAlmostEqual(covered_length(0, 100, [(0, 50), (20, 50)]), 70.0, places=4)

    def test_wrapped_az_range_is_a_short_arc_across_the_seam(self):
        regions = [_region((170, -170), (0, 40))]
        doors_a = [{"az0": 160.0, "width_deg": 40.0}]
        self.assertFalse(door_decides(regions, doors_a, []))


class TestGeometryLookup(unittest.TestCase):
    def test_saved_pair_is_used_and_produce_is_not_called(self):
        with temp_dir() as root:
            img_a = os.path.join(root, "house", "a.jpg")
            img_b = os.path.join(root, "house", "b.jpg")
            out = os.path.join(root, "roma_regions")
            os.makedirs(out)
            path = os.path.join(out, "house_a_b.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(_geometry(), fh)

            def produce(a, b):
                raise AssertionError("不該重跑")

            got = geometry_for(img_a, img_b, regions_dir=out, produce=produce)
        self.assertEqual(got["n"]["inlier"], 40)

    def test_missing_pair_runs_and_writes_back(self):
        with temp_dir() as root:
            img_a = os.path.join(root, "house", "a.jpg")
            img_b = os.path.join(root, "house", "b.jpg")
            out = os.path.join(root, "roma_regions")
            calls = []

            def produce(a, b):
                calls.append((a, b))
                return _geometry()

            geometry_for(img_a, img_b, regions_dir=out, produce=produce)
            saved = load_saved(out, img_a, img_b)
        self.assertEqual(len(calls), 1)
        self.assertEqual(saved["img_a"], "a")


class TestMethods(unittest.TestCase):
    def test_by_wedge_reads_the_saved_pair(self):
        with temp_dir() as root:
            img_a = os.path.join(root, "house", "a.jpg")
            img_b = os.path.join(root, "house", "b.jpg")
            out = os.path.join(root, "roma_regions")
            os.makedirs(out)
            with open(os.path.join(out, "house_a_b.json"), "w", encoding="utf-8") as fh:
                json.dump(_geometry(), fh)
            self.assertTrue(by_wedge(img_a, img_b, regions_dir=out, produce=lambda a, b: None))

    def test_by_door_uses_the_doors_of_each_image(self):
        with temp_dir() as root:
            img_a = os.path.join(root, "house", "a.jpg")
            img_b = os.path.join(root, "house", "b.jpg")
            out = os.path.join(root, "roma_regions")
            os.makedirs(out)
            regions = [_region((0, 20), (40, 100))]
            with open(os.path.join(out, "house_a_b.json"), "w", encoding="utf-8") as fh:
                json.dump(_geometry(regions=regions), fh)
            doors_a = [{"az0": 0.0, "width_deg": 40.0}]
            self.assertFalse(by_door(img_a, img_b, doors_a, [], regions_dir=out))

    def test_by_door_aligns_doors_when_the_cache_is_swapped(self):
        with temp_dir() as root:
            img_a = os.path.join(root, "house", "a.jpg")
            img_b = os.path.join(root, "house", "b.jpg")
            out = os.path.join(root, "roma_regions")
            os.makedirs(out)
            geometry = _geometry(regions=[_region((0, 20), (80, 150))])
            geometry["img_a"] = "b"
            geometry["img_b"] = "a"
            with open(os.path.join(out, "house_b_a.json"), "w", encoding="utf-8") as fh:
                json.dump(geometry, fh)
            doors_b = [{"az0": 0.0, "width_deg": 40.0}]
            self.assertFalse(by_door(img_a, img_b, [], doors_b, regions_dir=out,
                                     produce=lambda a, b: None))

    def test_named_lookup_finds_swapped_stems(self):
        with temp_dir() as root:
            out = os.path.join(root, "roma_regions")
            os.makedirs(out)
            with open(os.path.join(out, "house_b_a.json"), "w", encoding="utf-8") as fh:
                json.dump(_geometry(), fh)
            got = load_named(out, "house", "a", "b")
        self.assertEqual(got["img_a"], "a")


class TestKept(unittest.TestCase):
    def test_hallway_and_outdoor_labels(self):
        self.assertTrue(is_hallway("Hallway"))
        self.assertTrue(is_hallway("走廊"))
        self.assertFalse(is_hallway("Bathroom"))
        self.assertTrue(is_outdoor("Backyard"))
        self.assertTrue(is_outdoor("Street View"))
        self.assertFalse(is_outdoor("Bathroom upstairs"))

    def test_two_images_become_one_pair(self):
        with temp_dir() as root:
            room = os.path.join(root, "HouseA", "Hallway")
            os.makedirs(room)
            for name in ("aaa.jpg", "bbb.jpg"):
                open(os.path.join(room, name), "w").close()
            os.makedirs(os.path.join(root, "HouseB"))
            groups = load_groups(root)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["name"], "Hallway")
        self.assertEqual(pair_keys(groups), {("HouseA", frozenset(("aaa", "bbb")))})

    def test_json_lists_the_same_pairs_as_the_folder(self):
        with temp_dir() as root:
            room = os.path.join(root, "HouseA", "Hallway")
            os.makedirs(room)
            for name in ("aaa.jpg", "bbb.jpg"):
                open(os.path.join(room, name), "w").close()
            from_folder = load_groups(root)
            path = os.path.join(root, "kept.json")
            dump_groups(from_folder, path)
            from_json = load_groups(path)
        self.assertEqual(from_json, from_folder)
        self.assertEqual(pair_keys(from_json), {("HouseA", frozenset(("aaa", "bbb")))})

    def test_bundled_json_has_the_filtered_pairs(self):
        from roma_geo.same_room.mengtian import default_kept
        groups = load_groups(default_kept())
        self.assertEqual(len(pair_keys(groups)), 68)


class TestMengtian(unittest.TestCase):
    def test_ring_connected_and_other_connected_are_split(self):
        ring = {("H", frozenset(("a", "b"))), ("H", frozenset(("c", "d")))}
        summary = [
            {"house": "H", "img_a": "a", "img_b": "b", "gt_label": "1"},
            {"house": "H", "img_a": "a", "img_b": "x", "gt_label": "1"},
            {"house": "H", "img_a": "c", "img_b": "d", "gt_label": "0"},
        ]
        split = split_ring(ring, summary)
        self.assertEqual(split["ring_connected"], {("H", frozenset(("a", "b")))})
        self.assertEqual(split["other_connected"], {("H", frozenset(("a", "x")))})
        self.assertEqual(split["ring_unconnected"], {("H", frozenset(("c", "d")))})
        self.assertEqual(split["ring_missing"], set())

    def test_doors_follow_geometry_image_order(self):
        geometry = _geometry(regions=[_region((0, 20), (80, 150))])
        geometry["img_a"] = "left"
        geometry["img_b"] = "right"
        maps = {"H": {
            "left": [{"az0": 0.0, "width_deg": 40.0}],
            "right": [],
        }}
        doors_a, doors_b = doors_for_geometry(geometry, maps, "H", "right", "left")
        got = score_geometry(geometry, doors_a, doors_b)
        self.assertFalse(got["door"])

    def test_both_methods_are_scored_on_the_same_pair(self):
        geometry = _geometry(regions=[_region((80, 150), (10, 90))])
        doors_a = [{"az0": 0.0, "width_deg": 40.0}]
        got = score_geometry(geometry, doors_a, [])
        self.assertTrue(got["wedge"])
        self.assertTrue(got["door"])

    def test_region_inside_a_door_fails_door_but_can_still_pass_wedge(self):
        geometry = _geometry(regions=[_region((0, 20), (40, 100))])
        doors_a = [{"az0": 0.0, "width_deg": 40.0}]
        got = score_geometry(geometry, doors_a, [])
        self.assertTrue(got["wedge"])
        self.assertFalse(got["door"])

    def test_doors_come_from_each_image_in_the_house_map(self):
        with temp_dir() as root:
            path = os.path.join(root, "HouseA.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({
                    "house": "HouseA",
                    "doors": {"aaa": [{"az0": 10.0, "width_deg": 20.0}]},
                }, fh)
            maps = load_door_maps(root)
        self.assertEqual(doors_of(maps, "HouseA", "aaa")[0]["az0"], 10.0)
        self.assertEqual(doors_of(maps, "HouseA", "bbb"), [])

    def test_report_keeps_door_hits_on_their_own_table(self):
        wedge_rows = [
            {"wedge": True, "door": True, "has_pose": True, "n_inlier": 100,
             "wedge_a": 80.0, "wedge_b": 90.0},
            {"wedge": False, "door": True, "has_pose": True, "n_inlier": 40,
             "wedge_a": 20.0, "wedge_b": 70.0},
        ]
        door_rows = [wedge_rows[0]]
        text = render(
            [("全部連通", slice_stats(wedge_rows))],
            [("全部連通", slice_stats(door_rows))],
            ["只看總表裡標連通的 pair，共 1812 對。"],
        )
        wedge_part, door_part = text.split("## 門", 1)
        self.assertIn("楔形命中", wedge_part)
        self.assertNotIn("門命中", wedge_part)
        self.assertIn("門命中", door_part)
        self.assertIn("只看有語意圖的房子", door_part)
        self.assertIn("只看總表裡標連通的 pair", text)
        self.assertIn("全部連通", text)
        self.assertNotIn("未進總表", text)
        self.assertNotIn("資料夾", text)
        self.assertIn("同環的定義比較多元", text)
        self.assertIn("現有資料集也沒有合適的 GT", text)
        self.assertIn("適用於近距離、不靠門語意時", text)
        self.assertIn("適用於要抓廣義同一空間時", text)

    def test_house_with_a_map_counts_even_if_this_pair_has_no_door(self):
        maps = {"H": {"other": [{"az0": 0.0, "width_deg": 10.0}]}}
        self.assertTrue(has_semantic(maps, "H"))
        self.assertFalse(has_semantic(maps, "X"))
        self.assertEqual(doors_of(maps, "H", "aaa"), [])


class TestDrawCrossRoom(unittest.TestCase):
    def _write_pair(self, root, house, a, b, regions):
        out = os.path.join(root, "roma_regions")
        os.makedirs(out, exist_ok=True)
        geometry = _geometry(regions=regions)
        geometry["house"] = house
        geometry["img_a"] = a
        geometry["img_b"] = b
        path = os.path.join(out, "%s_%s_%s.json" % (house, a, b))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(geometry, fh)
        return out, path

    def test_lists_door_hits_that_the_label_calls_cross_room(self):
        with temp_dir() as root:
            regions = [_region((80, 150), (10, 90))]
            out, path = self._write_pair(root, "H", "a", "b", regions)
            summary = [
                {"house": "H", "img_a": "a", "img_b": "b", "gt_label": "1", "same_room": "0"},
                {"house": "H", "img_a": "a", "img_b": "c", "gt_label": "1", "same_room": "1"},
                {"house": "H", "img_a": "a", "img_b": "d", "gt_label": "0", "same_room": "0"},
            ]
            maps = {"H": {"a": [], "b": []}}
            hits = cross_room_door_hits(summary, maps, out)
        self.assertEqual([(h["house"], h["img_a"], h["img_b"]) for h in hits],
                         [("H", "a", "b")])
        self.assertEqual(hits[0]["json_path"], path)

    def test_a_region_only_inside_a_door_is_not_listed(self):
        with temp_dir() as root:
            regions = [_region((0, 20), (40, 100))]
            out, _ = self._write_pair(root, "H", "a", "b", regions)
            summary = [
                {"house": "H", "img_a": "a", "img_b": "b", "gt_label": "1", "same_room": "0"},
            ]
            maps = {"H": {"a": [{"az0": 0.0, "width_deg": 40.0}], "b": []}}
            hits = cross_room_door_hits(summary, maps, out)
        self.assertEqual(hits, [])

    def test_house_without_a_map_is_not_listed(self):
        with temp_dir() as root:
            regions = [_region((80, 150), (10, 90))]
            out, _ = self._write_pair(root, "H", "a", "b", regions)
            summary = [
                {"house": "H", "img_a": "a", "img_b": "b", "gt_label": "1", "same_room": "0"},
            ]
            hits = cross_room_door_hits(summary, {}, out)
        self.assertEqual(hits, [])

    def test_draw_paints_the_regions_left_after_the_door_filter(self):
        from roma_geo.same_room.draw import draw_hits
        inside = _region((0, 20), (40, 100))
        outside = _region((80, 150), (10, 90))
        geometry = _geometry(regions=[inside, outside])
        hit = {
            "house": "H", "img_a": "a", "img_b": "b",
            "json_path": "H_a_b.json",
            "geometry": geometry,
            "kept": [outside],
        }
        seen = []

        def render(json_path, out_dir, target_w, draw_az, geometry=None):
            seen.append((geometry.get("regions"), draw_az))
            return os.path.join(out_dir, "H_a_b.jpg")

        with temp_dir() as root:
            saved = draw_hits([hit], root, render=render)
        self.assertEqual(seen, [([outside], False)])
        self.assertTrue(saved[0].endswith("H_a_b.jpg"))
