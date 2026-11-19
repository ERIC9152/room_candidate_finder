# -*- coding: utf-8 -*-
"""蒙恬同環與其他連通：楔形、門兩種方法都跑一遍。"""
import argparse
import csv
import json
import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from roma_geo.config import Settings, datasets_root, regions_root
from roma_geo.names import find_image_file
from roma_geo.semantic import load_doors_json
from roma_geo.same_room.door import decides_same_room as door_decides
from roma_geo.same_room.geometry import geometry_for, load_named
from roma_geo.same_room.kept import is_hallway, is_outdoor, load_groups, pair_names
from roma_geo.same_room.wedge import _has_pose, _inliers, decides_same_room as wedge_decides

HERE = os.path.dirname(os.path.abspath(__file__))


def default_kept():
    return os.path.join(HERE, "mengtian.json")


def load_summary(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _key(house, a, b):
    return (house, frozenset((a, b)))


def split_ring(ring_keys, summary):
    connected = set()
    known = set()
    unconnected = set()
    for row in summary:
        key = _key(row["house"], row["img_a"], row["img_b"])
        known.add(key)
        if str(row.get("gt_label", "")).strip() == "1":
            connected.add(key)
        else:
            unconnected.add(key)
    return {
        "ring_connected": ring_keys & connected,
        "other_connected": connected - ring_keys,
        "ring_unconnected": ring_keys & unconnected,
        "ring_missing": ring_keys - known,
    }


def load_door_maps(directory):
    maps = {}
    if not directory or not os.path.isdir(directory):
        return maps
    for name in os.listdir(directory):
        if not name.endswith(".json"):
            continue
        house = name[:-5]
        maps[house] = load_doors_json(os.path.join(directory, name))
    return maps


def doors_of(maps, house, stem):
    return list((maps.get(house) or {}).get(stem) or [])


def has_semantic(maps, house):
    return house in maps


def doors_for_geometry(geometry, maps, house, stem_a, stem_b):
    a = (geometry or {}).get("img_a") or stem_a
    b = (geometry or {}).get("img_b") or stem_b
    return doors_of(maps, house, a), doors_of(maps, house, b)


def score_geometry(geometry, doors_a, doors_b):
    quality = (geometry or {}).get("quality") or {}
    a = quality.get("az_wedge_a_deg")
    b = quality.get("az_wedge_b_deg")
    doors_a = doors_a or []
    doors_b = doors_b or []
    return {
        "wedge": wedge_decides(geometry or {}),
        "door": door_decides((geometry or {}).get("regions") or [], doors_a, doors_b),
        "has_pose": _has_pose(geometry or {}),
        "n_inlier": _inliers(geometry or {}),
        "wedge_a": None if a is None else float(a),
        "wedge_b": None if b is None else float(b),
    }


def _median(values):
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    mid = (len(xs) - 1) / 2.0
    lo = int(mid)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] * (1.0 - (mid - lo)) + xs[hi] * (mid - lo)


def _narrower(row):
    a, b = row.get("wedge_a"), row.get("wedge_b")
    if a is None or b is None:
        return None
    return min(a, b)


def slice_stats(rows):
    n = len(rows)
    if n == 0:
        return {
            "n": 0, "pose_rate": None, "inlier_median": None, "wedge_median": None,
            "wedge_rate": None, "door_rate": None,
        }
    pose_n = sum(1 for row in rows if row.get("has_pose"))
    wedge_n = sum(1 for row in rows if row.get("wedge"))
    door_n = sum(1 for row in rows if row.get("door"))
    return {
        "n": n,
        "pose_rate": pose_n / n,
        "inlier_median": _median([row.get("n_inlier") for row in rows]),
        "wedge_median": _median([_narrower(row) for row in rows]),
        "wedge_rate": wedge_n / n,
        "door_rate": door_n / n,
    }


def _pct(value):
    if value is None:
        return "-"
    return "%.1f%%" % (value * 100)


def _num(value):
    if value is None:
        return "-"
    if isinstance(value, float) and abs(value - round(value)) > 0.05:
        return "%.1f" % value
    return "%g" % round(value)


def _wedge_line(title, stats):
    return "| %s | %d | %s | %s | %s | %s |" % (
        title, stats["n"], _pct(stats["pose_rate"]),
        _num(stats["inlier_median"]), _num(stats["wedge_median"]),
        _pct(stats["wedge_rate"]),
    )


def _door_line(title, stats):
    return "| %s | %d | %s | %s |" % (
        title, stats["n"], _pct(stats["wedge_rate"]), _pct(stats["door_rate"]),
    )


def render(wedge_sections, door_sections, notes):
    lines = [
        "# 蒙恬同環與其他連通",
        "",
        "表格只看總表裡標連通的 pair。",
        "同環是同一房間名底下、且在這批連通裡的配對。",
        "其他連通是這批連通裡、不在同環的配對。",
        "",
        "## 楔形",
        "",
        "有位姿、內點至少 30、兩側楔形都至少 60°。",
        "",
        "| | 對數 | 有位姿 | 內點中位 | 較窄側楔形中位 | 楔形命中 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for title, stats in wedge_sections:
        lines.append(_wedge_line(title, stats))
    lines.extend([
        "",
        "## 門",
        "",
        "只看有語意圖的房子。",
        "過濾掉某一側有八成以上落在門裡的共視後，還留得下一段才算同房。",
        "",
        "| | 對數 | 楔形命中 | 門命中 |",
        "|---|---:|---:|---:|",
    ])
    for title, stats in door_sections:
        lines.append(_door_line(title, stats))
    lines.append("")
    for note in notes:
        lines.append(note)
        lines.append("")
    lines.extend([
        "## 總結",
        "",
        "同環的定義比較多元。",
        "開放廚房加客廳，標註會切成兩個區域，實際看過去卻是同一個大空間。",
        "現有資料集也沒有合適的 GT。",
        "",
        "楔形比較能抓共視夠寬的近距離 pair。",
        "但距離一拉遠，共視區就會變窄、門檻也就過不了。",
        "走廊、戶外就是這種例子。",
        "適用於近距離、不靠門語意時。",
        "",
        "門會把落在門裡的共視拿掉，剩下的仍然可以是走廊、開放區、戶外。",
        "廣義上來講處於同一個空間，但狹義上來講，要處理真正在同一個房間的狀況準確率會較低。",
        "適用於要抓廣義同一空間時。",
        "",
    ])
    return "\n".join(lines)


def _stems(key):
    house, ids = key
    a, b = sorted(ids)
    return house, a, b


def find_image(datasets, house, stem, max_depth=5):
    cur = os.path.join(datasets, house)
    for _ in range(max_depth + 1):
        if not os.path.isdir(cur):
            return None
        found = find_image_file(cur, stem)
        if found:
            return found
        cur = os.path.join(cur, house)
    return None


def geometry_of(house, stem_a, stem_b, regions_dir, datasets, produce):
    saved = load_named(regions_dir, house, stem_a, stem_b)
    if saved is not None:
        return saved
    img_a = find_image(datasets, house, stem_a)
    img_b = find_image(datasets, house, stem_b)
    if not (img_a and img_b):
        return None
    try:
        return geometry_for(img_a, img_b, regions_dir=regions_dir, produce=produce)
    except (FileNotFoundError, OSError):
        return None


def score_key(key, names, maps, regions_dir, datasets, produce):
    house, stem_a, stem_b = _stems(key)
    mapped = has_semantic(maps, house)
    geometry = geometry_of(house, stem_a, stem_b, regions_dir, datasets, produce)
    if geometry is None:
        return {
            "wedge": False, "door": False, "has_pose": False, "n_inlier": 0,
            "wedge_a": None, "wedge_b": None, "has_semantic": mapped,
            "missing": True, "name": names.get(key),
        }
    doors_a, doors_b = doors_for_geometry(geometry, maps, house, stem_a, stem_b)
    row = score_geometry(geometry, doors_a, doors_b)
    row["missing"] = False
    row["has_semantic"] = mapped
    row["name"] = names.get(key)
    return row


def _by_name(keys, names, pred):
    return {key for key in keys if pred(names.get(key) or "")}


def _other_by_name(keys, relation, pred):
    chosen = set()
    for house, ids in keys:
        a, b = tuple(ids)
        left = (relation.get(house) or {}).get(a)
        right = (relation.get(house) or {}).get(b)
        if left and right and pred(left) and pred(right):
            chosen.add((house, ids))
    return chosen


def load_names(datasets):
    names = {}
    if not datasets or not os.path.isdir(datasets):
        return names
    for house in sorted(os.listdir(datasets)):
        house_dir = os.path.join(datasets, house)
        if not os.path.isdir(house_dir):
            continue
        inner = os.path.join(house_dir, house)
        base = inner if os.path.isdir(inner) else house_dir
        path = os.path.join(base, "relation.json")
        if not os.path.isfile(path):
            continue
        try:
            data = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        house_names = {}
        for pano in data.get("panos") or []:
            image_id = os.path.splitext(str(pano.get("id", "")))[0]
            name = (pano.get("name") or "").strip()
            if image_id and name:
                house_names[image_id] = name
        if house_names:
            names[house] = house_names
    return names


def _skip_produce(img_a, img_b):
    raise FileNotFoundError("沒有這一對的幾何結果")


def build(kept_root, summary_path, datasets, regions_dir, semantic_dir, produce):
    groups = load_groups(kept_root)
    names = pair_names(groups)
    ring_keys = set(names)
    split = split_ring(ring_keys, load_summary(summary_path))
    maps = load_door_maps(semantic_dir)
    relation = load_names(datasets)

    def scored(keys):
        rows = []
        for key in sorted(keys, key=lambda item: (item[0], tuple(sorted(item[1])))):
            rows.append(score_key(key, names, maps, regions_dir, datasets, produce))
        return rows

    ring = scored(split["ring_connected"])
    other = scored(split["other_connected"])
    hallway_ring = scored(_by_name(split["ring_connected"], names, is_hallway))
    outdoor_ring = scored(_by_name(split["ring_connected"], names, is_outdoor))
    indoor_ring = scored(_by_name(
        split["ring_connected"], names,
        lambda name: not is_hallway(name) and not is_outdoor(name),
    ))
    hallway_other = scored(_other_by_name(split["other_connected"], relation, is_hallway))
    outdoor_other = scored(_other_by_name(split["other_connected"], relation, is_outdoor))

    def mapped(rows):
        return [row for row in rows if row.get("has_semantic")]

    missing_n = sum(1 for row in ring if row.get("missing"))
    all_connected = ring + other
    notes = [
        "只看總表裡標連通的 pair，共 %d 對。其中同環 %d 對，其他連通 %d 對。"
        % (len(all_connected), len(ring), len(other)),
        "走廊只認名稱含 Hallway 或走廊。戶外認 yard、street、deck、road、driveway、porch、patio、dt out、陽台、戶外這類字。Tool Shed、attic、Garage 沒有這些字，留在其餘室內。",
    ]
    if missing_n:
        notes.append("連通同環裡有 %d 對沒讀到幾何結果。" % missing_n)
    wedge_sections = [
        ("全部連通", slice_stats(all_connected)),
        ("同環", slice_stats(ring)),
        ("其他連通", slice_stats(other)),
        ("同環走廊", slice_stats(hallway_ring)),
        ("同環戶外", slice_stats(outdoor_ring)),
        ("同環其餘室內", slice_stats(indoor_ring)),
        ("其他連通，兩端都是走廊", slice_stats(hallway_other)),
        ("其他連通，兩端都是戶外", slice_stats(outdoor_other)),
    ]
    door_sections = [
        ("全部連通", slice_stats(mapped(all_connected))),
        ("同環", slice_stats(mapped(ring))),
        ("其他連通", slice_stats(mapped(other))),
        ("同環走廊", slice_stats(mapped(hallway_ring))),
        ("同環戶外", slice_stats(mapped(outdoor_ring))),
        ("同環其餘室內", slice_stats(mapped(indoor_ring))),
        ("其他連通，兩端都是走廊", slice_stats(mapped(hallway_other))),
        ("其他連通，兩端都是戶外", slice_stats(mapped(outdoor_other))),
    ]
    return render(wedge_sections, door_sections, notes)


def main(argv=None):
    settings = Settings()
    parser = argparse.ArgumentParser(description="蒙恬同環：楔形與門兩種方法")
    parser.add_argument("--kept", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--regions", default=None)
    parser.add_argument("--semantic", default=None)
    parser.add_argument("--out", default=os.path.join(HERE, "mengtian_report.md"))
    parser.add_argument("--no-produce", action="store_true")
    args = parser.parse_args(argv)
    produce = _skip_produce if args.no_produce else None
    text = build(
        os.path.abspath(args.kept or default_kept()),
        os.path.abspath(args.summary or os.path.join(regions_root(), "_summary.csv")),
        os.path.abspath(args.datasets or datasets_root()),
        os.path.abspath(args.regions or settings.regions_root()),
        os.path.abspath(args.semantic or settings.semantic_out_dir()),
        produce,
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(args.out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
