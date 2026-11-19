# -*- coding: utf-8 -*-
"""把門語意判同環、標註卻跨房的 pair，畫出共識區域。"""
import argparse
import csv
import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ROOT = os.path.abspath(os.path.join(_SRC, ".."))
for path in (_SRC, _ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from roma_geo.config import Settings
from roma_geo.same_room.door import decides_same_room as door_decides, kept_regions
from roma_geo.same_room.geometry import existing_path, load_named
from roma_geo.same_room.mengtian import (
    doors_for_geometry,
    has_semantic,
    load_door_maps,
    load_summary,
)


def overlay_root(regions_dir):
    return os.path.join(regions_dir, "overlays", "door_true_cross_room")


def cross_room_door_hits(summary, maps, regions_dir):
    """連通、標註跨房、房子有語意圖、門語意仍判同房。"""
    hits = []
    for row in summary:
        if str(row.get("gt_label", "")).strip() != "1":
            continue
        if str(row.get("same_room", "")).strip() != "0":
            continue
        house = row["house"]
        if not has_semantic(maps, house):
            continue
        stem_a, stem_b = row["img_a"], row["img_b"]
        json_path = existing_path(regions_dir, house, stem_a, stem_b)
        if json_path is None:
            continue
        geometry = load_named(regions_dir, house, stem_a, stem_b)
        doors_a, doors_b = doors_for_geometry(geometry, maps, house, stem_a, stem_b)
        regions = (geometry or {}).get("regions") or []
        if not door_decides(regions, doors_a, doors_b):
            continue
        hits.append({
            "house": house,
            "img_a": stem_a,
            "img_b": stem_b,
            "json_path": json_path,
            "geometry": geometry,
            "doors_a": doors_a,
            "doors_b": doors_b,
            "kept": kept_regions(regions, doors_a, doors_b),
        })
    return hits


def _render_pair():
    from pose_regions.render_region_overlay import render_pair
    return render_pair


def draw_hits(hits, out_dir, render=None, target_w=2048):
    os.makedirs(out_dir, exist_ok=True)
    render = render or _render_pair()
    saved = []
    for hit in hits:
        geometry = dict(hit["geometry"] or {})
        geometry["regions"] = list(hit["kept"])
        try:
            path = render(hit["json_path"], out_dir, target_w, False, geometry)
        except SystemExit as err:
            print("skip %s: %s" % (os.path.basename(hit["json_path"]), err))
            continue
        saved.append(path)
    return saved


def write_index(hits, out_dir):
    path = os.path.join(out_dir, "_pairs.csv")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["house", "img_a", "img_b", "n_regions", "n_kept"])
        for hit in hits:
            geometry = hit.get("geometry") or {}
            writer.writerow([
                hit["house"], hit["img_a"], hit["img_b"],
                len(geometry.get("regions") or []),
                len(hit.get("kept") or []),
            ])
    return path


def main(argv=None):
    settings = Settings()
    parser = argparse.ArgumentParser(description="畫出門語意同環、標註跨房的共識區域")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--regions", default=None)
    parser.add_argument("--semantic", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args(argv)
    regions_dir = os.path.abspath(args.regions or settings.regions_root())
    summary = load_summary(os.path.abspath(args.summary or os.path.join(regions_dir, "_summary.csv")))
    maps = load_door_maps(os.path.abspath(args.semantic or settings.semantic_out_dir()))
    hits = cross_room_door_hits(summary, maps, regions_dir)
    if args.limit:
        hits = hits[:args.limit]
    out_dir = os.path.abspath(args.out or overlay_root(regions_dir))
    write_index(hits, out_dir)
    print("%d 對" % len(hits))
    print(os.path.join(out_dir, "_pairs.csv"))
    if args.index_only:
        return 0
    for path in draw_hits(hits, out_dir, target_w=args.width):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
