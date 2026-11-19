#!/usr/bin/env python3
"""
同學 val 10 房的完整鏈方位誤差。

有幾何位姿時用 az_ij/az_ji（來自 R、t），不足時用 az_coarse_*（同學方向預測）。
只量有 GT hotspot 方位的方向。語意門矯正這輪不算（fallback 規則還沒接）。

用法:
  python pose_regions/val10_chain_eval.py
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from roma_geo.config import matrices_version_root, PROJECT_ROOT          # noqa: E402
from roma_geo.semantic import circ_err                                   # noqa: E402

SIDES = (
    ("ij", "az_ij", "az_coarse_ij", "gt_az_ij"),
    ("ji", "az_ji", "az_coarse_ji", "gt_az_ji"),
)
HOUSE_DATA_DEFAULT = os.path.join(PROJECT_ROOT, "stitch_tool", "data")


def _num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def list_matrix_houses(matrices_dir):
    """矩陣目錄裡有 .npy 的房子，排序後回傳。"""
    names = []
    for fn in os.listdir(matrices_dir):
        if fn.endswith(".npy"):
            names.append(fn[:-4])
    return sorted(names)


def iter_gt_directions(pair):
    """有 GT 方位的每一側 → 幾何／同學／完整鏈誤差。"""
    source = pair.get("az_source")
    prob = _num(pair.get("prob"))
    for _side, az_k, coarse_k, gt_k in SIDES:
        gt = _num(pair.get(gt_k))
        if gt is None:
            continue
        combined = _num(pair.get(az_k))
        coarse = _num(pair.get(coarse_k))
        geo = combined if source == "geo" else None
        yield {
            "source": source,
            "prob": prob,
            "gt_connected": bool(pair.get("gt_connected")),
            "err_combined": None if combined is None else circ_err(combined, gt),
            "err_geo": None if geo is None else circ_err(geo, gt),
            "err_coarse": None if coarse is None else circ_err(coarse, gt),
        }


def summarize(rows, err_key="err_combined"):
    """n、中位、p90、≤22.5°／≤45°。空列表回 None。"""
    xs = [r[err_key] for r in rows if r.get(err_key) is not None]
    if not xs:
        return None
    arr = np.asarray(xs, dtype=float)
    n = len(arr)
    return {
        "n": n,
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "le22": 100.0 * float(np.mean(arr <= 22.5)),
        "le45": 100.0 * float(np.mean(arr <= 45.0)),
    }


def evaluate_houses(matrices_dir, data_dir):
    """只讀矩陣目錄裡的房子。回傳覆蓋率與分層誤差。"""
    houses = list_matrix_houses(matrices_dir)
    rows = []
    n_pairs = n_gt_pairs = n_gt_geo = n_gt_coarse = 0
    missing = []
    per_house = []
    for house in houses:
        path = os.path.join(data_dir, house + ".json")
        if not os.path.isfile(path):
            missing.append(house)
            continue
        data = json.load(open(path, encoding="utf-8"))
        house_rows = []
        gt_pairs = geo_pairs = coarse_pairs = 0
        for pair in data.get("pairs") or []:
            n_pairs += 1
            if pair.get("gt_connected"):
                n_gt_pairs += 1
                gt_pairs += 1
                if pair.get("az_source") == "geo":
                    n_gt_geo += 1
                    geo_pairs += 1
                else:
                    n_gt_coarse += 1
                    coarse_pairs += 1
            for row in iter_gt_directions(pair):
                row["house"] = house
                house_rows.append(row)
                rows.append(row)
        per_house.append({
            "house": house,
            "n_pairs": len(data.get("pairs") or []),
            "n_gt_pairs": gt_pairs,
            "n_gt_geo": geo_pairs,
            "n_gt_coarse": coarse_pairs,
            "combined": summarize(house_rows, "err_combined"),
        })
    p50 = [r for r in rows if r.get("prob") is not None and r["prob"] >= 0.5]
    return {
        "n_houses": len(houses) - len(missing),
        "houses": houses,
        "missing": missing,
        "n_pairs": n_pairs,
        "n_gt_pairs": n_gt_pairs,
        "n_gt_geo": n_gt_geo,
        "n_gt_coarse": n_gt_coarse,
        "combined": summarize(rows, "err_combined"),
        "geo": summarize([r for r in rows if r["source"] == "geo"], "err_geo"),
        "coarse": summarize([r for r in rows if r["source"] == "coarse"], "err_coarse"),
        "coarse_on_geo": summarize(
            [r for r in rows if r["source"] == "geo"], "err_coarse"),
        "combined_p50": summarize(p50, "err_combined"),
        "geo_p50": summarize([r for r in p50 if r["source"] == "geo"], "err_geo"),
        "coarse_p50": summarize([r for r in p50 if r["source"] == "coarse"], "err_coarse"),
        "per_house": per_house,
        "rows": rows,
    }


def _fmt(s):
    if not s:
        return "—"
    return "n=%d  中位 %.1f°  p90 %.1f°  ≤22.5° %.1f%%  ≤45° %.1f%%" % (
        s["n"], s["median"], s["p90"], s["le22"], s["le45"])


def _print_report(out):
    print("房子 %d 間（矩陣 %s）" % (out["n_houses"], ", ".join(out["houses"])))
    if out["missing"]:
        print("缺 house JSON：%s" % ", ".join(out["missing"]))
    print("配對 %d　其中 GT 連通 %d（geo %d / 同學 fallback %d）"
          % (out["n_pairs"], out["n_gt_pairs"], out["n_gt_geo"], out["n_gt_coarse"]))
    print()
    print("全部有 GT 方位的方向")
    print("  完整鏈（geo 否則同學）  ", _fmt(out["combined"]))
    print("  只看 geo 那條          ", _fmt(out["geo"]))
    print("  只看同學 fallback      ", _fmt(out["coarse"]))
    print("  同學方向（同一批 geo） ", _fmt(out["coarse_on_geo"]))
    print()
    print("prob ≥ 0.5（工具預設會顯示的對）")
    print("  完整鏈                  ", _fmt(out["combined_p50"]))
    print("  geo                     ", _fmt(out["geo_p50"]))
    print("  同學 fallback           ", _fmt(out["coarse_p50"]))
    print()
    print("分房（完整鏈）")
    for h in out["per_house"]:
        c = h["combined"]
        extra = _fmt(c) if c else "無 GT 方位"
        print("  %s  GT連通 %d（geo %d / fallback %d）  %s"
              % (h["house"], h["n_gt_pairs"], h["n_gt_geo"], h["n_gt_coarse"], extra))


def _write_csv(path, rows):
    if not rows:
        return
    keys = ["house", "source", "prob", "gt_connected",
            "err_combined", "err_geo", "err_coarse"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv=None):
    ap = argparse.ArgumentParser(description="10 房完整鏈方位誤差")
    ap.add_argument("--matrices", default=None)
    ap.add_argument("--data", default=HOUSE_DATA_DEFAULT)
    ap.add_argument("--csv", default=os.path.join(
        PROJECT_ROOT, "Output", "roma_regions", "_val10_chain.csv"))
    args = ap.parse_args(argv)
    mdir = args.matrices or matrices_version_root()
    out = evaluate_houses(mdir, args.data)
    _print_report(out)
    parent = os.path.dirname(os.path.abspath(args.csv))
    if parent:
        os.makedirs(parent, exist_ok=True)
    _write_csv(args.csv, out["rows"])
    print("\n明細 %s" % args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
