#!/usr/bin/env python3
"""
廠商 ROI 精確度評估。

廠商拿到的 ROI 是共視區域（polygon / az_range）加上通道方位 az_ab。
GT hotspot 是 ERP 上的一點，只有直接相鄰配對才有。本腳本量三件事：

  1. az_ab 與 GT 的圓角誤差（指向準不準）
  2. GT 是否落在主共視區 az_range（框有沒有包到通道）
  3. 以 az_ab 為中心、固定窗寬裁切時能否包住 GT（下游若只用方位當裁切中心）

族群 = Output/roma_regions 的 per-pair JSON 裡，有位姿且有 GT 方位的配對。
無位姿或非直接相鄰（沒有 hotspot 方位）不進入分母。

用法:
  python pose_regions/roi_accuracy_eval.py
輸出:
  pose_regions/roi_accuracy_report.md
  Output/roma_regions/_roi_accuracy.csv
"""
import argparse
import csv
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from roma_geo.config import regions_root                                    # noqa: E402
from roma_geo.semantic import az_contains, circ_err, region_az_span         # noqa: E402

WINDOW_DEGS = (7.5, 15, 22.5, 45, 90, 180)
OUT_MD = os.path.join(SCRIPT_DIR, "roi_accuracy_report.md")


def _num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def score_pair(rec):
    """單一 per-pair JSON → 列指標。usable=False 時其餘欄位可不看。"""
    pose = rec.get("pose") or {}
    gt = rec.get("gt") or {}
    flags = rec.get("flags") or {}
    regions = rec.get("regions") or []
    az_ab = _num(pose.get("az_ab_deg"))
    az_ba = _num(pose.get("az_ba_deg"))
    gt_ab = _num(gt.get("az_ab_deg"))
    gt_ba = _num(gt.get("az_ba_deg"))
    usable = az_ab is not None and gt_ab is not None
    out = {
        "usable": usable,
        "same_room": bool(gt.get("same_room")),
        "doorway_wedge": bool(flags.get("doorway_wedge")),
        "conf_used": _num(rec.get("conf_used")),
        "err_az_ab": circ_err(az_ab, gt_ab) if usable else None,
        "err_az_ba": (circ_err(az_ba, gt_ba)
                      if usable and az_ba is not None and gt_ba is not None else None),
        "region_hit_a": None,
        "region_hit_b": None,
        "region_width_a": None,
        "region_width_b": None,
        "window_hit": {w: None for w in WINDOW_DEGS},
    }
    if not usable:
        return out
    if regions:
        span_a = region_az_span(regions[0].get("a") or {})
        span_b = region_az_span(regions[0].get("b") or {})
        if span_a:
            out["region_hit_a"] = az_contains(gt_ab, span_a[0], span_a[1])
            out["region_width_a"] = span_a[2]
        if span_b and gt_ba is not None:
            out["region_hit_b"] = az_contains(gt_ba, span_b[0], span_b[1])
            out["region_width_b"] = span_b[2]
    for w in WINDOW_DEGS:
        out["window_hit"][w] = circ_err(az_ab, gt_ab) <= w
    return out


def _pct(n, d):
    return 100.0 * n / d if d else 0.0


def _median(xs):
    if not xs:
        return None
    ys = sorted(xs)
    return ys[len(ys) // 2]


def _p90(xs):
    if not xs:
        return None
    ys = sorted(xs)
    i = min(len(ys) - 1, int(round(0.9 * (len(ys) - 1))))
    return ys[i]


def _stats(rows, key="err_az_ab"):
    xs = [r[key] for r in rows if r.get(key) is not None]
    hits = [r for r in rows if r.get("region_hit_a")]
    widths = [r["region_width_a"] for r in rows if r.get("region_width_a") is not None]
    hit_w = [r["region_width_a"] for r in hits if r.get("region_width_a") is not None]
    n = len(rows)
    n_hit = sum(1 for r in rows if r.get("region_hit_a"))
    n_reg = sum(1 for r in rows if r.get("region_hit_a") is not None)
    return {
        "n": n,
        "med": _median(xs),
        "p90": _p90(xs),
        "r15": _pct(sum(1 for r in rows if r.get("window_hit", {}).get(15)), n),
        "r22": _pct(sum(1 for r in rows if r.get("window_hit", {}).get(22.5)), n),
        "r45": _pct(sum(1 for r in rows if r.get("window_hit", {}).get(45)), n),
        "hit": _pct(n_hit, n_reg),
        "n_reg": n_reg,
        "w_med": _median(widths),
        "w_hit": _median(hit_w),
    }


def _fmt(v, nd=1):
    return "—" if v is None else ("%." + str(nd) + "f") % v


def _row(name, s):
    return "| %s | %d | %s | %s | %s | %s | %s | %s | %s |" % (
        name, s["n"], _fmt(s["med"]), _fmt(s["p90"]),
        _fmt(s["r22"]), _fmt(s["r45"]),
        _fmt(s["hit"]), _fmt(s["w_med"]), _fmt(s["w_hit"]))


def load_scores(regions_dir):
    rows = []
    files = sorted(
        os.path.join(regions_dir, f)
        for f in os.listdir(regions_dir)
        if f.endswith(".json") and not f.startswith("_"))
    for path in files:
        try:
            rec = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        s = score_pair(rec)
        if not s["usable"]:
            continue
        rows.append({
            "house": rec.get("house"),
            "img_a": rec.get("img_a"),
            "img_b": rec.get("img_b"),
            **s,
        })
    return rows


def write_report(rows, out_md, n_files):
    all_s = _stats(rows)
    same = [r for r in rows if r["same_room"]]
    cross = [r for r in rows if not r["same_room"]]
    door = [r for r in rows if r["doorway_wedge"]]
    c65 = [r for r in rows if r["conf_used"] == 0.65]
    c30 = [r for r in rows if r["conf_used"] == 0.3]
    lines = [
        "# 廠商 ROI 精確度",
        "",
        "產生腳本:`roi_accuracy_eval.py`。族群 = 有位姿且有 GT hotspot 方位的配對"
        "（直接相鄰）。ROI = 主共視區 `az_range`（交付檔的區域欄）。",
        "",
        "掃描 %d 個 per-pair JSON，可用 %d 對。" % (n_files, len(rows)),
        "",
        "## 1. 指向誤差與 ROI 包住率",
        "",
        "| 組 | n | az 誤差中位 | p90 | ≤22.5° | ≤45° | ROI 包住 GT | 主區寬中位 | 包住時寬中位 |",
        "|---|---|---|---|---|---|---|---|---|",
        _row("全部", all_s),
        _row("同房", _stats(same)),
        _row("跨房", _stats(cross)),
        _row("doorway 旗標", _stats(door)),
        _row("conf 0.65", _stats(c65)),
        _row("conf 0.3 撈回", _stats(c30)),
        "",
        "ROI 包住 = GT 方位落在主共視區 `az_range` 內。寬度單位為度。"
        "包住率高但寬度也大，代表框到了、但框很鬆。",
        "",
        "## 2. 只用通道方位當裁切中心",
        "",
        "假設下游以 `az_ab` 為中心裁一段固定寬的 ERP。窗內有 GT 才算命中。",
        "",
        "| 半窗寬 | 命中率 |",
        "|---|---|",
    ]
    n = len(rows)
    for w in WINDOW_DEGS:
        hit = sum(1 for r in rows if r["window_hit"].get(w))
        lines.append("| ±%.1f° | %.1f%% |" % (w, _pct(hit, n)))
    lines += [
        "",
        "## 讀法",
        "",
        "- 指向誤差對照既有 `batch_pose_report.md` 的 `err_az_*`；此表只收有 GT 方位的對。",
        "- 廠商若用區域多邊形當 ROI，看「ROI 包住 GT」。",
        "- 廠商若只用通道方位當門位，看第 2 表的窗寬。",
        "",
    ]
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote", out_md)


def write_csv(rows, path):
    fields = ["house", "img_a", "img_b", "same_room", "doorway_wedge", "conf_used",
              "err_az_ab", "err_az_ba", "region_hit_a", "region_hit_b",
              "region_width_a", "region_width_b"]
    fields += ["window_%g" % w for w in WINDOW_DEGS]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=fields)
        wri.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in fields if not k.startswith("window_")}
            for w in WINDOW_DEGS:
                row["window_%g" % w] = r["window_hit"].get(w)
            wri.writerow(row)
    print("wrote", path, "(%d rows)" % len(rows))


def main():
    ap = argparse.ArgumentParser(description="廠商 ROI 精確度：GT 是否落在共視區")
    ap.add_argument("--regions_dir", default=None)
    ap.add_argument("--out_md", default=OUT_MD)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()
    regions_dir = args.regions_dir or regions_root()
    n_files = sum(1 for f in os.listdir(regions_dir)
                  if f.endswith(".json") and not f.startswith("_"))
    rows = load_scores(regions_dir)
    out_csv = args.out_csv or os.path.join(regions_dir, "_roi_accuracy.csv")
    write_report(rows, args.out_md, n_files)
    write_csv(rows, out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
