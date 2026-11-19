# -*- coding: utf-8 -*-
"""Door-aware ("open") wedge from SAM3 opening masks.

    python -m finder.openings --scan scans\\mydata [--npz-dir DIR]     compute open_scores.jsonl
    python -m finder.openings --scan scans\\mydata --list-todo FILE    panoramas still without a mask

Masks: <scan>/openings/<house>/<panorama id>_mask.png (SAM3 "open doorway" run by Start_Openings.ps1; red or
white pixels = opening; any size, compared at OPEN_MASK_W x OPEN_MASK_H).
Inliers per pair: pairs_inliers/<house>/<a>__<b>.json, else pairs/<house>/<a>__<b>.json (new scans keep
them), else --npz-dir/<house>__<a>__<b>.npz (key "inl", "size"; from same_room_benchmark).
Rule: drop inliers inside the (dilated) opening mask of A or of B, recompute the 80% wedge on each side
from the rest. Output, one line per pair with pose and >= 30 inliers:
  {"house", "a", "b", "open": {"n_keep", "door_frac", "wedge_a", "wedge_b", "wedge_min"}}
Needs numpy (scan environment); the review app only reads the JSONL.
"""
import argparse
import json
import math
import os
import sys
import time

from . import config

MIN_INLIERS = 30


def wedge_width(az_deg, frac=0.8):
    """Narrowest arc (degrees) holding frac of the azimuths (same as GeoVerify's az_wedge)."""
    import numpy as np
    az = np.sort(np.mod(np.asarray(az_deg, dtype=float), 360.0))
    m = len(az)
    if m == 0:
        return 0.0
    k = max(1, int(math.ceil(frac * m)))
    ext = np.concatenate([az, az + 360.0])
    return float((ext[np.arange(m) + k - 1] - az).min())


def load_mask(path, w=config.OPEN_MASK_W, h=config.OPEN_MASK_H):
    """Opening mask as a bool array (h, w), or None when the file does not exist."""
    import numpy as np
    from PIL import Image
    if not os.path.exists(path):
        return None
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB").resize((w, h), Image.NEAREST))
    return a[..., 0] > 128


def dilate(m, r=config.OPEN_DILATE_PX):
    """Dilation exactly as calibrated in same_room_benchmark/scripts/eval_open_wedge.py: r px left/right
    (wrapping around the panorama seam), and cumulative vertical shifts 1..r, which reach
    r(r+1)/2 px up/down (10 px for r=4). Keep it this way or the calibration table no longer applies."""
    import numpy as np
    if r <= 0:
        return m
    out = m.copy()
    for dx in range(-r, r + 1):
        out |= np.roll(m, dx, axis=1)
    for dy in range(1, r + 1):
        out[dy:] |= out[:-dy].copy()
        out[:-dy] |= out[dy:].copy()
    return out


def open_wedge(inliers, size, mask_a, mask_b, n_inlier=None):
    """inliers: (n, >=4) x_a, y_a, x_b, y_b in original pixels; size: (Wa, Ha, Wb, Hb);
    masks: dilated bool arrays (h, w) or None (= no opening). n_inlier: full E-inlier count when
    `inliers` is a subsample (GeoVerify keeps at most 2000)."""
    import numpy as np
    p = np.asarray(inliers, dtype=float)
    n = len(p)
    if n == 0 or p.ndim != 2 or p.shape[1] < 4:
        return dict(n_keep=0, door_frac=0.0, wedge_a=0.0, wedge_b=0.0, wedge_min=0.0)
    Wa, Ha, Wb, Hb = [float(v) for v in size]
    thru = np.zeros(n, bool)
    for m, xi, yi, W, H in ((mask_a, 0, 1, Wa, Ha), (mask_b, 2, 3, Wb, Hb)):
        if m is None:
            continue
        mh, mw = m.shape
        x = np.clip((p[:, xi] / W * mw).astype(int), 0, mw - 1)
        y = np.clip((p[:, yi] / H * mh).astype(int), 0, mh - 1)
        thru |= m[y, x]
    keep = ~thru
    scale = (n_inlier / float(n)) if n_inlier else 1.0
    wa = wedge_width(p[keep, 0] / Wa * 360.0) if keep.sum() >= 2 else 0.0
    wb = wedge_width(p[keep, 2] / Wb * 360.0) if keep.sum() >= 2 else 0.0
    return dict(n_keep=int(round(keep.sum() * scale)), door_frac=round(float(thru.mean()), 4),
                wedge_a=round(wa, 1), wedge_b=round(wb, 1), wedge_min=round(min(wa, wb), 1))


def mask_path(scan_dir, house, pid):
    return os.path.join(scan_dir, "openings", house, pid + "_mask.png")


def _pair_inliers(scan_dir, house, a, b, npz_dir):
    """(inliers as list of [xa, ya, xb, yb, ...], size, n_inlier) or None. Handles files stored as b__a."""
    for sub in ("pairs_inliers", "pairs"):
        for x, y, swap in ((a, b, False), (b, a, True)):
            fp = os.path.join(scan_dir, sub, house, "%s__%s.json" % (x, y))
            if not os.path.exists(fp):
                continue
            with open(fp, encoding="utf-8") as fh:
                g = json.load(fh)
            pts = g.get("inliers") or []
            if not pts:
                continue
            sa, sb = g.get("img_size_a"), g.get("img_size_b")
            if swap:
                pts = [[q[2], q[3], q[0], q[1]] for q in pts]; sa, sb = sb, sa
            return pts, (sa[0], sa[1], sb[0], sb[1]), int((g.get("n") or {}).get("inlier") or len(pts))
    if npz_dir:
        import numpy as np
        for x, y, swap in ((a, b, False), (b, a, True)):
            fp = os.path.join(npz_dir, "%s__%s__%s.npz" % (house, x, y))
            if os.path.exists(fp):
                z = np.load(fp)
                pts, s = z["inl"], [float(v) for v in z["size"]]
                n = int(z["n_inlier"]) if "n_inlier" in z.files else len(pts)
                if swap:
                    pts = pts[:, [2, 3, 0, 1]]; s = [s[2], s[3], s[0], s[1]]
                return pts.tolist(), s, n
    return None


def compute(scan_dir, npz_dir=None, say=print):
    """Write <scan>/open_scores.jsonl for every pair that can pass the wedge rule. Returns counts."""
    from .scores import load_scores
    scores = load_scores(scan_dir)
    with open(os.path.join(scan_dir, "manifest.json"), encoding="utf-8") as fh:
        houses = json.load(fh)["houses"]
    n_mask = sum(os.path.exists(mask_path(scan_dir, h, p["id"])) for h, v in houses.items() for p in v)
    n_pano = sum(len(v) for v in houses.values())
    say("opening masks: %d / %d panoramas" % (n_mask, n_pano))
    out_fp = os.path.join(scan_dir, "open_scores.jsonl")
    cnt = dict(written=0, no_inliers=0, no_mask=0)
    with open(out_fp + ".tmp", "w", encoding="utf-8") as out:
        for house, pairs in scores.items():
            cache = {}

            def mk(pid):
                if pid not in cache:
                    m = load_mask(mask_path(scan_dir, house, pid))
                    cache[pid] = dilate(m) if m is not None else None
                return cache[pid]
            for (a, b), s in sorted(pairs.items()):
                if not s.get("has_pose") or (s.get("n_inlier") or 0) < MIN_INLIERS:
                    continue
                ma, mb = mk(a), mk(b)
                if ma is None or mb is None:
                    cnt["no_mask"] += 1; continue            # left out: the app falls back to the plain wedge
                got = _pair_inliers(scan_dir, house, a, b, npz_dir)
                if got is None:
                    cnt["no_inliers"] += 1; continue
                pts, size, n = got
                o = open_wedge(pts, size, ma, mb, n_inlier=s.get("n_inlier") or n)
                out.write(json.dumps(dict(house=house, a=a, b=b, open=o)) + "\n")
                cnt["written"] += 1
    os.replace(out_fp + ".tmp", out_fp)
    with open(os.path.join(scan_dir, "open_scores_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), prompt=config.OPEN_PROMPT,
                       mask_size=[config.OPEN_MASK_W, config.OPEN_MASK_H], dilate_px=config.OPEN_DILATE_PX,
                       masks=n_mask, panoramas=n_pano, **cnt), fh, ensure_ascii=False, indent=1)
    say("open_scores.jsonl: %(written)d pairs; skipped %(no_mask)d without masks, %(no_inliers)d without inliers" % cnt)
    return cnt


def list_todo(scan_dir, out_file):
    """Panoramas without an opening mask, one per line: house <TAB> id <TAB> absolute image path
    (input list for Start_Openings.ps1)."""
    from .model import ReviewModel
    m = ReviewModel(scan_dir)
    todo = [(h, pid, os.path.abspath(m.image_path(h, pid))) for h, items in m.houses.items() for pid in items
            if not os.path.exists(mask_path(scan_dir, h, pid))]
    with open(out_file, "w", encoding="utf-8") as fh:
        fh.write("".join("%s\t%s\t%s\n" % t for t in todo))
    print("panoramas without an opening mask: %d" % len(todo))
    return todo


def main(argv=None):
    ap = argparse.ArgumentParser(description="Door-aware wedge scores from SAM3 opening masks")
    ap.add_argument("--scan", required=True)
    ap.add_argument("--npz-dir", default="", help="extra inlier source: <house>__<a>__<b>.npz (benchmark import)")
    ap.add_argument("--list-todo", default="", help="write the panoramas without a mask to this file and exit")
    a = ap.parse_args(argv)
    if a.list_todo:
        list_todo(a.scan, a.list_todo); return
    compute(a.scan, a.npz_dir or None)


if __name__ == "__main__":
    main()
