# -*- coding: utf-8 -*-
"""Pair scores from the scan, the wedge classifier, and the calibration lookup."""
import json
import os

from . import config


def summarize(geometry):
    """verify_pair result -> the few numbers the classifier and the UI need."""
    q = geometry.get("quality") or {}
    flags = geometry.get("flags") or {}
    n = geometry.get("n") or {}
    pose = geometry.get("pose") or {}
    has_pose = (not flags.get("no_covisibility")) and pose.get("az_ab_deg") not in (None, "")
    wa, wb = q.get("az_wedge_a_deg"), q.get("az_wedge_b_deg")
    return dict(has_pose=bool(has_pose), n_inlier=int(n.get("inlier") or 0),
                inlier_rate=geometry.get("inlier_rate"),
                wedge_a=wa, wedge_b=wb,
                wedge_min=min(wa, wb) if wa is not None and wb is not None else None,
                n_regions=len(geometry.get("regions") or []))


MODES = ("plain", "open")
RULE_TEXT = {
    "plain": "GeoVerify wedge (has pose, E inliers >= min_inliers, both wedges >= wedge_deg)",
    "open": "GeoVerify door-aware wedge (has pose, E inliers >= min_inliers; inliers inside a SAM3 'open doorway' "
            "mask of A or B removed; remaining inliers >= min_inliers and both recomputed wedges >= wedge_deg; "
            "pairs without opening masks use the plain wedge)",
}


def uses_open(s, mode):
    """True when this pair is judged with the door-aware wedge (open mode and open scores present)."""
    return mode == "open" and bool(s.get("open"))


def active_wedge(s, mode="plain"):
    """The narrower wedge the classifier compares in this mode."""
    return s["open"].get("wedge_min") if uses_open(s, mode) else s.get("wedge_min")


def passes(s, wedge_deg=config.DEFAULT_WEDGE_DEG, min_inliers=config.DEFAULT_MIN_INLIERS, mode="plain"):
    """plain: GeoVerify wedge classifier -- pose, inliers >= min_inliers, both wedges >= wedge_deg.
    open: additionally the inliers left after removing those seen through an opening must number
    >= min_inliers, and their recomputed wedges are used. Pairs without open scores use plain."""
    if not (bool(s.get("has_pose")) and (s.get("n_inlier") or 0) >= min_inliers):
        return False
    if uses_open(s, mode):
        o = s["open"]
        return (o.get("n_keep") or 0) >= min_inliers and (o.get("wedge_min") or 0) >= wedge_deg
    return s.get("wedge_min") is not None and s["wedge_min"] >= wedge_deg


def load_scores(scan_dir):
    """Read scores.jsonl (latest line per pair wins). Returns {house: {(a, b): summary}}."""
    out = {}
    fp = os.path.join(scan_dir, "scores.jsonl")
    if not os.path.exists(fp):
        return out
    with open(fp, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            j = json.loads(line)
            out.setdefault(j["house"], {})[(j["a"], j["b"])] = j["score"]
    ofp = os.path.join(scan_dir, "open_scores.jsonl")        # door-aware wedge (finder.openings)
    if os.path.exists(ofp):
        with open(ofp, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                j = json.loads(line)
                s = out.get(j["house"], {}).get((j["a"], j["b"]))
                if s is not None:
                    s["open"] = j["open"]
    return out


class Calibration:
    """Expected P/R for a slider position, from the Dollhouse benchmark (nearest table row).
    Tables: by_wedge_deg (plain wedge) and by_open_wedge_deg (door-aware wedge)."""

    def __init__(self, path=config.CALIBRATION_FILE):
        self.tables = {"plain": {}, "open": {}}
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            for mode, key in (("plain", "by_wedge_deg"), ("open", "by_open_wedge_deg")):
                self.tables[mode] = {float(k): v for k, v in data.get(key, {}).items()}
        except (OSError, ValueError):
            pass

    @property
    def rows(self):
        return self.tables["plain"]

    def expected(self, wedge_deg, mode="plain"):
        rows = self.tables.get(mode) or {}
        if not rows:
            return None
        k = min(rows, key=lambda d: abs(d - wedge_deg))
        r = rows[k]
        return dict(at=k, P=float(r["P"]), R=float(r["R"]))
