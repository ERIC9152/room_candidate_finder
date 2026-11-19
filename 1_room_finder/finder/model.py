# -*- coding: utf-8 -*-
"""Review state without any UI: candidates under the current thresholds, decisions, groups."""
import json
import os

from . import config
from .grouping import Decisions, build_groups, export_groups
from .scores import Calibration, active_wedge, load_scores, passes


class ReviewModel:
    def __init__(self, scan_dir, reviewer=""):
        self.scan_dir = os.path.abspath(scan_dir)
        with open(os.path.join(scan_dir, "manifest.json"), encoding="utf-8") as fh:
            man = json.load(fh)
        self.commit = man.get("geoverify_commit", "")
        root = man.get("root", "")
        self.root = root if os.path.isabs(root) else os.path.normpath(os.path.join(self.scan_dir, root))
        self.root_abs = man.get("root_abs", "")
        self.houses = {h: {p["id"]: p for p in items} for h, items in man["houses"].items()}
        self.scores = load_scores(scan_dir)
        self.decisions = Decisions(os.path.join(scan_dir, "decisions.jsonl"), reviewer)
        self.cal = Calibration()
        # door-aware wedge is available when finder.openings has written open_scores.jsonl
        self.n_open = sum(1 for pairs in self.scores.values() for s in pairs.values() if s.get("open"))
        self.has_open = self.n_open > 0
        self.min_inliers = config.DEFAULT_MIN_INLIERS
        self.set_mode("open" if self.has_open else "plain")

    def set_mode(self, mode):
        """'plain' = GeoVerify wedge, 'open' = door-aware wedge; resets the wedge threshold to the mode default."""
        if mode == "open" and not self.has_open:
            mode = "plain"
        self.mode = mode
        self.wedge_deg = config.DEFAULT_OPEN_WEDGE_DEG if mode == "open" else config.DEFAULT_WEDGE_DEG

    # ---- thresholds / candidates
    def thresholds(self):
        return dict(mode=self.mode, wedge_deg=self.wedge_deg, min_inliers=self.min_inliers)

    def candidates(self, house, hide_decided=False):
        """Pairs passing the current thresholds, strongest (widest narrower wedge) first."""
        out = []
        for (a, b), s in self.scores.get(house, {}).items():
            if not passes(s, self.wedge_deg, self.min_inliers, self.mode):
                continue
            rel = self.decisions.get(house, a, b)
            if hide_decided and rel is not None:
                continue
            out.append(dict(a=a, b=b, score=s, relation=rel, wedge=active_wedge(s, self.mode) or 0))
        out.sort(key=lambda c: (-c["wedge"], -c["score"]["n_inlier"], c["a"], c["b"]))
        return out

    def scanned_pairs(self, house):
        return len(self.scores.get(house, {}))

    def expected_pairs(self, house):
        n = len(self.houses.get(house, {}))
        return n * (n - 1) // 2

    def panos_without_candidates(self, house):
        linked = set()
        for c in self.candidates(house):
            linked.update((c["a"], c["b"]))
        return sorted(set(self.houses.get(house, {})) - linked)

    def expected_quality(self):
        return self.cal.expected(self.wedge_deg, self.mode)

    # ---- decisions / groups
    def decide(self, house, a, b, relation, note=""):
        s = self.scores.get(house, {}).get((a, b)) or self.scores.get(house, {}).get((b, a))
        return self.decisions.set(house, a, b, relation, note, score=s, thresholds=self.thresholds())

    def groups(self, house):
        return build_groups(house, list(self.houses.get(house, {})), self.decisions)

    def progress(self, house):
        cands = self.candidates(house)
        return dict(candidates=len(cands), decided=sum(1 for c in cands if c["relation"] is not None))

    def export(self, path):
        return export_groups(path, {h: list(v) for h, v in self.houses.items()}, self.decisions,
                             self.thresholds(), self.commit)

    def pair_detail(self, house, a, b):
        """GeoVerify result of one pair for the inlier overlay, or None.
        Looks in pairs_inliers/ (recomputed with inliers, tools/recompute_inliers.py) first, then pairs/.
        Returns the JSON with coordinates swapped when the file was stored as b__a."""
        for sub in ("pairs_inliers", "pairs"):
            for x, y, swap in ((a, b, False), (b, a, True)):
                fp = os.path.join(self.scan_dir, sub, house, "%s__%s.json" % (x, y))
                if os.path.exists(fp):
                    with open(fp, encoding="utf-8") as fh:
                        g = json.load(fh)
                    g["_source"] = sub
                    return _swap(g) if swap else g
        return None

    def opening_mask_path(self, house, pid):
        """SAM3 opening mask of one panorama (finder.openings), or None."""
        fp = os.path.join(self.scan_dir, "openings", house, pid + "_mask.png")
        return fp if os.path.exists(fp) else None

    def image_path(self, house, pid):
        """Relative paths resolve against the dataset root (portable); falls back to the
        absolute root recorded at scan time."""
        p = self.houses[house][pid]["path"]
        if os.path.isabs(p):
            return p
        cand = os.path.join(self.root, p)
        if not os.path.exists(cand) and self.root_abs:
            alt = os.path.join(self.root_abs, p)
            if os.path.exists(alt):
                return alt
        return cand


def _swap(g):
    """Exchange the A and B sides of a verify_pair result (only the fields the overlay uses)."""
    g = dict(g)
    g["img_size_a"], g["img_size_b"] = g.get("img_size_b"), g.get("img_size_a")
    g["inliers"] = [[p[2], p[3], p[0], p[1]] + list(p[4:]) for p in (g.get("inliers") or [])]
    g["regions"] = [dict(r, a=r.get("b"), b=r.get("a")) for r in (g.get("regions") or [])]
    return g
