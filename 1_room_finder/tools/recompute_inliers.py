# -*- coding: utf-8 -*-
"""Re-run GeoVerify (GPU) on candidate pairs of an existing scan, this time keeping the per-point inliers,
so the review app can mark them. Writes scans/<dataset>/pairs_inliers/<house>/<a>__<b>.json only
(scores.jsonl, decisions and pairs/ are not changed). Resumable: existing files are skipped.

    python tools/recompute_inliers.py --scan scans\\dollhouse [--house NAME ...] [--min-wedge 15] [--min-inliers 30] [--limit N]
    python tools/recompute_inliers.py --scan scans\\dollhouse --groups <room_groups.json or room_classification.json> [--groups ...]
    python tools/recompute_inliers.py --scan scans\\dollhouse --reviewed

Default: pairs that can appear in the review list at --min-wedge / --min-inliers (about 3 s per pair).
--groups: every pair inside the room groups of those files, whatever its wedge (houses are matched by panorama id).
--reviewed: every pair with a decision in decisions.jsonl.
Each new result is compared with the stored score (inlier count and wedges); differences are listed in
pairs_inliers/consistency.jsonl.
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from finder.model import ReviewModel
from finder.scores import passes, summarize
from finder.scan import _import_geoverify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--house", action="append")
    ap.add_argument("--min-wedge", type=float, default=15.0)
    ap.add_argument("--min-inliers", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--groups", action="append", help="room groups JSON; recompute every pair inside its groups")
    ap.add_argument("--reviewed", action="store_true", help="recompute every pair that has a decision")
    a = ap.parse_args()
    m = ReviewModel(a.scan)
    house_of = {pid: h for h, items in m.houses.items() for pid in items}
    wanted = None                                   # None = threshold mode
    if a.groups or a.reviewed:
        wanted, unknown = set(), 0
        for gf in a.groups or []:
            with open(gf, encoding="utf-8-sig") as fh:
                for g in json.load(fh)["groups"]:
                    ids = sorted(g["panorama_uuids"])
                    for i in range(len(ids)):
                        for j in range(i + 1, len(ids)):
                            h = house_of.get(ids[i])
                            if h is None or house_of.get(ids[j]) != h:
                                unknown += 1; continue
                            wanted.add((h, ids[i], ids[j]))
        if a.reviewed:
            wanted |= {(h, x, y) for (h, x, y), e in m.decisions.latest.items()}
        if unknown:
            print("pairs whose panoramas are not in this scan (skipped): %d" % unknown)
    todo, not_scored = [], 0
    for house, pairs in m.scores.items():
        if a.house and house not in a.house:
            continue
        for (x, y), s in pairs.items():
            fp = os.path.join(a.scan, "pairs_inliers", house, "%s__%s.json" % (x, y))
            hit = passes(s, a.min_wedge, a.min_inliers) if wanted is None else \
                ((house, x, y) in wanted or (house, y, x) in wanted)
            if hit and not os.path.exists(fp):
                todo.append((house, x, y, s, fp))
    if wanted is not None:
        scored = {(h, x, y) for h, p in m.scores.items() for (x, y) in p}
        not_scored = sum(1 for (h, x, y) in wanted if (h, x, y) not in scored and (h, y, x) not in scored)
        if not_scored:
            print("pairs never scanned (not recomputed; scan them first with Import_Dataset.ps1): %d" % not_scored)
    if a.limit:
        todo = todo[:a.limit]
    print("pairs to recompute: %d (%s)" % (len(todo), "pairs inside the given groups / reviewed pairs" if wanted is not None
          else "min wedge %.0f, min inliers %d" % (a.min_wedge, a.min_inliers)), flush=True)
    if not todo:
        return
    roma_geo, commit = _import_geoverify()
    cache = os.path.join(a.scan, "roma_cache"); os.makedirs(cache, exist_ok=True)
    matcher = roma_geo.get_matcher("roma_v2", None, True, cache)
    os.makedirs(os.path.join(a.scan, "pairs_inliers"), exist_ok=True)
    cons = open(os.path.join(a.scan, "pairs_inliers", "consistency.jsonl"), "a", encoding="utf-8")
    t0 = time.time()
    for i, (house, x, y, s, fp) in enumerate(todo, 1):
        try:
            g = roma_geo.verify_pair(m.image_path(house, x), m.image_path(house, y), matcher=matcher, mode="direct",
                                     house=house, keep_inliers=True)
        except Exception as e:
            print("  error %s %s %s: %r" % (house, x, y, e), flush=True); continue
        g["img_a"], g["img_b"], g["geoverify_commit"] = x, y, commit
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(g, fh, ensure_ascii=False, default=float)
        os.replace(fp + ".tmp", fp)
        n = summarize(g)
        same = n["n_inlier"] == s["n_inlier"] and all(
            (n[k] is None) == (s[k] is None) and (n[k] is None or abs(n[k] - s[k]) < 1.0) for k in ("wedge_a", "wedge_b"))
        cons.write(json.dumps(dict(house=house, a=x, b=y, same_as_scan=same, scan=s, now=n)) + "\n"); cons.flush()
        if i % 10 == 0 or i == len(todo):
            r = (time.time() - t0) / i
            print("[%d/%d] %.1fs/pair, eta %.1f min" % (i, len(todo), r, r * (len(todo) - i) / 60), flush=True)
    cons.close()
    print("done")


if __name__ == "__main__":
    main()
