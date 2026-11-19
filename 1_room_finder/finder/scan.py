# -*- coding: utf-8 -*-
"""Scan: run GeoVerify geometric verification on every panorama pair of every house.

    python -m finder.scan --root data\\mydata --out scans\\mydata [--house NAME] [--limit N]

Needs the scan environment (torch + romav2, GPU). Resumable: pairs already in scores.jsonl
are skipped. Writes into --out only:
  manifest.json            houses, panoramas (paths relative to the dataset root), skipped files
  scores.jsonl             one line per pair: house, a, b, score summary
  pairs/<house>/<a>__<b>.json   full verify_pair result (regions, pose, quality, inliers)
  status.json, scan.log, roma_cache/
"""
import argparse
import json
import os
import sys
import time
import traceback

from . import config, dataset
from .scores import summarize


def _import_geoverify():
    main = config.GEOVERIFY_MAIN
    if not os.path.exists(os.path.join(main, "SOURCE_COMMIT.txt")):
        sys.exit("GeoVerify code not found at %s (set RCF_GEOVERIFY_MAIN)" % main)
    sys.path[:0] = [os.path.join(main, "src"), main]
    import roma_geo
    if not os.path.abspath(roma_geo.__file__).startswith(os.path.abspath(main)):
        sys.exit("wrong roma_geo imported: " + roma_geo.__file__)
    return roma_geo, open(os.path.join(main, "SOURCE_COMMIT.txt")).read().strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Scan panorama pairs with GeoVerify (GPU).")
    ap.add_argument("--root", required=True, help="folder with one sub-folder per house")
    ap.add_argument("--out", required=True, help="output folder for this scan")
    ap.add_argument("--house", action="append", help="only these houses (repeatable)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N new pairs (smoke test)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    lock = os.path.join(a.out, "RUNNING.lock")
    if os.path.exists(lock):
        sys.exit("Another scan may be running (%s). If not, delete RUNNING.lock." % open(lock).read())
    open(lock, "w").write("pid=%d started=%s" % (os.getpid(), time.strftime("%Y-%m-%dT%H:%M:%S")))
    log = open(os.path.join(a.out, "scan.log"), "a", encoding="utf-8")

    def say(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True); log.write(line + "\n"); log.flush()
    try:
        houses, skipped = dataset.scan_houses(a.root)
        if a.house:
            houses = {h: v for h, v in houses.items() if h in a.house}
        total = sum(len(v) * (len(v) - 1) // 2 for v in houses.values())
        roma_geo, commit = _import_geoverify()
        root_abs = os.path.abspath(a.root)
        try:
            root_ref = os.path.relpath(root_abs, os.path.abspath(a.out))   # portable when both move together
        except ValueError:                                                # different drives
            root_ref = root_abs
        portable = {h: [dict(p, path=os.path.relpath(p["path"], root_abs)) for p in v] for h, v in houses.items()}
        manifest = dict(root=root_ref, root_abs=root_abs, geoverify_commit=commit, mode="direct",
                        params=roma_geo.DEFAULTS, houses=portable, skipped=skipped, total_pairs=total,
                        created=time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=1)
        say("houses %d, panoramas %d, pairs %d, skipped files %d" % (len(houses), sum(map(len, houses.values())), total, len(skipped)))
        score_fp = os.path.join(a.out, "scores.jsonl")
        done = set()
        if os.path.exists(score_fp):
            for line in open(score_fp, encoding="utf-8"):
                if line.strip():
                    j = json.loads(line); done.add((j["house"], j["a"], j["b"]))
        cache = os.path.join(a.out, "roma_cache"); os.makedirs(cache, exist_ok=True)
        matcher = roma_geo.get_matcher("roma_v2", None, True, cache)
        new = errors = 0; t0 = time.time(); idx = 0
        with open(score_fp, "a", encoding="utf-8") as out:
            for house, items in houses.items():
                pdir = os.path.join(a.out, "pairs", house); os.makedirs(pdir, exist_ok=True)
                for pa, pb in dataset.all_pairs(items):
                    idx += 1
                    if (house, pa["id"], pb["id"]) in done:
                        continue
                    try:
                        g = roma_geo.verify_pair(pa["path"], pb["path"], matcher=matcher, mode="direct",
                                                 house=house, keep_inliers=True)   # inliers kept for the review overlay (~100 KB per pair)
                        g["img_a"], g["img_b"] = pa["id"], pb["id"]
                        fp = os.path.join(pdir, "%s__%s.json" % (pa["id"], pb["id"]))
                        with open(fp + ".tmp", "w", encoding="utf-8") as fh:
                            json.dump(g, fh, ensure_ascii=False, default=float)
                        os.replace(fp + ".tmp", fp)
                        out.write(json.dumps(dict(house=house, a=pa["id"], b=pb["id"], score=summarize(g))) + "\n"); out.flush()
                        new += 1
                    except Exception as e:
                        errors += 1
                        with open(os.path.join(a.out, "errors.jsonl"), "a", encoding="utf-8") as fh:
                            fh.write(json.dumps(dict(house=house, a=pa["id"], b=pb["id"], error=repr(e),
                                                     trace=traceback.format_exc()[-1500:])) + "\n")
                    if new and new % 25 == 0:
                        rate = (time.time() - t0) / new
                        st = dict(state="running", index=idx, total=total, new=new, errors=errors,
                                  s_per_pair=round(rate, 2), eta_h=round(rate * (total - idx) / 3600, 2))
                        with open(os.path.join(a.out, "status.json"), "w") as fh:
                            json.dump(st, fh, indent=1)
                        say("[%d/%d] new %d err %d %.2fs/pair eta %.1fh" % (idx, total, new, errors, rate, st["eta_h"]))
                    if a.limit and new >= a.limit:
                        say("limit reached"); break
                if a.limit and new >= a.limit:
                    break
        state = "complete" if not a.limit and idx >= total else "partial"
        with open(os.path.join(a.out, "status.json"), "w") as fh:
            json.dump(dict(state=state, total=total, new=new, errors=errors, finished=time.strftime("%Y-%m-%dT%H:%M:%S")), fh, indent=1)
        say("done: %s, new %d, errors %d" % (state, new, errors))
    finally:
        log.close()
        if os.path.exists(lock):
            os.remove(lock)


if __name__ == "__main__":
    main()
