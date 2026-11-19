# -*- coding: utf-8 -*-
"""Copy SAM3 opening masks into <scan>/openings/<house>/<id>_mask.png (shrunk to
OPEN_MASK_W x OPEN_MASK_H, 1-bit, a few KB each, so the scan folder stays portable). Resumable.

    python tools/import_openings.py --scan scans\\dollhouse --src <sam3-pano-seg output folder>
        [--src ...more folders] [--budget-s 0]

A panorama's mask is taken from the first --src folder that has <house>__<id>_mask.png (the naming used
by Start_Openings.ps1) or <id>_mask.png (panorama ids unique across houses, e.g. Dollhouse UUIDs).
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from PIL import Image
from finder import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--src", action="append", required=True)
    ap.add_argument("--budget-s", type=float, default=0, help="stop after this many seconds (0 = no limit)")
    a = ap.parse_args()
    with open(os.path.join(a.scan, "manifest.json"), encoding="utf-8") as fh:
        ids = [(h, p["id"]) for h, v in json.load(fh)["houses"].items() for p in v]
    out = os.path.join(a.scan, "openings"); os.makedirs(out, exist_ok=True)
    t0, new, have, missing = time.time(), 0, 0, []
    for house, pid in ids:
        os.makedirs(os.path.join(out, house), exist_ok=True)
        dst = os.path.join(out, house, pid + "_mask.png")
        if os.path.exists(dst):
            have += 1; continue
        if a.budget_s and time.time() - t0 > a.budget_s:
            break
        names = ("%s__%s_mask.png" % (house, pid), pid + "_mask.png")
        src = next((os.path.join(d, n) for d in a.src for n in names if os.path.exists(os.path.join(d, n))), None)
        if src is None:
            missing.append(pid); continue
        with Image.open(src) as im:
            r = im.convert("RGB").resize((config.OPEN_MASK_W, config.OPEN_MASK_H), Image.NEAREST).getchannel("R")
        r.point(lambda v: 255 if v > 128 else 0).convert("1").save(dst + ".tmp.png")
        os.replace(dst + ".tmp.png", dst)
        new += 1
    print("panoramas %d: already %d, imported %d, without a source mask %d" % (len(ids), have, new, len(missing)))


if __name__ == "__main__":
    main()
