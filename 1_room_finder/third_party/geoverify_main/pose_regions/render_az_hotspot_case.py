#!/usr/bin/env python3
"""
簡報案例：左右並排 ERP，標預測方位與 HOTSPOT，不畫 correspondence。

    python pose_regions/render_az_hotspot_case.py
"""
import argparse
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
SRC = os.path.join(ROOT, "src")
TOOL = os.path.join(ROOT, "stitch_tool")
for p in (SRC, TOOL, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import gt_hotspot  # noqa: E402
from roma_geo.semantic import circ_err  # noqa: E402

try:
    from .estimate_relative_pose import find_house_image
except ImportError:
    from estimate_relative_pose import find_house_image  # noqa: E402

PRED = (51, 119, 221)
HOTSPOT = (46, 160, 70)
BAND = (51, 119, 221, 55)
WHITE = (255, 255, 255)
INK = (34, 34, 34)
MUTED = (90, 90, 90)
HALF_DEG = 22.5
PANE_W = 960
CASES = (
    ("accurate", "DollhouseTask_58553", 6, 8, "很準"),
    ("median", "DollhouseTask_58553", 8, 9, "接近中位"),
    ("off", "DollhouseTask_58599_NoOutdoor", 18, 21, "偏差較大"),
)


def az_to_x(az_deg, width):
    """方位 → ERP x。規約 az = ((x/W)×360 − 180) mod 360。"""
    return ((float(az_deg) + 180.0) % 360.0) / 360.0 * float(width)


def band_x_ranges(az_deg, half_deg, width):
    """預測方位 ±half_deg 對應的像素區間。過縫時拆成兩段。"""
    w = float(width)
    x0 = az_to_x(float(az_deg) - float(half_deg), w)
    x1 = az_to_x(float(az_deg) + float(half_deg), w)
    if float(half_deg) >= 180.0:
        return [(0.0, w)]
    if x0 <= x1:
        return [(x0, x1)]
    return [(x0, w), (0.0, x1)]


def default_data_dir():
    packed = os.path.join(ROOT, "Output", "9-14新上游", "stitch_tool", "data")
    if os.path.isdir(packed):
        return packed
    return os.path.join(ROOT, "stitch_tool", "data")


def _font(size):
    for path in (
        r"C:\Windows\Fonts\msjh.ttc",
        r"C:\Windows\Fonts\msjhbd.ttc",
        "msjh.ttc",
        "arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _find_pair(house_data, i, j):
    for pair in house_data.get("pairs") or []:
        if pair.get("i") == i and pair.get("j") == j:
            return pair
        if pair.get("i") == j and pair.get("j") == i:
            return pair
    return None


def _hotspot_xy(locations, src, tgt):
    hit = (locations.get(src) or {}).get(tgt)
    if hit is None:
        return None
    return hit


def _draw_band(overlay, az_deg, orig_w, pane_w, pane_h):
    dr = ImageDraw.Draw(overlay)
    scale = pane_w / float(orig_w)
    for x0, x1 in band_x_ranges(az_deg, HALF_DEG, orig_w):
        dr.rectangle([x0 * scale, 0, x1 * scale, pane_h], fill=BAND)


def _draw_pred(img, az_deg, orig_w):
    w, h = img.size
    x = az_to_x(az_deg, orig_w) * (w / float(orig_w))
    dr = ImageDraw.Draw(img)
    dr.line([(x, 0), (x, h)], fill=PRED, width=3)
    tip = [(x, 0), (x - 10, 18), (x + 10, 18)]
    dr.polygon(tip, fill=PRED)
    dr.text((x + 14, 6), "預測", fill=PRED, font=_font(22))


def _draw_hotspot(img, xy, orig_size):
    if xy is None:
        return
    ow, oh = orig_size
    w, h = img.size
    x = xy[0] * w / float(ow)
    y = xy[1] * h / float(oh)
    box = 22
    dr = ImageDraw.Draw(img)
    dr.rectangle([x - box, y - box, x + box, y + box], outline=HOTSPOT, width=4)
    dr.ellipse([x - 4, y - 4, x + 4, y + 4], fill=HOTSPOT)
    dr.text((x + box + 6, y - 12), "HOTSPOT", fill=HOTSPOT, font=_font(20))


def render_pane(path, az_deg, hotspot_xy, pane_w=PANE_W):
    im = Image.open(path).convert("RGB")
    orig = im.size
    pane_h = int(round(orig[1] * pane_w / orig[0]))
    im = im.resize((pane_w, pane_h), Image.BILINEAR)
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    if az_deg is not None:
        _draw_band(overlay, az_deg, orig[0], pane_w, pane_h)
    out = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
    if az_deg is not None:
        _draw_pred(out, az_deg, orig[0])
    _draw_hotspot(out, hotspot_xy, orig)
    return out


def render_case(house, i, j, title, data_dir, out_path, pane_w=PANE_W):
    house_path = os.path.join(data_dir, house + ".json")
    data = json.load(open(house_path, encoding="utf-8"))
    pair = _find_pair(data, i, j)
    if pair is None:
        raise SystemExit("找不到配對 %s %d-%d" % (house, i, j))
    images = data["images"]
    ia, ib = images[pair["i"]], images[pair["j"]]
    pa = find_house_image(house, os.path.splitext(ia)[0])
    pb = find_house_image(house, os.path.splitext(ib)[0])
    if not (pa and pb):
        raise SystemExit("找不到影像 %s %s / %s" % (house, ia, ib))
    task_dir = os.path.dirname(pa)
    hs = gt_hotspot.find_hotspot_files(task_dir)
    if not hs:
        hs = gt_hotspot.find_hotspot_files(os.path.dirname(task_dir))
    loc = gt_hotspot.load_gt_locations(hs)
    xy_a = _hotspot_xy(loc, ia, ib)
    xy_b = _hotspot_xy(loc, ib, ia)
    left = render_pane(pa, pair.get("az_ij"), xy_a, pane_w=pane_w)
    right = render_pane(pb, pair.get("az_ji"), xy_b, pane_w=pane_w)
    err_a = circ_err(pair["az_ij"], pair["gt_az_ij"]) if pair.get("gt_az_ij") is not None else None
    err_b = circ_err(pair["az_ji"], pair["gt_az_ji"]) if pair.get("gt_az_ji") is not None else None
    gap, head, foot = 12, 64, 44
    canvas = Image.new("RGB", (left.width + gap + right.width, head + left.height + foot), WHITE)
    dr = ImageDraw.Draw(canvas)
    err_txt = ""
    if err_a is not None and err_b is not None:
        err_txt = "  ·  偏差 %.1f° / %.1f°" % (err_a, err_b)
    dr.text((16, 18), title + err_txt, fill=INK, font=_font(26))
    src = "幾何驗證" if pair.get("az_source") == "geo" else "Fallback"
    dr.text((16, 44), "%s  ·  %s" % (house, src), fill=MUTED, font=_font(16))
    canvas.paste(left, (0, head))
    canvas.paste(right, (left.width + gap, head))
    ly = head + left.height + 12
    dr.rectangle([16, ly, 36, ly + 14], fill=PRED)
    dr.text((42, ly - 2), "藍線／淡帶：預測方位（帶寬 ±22.5°）", fill=INK, font=_font(16))
    dr.rectangle([420, ly, 440, ly + 14], outline=HOTSPOT, width=3)
    dr.text((446, ly - 2), "綠框：HOTSPOT", fill=INK, font=_font(16))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    canvas.save(out_path, quality=90)
    print("saved", out_path)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="畫預測方位 vs HOTSPOT 案例")
    ap.add_argument("--data", default=default_data_dir())
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "case_examples"))
    args = ap.parse_args(argv)
    outs = []
    for tag, house, i, j, title in CASES:
        path = os.path.join(args.out, "%s_%s_%d_%d.jpg" % (tag, house, i, j))
        outs.append(render_case(house, i, j, title, args.data, path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
