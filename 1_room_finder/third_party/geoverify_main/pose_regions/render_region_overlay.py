#!/usr/bin/env python3
"""
共視區域 overlay 渲染器(供 區域遮罩模式 參考與驗收視覺抽查)。

讀 Output/roma_regions/{pair}.json(schema wp1-step3-v1/v2),把 regions 畫在 A/B 兩張
ERP 上:同一 region 兩側同色半透明遮罩 + 邊框 + 編號;另標 az_ab / az_ba 方位線
(位姿平移方向 = 通道方位的正確估計)與 GT hotspot 方位(有的話,虛線)。
A 上、B 下堆疊 + 資訊列(conf_used / inlier 率 / 旗標 / 區域數)。

過縫處理:polygon x 可能超出 [0, W)(JSON 約定,保多邊形連續)。繪製時同一多邊形
平移 -W/0/+W 各畫一次,超界部分自然裁掉。前端請用同樣手法。

按工作計畫備註「overlay 不全量產出」:
  # 單對(on-demand)
  venv/Scripts/python.exe pose_regions/render_region_overlay.py --json Output/roma_regions/XXX.json
  # 抽樣批次(驗收用;每類 N 張,決定性選擇)
  venv/Scripts/python.exe pose_regions/render_region_overlay.py --sample 4
輸出:Output/roma_regions/overlays/(gitignore;預設寬 2048 的 jpg,單檔 ~0.3MB)
"""
import argparse, csv, glob, json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

# 模組化(2026-07-23):匯入零副作用(chdir 移入 main);套件/腳本雙模式匯入。
try:
    from .estimate_relative_pose import find_house_image
except ImportError:
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from estimate_relative_pose import find_house_image

PALETTE = [(230, 25, 75), (60, 180, 75), (67, 99, 216), (245, 130, 48),
           (145, 30, 180), (66, 212, 244), (240, 50, 230), (191, 239, 69)]
# 產出位置可用 ROMA_GEO_OUTPUT 覆寫(同 config.regions_root)。此處直接讀環境變數
# 而不 import config,以保住 pose_regions 可獨立匯入。
REG_DIR = os.path.join(
    os.path.abspath(os.path.expanduser(os.environ["ROMA_GEO_OUTPUT"]))
    if os.environ.get("ROMA_GEO_OUTPUT") else os.path.join(ROOT, "Output"),
    "roma_regions")
OUT_DIR = os.path.join(REG_DIR, "overlays")


def _font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_regions(img, regions, side, s, W):
    """在縮放後的 ERP 上畫該側所有 region(平移 -W/0/+W 各畫一次處理過縫)。"""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    for k, reg in enumerate(regions):
        col = PALETTE[k % len(PALETTE)]
        poly = reg[side]["polygon"]
        if len(poly) < 3:
            continue
        for dx in (-W, 0.0, W):
            pts = [((x + dx) * s, y * s) for x, y in poly]
            if max(p[0] for p in pts) < 0 or min(p[0] for p in pts) > img.width:
                continue
            dr.polygon(pts, fill=col + (80,))
            dr.line(pts + [pts[0]], fill=col + (255,), width=3)
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            if 0 <= cx < img.width:
                dr.text((cx, cy), f"R{k}", fill=col + (255,), font=_font(30), anchor="mm")
    return Image.alpha_composite(img.convert("RGBA"), layer)


def _az_line(img, az_deg, W, s, color, label, dashed=False):
    """在縮放後 ERP 上畫方位垂直線(az → x = (az+180)/360*W)。"""
    x = ((az_deg + 180.0) % 360.0) / 360.0 * W * s
    dr = ImageDraw.Draw(img)
    h = img.height
    if dashed:
        for y0 in range(0, h, 24):
            dr.line([(x, y0), (x, min(y0 + 12, h))], fill=color, width=3)
    else:
        dr.line([(x, 0), (x, h)], fill=color, width=3)
    dr.text((x + 6, 8), label, fill=color, font=_font(26))
    return img


def render_pair(json_path, out_dir=OUT_DIR, target_w=2048, draw_az=True, geometry=None):
    """draw_az=False:只畫共視區域遮罩,不畫 az/GT 方位線(deck 素材用。線條對非本組讀者不直觀)。"""
    d = geometry if geometry is not None else json.load(open(json_path, encoding="utf-8"))
    house, na, nb = d["house"], d["img_a"], d["img_b"]
    pa, pb = find_house_image(house, na), find_house_image(house, nb)
    if not (pa and pb):
        raise SystemExit(f"找不到圖:{house} {na}/{nb}")
    Wa, Ha = d["img_size_a"]
    Wb, Hb = d["img_size_b"]
    sa, sb = target_w / Wa, target_w / Wb
    A = Image.open(pa).resize((target_w, int(Ha * sa)), Image.BILINEAR)
    B = Image.open(pb).resize((target_w, int(Hb * sb)), Image.BILINEAR)
    regs = d.get("regions") or []
    A = _draw_regions(A, regs, "a", sa, Wa)
    B = _draw_regions(B, regs, "b", sb, Wb)
    pose, gt = d.get("pose"), d.get("gt") or {}
    if draw_az and pose:
        A = _az_line(A, pose["az_ab_deg"], Wa, sa, (255, 255, 0), "az_ab(→B)")
        B = _az_line(B, pose["az_ba_deg"], Wb, sb, (255, 255, 0), "az_ba(→A)")
    if draw_az and gt.get("az_ab_deg") is not None:
        A = _az_line(A, gt["az_ab_deg"], Wa, sa, (0, 255, 255), "GT hotspot", dashed=True)
    if draw_az and gt.get("az_ba_deg") is not None:
        B = _az_line(B, gt["az_ba_deg"], Wb, sb, (0, 255, 255), "GT hotspot", dashed=True)

    hdr_h, gap = 64, 8
    canvas = Image.new("RGB", (target_w, hdr_h + A.height + gap + B.height), (18, 18, 18))
    f = d.get("flags", {})
    info = (f"{house}  {na[:8]}->{nb[:8]}  conf_used={d.get('conf_used')}  "
            f"inlier={d.get('inlier_rate')}  regions={len(regs)}  "
            f"flags[low={int(f.get('low_matches', 0))} nocov={int(f.get('no_covisibility', 0))} "
            f"deg={int(f.get('degenerate_rot_only', 0))} door={int(f.get('doorway_wedge', 0))}]  "
            f"gt_label={gt.get('label')} same_room={gt.get('same_room')}")
    ImageDraw.Draw(canvas).text((12, hdr_h // 2), info, fill=(240, 240, 240), font=_font(24), anchor="lm")
    canvas.paste(A.convert("RGB"), (0, hdr_h))
    canvas.paste(B.convert("RGB"), (0, hdr_h + A.height + gap))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{house}_{na}_{nb}.jpg")   # 全名,與 JSON 檔名同約定(截斷會撞名)
    canvas.save(out, quality=88)
    print("saved ->", out)
    return out


def pick_samples(n_per):
    """驗收抽樣(決定性:各類按 house+img 排序取前 N):
    doorway / 同房 / 跨房非門洞 / 0.3 撈回 / 過縫 region / 多區(>=3)。"""
    rows = list(csv.DictReader(open(os.path.join(REG_DIR, "_summary.csv"), encoding="utf-8")))
    posed = [r for r in rows if r["az_ab_deg"]]
    key = lambda r: (r["house"], r["img_a"], r["img_b"])
    cats = {
        "doorway": [r for r in posed if r["flag_doorway_wedge"] == "1"],
        "same_room": [r for r in posed if r["same_room"] == "1"],
        "cross_room": [r for r in posed if r["same_room"] == "0" and r["flag_doorway_wedge"] == "0"],
        "retry03": [r for r in posed if r["conf_used"] == "0.3"],
        "multi_region": [r for r in posed if r["n_regions"] and int(r["n_regions"]) >= 3],
    }
    picked, seen = [], set()
    for cname, grp in cats.items():
        g = sorted(grp, key=key)
        g = g[::max(1, len(g) // max(n_per, 1))][:n_per]   # 等距取樣,跨房分散(仍決定性)
        for r in g:
            k = key(r)
            if k not in seen:
                seen.add(k)
                picked.append((cname, r))
    # 過縫案例:region1 az_range 上界 > 180(polygon x 會超出 [0,W))
    for f in sorted(glob.glob(os.path.join(REG_DIR, "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        regs = d.get("regions") or []
        if regs and regs[0]["a"]["az_range"][1] > 180.0:
            picked.append(("seam", {"house": d["house"], "img_a": d["img_a"], "img_b": d["img_b"]}))
            if sum(1 for c, _ in picked if c == "seam") >= n_per:
                break
    return picked


def main():
    os.chdir(ROOT)   # CLI 相對路徑(Output/ 等)以 repo 根為準
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="單對 JSON 路徑(on-demand)")
    ap.add_argument("--pair", nargs=3, metavar=("HOUSE", "IMG_A", "IMG_B"), help="以 id 指定單對")
    ap.add_argument("--sample", type=int, default=0, help="每類抽 N 對渲染(驗收批次)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--width", type=int, default=2048, help="輸出寬(原圖等比縮)")
    ap.add_argument("--no_az", action="store_true", help="不畫 az/GT 方位線,只畫共視區域遮罩(deck 素材)")
    a = ap.parse_args()
    if a.json:
        render_pair(a.json, a.out, a.width, draw_az=not a.no_az)
    elif a.pair:
        h, na, nb = a.pair
        render_pair(os.path.join(REG_DIR, f"{h}_{na}_{nb}.json"), a.out, a.width, draw_az=not a.no_az)
    elif a.sample:
        for cname, r in pick_samples(a.sample):
            jp = os.path.join(REG_DIR, f"{r['house']}_{r['img_a']}_{r['img_b']}.json")
            try:
                out = render_pair(jp, a.out, a.width)
                print(f"  [{cname}] {os.path.basename(out)}")
            except SystemExit as e:
                print(f"  [{cname}] skip: {e}")
    else:
        ap.error("需 --json / --pair / --sample 其一")


if __name__ == "__main__":
    main()
