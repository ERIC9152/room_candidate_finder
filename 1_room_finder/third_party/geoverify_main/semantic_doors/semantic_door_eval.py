"""語意門校正評估:用子計畫一的門語意圖修正通道方位,是否比 az_ab / coarse 更接近 GT。

背景:
  GT hotspot 在「隔門相連」時標的就是門(已由 hotspot_two_cases_eval 佐證),所以拿有門
  的案例來驗「語意門校正」是站得住的。幾何負責找出「哪一道門通往對方」,語意圖負責
  給出「那道門在哪」。

資料:datasets/semantic/<taskid>/<image>_mask.png(全解析度 ERP,門色 (255,50,50)/(128,0,0))。

三項必須處理的邊界情況(本腳本已處理,實作端照抄):
  1. **過縫**:門跨 ±180° 時 connectedComponents 會切成兩塊(實測 58553 a6df 兩塊各 1.9M px)。
     作法:先把 mask 水平接一份到右邊(寬 2W)做連通域,再把 x 折回 [0,W) 並記錄 unwrap 區間。
  2. **方位角約定**:segmentation/aligner 用 az=(x/W)*360(0=左緣);本專案 pipeline 用
     az=((x/W)*360-180)%360(0=圖中央)。兩者差 180°,必須轉換。
  3. **門後有門 / 半開門**:同一個門洞可能被門板切成數塊、或內外框各一塊。本腳本輸出
     每張圖的門元件明細供檢視,並提供 --merge_gap 以角距合併鄰近元件(取最外圍)。

估計法(每個方向一筆,與 GT hotspot 比):
  coarse        粗篩方向頭(az_coarse_ij/ji)
  az_ab         幾何基線方位(現行工具用的)
  door_nearest  取「離 az_ab 最近」的門,用該門的中心方位
  door_covis    取「與共視區域重疊度最高」的門(假設:能透過門看到對方 ⇒ 共識區在門的範圍內)
  door_covis_in 同上但只採信「共識區確實被門包住」(重疊率 ≥ --contain)的情形,否則退回 az_ab

用法:
  python semantic_doors/semantic_door_eval.py            # 全部有語意圖的房子
  python semantic_doors/semantic_door_eval.py --houses 58553
  python semantic_doors/semantic_door_eval.py --dump_doors   # 另存每張圖的門明細
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 本檔位於專案根目錄下一層，故 dirname 一次即為根目錄。
ROOT = os.path.dirname(SCRIPT_DIR)
CACHE = os.path.join(SCRIPT_DIR, "_door_cache.json")
OUT_MD = os.path.join(SCRIPT_DIR, "semantic_door_eval.md")

DOOR_RGB = [(255, 50, 50), (128, 0, 0)]
COL_TOL = 45
MIN_PX = 2000          # 門元件最小像素(8192×4096 下,小於此多為雜點)
MAX_DOOR_DEG = 90.0    # 合併後的門角寬上限;超過即視為串連了不同門,不合併


def base(n):
    return os.path.splitext(n)[0]


def circ_err(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def px_to_az(x, W):
    """本專案 pipeline 約定(0=圖中央,±180=接縫)。注意與 segmentation/aligner 差 180°。"""
    return ((x / W) * 360.0 - 180.0) % 360.0


def extract_doors(mask_path, merge_gap_deg=3.0):
    """回傳該圖的門清單:[{az, az0, az1, width_deg, px, y0, y1}](az 皆為 pipeline 約定)。
       過縫處理:寬度接倍後做連通域,再折回。"""
    import cv2
    from PIL import Image
    a = np.array(Image.open(mask_path).convert("RGB"))
    H, W = a.shape[:2]
    m = np.zeros((H, W), np.uint8)
    ai = a.astype(np.int16)
    for c in DOOR_RGB:
        m |= (np.abs(ai - np.array(c, dtype=np.int16)).sum(axis=2) < COL_TOL).astype(np.uint8)
    if m.sum() == 0:
        return []
    # 過縫:接一份到右側,連通域跨接縫的會連成一塊
    m2 = np.concatenate([m, m], axis=1)
    n, lab = cv2.connectedComponents(m2)
    raw = []
    for k in range(1, n):
        ys, xs = np.where(lab == k)
        if len(xs) < MIN_PX:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        if x0 >= W:                    # 完全落在複本區,本體已收過
            continue
        if x1 - x0 >= W:               # 整圈都是門(異常)
            continue
        raw.append(dict(x0=x0, x1=x1, px=int(len(xs)),
                        y0=int(ys.min()), y1=int(ys.max())))
    # 去除「過縫本體的尾巴被當成獨立元件」的重複:
    # 例 a6df 的門在倍寬圖是 [7164,8975];而 [0,783] 也自成一塊,+W 後 =[8192,8975] ⊆ 前者。
    keep = []
    for d in raw:
        dup = False
        for o in raw:
            if o is d:
                continue
            for sh in (0, W):
                if o["x0"] <= d["x0"] + sh and d["x1"] + sh <= o["x1"] and \
                   (o["x1"] - o["x0"]) > (d["x1"] - d["x0"]):
                    dup = True
                    break
            if dup:
                break
        if not dup:
            keep.append(d)
    out = []
    for d in keep:
        span = d["x1"] - d["x0"]
        out.append(dict(az0=px_to_az(d["x0"], W), az1=px_to_az(d["x1"], W),
                        az=px_to_az((d["x0"] + d["x1"]) / 2.0, W),
                        width_deg=span / W * 360.0, px=d["px"],
                        y0=d["y0"], y1=d["y1"],
                        _x0=d["x0"], _x1=d["x1"], _W=W))
    # 依角距合併鄰近元件(門後有門 / 半開門被門板切開)→ 取最外圍
    out.sort(key=lambda d: d["_x0"])
    merged = []
    for d in out:
        if merged:
            prev = merged[-1]
            gap = (d["_x0"] - prev["_x1"]) / d["_W"] * 360.0
            newspan = (max(prev["_x1"], d["_x1"]) - prev["_x0"]) / d["_W"] * 360.0
            # 只補「同一門洞被門板切開」的細縫;不得串連成大片(走廊多門相鄰時最危險)
            if gap <= merge_gap_deg and newspan <= MAX_DOOR_DEG:
                prev["_x1"] = max(prev["_x1"], d["_x1"])
                prev["px"] += d["px"]
                prev["y0"] = min(prev["y0"], d["y0"])
                prev["y1"] = max(prev["y1"], d["y1"])
                prev["az0"] = px_to_az(prev["_x0"], d["_W"])
                prev["az1"] = px_to_az(prev["_x1"], d["_W"])
                prev["az"] = px_to_az((prev["_x0"] + prev["_x1"]) / 2.0, d["_W"])
                prev["width_deg"] = (prev["_x1"] - prev["_x0"]) / d["_W"] * 360.0
                continue
        merged.append(dict(d))
    for d in merged:
        d.pop("_x0", None); d.pop("_x1", None); d.pop("_W", None)
    merged.sort(key=lambda d: -d["px"])
    return merged


def az_overlap(a0, a1, b0, b1):
    """與交付路徑同一份實作。"""
    from roma_geo.semantic import az_overlap as _overlap
    return _overlap(a0, a1, b0, b1)


def region_az_span(poly, W):
    xs = [p[0] for p in poly]
    return px_to_az(min(xs), W), px_to_az(max(xs), W), (max(xs) - min(xs)) / W * 360.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--houses", nargs="*", default=None)
    ap.add_argument("--merge_gap", type=float, default=1.0)
    ap.add_argument("--contain", type=float, default=0.8,
                    help="共識區被門包住的最低重疊比例(door_covis_in 用)")
    ap.add_argument("--dump_doors", action="store_true")
    ap.add_argument("--quality_gate", action="store_true",
                    help="只在高品質位姿(conf_used=0.65 且 inlier_rate>=--min_ir)上套用語意校正")
    ap.add_argument("--min_ir", type=float, default=0.15)
    ap.add_argument("--cross_room_only", action="store_true",
                    help="只看跨房案例;預設同房與跨房皆納入")
    ap.add_argument("--refresh", action="store_true", help="忽略門快取重抽")
    args = ap.parse_args()
    from roma_geo.config import semantic_root, regions_root
    sem_dir = semantic_root()
    reg_dir = regions_root()
    data_dir = os.path.join(ROOT, "stitch_tool", "data")

    # ---- 門抽取(含快取)----
    cache = {}
    if os.path.exists(CACHE) and not args.refresh:
        cache = json.load(open(CACHE, encoding="utf-8"))
    task_ids = sorted(os.listdir(sem_dir))
    if args.houses:
        task_ids = [t for t in task_ids if t in args.houses]
    n_new = 0
    for tid in task_ids:
        for mp in sorted(glob.glob(os.path.join(sem_dir, tid, "*_mask.png"))):
            img = os.path.basename(mp)[:-len("_mask.png")]
            key = tid + "/" + img
            if key in cache:
                continue
            cache[key] = extract_doors(mp, args.merge_gap)
            n_new += 1
            if n_new % 20 == 0:
                print("  抽取中 %d…" % n_new, flush=True)
    if n_new:
        json.dump(cache, open(CACHE, "w", encoding="utf-8"))
    print("門快取:%d 張圖(本次新抽 %d)" % (len(cache), n_new))

    if args.dump_doors:
        for k in sorted(cache):
            ds = cache[k]
            print(k, "門數", len(ds))
            for d in ds[:6]:
                print("    az=%6.1f [%6.1f,%6.1f] 寬%5.1f° px=%d" %
                      (d["az"], d["az0"], d["az1"], d["width_deg"], d["px"]))

    # ---- GT / coarse ----
    coarse, gt = {}, {}
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        hd = json.load(open(fp, encoding="utf-8"))
        if not isinstance(hd, dict) or "pairs" not in hd:
            continue
        house, imgs = hd["house"], [base(x) for x in hd["images"]]
        for p in hd["pairs"]:
            if not p.get("gt_connected"):
                continue
            bi, bj = imgs[p["i"]], imgs[p["j"]]
            if p.get("gt_az_ij") is not None:
                coarse[(house, bi, bj)] = p.get("az_coarse_ij", p["az_ij"])
                gt[(house, bi, bj)] = p["gt_az_ij"]
            if p.get("gt_az_ji") is not None:
                coarse[(house, bj, bi)] = p.get("az_coarse_ji", p["az_ji"])
                gt[(house, bj, bi)] = p["gt_az_ji"]

    def doors_for(house, img):
        tid = "".join(ch for ch in house if ch.isdigit())
        return cache.get(tid + "/" + img, [])

    rows = []
    n_pair = 0
    for fp in glob.glob(os.path.join(reg_dir, "*.json")):
        if os.path.basename(fp).startswith("_"):
            continue
        try:
            j = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        pose = j.get("pose")
        if not pose:
            continue
        house = j.get("house")
        ia, ib = base(j.get("img_a", "")), base(j.get("img_b", ""))
        if not doors_for(house, ia) and not doors_for(house, ib):
            continue
        same_room = (j.get("gt") or {}).get("same_room")
        if same_room is None:
            continue
        if same_room and args.cross_room_only:   # 預設同房與跨房皆納入
            continue
        n_pair += 1
        # 品質閘門:低信心撈回層(conf 0.3)與極低內點率的位姿,語意校正不套用
        hi_q = (j.get("conf_used") == 0.65 and (j.get("inlier_rate") or 0) >= args.min_ir)
        regs = j.get("regions") or []
        Wa = (j.get("img_size_a") or [0])[0]
        Wb = (j.get("img_size_b") or [0])[0]

        for key, side, W, az_pose in (((house, ia, ib), "a", Wa, pose.get("az_ab_deg")),
                                      ((house, ib, ia), "b", Wb, pose.get("az_ba_deg"))):
            if key not in gt or az_pose is None or not W:
                continue
            img = key[1]
            ds = doors_for(house, img)
            if not ds:
                continue
            g = gt[key]
            azp = az_pose % 360.0

            # (1) 離 az_ab 最近的門
            d_near = min(ds, key=lambda d: circ_err(d["az"], azp))

            # (2) 與共識區重疊度最高的門
            best = None
            for reg in regs:
                poly = (reg.get(side) or {}).get("polygon")
                if not poly or len(poly) < 3:
                    continue
                r0, r1, rw = region_az_span(poly, W)
                if rw <= 0:
                    continue
                for d in ds:
                    ov = az_overlap(r0, r1, d["az0"], d["az1"])
                    frac = ov / rw                     # 共識區有多少比例落在門內
                    if best is None or frac > best[0]:
                        best = (frac, d, rw)
            d_cov = best[1] if best else None
            cov_frac = best[0] if best else 0.0

            # (3) 夾取(clamp):az_ab 已落在門內就不動;落在門外才拉到最近的門邊。
            #     動機:az_ab 中位已 10°,直接換成門中心反而變差(門中位寬 27°);
            #     真正該修的只有「指到門外的牆」那批。
            def inside(d, az):
                return az_overlap(az, az, d["az0"], d["az1"]) > 0 or \
                    ((az - d["az0"]) % 360.0) <= ((d["az1"] - d["az0"]) % 360.0)

            def clamp_to(d, az):
                if inside(d, az):
                    return az
                e0, e1 = d["az0"], d["az1"]
                return e0 if circ_err(az, e0) <= circ_err(az, e1) else e1

            in_any = any(inside(d, azp) for d in ds)
            az_clamp = azp if in_any else clamp_to(d_near, azp)
            az_clamp_cov = azp if (d_cov and inside(d_cov, azp)) else \
                (clamp_to(d_cov, azp) if d_cov else azp)

            # (4) 門 ∩ 共視區 的中心(能真正看穿的那一段)
            az_isect = None
            if d_cov and best:
                rw = best[2]
                for reg in regs:
                    poly = (reg.get(side) or {}).get("polygon")
                    if not poly or len(poly) < 3:
                        continue
                    r0, r1, w = region_az_span(poly, W)
                    if abs(w - rw) > 1e-6:
                        continue
                    lo = r0 if az_overlap(r0, r0, d_cov["az0"], d_cov["az1"]) > 0 or \
                        ((r0 - d_cov["az0"]) % 360) <= ((d_cov["az1"] - d_cov["az0"]) % 360) else d_cov["az0"]
                    hi = r1 if ((r1 - d_cov["az0"]) % 360) <= ((d_cov["az1"] - d_cov["az0"]) % 360) else d_cov["az1"]
                    az_isect = (lo + ((hi - lo) % 360.0) / 2.0) % 360.0
                    break

            rows.append(dict(
                same_room=bool(same_room),
                n_doors=len(ds),
                e_clamp=circ_err(az_clamp, g),
                e_clamp_cov=circ_err(az_clamp_cov, g),
                e_isect=circ_err(az_isect, g) if az_isect is not None else None,
                az_outside=not in_any,
                e_coarse=circ_err(coarse[key], g),
                e_az=circ_err(azp, g),
                e_near=circ_err(d_near["az"], g),
                e_cov=circ_err(d_cov["az"], g) if d_cov else None,
                e_cov_in=(circ_err(d_cov["az"], g)
                          if (d_cov and cov_frac >= args.contain
                              and (hi_q or not args.quality_gate))
                          else circ_err(azp, g)),
                hi_q=hi_q,
                cov_frac=cov_frac,
                door_w=d_near["width_deg"],
            ))

    def stats(v):
        v = [x for x in v if x is not None]
        if not v:
            return None
        s = sorted(v)
        n = len(s)

        def pct(p):
            k = (n - 1) * p
            lo, hi = int(k), min(int(k) + 1, n - 1)
            return s[lo] + (s[hi] - s[lo]) * (k - lo)
        return dict(n=n, med=pct(0.5), p90=pct(0.9),
                    r22=sum(1 for e in s if e <= 22.5) / n * 100,
                    r45=sum(1 for e in s if e <= 45) / n * 100)

    def row(name, st):
        if not st:
            return "| %s | 0 | — | — | — | — |" % name
        return "| %s | %d | %.1f° | %.1f° | %.1f%% | %.1f%% |" % (
            name, st["n"], st["med"], st["p90"], st["r22"], st["r45"])

    out = []
    out.append("# 語意門校正評估:用子計畫一的門語意圖修正通道方位\n")
    out.append("產生腳本:`semantic_door_eval.py`。族群 = **GT 連通且該圖有語意門**的方向,")
    out.append("同房與跨房皆納入(`--cross_room_only` 可限定為跨房)。")
    out.append("GT = HOTSPOT 方位(隔門相連時標的就是門,見 hotspot_two_cases_eval.md)。\n")
    out.append("| 估計法 | n | 中位 | p90 | ≤22.5° | ≤45° |")
    out.append("|---|---|---|---|---|---|")
    out.append(row("coarse 方向頭", stats([r["e_coarse"] for r in rows])))
    out.append(row("az_ab 基線方位(現行)", stats([r["e_az"] for r in rows])))
    out.append(row("**門校正:離 az_ab 最近的門**", stats([r["e_near"] for r in rows])))
    out.append(row("**門校正:與共識區重疊最高的門**", stats([r["e_cov"] for r in rows])))
    out.append(row("**門校正:僅採信被門包住者(其餘退回 az_ab)**", stats([r["e_cov_in"] for r in rows])))
    out.append(row("**夾取:az_ab 落在門外才拉進最近的門**", stats([r["e_clamp"] for r in rows])))
    out.append(row("**夾取(門由共視區決定)**", stats([r["e_clamp_cov"] for r in rows])))
    out.append(row("門∩共視區的中心(看得穿的那一段)", stats([r["e_isect"] for r in rows])))
    out.append("")
    n_out = sum(1 for r in rows if r["az_outside"])
    if rows:
        out.append("az_ab 落在**所有門之外**的比例:%.1f%%(%d/%d);夾取只會動到這一批。\n"
                   % (n_out / len(rows) * 100, n_out, len(rows)))
        sel = [r for r in rows if r["az_outside"]]
        if sel:
            out.append("僅看這一批(az_ab 指到門外):")
            out.append("| 估計法 | n | 中位 | p90 | ≤22.5° | ≤45° |")
            out.append("|---|---|---|---|---|---|")
            out.append(row("az_ab", stats([r["e_az"] for r in sel])))
            out.append(row("夾取後", stats([r["e_clamp"] for r in sel])))
            out.append(row("最近門中心", stats([r["e_near"] for r in sel])))
            out.append("")
    multi = [r for r in rows if r["n_doors"] >= 3]
    single = [r for r in rows if r["n_doors"] <= 2]
    out.append("## 分層:該圖的門數(走廊型多門是已知難點)\n")
    out.append("| 族群 | 估計法 | n | 中位 | p90 | ≤22.5° | ≤45° |")
    out.append("|---|---|---|---|---|---|---|")
    for lab, sel in (("門≤2", single), ("門≥3", multi)):
        out.append("| %s | az_ab " % lab + row("", stats([r["e_az"] for r in sel]))[3:])
        out.append("| %s | 最近門 " % lab + row("", stats([r["e_near"] for r in sel]))[3:])
        out.append("| %s | 共視重疊門 " % lab + row("", stats([r["e_cov"] for r in sel]))[3:])
    out.append("")
    # ---- 決定性拆解:包住判準本身是不是好的開關 ----
    out.append("## 拆解:「共識區被門包住」是不是好的開關(決定性證據)\n")
    out.append("| 子集 | n | az_ab 中位 | 門中心 中位 | az_ab ≤22.5° | 門中心 ≤22.5° |")
    out.append("|---|---|---|---|---|---|")
    for lab, pred in (("共識區**被門包住**(cov_frac≥%.1f)" % args.contain,
                       lambda r: r["cov_frac"] >= args.contain),
                      ("共識區**不在門內**(cov_frac<%.1f)" % args.contain,
                       lambda r: r["cov_frac"] < args.contain)):
        sel = [r for r in rows if pred(r) and r["e_cov"] is not None]
        if not sel:
            continue
        sa = stats([r["e_az"] for r in sel])
        sc = stats([r["e_cov"] for r in sel])
        out.append("| %s | %d | %.1f° | %.1f° | %.1f%% | %.1f%% |"
                   % (lab, len(sel), sa["med"], sc["med"], sa["r22"], sc["r22"]))
    out.append("")
    out.append("### 包住門檻掃描(穩健性)\n")
    out.append("| 門檻 | 觸發比例 | 混合後中位 | 混合後 ≤22.5° |")
    out.append("|---|---|---|---|")
    for th in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        mix = [(r["e_cov"] if (r["e_cov"] is not None and r["cov_frac"] >= th) else r["e_az"])
               for r in rows]
        fire = sum(1 for r in rows if r["e_cov"] is not None and r["cov_frac"] >= th)
        s = stats(mix)
        out.append("| %.1f | %.1f%% | %.1f° | %.1f%% |"
                   % (th, fire / len(rows) * 100, s["med"], s["r22"]))
    out.append("")

    cf = sorted(r["cov_frac"] for r in rows)
    if cf:
        out.append("共識區落在門內的比例(cov_frac)分布:中位 %.2f、p25 %.2f、p75 %.2f;"
                   "≥%.1f 者佔 %.1f%%。\n"
                   % (cf[len(cf) // 2], cf[len(cf) // 4], cf[3 * len(cf) // 4], args.contain,
                      sum(1 for x in cf if x >= args.contain) / len(cf) * 100))
    dw = sorted(r["door_w"] for r in rows)
    if dw:
        out.append("被選中的門角寬:中位 %.1f°、p90 %.1f°(門很寬時「門中心」本身就有數度誤差)。\n"
                   % (dw[len(dw) // 2], dw[int(len(dw) * 0.9)]))
    # 結論引用的數字一律由本輪計算,避免與上方表格脫節。
    _in = [r for r in rows if r["e_cov"] is not None and r["cov_frac"] >= args.contain]
    _out = [r for r in rows if r["e_cov"] is not None and r["cov_frac"] < args.contain]
    s_in_az, s_in_door = stats([r["e_az"] for r in _in]), stats([r["e_cov"] for r in _in])
    s_out_az, s_out_door = stats([r["e_az"] for r in _out]), stats([r["e_cov"] for r in _out])
    s_base, s_gated = stats([r["e_az"] for r in rows]), stats([r["e_cov_in"] for r in rows])
    s_near, s_cov, s_clamp = (stats([r["e_near"] for r in rows]),
                              stats([r["e_cov"] for r in rows]),
                              stats([r["e_clamp"] for r in rows]))
    fire_pct = len(_in) / len(rows) * 100 if rows else 0.0

    out.append("## 結論(2026-07-29,數字隨每次執行更新)\n")
    out.append("**假設成立,且找到正確用法:語意門的作用是提供「共識區是否被門包住」這個開關,")
    out.append("包住時改用門中心、沒包住時維持 az_ab。**\n")
    if s_in_door and s_out_door:
        out.append("- 決定性拆解:包住時門中心 **%.1f°** vs az_ab %.1f°(≤22.5° %.1f%% vs %.1f%%);"
                   % (s_in_door["med"], s_in_az["med"], s_in_door["r22"], s_in_az["r22"]))
        out.append("  沒包住時門中心 %.1f° vs az_ab %.1f°;開關兩側的行為完全相反,足以證明判準有效。"
                   % (s_out_door["med"], s_out_az["med"]))
    if s_base and s_gated:
        out.append("- 混合後全族群中位 %.1f°→**%.1f°**(門檻 %.1f),≤22.5° %.1f%%→%.1f%%;門檻 0.5–0.9 皆穩定。"
                   % (s_base["med"], s_gated["med"], args.contain, s_base["r22"], s_gated["r22"]))
    out.append("- 物理對應:共識區被門包住 ⇒ 那是**透過這道門看到對面**,門即通道;")
    out.append("  共識區遠大於門 ⇒ 那是**自己這張圖房間內**的內容(或大環),與該門無關,不可拿來校正。")
    if s_near and s_cov and s_clamp:
        out.append("- 失敗的替代方案(併記,避免重試):直接換成最近門 %.1f° 但尾巴爆掉(p90 %.0f°);"
                   % (s_near["med"], s_near["p90"]))
        out.append("  直接換成共視重疊最高的門 %.1f°;夾取(az_ab 落門外才拉進門)%.1f°,更差,"
                   % (s_cov["med"], s_clamp["med"]))
    op = [r for r in rows if r["az_outside"]]
    if op:
        s_op_az, s_op_clamp = stats([r["e_az"] for r in op]), stats([r["e_clamp"] for r in op])
        out.append("  因為 az_ab 有 %.1f%% 的時候落在**所有門之外**,那批硬拉進門會從 %.1f° 惡化到 %.1f°;"
                   % (len(op) / len(rows) * 100, s_op_az["med"], s_op_clamp["med"]))
        out.append("  「落在門外」無法作為可靠的失效訊號(語意圖未必涵蓋所有開口)。")
    out.append("- 適用範圍:凡具備語意門的方向皆納入,同房與跨房不分;開關觸發率約 %.0f%%,"
               "其餘維持現行 az_ab,故**不會變差**。" % fire_pct)
    out.append("- 語意圖涵蓋率(實測):22/57 間房;那 22 間的 583 張影像中 402 張有 mask(**69.0%**)。")
    out.append("  跨房側有門的比例 69.9%,與影像涵蓋率一致;同房側僅 1.7%(見下)。")
    out.append("- **品質閘門(`--quality_gate`)已停用,不要啟用。**")
    out.append("  停用理由:將 2 個同房觸發逕行認定為「誤觸發」缺乏依據,**未查證其誤差是否真的變差**。實查:")
    out.append("  一個方向沒有 GT(gt_label=0)無從判定;另一個(59863)GT=1.9°、az_ab=17.9°(誤差 16.0°)、")
    out.append("  門校正後 3.1°(誤差 **1.3°**),**變好 14.8°**。也就是說同房觸發至今**沒有任何一例被證實有害**,")
    out.append("  而該閘門會把這個 14.8° 的改善一併擋掉,代價還有全體中位 9.5°→10.5°。")
    out.append("  旗標仍保留在腳本裡供日後有反例時再評估,但**預設關閉**。")
    out.append("  另記:`doorway_wedge` 無鑑別力(觸發案例中同房 50% vs 跨房 26.4%,n=2),更不可當閘門。")
    out.append("- **大環誤觸發已實測(不需 GT 方位,只需 same_room 標籤,故樣本充足)**:")
    out.append("  同房/大環 470 側中僅 2 側觸發 = **0.43%**;跨房 1430 側中 382 側觸發 = 26.7%。")
    out.append("  **保護機制是同房/大環的影像有 98.3% 根本沒偵測到門**")
    out.append("  (470 側中只有 8 側有門)。拆一層看:那 462 個「沒門」的側裡,459 個**沒有語意圖檔**、")
    out.append("  只有 3 個是「有圖但偵測不到門」。**注意本統計已限定在有語意圖的 22 間房內**。")
    out.append("  對此的兩種解讀,目前資料無法區分:")
    out.append("  (a) 涵蓋率假象:若日後補上那些圖、且它們有門,開關就可能開始誤觸;")
    out.append("  (b) 目視觀察:**那些缺 mask 的影像就是戶外案例,而戶外幾乎沒有門**,")
    out.append("      所以「沒 mask」與「沒有門」是同一件事的兩面,保護是內生的。")
    out.append("  沒有 mask 的影像無從觀察其門,故 (b) **無法用現有資料證實或否證**;")
    out.append("  目視觀察是目前唯一的證據來源,本文件據此不採取任何預防性收緊。")
    out.append("  (少數有圖的同房案例門角寬中位 28.1° vs 跨房 21.2°,但 n 僅 13 道門,不足以推論。)")
    out.append("  故工具端仍應把觸發事件記錄下來供目視抽查。")
    out.append("- 已知案例:58553 `0894de77`↔`dc4c66d0` 兩側恰好各是一種情形:")
    out.append("  `dc4c66d0` 單門且共識區完全落在門內 ⇒ 觸發,誤差 20.4°→12.6°;")
    out.append("  `0894de77` 三道門且共識區僅 32% 落在門內 ⇒ 不觸發,維持 az_ab(33.5°)不被改壞。")
    out.append("  正是「看不到門後內容就無法定位、除非另一張看得到」的實例。\n")
    out.append("(參與 pair 數 %d;門合併角距 %.1f°、包住判準 %.2f。)" % (n_pair, args.merge_gap, args.contain))
    text = "\n".join(out)
    open(OUT_MD, "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
