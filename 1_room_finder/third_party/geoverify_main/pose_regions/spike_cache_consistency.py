#!/usr/bin/env python3
"""
 縮水版 spike:驗證 roma_match/cache/*.npz 的座標約定。

問題:cache 的 matches (K,5) 座標到底是「原圖 ERP 像素」還是 512 處理空間?
(RoMaV2 輸入會被縮成 512×512 處理;下游 pose_regions / stitch_tool 都當原圖像素用,
 但從未正式驗證過。)

方法:抽 3 對(3 棟不同房子,取 conf>=0.65 點數最多的 pair,偏向真重疊 pair),
對每對比較「cache 讀出」vs「同一對 live 重跑 RoMa(繞過快取)」:
  1. range/scale:兩邊座標都落在 [0,W]x[0,H],且 max 比值 ~1(若 cache 是 512 空間,
     比值會是 ~16 倍,一眼識破)。
  2. NN 距離:cache 點到 live 點的最近鄰距離中位數(A 側 2D)。sample() 是隨機抽樣,
     兩次 draw 不會逐點相同,但同約定下兩組點蓋同一分佈,NN 中位數應在十幾 px 等級
     (8192 寬圖);若約定不同會是數千 px。
  3. 幾何交叉驗證:用 cache 點 fit RANSAC essential(球面射線),拿同一個 E 去算
     live 點的內點率,反向亦然。同約定下交叉內點率 ≈ 自身內點率;約定錯了射線全歪,
     交叉內點率會塌掉。
  4. 邊界指紋(參考用):兩邊的 x 極值是否同為半格 margin(W=8192 時 5.12 / 8186.88)。

另附全量掃描(--skip_scan 可跳過):逐一檢查每個快取檔的座標範圍不超出該房原圖 [0,W]x[0,H]。

輸出:stdout 報告 + spike_cache_consistency_report.md(供 README 引用)。
用法:  venv/Scripts/python.exe pose_regions/spike_cache_consistency.py  [--skip_scan]
(需 GPU venv;live 重跑 3 對約 1-2 分鐘 + 模型載入。)
"""
import argparse, glob, os, sys, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

# 模組化(2026-07-23):匯入零副作用(chdir 移入 __main__);套件/腳本雙模式匯入。
try:
    from .estimate_relative_pose import (parse_cache_name, find_house_image,
                                         erp_to_bearing, ransac_E, sampson_residual)
except ImportError:
    for p in (ROOT, SCRIPT_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    from estimate_relative_pose import (parse_cache_name, find_house_image,
                                        erp_to_bearing, ransac_E, sampson_residual)

CACHE_DIR = (os.path.abspath(os.path.expanduser(os.environ["ROMA_GEO_CACHE"]))
             if os.environ.get("ROMA_GEO_CACHE")          # 同 config.cache_root
             else os.path.join(ROOT, "roma_match", "cache"))
CONF = 0.65          # 分析用門檻(cache 底床 0.1 的低段雜訊多)
FIT_MAX = 3000       # RANSAC fit 抽樣上限
THRESH_DEG = 0.5     # 對極內點角度門檻(同 estimate_relative_pose 預設)


def list_cache_by_house():
    by = {}
    for f in sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz"))):
        h, _, _ = parse_cache_name(f)
        by.setdefault(h, []).append(f)
    return by


def pick_pairs(by_house, probe=40):
    """3 棟(字母序首/中/尾),各取前 probe 檔中 conf>=CONF 點數最多的 pair。決定性選擇,可重跑。"""
    houses = sorted(by_house)
    picks = []
    for h in (houses[0], houses[len(houses) // 2], houses[-1]):
        best, best_k = None, -1
        for f in by_house[h][:probe]:
            m = np.load(f)["matches"]
            k = int((m[:, 4] >= CONF).sum()) if m.size else 0
            if k > best_k:
                best_k, best = k, f
        picks.append((h, best, best_k))
    return picks


def nn_median(P, Q, sub=2000, seed=0):
    """P 每點到 Q 的最近鄰歐氏距離中位數(P 抽樣至 sub;分塊暴力)。"""
    rng = np.random.default_rng(seed)
    if len(P) > sub:
        P = P[rng.choice(len(P), sub, replace=False)]
    p2 = (P ** 2).sum(1)[:, None]
    best = np.full(len(P), np.inf)
    for i in range(0, len(Q), 8192):
        c = Q[i:i + 8192]
        d2 = p2 + (c ** 2).sum(1)[None, :] - 2.0 * (P @ c.T)
        best = np.minimum(best, d2.min(1))
    return float(np.sqrt(max(np.median(best), 0.0)))


def fit_E(pa, pb, W, H, seed=0):
    da, db = erp_to_bearing(pa, W, H), erp_to_bearing(pb, W, H)
    rng = np.random.default_rng(seed)
    if len(da) > FIT_MAX:
        idx = rng.choice(len(da), FIT_MAX, replace=False)
        da, db = da[idx], db[idx]
    thr = np.sin(np.radians(THRESH_DEG))
    E, inl = ransac_E(da, db, thr)
    return E, float(inl.mean()), thr


def eval_E(E, thr, pa, pb, W, H):
    da, db = erp_to_bearing(pa, W, H), erp_to_bearing(pb, W, H)
    return float((sampson_residual(E, da, db) < thr).mean())


def spike_three_pairs(report_lines):
    from PIL import Image
    by_house = list_cache_by_house()
    picks = pick_pairs(by_house)
    print(f"[spike] 抽到 3 對(房子字母序 首/中/尾,各取 conf>={CONF} 點數最大 pair):")
    for h, f, k in picks:
        print(f"  {h}  {os.path.basename(f)}  (conf>={CONF}: {k} 點)")

    print("[spike] 載入 RoMaMatcher(use_cache=False,live 重跑)...")
    t0 = time.time()
    from roma_match.matcher import RoMaMatcher
    matcher = RoMaMatcher(conf_threshold=0.1, use_cache=False, model_type="roma_v2")
    print(f"[spike] 模型載入 {time.time()-t0:.1f}s")

    all_ok = True
    report_lines.append("| pair | K cache/live | max 比值 x;y | NN 中位數(A側,px) | 內點率 cache-E: c/l | live-E: l/c | 邊界 x min/max 同值 | 判定 |")
    report_lines.append("|---|---|---|---|---|---|---|---|")

    for h, f, _ in picks:
        _, na, nb = parse_cache_name(f)
        img_a, img_b = find_house_image(h, na), find_house_image(h, nb)
        img_a, img_b = os.path.relpath(img_a, os.getcwd()), os.path.relpath(img_b, os.getcwd())
        with Image.open(img_a) as im:
            W, H = im.size

        cache = np.load(f)["matches"].astype(float)
        t0 = time.time()
        live, _ = matcher.match_direct(img_a, img_b)
        live = np.asarray(live, float)
        dt = time.time() - t0
        c = cache[cache[:, 4] >= CONF]
        l = live[live[:, 4] >= CONF]
        print(f"\n=== {h} {na[:8]}->{nb[:8]}  (live 重跑 {dt:.1f}s) ===")
        print(f"  K(conf>={CONF}): cache {len(c)} | live {len(l)}")

        # 1) range/scale
        in_range = lambda m: bool((m[:, 0] >= 0).all() and (m[:, 0] <= W).all() and (m[:, 1] >= 0).all() and (m[:, 1] <= H).all()
                                  and (m[:, 2] >= 0).all() and (m[:, 2] <= W).all() and (m[:, 3] >= 0).all() and (m[:, 3] <= H).all())
        range_ok = in_range(c) and in_range(l)
        rx = c[:, 0].max() / max(l[:, 0].max(), 1e-9)
        ry = c[:, 1].max() / max(l[:, 1].max(), 1e-9)
        scale_ok = 0.8 < rx < 1.25 and 0.8 < ry < 1.25
        print(f"  range∈[0,{W}]x[0,{H}]: cache/live {in_range(c)}/{in_range(l)} | max 比值 x {rx:.4f}, y {ry:.4f}")

        # 2) NN
        nn_a = nn_median(c[:, :2], l[:, :2])
        nn_b = nn_median(c[:, 2:4], l[:, 2:4])
        nn_ok = nn_a < 40 and nn_b < 40
        print(f"  NN 中位數 cache→live: A側 {nn_a:.1f}px, B側 {nn_b:.1f}px (門檻 40px;同約定應為十幾 px)")

        # 3) E 交叉驗證
        E_c, self_c, thr = fit_E(c[:, :2], c[:, 2:4], W, H)
        cross_cl = eval_E(E_c, thr, l[:, :2], l[:, 2:4], W, H)
        E_l, self_l, _ = fit_E(l[:, :2], l[:, 2:4], W, H)
        cross_lc = eval_E(E_l, thr, c[:, :2], c[:, 2:4], W, H)
        weak = self_c < 0.15 or self_l < 0.15
        geom_ok = cross_cl >= 0.7 * self_c and cross_lc >= 0.7 * self_l
        print(f"  E 交叉內點率: cache-E 自身 {self_c:.3f} → live {cross_cl:.3f} | live-E 自身 {self_l:.3f} → cache {cross_lc:.3f}"
              + (" [低內點率,弱證據]" if weak else ""))

        # 4) 邊界指紋(參考)
        bnd = (abs(c[:, 0].min() - l[:, 0].min()) < 1e-6, abs(c[:, 0].max() - l[:, 0].max()) < 1e-6)
        print(f"  邊界指紋 x min/max: cache ({c[:,0].min():.2f}, {c[:,0].max():.2f}) live ({l[:,0].min():.2f}, {l[:,0].max():.2f}) 同值 {bnd}")

        ok = range_ok and scale_ok and nn_ok and geom_ok
        all_ok &= ok
        print(f"  => {'PASS' if ok else 'FAIL'}")
        report_lines.append(
            f"| {h.replace('DollhouseTask_','').replace('_NoOutdoor','')} {na[:8]}→{nb[:8]} | {len(c)}/{len(l)} | {rx:.3f};{ry:.3f} "
            f"| A {nn_a:.1f} / B {nn_b:.1f} | {self_c:.3f}→{cross_cl:.3f} | {self_l:.3f}→{cross_lc:.3f} "
            f"| {'✓' if all(bnd) else '✗'} | {'PASS' if ok else 'FAIL'}{'(弱)' if weak else ''} |")
    return all_ok


def scan_all(report_lines):
    """全量掃描:每檔座標必須落在該房原圖 [0,W]x[0,H] 內。"""
    from PIL import Image
    by_house = list_cache_by_house()
    n_files, n_bad, t0 = 0, [], time.time()
    house_wh = {}
    for h, files in sorted(by_house.items()):
        _, na, _ = parse_cache_name(files[0])
        with Image.open(find_house_image(h, na)) as im:
            house_wh[h] = im.size
        W, H = house_wh[h]
        for f in files:
            m = np.load(f)["matches"]
            n_files += 1
            if m.size == 0:
                continue
            if not (m[:, [0, 2]].min() >= 0 and m[:, [0, 2]].max() <= W and m[:, [1, 3]].min() >= 0 and m[:, [1, 3]].max() <= H):
                n_bad.append(os.path.basename(f))
    dt = time.time() - t0
    sizes = sorted(set(house_wh.values()))
    print(f"\n[scan] {n_files} 檔掃完 ({dt:.0f}s) | 原圖尺寸種類: {sizes} | 超界檔數: {len(n_bad)}")
    for f in n_bad[:5]:
        print("   超界:", f)
    report_lines.append(f"\n全量掃描:{n_files} 檔全數座標落在該房原圖範圍內(超界 {len(n_bad)} 檔);"
                        f"原圖尺寸種類 {sizes};耗時 {dt:.0f}s。")
    return len(n_bad) == 0


if __name__ == "__main__":
    os.chdir(ROOT)   # CLI 相對路徑以 repo 根為準(repo 上層路徑含中文,部分函式庫吃不了)
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_scan", action="store_true", help="跳過全量範圍掃描")
    a = ap.parse_args()

    lines = ["# cache npz 座標約定 spike 報告()", "",
             f"日期:{time.strftime('%Y-%m-%d %H:%M')};方法與判準見 `spike_cache_consistency.py` docstring。", ""]
    ok = spike_three_pairs(lines)
    if not a.skip_scan:
        ok &= scan_all(lines)
    lines += ["", f"**總判定:{'PASS：cache 座標約定 = 原圖 ERP 像素 (x_a,y_a,x_b,y_b,conf),與 live 重跑同約定' if ok else 'FAIL：有不一致,見上表'}**"]
    out = os.path.join(SCRIPT_DIR, "spike_cache_consistency_report.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[spike] 報告已寫 {out}\n總判定: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
