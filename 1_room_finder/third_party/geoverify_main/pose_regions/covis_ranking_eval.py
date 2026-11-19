#!/usr/bin/env python3
"""
共視/幾何驗證訊號當「連通性排序分數」的判別力評估(2026-07-26)。

問題:粗篩存活、被 RoMa 跑過的 pair 裡,幾何驗證後的共視證據
(inlier 數/率、共視區域)能不能把「真正連通」的 pair 排到前面?
2026-07-18 曾用純 RoMa match_count 排序 → 分不出明顯界線。之後多了對極驗證與
共視區域,本腳本重測一次,並回答「上限在哪」。

族群 = Output/roma_regions/_summary.csv 全部列(= coarse filter R90 存活者)。
GT = datasets/*_HOTSPOT.json 完整鄰接(不用 metadata/*.csv,那份只覆蓋 68% 的 cache pair),
     另以 BFS 算 topology distance,供「殘餘 FP 落在第幾層鄰居」的組成分析。
無位姿(no_covisibility)的 pair 幾何分數一律 0(誠實排在最後),不排除。排除會
把覆蓋率問題藏起來。

對照的分數:
  coarse refined / base logit   現役模型的對稱 logit(注意:此族群本來就由它挑出,
                                分數範圍被截斷,AP 不可與全矩陣評測相比)
  RoMa n_conf / n_raw           匹配點數(2026-07-18 用過的訊號)
  幾何 n_inlier / inlier_rate   對極 RANSAC 驗證後的內點數/率
  區域 region1_frac / n_regions 共視區域量
  組合                          rank 平均(免訓練)與 house-grouped 5-fold logistic(OOF)

用法:
  python pose_regions/covis_ranking_eval.py
輸出:
  pose_regions/covis_ranking_report.md          結果報告(自動生成)
  Output/roma_regions/_ranking_eval.csv         每 pair 一列(含 topology distance)
"""
import argparse
import csv
import json
import os
from collections import defaultdict, deque

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _gt_adjacency(house_dir):
    """datasets/<house> 底下所有 *_HOTSPOT.json → 對稱鄰接(鍵去掉 .jpg 副檔名)。

    就地實作,以免本腳本相依粗篩套件(那條匯入路徑會拉進 torch)。
    """
    from pathlib import Path
    adj = defaultdict(set)
    for h_file in Path(house_dir).rglob("*_HOTSPOT.json"):
        if os.path.getsize(h_file) == 0:
            continue
        try:
            with open(h_file, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        for room in data.get("HOTSPOTOFROOM", []):
            a = room.get("IDName")
            for b in room.get("ToIDName", {}).get("IDName", []):
                adj[a].add(b)
                adj[b].add(a)
    strip = lambda s: s[:-4] if s.endswith(".jpg") else s
    return {strip(k): {strip(v) for v in vs} for k, vs in adj.items()}


def _bfs(adj, src):
    d, q = {src: 0}, deque([src])
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in d:
                d[v] = d[u] + 1
                q.append(v)
    return d


def _load_matrix(score_dir, house):
    p = os.path.join(score_dir, f"{house}.npy")
    ip = os.path.join(score_dir, f"{house}_images.json")
    if not (os.path.exists(p) and os.path.exists(ip)):
        return None, None
    with open(ip, encoding="utf-8") as f:
        imgs = json.load(f)
    idx = {os.path.splitext(os.path.basename(q.replace("\\", "/")))[0]: i
           for i, q in enumerate(imgs)}
    return np.load(p), idx


def _f(v, default=0.0):
    return default if v in ("", None) else float(v)


# ---------- 評估工具 ----------
def sweep(y, s):
    """分數降冪、同分一起納入;每個 tie-group 邊界回傳 (N, TP, FP, precision, recall)。"""
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    n, npos = len(y), int(y.sum())
    out, i = [], 0
    while i < n:
        j = i + 1
        while j < n and ss[j] == ss[i]:
            j += 1
        tp = int(ys[:j].sum())
        out.append((j, tp, j - tp, tp / j, tp / npos))
        i = j
    return out


def average_precision(y, s):
    """tie-aware AP:每個 tie-group 以其結束時的 precision 乘上 recall 增量。"""
    ap, prev = 0.0, 0.0
    for _, _, _, p, r in sweep(y, s):
        ap += p * (r - prev)
        prev = r
    return ap


def p_at_r(rows, target):
    for n, tp, fp, p, r in rows:
        if r >= target:
            return dict(N=n, TP=tp, FP=fp, P=p, R=r)
    return None


def rank01(v):
    return np.argsort(np.argsort(v, kind="mergesort"), kind="mergesort") / (len(v) - 1)


def logistic_oof(X, y, groups, folds=5, iters=400, lr=0.5, l2=1e-3, seed=0):
    """house-grouped 5-fold 的小型 logistic(純 numpy 梯度下降);回傳 out-of-fold 分數。

    折的分配只依房名排序後輪流,與訓練用 fold_map 無關。這裡只是要一個「組合權重
    沒看過該房」的乾淨估計,coarse 分數本身已是 OOF。
    """
    gs = sorted(set(groups))
    fold_of = {g: i % folds for i, g in enumerate(gs)}
    fid = np.array([fold_of[g] for g in groups])
    Xz = np.hstack([(X - X.mean(0)) / (X.std(0) + 1e-9), np.ones((len(X), 1))])
    oof = np.zeros(len(y))
    for k in range(folds):
        tr, te = fid != k, fid == k
        w = np.zeros(Xz.shape[1])
        Xt, yt = Xz[tr], y[tr].astype(float)
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-Xt @ w))
            w -= lr * (Xt.T @ (p - yt) / len(yt) + l2 * w)
        oof[te] = Xz[te] @ w
    return oof


# ---------- 路徑（統一走套件內 config）----------
def _datasets_root():
    from roma_geo.config import datasets_root
    return datasets_root()


def _matrix_dir(variant):
    from roma_geo.config import matrices_root
    return matrices_root(variant)


def _regions_dir():
    from roma_geo.config import regions_root
    return regions_root()


def write_csv(path, rec):
    """輸出 pair-level CSV,供 prepare_house_data 的佇列排序使用。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["house", "img_a", "img_b", "y", "topo_d", "same_room", "posed",
                    "conf_used", "coarse_refined", "n_conf", "n_inlier", "inlier_rate",
                    "n_regions"])
        for q in rec:
            w.writerow([q["house"], q["img_a"], q["img_b"], q["y"], q["d"], q["same_room"],
                        q["posed"], q["conf_used"], f"{q['coarse']:.4f}", int(q["n_conf"]),
                        int(q["n_inlier"]), q["inlier_rate"], int(q["n_regions"])])


# ---------- 主流程 ----------
def build_records(summary_csv, datasets=None, refined_dir=None, oof_dir=None):
    # 矩陣來源由呼叫端傳入，未傳入才讀套件設定。
    datasets = datasets or _datasets_root()
    refined_dir = refined_dir or _matrix_dir("refined")
    oof_dir = oof_dir or _matrix_dir("oof")
    rows = list(csv.DictReader(open(summary_csv, encoding="utf-8")))
    houses = sorted({r["house"] for r in rows})
    graphs = {h: _gt_adjacency(os.path.join(datasets, h)) for h in houses}
    dists = {h: {u: _bfs(g, u) for u in g} for h, g in graphs.items()}
    mats = {}
    for h in houses:
        mats[h] = {
            "refined": _load_matrix(refined_dir, h),
            "oof": _load_matrix(oof_dir, h),
        }

    rec, skipped = [], defaultdict(int)
    for r in rows:
        h, a, b = r["house"], r["img_a"], r["img_b"]
        g = graphs[h]
        if a not in g or b not in g:
            skipped["不在 HOTSPOT 圖中"] += 1
            continue
        M, idx = mats[h]["refined"]
        if M is None or a not in idx or b not in idx:
            skipped["無 coarse 矩陣"] += 1
            continue
        i, j = idx[a], idx[b]
        Mb, idxb = mats[h]["oof"]
        rec.append(dict(
            house=h, img_a=a, img_b=b,
            y=int(b in g[a]), d=dists[h][a].get(b, 99),
            same_room=r["same_room"], conf_used=r["conf_used"],
            posed=int(r["flag_no_covisibility"] != "1" and r["conf_used"] != ""),
            coarse=0.5 * (float(M[i, j, 0]) + float(M[j, i, 0])),
            coarse_base=(0.5 * (float(Mb[idxb[a], idxb[b], 0]) + float(Mb[idxb[b], idxb[a], 0]))
                         if Mb is not None and a in idxb and b in idxb else float("nan")),
            n_raw=_f(r["n_raw"]), n_conf=_f(r["n_conf"]),
            n_inlier=_f(r["n_inlier"]), inlier_rate=_f(r["inlier_rate"]),
            n_regions=_f(r["n_regions"]), region1_frac=_f(r["region1_frac"]),
        ))
    return rec, skipped, len(houses)


def run_ranking_eval(summary_csv, out_csv, datasets=None, refined_dir=None, oof_dir=None):
    """寫出 _ranking_eval.csv。回傳列數與略過原因。"""
    rec, skipped, n_houses = build_records(
        summary_csv, datasets=datasets, refined_dir=refined_dir, oof_dir=oof_dir)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    write_csv(out_csv, rec)
    return {"n_rows": len(rec), "n_houses": n_houses, "skipped": dict(skipped), "csv": out_csv}


def main():
    ap = argparse.ArgumentParser()
    # 產出位置可用 ROMA_GEO_OUTPUT 覆寫;換矩陣版本設 config.MATRICES_VERSION 或
    # ROMA_GEO_MATRICES(見 matrices/README.md)。
    reg = _regions_dir()
    ap.add_argument("--summary", default=os.path.join(reg, "_summary.csv"))
    ap.add_argument("--report", default=os.path.join(SCRIPT_DIR, "covis_ranking_report.md"))
    ap.add_argument("--csv", default=os.path.join(reg, "_ranking_eval.csv"))
    a = ap.parse_args()

    rec, skipped, n_houses = build_records(a.summary)

    # 先寫 CSV 再做統計:後面的比率計算在小樣本會除以零,不應連帶影響 CSV。
    write_csv(a.csv, rec)
    if not rec:
        print("[warn] 沒有任何可用的配對記錄,只寫出空的 %s,略過報告。" % a.csv)
        return
    y = np.array([q["y"] for q in rec])
    d = np.array([q["d"] for q in rec])
    posed = np.array([q["posed"] for q in rec])
    house = [q["house"] for q in rec]
    n, npos = len(y), int(y.sum())
    S = lambda k: np.array([q[k] for q in rec], float)

    cands = {
        "coarse refined logit(現役)": S("coarse"),
        "coarse base logit": np.nan_to_num(S("coarse_base"), nan=-1e30),
        "RoMa n_conf(@0.65 匹配數)": S("n_conf"),
        "RoMa n_raw(@0.1 匹配數)": S("n_raw"),
        "幾何 n_inlier": S("n_inlier"),
        "幾何 inlier_rate": S("inlier_rate"),
        "區域 region1_frac": S("region1_frac"),
        "區域 n_regions": S("n_regions"),
    }
    comb_rank = (rank01(S("coarse")) + rank01(S("n_inlier"))) / 2
    X = np.column_stack([S("coarse"), np.log1p(S("n_inlier")), S("inlier_rate"),
                         np.log1p(S("n_conf")), posed.astype(float), S("region1_frac")])
    comb_lr = logistic_oof(X, y, house)
    cands["組合:rank(coarse)+rank(n_inlier) 平均"] = comb_rank
    cands["組合:logistic(coarse+幾何, house-5fold OOF)"] = comb_lr
    cands["幾何 only:logistic(house-5fold OOF)"] = logistic_oof(X[:, 1:], y, house)

    L = []
    L.append("# 共視/幾何訊號當連通性排序分數:判別力與上限(2026-07-26)\n")
    L.append(f"族群 = `{a.summary}` 全部 **{n:,} 對**(coarse filter R90 存活者,{n_houses} 房),"
             f"正例 **{npos:,}({100*npos/n:.1f}%)**。"
             f"GT 由 `datasets/*_HOTSPOT.json` 完整鄰接重建 + BFS topology distance。"
             + (f"略過:{dict(skipped)}。" if skipped else "略過:0。"))
    L.append(f"\n有位姿的 pair **{int(posed.sum()):,}({100*posed.mean():.1f}%)**;"
             f"正例中有位姿 {int(y[posed == 1].sum()):,}/{npos:,}"
             f"({100*y[posed == 1].sum()/npos:.1f}%),"
             f"有任何 inlier 的正例 {int(((S('n_inlier') > 0) & (y == 1)).sum()):,}"
             f"({100*((S('n_inlier') > 0) & (y == 1)).sum()/npos:.1f}%)。\n")

    L.append("## 1. 分數對照(全族群)\n")
    L.append("| 分數 | AP | P@R90 | P@R85 | FP@R90 |")
    L.append("|---|---|---|---|---|")
    res = {}
    for name, s in cands.items():
        rws = sweep(y, s)
        res[name] = dict(ap=average_precision(y, s), o90=p_at_r(rws, 0.90),
                         o85=p_at_r(rws, 0.85), s=s)
        r = res[name]
        L.append(f"| {name} | {r['ap']:.3f} | {100*r['o90']['P']:.1f}% | "
                 f"{100*r['o85']['P']:.1f}% | {r['o90']['FP']:,} |")
    base = npos / n
    L.append(f"| *(不排序,全留)* | *{base:.3f}* | *{100*base:.1f}%* | *{100*base:.1f}%* "
             f"| *{n-npos:,}* |")
    L.append("\n**注意**:此族群本來就是 coarse filter 自己挑出來的(分數範圍被截斷),"
             "它的 AP 不可與全矩陣評測的 AP 相比,也不可讀成「幾何贏過模型」。\n")

    L.append("## 2. 上限在哪:按 topology distance 切開\n")
    scen = [("d1 vs d2(最難的牆)", (d == 1) | (d == 2)),
            ("d1 vs d3 以上(較遠的負例)", (d == 1) | (d >= 3)),
            ("d1 vs d2,且兩邊都有位姿", ((d == 1) | (d == 2)) & (posed == 1))]
    L.append("| 子族群 | n | 正例率 | coarse AP | 幾何 n_inlier AP | 組合(rank)AP |")
    L.append("|---|---|---|---|---|---|")
    for nm, m in scen:
        yy = y[m]
        cr = (rank01(S("coarse")[m]) + rank01(S("n_inlier")[m])) / 2
        L.append(f"| {nm} | {int(m.sum()):,} | {100*yy.mean():.1f}% | "
                 f"{average_precision(yy, S('coarse')[m]):.3f} | "
                 f"{average_precision(yy, S('n_inlier')[m]):.3f} | "
                 f"{average_precision(yy, cr):.3f} |")
    L.append("")
    L.append("| topology distance | n | 正例 | 有位姿率 | n_inlier 中位 |")
    L.append("|---|---|---|---|---|")
    for k in sorted(set(d.tolist())):
        m = d == k
        lab = "∞(不連通)" if k == 99 else f"d{k}"
        L.append(f"| {lab} | {int(m.sum()):,} | {int(y[m].sum()):,} | "
                 f"{100*posed[m].mean():.1f}% | {np.median(S('n_inlier')[m]):.0f} |")

    L.append("\n## 3. 組合到底移除了哪些 FP\n")
    def topset(s, t=0.90):
        for nn, tp, fp, p, r in sweep(y, s):
            if r >= t:
                return np.argsort(-s, kind="mergesort")[:nn], fp
        return None, None
    A, fa = topset(S("coarse"))
    B, fb = topset(comb_rank)
    if not fa or not fb or not npos:
        # 樣本不足以湊出 R≥0.90 的操作點,跳過 FP 組成分析。
        L.append("\n（樣本不足,略過 FP 組成與覆蓋率分析。）\n")
        with open(a.report, "w", encoding="utf-8") as f:
            f.write("\n".join(L) + "\n")
        print("\n".join(L))
        print(f"\nwrote {a.report}\nwrote {a.csv} ({len(rec):,} rows)")
        return
    fpA, fpB = set(A[y[A] == 0].tolist()), set(B[y[B] == 0].tolist())
    rem = np.array(sorted(fpA - fpB))
    add = np.array(sorted(fpB - fpA))
    compA = {int(k): int((d[A[y[A] == 0]] == k).sum()) for k in sorted(set(d[A[y[A] == 0]].tolist()))}
    compB = {int(k): int((d[B[y[B] == 0]] == k).sum()) for k in sorted(set(d[B[y[B] == 0]].tolist()))}
    fmt = lambda c, tot: "、".join(f"{'d∞' if k == 99 else 'd'+str(k)} {v}({100*v/tot:.0f}%)"
                                   for k, v in list(c.items())[:4]) + " …"
    L.append(f"R≥0.90 操作點:coarse 單獨 FP **{fa:,}** → 組合 FP **{fb:,}**(淨減 {fa-fb:,},"
             f"−{100*(fa-fb)/fa:.0f}%)。")
    L.append(f"\n- coarse FP 組成:{fmt(compA, fa)}(d2 佔 {100*compA.get(2,0)/fa:.0f}%)")
    L.append(f"- 組合 FP 組成:{fmt(compB, fb)}(d2 佔 **{100*compB.get(2,0)/fb:.0f}%**)")
    L.append(f"- 被移除的 {len(rem):,} 個 FP 中,**d3 以上佔 {100*(d[rem] >= 3).mean():.0f}%**,"
             f"d2 只佔 {100*(d[rem] == 2).mean():.0f}%;新進 FP {len(add):,} 個。")
    L.append("\n**結論**:幾何驗證清掉的幾乎都是遠距離負例;在 d2(看得到、走不到)這道牆上,"
             "它與 coarse filter 量的是同一件事,提供不了新資訊。殘餘誤差被濃縮到牆上,"
             "而不是穿過牆。共視類證據到此榨乾,要再往上需要的是**可達性**證據。\n")
    L.append(f"**覆蓋率另一道限制**:只有 {100*((S('n_inlier') > 0) & (y == 1)).sum()/npos:.1f}% "
             f"的正例拿得到任何 inlier、{100*y[posed == 1].sum()/npos:.1f}% 拿得到位姿,"
             "所以幾何分數單獨用時 recall 上不了 90%(要衝 90% 得收進整個零分群,"
             "精確度掉回基準率。表 1 的 P@R90 欄就是這個現象)。"
             "它只能當特徵,不能當獨立過濾器。\n")

    with open(a.report, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {a.report}\nwrote {a.csv} ({len(rec):,} rows)")


if __name__ == "__main__":
    main()
