"""
[腳本用途]
跨環境一致性檢查：在目前環境重跑一批配對，與 `Output/roma_regions/_summary.csv`
的既有結果逐對比較，回答「換到這個環境之後，幾何結論還一不一樣」。

經對外入口 `roma_geo.verify_pairs()` 執行。

比較的是幾何結論（RoMa 的取樣帶隨機性，不同 torch 版本的浮點結果
亦有微差）。判準為：

  1. 位姿有無的一致率：兩邊都有位姿 / 都沒有位姿
  2. 共視區域數的一致率
  3. 通道方位角的圓角差：中位數與 ≤5° 的比例

其中 1 與 3 決定 PASS / CHECK。⚠ 3 只看「兩邊 n_inlier 皆 ≥ MIN_INLIER_FOR_AZ_GATE」
的配對；低證據的配對逐次執行本來就會跳動，放進判準會讓這個檢查頻繁誤報
（理由與實測數字見該常數的說明）。

[用法]
  python environments/cross_env_check.py --n 30
  python environments/cross_env_check.py --n 30 --tag kaggle-t4 --out_dir /kaggle/working

  --n         取樣對數（決定性等距抽樣；相同 --n 在任何環境都抽到同一批）
  --limit_gt  只取 GT 連通的配對（訊號較強，適合小樣本快速檢查）
  --no_cache  關閉 RoMa 匹配快取（預設關閉，確保真的重算）
"""
import argparse
import csv
import json
import os
import statistics as st
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SUMMARY = os.path.join(ROOT, "Output", "roma_regions", "_summary.csv")

# 判準門檻：容許一定比例的位姿有無翻面。
MIN_POSED_AGREE = 0.85
MAX_MEDIAN_AZ_DEG = 5.0

# 方位角判準只看「證據充足」的配對：兩邊 n_inlier 皆 ≥ 此值。
#
# 理由是統計量的可測性。抽 30 對時兩邊都有位姿的約 12 對，而方位角差是重尾分佈
# （多數 <1°、少數數十度），12 個樣本的中位數極不穩定：以全量資料 bootstrap，
# 母體中位 1.99° 明明遠低於門檻，卻有 25.6% 的機率抽出 >5° 而誤報。
# 改看強證據子集後，母體中位降到 0.63°、誤報率降至 1.6%（n=30）與 0.1%（n=60）。
#
# 這與下游的既有規則一致：低 inlier 與 conf 0.3 撈回層本來就被列為必須 gate 的對象
# （見 pose_regions/README.md），不應拿來當環境是否正常的判斷依據。
MIN_INLIER_FOR_AZ_GATE = 150


def circ(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def num(s):
    s = (s or "").strip()
    return float(s) if s else None


def load_reference():
    if not os.path.exists(SUMMARY):
        raise SystemExit("找不到參考結果 %s" % SUMMARY)
    with open(SUMMARY, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (r["house"], r["img_a"], r["img_b"]))
    return rows


def sample(rows, n):
    """決定性等距抽樣：相同 n 在任何環境都取到同一批，結果才可逐對相比。"""
    if n >= len(rows):
        return rows
    step = len(rows) / float(n)
    return [rows[int(i * step)] for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description="跨環境一致性檢查")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--tag", default="current")
    ap.add_argument("--out_dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--limit_gt", action="store_true", help="只取 GT 連通的配對")
    ap.add_argument("--no_cache", action="store_true", default=True)
    ap.add_argument("--use_cache", dest="no_cache", action="store_false")
    args = ap.parse_args()

    from pose_regions import find_house_image
    from roma_geo import verify_pairs, DEFAULTS

    rows = load_reference()
    if args.limit_gt:
        rows = [r for r in rows if r.get("gt_label") == "1"]
    picked = sample(rows, args.n)

    # 解析影像路徑；找不到圖的配對直接略過（資料集可能只掛了部分房子）
    tasks, refs, missing = [], [], 0
    for r in picked:
        pa = find_house_image(r["house"], r["img_a"])
        pb = find_house_image(r["house"], r["img_b"])
        if not (pa and pb):
            missing += 1
            continue
        tasks.append((pa, pb))
        refs.append(r)

    if not tasks:
        raise SystemExit("沒有任何配對的影像可解析；請確認資料集路徑（見 config.py）。")

    print("=" * 72)
    print("  跨環境一致性檢查 | tag=%s | %d 對（%d 對因缺圖略過）"
          % (args.tag, len(tasks), missing))
    print("  參數：%s" % json.dumps(DEFAULTS, ensure_ascii=False))
    print("=" * 72, flush=True)

    recs = []
    t_start = time.time()
    for i, res in enumerate(verify_pairs(tasks, use_cache=not args.no_cache,
                                         keep_inliers=False)):
        r = refs[i]
        ref_posed = (r.get("conf_used") or "").strip() != ""
        new_posed = res.get("pose") is not None
        pose = res.get("pose") or {}
        ref_ab, ref_ba = num(r.get("az_ab_deg")), num(r.get("az_ba_deg"))
        d_ab = circ(pose["az_ab_deg"], ref_ab) if (new_posed and ref_ab is not None) else None
        d_ba = circ(pose["az_ba_deg"], ref_ba) if (new_posed and ref_ba is not None) else None
        recs.append({
            "house": r["house"], "img_a": r["img_a"], "img_b": r["img_b"],
            "gt_label": r.get("gt_label", ""),
            "ref_posed": int(ref_posed), "new_posed": int(new_posed),
            "ref_n_regions": r.get("n_regions", ""), "new_n_regions": len(res.get("regions") or []),
            "ref_inlier_rate": r.get("inlier_rate", ""),
            "new_inlier_rate": res.get("inlier_rate"),
            "ref_n_inlier": r.get("n_inlier", ""),
            "new_n_inlier": (res.get("n") or {}).get("inlier", 0),
            "d_az_ab_deg": d_ab, "d_az_ba_deg": d_ba,
            "match_s": res["timing"]["match_s"], "verify_s": res["timing"]["verify_s"],
        })
        if (i + 1) % 10 == 0:
            print("  ... %d/%d" % (i + 1, len(tasks)), flush=True)
    elapsed = time.time() - t_start

    # ---- 彙總 ----
    n = len(recs)
    posed_agree = sum(1 for x in recs if x["ref_posed"] == x["new_posed"])
    both_posed = [x for x in recs if x["ref_posed"] and x["new_posed"]]
    region_agree = sum(1 for x in both_posed
                       if str(x["ref_n_regions"]) == str(x["new_n_regions"]))
    dz = [x["d_az_ab_deg"] for x in both_posed if x["d_az_ab_deg"] is not None]
    dz += [x["d_az_ba_deg"] for x in both_posed if x["d_az_ba_deg"] is not None]
    med_az = st.median(dz) if dz else None
    within5 = (sum(1 for d in dz if d <= 5.0) / len(dz)) if dz else None

    # 判準用的子集：兩邊 n_inlier 皆 ≥ MIN_INLIER_FOR_AZ_GATE（理由見該常數的說明）。
    strong = [x for x in both_posed
              if (num(x["ref_n_inlier"]) or 0) >= MIN_INLIER_FOR_AZ_GATE
              and (x["new_n_inlier"] or 0) >= MIN_INLIER_FOR_AZ_GATE]
    sz = [x["d_az_ab_deg"] for x in strong if x["d_az_ab_deg"] is not None]
    sz += [x["d_az_ba_deg"] for x in strong if x["d_az_ba_deg"] is not None]
    med_az_strong = st.median(sz) if sz else None
    # 計時僅供參考；要引用的效能數字用 local_timing_bench.py。
    warm = recs[2:] if len(recs) > 4 else recs          # 排除前兩對的 CUDA 暖機
    m_times = [x["match_s"] for x in warm]
    v_times = [x["verify_s"] for x in warm]

    print("\n--- 結果 ---")
    print("  位姿有無一致      : %d/%d (%.1f%%)" % (posed_agree, n, 100.0 * posed_agree / n))
    print("  兩邊都有位姿      : %d 對" % len(both_posed))
    if both_posed:
        print("  共視區域數一致    : %d/%d (%.1f%%)"
              % (region_agree, len(both_posed), 100.0 * region_agree / len(both_posed)))
    if med_az is not None:
        print("  方位角差 中位     : %.2f°   ≤5° 比例 %.1f%%（全體，僅供參考）"
              % (med_az, 100.0 * within5))
    if med_az_strong is not None:
        print("  方位角差 中位     : %.2f°（**判準用**：兩邊 n_inlier ≥ %d 的 %d 對）"
              % (med_az_strong, MIN_INLIER_FOR_AZ_GATE, len(strong)))
    else:
        print("  方位角差 中位     : 本次抽樣沒有 n_inlier ≥ %d 的配對，方位角一項不判"
              % MIN_INLIER_FOR_AZ_GATE)
    print("  單對耗時 中位     : 匹配 %.3fs + 驗證 %.3fs（僅供參考，排除前兩對暖機）"
          % (st.median(m_times), st.median(v_times)))
    print("  總耗時            : %.1fs" % elapsed)

    ok = (posed_agree / n >= MIN_POSED_AGREE
          and (med_az_strong is None or med_az_strong <= MAX_MEDIAN_AZ_DEG))
    print("\n" + "=" * 72)
    print("  結論：%s" % ("PASS：本環境與參考結果的幾何結論一致"
                          if ok else "CHECK：未達判準，見上方數字"))
    print("  判準：位姿一致率 ≥%.0f%% 且 方位角差中位 ≤%.0f°"
          "（後者只看兩邊 n_inlier ≥ %d 的配對）"
          % (100 * MIN_POSED_AGREE, MAX_MEDIAN_AZ_DEG, MIN_INLIER_FOR_AZ_GATE))
    print("=" * 72)

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "cross_env_check_%s.csv" % args.tag)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print("逐對明細 -> %s" % out_csv)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
