"""
[腳本用途]
批次填滿 RoMa 匹配快取。這是完整鏈條的第一個重活：

    batch_match.py            → roma_match/cache/*.npz          （本腳本）
    batch_pose_from_cache.py  → Output/roma_regions/*.json + _summary.csv
    covis_ranking_eval.py     → Output/roma_regions/_ranking_eval.csv
    prepare_house_data.py     → stitch_tool/data/*.json
    prepare_semantic_doors.py → stitch_tool/semantic/*.json

配對來源優先序：`--pairs` > `_pairs.csv` > 從 `_summary.csv` 快照一份。

`_pairs.csv` 有兩種來源：
  1. `environments/select_pairs.py`：從連通性矩陣按門檻挑出（換上游模型走這條）
  2. 從既有 `_summary.csv` 快照（粗篩存活者）。與既有結果同一族群，
     重跑後才能逐對比較，驗證遷移正確性時用這條。

**多 GPU**：偵測到多張 GPU 時走 `roma_match.parallel_matcher.ParallelScorer`，
每張卡一個 worker（`CUDA_VISIBLE_DEVICES` 隔離），可在雙 T4 環境約略減半總時間。
單張卡則走序列路徑（省去多進程開銷，且進度回報較細）。

**可續跑**：已存在對應 npz 的配對會直接略過，因此工作階段中斷後重跑即可接續
（需把上一輪的 roma_cache.zip 掛回來或留在 working 目錄）。

快取檔名由 `roma_match.matcher._get_cache_info()` 決定，本腳本不自行拼字串，
避免與讀取端的約定分歧。

[用法]
  python environments/batch_match.py                     # 全部，自動用上所有 GPU
  python environments/batch_match.py --limit 200         # 只做前 200 對（決定性順序）
  python environments/batch_match.py --workers 1         # 強制序列（除錯用）
  python environments/batch_match.py --time_budget_h 8   # 逼近時限就停，保留已完成的
"""
import argparse
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from roma_geo.config import regions_root                   # noqa: E402

REGIONS_DIR = regions_root()                    # 可用 ROMA_GEO_OUTPUT 覆寫，見 config.py
SUMMARY = os.path.join(REGIONS_DIR, "_summary.csv")

# 配對來源與批次產物分開存放:batch_pose_from_cache.py 會把 _summary.csv 重寫成
# 「當下快取有幾對」,直接拿它當來源會在續跑時遺失尚未完成的配對。
# 首次執行時另存一份快照,之後只讀這一份。
PAIRS = os.path.join(REGIONS_DIR, "_pairs.csv")


def ensure_pair_source(explicit=None):
    """回傳穩定的配對來源路徑;必要時從 _summary.csv 建立一份快照。"""
    if explicit:
        return explicit
    if os.path.exists(PAIRS):
        return PAIRS
    if not os.path.exists(SUMMARY):
        raise SystemExit(
            "找不到配對來源:%s 與 %s 都不存在。\n"
            "        _summary.csv 應隨 repo 入庫,請確認 clone 完整;\n"
            "        或用 environments/select_pairs.py 從連通性矩陣挑一份出來。" % (PAIRS, SUMMARY))
    with open(SUMMARY, encoding="utf-8", newline="") as fi:
        rows = list(csv.DictReader(fi))
    os.makedirs(REGIONS_DIR, exist_ok=True)
    with open(PAIRS, "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["house", "img_a", "img_b"])
        for r in rows:
            w.writerow([r["house"], r["img_a"], r["img_b"]])
    print("[init] 已從 _summary.csv 建立配對來源快照:%s（%d 對）" % (PAIRS, len(rows)))
    return PAIRS


def load_tasks(summary_csv, limit, mode, matcher):
    """讀配對來源，解析影像路徑，並濾掉已有快取的配對。

    回傳 (tasks, n_hit, n_miss)：tasks 為 [(img_a, img_b), ...]。
    """
    from pose_regions import find_house_image

    if not os.path.exists(summary_csv):
        raise SystemExit("找不到配對來源 %s" % summary_csv)
    with open(summary_csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (r["house"], r["img_a"], r["img_b"]))
    if limit:
        rows = rows[:limit]

    cache_mode = "romascan" if mode == "core" else mode
    tasks, n_hit, n_miss = [], 0, 0
    for r in rows:
        pa = find_house_image(r["house"], r["img_a"])
        pb = find_house_image(r["house"], r["img_b"])
        if not (pa and pb):
            n_miss += 1
            continue
        # 用 matcher 自己的命名規則判斷是否已有快取，不自行拼檔名。
        cache_path, _ = matcher._get_cache_info(pa, pb, cache_mode)
        if os.path.exists(cache_path):
            n_hit += 1
            continue
        tasks.append((pa, pb))
    return tasks, n_hit, n_miss


def run_sequential(tasks, matcher, mode, budget, t0):
    """單 GPU 序列路徑。進度回報較細，也不必付多進程的啟動成本。"""
    n_done = n_fail = 0
    for i, (pa, pb) in enumerate(tasks, 1):
        try:
            if mode == "direct":
                matcher.match_direct(pa, pb)
            else:
                matcher.match_core(pa, pb)
            n_done += 1
        except Exception as e:                               # noqa: BLE001
            n_fail += 1
            print("  [fail] %s / %s: %s: %s"
                  % (os.path.basename(pa)[:8], os.path.basename(pb)[:8],
                     type(e).__name__, e), flush=True)

        if n_done and n_done % 25 == 0:
            el = time.time() - t0
            rate = el / n_done
            print("  [%d/%d] 已用 %.1f min · %.2f s/對 · 預估剩餘 %.1f min"
                  % (i, len(tasks), el / 60, rate, rate * (len(tasks) - i) / 60), flush=True)

        if budget and (time.time() - t0) > budget:
            print("\n[stop] 達到時間預算，於第 %d/%d 對停止；已完成的快取保留，"
                  "重跑本腳本會接續。" % (i, len(tasks)), flush=True)
            break
    return n_done, n_fail


def run_parallel(tasks, n_workers, mode, budget):
    """多 GPU 路徑：每張卡一個 worker，由 ParallelScorer 分派。

    時間預算以分批方式套用：run_stage 為阻塞式，故切成數批，每批結束時檢查預算。
    """
    from roma_match.parallel_matcher import ParallelScorer

    scorer = ParallelScorer(workers=n_workers, mode=mode, model_type="roma_v2",
                            conf=0.1, use_cache=True)
    print("  使用 %d 個 worker / %d 張 GPU" % (scorer.n_workers, scorer.num_gpus), flush=True)

    # 分批以取得時間預算的檢查點(run_stage 阻塞,中途無法中止)。
    # 每批重新 spawn worker 並重載權重,故批次不宜過小。
    chunk = max(scorer.n_workers * 200, 200)
    n_done = n_fail = 0
    t0 = time.time()
    for s in range(0, len(tasks), chunk):
        part = tasks[s:s + chunk]
        got = scorer.run_stage(part, stage="match")
        n_done += len(got)
        n_fail += len(part) - len(got)      # run_stage 只回傳成功項,差額即失敗數
        el = time.time() - t0
        rate = el / max(n_done, 1)
        print("  [%d/%d] 已用 %.1f min · %.2f s/對(整體) · 預估剩餘 %.1f min"
              % (min(s + chunk, len(tasks)), len(tasks), el / 60, rate,
                 rate * (len(tasks) - s - len(part)) / 60), flush=True)
        if budget and el > budget:
            print("\n[stop] 達到時間預算，已完成 %d/%d 對；快取保留，重跑會接續。"
                  % (min(s + chunk, len(tasks)), len(tasks)), flush=True)
            break
    return n_done, n_fail


def aggregate_scores(summary_csv, limit, conf_threshold=0.65):
    """從快取彙總每對的 RoMa 證據強度，寫入 roma_match/roma_scores.npz。

    `stitch_tool/prepare_house_data.py` 讀這個檔補上 roma_match_count /
    roma_conf_sum / roma_conf_mean 欄位。

    只讀快取、不跑模型：matcher 以 workers=1 建構，每對都應命中快取，
    缺漏時才退化成一次即時匹配。conf_sum = Σ conf_i，門檻 conf ≥ 0.65。
    """
    import numpy as np
    from roma_match.matcher import RoMaMatcher
    from roma_match.roma_score_cache import RomaScoreCache
    from pose_regions import find_house_image

    with open(summary_csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (r["house"], r["img_a"], r["img_b"]))
    if limit:
        rows = rows[:limit]

    matcher = RoMaMatcher(device="cpu", conf_threshold=conf_threshold,
                          use_cache=True, model_type="roma_v2", workers=1)
    cache = RomaScoreCache()
    n_ok = n_fail = 0
    for r in rows:
        pa = find_house_image(r["house"], r["img_a"])
        pb = find_house_image(r["house"], r["img_b"])
        if not (pa and pb):
            continue
        try:
            matches, _ = matcher.match_direct(pa, pb)
            conf_sum = float(np.sum([m[4] for m in matches])) if matches else 0.0
            cache.put(r["house"], os.path.basename(pa), os.path.basename(pb),
                      match_count=len(matches), conf_sum=conf_sum)
            n_ok += 1
        except Exception as e:                               # noqa: BLE001
            n_fail += 1
            print("  [warn] 彙總失敗 %s / %s: %s"
                  % (r["img_a"][:8], r["img_b"][:8], e), flush=True)

    out = os.path.join(ROOT, "roma_match", "roma_scores.npz")
    cache.save(out)
    print("  彙總 %d 對（成功 %d / 失敗 %d）-> %s"
          % (cache.num_edges(), n_ok, n_fail, out))


def main():
    ap = argparse.ArgumentParser(description="批次填滿 RoMa 匹配快取")
    ap.add_argument("--limit", type=int, default=0, help="只做前 N 對；0 = 全部")
    ap.add_argument("--time_budget_h", type=float, default=0.0,
                    help="時間預算（小時）；超過即停止並保留已完成的部分。0 = 不限")
    # 不提供 cubemap:其快取檔名為 ..._cubemap_...npz,而 batch_pose_from_cache
    # 只收 ..._direct_...npz。
    ap.add_argument("--mode", default="direct", choices=["direct", "core"])
    ap.add_argument("--workers", default="auto",
                    help="平行 worker 數；auto = GPU 張數，1 = 序列")
    # 預設留 None：交由 ensure_pair_source() 建立/沿用 _pairs.csv 快照。
    # 若給了明確路徑就照用（除錯或自訂族群時）。
    ap.add_argument("--summary", default=None,
                    help="配對來源 CSV；省略則使用 _pairs.csv 快照（首次從 _summary.csv 建立）")
    ap.add_argument("--aggregate_scores", action="store_true",
                    help="結束後重算 roma_match/roma_scores.npz（供 prepare_house_data 使用）")
    args = ap.parse_args()

    import torch
    from roma_match.matcher import RoMaMatcher

    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        raise SystemExit(
            "[abort] 偵測不到 CUDA GPU；本階段需要 GPU。\n"
            "        請確認 Notebook 已開啟加速器（T4 或 L4）。")
    if str(args.workers).lower() == "auto":
        n_workers = max(1, n_gpu)
    else:
        # ParallelScorer 以 gpu_id = i % num_gpus 分派，允許超額配置；
        # 同卡多 worker 會搶 VRAM。
        n_workers = max(1, int(args.workers))
        if n_workers > n_gpu:
            print("[warn] 指定 %d 個 worker 但只有 %d 張 GPU；同卡多 worker 會搶 VRAM。"
                  % (n_workers, n_gpu))

    # workers=1 讓 __init__ 不載入權重：列舉階段只用到純字串的 _get_cache_info()。
    # 平行路徑的模型由各 worker 自行載入；序列路徑首次呼叫 match_* 時延遲載入。
    m = RoMaMatcher(conf_threshold=0.1, use_cache=True, model_type="roma_v2", workers=1)

    pair_src = ensure_pair_source(args.summary)
    tasks, n_hit, n_miss = load_tasks(pair_src, args.limit, args.mode, m)
    budget = args.time_budget_h * 3600.0

    print("=" * 72)
    print("  批次匹配 | 待算 %d 對 | 已快取 %d 對 | 缺圖 %d 對" % (len(tasks), n_hit, n_miss))
    print("  mode=%s | GPU %d 張 | worker %d%s"
          % (args.mode, n_gpu, n_workers,
             "" if not budget else " | 時間預算 %.1f h" % args.time_budget_h))
    print("=" * 72, flush=True)

    t0 = time.time()
    n_done = n_fail = 0
    if not tasks:
        # 仍往下走：--aggregate_scores 是獨立於匹配的後處理。
        print("沒有待算的配對（全部已有快取）。")
    elif n_workers > 1:
        n_done, n_fail = run_parallel(tasks, n_workers, args.mode, budget)
    else:
        n_done, n_fail = run_sequential(tasks, m, args.mode, budget, t0)
    el = time.time() - t0

    if args.aggregate_scores:
        print("\n--- 彙總 RoMa 證據強度（roma_scores.npz）---", flush=True)
        aggregate_scores(pair_src, args.limit)

    print("\n" + "=" * 72)
    print("  新算 %d 對 · 快取命中 %d 對 · 缺圖 %d 對 · 失敗 %d 對"
          % (n_done, n_hit, n_miss, n_fail))
    print("  耗時 %.1f min%s" % (el / 60,
          ("（%.2f s/對）" % (el / n_done)) if n_done else ""))
    print("=" * 72)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
