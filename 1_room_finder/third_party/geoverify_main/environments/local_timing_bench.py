"""本機（PC）RoMa v2 + 幾何驗證 端到端計時基準（兩段分開計時）。

「匹配」與「幾何驗證」串在一起跑（部署時為同一階段），但兩段時間分開計。

量測設計（數字要能對外引用就必須照這樣）：
 * **RoMa 一律關快取**（use_cache=False）。抽到的配對多半已在 roma_match/cache 內，
    開快取只會量到讀檔速度。關快取也不會污染既有 cache（match_direct 不寫回）。
  * matcher.conf_threshold 設 0.1 → 取回與 cache 同一口徑的原始匹配（floor 0.1），
    驗證階段才和生產一致（analyze_pair 內部自己用 conf 0.65 / 撈回 0.3）。
  * **暖機**：第一對含 CUDA 初始化與權重載入，統計時排除前 --warmup 對，另外單獨報
    「模型建構耗時」與「暖機對耗時」。
  * 影像尺寸讀取（PIL 讀檔頭）在兩個計時器之外，避免把 I/O 算進推論。
 * 抽樣：族群 = `Output/roma_regions/_summary.csv` 全部列。
 **決定性等距抽樣**（排序後取等間隔），跨房分散、可重現。

用法：
  venv/Scripts/python.exe environments/local_timing_bench.py --n 120
  venv/Scripts/python.exe environments/local_timing_bench.py --n 20 --out /tmp/smoke.csv   # 煙霧測試
本腳本亦可於容器內以相同參數執行，改用容器內的直譯器路徑即可。
"""
import argparse
import csv
import os
import statistics
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 本檔位於 environments/，專案根目錄為上一層。
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SUMMARY = os.path.join(ROOT, "Output", "roma_regions", "_summary.csv")


def env_report():
    """記錄環境：GPU、torch、以及 fused-local-corr 加速核心**是否真的可用**。

    ⚠ 兩項已知陷阱（2026-07-30 實測確認）：
      1. 套件名是 `fused-local-corr`，但**匯入名是 `local_corr`**。查錯名字會誤報「未安裝」。
      2. 「裝了」≠「能用」。該 wheel 是預編的，torch ABI 不合時 `import` 會丟
         `undefined symbol: _ZNR5torch7Library4_def...`。所以必須**真的 import 一次**，
         只看 find_spec 會把「裝了但壞掉」誤判成可用。
    """
    import importlib.util
    info = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"
    except Exception as e:
        info["torch"] = "import failed: %s" % e
        info["cuda_available"] = False
        info["gpu"] = "-"
    installed = bool(importlib.util.find_spec("local_corr"))
    usable, why = False, "未安裝"
    if installed:
        try:
            import local_corr  # noqa: F401
            usable, why = True, "可用"
        except Exception as e:
            why = "已安裝但 import 失敗（%s: %s）" % (type(e).__name__, str(e)[:90])
    info["fused_kernel_installed"] = installed
    info["fused_kernel_usable"] = usable
    info["fused_kernel_note"] = why
    return info


#: 母體規模與母體位姿率，由 load_population 從 _summary.csv 現算。
#: 報告字串一律引用這兩個值，不可寫死成某次執行的數字（換資料集即不成立）。
n_pop = 0
n_pop_posed = None


def load_population():
    global n_pop, n_pop_posed
    rows, posed = [], 0
    with open(SUMMARY, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["house"], r["img_a"], r["img_b"]))
            if r.get("az_ab_deg"):
                posed += 1
    rows.sort()
    n_pop, n_pop_posed = len(rows), posed
    return rows


def sample_pairs(rows, n):
    if n >= len(rows):
        return rows
    step = len(rows) / float(n)
    return [rows[int(i * step)] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="抽樣對數（決定性等距抽樣）")
    ap.add_argument("--warmup", type=int, default=2, help="統計時排除的前 N 對（含 CUDA 暖機）")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "local_timing_bench.csv"))
    ap.add_argument("--md", default=os.path.join(SCRIPT_DIR, "local_timing_bench.md"))
    ap.add_argument("--tag", default="local-venv", help="這次跑的標籤（例：local-venv / docker-fused）")
    ap.add_argument("--use_cache", action="store_true",
                    help="（不建議）開 RoMa 快取；只會量到讀檔速度，不是推論時間")
    ap.add_argument("--batch_bidir", action="store_true",
                    help="把 (A,B)+(B,A) 打包成 batch=2 一次 forward(VRAM ×2)。預設關 = 串行跑"
                         "兩次 forward(見 core_roma_match/matchers.py match() 的 batch_bidir 分支)")
    ap.add_argument("--mode", default="direct", choices=["direct", "core"],
                    help="direct=單次 ERP↔ERP 匹配（生產路徑，run_roma_above_threshold 用的）；"
                         "core=RoMaScan cubemap 分塊路徑（一個 pair 內含多次子匹配）")
    # 驗證階段參數：預設與 batch_pose_from_cache.py 生產值一致，數字才可與 151ms/對 對照
    ap.add_argument("--conf", type=float, default=0.65)
    ap.add_argument("--retry_conf", type=float, default=0.3)
    ap.add_argument("--max_pts", type=int, default=4000)
    ap.add_argument("--thresh_deg", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--sampler", default="stratified")
    ap.add_argument("--lat_filter", type=float, default=60.0)
    args = ap.parse_args()

    import numpy as np
    from PIL import Image

    # core_roma_match 位於本專案內，可直接 import。
    from pose_regions import analyze_pair, find_house_image
    from roma_match.matcher import RoMaMatcher

    env = env_report()
    print("[env] " + " | ".join("%s=%s" % kv for kv in env.items()), flush=True)
    if not env["fused_kernel_usable"]:
        print("[env] ⚠ fused-local-corr 加速核心**不可用**（%s）。這是「未加速」的數字。"
              % env["fused_kernel_note"], flush=True)

    pairs = sample_pairs(load_population(), args.n)
    print("[sample] 母體 %d 對中決定性抽樣 %d 對，跨 %d 間房"
          % (n_pop, len(pairs), len({p[0] for p in pairs})), flush=True)

    t0 = time.perf_counter()
    matcher = RoMaMatcher(conf_threshold=0.1, use_cache=args.use_cache, model_type="roma_v2",
                          batch_bidir=args.batch_bidir)
    t_build = time.perf_counter() - t0
    print("[init] matcher 建構 %.2fs（模型載入；不計入每對成本）" % t_build, flush=True)

    recs = []
    for k, (house, ia, ib) in enumerate(pairs):
        pa = find_house_image(house, ia)
        pb = find_house_image(house, ib)
        if not pa or not pb:
            print("  [skip] %s %s/%s 影像找不到" % (house, ia[:8], ib[:8]), flush=True)
            continue
        # 尺寸讀檔頭：兩個計時器之外
        with Image.open(pa) as im:
            Wa, Ha = im.size
        with Image.open(pb) as im:
            Wb, Hb = im.size

        t1 = time.perf_counter()
        try:
            # core 模式走 cubemap 分塊(RoMaScanPipeline);一次呼叫仍是「一個 pair」,
            # 分塊在私有核心的 run_roma_match 內部完成。
            matches, _info = (matcher.match_core(pa, pb) if args.mode == "core"
                              else matcher.match_direct(pa, pb))
        except Exception as e:
            print("  [err match] %s %s/%s: %s" % (house, ia[:8], ib[:8], e), flush=True)
            continue
        t_match = time.perf_counter() - t1

        m = np.asarray(matches, dtype=float)
        if m.ndim != 2 or m.shape[0] == 0:
            m = np.zeros((0, 5), dtype=float)
        t2 = time.perf_counter()
        try:
            res = analyze_pair(m, Wa, Ha, Wb, Hb, args.conf, args.max_pts, args.thresh_deg,
                               args.iters, seed=0, sampler=args.sampler,
                               retry_conf=args.retry_conf, lat_max=args.lat_filter)
        except Exception as e:
            print("  [err verify] %s %s/%s: %s" % (house, ia[:8], ib[:8], e), flush=True)
            continue
        t_verify = time.perf_counter() - t2

        posed = bool(res.get("pose"))
        recs.append(dict(house=house, img_a=ia, img_b=ib, w=Wa, h=Ha,
                         n_raw=len(m), n_inlier=(res.get("n") or {}).get("inlier", 0),
                         posed=int(posed), conf_used=res.get("conf_used") or "",
                         t_match_s=round(t_match, 4), t_verify_s=round(t_verify, 4)))
        if (k + 1) % 10 == 0 or k < args.warmup:
            print("  [%3d/%d] %s 匹配 %.2fs · 驗證 %.3fs · 點 %d → 內點 %d%s"
                  % (k + 1, len(pairs), house[-5:], t_match, t_verify, len(m),
                     recs[-1]["n_inlier"], "" if posed else " (無位姿)"), flush=True)

    if not recs:
        print("沒有成功的樣本，無法統計。")
        return

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)

    stat = recs[args.warmup:] if len(recs) > args.warmup else recs
    tm = sorted(r["t_match_s"] for r in stat)
    tv = sorted(r["t_verify_s"] for r in stat)

    def pct(v, p):
        k = (len(v) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (k - lo)

    out = []
    out.append("# 本機 RoMa v2 + 幾何驗證 計時基準（%s）\n" % args.tag)
    out.append("產生腳本：`local_timing_bench.py`；原始逐對數據：`%s`\n" % os.path.basename(args.out))
    out.append("環境：GPU %s · torch %s · **fused-local-corr 加速核心：%s**\n"
               % (env["gpu"], env["torch"],
                  "可用" if env["fused_kernel_usable"] else "**不可用**（%s）" % env["fused_kernel_note"]))
    out.append("樣本：母體 %s 對中決定性等距抽樣 %d 對，成功 %d 對；"
               "統計已排除前 %d 對暖機。RoMa 快取：%s。\n"
               % (f"{n_pop:,}", len(pairs), len(recs), args.warmup,
                  "開（不可用於計時）" if args.use_cache else "關"))
    out.append("模型建構（一次性，不計入每對）：%.2f s\n" % t_build)
    out.append("雙向推論策略：**%s**（一對 = A→B ＋ B→A；串行=兩次 forward、"
               "batch=打包成 batch2 一次 forward、VRAM ×2）\n"
               % ("batch_bidir" if args.batch_bidir else "串行(預設)"))
    out.append("匹配模式：**%s**（%s）\n"
               % (args.mode,
                  "單次 ERP↔ERP，生產路徑（run_roma_above_threshold.py mode=\"direct\"）"
                  if args.mode == "direct" else
                  "RoMaScan cubemap 分塊，一個 pair 內含多次子匹配"))
    out.append("| 階段 | 中位 | 平均 | p90 | 最小 | 最大 |")
    out.append("|---|---|---|---|---|---|")
    out.append("| RoMa v2 匹配（%s） | %.3f s | %.3f s | %.3f s | %.3f s | %.3f s |"
               % (args.mode, pct(tm, .5), statistics.fmean(tm), pct(tm, .9), tm[0], tm[-1]))
    out.append("| 幾何驗證（位姿+區域） | %.3f s | %.3f s | %.3f s | %.3f s | %.3f s |"
               % (pct(tv, .5), statistics.fmean(tv), pct(tv, .9), tv[0], tv[-1]))
    out.append("| **合計（單對端到端）** | %.3f s | %.3f s | — | — | — |"
               % (pct(tm, .5) + pct(tv, .5), statistics.fmean(tm) + statistics.fmean(tv)))
    out.append("")
    # 驗證時間是雙峰的:no_covisibility 會提早退出(很便宜),真的算出位姿才跑完整 RANSAC。
    # 只報總體中位會嚴重低估「有位姿那些對」的成本,所以拆開。
    tv_yes = sorted(r["t_verify_s"] for r in stat if r["posed"])
    tv_no = sorted(r["t_verify_s"] for r in stat if not r["posed"])
    out.append("驗證時間是**雙峰**的（無共視證據會提早退出，算出位姿才跑完整 RANSAC＋區域萃取）：\n")
    out.append("| 驗證子集 | n | 中位 | 平均 | p90 |")
    out.append("|---|---|---|---|---|")
    if tv_yes:
        out.append("| 有位姿 | %d | %.3f s | %.3f s | %.3f s |"
                   % (len(tv_yes), pct(tv_yes, .5), statistics.fmean(tv_yes), pct(tv_yes, .9)))
    if tv_no:
        out.append("| 無位姿（提早退出） | %d | %.3f s | %.3f s | %.3f s |"
                   % (len(tv_no), pct(tv_no, .5), statistics.fmean(tv_no), pct(tv_no, .9)))
    out.append("")
    tot = statistics.fmean(tm) + statistics.fmean(tv)
    out.append("驗證佔單對總時間比重（全樣本平均）：**%.1f%%**；只看有位姿的對則是 %s。"
               "主成本在匹配。\n"
               % (statistics.fmean(tv) / tot * 100,
                  ("%.1f%%" % (statistics.fmean(tv_yes) /
                               (statistics.fmean(tm) + statistics.fmean(tv_yes)) * 100))
                  if tv_yes else "本樣本無位姿成功對，無法計算"))
    out.append("外推全量 %s 對（單卡序列、含驗證）：**%.1f 分鐘**（%.2f 小時）。\n"
               % (f"{n_pop:,}", tot * n_pop / 60, tot * n_pop / 3600))
    n_posed = sum(r["posed"] for r in stat)
    # 全量位姿率由母體現算，不可寫死成某次執行的數字（換資料集即不成立）。
    out.append("位姿成功率（本樣本）：%d/%d = %.1f%%%s\n"
               % (n_posed, len(stat), n_posed / len(stat) * 100,
                  "（全量 %.1f%%，用來確認抽樣沒偏）" % (100.0 * n_pop_posed / n_pop)
                  if n_pop_posed is not None and n_pop else "。"))
    text = "\n".join(out)
    with open(args.md, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
