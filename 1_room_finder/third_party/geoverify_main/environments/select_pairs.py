"""
[腳本用途]
連通性矩陣 → 配對清單（`_pairs.csv`）。本腳本位於上游的交界點：

    <上游粗篩模型>  →  matrices/<版本>/<house>.npy      （ROMA_GEO_MATRICES 指到這裡）
                              │  ch0 分數 ≥ threshold
                              ▼
    select_pairs.py     →  Output/roma_regions/_pairs.csv （本腳本）
    batch_match.py      →  roma_match/cache/*.npz
    batch_pose_from_cache.py → Output/roma_regions/*.json + _summary.csv
    covis_ranking_eval.py    → _ranking_eval.csv
    prepare_house_data.py    → stitch_tool/data/*.json

在此之前 `_pairs.csv` 只能由既有的 `_summary.csv` 快照而來，換上游模型時配對清單
不會隨之變動。本腳本補上這一段。

[門檻]
`--threshold` > 環境變數 `ROMA_GEO_PAIR_THRESHOLD` > `config.PAIR_THRESHOLD`。
單位由矩陣決定，不做 sigmoid 轉換；⚠ **換上游來源必須重新校準**。

[方向性]
`M[i,j,0]` 與 `M[j,i,0]` 不保證相等。配對本身無方向，故預設取兩向平均
（與 `covis_ranking_eval.py` 一致）；上游若確認為有向分數，用 `--pool` 指定
max/min/upper。

[用法]
  python environments/select_pairs.py                        # 用預設門檻產生 _pairs.csv
  python environments/select_pairs.py --threshold -3.0       # 指定門檻
  python environments/select_pairs.py --force                # 覆寫既有清單
  python environments/select_pairs.py --inspect              # 僅列出分數分佈與門檻曲線

不確定新來源的分數尺度時先跑 `--inspect`，它不寫檔。
"""
import argparse
import csv
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 套件未安裝時（直跑本腳本）補上 src，僅為了找到 roma_geo。能匯入就完全不動搜尋
# 路徑：本檔也會被流程步驟以檔案路徑載入，那時不該動到呼叫端的搜尋路徑。
if importlib.util.find_spec("roma_geo") is None:
    sys.path.append(os.path.join(ROOT, "src"))

from roma_geo.config import matrices_root, pair_threshold, regions_root   # noqa: E402
from roma_geo.errors import RomaGeoError  # noqa: E402
from roma_geo.names import glob_in_dir  # noqa: E402


def load_house(mdir, house):
    """回傳 (images_without_ext, ch0)；缺檔回 (None, None)。"""
    npy = os.path.join(mdir, house + ".npy")
    js = os.path.join(mdir, house + "_images.json")
    if not (os.path.exists(npy) and os.path.exists(js)):
        return None, None
    M = np.load(npy)
    if M.ndim != 3 or M.shape[0] != M.shape[1] or M.shape[2] < 1:
        raise RomaGeoError(
            "%s.npy 形狀 %s 不合契約 (N,N,>=1)" % (house, M.shape),
            input_name="連通矩陣",
            condition="見 matrices/README.md",
        )
    imgs = [os.path.splitext(x)[0] for x in json.load(open(js, encoding="utf-8"))]
    if len(imgs) != M.shape[0]:
        raise RomaGeoError(
            "%s：_images.json 長度 %s != 矩陣 %s，row/col 對不上"
            % (house, len(imgs), M.shape[0]),
            input_name="連通矩陣",
            condition="見 matrices/README.md",
        )
    return imgs, M[:, :, 0].astype(np.float64)


def pooled(ch0, how):
    if how == "mean":
        return (ch0 + ch0.T) / 2.0
    if how == "max":
        return np.maximum(ch0, ch0.T)
    if how == "min":
        return np.minimum(ch0, ch0.T)
    return ch0                      # upper：直接用 M[i,j]，不理另一向


def list_houses(mdir):
    out = []
    for p in sorted(glob_in_dir(mdir, "*.npy")):
        name = os.path.splitext(os.path.basename(p))[0]
        if name.endswith("_doormap"):
            continue
        out.append(name)
    return out


def read_pairs(path):
    """既有配對清單 → {(house, frozenset(a,b))}。"""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", newline="") as f:
        return {(r["house"], frozenset((r["img_a"], r["img_b"])))
                for r in csv.DictReader(f)}


def collect(mdir, houses, how):
    """回傳 [(house, a, b, score)]，全部可能配對（上三角）。"""
    rows = []
    for h in houses:
        imgs, ch0 = load_house(mdir, h)
        if imgs is None:
            print(f"    [skip] {h}：缺 .npy 或 _images.json")
            continue
        S = pooled(ch0, how)
        n = len(imgs)
        for i in range(n):
            for j in range(i + 1, n):
                rows.append((h, imgs[i], imgs[j], float(S[i, j])))
    return rows


def inspect(mdir, houses, how, rows, compare_set):
    print("\n--- 對稱性（ch0 vs ch0ᵀ）---")
    amax = []
    for h in houses:
        imgs, ch0 = load_house(mdir, h)
        if imgs is None:
            continue
        amax.append(float(np.abs(ch0 - ch0.T).max()))
    if amax:
        mx = max(amax)
        print(f"    max|M-Mᵀ| = {mx:.4f}（各房中位 {np.median(amax):.4f}）")
        print(f"    ⇒ {'對稱，pool 選項不影響結果' if mx < 1e-5 else '**不對稱**，pool 的選擇會改變結果'}")

    s = np.array([r[3] for r in rows])
    print(f"\n--- 分數分佈（pool={how}，{len(rows):,} 個可能配對 / {len(houses)} 房）---")
    print(f"    min {s.min():.3f} · p5 {np.percentile(s,5):.3f} · 中位 {np.median(s):.3f} "
          f"· p95 {np.percentile(s,95):.3f} · max {s.max():.3f}")
    prob_like = s.min() >= -0.01 and s.max() <= 1.01
    print(f"    單位推測：{'機率(0~1)' if prob_like else 'logit（未經 sigmoid；門檻應為負數）'}")

    print("\n--- 門檻 → 選出對數 ---")
    head = f"{'門檻':>8} {'選出':>9}"
    if compare_set is not None:
        head += f" {'涵蓋既有':>9} {'既有涵蓋率':>11} {'新進':>8}"
    print(head)
    lo, hi = np.percentile(s, 1), np.percentile(s, 99)
    for t in np.linspace(lo, hi, 11):
        m = s >= t
        line = f"{t:8.2f} {int(m.sum()):9,}"
        if compare_set is not None:
            hit = sum(1 for r, keep in zip(rows, m)
                      if keep and (r[0], frozenset((r[1], r[2]))) in compare_set)
            line += f" {hit:9,} {hit/max(len(compare_set),1):10.1%} {int(m.sum())-hit:8,}"
        print(line)


def main():
    ap = argparse.ArgumentParser(description="連通性矩陣 → _pairs.csv")
    ap.add_argument("--matrices", default=None, help="矩陣目錄（預設 config.matrices_root('refined')）")
    ap.add_argument("--threshold", type=float, default=None,
                    help="ch0 分數門檻，單位由矩陣決定（預設 config.pair_threshold()）")
    ap.add_argument("--pool", choices=["mean", "max", "min", "upper"], default="mean",
                    help="兩個方向的分數如何合併（預設 mean，與 covis_ranking_eval.py 一致）")
    ap.add_argument("--out", default=None, help="輸出路徑（預設 <regions_root>/_pairs.csv）")
    ap.add_argument("--compare", default=None,
                    help="與既有配對清單比對涵蓋率（預設以 --out 的現有內容為對照）")
    ap.add_argument("--inspect", action="store_true", help="僅診斷，不寫檔")
    ap.add_argument("--force", action="store_true", help="允許覆寫既有 _pairs.csv")
    a = ap.parse_args()

    mdir = a.matrices or matrices_root("refined")
    out = a.out or os.path.join(regions_root(), "_pairs.csv")
    thr = a.threshold if a.threshold is not None else pair_threshold()

    print("=" * 72)
    print("  矩陣 → 配對清單")
    print("=" * 72)
    print(f"    矩陣目錄  : {mdir}")
    print(f"    輸出      : {out}")
    print(f"    門檻      : {thr}（pool={a.pool}）")
    if not os.path.isdir(mdir):
        raise SystemExit(f"[FAIL] 矩陣目錄不存在：{mdir}\n"
                         "        設 ROMA_GEO_MATRICES 或 config.MATRICES_VERSION，見 matrices/README.md")

    houses = list_houses(mdir)
    if not houses:
        raise SystemExit(f"[FAIL] {mdir} 底下找不到任何 <house>.npy")
    print(f"    房子數    : {len(houses)}")

    rows = collect(mdir, houses, a.pool)
    compare_set = read_pairs(a.compare) if a.compare else read_pairs(out)
    if compare_set is not None:
        src = a.compare or out
        print(f"    對照清單  : {src}（{len(compare_set):,} 對）")

    if a.inspect:
        inspect(mdir, houses, a.pool, rows, compare_set)

    sel = [r for r in rows if r[3] >= thr]
    print(f"\n--- 選出 {len(sel):,} 對（門檻 {thr}）---")
    if compare_set is not None:
        keys = {(r[0], frozenset((r[1], r[2]))) for r in sel}
        print(f"    與對照清單：共有 {len(keys & compare_set):,} · "
              f"新進 {len(keys - compare_set):,} · 消失 {len(compare_set - keys):,}")

    if a.inspect:
        print("\n[inspect] 未寫檔。門檻確定後移除 --inspect（覆寫既有清單需加 --force）。")
        return 0

    if os.path.exists(out) and not a.force:
        raise SystemExit(
            f"\n[STOP] {out} 已存在。\n"
            "        該檔為 batch_match.py 的配對來源，覆寫會改變本批次的計算範圍。\n"
            "        請先確認上方的「新進 / 消失」數量，再加 --force 執行寫入。")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["house", "img_a", "img_b", "score"])
        for h, x, y, s in sorted(sel):
            w.writerow([h, x, y, round(s, 4)])
    print(f"\n[ok] 已寫入 {out}（{len(sel):,} 對）")
    print("     下一步：python environments/batch_match.py --workers auto")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RomaGeoError as e:
        raise SystemExit("[FAIL] %s" % e)
