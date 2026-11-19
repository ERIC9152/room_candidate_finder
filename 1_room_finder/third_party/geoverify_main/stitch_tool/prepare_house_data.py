"""
[腳本用途]
把上游粗篩產出的連通性矩陣整理成前端拼接工具可直接讀取的 house JSON。

輸入 (每間房子):
  oof_matrices/<exp>/<house>.npy            (N, N, 3)  ch0=連通 logit(對稱), ch1=sin(az), ch2=cos(az)
  oof_matrices/<exp>/<house>_images.json    長度 N 的影像檔名 (與矩陣 row/col 對齊)
  datasets/<house>/.../*.jpg                 ERP 全景圖 (讀寬高)
  datasets/<house>/.../*_HOTSPOT.json        Ground Truth 連接 (可選, 供除錯對照)
  Output/roma_regions/_ranking_eval.csv       幾何驗證排序欄位 (geo_*，可選；join key = house+img 檔名集合)
  Output/roma_regions/_summary.csv           幾何驗證方位 az_ab/az_ba (取代粗篩方向頭，可選；同上 join)

輸出:
  stitch_tool/data/<house>.json

方位規約 (與 dataset_builder._calculate_azimuth 完全一致):
  az(x) = ((x / W) * 360 - 180) mod 360
  反解 (前端用): u = ((az + 180) mod 360) / 360     # ERP 影像水平比例 / three.js 貼圖 u

[用法]
  python stitch_tool/prepare_house_data.py                # 全部房子
  python stitch_tool/prepare_house_data.py --houses DollhouseTask_58472_NoOutdoor
"""
import os
import sys
import csv
import json
import glob
import math
import argparse
import importlib.util
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)  # 專案根目錄
# 套件未安裝時（直跑本腳本）補上 src，僅為了找到 roma_geo。能匯入就完全不動搜尋
# 路徑：本檔也會被流程步驟以檔案路徑載入，那時不該動到呼叫端的搜尋路徑。
if importlib.util.find_spec("roma_geo") is None:
    sys.path.append(os.path.join(ROOT_DIR, "src"))
from roma_geo import upstream_direction                          # noqa: E402
from roma_geo.config import datasets_root, matrices_root, regions_root   # noqa: E402
from roma_geo.project_modules import load_project_module         # noqa: E402

DATA_ROOT = datasets_root()                        # 可用 ROMA_GEO_DATASETS 覆寫，見 config.py
# 連通性矩陣來源。優先序：--oof_dir > 環境變數 ROMA_GEO_MATRICES > matrices/<版本>/
DEFAULT_OOF = matrices_root()
OUT_DIR = os.path.join(SCRIPT_DIR, "data")
# Optional: run_roma_above_threshold.py's aggregated per-pair RoMa evidence (match_count +
# confidence sum/mean), written by its aggregate_roma_scores() step. Only covers pairs whose
# coarse-filter score cleared the LOO threshold in that run, so most houses/pairs will have none -
# every pair without a cached entry simply gets roma_* = null (fully backward compatible).
DEFAULT_ROMA_SCORES = os.path.join(ROOT_DIR, "roma_match", "roma_scores.npz")
# Optional: 幾何驗證批次輸出的 pair-level CSV（皆已入 git）。
# _ranking_eval.csv -> geo_n_inlier/geo_inlier_rate/geo_n_regions/geo_conf_used/geo_posed
#   （佇列「幾何驗證」排序取代原始 roma_conf_* 聚合，見 queue.js）。
# _summary.csv      -> az_ab_deg/az_ba_deg 取代粗篩方向頭 az_ij/az_ji（粗篩模型不輸出通道方位，
#   規則見 matrices/README.md「方位通道：型態可換」）；並在 _ranking_eval.csv
#   缺此對時作為 geo_* 的來源（外來資料無 *_HOTSPOT.json，排序評估產不出來，見 build_house）。
# 兩者都缺此對時該 house 完全不受影響：geo_* 全 null、az_ij/az_ji 保留 coarse 舊行為。
# 產出位置可用 ROMA_GEO_OUTPUT 覆寫（批次在別台機器跑時），見 config.py。
DEFAULT_RANKING_EVAL = os.path.join(regions_root(), "_ranking_eval.csv")
DEFAULT_SUMMARY_CSV = os.path.join(regions_root(), "_summary.csv")

try:
    from roma_match.roma_score_cache import RomaScoreCache
except ImportError:
    RomaScoreCache = None  # optional: RoMa-score sorting is skipped, everything else still works

# 同目錄模組以檔案路徑載入。本檔會被流程步驟以檔案路徑載入，那時 stitch_tool 不在
# 搜尋路徑上，裸名 `import house_dirs` 找不到；也不該為此把本目錄加進搜尋路徑，
# 那會讓這些泛用檔名變成呼叫端的頂層模組。
resolve_task_dir = load_project_module(SCRIPT_DIR, "house_dirs").resolve_task_dir
_gt_hotspot = load_project_module(SCRIPT_DIR, "gt_hotspot")
find_hotspot_files = _gt_hotspot.find_hotspot_files
load_gt = _gt_hotspot.load_gt
gt_pair_fields = _gt_hotspot.gt_pair_fields

def read_image_size(path: str):
    """只讀 header 拿寬高，避免載入整張大圖。"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.size[0]), int(im.size[1])
    except Exception:
        return 8192, 4096  # ERP 常見預設


def az_from_matrix(M, i, j):
    """方位(度) i->j = atan2(ch1=sin, ch2=cos)。已對 localization CSV 驗證一致。

    解碼本身在 `upstream_direction.py`：上游粗篩之後會由子計畫的模型整體取代，
    屆時方向可能是影像上的框，換來源時只改那個模組。
    """
    return upstream_direction.from_sincos(M[i, j, 1], M[i, j, 2])


def _csv_num(s, cast=float):
    """CSV 欄位轉數值；空字串一律轉 None。posed=0 列的 conf_used 等欄位常是
    空字串，用這個 helper 才不會意外炸成 ValueError 或混進 NaN。"""
    s = (s or "").strip()
    return cast(s) if s != "" else None


def load_pair_csv(path: str) -> dict:
    """讀取 pose_experiment 幾何驗證批次輸出的 pair-level CSV（_ranking_eval.csv 或
    _summary.csv，兩者前三欄都是 house,img_a,img_b），回傳
    {(house, frozenset({img_a, img_b})): row_dict}。
    img_a/img_b 檔名不含副檔名、pair 無序。用 frozenset 當 key 一次處理這兩項，
    呼叫端 join 時只需把影像檔名去副檔名即可比對。檔案不存在回傳空 dict（呼叫端等同「不 join」）。"""
    lookup = {}
    if not os.path.exists(path):
        return lookup
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            lookup[(row["house"], frozenset({row["img_a"], row["img_b"]}))] = row
    return lookup


def build_house(house: str, oof_dir: str, roma_cache=None, ranking_lookup=None, summary_lookup=None,
                datasets_root=None) -> dict | None:
    ds_root = datasets_root if datasets_root is not None else DATA_ROOT
    npy = os.path.join(oof_dir, f"{house}.npy") if oof_dir else ""
    imgs_json = os.path.join(oof_dir, f"{house}_images.json") if oof_dir else ""
    M = None
    images = None
    if npy and os.path.exists(npy) and os.path.exists(imgs_json):
        M = np.load(npy)                       # (N, N, 3)
        images = json.load(open(imgs_json, encoding="utf-8"))

    task_dir = resolve_task_dir(ds_root, house)
    if images is None:
        if task_dir is None:
            print(f"[skip] {house}: 無連通矩陣，也找不到影像資料夾")
            return None
        from roma_geo.names import is_image_filename
        images = sorted(
            n for n in os.listdir(task_dir)
            if is_image_filename(n) and os.path.isfile(os.path.join(task_dir, n))
        )
        if not images:
            print(f"[skip] {house}: 影像資料夾沒有 jpg")
            return None

    N = len(images)
    if task_dir is None:
        print(f"[warn] {house}: 找不到 datasets 影像資料夾，仍輸出(無 GT/寬高預設)")
        W, H = 8192, 4096
    else:
        W, H = read_image_size(os.path.join(task_dir, images[0]))

    # HOTSPOT 尋找＋解析＋排序邏輯集中在 gt_hotspot.py（與 server.py 即時補 GT 共用同一份，
    # 見該檔模組說明）；寬度直接沿用上面剛算出的 W，不必再開一次圖。
    gt = load_gt(find_hotspot_files(task_dir), W) if task_dir else {}

    # 建立 pairs (上三角 i<j)。有連通矩陣時依連通機率由高到低。
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            ai, aj = images[i], images[j]
            ai_base = os.path.splitext(ai)[0]
            aj_base = os.path.splitext(aj)[0]
            srow = summary_lookup.get((house, frozenset({ai_base, aj_base}))) if summary_lookup else None

            if M is not None:
                logit = float(M[i, j, 0])
                prob = 1.0 / (1.0 + math.exp(-logit))
                az_ij_coarse = az_from_matrix(M, i, j)
                az_ji_coarse = az_from_matrix(M, j, i)
                az_ij, az_ji, az_source = az_ij_coarse, az_ji_coarse, "coarse"
            else:
                logit = prob = az_ij_coarse = az_ji_coarse = None
                az_ij = az_ji = az_source = None

            if srow is not None:
                az_ab = _csv_num(srow.get("az_ab_deg"))
                az_ba = _csv_num(srow.get("az_ba_deg"))
                if az_ab is not None and az_ba is not None:
                    if srow["img_a"] == ai_base:
                        az_ij, az_ji = az_ab % 360.0, az_ba % 360.0
                    else:
                        az_ij, az_ji = az_ba % 360.0, az_ab % 360.0
                    az_source = "geo"

            geo_blank = {
                "roma_match_count": None,
                "roma_conf_sum": None,
                "roma_conf_mean": None,
                "geo_n_inlier": None,
                "geo_inlier_rate": None,
                "geo_n_regions": None,
                "geo_conf_used": None,
                "geo_posed": None,
                "geo_az_wedge_i_deg": None,
                "geo_az_wedge_j_deg": None,
            }
            if M is not None:
                pair = {
                    "i": i, "j": j,
                    "logit": round(logit, 4),
                    "prob": round(prob, 4),
                    "az_ij": round(az_ij, 2) % 360.0,
                    "az_ji": round(az_ji, 2) % 360.0,
                    "az_source": az_source,
                    "az_coarse_ij": round(az_ij_coarse, 2) % 360.0,
                    "az_coarse_ji": round(az_ji_coarse, 2) % 360.0,
                    **gt_pair_fields(gt, ai, aj),
                    **geo_blank,
                }
            else:
                pair = {
                    "i": i, "j": j,
                    "az_ij": None if az_ij is None else round(az_ij, 2) % 360.0,
                    "az_ji": None if az_ji is None else round(az_ji, 2) % 360.0,
                    **gt_pair_fields(gt, ai, aj),
                    **geo_blank,
                }
                if az_source is not None:
                    pair["az_source"] = az_source
            if roma_cache is not None:
                rec = roma_cache.get(house, ai, aj)
                if rec is not None:
                    pair["roma_match_count"] = rec["match_count"]
                    pair["roma_conf_sum"] = round(rec["conf_sum"], 4)
                    pair["roma_conf_mean"] = round(rec["conf_mean"], 4)
            grow = ranking_lookup.get((house, frozenset({ai_base, aj_base}))) if ranking_lookup else None
            if grow is not None:
                pair["geo_n_inlier"] = _csv_num(grow.get("n_inlier"), int)
                pair["geo_inlier_rate"] = _csv_num(grow.get("inlier_rate"), float)
                pair["geo_n_regions"] = _csv_num(grow.get("n_regions"), int)
                pair["geo_conf_used"] = _csv_num(grow.get("conf_used"), float)
                pair["geo_posed"] = _csv_num(grow.get("posed"), int)
            elif srow is not None:
                # 退回 _summary.csv。_ranking_eval.csv 需要 GT 才產得出來（它要算 y 與
                # topology distance），但 geo_* 這五欄本身與 GT 無關。外來資料沒有
                # *_HOTSPOT.json 時走這條，佇列的「幾何驗證」排序才不會整欄落空。
                # posed 的定義與 covis_ranking_eval.py 相同：有位姿且非 no_covisibility。
                pair["geo_n_inlier"] = _csv_num(srow.get("n_inlier"), int)
                pair["geo_inlier_rate"] = _csv_num(srow.get("inlier_rate"), float)
                pair["geo_n_regions"] = _csv_num(srow.get("n_regions"), int)
                pair["geo_conf_used"] = _csv_num(srow.get("conf_used"), float)
                pair["geo_posed"] = int(srow.get("flag_no_covisibility") != "1"
                                        and srow.get("conf_used", "") != "")
            if srow is not None:
                # 側別正規化與 az_ij/az_ji 用同一個判斷：srow 的 img_a 是不是 i 圖。
                wa = _csv_num(srow.get("az_wedge_a_deg"), float)
                wb = _csv_num(srow.get("az_wedge_b_deg"), float)
                if srow["img_a"] == ai_base:
                    pair["geo_az_wedge_i_deg"], pair["geo_az_wedge_j_deg"] = wa, wb
                else:
                    pair["geo_az_wedge_i_deg"], pair["geo_az_wedge_j_deg"] = wb, wa
            pairs.append(pair)
    if M is not None:
        pairs.sort(key=lambda p: p["prob"], reverse=True)
    else:
        pairs.sort(key=lambda p: (p["i"], p["j"]))

    out = {
        "house": house,
        "n_images": N,
        "width": W,
        "height": H,
        "images": images,
        "pairs": pairs,
        "az_convention": upstream_direction.CONVENTION,
        "has_gt": bool(gt),
        "n_gt_connected_pairs": sum(1 for p in pairs if p["gt_connected"]),
        "has_connectivity_matrix": M is not None,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof_dir", default=DEFAULT_OOF)
    ap.add_argument("--houses", nargs="*", default=None, help="指定房子；預設全部")
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--roma_scores", default=DEFAULT_ROMA_SCORES,
                     help="run_roma_above_threshold.py 產出的 roma_scores.npz（可選；不存在則跳過）")
    ap.add_argument("--no_roma_scores", action="store_true", help="即使檔案存在也不附加 RoMa 分數")
    ap.add_argument("--ranking_eval", default=DEFAULT_RANKING_EVAL,
                     help="_ranking_eval.csv：附加幾何驗證排序欄位 geo_*（可選；不存在則全 null）")
    ap.add_argument("--no_ranking_eval", action="store_true", help="即使檔案存在也不附加幾何驗證欄位")
    ap.add_argument("--summary_csv", default=DEFAULT_SUMMARY_CSV,
                     help="_summary.csv：az_ij/az_ji 改吃幾何驗證 az_ab/az_ba（可選；不存在則保留 coarse 方位）")
    ap.add_argument("--no_summary_csv", action="store_true", help="即使檔案存在也不做方位遷移（保留 coarse）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    roma_cache = None
    if not args.no_roma_scores and RomaScoreCache is not None and os.path.exists(args.roma_scores):
        try:
            roma_cache = RomaScoreCache.load(args.roma_scores)
            print(f"[roma] Loaded {roma_cache.num_edges()} aggregated RoMa pair scores from {args.roma_scores}")
        except (ValueError, OSError, KeyError) as e:
            print("[roma] 略過分數快取（%s）。舊格式請重新產生。" % e)
            roma_cache = None

    ranking_lookup = None
    if not args.no_ranking_eval and os.path.exists(args.ranking_eval):
        ranking_lookup = load_pair_csv(args.ranking_eval)
        print(f"[geo] Loaded {len(ranking_lookup)} ranking-eval rows from {args.ranking_eval}")

    summary_lookup = None
    if not args.no_summary_csv and os.path.exists(args.summary_csv):
        summary_lookup = load_pair_csv(args.summary_csv)
        print(f"[geo] Loaded {len(summary_lookup)} summary rows from {args.summary_csv}")

    if args.houses:
        houses = args.houses
    else:
        houses = sorted({
            os.path.basename(p)[:-4]
            for p in glob.glob(os.path.join(args.oof_dir, "*.npy"))
            if not p.endswith("_doormap.npy")   # 舊矩陣目錄可能還有這種檔，不是房屋矩陣
        })
        # 沒有輸入就中止：繼續下去會把 data/_index.json 覆寫成空陣列，
        # 而 server.py 優先讀該檔，導致 /api/houses 回空。
        if not houses:
            raise SystemExit(
                "[abort] %s 底下找不到任何 *.npy（連通性矩陣）。\n"
                "        沒有輸入就不會產生任何 house JSON，繼續執行只會把\n"
                "        %s/_index.json 覆寫成空陣列並讓工具打不開。\n"
                "        請用 --oof_dir 指定正確路徑，或先取得矩陣檔。"
                % (args.oof_dir, args.out_dir))

    index = []
    for h in houses:
        data = build_house(h, args.oof_dir, roma_cache=roma_cache,
                            ranking_lookup=ranking_lookup, summary_lookup=summary_lookup)
        if data is None:
            continue
        out = os.path.join(args.out_dir, f"{h}.json")
        json.dump(data, open(out, "w", encoding="utf-8"), ensure_ascii=False)
        index.append({
            "house": h,
            "n_images": data["n_images"],
            "n_pairs": len(data["pairs"]),
            "n_gt_connected_pairs": data["n_gt_connected_pairs"],
        })
        tag = "[ok]"
        extra = ""
        if data.get("has_connectivity_matrix") is False:
            tag = "[degraded]"
            extra = " 連通矩陣缺席，改以影像兩兩配對。"
        print("%s %s: N=%s pairs=%s gt_conn=%s -> %s%s"
              % (tag, h, data["n_images"], len(data["pairs"]),
                 data["n_gt_connected_pairs"], out, extra))

    json.dump(index, open(os.path.join(args.out_dir, "_index.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[done] {len(index)} 房子 -> {args.out_dir}/_index.json")


if __name__ == "__main__":
    main()
