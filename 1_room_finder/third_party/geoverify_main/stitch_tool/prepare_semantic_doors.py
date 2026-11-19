"""
[腳本用途]
把子計畫一的門語意圖（經 semantic_doors/semantic_door_eval.py 的 extract_doors() 抽取、快取於
_door_cache.json）轉存成 stitch_tool/semantic/<house>.json，供 server.py 的
GET /api/semantic_doors/<house> 讀取。

門抽取邏輯（過縫接倍去重 / pipeline 方位角約定 / 門後有門合併但角寬不超過 90°）完全沿用
semantic_door_eval.extract_doors()（直接 import，不複製一份）。

輸出格式（每個房子一個檔案）:
  {"house": "DollhouseTask_58553",
   "doors": {"<image檔名不含副檔名>": [{"az":174.6,"az0":134.8,"az1":214.4,"width_deg":79.6,"px":3846336}, ...]}}
只收錄「至少有一道門」的圖；沒有語意圖的房子完全不輸出檔案（server 端 404，前端顯示「此房無語意圖」）。

[用法]
  venv/Scripts/python.exe stitch_tool/prepare_semantic_doors.py
      預設模式：只讀現成的 semantic_doors/_door_cache.json 轉檔（不重跑抽取，
      不需要 cv2/PIL，秒級完成）。

  venv/Scripts/python.exe stitch_tool/prepare_semantic_doors.py --refresh
      忽略快取重新對 datasets/semantic/ 跑 extract_doors()（約 25 分鐘，需要 cv2/PIL/numpy）。
      merge_gap 固定用 1.0（與現有 _door_cache.json 一致，對應 semantic_door_eval.py 的 CLI 預設）。

  venv/Scripts/python.exe stitch_tool/prepare_semantic_doors.py --houses DollhouseTask_58553
      只轉換（--refresh 時也只重抽）指定的 stitch_tool 房子名（可多個，空白分隔）。
"""
import argparse
import glob
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))            # stitch_tool/
ROOT_DIR = os.path.dirname(SCRIPT_DIR)                              # 專案根目錄
# 門抽取程式與參考快取位於根目錄的 semantic_doors/。
EVAL_DIR = os.path.join(ROOT_DIR, "semantic_doors")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")                          # prepare_house_data.py 產物
OUT_DIR = os.path.join(SCRIPT_DIR, "semantic")                       # 本腳本產物
CACHE = os.path.join(EVAL_DIR, "_door_cache.json")
_SRC = os.path.join(ROOT_DIR, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from roma_geo.config import semantic_root                                     # noqa: E402

SEM_DIR = semantic_root()                                            # <tid>/<image>_mask.png；見 config.py


def base(name):
    return os.path.splitext(name)[0]


def house_tid(house):
    """stitch_tool 房子名 -> datasets/semantic/ 的數字資料夾名（與 semantic_door_eval.doors_for 同規則）。"""
    return "".join(ch for ch in house if ch.isdigit())


def load_cache():
    if os.path.exists(CACHE):
        return json.load(open(CACHE, encoding="utf-8"))
    return {}


def refresh_cache(tids, cache):
    """對指定 tid 的 datasets/semantic/<tid>/*_mask.png 全部重抽（延遲 import：只有 --refresh
    才需要 cv2/PIL，預設的「讀現成快取轉檔」路徑不需要這些重依賴）。"""
    sys.path.insert(0, EVAL_DIR)
    from semantic_door_eval import extract_doors
    n_new = 0
    for tid in tids:
        d = os.path.join(SEM_DIR, tid)
        if not os.path.isdir(d):
            continue
        for mp in sorted(glob.glob(os.path.join(d, "*_mask.png"))):
            img = os.path.basename(mp)[:-len("_mask.png")]
            cache[tid + "/" + img] = extract_doors(mp, 1.0)
            n_new += 1
            if n_new % 20 == 0:
                print("  重抽中 %d…" % n_new, flush=True)
    if n_new:
        json.dump(cache, open(CACHE, "w", encoding="utf-8"))
        print("已更新門快取（%s）：重抽 %d 張圖" % (CACHE, n_new))
    else:
        print("--refresh 但沒有找到任何 *_mask.png（tid 清單：%s）" % (tids,))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--houses", nargs="*", default=None,
                    help="只處理指定的 stitch_tool 房子名（如 DollhouseTask_58553），預設全部")
    ap.add_argument("--refresh", action="store_true",
                    help="忽略 _door_cache.json，對 datasets/semantic/ 重新跑 extract_doors()（約 25 分鐘）")
    args = ap.parse_args()

    houses = []
    for fp in sorted(glob.glob(os.path.join(DATA_DIR, "*.json"))):
        name = base(os.path.basename(fp))
        if name.startswith("_"):
            continue
        if args.houses and name not in args.houses:
            continue
        houses.append(name)
    if not houses:
        print("沒有符合的房子（檢查 --houses 或 stitch_tool/data/*.json 是否存在）")
        return

    cache = load_cache()
    if args.refresh:
        refresh_cache(sorted({house_tid(h) for h in houses}), cache)

    os.makedirs(OUT_DIR, exist_ok=True)
    n_written = n_skipped = 0
    for house in houses:
        tid = house_tid(house)
        hd = json.load(open(os.path.join(DATA_DIR, house + ".json"), encoding="utf-8"))
        doors = {}
        for img in hd["images"]:
            ds = cache.get(tid + "/" + base(img))
            if ds:
                doors[base(img)] = ds
        if not doors:
            n_skipped += 1
            continue
        out_fp = os.path.join(OUT_DIR, house + ".json")
        json.dump({"house": house, "doors": doors}, open(out_fp, "w", encoding="utf-8"),
                   ensure_ascii=False)
        n_written += 1
        print("%-45s %2d/%2d 張圖有門" % (house, len(doors), hd["n_images"]))

    print("\n完成：%d 個房子寫入語意門資料（%s），%d 個房子無語意圖（略過，前端顯示 404）。"
          % (n_written, OUT_DIR, n_skipped))


if __name__ == "__main__":
    main()
