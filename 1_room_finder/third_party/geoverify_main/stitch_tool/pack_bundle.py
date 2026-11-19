"""
[腳本用途]
把流程各階段的產出，組成一個「自足的房子產出包（bundle）」，供標註/檢視工具直接載入。
工具因此不再依賴完整的資料集樹，改為「使用時手動選取一或數棟房子的產出包」。面向未知資料。

    <bundle_root>/<house>/
        <house>.json          幾何流程整合後的房子資料（連通、通道方位、共視統計）
        images/*.jpg          該房 ERP 全景影像（隨包攜帶，工具不再回頭讀資料集）
        regions/*.json        共視區域（每對一檔）
        semantic.json         （可選）門語意輔助資料
        <house>_HOTSPOT.json  （可選）標註真值；<house>.json 若還沒有 GT 欄位（例如下面
                               的交付檔模式），伺服器讀取時會用這個檔案即時算出 GT 對照
                               並回報比對結果，房號對不上會在伺服器 log 明確提示，不會
                               安靜地不顯示（見 stitch_tool/gt_hotspot.py）。若房子資料
                               已經內建 GT，這個檔案就只是隨包留存的原始標註，不影響顯示。

[交付檔模式 --from_public]
改以 `export_public.py` 的**對外交付檔**當作包內的房子資料，且只附影像：

    <bundle_root>/<house>/
        <house>.json          對外交付檔本身，未經任何修改
        images/*.jpg          該房 ERP 全景影像

不放區域檔（交付檔已內嵌）、不放門語意、不放真值。下游收到的就是這些。工具能不能
把它渲染出來，即為「交付內容是否足夠」的直接檢驗；缺什麼會當場看得出來，
而不是等下游回報。

[路徑彈性]
輸入各段路徑、輸出包根目錄、要打包哪些房子，全部可由參數指定；預設值取自 config 與既有產出位置。

[影像處理]
預設以硬連結節省空間（同磁碟時等效）；要做成可搬到別台機器的可攜包，加 --copy 實體複製。
"""
import argparse
import glob
import json
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
_SRC = os.path.join(ROOT_DIR, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from roma_geo.config import datasets_root, regions_root                 # noqa: E402
from house_dirs import resolve_task_dir                        # noqa: E402


def _place(src, dst, copy):
    """把單一檔案放到目的地：copy=實體複製；否則硬連結（失敗時退回複製）。"""
    if os.path.exists(dst):
        os.remove(dst)
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_images(hd, task_dir, dst, copy):
    """依房子資料的 images 清單，從資料集資料夾取影像放進包內。回傳 (放入數, 缺漏數)。"""
    if not task_dir:
        return 0, len(hd.get("images", []))
    n_ok = n_miss = 0
    for fn in hd.get("images", []):
        src = os.path.join(task_dir, fn)
        if os.path.exists(src):
            _place(src, os.path.join(dst, "images", fn), copy)
            n_ok += 1
        else:
            n_miss += 1
    return n_ok, n_miss


def pack_house_public(house, public_dir, datasets, out_root, copy):
    """對外交付檔 + 影像 → 產出包（見模組說明的「交付檔模式」）。"""
    hj = os.path.join(public_dir, f"{house}.json")
    if not os.path.exists(hj):
        print(f"[skip] {house}: 找不到交付檔 {hj}")
        return False
    hd = json.load(open(hj, encoding="utf-8"))
    dst = os.path.join(out_root, house)
    os.makedirs(os.path.join(dst, "images"), exist_ok=True)

    _place(hj, os.path.join(dst, f"{house}.json"), copy=True)   # 原樣複製，不做任何改寫
    n_img, n_miss = copy_images(hd, resolve_task_dir(datasets, house), dst, copy)
    n_reg = sum(1 for p in hd.get("pairs", []) if p.get("regions"))

    print(f"[ok] {house}: 影像 {n_img}(缺 {n_miss}) · 內嵌共視區域 {n_reg} 對 · "
          f"{len(hd.get('pairs', []))} 對（交付檔模式：無語意、無 GT）")
    return True


def pack_house(house, data_dir, regions_dir, semantic_dir, datasets, out_root, copy):
    hj = os.path.join(data_dir, f"{house}.json")
    if not os.path.exists(hj):
        print(f"[skip] {house}: 找不到房子資料 {hj}")
        return False
    hd = json.load(open(hj, encoding="utf-8"))
    dst = os.path.join(out_root, house)
    os.makedirs(os.path.join(dst, "images"), exist_ok=True)
    os.makedirs(os.path.join(dst, "regions"), exist_ok=True)

    _place(hj, os.path.join(dst, f"{house}.json"), copy=True)   # 房子資料一律實體複製（小、且是核心）

    # 影像：依房子資料的 images 清單，從資料集資料夾取。
    task_dir = resolve_task_dir(datasets, house)
    n_img, n_img_miss = copy_images(hd, task_dir, dst, copy)

    # 共視區域：本房所有 <house>_<a>_<b>.json。
    n_reg = 0
    for rp in glob.glob(os.path.join(regions_dir, f"{house}_*.json")):
        _place(rp, os.path.join(dst, "regions", os.path.basename(rp)), copy=True); n_reg += 1

    # 門語意（可選）。
    sem = os.path.join(semantic_dir, f"{house}.json")
    has_sem = os.path.exists(sem)
    if has_sem:
        _place(sem, os.path.join(dst, "semantic.json"), copy=True)

    # 標註真值（可選）：帶了工具才顯示 GT 對照。
    has_gt = False
    if task_dir:
        hs = glob.glob(os.path.join(task_dir, "*_HOTSPOT.json"))
        for h in hs:
            _place(h, os.path.join(dst, os.path.basename(h)), copy=True); has_gt = True

    print(f"[ok] {house}: 影像 {n_img}(缺 {n_img_miss}) · 區域 {n_reg} · "
          f"語意 {'有' if has_sem else '無'} · GT {'有' if has_gt else '無'}")
    return True


def main():
    ap = argparse.ArgumentParser(description="流程產出 → 自足房子產出包")
    ap.add_argument("--data_dir", default=os.path.join(SCRIPT_DIR, "data"),
                    help="房子資料（<house>.json）目錄")
    ap.add_argument("--regions_dir", default=None,
                    help="共視區域目錄（預設 config.regions_root()）")
    ap.add_argument("--semantic_dir", default=os.path.join(SCRIPT_DIR, "semantic"),
                    help="門語意目錄")
    ap.add_argument("--datasets", default=None,
                    help="影像與 HOTSPOT 來源資料集根（預設 config.datasets_root()）")
    ap.add_argument("--out", required=True, help="輸出包根目錄 <bundle_root>")
    ap.add_argument("--houses", nargs="*", default=None, help="指定房子；預設來源目錄內全部")
    ap.add_argument("--copy", action="store_true",
                    help="影像實體複製（可攜包）；預設硬連結（本機省空間）")
    ap.add_argument("--from_public", nargs="?", const="", default=None,
                    metavar="DIR",
                    help="改用對外交付檔打包（只放交付檔與影像）；"
                         "省略路徑時取 <ROOT>/Output/public_houses")
    a = ap.parse_args()

    datasets = a.datasets or datasets_root()
    src_dir = a.data_dir
    if a.from_public is not None:
        src_dir = a.from_public or os.path.join(ROOT_DIR, "Output", "public_houses")
    houses = a.houses or sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(src_dir, "*.json"))
        if not os.path.basename(p).startswith("_"))
    if not houses:
        raise SystemExit(f"[error] {src_dir} 底下沒有房子資料")
    os.makedirs(a.out, exist_ok=True)

    ok = 0
    if a.from_public is not None:
        for h in houses:
            ok += pack_house_public(h, src_dir, datasets, a.out, a.copy)
        print(f"\n打包 {ok}/{len(houses)} 房（交付檔模式）→ {a.out}")
        print(f"來源 {src_dir}；包內 <house>.json 即下游收到的檔案，未經修改。")
    else:
        regions_dir = a.regions_dir or regions_root()
        for h in houses:
            ok += pack_house(h, src_dir, regions_dir, a.semantic_dir, datasets, a.out, a.copy)
        print(f"\n打包 {ok}/{len(houses)} 房 → {a.out}")


if __name__ == "__main__":
    main()
