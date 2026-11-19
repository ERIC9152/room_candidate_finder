#!/usr/bin/env python3
"""
[模組用途]
把每間房的內部 JSON 裁成「對外交付」的乾淨子集：只留下游要的欄位，
擋掉評估與排序用的內部欄位。與 `roma_geo.to_public()`（單對契約）同一個原則，
差別在這裡處理的是整間房的連接圖。

    <內部 house JSON 目錄>/<house>.json  →  <out_dir>/<house>.json

[對外保留]
    房層  house / n_images / width / height / images / az_convention
          schema_version / has_connectivity_matrix
    每對  i, j                連接圖的兩個端點（index 對應 images）
          prob                上游給的連通機率
          az_ij / az_ji       邊方向（在 i 圖看向 j 的方位角，反向同理）
          az_source           該方位從哪來。降級必須看得出來，不可與幾何驗證混為一談
          has_geometry        有沒有幾何驗證結果（false = 共視不足或未跑，方向來自上游）
          n_inlier / inlier_rate / conf_used   幾何證據強度，供下游自行 gate
          az_wedge_i_deg / az_wedge_j_deg      共視楔形寬：包住 80% 內點的最窄圓弧
                              兩側各一個，側別與 az_ij/az_ji 同樣正規化。窄代表共視證據
                              集中在單一方向，位姿條件較差；判斷可靠度時取兩者較小者
          R / t               相對位姿。僅在 has_geometry 為真時出現。t 長度為 1，跨對無共用尺度
          semantic_corrected  輸出一是否被語意門中心改寫
          regions             共視區域（有幾何驗證結果的對才有；`--no_regions` 可關閉）
                              欄位與單對契約相同：n / frac_of_inliers / a、b 兩側的
                              polygon、area_frac、az_range

[刻意不對外]
    gt_*         真值對照，只供內部評估
    az_coarse_*  幾何覆寫前的上游原值，只供內部比對
    roma_*       匹配聚合分數，內部排序用
    inliers[]    匹配內點明細，只在逐對內部結果裡

共視區域取自 per-pair 產出，預設併入，與 `roma_geo.to_public()` 的單對契約一致
（共視區域、通道方位角、可信度）。只有跑出幾何驗證結果的對才有區域，
其餘對不帶此欄位。不需要時以 `--no_regions` 關閉。

區域的欄位裁切直接呼叫 `roma_geo.region_to_public()`，不在此另寫一份：兩條對外路徑
若各自維護裁切規則，改動時漏改一邊不會有任何徵兆。內部的 `mean_conf` 與可由
`az_range` 推得的 `az_width_deg` 均不外流。

⚠ 區域的側別已在此正規化：**a 側永遠對應 i 圖、b 側永遠對應 j 圖**。per-pair 檔名的
兩張圖順序由匹配那一輪決定，未必等於 (i, j)，反序時本模組會把兩側對調後才寫出，
故下游不必也不應再自行判斷側別。
"""
import argparse
import json
import os
import tempfile

from .config import Settings
from .errors import RomaGeoError
from .names import dest_in_dir, glob_in_dir

#: 每對保留的欄位：內部名 → 對外名（改名處的理由見模組說明）。
PAIR_FIELDS = {
    "i": "i",
    "j": "j",
    "prob": "prob",
    "az_ij": "az_ij",
    "az_ji": "az_ji",
    "az_source": "az_source",
    "geo_posed": "has_geometry",
    "geo_n_inlier": "n_inlier",
    "geo_inlier_rate": "inlier_rate",
    "geo_conf_used": "conf_used",
    "geo_az_wedge_i_deg": "az_wedge_i_deg",
    "geo_az_wedge_j_deg": "az_wedge_j_deg",
    "pose_R": "R",
    "pose_t": "t",
    "semantic_corrected": "semantic_corrected",
}

HOUSE_SCHEMA_VERSION = "house-v2"

HOUSE_FIELDS = ["house", "n_images", "width", "height", "images", "az_convention",
                "has_connectivity_matrix"]

#: 出現這些前綴就是內部欄位漏出去了（輸出前自檢用）。
FORBIDDEN_PREFIXES = ("gt_", "az_coarse_", "roma_")


def pair_to_public(p: dict) -> dict:
    out = {}
    for src, dst in PAIR_FIELDS.items():
        if src in p:
            out[dst] = p[src]
    if "has_geometry" in out:
        out["has_geometry"] = bool(out["has_geometry"])
    out.setdefault("semantic_corrected", False)
    return out


def _atomic_write_json(path, obj):
    """寫入 JSON。先寫同目錄暫存檔再置換，中斷時目的檔保持原狀。"""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="._pub_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _dest_in_dir(out_dir, house):
    return dest_in_dir(out_dir, house, suffix=".json", input_name="房子 %s" % house)


def _load_json(path, input_name):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise RomaGeoError(
            "%s：%s" % (type(e).__name__, e),
            input_name=input_name,
            condition="無法解析為 JSON",
        ) from e


class ExportResult(list):
    """寫出的路徑清單。`pose_status` 為 complete（相對位姿齊全）或 absent（整批缺席）。"""

    def __init__(self, paths, pose_status="complete"):
        list.__init__(self, paths)
        self.pose_status = pose_status


def _pair_stems(pub, p):
    names = pub.get("images") or []
    try:
        a = os.path.splitext(names[p["i"]])[0]
        b = os.path.splitext(names[p["j"]])[0]
        return a, b
    except (IndexError, KeyError, TypeError):
        return str(p.get("i")), str(p.get("j"))


def _pose_status_or_raise(prepared, regions_dir):
    """依每一對自己的產物檔判定。檔在卻沒位姿或部份缺漏則中止；檔都不在才降級。"""
    missing = []
    empty_pose = []
    n_rt = 0
    for _f, house, pub in prepared:
        for p in pub.get("pairs") or []:
            has = bool(p.get("has_geometry"))
            has_rt = "R" in p and "t" in p
            if has_rt and not has:
                raise RomaGeoError(
                    "無位姿卻帶 R/t",
                    input_name="房子 %s" % house,
                    condition="has_geometry 與相對位姿不同步",
                )
            if not has:
                continue
            a, b = _pair_stems(pub, p)
            path, _swap = _find_pair_json(regions_dir, house, a, b)
            if has_rt:
                n_rt += 1
            elif path:
                empty_pose.append((house, a, b))
            else:
                missing.append((house, a, b))
    if empty_pose:
        house, a, b = empty_pose[0]
        raise RomaGeoError(
            "配對 %s/%s 的逐對產物沒有相對位姿" % (a, b),
            input_name="房子 %s" % house,
            condition="has_geometry 與相對位姿不同步",
        )
    if missing and n_rt:
        house, a, b = missing[0]
        raise RomaGeoError(
            "配對 %s/%s 有 has_geometry 但逐對產物缺席" % (a, b),
            input_name="房子 %s" % house,
            condition="相對位姿部分缺漏",
        )
    if missing:
        return "absent"
    return "complete"


def house_to_public(d: dict) -> dict:
    """內部 house JSON dict → 對外子集（純函式，不碰檔案）。"""
    out = {k: d[k] for k in HOUSE_FIELDS if k in d}
    out["schema_version"] = d.get("schema_version", HOUSE_SCHEMA_VERSION)
    out["pairs"] = [pair_to_public(p) for p in d.get("pairs", [])]
    return out


def leaked_fields(d: dict) -> set:
    """回傳輸出裡仍帶內部前綴的欄位名（正常應為空集合）。"""
    bad = set()
    for p in d.get("pairs", []):
        bad |= {k for k in p if k.startswith(FORBIDDEN_PREFIXES)}
    bad |= {k for k in d if k.startswith(FORBIDDEN_PREFIXES)}
    return bad


def swap_sides(region: dict) -> dict:
    """對調共視區域的 a/b 兩側。per-pair 檔的圖序與 (i, j) 相反時用來正規化。"""
    return dict(region, a=region["b"], b=region["a"])


def attach_regions(house: str, pub: dict, regions_dir: str) -> int:
    """把 per-pair JSON 的共視區域併進每一對；回傳成功併入的對數。

    per-pair 檔名的兩張圖順序由匹配那一輪決定，未必等於 (i, j)。兩種順序都找，找到
    反序的就把兩側對調，使輸出中**區域的 a 側永遠是 i 圖、b 側永遠是 j 圖**。
    側別統一在此處理，下游不必判斷；若留給下游判斷，漏判的症狀是左右對調而畫面
    仍然合理，不會有任何錯誤訊息。

    併入前一律經 region_to_public() 裁成對外欄位。
    """
    from . import region_to_public
    names = pub.get("images", [])
    n = 0
    for p in pub["pairs"]:
        try:
            a, b = (os.path.splitext(names[k])[0] for k in (p["i"], p["j"]))
        except (IndexError, KeyError):
            continue
        for swap, path in ((False, os.path.join(regions_dir, f"{house}_{a}_{b}.json")),
                           (True, os.path.join(regions_dir, f"{house}_{b}_{a}.json"))):
            if not os.path.isfile(path):
                continue
            regions = _load_json(path, "逐對產物 %s" % os.path.basename(path)).get("regions") or []
            if regions:
                p["regions"] = [region_to_public(swap_sides(r) if swap else r)
                                for r in regions]
                n += 1
            break
    return n


def _find_pair_json(regions_dir, house, a, b):
    fwd = os.path.join(regions_dir, "%s_%s_%s.json" % (house, a, b))
    rev = os.path.join(regions_dir, "%s_%s_%s.json" % (house, b, a))
    if os.path.isfile(fwd):
        return fwd, False
    if os.path.isfile(rev):
        return rev, True
    return None, False


def attach_pose(house, pub, regions_dir):
    """把逐對結果的 R、t 併入有位姿的對。側別與 (i, j) 對齊。缺席時不填充。"""
    from .semantic import invert_pose
    names = pub.get("images", [])
    n = 0
    for p in pub["pairs"]:
        if not p.get("has_geometry"):
            p.pop("R", None)
            p.pop("t", None)
            continue
        try:
            a, b = (os.path.splitext(names[k])[0] for k in (p["i"], p["j"]))
        except (IndexError, KeyError):
            continue
        path, swap = _find_pair_json(regions_dir, house, a, b)
        if not path:
            continue
        pose = _load_json(path, "逐對產物 %s" % os.path.basename(path)).get("pose")
        if not pose or pose.get("R") is None or pose.get("t") is None:
            continue
        R, t = pose["R"], pose["t"]
        if swap:
            R, t = invert_pose(R, t)
        else:
            R = [[round(float(x), 6) for x in row] for row in R]
            t = [round(float(x), 6) for x in t]
        p["R"] = R
        p["t"] = t
        n += 1
    return n


#: 受限來源的標記檔名。資料集裡某間房的目錄底下出現這個檔，即表示該房的來源帶有
#: 使用範圍限制（例如授權僅及於研究評測），不得出現在對外交付內容中。
#: 檔案內容為限制說明的文字，會被引用到錯誤訊息裡。
RESTRICTED_MARKER = "_RESTRICTED"


def restricted_reason(house, datasets_dir=None):
    """房屋是否來自受限來源；是則回傳限制說明，否則回 None。

    判定依據是資料集目錄底下的標記檔，不認任何特定來源的名稱。日後接上其他受限
    來源時，只要在該來源的房屋目錄放上同名標記檔即可，這裡不必再改。
    房屋目錄可能同名巢狀，故往下遞迴尋找。
    """
    if not datasets_dir:
        from .config import datasets_root
        datasets_dir = datasets_root()
    base = os.path.join(datasets_dir, house)
    if not os.path.isdir(base):
        return None
    hits = [os.path.join(base, RESTRICTED_MARKER)]
    hits += glob_in_dir(base, os.path.join("**", RESTRICTED_MARKER), recursive=True)
    for p in hits:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return f.read().strip() or "（標記檔未附說明）"
            except OSError:
                return "（標記檔無法讀取）"
    return None


def export_houses(in_dir, out_dir, regions_dir=None, with_regions=True,
                  datasets_dir=None, houses=None, semantic_correct=False,
                  semantic_dir=None):
    """把內部房子 JSON 裁成對外交付子集。失敗時丟 RomaGeoError。"""

    files = [f for f in sorted(glob_in_dir(in_dir, "*.json"))
             if not os.path.basename(f).startswith("_")]
    loaded = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        try:
            d = _load_json(f, "房子資料 %s" % os.path.basename(f))
        except RomaGeoError:
            if houses is not None and stem not in houses:
                continue
            raise
        house = d.get("house") or stem
        loaded.append((f, house, d))

    if houses is not None:
        want = set(houses)
        loaded = [(f, h, d) for f, h, d in loaded if h in want]
        missing = want - {h for _, h, _ in loaded}
        if missing:
            name = sorted(missing)[0]
            raise RomaGeoError(
                "找不到 house 欄位為 %s 的來源" % name,
                input_name="房子 %s" % name,
                condition="比對依據為 JSON 的 house 欄位；目錄 %s" % in_dir,
            )
    if not loaded:
        raise RomaGeoError(
            "底下沒有 house JSON",
            input_name="房子資料",
            condition="目錄 %s 沒有可匯出的 JSON" % in_dir,
        )

    seen_files = {}
    for f, house, _d in loaded:
        dest_in_dir(out_dir, house, suffix=".json", input_name="房子 %s" % house)
        if house in seen_files:
            raise RomaGeoError(
                "識別名稱 %s 同時出現在 %s 與 %s"
                % (house, os.path.basename(seen_files[house]), os.path.basename(f)),
                input_name="房子 %s" % house,
                condition="同一批來源的識別名稱重複",
            )
        seen_files[house] = f

    if not regions_dir:
        from .config import regions_root
        regions_dir = regions_root()

    names = [h for _f, h, _d in loaded]
    existing = []
    if os.path.isdir(out_dir):
        existing = [os.path.splitext(os.path.basename(p))[0]
                    for p in glob_in_dir(out_dir, "*.json")
                    if not os.path.basename(p).startswith("_")]
    blocked = []
    seen = set()
    for house in names + existing:
        if house in seen:
            continue
        seen.add(house)
        why = restricted_reason(house, datasets_dir=datasets_dir)
        if why:
            blocked.append((house, why))
    if blocked:
        os.makedirs(out_dir, exist_ok=True)
        for house, _why in blocked:
            leftover = _dest_in_dir(out_dir, house)
            if os.path.isfile(leftover):
                os.remove(leftover)
        house, why = blocked[0]
        raise RomaGeoError(
            why,
            input_name="房子 %s" % house,
            condition="來自受限來源，不得對外交付",
        )

    os.makedirs(out_dir, exist_ok=True)
    prepared = []
    tot_regions = 0
    for f, house, d in loaded:
        pub = house_to_public(d)
        need_regions = with_regions or semantic_correct
        if need_regions:
            tot_regions += attach_regions(house, pub, regions_dir)
        attach_pose(house, pub, regions_dir)
        if semantic_correct:
            from .semantic import apply_semantic_correction, load_doors_json
            doors = {}
            if semantic_dir:
                doors = load_doors_json(os.path.join(semantic_dir, house + ".json"))
            apply_semantic_correction(pub, doors)
        if need_regions and not with_regions:
            for p in pub["pairs"]:
                p.pop("regions", None)
        bad = leaked_fields(pub)
        if bad:
            raise RomaGeoError(
                "輸出帶有內部欄位：%s" % sorted(bad),
                input_name="房子 %s" % house,
                condition="對外子集混入了內部欄位",
            )
        prepared.append((f, house, pub))

    pose_status = _pose_status_or_raise(prepared, regions_dir)

    tot_in = tot_out = tot_pairs = 0
    written = []
    for f, house, pub in prepared:
        out_path = _dest_in_dir(out_dir, house)
        _atomic_write_json(out_path, pub)
        tot_in += os.path.getsize(f)
        tot_out += os.path.getsize(out_path)
        tot_pairs += len(pub["pairs"])
        written.append(out_path)

    print(f"{len(loaded)} 房 · {tot_pairs} 對 → {out_dir}")
    if tot_in:
        print(f"體積 {tot_in/1e6:.2f} MB → {tot_out/1e6:.2f} MB（{tot_out/tot_in:.0%}）")
    if with_regions:
        print(f"併入共視區域：{tot_regions} 對（來源 {regions_dir}；a 側 = i 圖）")
    else:
        print("未併入共視區域（--no_regions）")
    ds = datasets_dir
    if not ds:
        from .config import datasets_root
        ds = datasets_root()
    if os.path.isdir(ds):
        print("自檢：輸出無 gt_* / az_coarse_* / roma_* 欄位，且無受限來源的房屋")
    else:
        print("自檢：輸出無 gt_* / az_coarse_* / roma_* 欄位；受限來源未驗到（資料集目錄不存在）")
    if pose_status == "absent":
        print("降級：相對位姿整批缺席")
    return ExportResult(written, pose_status)


def main():
    s = Settings()
    ap = argparse.ArgumentParser(description="每間房的內部 JSON → 對外交付子集")
    ap.add_argument("--in_dir", default=None,
                    help="內部 house JSON 目錄（預設 Settings.house_data_dir()）")
    ap.add_argument("--out_dir", default=None,
                    help="輸出目錄（預設 Settings.public_dir()）")
    ap.add_argument("--no_regions", action="store_true",
                    help="不併入共視區域（預設併入，來源 --regions_dir）")
    ap.add_argument("--regions_dir", default=None,
                    help="per-pair JSON 目錄（預設 config.regions_root()）")
    a = ap.parse_args()
    in_dir = a.in_dir or s.house_data_dir()
    out_dir = a.out_dir or s.public_dir()
    try:
        result = export_houses(in_dir, out_dir, regions_dir=a.regions_dir,
                               with_regions=not a.no_regions,
                               datasets_dir=s.datasets_root())
    except RomaGeoError as e:
        raise SystemExit("[error] %s" % e)
    if getattr(result, "pose_status", "complete") == "absent":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
