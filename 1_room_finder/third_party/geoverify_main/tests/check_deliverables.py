#!/usr/bin/env python3
"""
[腳本用途]
檢查磁碟上**已產出的交付物**是否仍符合契約。與其他測試檔的差別在於：那些驗證程式碼的
行為，本檔驗證資料本身。

用途有二。其一，批次跑完後確認產出可交付；其二，收到別處（其他機器、其他版本、
上游）產生的檔案時，先確認格式相符再接。

檢查五類產物，各自在來源不存在時回報「略過」。外部環境本來就不會五類齊備：

    per-pair JSON      共視區域與位姿，schema wp1-step3-v2
    _summary.csv       逐對指標總表，欄位須與批次程式宣告的一致
    連通性矩陣          上游輸入，(N,N,3) 與影像清單對齊
    house JSON         工具用的每房連接圖
    對外交付子集        裁切後的每房連接圖，不得含內部欄位

[用法]
    venv\\Scripts\\python.exe tests\\check_deliverables.py            # 各類抽驗 200 份
    venv\\Scripts\\python.exe tests\\check_deliverables.py --all      # 全量
    venv\\Scripts\\python.exe tests\\check_deliverables.py --regions_dir <路徑>

離開碼 0 表示沒有任何一項失敗（含全部略過的情況）。
"""
import argparse
import csv
import json
import math
import os
import sys

from _support import ROOT  # noqa: F401（匯入時補 sys.path）

import export_public
import roma_geo
import upstream_direction
from config import Settings, datasets_root
from roma_geo.names import glob_in_dir
from pose_regions.batch_pose_from_cache import SCHEMA_VERSION, SUMMARY_COLS

OK, FAIL, SKIP = "OK", "FAIL", "SKIP"

#: 每項最多列出幾筆問題明細，避免整批不符時洗版。
MAX_DETAIL = 5


class Result:
    def __init__(self, name, status, detail="", problems=None, n=0):
        self.name, self.status, self.detail = name, status, detail
        self.problems, self.n = problems or [], n

    def line(self):
        return "[%-4s] %-16s %s" % (self.status, self.name, self.detail)


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sample(files, limit):
    """決定性等距抽樣：同一批檔案在任何機器上都取到同一組。"""
    if not limit or len(files) <= limit:
        return files
    step = len(files) / float(limit)
    return [files[int(i * step)] for i in range(limit)]


def az_formula(s):
    """取方位規約字串中的公式部分（分號之前）。

    分號之後是給人看的註記，各處寫法不同：house JSON 附前端反解 `u=...`，
    `upstream_direction.CONVENTION` 附「0deg = image center」。定義方向的是公式，
    註記不同不構成不一致，公式不同才是。
    """
    return (s or "").split(";")[0].strip().replace(" ", "")


# ---------- per-pair JSON ----------
def check_pair_jsons(regions_dir, limit):
    files = sorted(f for f in glob_in_dir(regions_dir, "*.json")
                   if not os.path.basename(f).startswith("_"))
    if not files:
        return Result("per-pair JSON", SKIP, "%s 底下沒有 per-pair JSON" % regions_dir)

    chosen = _sample(files, limit)
    problems, n_posed, n_noco, versions = [], 0, 0, {}

    for f in chosen:
        base = os.path.basename(f)
        try:
            d = _load(f)
        except Exception as e:
            problems.append("%s: 無法解析（%s）" % (base, type(e).__name__))
            continue

        v = d.get("schema_version")
        versions[v] = versions.get(v, 0) + 1
        for k in ("img_size_a", "img_size_b", "params", "n", "conf_used",
                  "inlier_rate", "pose", "flags", "regions"):
            if k not in d:
                problems.append("%s: 缺欄位 %s" % (base, k))

        n = d.get("n") or {}
        flags = d.get("flags") or {}
        pose = d.get("pose")
        noco = bool(flags.get("no_covisibility"))

        # 位姿的有無必須與旗標、conf_used 三者一致，下游據此分辨「沒證據」與「沒跑」。
        if noco != (pose is None):
            problems.append("%s: no_covisibility=%s 但 pose=%s" % (base, noco, pose is not None))
        if noco != (d.get("conf_used") is None):
            problems.append("%s: no_covisibility 與 conf_used 不一致" % base)

        if pose is None:
            n_noco += 1
        else:
            n_posed += 1
            az = pose.get("az_ab_deg")
            if az is None or not (-180.0 <= az <= 180.0):
                problems.append("%s: az_ab_deg 超出範圍（%s）" % (base, az))
            if not (0.0 <= float(pose.get("baseline_txz", 0.0)) <= 1.0001):
                problems.append("%s: baseline_txz 非單位向量分量" % base)
            used, inl = n.get("used", 0), n.get("inlier", 0)
            if used and abs(d.get("inlier_rate", 0) - inl / used) > 5e-4:
                problems.append("%s: inlier_rate 與 n.inlier/n.used 不符" % base)
            if inl > used:
                problems.append("%s: n.inlier > n.used" % base)
            if inl < 8:
                problems.append("%s: 有位姿但內點 < 8" % base)

        total_frac = sum(r.get("frac_of_inliers", 0.0) for r in (d.get("regions") or []))
        if total_frac > 1.001:
            problems.append("%s: 區域內點占比總和 %.3f > 1" % (base, total_frac))
        for r in (d.get("regions") or []):
            if r.get("n", 0) < 8:
                problems.append("%s: 區域成員數 < 8" % base)
            for side in ("a", "b"):
                g = r.get(side) or {}
                if not (0.0 <= g.get("area_frac", -1) <= 1.0):
                    problems.append("%s: 區域 %s 側 area_frac 超出 [0,1]" % (base, side))
                if not g.get("polygon"):
                    problems.append("%s: 區域 %s 側缺多邊形" % (base, side))

        inliers = d.get("inliers") or []
        if len(inliers) > 2000:
            problems.append("%s: inliers 超過 2000 點上限" % base)
        if inliers and len(inliers[0]) != 5:
            problems.append("%s: inliers 每列須為 5 欄" % base)

    other = {k: v for k, v in versions.items() if k != SCHEMA_VERSION}
    if other:
        problems.append("非 %s 的 schema：%s" % (SCHEMA_VERSION, other))

    detail = ("%d 份（抽驗 %d）· 有位姿 %d · 無共視 %d · schema %s"
              % (len(files), len(chosen), n_posed, n_noco, SCHEMA_VERSION))
    return Result("per-pair JSON", FAIL if problems else OK, detail, problems, len(chosen))


# ---------- _summary.csv ----------
def check_summary_csv(regions_dir):
    path = os.path.join(regions_dir, "_summary.csv")
    if not os.path.isfile(path):
        return Result("_summary.csv", SKIP, "不存在")
    problems = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    if cols != list(SUMMARY_COLS):
        missing = [c for c in SUMMARY_COLS if c not in cols]
        extra = [c for c in cols if c not in SUMMARY_COLS]
        problems.append("欄位與批次程式宣告不符（缺 %s／多 %s）" % (missing or "無", extra or "無"))
    for i, r in enumerate(rows):
        if r.get("flag_no_covisibility") == "1" and r.get("az_ab_deg"):
            problems.append("第 %d 列：標為無共視卻有方位角" % (i + 2))
        if len(problems) >= MAX_DETAIL:
            break
    return Result("_summary.csv", FAIL if problems else OK,
                  "%d 列 · %d 欄" % (len(rows), len(cols)), problems, len(rows))


# ---------- 連通性矩陣 ----------
def check_matrices(matrices_dir):
    npys = sorted(glob_in_dir(matrices_dir, "*.npy"))
    npys = [f for f in npys if not f.endswith("_doormap.npy")]
    if not npys:
        return Result("連通性矩陣", SKIP, "%s 底下沒有 .npy" % matrices_dir)

    import numpy as np
    problems, total_n = [], 0
    for f in npys:
        house = os.path.splitext(os.path.basename(f))[0]
        imgs_path = os.path.join(matrices_dir, house + "_images.json")
        if not os.path.isfile(imgs_path):
            problems.append("%s: 缺影像清單 %s_images.json" % (house, house))
            continue
        M = np.load(f)
        try:
            images = _load(imgs_path)
        except Exception as e:
            problems.append("%s: 無法解析影像清單（%s）" % (house, type(e).__name__))
            continue
        total_n += len(images)
        if M.ndim != 3 or M.shape[2] != 3:
            problems.append("%s: 形狀應為 (N,N,3)，實得 %s" % (house, (M.shape,)))
            continue
        if M.shape[0] != M.shape[1] or M.shape[0] != len(images):
            problems.append("%s: 矩陣 %s 與影像清單長度 %d 不符"
                            % (house, (M.shape,), len(images)))
            continue
        if not np.isfinite(M).all():
            problems.append("%s: 含 NaN 或 inf" % house)
        if not np.allclose(M[:, :, 0], M[:, :, 0].T, atol=1e-4):
            problems.append("%s: ch0（連通分數）不對稱" % house)
        r = np.hypot(M[:, :, 1], M[:, :, 2])
        off = ~np.eye(len(images), dtype=bool)
        bad = ((r[off] > 1e-4) & (np.abs(r[off] - 1.0) > 1e-3)).sum()
        if bad:
            problems.append("%s: %d 個方位項既非未填(0)也非單位長" % (house, bad))
        if any(not isinstance(x, str) or "." not in x for x in images):
            problems.append("%s: 影像清單須為含副檔名的檔名" % house)
    return Result("連通性矩陣", FAIL if problems else OK,
                  "%d 房 · 共 %d 張影像" % (len(npys), total_n), problems, len(npys))


# ---------- house JSON ----------
def check_house_json(data_dir):
    files = sorted(f for f in glob_in_dir(data_dir, "*.json")
                   if not os.path.basename(f).startswith("_"))
    if not files:
        return Result("house JSON", SKIP, "%s 底下沒有 house JSON" % data_dir)

    problems, tot_pairs, n_geo = [], 0, 0
    for f in files:
        house = os.path.splitext(os.path.basename(f))[0]
        try:
            d = _load(f)
        except Exception as e:
            problems.append("%s: 無法解析（%s）" % (house, type(e).__name__))
            continue
        for k in ("house", "n_images", "width", "height", "images", "pairs", "az_convention"):
            if k not in d:
                problems.append("%s: 缺欄位 %s" % (house, k))
        if az_formula(d.get("az_convention")) != az_formula(upstream_direction.CONVENTION):
            problems.append("%s: 方位規約公式與 upstream_direction.CONVENTION 不符（%r）"
                            % (house, d.get("az_convention")))
        n_img = len(d.get("images") or [])
        if d.get("n_images") != n_img:
            problems.append("%s: n_images 與影像清單長度不符" % house)
        for p in d.get("pairs") or []:
            tot_pairs += 1
            if not (0 <= p.get("i", -1) < n_img and 0 <= p.get("j", -1) < n_img):
                problems.append("%s: pair 索引超出影像範圍" % house)
            src = p.get("az_source")
            degraded = d.get("has_connectivity_matrix") is False
            if src is None:
                if not degraded:
                    problems.append("%s: az_source 缺席，且此房不是連通矩陣缺席的降級" % house)
            elif src not in ("geo", "coarse"):
                problems.append("%s: az_source 非 geo/coarse（%s）" % (house, src))
            if p.get("az_source") == "geo":
                n_geo += 1
                if not p.get("geo_posed"):
                    problems.append("%s: az_source=geo 但 geo_posed 為假" % house)
            vij, vji = p.get("az_ij"), p.get("az_ji")
            if vij is None and vji is None:
                if not degraded:
                    problems.append("%s: 通道方位為空值，且此房不是連通矩陣缺席的降級" % house)
            elif vij is None or vji is None:
                problems.append("%s: az_ij 與 az_ji 須同為數值或同為空值" % house)
            else:
                for key, v in (("az_ij", vij), ("az_ji", vji)):
                    if not (0.0 <= v < 360.0):
                        problems.append("%s: %s 超出 [0,360)（%s）" % (house, key, v))
            if len(problems) > 50:
                break
    pct = (100.0 * n_geo / tot_pairs) if tot_pairs else 0.0
    return Result("house JSON", FAIL if problems else OK,
                  "%d 房 · %d 對 · 方位來自幾何 %d 對（%.1f%%）"
                  % (len(files), tot_pairs, n_geo, pct), problems, len(files))


# ---------- 對外交付子集 ----------
def check_public_houses(pub_dir, datasets_dir=None):
    files = sorted(f for f in glob_in_dir(pub_dir, "*.json")
                   if not os.path.basename(f).startswith("_"))
    if not files:
        return Result("對外交付子集", SKIP, "%s 底下沒有輸出" % pub_dir)

    allowed_pair = set(export_public.PAIR_FIELDS.values()) | {"regions"}
    # 區域欄位與單對契約同一個定義，兩條路徑共用 roma_geo 的常數。
    allowed_region = set(roma_geo.PUBLIC_REGION_FIELDS) | {"a", "b"}
    allowed_side = set(roma_geo.PUBLIC_REGION_SIDE_FIELDS)
    problems, tot_pairs, n_regions = [], 0, 0
    geom_missing, n_geom_rt = [], 0
    ds = datasets_dir if datasets_dir is not None else datasets_root()
    ds_missing = not os.path.isdir(ds)
    for f in files:
        house = os.path.splitext(os.path.basename(f))[0]
        try:
            d = _load(f)
        except Exception as e:
            problems.append("%s: 無法解析（%s）" % (house, type(e).__name__))
            continue
        leaked = export_public.leaked_fields(d)
        if leaked:
            problems.append("%s: 含內部欄位 %s" % (house, sorted(leaked)))
        if not ds_missing:
            why = export_public.restricted_reason(house, datasets_dir=ds)
            if why:
                problems.append("%s: 來自受限來源，不得對外交付（%s）"
                                % (house, why.splitlines()[0]))
        n_img = len(d.get("images") or [])
        for p in d.get("pairs") or []:
            tot_pairs += 1
            extra = set(p) - allowed_pair
            if extra:
                problems.append("%s: pair 含契約外欄位 %s" % (house, sorted(extra)))
            if "has_geometry" in p and not isinstance(p["has_geometry"], bool):
                problems.append("%s: has_geometry 非布林" % house)
            if d.get("schema_version") == export_public.HOUSE_SCHEMA_VERSION:
                if "semantic_corrected" not in p:
                    problems.append("%s: house-v2 缺 semantic_corrected" % house)
                elif not isinstance(p["semantic_corrected"], bool):
                    problems.append("%s: semantic_corrected 非布林" % house)
                has = bool(p.get("has_geometry"))
                has_rt = "R" in p and "t" in p
                if has and has_rt:
                    n_geom_rt += 1
                elif has:
                    imgs = d.get("images") or []
                    try:
                        a = os.path.splitext(imgs[p["i"]])[0]
                        b = os.path.splitext(imgs[p["j"]])[0]
                    except (IndexError, KeyError, TypeError):
                        a, b = str(p.get("i")), str(p.get("j"))
                    geom_missing.append("%s: 相對位姿缺席（配對 %s/%s）" % (house, a, b))
                elif has_rt or "R" in p or "t" in p:
                    problems.append("%s: 無位姿卻帶 R/t" % house)
                if "az_ij" not in p or "az_ji" not in p:
                    problems.append("%s: 通道方位缺席" % house)
                else:
                    degraded = d.get("has_connectivity_matrix") is False
                    vij, vji = p.get("az_ij"), p.get("az_ji")
                    empty = vij is None and vji is None
                    if empty:
                        if not degraded:
                            problems.append(
                                "%s: 通道方位為空值，且此房不是連通矩陣缺席的降級" % house)
                        if "az_source" in p:
                            problems.append("%s: 無方位卻帶 az_source" % house)
                    elif vij is None or vji is None:
                        problems.append("%s: az_ij 與 az_ji 須同為數值或同為空值" % house)
                    else:
                        for key, v in (("az_ij", vij), ("az_ji", vji)):
                            if not (0.0 <= v < 360.0):
                                problems.append("%s: %s 超出 [0,360)（%s）"
                                                % (house, key, v))
                        src = p.get("az_source")
                        if src not in ("geo", "coarse"):
                            problems.append("%s: 有方位卻缺 az_source" % house)
            if not (0 <= p.get("i", -1) < n_img and 0 <= p.get("j", -1) < n_img):
                problems.append("%s: pair 索引超出影像範圍" % house)
            if "regions_order" in p:
                # 舊版輸出把側別判斷留給下游；現行契約已在匯出時正規化並移除此欄位。
                problems.append("%s: 帶有已移除的 regions_order 欄位，請重新匯出" % house)
            if "regions" in p:
                n_regions += 1
                for r in p["regions"]:
                    extra_r = set(r) - allowed_region
                    if extra_r:
                        problems.append("%s: 區域含契約外欄位 %s" % (house, sorted(extra_r)))
                    for side in ("a", "b"):
                        g = r.get(side) or {}
                        if not g.get("polygon"):
                            problems.append("%s: 區域缺 %s 側多邊形" % (house, side))
                        extra_s = set(g) - allowed_side
                        if extra_s:
                            problems.append("%s: 區域 %s 側含契約外欄位 %s"
                                            % (house, side, sorted(extra_s)))
    detail = "%d 房 · %d 對" % (len(files), tot_pairs)
    if n_regions:
        detail += " · 含共視區域 %d 對" % n_regions
    pose_partial = bool(geom_missing) and n_geom_rt > 0
    pose_absent = bool(geom_missing) and n_geom_rt == 0
    if pose_partial:
        problems = ["相對位姿部分缺漏"] + geom_missing[:MAX_DETAIL] + problems
    if pose_absent:
        note = "相對位姿整批缺席（%d 對 has_geometry 而無 R/t）" % len(geom_missing)
        problems = [note] + problems
        detail += "；%s" % note
    skip_notes = []
    if ds_missing:
        skip_notes.append("受限來源未驗到：資料集目錄不存在（%s）" % ds)
    if problems:
        return Result("對外交付子集", FAIL, detail, skip_notes + problems, len(files))
    if skip_notes:
        return Result("對外交付子集", SKIP,
                      "%s；%s" % (detail, "；".join(skip_notes)),
                      [], len(files))
    return Result("對外交付子集", OK, detail, [], len(files))


# ---------- 匯總 ----------
def run_checks(limit=200, regions_dir=None, matrices_dir=None,
               house_dir=None, public_dir=None, datasets_dir=None):
    s = Settings()
    regions_dir = regions_dir or s.regions_root()
    matrices_dir = matrices_dir or s.matrices_root()
    house_dir = house_dir or s.house_data_dir()
    public_dir = public_dir or s.public_dir()
    return [check_pair_jsons(regions_dir, limit),
            check_summary_csv(regions_dir),
            check_matrices(matrices_dir),
            check_house_json(house_dir),
            check_public_houses(public_dir, datasets_dir=datasets_dir)]


def report(results, stream=sys.stdout):
    for r in results:
        print(r.line(), file=stream)
        for p in r.problems[:MAX_DETAIL]:
            print("         · %s" % p, file=stream)
        if len(r.problems) > MAX_DETAIL:
            print("         · …另有 %d 項" % (len(r.problems) - MAX_DETAIL), file=stream)
    n_fail = sum(1 for r in results if r.status == FAIL)
    n_skip = sum(1 for r in results if r.status == SKIP)
    skip_note = "（來源不存在）" if n_skip else ""
    print("\n%d 項通過 · %d 項失敗 · %d 項略過%s"
          % (len(results) - n_fail - n_skip, n_fail, n_skip, skip_note), file=stream)
    return n_fail


def main(argv=None):
    ap = argparse.ArgumentParser(description="檢查已產出的交付物是否符合契約")
    ap.add_argument("--all", action="store_true", help="全量檢查（預設每類抽驗 200 份）")
    ap.add_argument("--limit", type=int, default=200, help="每類抽驗份數")
    ap.add_argument("--regions_dir", default=None, help="per-pair JSON 目錄")
    ap.add_argument("--matrices_dir", default=None, help="連通性矩陣目錄")
    ap.add_argument("--house_dir", default=None, help="house JSON 目錄")
    ap.add_argument("--public_dir", default=None, help="對外交付子集目錄")
    ap.add_argument("--datasets", default=None, help="資料集根目錄（判定受限來源）")
    a = ap.parse_args(argv)

    results = run_checks(limit=0 if a.all else a.limit,
                         regions_dir=a.regions_dir, matrices_dir=a.matrices_dir,
                         house_dir=a.house_dir, public_dir=a.public_dir,
                         datasets_dir=a.datasets)
    return 1 if report(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
