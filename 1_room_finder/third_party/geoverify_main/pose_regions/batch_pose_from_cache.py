#!/usr/bin/env python3
"""
全量批次:快取內每一對逐對估計相對位姿 + 品質指標,輸出 per-pair JSON。

輸入:roma_match/cache/*.npz(座標約定 = 原圖 ERP 像素,見 README)
輸出:
  Output/roma_regions/{house}_{imgA}_{imgB}.json   per-pair 結果(schema 見 README)
  Output/roma_regions/_summary.csv                 每 pair 一列的指標總表(供驗收/出圖)
  Output/roma_regions/batch_pose_report.md         耗時 + 指標分佈報告(自動生成)

品質指標:
  - inlier 率(對極 RANSAC 內點 / 使用點數)
  - 方位覆蓋度 / 楔形寬度:內點在 A/B 兩側的方位角分佈：「涵蓋幾成 10° bin」與
    「包住 80% 內點的最窄圓弧」。跨房穿門 pair 的 match 本來就集中在門洞楔形,
    要與退化垃圾區分,不能一律當退化丟。
  - rot-only 殘差:內點被純旋轉解釋的中位角誤差(近零基線偵測)
  - 重投影(Sampson)殘差統計:內點的中位數 / p90(角度)

自適應 conf:基準 conf 0.65;若「點數<30 或 RANSAC 內點<8」
則降到 retry_conf(0.3)重試;實際使用的門檻記入 JSON `conf_used`。兩級都失敗才標
`no_covisibility`。穿門窄楔形 pair 在 0.65 常只剩 <20 點,這一步是拼接位姿覆蓋率的關鍵。

旗標(定義同步寫在 README):
  low_matches         conf>=0.65 的點 < 30(品質註記;自適應後不再是「跳過原因」)
  no_covisibility     兩級 conf 都湊不出 >=8 內點 → 無位姿(語意=無共視證據,非「不連通」)
  degenerate_rot_only rot-only 中位殘差 < 1.5×thresh_deg(近零基線,t 方向不可靠)
  doorway_wedge       非退化 且 inlier_rate>=0.3 且 n_inlier>=30 且兩側楔形寬 <=75°
                      (窄但有效的門洞叢集)

GT 欄位(供驗收交叉比對,非計算輸入):
  gt_label   由 *_HOTSPOT.json 的 ToIDName 推導(該房完全沒有 HOTSPOT 檔時為空)
  gt_az_*    datasets 內 *_HOTSPOT.json 的 GT hotspot 方位(僅直接相鄰 pair 有)。
             注意語意:hotspot 標的是「通道位置」;只有同房 pair 才近似指向對方相機。
             但直接相鄰 pair 的 B 相機通常就在通道後方,兩語意的方位高度重合,
             故 err_az_* 當一致性檢核用,不當嚴格位姿精度。
  same_room  兩 pano 的 HOTSPOT.json (floor, Roomidx) 相同 → 同房(缺 Roomidx 則空)
  位姿方位 az_ab = A 幀中 B 相機方向(-R^T t);與 gt_az_ab 的圓角誤差 = err_az_ab

用法:
  venv/Scripts/python.exe pose_regions/batch_pose_from_cache.py            # 全量
  venv/Scripts/python.exe pose_regions/batch_pose_from_cache.py --limit 20 # 冒煙
  --resume 會跳過已存在的 JSON(重讀其內容補 summary 列),中斷可續跑。
"""
import argparse, csv, importlib.util, json, os, sys, tempfile, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
# 套件未安裝時(直跑本腳本)補上 src,僅為了讓下面的 roma_geo 延遲匯入找得到套件。
# 能匯入就完全不動搜尋路徑:本模組會被 roma_geo 匯入,不該因此改變呼叫端的搜尋路徑。
if importlib.util.find_spec("roma_geo") is None:
    sys.path.append(os.path.join(ROOT, "src"))

# 模組化(2026-07-23):匯入零副作用(chdir 移入 main());套件(from pose_regions import …)
# 與腳本直跑(python pose_regions/batch_pose_from_cache.py)雙模式匯入。
try:
    from .estimate_relative_pose import (parse_cache_name, find_house_image, erp_to_bearing,
                                         ransac_E, select_pose, rotation_only_fit, sampson_residual,
                                         CACHE_SUFFIX, match_cache_to_dataset, identify_cache_file)
except ImportError:
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from estimate_relative_pose import (parse_cache_name, find_house_image, erp_to_bearing,
                                        ransac_E, select_pose, rotation_only_fit, sampson_residual,
                                        CACHE_SUFFIX, match_cache_to_dataset, identify_cache_file)

SCHEMA_VERSION = "wp1-step3-v2"   # v2(2026-07-23):nadir 前濾預設開啟(lat_filter 60);params.lat_max / n.lat_drop 欄位

# 快取與產出位置可由環境變數覆寫(同 config.cache_root / regions_root)。此處直接讀取
# 環境變數而不 import config,以維持 pose_regions 可獨立匯入;變數名異動需三處一致。
CACHE_DIR = (os.path.abspath(os.path.expanduser(os.environ["ROMA_GEO_CACHE"]))
             if os.environ.get("ROMA_GEO_CACHE")
             else os.path.join(ROOT, "roma_match", "cache"))
DEFAULT_OUT = os.path.join(
    os.path.abspath(os.path.expanduser(os.environ["ROMA_GEO_OUTPUT"]))
    if os.environ.get("ROMA_GEO_OUTPUT") else os.path.join(ROOT, "Output"),
    "roma_regions")


# ---------- 方位統計 ----------
def wedge_width(az_deg, frac=0.8):
    """包住 frac 比例點的最窄圓弧寬度(度)。"""
    az = np.sort(np.mod(az_deg, 360.0))
    m = len(az)
    if m == 0:
        return None
    k = max(1, int(np.ceil(frac * m)))
    ext = np.concatenate([az, az + 360.0])
    return float((ext[np.arange(m) + k - 1] - az).min())


def az_coverage(az_deg, bin_deg=10.0):
    """有內點的 bin_deg° bin 占全圓比例。"""
    if len(az_deg) == 0:
        return None
    occ = np.unique((np.mod(az_deg, 360.0) // bin_deg).astype(int))
    return float(len(occ) / (360.0 / bin_deg))


def circ_err(a, b):
    """兩方位角(度)的圓角差絕對值。"""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def px_to_az(x, W):
    return (x / W) * 360.0 - 180.0


# ---------- GT 載入(per house,lazy) ----------
class HouseGT:
    def __init__(self):
        self.labels, self.hs_az, self.sizes, self.room = {}, {}, {}, {}
        self._loaded = set()

    def img_size(self, path):
        if path not in self.sizes:
            from PIL import Image
            with Image.open(path) as im:
                self.sizes[path] = im.size
        return self.sizes[path]

    def _load(self, house, sample_img_path):
        if house in self._loaded:
            return
        self._loaded.add(house)
        d = os.path.dirname(sample_img_path)
        # 連通 label:由 dataset 的 HOTSPOT.json 推導,ToIDName = 直接相鄰清單。
        # 與 stitch_tool/prepare_house_data.load_gt() 同源。
        # ⚠ 必須排序:多樓層房子每層一個 HOTSPOT 檔,樓梯配對在相鄰兩層皆有標註且座標不同;
        # 下方 self.room 與 GT 方位為後寫覆蓋前寫,未排序時採用哪一筆取決於檔案系統的
        # 列舉順序(Linux 與 Windows 不一致)。與 prepare_house_data.load_gt 同源同修。
        from roma_geo.names import glob_in_dir
        _hs_files = sorted(glob_in_dir(d, "*_HOTSPOT.json"))
        for hj in _hs_files:
            try:
                data = json.load(open(hj, encoding="utf-8"))
            except Exception:
                continue
            for ent in data.get("HOTSPOTOFROOM", []):
                owner = os.path.splitext(ent.get("IDName", ""))[0]
                if not owner:
                    continue
                for t in ent.get("ToIDName", {}).get("IDName", []) or []:
                    self.labels[(house, owner, os.path.splitext(t)[0])] = 1
        # 未列在 ToIDName 的 pair = 不相連;由 get() 以「有讀到 HOTSPOT 才視為 0」
        # 處理(完全沒有 HOTSPOT 檔的房子仍回 None,避免把「不知道」誤報成「不連通」)。
        self._has_hs = getattr(self, "_has_hs", set())
        if _hs_files:
            self._has_hs.add(house)
        # GT hotspot 方位(owner pano 的 ERP x → 方位)
        for hj in _hs_files:                      # 已排序,理由見上
            try:
                data = json.load(open(hj, encoding="utf-8"))
            except Exception:
                continue
            for ent in data.get("HOTSPOTOFROOM", []):
                owner = os.path.splitext(ent.get("IDName", ""))[0]
                op = os.path.join(d, owner + ".jpg")
                if not (owner and os.path.exists(op)):
                    continue
                self.room[(house, owner)] = (os.path.basename(hj), ent.get("Roomidx"))
                W = self.img_size(op)[0]
                tgts = ent.get("ToIDName", {}).get("IDName", [])
                coords = ent.get("HSLoc", {}).get("Coordinate", [])
                for t, c in zip(tgts, coords):
                    key = (house, owner, os.path.splitext(t)[0])
                    if key not in self.hs_az:
                        self.hs_az[key] = px_to_az(float(c["x"]), W)

    def get(self, house, na, nb, sample_img_path):
        self._load(house, sample_img_path)
        lab = self.labels.get((house, na, nb), self.labels.get((house, nb, na)))
        if lab is None and house in getattr(self, "_has_hs", ()):
            lab = 0          # 這棟房有 HOTSPOT 但這對不在相鄰清單裡 ⇒ 明確的不連通
        ra, rb = self.room.get((house, na)), self.room.get((house, nb))
        same_room = (ra == rb) if (ra is not None and rb is not None) else None
        return lab, self.hs_az.get((house, na, nb)), self.hs_az.get((house, nb, na)), same_room


# ---------- 共視區域萃取 ----------
REGION_EPS_DEG = 12.0     # DBSCAN 鄰接半徑:兩側「各約 12°」的 6D 弦距(鏈接距離,非區域大小)
REGION_MIN_FRAC = 0.02    # min_samples = max(8, ceil(2% × 內點數)):速斑/殘噪不成區


def _dbscan(X, eps, min_pts):
    """numpy 版 DBSCAN(決定性,無 sklearn 依賴)。回傳 labels,-1 = 噪點。"""
    n = len(X)
    x2 = (X ** 2).sum(1)
    nbrs = []
    for i0 in range(0, n, 512):
        d2 = x2[i0:i0 + 512, None] + x2[None, :] - 2.0 * (X[i0:i0 + 512] @ X.T)
        for r in d2:
            nbrs.append(np.where(r <= eps * eps)[0])
    core = np.array([len(a) >= min_pts for a in nbrs])
    labels = np.full(n, -1)
    cid = 0
    for i in range(n):
        if labels[i] != -1 or not core[i]:
            continue
        labels[i] = cid
        stack = [i]
        while stack:
            j = stack.pop()
            for k in nbrs[j]:
                if labels[k] == -1:
                    labels[k] = cid
                    if core[k]:
                        stack.append(int(k))
        cid += 1
    return labels


def _convex_hull(pts):
    """Andrew 單調鏈凸包,pts (n,2) → 頂點 (m,2)(逆時針)。共線退化時回傳端點。"""
    P = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    P = np.unique(P, axis=0)
    if len(P) <= 2:
        return P
    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], p - h[-2]) <= 0:
                h.pop()
            h.append(p)
        return h[:-1]
    return np.array(half(list(P)) + half(list(P[::-1])))


def _side_geom(px, py, W, H):
    """單側區域幾何:過縫用「最大空缺弧」連續化(x 可能超出 [0,W),繪製時 mod W)。"""
    az = px / W * 360.0 - 180.0
    s = np.sort(az)
    gaps = np.diff(np.concatenate([s, [s[0] + 360.0]]))
    i = int(np.argmax(gaps))
    start = s[(i + 1) % len(s)]                 # 最大空缺弧之後 = 區域起點
    az_c = (az - start) % 360.0 + start         # 連續化方位
    x_c = (az_c + 180.0) / 360.0 * W
    hull = _convex_hull(np.stack([x_c, py], 1))
    area = 0.0
    if len(hull) >= 3:
        x, y = hull[:, 0], hull[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    width = 360.0 - gaps[i]
    return {"polygon": [[round(float(a), 1), round(float(b), 1)] for a, b in hull],
            "area_frac": round(area / (W * H), 4),
            "az_range": [round(float(start), 1), round(float(start + width), 1)],
            "az_width_deg": round(float(width), 1)}


def extract_regions(mi, da_i, db_i, Wa, Ha, Wb, Hb):
    """內點 → 共視區域。6D 聯合聚類(A、B bearing 串接):同一區域必須兩側都連續,
    A 側同牆但 B 側分裂的匹配會被拆開。回傳依成員數降冪的 region list。"""
    n = len(mi)
    if n < 8:
        return []
    eps = np.sqrt(2.0) * 2.0 * np.sin(np.radians(REGION_EPS_DEG) / 2.0)
    labels = _dbscan(np.hstack([da_i, db_i]), eps, max(8, int(np.ceil(REGION_MIN_FRAC * n))))
    out = []
    for cid in range(labels.max() + 1):
        sel = labels == cid
        k = int(sel.sum())
        if k < 8:
            continue
        out.append({"n": k, "frac_of_inliers": round(k / n, 3),
                    "mean_conf": round(float(mi[sel, 4].mean()), 3),
                    "a": _side_geom(mi[sel, 0], mi[sel, 1], Wa, Ha),
                    "b": _side_geom(mi[sel, 2], mi[sel, 3], Wb, Hb)})
    out.sort(key=lambda r: -r["n"])
    return out


# ---------- 單 pair 分析 ----------
def analyze_pair(m_all, Wa, Ha, Wb, Hb, conf_min, max_pts, thresh_deg, iters,
                 seed=0, sampler="stratified", retry_conf=0.3, lat_max=None):
    """m_all = cache matches (K,5) 原圖像素。回傳 result dict(JSON 主體,不含 gt/檔名欄)。
    自適應 conf:先試 conf_min(需 >=30 點);點數不足或內點 <8 → 降 retry_conf(需 >=8 點)
    重試;實際門檻記入 conf_used。兩級皆敗 → no_covisibility,pose=null。
    lat_max(nadir A/B 實驗,2026-07-23):不為 None 時,RANSAC 前先丟掉「任一側 |lat|>lat_max」
    的匹配點(天底=腳架帶/天頂=天花板,系統性假匹配);丟掉數記入 n.lat_drop。"""
    n_raw = len(m_all)
    n_lat_drop = 0
    if lat_max is not None and n_raw:
        lat_a = 90.0 - (m_all[:, 1] / Ha) * 180.0
        lat_b = 90.0 - (m_all[:, 3] / Hb) * 180.0
        keep = (np.abs(lat_a) <= lat_max) & (np.abs(lat_b) <= lat_max)
        n_lat_drop = int((~keep).sum())
        m_all = m_all[keep]
    n_conf065 = int((m_all[:, 4] >= conf_min).sum()) if len(m_all) else 0
    res = {"schema_version": SCHEMA_VERSION,
           "img_size_a": [Wa, Ha], "img_size_b": [Wb, Hb],
           "params": {"conf_min": conf_min, "retry_conf": retry_conf, "max_pts": max_pts,
                      "thresh_deg": thresh_deg, "ransac_iters": iters, "seed": seed, "sampler": sampler,
                      **({"lat_max": lat_max} if lat_max is not None else {})},
           "n": {"raw": n_raw, "conf": n_conf065, "used": 0, "inlier": 0,
                 **({"lat_drop": n_lat_drop} if lat_max is not None else {})},
           "conf_used": None,
           "inlier_rate": None, "pose": None, "quality": None,
           "flags": {"low_matches": n_conf065 < 30, "no_covisibility": False,
                     "degenerate_rot_only": False, "doorway_wedge": False},
           "regions": [], "inliers": []}

    rng = np.random.default_rng(seed)
    thr = np.sin(np.radians(thresh_deg))
    m = da = db = inl = E = None
    for cu, floor in ((conf_min, 30), (retry_conf, 8)):
        mc = m_all[m_all[:, 4] >= cu]
        if len(mc) < floor:
            continue
        if max_pts and len(mc) > max_pts:
            mc = mc[rng.choice(len(mc), max_pts, replace=False)]
        dca = erp_to_bearing(mc[:, :2], Wa, Ha)
        dcb = erp_to_bearing(mc[:, 2:4], Wb, Hb)
        Ec, inlc = ransac_E(dca, dcb, thr, iters=iters, seed=seed, conf=mc[:, 4], sampler=sampler)
        res["n"]["conf"] = int((m_all[:, 4] >= cu).sum())
        res["n"]["used"] = len(mc)
        res["n"]["inlier"] = int(inlc.sum())
        res["inlier_rate"] = round(int(inlc.sum()) / len(mc), 4)
        res["conf_used"] = cu
        if int(inlc.sum()) >= 8:
            m, da, db, E, inl = mc, dca, dcb, Ec, inlc
            break
    if m is None:                       # 兩級都湊不出 >=8 內點
        res["flags"]["no_covisibility"] = True
        res["conf_used"] = None
        return res
    n_in = int(inl.sum())

    R, t = select_pose(E, da, db, inl)
    _, rot_err = rotation_only_fit(da[inl], db[inl])
    rot_only_med = float(np.median(rot_err))
    samp_deg = np.degrees(np.arcsin(np.clip(sampson_residual(E, da[inl], db[inl]), -1, 1)))
    Cb = -R.T @ t                     # A 幀中的 B 相機方向
    az_a = px_to_az(m[inl, 0], Wa)    # 內點方位(A 側)
    az_b = px_to_az(m[inl, 2], Wb)
    wa_, wb_ = wedge_width(az_a), wedge_width(az_b)

    degenerate = rot_only_med < thresh_deg * 1.5
    doorway = (not degenerate and res["inlier_rate"] >= 0.3 and n_in >= 30
               and wa_ is not None and wb_ is not None and wa_ <= 75 and wb_ <= 75)
    res["flags"]["degenerate_rot_only"] = bool(degenerate)
    res["flags"]["doorway_wedge"] = bool(doorway)

    res["pose"] = {"R": np.round(R, 6).tolist(), "t": np.round(t, 6).tolist(),
                   "cam_b_in_a": np.round(Cb, 6).tolist(),
                   "az_ab_deg": round(float(np.degrees(np.arctan2(Cb[0], Cb[2]))), 2),
                   "az_ba_deg": round(float(np.degrees(np.arctan2(t[0], t[2]))), 2),
                   "rot_angle_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))), 2),
                   "baseline_txz": round(float(np.linalg.norm(t[[0, 2]])), 4)}
    res["quality"] = {"sampson_med_deg": round(float(np.median(samp_deg)), 4),
                      "sampson_p90_deg": round(float(np.percentile(samp_deg, 90)), 4),
                      "rot_only_med_deg": round(rot_only_med, 3),
                      "az_wedge_a_deg": round(wa_, 1), "az_wedge_b_deg": round(wb_, 1),
                      "az_cov_a": round(az_coverage(az_a), 3), "az_cov_b": round(az_coverage(az_b), 3)}

    res["regions"] = extract_regions(m[inl], da[inl], db[inl], Wa, Ha, Wb, Hb)

    keep = m[inl]
    if len(keep) > 2000:
        keep = keep[rng.choice(len(keep), 2000, replace=False)]
    res["inliers"] = [[round(float(x), 1), round(float(y), 1), round(float(u), 1),
                       round(float(v), 1), round(float(c), 3)] for x, y, u, v, c in keep]
    return res


# ---------- summary 列 ----------
def _csv_fields_or_raise(fields, expected, path):
    have = list(fields or [])
    extra = [c for c in have if c not in expected]
    missing = [c for c in expected if c not in have]
    if extra or missing:
        from roma_geo.errors import RomaGeoError
        raise RomaGeoError(
            "多了 %s，少了 %s" % (extra or "無", missing or "無"),
            input_name="彙總檔",
            condition="既有檔 %s 的欄位與現行不一致" % path,
        )


def _merge_summary_rows(path, new_rows):
    """以 (house, {img_a, img_b}) 合併既有彙總列。指定配對時累積，不覆寫其他房子。"""
    by = {}
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            _csv_fields_or_raise(reader.fieldnames, SUMMARY_COLS, path)
            for row in reader:
                key = (row["house"], frozenset({row["img_a"], row["img_b"]}))
                by[key] = row
    for row in new_rows:
        key = (row["house"], frozenset({row["img_a"], row["img_b"]}))
        by[key] = row
    return [by[k] for k in sorted(by, key=lambda k: (k[0], min(k[1]), max(k[1])))]


def _atomic_write_csv(path, fieldnames, rows):
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="._summary_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="raise")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


SUMMARY_COLS = ["house", "img_a", "img_b", "n_raw", "n_conf", "n_used", "n_inlier", "inlier_rate",
                "conf_used",
                "az_ab_deg", "az_ba_deg", "rot_angle_deg", "baseline_txz",
                "sampson_med_deg", "sampson_p90_deg", "rot_only_med_deg",
                "az_wedge_a_deg", "az_wedge_b_deg", "az_cov_a", "az_cov_b",
                "flag_low_matches", "flag_no_covisibility", "flag_degenerate_rot_only", "flag_doorway_wedge",
                "n_regions", "region1_frac", "region1_az_width_a",
                "gt_label", "same_room", "gt_az_ab", "gt_az_ba", "err_az_ab", "err_az_ba", "time_ms"]


def to_summary_row(house, na, nb, res, gt_label, gt_ab, gt_ba, same_room, dt_ms):
    p, q, f = res.get("pose") or {}, res.get("quality") or {}, res["flags"]
    az_ab, az_ba = p.get("az_ab_deg"), p.get("az_ba_deg")
    # 舊 schema(wp1-step2-v1)JSON 相容:有位姿 = conf 0.65;無 conf_used/no_covisibility 欄
    conf_used = res.get("conf_used", 0.65 if res.get("pose") else None)
    row = {"house": house, "img_a": na, "img_b": nb,
           "n_raw": res["n"]["raw"], "n_conf": res["n"]["conf"], "n_used": res["n"]["used"],
           "n_inlier": res["n"]["inlier"], "inlier_rate": res["inlier_rate"],
           "conf_used": conf_used,
           "az_ab_deg": az_ab, "az_ba_deg": az_ba,
           "rot_angle_deg": p.get("rot_angle_deg"), "baseline_txz": p.get("baseline_txz"),
           "sampson_med_deg": q.get("sampson_med_deg"), "sampson_p90_deg": q.get("sampson_p90_deg"),
           "rot_only_med_deg": q.get("rot_only_med_deg"),
           "az_wedge_a_deg": q.get("az_wedge_a_deg"), "az_wedge_b_deg": q.get("az_wedge_b_deg"),
           "az_cov_a": q.get("az_cov_a"), "az_cov_b": q.get("az_cov_b"),
           "flag_low_matches": int(f["low_matches"]), "flag_no_covisibility": int(f.get("no_covisibility", False)),
           "flag_degenerate_rot_only": int(f["degenerate_rot_only"]),
           "flag_doorway_wedge": int(f["doorway_wedge"]),
           "n_regions": len(res.get("regions") or []),
           "region1_frac": (res.get("regions") or [{}])[0].get("frac_of_inliers"),
           "region1_az_width_a": ((res.get("regions") or [{}])[0].get("a") or {}).get("az_width_deg"),
           "gt_label": gt_label, "same_room": None if same_room is None else int(same_room),
           "gt_az_ab": None if gt_ab is None else round(gt_ab, 2),
           "gt_az_ba": None if gt_ba is None else round(gt_ba, 2),
           "err_az_ab": None, "err_az_ba": None, "time_ms": round(dt_ms, 1)}
    if az_ab is not None and gt_ab is not None:
        row["err_az_ab"] = round(circ_err(az_ab, gt_ab), 2)
    if az_ba is not None and gt_ba is not None:
        row["err_az_ba"] = round(circ_err(az_ba, gt_ba), 2)
    return row


# ---------- 報告 ----------
def q(vals, p):
    return round(float(np.percentile(vals, p)), 3) if len(vals) else None


def write_report(rows, total_s, args, out_path):
    r_all = [r for r in rows if r["inlier_rate"] is not None]
    by_lab = {lab: [r for r in r_all if r["gt_label"] == lab] for lab in (1, 0)}          # 有跑 RANSAC 的
    by_lab_all = {lab: [r for r in rows if r["gt_label"] == lab] for lab in (1, 0)}      # 全部(含 low_matches)
    # 外來資料通常不含 *_HOTSPOT.json ⇒ gt_label / same_room / gt_az 全為 None。
    # 此時整段略過 GT 相關小節,避免輸出大量 None 而被誤判為計算失敗。
    has_gt = any(r["gt_label"] is not None for r in rows)
    has_gt_az = any(r["err_az_ab"] is not None for r in rows)
    NO_GT = "*(此資料集無 GT 標註,略過。GT 來自 datasets 內的 `*_HOTSPOT.json`。)*"
    compute_s = sum(r["time_ms"] for r in rows) / 1000.0    # 累計計算時間(resume 重生報告也不失真)
    lines = ["# +3 全量批次報告(自動生成)", "",
             f"日期:{time.strftime('%Y-%m-%d %H:%M')} | 參數:conf>={args.conf}, max_pts={args.max_pts}, "
             f"thresh={args.thresh_deg}°, iters={args.iters}, sampler={args.sampler}"
             + (f", lat_filter=±{args.lat_filter}°" if getattr(args, 'lat_filter', 0) and args.lat_filter > 0 else ""), "",
             f"**{len(rows)} pair 跑完,累計計算 {compute_s/60:.1f} 分鐘"
             f"(平均 {1000*compute_s/max(len(rows),1):.0f} ms/pair,含 JSON 寫出;本機 CPU 向量化 RANSAC,無 GPU)**", "",
             "## inlier 率分佈(gt_label 由 datasets 的 `*_HOTSPOT.json` 推導;"
             "有跑 RANSAC 的 pair,按 conf 層拆開)", "",
             "| 組 | pair 數 | inlier率 p25 | 中位 | p75 |",
             "|---|---|---|---|---|"]
    tiers = [("conf0.65 級", lambda r: r["conf_used"] == 0.65),
             ("conf0.3 撈回級", lambda r: r["conf_used"] == args.retry_conf)]
    if has_gt:
        for lab_name, lab in (("label=1", 1), ("label=0", 0)):
            for t_name, t_sel in tiers:
                grp = [r for r in by_lab[lab] if t_sel(r)]
                ir = [r["inlier_rate"] for r in grp]
                lines.append(f"| {lab_name}・{t_name} | {len(grp)} | {q(ir,25)} | {q(ir,50)} | {q(ir,75)} |")
    else:
        for t_name, t_sel in tiers:
            grp = [r for r in r_all if t_sel(r)]
            ir = [r["inlier_rate"] for r in grp]
            lines.append(f"| {t_name}(無 GT,不分 label) | {len(grp)} | {q(ir,25)} | {q(ir,50)} | {q(ir,75)} |")
    ir = [r["inlier_rate"] for r in r_all]
    lines.append(f"| 全部(含無標註,兩層混合) | {len(r_all)} | {q(ir,25)} | {q(ir,50)} | {q(ir,75)} |")
    lines += ["", f"## 旗標統計(全 {len(rows)} pair;label 欄含 low_matches pair)", ""]
    if has_gt:
        lines += [f"| 旗標 | 全部 | label=1 (n={len(by_lab_all[1])}) | label=0 (n={len(by_lab_all[0])}) |",
                  "|---|---|---|---|"]
        for fl in ("flag_low_matches", "flag_no_covisibility", "flag_degenerate_rot_only", "flag_doorway_wedge"):
            cnt = lambda g: sum(r[fl] for r in g)
            lines.append(f"| {fl} | {cnt(rows)} | {cnt(by_lab_all[1])} | {cnt(by_lab_all[0])} |")
    else:
        lines += ["| 旗標 | 全部 |", "|---|---|"]
        for fl in ("flag_low_matches", "flag_no_covisibility", "flag_degenerate_rot_only", "flag_doorway_wedge"):
            lines.append(f"| {fl} | {sum(r[fl] for r in rows)} |")
    n_p65 = sum(1 for r in rows if r["conf_used"] == 0.65 and r["az_ab_deg"] is not None)
    n_p30 = sum(1 for r in rows if r["conf_used"] == args.retry_conf and r["az_ab_deg"] is not None)
    n_noco = sum(r["flag_no_covisibility"] for r in rows)
    rec1 = sum(1 for r in rows if r["conf_used"] == args.retry_conf and r["az_ab_deg"] is not None and r["gt_label"] == 1)
    noco1 = sum(1 for r in rows if r["flag_no_covisibility"] and r["gt_label"] == 1)
    err_t65 = [r["err_az_ab"] for r in rows if r["conf_used"] == 0.65
               and r["err_az_ab"] is not None and not r["flag_degenerate_rot_only"]]
    err_t30 = [r["err_az_ab"] for r in rows if r["conf_used"] == args.retry_conf
               and r["err_az_ab"] is not None and not r["flag_degenerate_rot_only"]]
    n = max(len(rows), 1)
    lines += ["", "## 自適應 conf(0.65 → " + str(args.retry_conf) + " 重試)", "",
              f"- 位姿來自 conf 0.65:{n_p65} 對;**降級 {args.retry_conf} 撈回:{n_p30} 對**"
              + (f"(其中 GT 連通 {rec1} 對)" if has_gt else "")
              + f";no_covisibility:{n_noco} 對" + (f"(GT 連通僅 {noco1})" if has_gt else ""),
              # 「升級前」= 僅有 conf 0.65 一級時的覆蓋率,須由本批次實算,
              # 不可寫死成某次執行的數字(換資料集即不成立)。
              f"- 位姿覆蓋率:{(n_p65+n_p30)}/{len(rows)} = {100*(n_p65+n_p30)/n:.1f}%"
              f"(升級前,即僅 conf {args.conf} 級:{n_p65}/{len(rows)} = {100*n_p65/n:.1f}%)"]
    if has_gt_az:
        lines.append(
            f"- **分層品質(az 誤差 vs GT,非退化)**:0.65 級中位 {q(err_t65,50)}°/p90 {q(err_t65,90)}°"
            f"(n={len(err_t65)});{args.retry_conf} 撈回級中位 {q(err_t30,50)}°/p90 {q(err_t30,90)}°"
            f"(n={len(err_t30)})。撈回級重尾,下游必須以 conf_used 分層 gate")
    else:
        lines.append("- 分層品質(az 誤差 vs GT):" + NO_GT
                     + " 撈回級品質重尾是既有結論,下游仍必須以 `conf_used` 分層 gate。")
    ok = [r for r in r_all if r["err_az_ab"] is not None and not r["flag_degenerate_rot_only"]]
    err_same = [r["err_az_ab"] for r in ok if r["same_room"] == 1]
    err_cross = [r["err_az_ab"] for r in ok if r["same_room"] == 0]
    err_unk = [r["err_az_ab"] for r in ok if r["same_room"] is None]
    err_deg = [r["err_az_ab"] for r in r_all if r["err_az_ab"] is not None and r["flag_degenerate_rot_only"]]
    lines += ["", "## 位姿方位 vs GT hotspot 方位(直接相鄰 pair 才有 GT;A→B 方向)", ""]
    if has_gt_az:
        lines += ["GT hotspot 標的是「**通道位置**」,只有同房 pair 才近似指向對方相機;但直接相鄰",
                  "pair 的 B 相機通常就在通道正後方,兩種語意的方位高度重合,故此表當**一致性檢核**",
                  "看,不當嚴格位姿精度。", "",
                  "| 組 | n | 圓角誤差中位 | p90 |", "|---|---|---|---|",
                  f"| 同房(hotspot≈對方相機方向) | {len(err_same)} | {q(err_same,50)}° | {q(err_same,90)}° |",
                  f"| 跨房(hotspot=通道方位) | {len(err_cross)} | {q(err_cross,50)}° | {q(err_cross,90)}° |"]
        if err_unk:
            lines.append(f"| 房間歸屬不明 | {len(err_unk)} | {q(err_unk,50)}° | {q(err_unk,90)}° |")
        lines += [f"| 退化(近零基線)對照 | {len(err_deg)} | {q(err_deg,50)}° | — |", "",
                  "退化組 t 方向本就不可靠,誤差≈隨機,恰證明退化旗標有效。"]
    else:
        lines.append(NO_GT)
    lines += ["", "## 楔形寬度(A 側,非退化、有位姿的 pair)", "",
              "| 組 | 中位楔形寬° | 中位方位覆蓋度 |", "|---|---|---|"]
    wedge_groups = ([("label=1", by_lab[1]), ("label=0", by_lab[0])] if has_gt
                    else [("全部(無 GT,不分 label)", r_all)])
    for name, grp in wedge_groups:
        g = [r for r in grp if r["az_wedge_a_deg"] is not None and not r["flag_degenerate_rot_only"]]
        w = [r["az_wedge_a_deg"] for r in g]; c = [r["az_cov_a"] for r in g]
        lines.append(f"| {name} | {q(w,50)} | {q(c,50)} |")
    posed = [r for r in r_all if r["az_ab_deg"] is not None]
    door = [r for r in posed if r["flag_doorway_wedge"]]
    sr = [r for r in posed if r["same_room"] == 1]
    cr = [r for r in posed if r["same_room"] == 0]
    lines += ["", "## 共視區域(6D 聯合 DBSCAN;預期:門 pair 單一門洞區、同房多區/大覆蓋)", "",
              "| 組 | n | 區域數中位 | =1區占比 | 主區內點占比中位 | 主區A側方位寬中位 |",
              "|---|---|---|---|---|---|"]
    for name, grp in [("doorway 旗標 pair", door), ("同房直接相鄰", sr), ("跨房直接相鄰", cr), ("全部有位姿", posed)]:
        if not grp:
            continue
        nr = [r["n_regions"] for r in grp]
        one = 100 * sum(1 for r in grp if r["n_regions"] == 1) / len(grp)
        fr = [r["region1_frac"] for r in grp if r["region1_frac"] is not None]
        wd = [r["region1_az_width_a"] for r in grp if r["region1_az_width_a"] is not None]
        lines.append(f"| {name} | {len(grp)} | {q(nr,50)} | {one:.0f}% | {q(fr,50)} | {q(wd,50)}° |")
    # 產出位於 repo 內時輸出相對路徑;經 ROMA_GEO_OUTPUT 導向外部時輸出絕對路徑。
    sm = os.path.join(args.out, "_summary.csv")
    try:
        rel = os.path.relpath(sm, ROOT)
        if not rel.startswith(".."):
            sm = rel.replace(os.sep, "/")
    except ValueError:                      # 跨磁碟機時 relpath 會拋出例外
        pass
    lines += ["", "指標定義與旗標閾值見 `pose_regions/README.md`;逐 pair 明細見 "
              f"`{sm}`(JSON 同資料夾)。"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------- 主流程 ----------
def run_pose_batch(cache_dir, out_dir, datasets_root=None, pairs=None, limit=0, resume=False,
                   conf=0.65, retry_conf=0.3, max_pts=4000, thresh_deg=0.5, iters=2000,
                   sampler="stratified", lat_filter=60.0, report=None, on_missing="skip"):
    """對匹配快取做幾何驗證，寫出逐對 JSON 與 _summary.csv。不切換工作目錄。

    pairs 為 [(house, img_a, img_b), ...] 時只處理這些配對；為 None 時掃描 cache_dir。
    on_missing: skip 略過缺圖；error 則丟出 RomaGeoError。
    回傳寫入 summary 的列數。
    """
    from types import SimpleNamespace
    jobs = []  # (house, na, nb, npz_path, swapped)
    skipped = []
    if pairs is not None:
        from roma_geo.names import require_pair_names
        for house, na, nb in pairs:
            require_pair_names(house, na, nb, input_name="房子 %s" % house)
            p = os.path.join(cache_dir, "%s_%s_%s%s" % (house, na, nb, CACHE_SUFFIX))
            pr = os.path.join(cache_dir, "%s_%s_%s%s" % (house, nb, na, CACHE_SUFFIX))
            if os.path.isfile(p):
                jobs.append((house, na, nb, p, False))
            elif os.path.isfile(pr):
                jobs.append((house, na, nb, pr, True))
            else:
                from roma_geo.errors import RomaGeoError
                raise RomaGeoError(
                    "檔案不存在：%s" % p,
                    input_name="匹配快取",
                    condition="尚未寫入匹配快取。若匹配步驟曾回報失敗，原因在該步的 failures"
                    "（配對 %s / %s / %s；目錄 %s）"
                    % (house, na, nb, cache_dir),
                )
    else:
        from roma_geo.names import glob_in_dir
        all_npz = sorted(glob_in_dir(cache_dir, "*.npz"))
        files = [f for f in all_npz if f.endswith(CACHE_SUFFIX)]
        n_other = len(all_npz) - len(files)
        if n_other:
            print(f"[info] 快取目錄有 {n_other} 個非 {CACHE_SUFFIX} 的檔案,已略過。")
        for f in files:
            ident = identify_cache_file(f, datasets_root)
            if ident:
                jobs.append((ident[0], ident[1], ident[2], f, False))
            else:
                skipped.append((os.path.basename(f), "無法確定房子與影像名"))
        if skipped:
            print("[skip] 無法由檔名回推房子與影像名：")
            for name, why in skipped:
                print("  %s：%s" % (name, why))
    if limit:
        jobs = jobs[:limit]
    os.makedirs(out_dir, exist_ok=True)
    if not jobs and skipped:
        print("[skip] 本批 %d 個快取檔全部略過。" % len(skipped))
    gt = HouseGT()
    rows, t_start = [], time.time()
    try:
        from tqdm import tqdm
        seq = tqdm(jobs, desc="batch pose") if sys.stderr.isatty() else jobs
    except Exception:
        seq = jobs
    missing_imgs = []
    for house, na, nb, f, swapped in seq:
        jpath = os.path.join(out_dir, f"{house}_{na}_{nb}.json")
        if resume and os.path.exists(jpath):
            old = json.load(open(jpath, encoding="utf-8"))
            if (old.get("schema_version") == SCHEMA_VERSION
                    and (old.get("pose") is not None or old.get("flags", {}).get("no_covisibility"))):
                og = old.get("gt", {})
                rows.append(to_summary_row(house, na, nb, old, og.get("label"),
                                           og.get("az_ab_deg"), og.get("az_ba_deg"),
                                           og.get("same_room"), old.get("time_ms", 0)))
                continue
        t0 = time.time()
        img_a = find_house_image(house, na, datasets_root=datasets_root)
        img_b = find_house_image(house, nb, datasets_root=datasets_root)
        if not (img_a and img_b):
            if on_missing == "error":
                from roma_geo.errors import RomaGeoError
                raise RomaGeoError(
                    "找不到 %s.jpg / %s.jpg" % (na, nb),
                    input_name="全景影像",
                    condition="房子 %s 在 %s 底下沒有這兩張圖" % (house, datasets_root or "datasets"),
                )
            missing_imgs.append("%s %s/%s" % (house, na, nb))
            print(f"[skip] 找不到圖 {house} {na}/{nb}")
            continue
        Wa, Ha = gt.img_size(img_a)
        Wb, Hb = gt.img_size(img_b)
        m = np.load(f)["matches"]
        m = m.astype(float) if m.size else np.zeros((0, 5))
        if swapped and m.size:
            m = m[:, [2, 3, 0, 1, 4]]
        res = analyze_pair(m, Wa, Ha, Wb, Hb, conf, max_pts, thresh_deg, iters,
                           sampler=sampler, retry_conf=retry_conf,
                           lat_max=(lat_filter if lat_filter and lat_filter > 0 else None))
        lab, gt_ab, gt_ba, same_room = gt.get(house, na, nb, img_a)
        dt_ms = (time.time() - t0) * 1000
        res.update({"house": house, "img_a": na, "img_b": nb,
                    "gt": {"label": lab,
                           "same_room": None if same_room is None else bool(same_room),
                           "az_ab_deg": None if gt_ab is None else round(gt_ab, 2),
                           "az_ba_deg": None if gt_ba is None else round(gt_ba, 2)},
                    "time_ms": round(dt_ms, 1)})
        with open(jpath, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, separators=(",", ":"))
        rows.append(to_summary_row(house, na, nb, res, lab, gt_ab, gt_ba, same_room, dt_ms))

    if missing_imgs and not rows:
        print("[skip] 本批因找不到影像而全部略過：")
        for item in missing_imgs:
            print("  %s" % item)

    total_s = time.time() - t_start
    summary_path = os.path.join(out_dir, "_summary.csv")
    summary_rows = rows
    if pairs is not None and os.path.isfile(summary_path):
        summary_rows = _merge_summary_rows(summary_path, rows)
    _atomic_write_csv(summary_path, SUMMARY_COLS, summary_rows)
    args_ns = SimpleNamespace(conf=conf, max_pts=max_pts, thresh_deg=thresh_deg, iters=iters,
                              sampler=sampler, lat_filter=lat_filter, retry_conf=retry_conf,
                              out=out_dir)
    if report is None:
        report = os.path.join(out_dir, "batch_pose_report.md")
    write_report(rows, total_s, args_ns, report)
    print(f"\n{len(rows)} pairs | {total_s/60:.1f} min | summary -> {out_dir}/_summary.csv | "
          f"report -> {report}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.65)
    ap.add_argument("--retry_conf", type=float, default=0.3,
                    help="自適應重試門檻:conf 級點數<30 或內點<8 時降到此值重試")
    ap.add_argument("--max_pts", type=int, default=4000)
    ap.add_argument("--thresh_deg", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--sampler", default="stratified", choices=["stratified", "uniform"],
                    help="RANSAC 取樣:stratified=conf 漸進池+方位分桶(預設);uniform=均勻(對照)")
    ap.add_argument("--lat_filter", type=float, default=60.0,
                    help="RANSAC 前丟掉任一側 |lat|>此值(度)的匹配點(天底腳架/天頂帶假匹配;"
                         "2026-07-23 A/B 拍板採用,預設 60;設 0 關閉=舊行為,見 nadir_ab_report.md)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="產出目錄(預設 <ROMA_GEO_OUTPUT|Output>/roma_regions)")
    ap.add_argument("--report", default=None,
                    help="報告輸出路徑(預設 <產出目錄>/batch_pose_report.md)")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 對(冒煙用)")
    ap.add_argument("--resume", action="store_true", help="跳過已有 JSON 的 pair")
    a = ap.parse_args()
    # 報告跟著產出目錄走,不寫死在本檔旁邊:換 --out(隔離執行、只跑部分房子、
    # 在別台機器跑)時,報告會跟該批產出放在一起,不會覆蓋掉 repo 內那份與
    # _summary.csv 同批的全量報告。
    if a.report is None:
        a.report = os.path.join(a.out, "batch_pose_report.md")

    run_pose_batch(CACHE_DIR, a.out, datasets_root=None, pairs=None, limit=a.limit,
                   resume=a.resume, conf=a.conf, retry_conf=a.retry_conf, max_pts=a.max_pts,
                   thresh_deg=a.thresh_deg, iters=a.iters, sampler=a.sampler,
                   lat_filter=a.lat_filter, report=a.report, on_missing="skip")


if __name__ == "__main__":
    main()
