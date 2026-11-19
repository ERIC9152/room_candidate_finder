"""語意門抽取與通道方位矯正。矯正只改輸出一，不改相對位姿。"""
import json
import os

from .names import glob_in_dir, require_path_component

CONTAIN_FRAC = 0.8
MERGE_GAP_DEG = 1.0


def _arc_segments(a0, a1):
    """把 [a0, a1] 正向弧拆成 [0,360) 上的線性段。空區間回空列表，整圈回一段 360°。"""
    raw = float(a1) - float(a0)
    start = float(a0) % 360.0
    width = raw % 360.0
    if width == 0.0:
        if abs(raw) < 1e-12:
            return []
        return [(0.0, 360.0)]
    end = start + width
    if end <= 360.0 + 1e-12:
        return [(start, min(end, 360.0))]
    return [(start, 360.0), (0.0, end - 360.0)]


def az_overlap(a0, a1, b0, b1):
    """兩個可過縫角區間的重疊角度。區間以 [a0, a1] 表示，沿正向走到 a1。

    評估腳本與標註工具前端引用同一規則。兩段都跨縫且合計超過一整圈時加總兩段，
    否則與只取單段最大的舊算法相同。
    """
    total = 0.0
    for s1, e1 in _arc_segments(a0, a1):
        for s2, e2 in _arc_segments(b0, b1):
            lo = max(s1, s2)
            hi = min(e1, e2)
            if hi > lo:
                total += hi - lo
    return total


def circ_err(a, b):
    """兩方位的最短圓角距離（度，0~180）。有號與 0~360 皆可。"""
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def az_contains(az, a0, a1):
    """點是否落在 [a0, a1] 正向弧上（可過縫）。空區間為 False，整圈為 True。"""
    segs = _arc_segments(a0, a1)
    if not segs:
        return False
    x = float(az) % 360.0
    for start, end in segs:
        if start - 1e-9 <= x <= end + 1e-9:
            return True
        if start - 1e-9 <= x + 360.0 <= end + 1e-9:
            return True
    return False


def region_az_span(side):
    """從共視區域一側取出方位區間 (az0, az1, width_deg)。沒有 az_range 時回 None。"""
    if not side:
        return None
    ar = side.get("az_range")
    if not ar or len(ar) < 2:
        return None
    a0, a1 = float(ar[0]), float(ar[1])
    w = (a1 - a0) % 360.0
    if w <= 0:
        w = abs(a1 - a0)
    return a0, a1, w


def choose_door_correction(az, side_regions, doors, contain=CONTAIN_FRAC):
    """取與共視區域重疊度最高的門；共視區域有 contain 以上落在該門內時改用門中心。

    回傳 (方位, 是否矯正)。
    """
    if not doors or not side_regions:
        return az, False
    best = None
    for side in side_regions:
        span = region_az_span(side)
        if span is None:
            continue
        r0, r1, rw = span
        if rw <= 0:
            continue
        for d in doors:
            ov = az_overlap(r0, r1, d["az0"], d["az1"])
            frac = ov / rw
            if best is None or frac > best[0]:
                best = (frac, d)
    if best is None or best[0] < contain:
        return az, False
    return float(best[1]["az"]) % 360.0, True


def invert_pose(R, t, ndigits=6):
    """x_b = R x_a + t 的逆：x_a = Rᵀ x_b - Rᵀ t。t 保持單位長。數值取 ndigits 位小數。"""
    import numpy as np
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)
    Ri = R.T
    ti = -R.T @ t
    n = float(np.linalg.norm(ti))
    if n:
        ti = ti / n
    return [[round(float(x), ndigits) for x in row] for row in Ri.tolist()], \
        [round(float(x), ndigits) for x in ti.tolist()]


def azimuth_from_pose(R, t):
    """A 幀中 B 相機方位（度，0~360）。cam_b_in_a = -Rᵀ t。"""
    import math
    import numpy as np
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)
    cb = -R.T @ t
    return math.degrees(math.atan2(cb[0], cb[2])) % 360.0


def _house_tid(house):
    return "".join(ch for ch in house if ch.isdigit())


def _load_extract_doors():
    """取得門抽取函式。不把專案根目錄插進模組搜尋路徑。"""
    try:
        from semantic_doors.semantic_door_eval import extract_doors
        return extract_doors
    except ImportError:
        pass
    from .config import PROJECT_ROOT
    from .errors import RomaGeoError
    import importlib.util
    path = os.path.join(PROJECT_ROOT, "semantic_doors", "semantic_door_eval.py")
    if not os.path.isfile(path):
        raise RomaGeoError(
            "找不到語意門抽取實作",
            input_name="semantic_maps",
            condition="目前環境沒有 semantic_doors 模組",
        )
    spec = importlib.util.spec_from_file_location("_roma_geo_semantic_door_eval", path)
    if spec is None or spec.loader is None:
        raise RomaGeoError(
            "無法載入語意門抽取實作",
            input_name="semantic_maps",
            condition="semantic_door_eval.py 無法以檔案方式載入",
        )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.extract_doors


def extract_house_doors(semantic_root, house, merge_gap_deg=MERGE_GAP_DEG):
    """對一間房的語意圖跑門抽取。回傳 {image_stem: [door, ...]}。"""
    require_path_component(house, input_name="房子 %s" % house)
    extract_doors = _load_extract_doors()
    tid = _house_tid(house)
    dirs = [os.path.join(semantic_root, house)]
    if tid:
        dirs.append(os.path.join(semantic_root, tid))
    doors = {}
    seen = set()
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for mp in sorted(glob_in_dir(d, "*_mask.png")):
            img = os.path.basename(mp)[:-len("_mask.png")]
            if img in seen:
                continue
            seen.add(img)
            doors[img] = extract_doors(mp, merge_gap_deg)
    return doors


def apply_semantic_correction(pub, doors_by_image):
    """就地矯正 pub['pairs'] 的 az_ij / az_ji。回傳矯正對數。

    只改通道方位。R / t 不動。
    """
    names = pub.get("images") or []
    n = 0
    for p in pub.get("pairs") or []:
        p["semantic_corrected"] = False
        if p.get("az_source") != "geo":
            continue
        regions = p.get("regions") or []
        if not regions:
            continue
        try:
            ni, nj = names[p["i"]], names[p["j"]]
        except (IndexError, KeyError, TypeError):
            continue
        si = os.path.splitext(ni)[0]
        sj = os.path.splitext(nj)[0]
        sides_a = [r.get("a") or {} for r in regions]
        sides_b = [r.get("b") or {} for r in regions]
        corr_i = corr_j = False
        if "az_ij" in p and si in doors_by_image:
            new_az, hit = choose_door_correction(p["az_ij"], sides_a, doors_by_image[si])
            if hit:
                p["az_ij"] = round(new_az, 2) % 360.0
                corr_i = True
        if "az_ji" in p and sj in doors_by_image:
            new_az, hit = choose_door_correction(p["az_ji"], sides_b, doors_by_image[sj])
            if hit:
                p["az_ji"] = round(new_az, 2) % 360.0
                corr_j = True
        if corr_i or corr_j:
            p["semantic_corrected"] = True
            n += 1
    return n


def load_doors_json(path):
    if not path or not os.path.isfile(path):
        return {}
    data = json.load(open(path, encoding="utf-8"))
    return data.get("doors") or {}
