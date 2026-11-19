"""
[模組用途]
測試共用的輔助工具：合成資料產生器、環境變數暫時覆寫、臨時目錄。

本模組匯入時會把專案根目錄補進 `sys.path`，因此每個測試檔的第一個匯入都應該是
本模組，之後才匯入受測模組。

合成資料的設計原則：期望值由**規約或獨立算法**推得，不從受測程式碼反推。
`synth_matches()` 以已知的 R、t 正向投影出兩側 ERP 像素，受測程式碼若解回同一組
R、t 才算通過；若只是複製實作的算式，測試將只能鎖住現況而無法發現錯誤。
"""
import contextlib
import io
import math
import os
import shutil
import sys
import tempfile

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if ROOT not in sys.path:
    sys.path.append(ROOT)

# 專案全域採用 `json.load(open(...))` 的寫法（三十餘處），unittest 會為此發出
# ResourceWarning。CPython 在參考計數歸零時即關檔，實際行為無誤；若不過濾，每次執行
# 都會出現數十行與測試結論無關的警告，久了會使人忽略輸出。故在此具名關閉，
# 其他類別的警告一律保留。
import warnings  # noqa: E402
warnings.filterwarnings("ignore", category=ResourceWarning)

#: 資料集常見的 ERP 尺寸，供合成資料使用（實際資料 per-house 不同，見 upstream_direction）。
W_DEF, H_DEF = 8192, 4096


# ---------- 幾何：ERP 與球面射線 ----------
def bearing_to_erp(d, W=W_DEF, H=H_DEF):
    """單位方向向量 → ERP 像素座標。

    為 `pose_regions.erp_to_bearing` 的解析反函式，由規約直接推得：
        lon = (x / W) · 2π − π       lat = π/2 − (y / H) · π
    """
    d = np.asarray(d, dtype=float)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    lat = np.arcsin(np.clip(d[:, 1], -1.0, 1.0))
    lon = np.arctan2(d[:, 0], d[:, 2])
    x = (lon + np.pi) / (2.0 * np.pi) * W
    y = (np.pi / 2.0 - lat) / np.pi * H
    return np.stack([x, y], axis=1)


def rot_matrix(axis, deg):
    """羅德里格斯公式：繞 `axis` 轉 `deg` 度的旋轉矩陣。"""
    ax = np.asarray(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    a = math.radians(deg)
    K = np.array([[0.0, -ax[2], ax[1]],
                  [ax[2], 0.0, -ax[0]],
                  [-ax[1], ax[0], 0.0]])
    return np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * (K @ K)


def az_ab_of(R, t):
    """由已知的 R、t 推得 A 幀中 B 相機的方位角（度）。

    定義與 `batch_pose_from_cache.analyze_pair` 相同：`cam_b_in_a = −Rᵀt`，
    方位取 `atan2(x, z)`。此處由定義獨立寫出，作為受測結果的對照。
    """
    Cb = -np.asarray(R, dtype=float).T @ np.asarray(t, dtype=float)
    return math.degrees(math.atan2(Cb[0], Cb[2]))


def circ_diff(a, b):
    """兩方位角（度）的圓角差絕對值。"""
    return abs((a - b + 180.0) % 360.0 - 180.0)


# ---------- 合成匹配 ----------
def synth_matches(n=400, R=None, t=None, W=W_DEF, H=H_DEF, seed=0,
                  outlier_frac=0.0, conf_in=(0.7, 1.0), conf_out=(0.4, 0.8),
                  center=(0.0, 0.0, 10.0), spread=(5.0, 2.0, 3.0),
                  noise_px=0.0):
    """產生一組已知位姿的合成匹配，格式同 RoMa 快取：(K,5) = [x_a, y_a, x_b, y_b, conf]。

    正向流程：在 A 相機座標系取 3D 點 → 兩側各自投影成 ERP 像素。
    B 幀的點為 `X_b = R · X_a + t`，故 `−Rᵀt` 即 A 幀中的 B 相機位置。

    Args:
        n: 匹配點數。
        R, t: 真值位姿；預設為繞 Y 軸 15° 與 x 方向的基線。
        outlier_frac: 外點比例；外點的 B 側方向改為隨機，模擬誤匹配。
        conf_in / conf_out: 內點與外點的 confidence 取值範圍（仿 RoMa 的重疊分佈）。
        center, spread: 3D 點雲的中心與各軸標準差（相機座標系，Z 為前方）。
        noise_px: B 側像素雜訊的標準差。

    Returns:
        (matches, R, t)：matches 前 `n_out` 列為外點，其餘為內點。
    """
    rng = np.random.default_rng(seed)
    R = rot_matrix([0.0, 1.0, 0.0], 15.0) if R is None else np.asarray(R, dtype=float)
    t = np.array([1.0, 0.0, 0.0]) if t is None else np.asarray(t, dtype=float)

    X = rng.normal(size=(n, 3)) * np.asarray(spread, dtype=float) + np.asarray(center, dtype=float)
    X[:, 2] = np.abs(X[:, 2]) + 1.0                      # 保證在相機前方
    da = X / np.linalg.norm(X, axis=1, keepdims=True)
    Xb = (R @ X.T).T + t
    db = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)

    n_out = int(round(outlier_frac * n))
    if n_out:
        rnd = rng.normal(size=(n_out, 3))
        db[:n_out] = rnd / np.linalg.norm(rnd, axis=1, keepdims=True)

    pa = bearing_to_erp(da, W, H)
    pb = bearing_to_erp(db, W, H)
    if noise_px:
        pb = pb + rng.normal(scale=noise_px, size=pb.shape)
        pb[:, 0] = np.mod(pb[:, 0], W)
        pb[:, 1] = np.clip(pb[:, 1], 0.0, H - 1e-6)

    conf = np.empty(n, dtype=float)
    conf[:n_out] = rng.uniform(conf_out[0], conf_out[1], n_out)
    conf[n_out:] = rng.uniform(conf_in[0], conf_in[1], n - n_out)

    return np.column_stack([pa, pb, conf]), R, t


def cone_matches(n=200, half_angle_deg=15.0, R=None, t=None, W=W_DEF, H=H_DEF,
                 seed=0, conf=(0.8, 1.0), depth_range=(3.0, 20.0)):
    """產生方位集中於窄錐的合成匹配，用於楔形（門洞）旗標的測試。

    3D 點聚在單一方向的窄錐內，兩側的方位分佈因此都很窄。深度須有足夠變化，
    否則窄錐加淺深度可由純旋轉解釋，會被判為退化。
    """
    rng = np.random.default_rng(seed)
    R = rot_matrix([0.0, 1.0, 0.0], 10.0) if R is None else np.asarray(R, dtype=float)
    t = np.array([0.6, 0.0, 0.0]) if t is None else np.asarray(t, dtype=float)

    ha = math.radians(half_angle_deg)
    az = rng.uniform(-ha, ha, n)
    el = rng.uniform(-ha, ha, n)
    d = np.stack([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)], axis=1)
    X = d * rng.uniform(depth_range[0], depth_range[1], (n, 1))

    da = X / np.linalg.norm(X, axis=1, keepdims=True)
    Xb = (R @ X.T).T + t
    db = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
    m = np.column_stack([bearing_to_erp(da, W, H), bearing_to_erp(db, W, H),
                         rng.uniform(conf[0], conf[1], n)])
    return m, R, t


# ---------- 測試環境 ----------
@contextlib.contextmanager
def env_override(**kw):
    """暫時覆寫環境變數；值為 None 表示刪除該變數。離開時還原原值。"""
    old = {k: os.environ.get(k) for k in kw}
    try:
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def quiet():
    """收走區塊內的 stdout。

    受測的腳本函式多半設計成 CLI 使用而會逐項印出進度；測試只看回傳值與副作用，
    讓那些輸出混進測試結果只會讓真正的失敗訊息被淹沒。
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


@contextlib.contextmanager
def temp_dir():
    """建立臨時目錄，離開時整棵刪除。"""
    d = tempfile.mkdtemp(prefix="roma_geo_test_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def write_json(path, obj):
    """寫出 UTF-8 JSON，必要時建立上層目錄。"""
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    return path
