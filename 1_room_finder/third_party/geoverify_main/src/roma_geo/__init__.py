"""Roma_Geo 套件入口：單對幾何驗證與整房流程。匯入本套件不修改 sys.path。"""
import json
import os
import time

from .names import house_of as house_of  # re-export
from .config import Settings  # noqa: F401
from .errors import RomaGeoError  # noqa: F401

__all__ = [
    "verify_pair", "verify_pairs", "to_public", "region_to_public", "get_matcher",
    "house_of", "DEFAULTS", "SCHEMA_VERSION", "MATCH_MODE_DEFAULT",
    "Settings", "RomaGeoError",
    "format_dependency_error", "require_modules", "require_matching_deps",
]

MATCH_MODE_DEFAULT = "direct"

DEFAULTS = {
    "conf_min": 0.65,
    "retry_conf": 0.3,
    "max_pts": 4000,
    "thresh_deg": 0.5,
    "iters": 2000,
    "seed": 0,
    "sampler": "stratified",
    "lat_max": 60.0,
}

#: 從 verify_pair 結果原樣複製的欄位。不是對外輸出的完整白名單。
#: az_ab_deg、az_wedge_*、no_covisibility、regions 由 pose / quality / flags 提升而來。
PUBLIC_FIELDS = ("schema_version", "house", "img_a", "img_b",
                 "img_size_a", "img_size_b", "conf_used", "inlier_rate")

PUBLIC_REGION_FIELDS = ("n", "frac_of_inliers")
PUBLIC_REGION_SIDE_FIELDS = ("polygon", "area_frac", "az_range")

_matchers = {}

_RUNTIME_DEPS = (("numpy", "numpy"), ("PIL", "pillow"))
_MATCH_DEPS = (("torch", "torch"),)


def format_dependency_error(missing, action):
    """缺少執行相依時給呼叫端看的說明。"""
    return (
        "無法%s：缺少 %s。"
        "本專案的相依版本見 environments/requirements-core.txt，"
        "請執行 environments/setup_env.ps1 或 environments/setup_env.sh。"
        % (action, "、".join(missing))
    )


def require_modules(mod_and_dist, action):
    """mod_and_dist 為 (匯入名, 套件名) 列表。缺任一項即 ImportError，並指出依據。"""
    missing = []
    for mod, dist in mod_and_dist:
        try:
            __import__(mod)
        except ImportError:
            missing.append(dist)
    if missing:
        raise ImportError(format_dependency_error(missing, action))


def require_matching_deps(action="載入匹配模型"):
    """匹配路徑會用到的相依：數值運算、影像、torch。"""
    require_modules(_RUNTIME_DEPS + _MATCH_DEPS, action)


def _load_analyze_pair():
    from pose_regions import analyze_pair
    return analyze_pair


def _load_matcher_class():
    from roma_match.matcher import RoMaMatcher
    return RoMaMatcher


try:
    from pose_regions.batch_pose_from_cache import SCHEMA_VERSION
except Exception:                                     # pragma: no cover
    SCHEMA_VERSION = "wp1-step3-v2"


def get_matcher(model_type="roma_v2", device=None, use_cache=True, cache_dir=None):
    """取得（或建立）共用的 RoMaMatcher 單例。"""
    require_matching_deps("載入匹配模型")
    key = (model_type, device, bool(use_cache), cache_dir)
    m = _matchers.get(key)
    if m is None:
        RoMaMatcher = _load_matcher_class()
        m = RoMaMatcher(device=device, conf_threshold=0.1, use_cache=use_cache,
                        model_type=model_type, cache_dir=cache_dir)
        _matchers[key] = m
    return m


def verify_pair(img_a, img_b, matcher=None, mode=MATCH_MODE_DEFAULT, model_type="roma_v2",
                device=None, use_cache=True, cache_dir=None, house=None,
                keep_inliers=True, **params):
    """對一組 ERP 全景圖做幾何驗證。"""
    if matcher is None:
        require_matching_deps("做幾何驗證")
    else:
        require_modules(_RUNTIME_DEPS, "做幾何驗證")
    bad = set(params) - set(DEFAULTS)
    if bad:
        raise TypeError("未知參數 %s；可覆寫的只有 %s" % (sorted(bad), sorted(DEFAULTS)))
    p = dict(DEFAULTS, **params)

    import numpy as np
    analyze_pair = _load_analyze_pair()

    m = matcher if matcher is not None else get_matcher(model_type, device, use_cache, cache_dir)

    t0 = time.perf_counter()
    if mode == "direct":
        matches, info = m.match_direct(img_a, img_b)
    elif mode == "core":
        matches, info = m.match_core(img_a, img_b)
    else:
        raise ValueError("mode 必須是 direct 或 core，收到 %r" % (mode,))
    t_match = time.perf_counter() - t0

    W, H = info["w"], info["h"]
    Wb, Hb = _size(img_b, W, H)

    arr = np.asarray(matches, dtype=float)
    if arr.size == 0:
        arr = np.empty((0, 5), dtype=float)

    t0 = time.perf_counter()
    res = analyze_pair(arr, W, H, Wb, Hb,
                       conf_min=p["conf_min"], max_pts=p["max_pts"],
                       thresh_deg=p["thresh_deg"], iters=p["iters"],
                       seed=p["seed"], sampler=p["sampler"],
                       retry_conf=p["retry_conf"], lat_max=p["lat_max"])
    t_geo = time.perf_counter() - t0

    res["house"] = house if house is not None else house_of(img_a)
    res["img_a"] = os.path.splitext(os.path.basename(str(img_a)))[0]
    res["img_b"] = os.path.splitext(os.path.basename(str(img_b)))[0]
    res["timing"] = {"match_s": round(t_match, 4),
                     "verify_s": round(t_geo, 4),
                     "total_s": round(t_match + t_geo, 4)}
    if not keep_inliers:
        res.pop("inliers", None)
    return res


def _size(path, default_w, default_h):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size[0], im.size[1]
    except Exception:
        return default_w, default_h


def verify_pairs(pairs, progress=False, **kw):
    """對多組圖做幾何驗證，全程重用同一個 matcher。pairs 可以是產生器。"""
    if kw.get("matcher") is None:
        require_matching_deps("做幾何驗證")
    m = kw.pop("matcher", None) or get_matcher(
        kw.get("model_type", "roma_v2"), kw.get("device"),
        kw.get("use_cache", True), kw.get("cache_dir"))
    seq = list(pairs) if progress else pairs
    n = len(seq) if progress else None
    for i, (a, b) in enumerate(seq, 1):
        res = verify_pair(a, b, matcher=m, **kw)
        if progress:
            posed = "有位姿" if res.get("pose") else "無位姿"
            print("[%d/%d] %s %s · %d 區域 · %.2fs"
                  % (i, n, res["img_a"][:8], posed,
                     len(res.get("regions") or []), res["timing"]["total_s"]), flush=True)
        yield res


def region_to_public(r):
    """把單一共視區域裁成對外欄位。"""
    out = {k: r[k] for k in PUBLIC_REGION_FIELDS if k in r}
    for side in ("a", "b"):
        g = r.get(side) or {}
        out[side] = {k: g[k] for k in PUBLIC_REGION_SIDE_FIELDS if k in g}
    return out


def to_public(res):
    """單對對外契約：通道方位、共視區域、可信度。不含相對位姿。"""
    pose = res.get("pose") or {}
    q = res.get("quality") or {}
    out = {k: res.get(k) for k in PUBLIC_FIELDS}
    out["az_ab_deg"] = pose.get("az_ab_deg")
    out["az_ba_deg"] = pose.get("az_ba_deg")
    out["az_wedge_a_deg"] = q.get("az_wedge_a_deg")
    out["az_wedge_b_deg"] = q.get("az_wedge_b_deg")
    out["no_covisibility"] = bool((res.get("flags") or {}).get("no_covisibility"))
    out["regions"] = [region_to_public(r) for r in (res.get("regions") or [])]
    return out
