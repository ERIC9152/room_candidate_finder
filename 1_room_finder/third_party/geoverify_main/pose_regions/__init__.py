"""
pose_regions：相對位姿與共視區域估計(RoMa 匹配的幾何驗證層)

從 ERP 全景的像素對應點(RoMa matches)估計兩張圖的相對位姿(R、t 方向),並萃取
幾何驗證過的共視區域多邊形。

核心 API(純 numpy,無 GPU / cv2 依賴,匯入無副作用,cwd 無關):

    from pose_regions import analyze_pair
    res = analyze_pair(matches, Wa, Ha, Wb, Hb,
                       conf_min=0.65, max_pts=4000, thresh_deg=0.5, iters=2000)
    # matches: (K,5) ndarray = [x_a, y_a, x_b, y_b, conf](原圖 ERP 像素)
    # res: dict = per-pair JSON 主體(schema 見 README「per-pair JSON schema」)
    #      pose / regions / flags / conf_used / quality ...

批次 CLI(全量 cache → Output/roma_regions/,用法見各檔 docstring):
    venv/Scripts/python.exe pose_regions/batch_pose_from_cache.py
    venv/Scripts/python.exe pose_regions/estimate_relative_pose.py --selftest
    venv/Scripts/python.exe pose_regions/render_region_overlay.py --sample
    venv/Scripts/python.exe pose_regions/roi_accuracy_eval.py
"""

from .estimate_relative_pose import (
    erp_to_bearing,
    ransac_E,
    select_pose,
    rotation_only_fit,
    sampson_residual,
    parse_cache_name,
    find_house_image,
)
from .batch_pose_from_cache import (
    SCHEMA_VERSION,
    analyze_pair,
    extract_regions,
    wedge_width,
    az_coverage,
)

__all__ = [
    "SCHEMA_VERSION",
    "analyze_pair",
    "extract_regions",
    "erp_to_bearing",
    "ransac_E",
    "select_pose",
    "rotation_only_fit",
    "sampson_residual",
    "wedge_width",
    "az_coverage",
    "parse_cache_name",
    "find_house_image",
]
