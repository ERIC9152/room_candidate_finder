# -*- coding: utf-8 -*-
"""有位姿、內點夠多、兩側共視楔形都夠寬，才判同房。"""

INLIERS = 30
WEDGE_DEG = 60.0


def _inliers(geometry):
    n = geometry.get("n") or {}
    if "inlier" in n and n["inlier"] is not None:
        return int(n["inlier"])
    return int(geometry.get("n_inlier") or 0)


def _has_pose(geometry):
    if (geometry.get("flags") or {}).get("no_covisibility"):
        return False
    az = (geometry.get("pose") or {}).get("az_ab_deg")
    return az is not None and az != ""


def decides_same_room(geometry, min_inliers=INLIERS, min_wedge_deg=WEDGE_DEG):
    if not _has_pose(geometry):
        return False
    if _inliers(geometry) < min_inliers:
        return False
    quality = geometry.get("quality") or {}
    a = quality.get("az_wedge_a_deg")
    b = quality.get("az_wedge_b_deg")
    if a is None or b is None:
        return False
    return float(a) >= min_wedge_deg and float(b) >= min_wedge_deg
