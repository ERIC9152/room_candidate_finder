"""Per-column ground truth for a pano pair: floor boundary, angular
correspondence and co-visibility.

All three targets live on the same 1D column grid (``n_cols`` columns spanning
the full 360 deg FoV), which is exactly the grid the network predicts on.

Co-visibility definition (an assumption -- the paper does not publish the
exact rule, see README):
    column ``j`` of pano A is co-visible with pano B iff the floor-wall
    boundary point that A sees at column ``j``, after being mapped into B's
    frame, coincides with the floor-wall boundary point B sees along the same
    azimuth.  "Coincides" = radial distances agree within
    ``max(abs_tol, rel_tol * r)``.  A point that is behind one of B's own walls
    ends up with a much smaller ``r_B`` and is correctly marked not co-visible,
    so this rule handles occlusion without any extra ray tracing.

    ...and both cameras must be on the same side of that wall (A17).  Distance
    agreement alone cannot tell the two faces of one wall apart: for panos in
    two adjacent rooms, the wall they share is a few centimetres thick in
    ZInD's floor plan -- inside the distance tolerance -- so 98 % of the
    "co-visible" columns of cross-room pairs were the same wall seen from
    opposite sides.  Two cameras inside one room are always on its inner side,
    so the test changes nothing for intra-room pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .geometry import (
    col_centers_theta,
    dir_of_theta,
    dist_to_bfloor,
    ray_cast_polygon,
    relative_pose,
    theta_of,
    transform_points,
    wrap_pi,
)
from .zind_parse import PanoRecord

# Tolerance for calling two boundary distances "the same wall point".
# Units are camera heights (1 unit ~ 1.5 m in ZInD).
DEFAULT_ABS_TOL = 0.10
DEFAULT_REL_TOL = 0.05


@dataclass
class PairTargets:
    """Ground truth for one ordered pair, plus the reverse direction.

    Arrays are (n_cols,) float32 / bool unless noted.
    """

    n_cols: int
    # per-pano floor-wall boundary, HorizonNet parameterisation (rad, > 0)
    bon_a: np.ndarray
    bon_b: np.ndarray
    # validity of the boundary itself (False where the polygon has no hit)
    bon_valid_a: np.ndarray
    bon_valid_b: np.ndarray
    # angular correspondence: azimuth in the *other* pano, rad in [-pi, pi)
    corr_a2b: np.ndarray
    corr_b2a: np.ndarray
    # co-visibility
    covis_a: np.ndarray
    covis_b: np.ndarray
    # relative pose of A expressed in B's frame:  p_b = k * Rot(yaw) p_a + t
    t_ab: np.ndarray  # (2,)
    yaw_ab: float
    scale_ab: float
    # boundary points of A mapped into B's frame (used by the RANSAC baseline)
    pts_a_in_b: np.ndarray  # (n_cols, 2)
    pts_b_in_a: np.ndarray  # (n_cols, 2)


def _boundary(rec: PanoRecord, n_cols: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column radial distance, boundary points and validity for one pano."""
    theta = col_centers_theta(n_cols)
    dirs = dir_of_theta(theta)
    r = ray_cast_polygon(np.zeros(2), dirs, rec.vertices, closed=True)
    valid = np.isfinite(r) & (r > 1e-3)
    r_safe = np.where(valid, r, 1.0)
    pts = dirs * r_safe[:, None]
    return r_safe, pts, valid


# A hit this close to an edge end is treated as hitting the corner, where
# either of the two walls meeting there may be the one the other camera faces.
_CORNER_S = 1e-3


def _same_side(cam: np.ndarray, poly: np.ndarray, edge: np.ndarray) -> np.ndarray:
    """Is ``cam`` on the same side of each ``edge``'s line as the origin?"""
    a = poly[edge]
    e = np.roll(poly, -1, axis=0)[edge] - a
    side_origin = e[:, 0] * (0.0 - a[:, 1]) - e[:, 1] * (0.0 - a[:, 0])
    side_cam = e[:, 0] * (cam[1] - a[:, 1]) - e[:, 1] * (cam[0] - a[:, 0])
    return side_origin * side_cam > 0.0


def _covisible(
    q: np.ndarray,
    other: PanoRecord,
    abs_tol: float,
    rel_tol: float,
    cam: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Are the points ``q`` (in ``other``'s frame) on ``other``'s visible boundary?

    ``cam`` is the position, in ``other``'s frame, of the camera that sees
    ``q``; it must face the wall from the same side as ``other``'s camera.

    Returns ``(covis_mask, theta_in_other)``.
    """
    theta_q = theta_of(q)
    d_q = np.linalg.norm(q, axis=-1)
    dirs = dir_of_theta(theta_q)
    poly = np.asarray(other.vertices, dtype=np.float64)
    r_other, edge, s = ray_cast_polygon(np.zeros(2), dirs, poly, closed=True,
                                        return_edge=True)

    finite = np.isfinite(r_other)
    r_cmp = np.where(finite, r_other, 0.0)
    tol = np.maximum(abs_tol, rel_tol * np.minimum(d_q, np.where(finite, r_cmp, d_q)))
    covis = finite & (np.abs(d_q - r_cmp) <= tol)

    m = len(poly)
    cam = np.asarray(cam, dtype=np.float64).reshape(2)
    facing = _same_side(cam, poly, edge)
    facing |= (s < _CORNER_S) & _same_side(cam, poly, (edge - 1) % m)
    facing |= (s > 1.0 - _CORNER_S) & _same_side(cam, poly, (edge + 1) % m)
    return covis & facing, theta_q


def compute_pair_targets(
    rec_a: PanoRecord,
    rec_b: PanoRecord,
    n_cols: int = 256,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> PairTargets:
    """Build the full per-column ground truth for the ordered pair (A, B)."""
    r_a, pts_a, valid_a = _boundary(rec_a, n_cols)
    r_b, pts_b, valid_b = _boundary(rec_b, n_cols)

    # Pose of A in B's frame, and its inverse.
    t_ab, yaw_ab, k_ab = relative_pose(rec_a.pose, rec_b.pose)
    t_ba, yaw_ba, k_ba = relative_pose(rec_b.pose, rec_a.pose)

    pts_a_in_b = transform_points(pts_a, t_ab, yaw_ab, k_ab)
    pts_b_in_a = transform_points(pts_b, t_ba, yaw_ba, k_ba)

    # A's camera sits at t_ab in B's frame, and B's at t_ba in A's.
    covis_a, corr_a2b = _covisible(pts_a_in_b, rec_b, abs_tol, rel_tol, cam=t_ab)
    covis_b, corr_b2a = _covisible(pts_b_in_a, rec_a, abs_tol, rel_tol, cam=t_ba)

    covis_a &= valid_a
    covis_b &= valid_b

    return PairTargets(
        n_cols=n_cols,
        bon_a=dist_to_bfloor(r_a).astype(np.float32),
        bon_b=dist_to_bfloor(r_b).astype(np.float32),
        bon_valid_a=valid_a,
        bon_valid_b=valid_b,
        corr_a2b=wrap_pi(corr_a2b).astype(np.float32),
        corr_b2a=wrap_pi(corr_b2a).astype(np.float32),
        covis_a=covis_a,
        covis_b=covis_b,
        t_ab=t_ab.astype(np.float64),
        yaw_ab=float(yaw_ab),
        scale_ab=float(k_ab),
        pts_a_in_b=pts_a_in_b,
        pts_b_in_a=pts_b_in_a,
    )
