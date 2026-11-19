"""RANSAC rigid registration from predicted column correspondences.

The alternative pose estimator described in the paper: instead of reading the
pose off the conv decoder, turn the dense per-column predictions into 2D-2D
point correspondences on the floor plane and fit a rigid transform.

How a correspondence becomes a point pair
-----------------------------------------
For column ``j`` of pano A:

* A's own boundary gives the radial distance ``r_a[j]``, so the observed floor
  point in **A's frame** is ``P_j = r_a[j] * dir(theta_j)``.
* The correspondence head says that point is seen from B at azimuth
  ``phi_j = corr_a[j]``.  B's own boundary, sampled at that azimuth, gives the
  distance ``r_b(phi_j)``, so the same point in **B's frame** is
  ``Q_j = r_b(phi_j) * dir(phi_j)``.

``(P_j, Q_j)`` is then a 2D-2D correspondence of one physical point, and
``Q = Rot(yaw) P + t`` is a rigid fit (no scale -- see assumption A6).
Columns the co-visibility head rejects are never proposed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .geometry import bfloor_to_dist, col_centers_theta, dir_of_theta, wrap_pi


@dataclass
class RansacResult:
    t: np.ndarray            # (2,) translation, A's origin in B's frame
    yaw: float               # radians
    inliers: np.ndarray      # (n,) bool over the proposed correspondences
    n_proposed: int
    ok: bool                 # False if there were too few correspondences


def sample_boundary_at_azimuth(bon: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    """Circularly interpolate a per-column boundary at arbitrary azimuths.

    ``bon`` is the HorizonNet boundary over ``n_cols`` columns; returns the
    interpolated *radial distance* at each requested azimuth.
    """
    n = len(bon)
    theta = col_centers_theta(n)
    r = bfloor_to_dist(bon)
    # np.interp with period handles the wraparound for us
    return np.interp(wrap_pi(azimuth), theta, r, period=2 * math.pi)


def kabsch_2d(p: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, float]:
    """Least-squares rigid fit ``q ~= Rot(yaw) p + t`` for 2D point sets."""
    pc, qc = p.mean(0), q.mean(0)
    p0, q0 = p - pc, q - qc
    # closed form for 2D: yaw = atan2(sum cross, sum dot)
    cross = (p0[:, 0] * q0[:, 1] - p0[:, 1] * q0[:, 0]).sum()
    dot = (p0 * q0).sum()
    yaw = math.atan2(cross, dot)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    t = qc - R @ pc
    return t, yaw


def build_correspondences(
    bon_a: np.ndarray,
    bon_b: np.ndarray,
    corr_a: np.ndarray,
    covis_a: np.ndarray,
    max_dist: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn per-column predictions into matched 2D point sets ``(P, Q)``."""
    n = len(bon_a)
    theta = col_centers_theta(n)
    r_a = bfloor_to_dist(bon_a)
    phi = wrap_pi(corr_a)
    r_b = sample_boundary_at_azimuth(bon_b, phi)

    keep = (
        np.asarray(covis_a, dtype=bool)
        & np.isfinite(r_a) & np.isfinite(r_b)
        & (r_a > 1e-3) & (r_b > 1e-3)
        & (r_a < max_dist) & (r_b < max_dist)
    )
    P = dir_of_theta(theta[keep]) * r_a[keep][:, None]
    Q = dir_of_theta(phi[keep]) * r_b[keep][:, None]
    return P, Q


def ransac_rigid(
    P: np.ndarray,
    Q: np.ndarray,
    threshold: float = 0.10,
    iters: int = 500,
    min_points: int = 6,
    seed: int = 0,
    refine: bool = True,
) -> RansacResult:
    """RANSAC + least-squares refit for ``Q ~= Rot(yaw) P + t``.

    ``threshold`` is the inlier residual in camera heights. The default 0.10
    deliberately matches ``targets.DEFAULT_ABS_TOL``: a column is only labelled
    co-visible when the two boundary distances agree to within that, so the
    correspondences fed in here already carry up to that much slack, and a
    looser RANSAC threshold just lets the slack bias the least-squares refit.
    Measured on ground-truth inputs (house 0000, 102 pairs, n_cols=256):
    median |t| error 0.030 at threshold 0.25 vs **0.015 at 0.10**, with the same
    number of pairs solved. See docs/ASSUMPTIONS.md A11.

    Two correspondences determine a 2D rigid transform, so the minimal sample
    is 2.
    """
    n = len(P)
    if n < min_points:
        return RansacResult(np.zeros(2), 0.0, np.zeros(n, bool), n, False)

    rng = np.random.default_rng(seed)
    best_inl = np.zeros(n, bool)
    best_count = -1
    best = (np.zeros(2), 0.0)

    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        if np.linalg.norm(P[i] - P[j]) < 1e-3:
            continue
        t, yaw = kabsch_2d(P[[i, j]], Q[[i, j]])
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        res = np.linalg.norm((R @ P.T).T + t - Q, axis=1)
        inl = res < threshold
        cnt = int(inl.sum())
        if cnt > best_count:
            best_count, best_inl, best = cnt, inl, (t, yaw)

    t, yaw = best
    if refine and best_inl.sum() >= 2:
        t, yaw = kabsch_2d(P[best_inl], Q[best_inl])
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        res = np.linalg.norm((R @ P.T).T + t - Q, axis=1)
        best_inl = res < threshold

    return RansacResult(np.asarray(t), float(wrap_pi(yaw)), best_inl, n, True)


def pose_from_predictions(
    bon_a: np.ndarray,
    bon_b: np.ndarray,
    corr_a: np.ndarray,
    covis_a: np.ndarray,
    covis_thresh: float = 0.5,
    **kw,
) -> RansacResult:
    """End-to-end convenience: per-column predictions -> RANSAC pose."""
    mask = np.asarray(covis_a)
    if mask.dtype != bool:
        mask = mask > covis_thresh
    P, Q = build_correspondences(bon_a, bon_b, corr_a, mask)
    return ransac_rigid(P, Q, **kw)
