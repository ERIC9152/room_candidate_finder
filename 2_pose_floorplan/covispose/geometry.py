"""Panorama / floor-plan geometry for the CoVisPose reimplementation.

Coordinate conventions (fixed here, used everywhere else)
--------------------------------------------------------
Pano-local frame (this is ZInD's per-pano "room shape" frame):
    * camera at the origin, camera height normalised to 1, so the floor plane
      is z = -1 and a floor point is (x, y, -1).
    * azimuth  theta = atan2(-x, y)  in [-pi, pi)   (ZInD's x-axis flip, see
      ``zind/code/transformations.py::TransformationSpherical.cartesian_to_sphere``)
      => the unit direction for a given theta is (-sin theta, cos theta).

Equirectangular image of size H x W:
    * column j  <->  theta_j = ((j + 0.5) / W - 0.5) * 2 * pi
    * row    i  <->  b_i     = ((i + 0.5) / H - 0.5) * pi
      ``b`` is HorizonNet's boundary parameterisation ("bon"): it grows
      downwards, so the floor-wall boundary has b > 0 and the ceiling-wall
      boundary has b < 0.  ZInD's elevation phi (up-positive) is  phi = -b.
    NOTE: ZInD's own ``sphere_to_pixel`` maps with (W-1)/(H-1) rather than
    W/H.  The two differ by half a pixel at W=1024; we use the HorizonNet
    convention because the network's column grid is defined that way.  This
    is documented in README.md under "assumptions".

Floor-wall boundary <-> radial distance, at camera height 1:
    b_floor = atan2(1, r)   (in (0, pi/2)),      r = 1 / tan(b_floor)

Floor-plan (global) frame, from ZInD's ``floor_plan_transformation``:
    p_global = scale * Rot(rotation) @ p_local + translation
  with Rot() the standard CCW rotation.  (Matches ``Transformation2D.to_global``,
  which right-multiplies a row vector by [[c, s], [-s, c]].)
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

__all__ = [
    "col_centers_theta",
    "theta_to_col",
    "row_centers_b",
    "b_to_row",
    "dist_to_bfloor",
    "bfloor_to_dist",
    "theta_of",
    "dir_of_theta",
    "PanoPose",
    "relative_pose",
    "ray_cast_polygon",
    "polygon_to_column_distance",
    "wrap_pi",
]


# --------------------------------------------------------------------------- #
# column / row <-> angle
# --------------------------------------------------------------------------- #
def col_centers_theta(width: int) -> np.ndarray:
    """Azimuth (rad) at the centre of every image column. Shape (W,)."""
    j = np.arange(width, dtype=np.float64)
    return ((j + 0.5) / width - 0.5) * 2.0 * math.pi


def theta_to_col(theta: np.ndarray, width: int) -> np.ndarray:
    """Continuous (sub-pixel) column index for an azimuth. Wrapped to [0, W)."""
    col = (np.asarray(theta) / (2.0 * math.pi) + 0.5) * width - 0.5
    return np.mod(col, width)


def row_centers_b(height: int) -> np.ndarray:
    """HorizonNet boundary value at the centre of every image row. Shape (H,)."""
    i = np.arange(height, dtype=np.float64)
    return ((i + 0.5) / height - 0.5) * math.pi


def b_to_row(b: np.ndarray, height: int) -> np.ndarray:
    """Continuous row index for a HorizonNet boundary value."""
    return (np.asarray(b) / math.pi + 0.5) * height - 0.5


def wrap_pi(a: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + math.pi) % (2.0 * math.pi) - math.pi


# --------------------------------------------------------------------------- #
# floor boundary <-> radial distance (camera height == 1)
# --------------------------------------------------------------------------- #
def dist_to_bfloor(r: np.ndarray) -> np.ndarray:
    """Radial distance on the floor plane -> HorizonNet floor boundary value."""
    return np.arctan2(1.0, np.asarray(r))


def bfloor_to_dist(b: np.ndarray) -> np.ndarray:
    """HorizonNet floor boundary value -> radial distance on the floor plane."""
    b = np.asarray(b)
    # b is in (0, pi/2); clamp away from the horizon to keep r finite.
    b = np.clip(b, 1e-4, math.pi / 2 - 1e-6)
    return 1.0 / np.tan(b)


# --------------------------------------------------------------------------- #
# pano-local 2D helpers
# --------------------------------------------------------------------------- #
def theta_of(p: np.ndarray) -> np.ndarray:
    """Azimuth of 2D pano-local point(s) ``p`` of shape (..., 2)."""
    p = np.asarray(p)
    return np.arctan2(-p[..., 0], p[..., 1])


def dir_of_theta(theta: np.ndarray) -> np.ndarray:
    """Unit 2D direction(s) for azimuth(s). Returns shape (..., 2)."""
    theta = np.asarray(theta)
    return np.stack([-np.sin(theta), np.cos(theta)], axis=-1)


# --------------------------------------------------------------------------- #
# poses
# --------------------------------------------------------------------------- #
def _rot(a_rad: float) -> np.ndarray:
    c, s = math.cos(a_rad), math.sin(a_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


class PanoPose:
    """ZInD ``floor_plan_transformation`` for one pano: local -> floor-plan global."""

    __slots__ = ("t", "rot_deg", "scale")

    def __init__(self, translation, rotation_deg: float, scale: float):
        self.t = np.asarray(translation, dtype=np.float64).reshape(2)
        self.rot_deg = float(rotation_deg)
        self.scale = float(scale)

    @classmethod
    def from_zind(cls, d: dict) -> "PanoPose":
        return cls(d["translation"], d["rotation"], d["scale"])

    @property
    def R(self) -> np.ndarray:
        return _rot(math.radians(self.rot_deg))

    def to_global(self, p_local: np.ndarray) -> np.ndarray:
        p = np.asarray(p_local, dtype=np.float64)
        return (self.R @ (p * self.scale).T).T + self.t


def relative_pose(src: PanoPose, dst: PanoPose) -> Tuple[np.ndarray, float, float]:
    """Pose of pano ``src`` expressed in pano ``dst``'s local frame.

    Returns ``(t_rel, yaw_rel, k)`` such that for any 2D point ``p_src`` in
    ``src``'s local frame::

        p_dst = k * Rot(yaw_rel) @ p_src + t_rel

    * ``t_rel``    : (2,) position of ``src``'s camera centre in ``dst``'s frame,
                     in units of ``dst``'s camera height.
    * ``yaw_rel``  : relative yaw in radians, wrapped to [-pi, pi).
    * ``k``        : scale ratio ``src.scale / dst.scale``.  Because ZInD
                     normalises every pano's local frame by its own camera
                     height, ``k`` is the camera-height ratio of the two panos.
                     It is 1.0 for the vast majority of ZInD pairs.
    """
    yaw = wrap_pi(math.radians(src.rot_deg - dst.rot_deg)).item()
    k = src.scale / dst.scale
    t_rel = _rot(-math.radians(dst.rot_deg)) @ (src.t - dst.t) / dst.scale
    return t_rel, yaw, k


def transform_points(p_src: np.ndarray, t_rel, yaw_rel: float, k: float) -> np.ndarray:
    """Apply the relative pose returned by :func:`relative_pose` to 2D points."""
    p = np.asarray(p_src, dtype=np.float64)
    return (_rot(yaw_rel) @ (p * k).T).T + np.asarray(t_rel, dtype=np.float64).reshape(2)


# --------------------------------------------------------------------------- #
# ray casting against a room polygon
# --------------------------------------------------------------------------- #
def ray_cast_polygon(
    origin: np.ndarray,
    dirs: np.ndarray,
    poly: np.ndarray,
    closed: bool = True,
    return_edge: bool = False,
):
    """Nearest positive-t hit of rays ``origin + t * dirs`` against a polygon.

    Args:
        origin: (2,) ray origin.
        dirs:   (N, 2) ray directions (need not be normalised; ``t`` is in
                units of ``|dirs|``, so pass unit vectors for a metric ``t``).
        poly:   (M, 2) polygon vertices.
        closed: whether to include the closing edge poly[-1] -> poly[0].
        return_edge: also return which edge each ray hit and where along it.

    Returns:
        (N,) array of hit distances; ``inf`` where a ray misses everything.
        With ``return_edge``: ``(t, edge, s)`` -- the index of the hit edge
        (poly[edge] -> poly[edge + 1]) and the hit's fraction ``s`` in [0, 1]
        along it; both meaningless where ``t`` is ``inf``.
    """
    origin = np.asarray(origin, dtype=np.float64).reshape(2)
    dirs = np.asarray(dirs, dtype=np.float64).reshape(-1, 2)
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)

    a = poly if closed else poly[:-1]
    b = np.roll(poly, -1, axis=0) if closed else poly[1:]
    a = a - origin  # (M, 2), edge start relative to the ray origin
    e = b - (a + origin)  # (M, 2), edge vector

    # t * d - s * e = a
    #   cross with e:  t * (d x e)             = a x e   ->  t = (a x e) / (d x e)
    #   cross with d:            - s * (e x d) = a x d   ->  s = (a x d) / (d x e)
    dxe = dirs[:, None, 0] * e[None, :, 1] - dirs[:, None, 1] * e[None, :, 0]  # (N, M)
    axe = a[None, :, 0] * e[None, :, 1] - a[None, :, 1] * e[None, :, 0]  # (1, M)
    axd = a[None, :, 0] * dirs[:, None, 1] - a[None, :, 1] * dirs[:, None, 0]  # (N, M)

    with np.errstate(divide="ignore", invalid="ignore"):
        t = axe / dxe
        s = axd / dxe

    ok = np.isfinite(t) & (np.abs(dxe) > 1e-12) & (t > 1e-9) & (s >= 0.0) & (s <= 1.0)
    t = np.where(ok, t, np.inf)
    if not return_edge:
        return t.min(axis=1)
    edge = t.argmin(axis=1)
    rows = np.arange(t.shape[0])
    return t[rows, edge], edge, np.where(ok, s, 0.0)[rows, edge]


def polygon_to_column_distance(poly: np.ndarray, width: int) -> np.ndarray:
    """Per-column radial distance to the room polygon, seen from the camera.

    This is the 1D "floor-wall boundary" label in distance form; convert it
    with :func:`dist_to_bfloor` to get the HorizonNet boundary representation.

    Returns (W,) float array; entries are ``inf`` for columns with no hit
    (which happens for open / partial polygons such as ``layout_visible``).
    """
    theta = col_centers_theta(width)
    dirs = dir_of_theta(theta)
    return ray_cast_polygon(np.zeros(2), dirs, poly, closed=True)
