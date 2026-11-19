"""Equirectangular geometry: pixel <-> ray <-> floor/ceiling point.

Fixed convention (checked by tests/test_equirect.py and tests/test_seam.py):

* Pixel ``u`` grows to the right, ``v`` grows downward.  Most functions take
  *normalized* coordinates ``un = u / W`` and ``vn = v / H`` in [0, 1).
* Azimuth   ``theta = (un - 0.5) * 2*pi``   (0 at image centre, + to the right).
* Elevation ``phi   = (0.5 - vn) * pi``     (y-up: + above the horizon).
* Ray       ``d = [cos(phi) sin(theta), sin(phi), cos(phi) cos(theta)]``.
* Camera at the origin, floor plane ``y = -h``.  A floor pixel has ``d_y < 0``;
  ``lambda = -h / d_y`` and the 2D floor point is ``[lambda d_x, lambda d_z]``.
* The 2D top-down frame is therefore ``(x, z)``: +z is the direction of the
  image centre column, +x is the direction of the column at ``un = 0.75``.
  Plotting x to the right and z upward gives a non-mirrored view from above.
* Default ``h = 1``: every 2D length is in **camera_height** units, not metres.
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi
UNIT = "camera_height"


def uv_to_angles(un, vn):
    un = np.asarray(un, dtype=float)
    vn = np.asarray(vn, dtype=float)
    theta = (un - 0.5) * TWO_PI
    phi = (0.5 - vn) * np.pi
    return theta, phi


def angles_to_uv(theta, phi):
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    un = np.mod(theta / TWO_PI + 0.5, 1.0)
    vn = 0.5 - phi / np.pi
    return un, vn


def uv_to_ray(un, vn):
    theta, phi = uv_to_angles(un, vn)
    return np.stack([np.cos(phi) * np.sin(theta),
                     np.sin(phi),
                     np.cos(phi) * np.cos(theta)], axis=-1)


def ray_to_uv(d):
    d = np.asarray(d, dtype=float)
    theta = np.arctan2(d[..., 0], d[..., 2])
    phi = np.arctan2(d[..., 1], np.hypot(d[..., 0], d[..., 2]))
    return angles_to_uv(theta, phi)


def floor_uv_to_xy(un, vn, h=1.0):
    """Intersect floor-pixel rays with the plane y = -h. Returns (..., 2) [x, z].

    Pixels at or above the horizon (d_y >= 0) give NaN.
    """
    d = uv_to_ray(un, vn)
    dy = d[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(dy < 0, -h / dy, np.nan)
    return np.stack([lam * d[..., 0], lam * d[..., 2]], axis=-1)


def ceiling_uv_to_xy(un, vn, hc):
    """Intersect ceiling-pixel rays with the plane y = +hc."""
    d = uv_to_ray(un, vn)
    dy = d[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(dy > 0, hc / dy, np.nan)
    return np.stack([lam * d[..., 0], lam * d[..., 2]], axis=-1)


def xy_to_floor_uv(xy, h=1.0):
    xy = np.asarray(xy, dtype=float)
    p = np.stack([xy[..., 0], np.full(xy.shape[:-1], -h), xy[..., 1]], axis=-1)
    return ray_to_uv(p)


def xy_to_ceiling_uv(xy, hc):
    xy = np.asarray(xy, dtype=float)
    p = np.stack([xy[..., 0], np.full(xy.shape[:-1], hc), xy[..., 1]], axis=-1)
    return ray_to_uv(p)


def ceiling_ratio_from_pair(vn_ceil, vn_floor):
    """Ceiling-above-camera / camera-height from a ceiling & floor pixel of one corner column."""
    phi_c = (0.5 - np.asarray(vn_ceil, float)) * np.pi
    phi_f = (0.5 - np.asarray(vn_floor, float)) * np.pi
    return np.tan(phi_c) / np.tan(-phi_f)


def circ_du(a, b):
    """Signed circular difference a - b of normalized u, in [-0.5, 0.5)."""
    return np.mod(np.asarray(a, float) - np.asarray(b, float) + 0.5, 1.0) - 0.5


def split_at_seam(un, vn, jump=0.5):
    """Split a polyline in normalized uv wherever u wraps around the seam.

    Returns a list of (un, vn) arrays; each can be drawn without a line that
    crosses the whole image.
    """
    un = np.asarray(un, float)
    vn = np.asarray(vn, float)
    if un.size == 0:
        return []
    cut = np.where(np.abs(np.diff(un)) > jump)[0] + 1
    return [(a, b) for a, b in zip(np.split(un, cut), np.split(vn, cut)) if a.size]
