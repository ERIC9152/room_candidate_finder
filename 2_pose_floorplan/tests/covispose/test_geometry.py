"""Self-checks for the geometry conventions.

Run with:  python -m pytest tests/  (or just: python tests/test_geometry.py)

The important checks cross-validate our conventions against ZInD's own
``code/transformations.py`` rather than against our own re-derivation.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _requires import needs_zind_code  # noqa: E402

from covispose.geometry import (  # noqa: E402
    PanoPose,
    bfloor_to_dist,
    col_centers_theta,
    dir_of_theta,
    dist_to_bfloor,
    polygon_to_column_distance,
    ray_cast_polygon,
    relative_pose,
    theta_of,
    theta_to_col,
    transform_points,
    wrap_pi,
)


def test_raycast_square():
    """Unit square [-1,1]^2, camera at the origin: analytic distance is
    1 / max(|cos|, |sin|) of the ray direction in the *cartesian* frame."""
    poly = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    theta = col_centers_theta(256)
    dirs = dir_of_theta(theta)
    r = ray_cast_polygon(np.zeros(2), dirs, poly, closed=True)
    expect = 1.0 / np.maximum(np.abs(dirs[:, 0]), np.abs(dirs[:, 1]))
    assert np.all(np.isfinite(r)), "closed polygon around the origin must hit every ray"
    assert np.allclose(r, expect, atol=1e-9), np.abs(r - expect).max()
    print("  ok  test_raycast_square")


def test_raycast_offcenter():
    """Same square but casting from an interior point that is not the origin."""
    poly = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    o = np.array([0.3, -0.45])
    dirs = dir_of_theta(col_centers_theta(64))
    r = ray_cast_polygon(o, dirs, poly, closed=True)
    hit = o[None, :] + dirs * r[:, None]
    assert np.all(np.isfinite(r))
    # every hit must be on the square's border
    on_border = np.isclose(np.abs(hit).max(axis=1), 1.0, atol=1e-9)
    assert on_border.all(), np.abs(hit).max(axis=1)[~on_border]
    print("  ok  test_raycast_offcenter")


def test_boundary_roundtrip():
    r = np.array([0.5, 1.0, 2.0, 7.0, 40.0])
    assert np.allclose(bfloor_to_dist(dist_to_bfloor(r)), r, rtol=1e-6)
    # a wall at exactly camera height distance sits at 45 deg below the horizon
    assert abs(dist_to_bfloor(np.array([1.0]))[0] - math.pi / 4) < 1e-12
    print("  ok  test_boundary_roundtrip")


@needs_zind_code
def test_azimuth_matches_zind():
    """Our theta / column mapping must agree with ZInD's sphere_to_pixel to
    within the known half-pixel (W vs W-1) convention difference."""
    from transformations import TransformationSpherical  # ZInD's module

    rng = np.random.default_rng(0)
    pts2d = rng.normal(size=(200, 2)) * 3.0
    pts3d = np.hstack([pts2d, -np.ones((len(pts2d), 1))])

    W = 1024
    zind_px = TransformationSpherical.cartesian_to_pixel(pts3d, W)[:, 0]
    ours_theta = theta_of(pts2d)
    ours_px = theta_to_col(ours_theta, W)

    # ZInD maps theta in [-pi,pi] onto [0, W-1]; we map onto column centres of
    # a W-wide grid.  Convert ZInD's to our parameterisation before comparing.
    zind_theta = zind_px / (W - 1) * 2 * math.pi - math.pi
    assert np.max(np.abs(wrap_pi(zind_theta - ours_theta))) < 1e-9

    # and the pixel columns differ by less than one pixel
    d = np.abs(wrap_pi((zind_px - ours_px) / W * 2 * math.pi)) / (2 * math.pi) * W
    assert d.max() < 1.0, d.max()
    print(f"  ok  test_azimuth_matches_zind (max column delta {d.max():.3f}px)")


@needs_zind_code
def test_elevation_matches_zind():
    """Our floor-boundary value b must match ZInD's row for the same 3D point."""
    from transformations import TransformationSpherical

    W, H = 1024, 512
    r = np.array([0.4, 1.0, 2.5, 9.0])
    pts3d = np.stack([np.zeros_like(r), r, -np.ones_like(r)], axis=1)  # +y, floor
    zind_row = TransformationSpherical.cartesian_to_pixel(pts3d, W)[:, 1]
    ours_b = dist_to_bfloor(r)
    ours_row = (ours_b / math.pi + 0.5) * H - 0.5
    # ZInD uses (H-1); rescale before comparing
    zind_b = (zind_row / (H - 1) - 0.5) * math.pi
    assert np.max(np.abs(zind_b - ours_b)) < 1e-9, np.abs(zind_b - ours_b).max()
    assert np.max(np.abs(zind_row - ours_row)) < 1.0
    print("  ok  test_elevation_matches_zind")


def test_relative_pose_against_global():
    """Composing through the global floor-plan frame must equal the direct
    relative transform returned by relative_pose()."""
    rng = np.random.default_rng(3)
    for _ in range(50):
        pa = PanoPose(rng.normal(size=2) * 5, rng.uniform(-180, 180), abs(rng.normal(1, 0.2)) + 0.1)
        pb = PanoPose(rng.normal(size=2) * 5, rng.uniform(-180, 180), abs(rng.normal(1, 0.2)) + 0.1)
        p_local_a = rng.normal(size=(20, 2)) * 2

        # route 1: A -> global -> B
        g = pa.to_global(p_local_a)
        ref = ((pb.R.T @ (g - pb.t).T).T) / pb.scale

        # route 2: direct
        t, yaw, k = relative_pose(pa, pb)
        got = transform_points(p_local_a, t, yaw, k)
        assert np.allclose(ref, got, atol=1e-9), np.abs(ref - got).max()

        # the origin of A must map to t
        assert np.allclose(transform_points(np.zeros((1, 2)), t, yaw, k)[0], t, atol=1e-12)
    print("  ok  test_relative_pose_against_global")


def test_relative_pose_inverse():
    rng = np.random.default_rng(7)
    pa = PanoPose(rng.normal(size=2), 37.0, 0.4)
    pb = PanoPose(rng.normal(size=2), -155.0, 0.4)
    t_ab, yaw_ab, k_ab = relative_pose(pa, pb)
    t_ba, yaw_ba, k_ba = relative_pose(pb, pa)
    p = rng.normal(size=(10, 2))
    back = transform_points(transform_points(p, t_ab, yaw_ab, k_ab), t_ba, yaw_ba, k_ba)
    assert np.allclose(p, back, atol=1e-9)
    assert abs(wrap_pi(yaw_ab + yaw_ba)) < 1e-12
    assert abs(k_ab * k_ba - 1.0) < 1e-12
    print("  ok  test_relative_pose_inverse")


def test_column_distance_closed_polygon():
    """A camera strictly inside a closed polygon must have no missing column."""
    rng = np.random.default_rng(11)
    for _ in range(20):
        n = int(rng.integers(6, 14))
        # Evenly spaced angles + bounded jitter, so no angular gap ever reaches
        # pi -- that keeps the polygon star-shaped about the origin, which is
        # what makes "every ray hits" the correct expectation.
        base = np.arange(n) * (2 * math.pi / n)
        ang = base + rng.uniform(-0.4, 0.4, size=n) * (2 * math.pi / n) / 2
        rad = rng.uniform(0.8, 4.0, size=n)
        poly = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
        assert np.max(np.diff(np.concatenate([ang, [ang[0] + 2 * math.pi]]))) < math.pi
        r = polygon_to_column_distance(poly, 256)
        assert np.all(np.isfinite(r)), np.mean(~np.isfinite(r))
    print("  ok  test_column_distance_closed_polygon")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{len(fns)} geometry checks passed")
