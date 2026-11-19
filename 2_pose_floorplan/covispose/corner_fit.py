"""Straight polygon fitting with fixed corner bearings; cameras are never optimized."""
import numpy as np
from scipy.optimize import least_squares
from shapely.geometry import Polygon, Point
from .geometry import col_centers_theta, dir_of_theta, bfloor_to_dist, theta_of, theta_to_col, polygon_to_column_distance


def fit_corner_polygon(bon, columns, max_range=20.0):
    bon = np.asarray(bon, float)
    width = len(bon)
    columns = np.unique(np.asarray(columns, float) % width)
    if len(columns) < 3 or len(columns) > 64:
        return None, 'corner_count'
    gaps = (np.roll(columns, -1)-columns) % width
    if np.max(gaps) >= width/2 or np.min(gaps) < 1:
        return None, 'invalid_bearings'
    grid = np.arange(width)
    previous = (np.searchsorted(columns, grid, side='right')-1) % len(columns)
    following = (previous+1) % len(columns)
    fraction = (grid-columns[previous]) % width
    span = gaps[previous]*2*np.pi/width
    angle = fraction*2*np.pi/width
    # Inverse range to a line is linear in its endpoint inverse ranges.
    A = np.zeros((width, len(columns)))
    A[grid, previous] = np.sin(span-angle)/np.sin(span)
    A[grid, following] = np.sin(angle)/np.sin(span)
    init = np.interp(columns, np.arange(width), np.tan(bon), period=width)
    result = least_squares(lambda x: np.arctan(A@x)-bon,
        np.clip(init, 1/max_range+1e-5, 9.99),
        jac=lambda x: A/(1+(A@x)**2)[:, None], bounds=(1/max_range, 10),
        loss='soft_l1', f_scale=np.radians(1.5), max_nfev=80)
    theta = ((columns+.5)/width-.5)*2*np.pi
    vertices = dir_of_theta(theta)/result.x[:, None]
    shape = Polygon(vertices)
    if not result.success or not shape.is_valid or not shape.contains(Point(0, 0)):
        return None, 'invalid_polygon'
    return vertices, 'ok'


def boundary_polygon(bon):
    return dir_of_theta(col_centers_theta(len(bon)))*np.clip(bfloor_to_dist(bon), .1, 20)[:, None]


def boundary_only_corners(bon, tolerance=.15):
    poly = Polygon(boundary_polygon(bon)).simplify(tolerance, preserve_topology=True)
    return theta_to_col(theta_of(np.asarray(poly.exterior.coords)[:-1]), len(bon))


def refine_corner_polygon(bon, columns, penalty=.1, trigger_deg=2.0,
                          search_deg=8.0, max_splits=3):
    """Split a merged bearing using boundary turns on either side of that peak.

    No GT is consulted. Penalized robust reprojection loss rejects noise-driven
    splits. Search is restricted to existing peaks, avoiding free reconstruction
    of distant doorway floor protrusions. Cameras and boundary remain fixed.
    """
    bon = np.asarray(bon, float)
    width = len(bon)
    bearings = np.unique(np.asarray(columns, float) % width)
    poly, status = fit_corner_polygon(bon, bearings)
    if poly is None:
        return None, dict(status=status, accepted=[])
    candidates = np.sort(boundary_only_corners(bon))
    def residual(p):
        b = np.arctan2(1, polygon_to_column_distance(p, width))
        return np.degrees(b-bon)
    def score(p, count):
        r = residual(p)/1.5
        return float(np.mean(2*1.5**2*(np.sqrt(1+r*r)-1))+penalty*count)
    accepted = []
    for _ in range(max_splits):
        current = score(poly, len(bearings))
        errors = residual(poly)
        best = None
        for k, center in enumerate(bearings):
            offsets = (candidates-center+width/2) % width-width/2
            limit = search_deg*width/360
            left = candidates[(offsets < -1) & (offsets >= -limit)]
            right = candidates[(offsets > 1) & (offsets <= limit)]
            if not len(left) or not len(right):
                continue
            # Nearest boundary turns bracketing the merged peak.
            lo = left[np.argmin((center-left) % width)]
            hi = right[np.argmin((right-center) % width)]
            local = np.abs((np.arange(width)-center+width/2) % width-width/2) <= limit
            if np.percentile(np.abs(errors[local]), 80) < trigger_deg:
                continue
            others = np.delete(bearings, k)
            proposed = np.unique(np.r_[others, lo, hi])
            if len(proposed) != len(bearings)+1:
                continue
            fitted, state = fit_corner_polygon(bon, proposed)
            if fitted is None:
                continue
            value = score(fitted, len(proposed))
            if value < current-1e-4 and (best is None or value < best[0]):
                best = (value, fitted, proposed, dict(replaced=float(center),
                    inserted=[float(lo), float(hi)], objective_before=current, objective_after=value))
        if best is None:
            break
        _, poly, bearings, event = best
        accepted.append(event)
    return poly, dict(status='ok', accepted=accepted, final_columns=bearings.tolist())
