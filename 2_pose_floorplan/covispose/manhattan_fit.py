"""Room-local soft Manhattan fitting with fixed camera and corner bearings."""
import numpy as np
from scipy.optimize import least_squares
from shapely.geometry import Polygon, Point
from .geometry import theta_of, theta_to_col, dir_of_theta, polygon_to_column_distance


def axis_offset(angles, axis):
    return (angles-axis+np.pi/4) % (np.pi/2)-np.pi/4


def dominant_axis(poly):
    edges = np.roll(poly, -1, axis=0)-poly
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    lengths = np.linalg.norm(edges, axis=1)
    scores = [lengths[np.abs(axis_offset(angles, a)) < np.radians(15)].sum() for a in angles]
    center = angles[int(np.argmax(scores))]
    offsets = axis_offset(angles, center)
    keep = np.abs(offsets) < np.radians(15)
    return float((center+np.average(offsets[keep], weights=lengths[keep])) % (np.pi/2))


def direction_error(poly, axis=None):
    axis = dominant_axis(poly) if axis is None else axis
    edges = np.roll(poly, -1, axis=0)-poly
    lengths = np.linalg.norm(edges, axis=1)
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    return float(np.average(np.degrees(np.abs(axis_offset(angles, axis))), weights=lengths))


def manhattan_polygon(bon, poly, weight=.1):
    bon = np.asarray(bon, float); poly = np.asarray(poly, float)
    width = len(bon)
    cols = theta_to_col(theta_of(poly), width)
    order = np.argsort(cols); cols, poly = cols[order], poly[order]
    gaps = (np.roll(cols, -1)-cols) % width
    if len(poly) < 3 or np.max(gaps) >= width/2 or np.min(gaps) < 1:
        return poly, dict(status='invalid_bearings')
    theta = ((cols+.5)/width-.5)*2*np.pi
    dirs = dir_of_theta(theta)
    grid = np.arange(width)
    left = (np.searchsorted(cols, grid, side='right')-1) % len(cols)
    span = gaps[left]*2*np.pi/width
    offset = ((grid-cols[left]) % width)*2*np.pi/width
    A = np.zeros((width, len(cols)))
    A[grid, left] = np.sin(span-offset)/np.sin(span)
    A[grid, (left+1) % len(cols)] = np.sin(offset)/np.sin(span)
    axis = dominant_axis(poly)
    edge = np.roll(poly, -1, axis=0)-poly
    lengths = np.linalg.norm(edge, axis=1)
    angles = np.arctan2(edge[:, 1], edge[:, 0])
    # Preserve strongly non-Manhattan walls rather than assigning them an axis.
    gate = np.abs(axis_offset(angles, axis)) <= np.radians(20)
    support = lengths/max(lengths.sum(), 1e-12)
    scale = np.sqrt(width*weight*support[gate])
    initial = np.r_[np.clip(1/np.linalg.norm(poly, axis=1), .050001, 9.99), axis]
    def residual(x):
        points = dirs/x[:-1, None]
        edges = np.roll(points, -1, axis=0)-points
        wall = np.arctan2(edges[:, 1], edges[:, 0])
        data = np.degrees(np.arctan(A@x[:-1])-bon)
        prior = np.degrees(axis_offset(wall[gate], x[-1]))*scale
        return np.r_[data, prior]
    def mixed_loss(z):
        # Robust boundary evidence; quadratic direction prior for gated walls.
        rho = np.vstack([z.copy(), np.ones_like(z), np.zeros_like(z)])
        r = np.sqrt(1+z[:width])
        rho[0, :width] = 2*(r-1)
        rho[1, :width] = 1/r
        rho[2, :width] = -.5/r**3
        return rho
    fit = least_squares(residual, initial, bounds=(np.r_[np.full(len(cols), .05), axis-np.pi/8],
        np.r_[np.full(len(cols), 10.), axis+np.pi/8]), loss=mixed_loss, f_scale=1.5, max_nfev=120)
    candidate = dirs/fit.x[:-1, None]
    shape = Polygon(candidate)
    if not fit.success or not shape.is_valid or not shape.contains(Point(0, 0)):
        return poly, dict(status='fallback', axis_deg=float(np.degrees(axis)))
    return candidate, dict(status='ok', axis_deg=float(np.degrees(fit.x[-1]) % 90),
        constrained_edges=int(gate.sum()), unconstrained_edges=int((~gate).sum()),
        direction_error_deg=direction_error(candidate, fit.x[-1]), columns=cols.tolist())


def merge_collinear(bon, poly, weight=.1, turn_deg=10., vertex_penalty=.1, refit_initial=True):
    """Delete near-straight redundant corners only if evidence cost permits it."""
    if refit_initial:
        current, report = manhattan_polygon(bon, poly, weight)
    else:
        current = np.asarray(poly, float).copy()
        report = dict(status='ok',axis_deg=float(np.degrees(dominant_axis(current))),
                      direction_error_deg=direction_error(current))
    accepted = []
    def objective(p):
        b = np.arctan2(1, polygon_to_column_distance(p, len(bon)))
        r = np.degrees(b-bon)/1.5
        return float(np.mean(2*1.5**2*(np.sqrt(1+r*r)-1))+vertex_penalty*len(p))
    for _ in range(8):
        if len(current) <= 4:
            break
        edges = np.roll(current, -1, axis=0)-current
        lengths = np.linalg.norm(edges, axis=1)
        unit = edges/np.maximum(lengths[:, None], 1e-12)
        turns = np.degrees(np.arccos(np.clip((np.roll(unit, 1, axis=0)*unit).sum(1), -1, 1)))
        base_score = objective(current); best = None
        for k in np.flatnonzero(turns < turn_deg):
            proposal, info = manhattan_polygon(bon, np.delete(current, k, axis=0), weight)
            if info['status'] != 'ok':
                continue
            score = objective(proposal)
            if score < base_score and (best is None or score < best[0]):
                best = score, proposal, info, dict(removed_column=float(theta_to_col(theta_of(current[k]),len(bon))),
                    turn_deg=float(turns[k]), objective_before=base_score, objective_after=score)
        if best is None:
            break
        _, current, report, event = best
        accepted.append(event)
    return current, dict(**report, merges=accepted)
