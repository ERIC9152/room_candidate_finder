"""Wall fitting and multi-view bundle adjustment over the per-column boundary.

The network predicts one boundary value per image column, independently. Nothing
in it knows that a wall is straight or that two panoramas looking at the same
wall must agree about where it is. This module adds both, after the fact:

    per-column boundary -> wall segments (per pano)
                        -> shared walls (across panos)
                        -> joint refinement of poses AND walls

Why the residual is an angle
----------------------------
The boundary is an angle, ``b = atan2(1, r)``, so ``dr/db = -(1 + r^2)``: a fixed
angular error becomes a radial error that grows with the *square* of the range.
Measuring residuals in metres would therefore weight a wall 5 camera heights
away 26x more than one at 1 -- the opposite of their reliability. So a candidate
wall is projected back into each panorama and compared in boundary-angle space,
where the network's error is roughly homoscedastic (measured: 1.6 deg on ZInD,
6.0 deg out of domain, both roughly flat in range).

What this can and cannot do
---------------------------
It is a local optimisation. Given poses that are roughly right it will average
per-column noise down over hundreds of observations and tighten the plan. Given
a pose that is 90 or 180 degrees wrong -- 20 % of pairs on the tour data, even
in-distribution -- it will converge confidently to a wrong answer that *looks*
cleaner than the input. :func:`pose_outliers` exists for exactly that reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import bfloor_to_dist, col_centers_theta, dir_of_theta, wrap_pi


# --------------------------------------------------------------------------- #
# per-panorama wall segments
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    """One straight wall piece seen by one panorama."""

    pano: str
    cols: np.ndarray              # column indices contributing to it
    normal: np.ndarray            # (2,) unit normal, in the pano's local frame
    offset: float                 # n . x = offset
    residual: float               # RMS perpendicular distance, camera heights

    @property
    def span(self) -> int:
        return len(self.cols)


def _fit_line(pts: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Total-least-squares line through 2D points -> (unit normal, offset, rms)."""
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[1]                                   # the least-variance direction
    n = n / (np.linalg.norm(n) + 1e-12)
    d = float(n @ c)
    rms = float(np.sqrt(np.mean((pts @ n - d) ** 2)))
    return n, d, rms


def _tls_from_moments(m) -> Tuple[np.ndarray, float, float]:
    """Total-least-squares line from accumulated moments -> (normal, offset, rms)."""
    n, sx, sy, sxx, syy, sxy = m
    mx, my = sx / n, sy / n
    cxx, cyy, cxy = sxx / n - mx * mx, syy / n - my * my, sxy / n - mx * my
    # smaller eigenvector of [[cxx, cxy], [cxy, cyy]] is the normal
    tr, det = cxx + cyy, cxx * cyy - cxy * cxy
    disc = max(tr * tr / 4 - det, 0.0)
    lam = tr / 2 - math.sqrt(disc)                      # smaller eigenvalue
    nx, ny = (cxy, lam - cxx) if abs(cxy) > 1e-12 else ((1.0, 0.0) if cxx <= cyy else (0.0, 1.0))
    norm = math.hypot(nx, ny) or 1.0
    normal = np.array([nx / norm, ny / norm])
    return normal, float(normal[0] * mx + normal[1] * my), math.sqrt(max(lam, 0.0))


def _moments(pts: np.ndarray):
    x, y = pts[:, 0], pts[:, 1]
    return np.array([len(pts), x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()])


def line_boundary(normal: np.ndarray, offset: float, dirs: np.ndarray) -> np.ndarray:
    """Boundary angle a wall line would produce, seen from the local origin.

    ``b = atan2(1, r)`` with ``r`` the range along each column direction; columns
    the line is behind or parallel to get ``nan``.
    """
    den = dirs @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        r = offset / den
    r = np.where(np.isfinite(r) & (r > 1e-3), r, np.nan)
    return np.arctan2(1.0, r)


def _fit_run(b, dirs, pts, idx, tol):
    """Fit one column run -> (normal, offset, p90 angular error, per-column error)."""
    normal, offset, _ = _tls_from_moments(_moments(pts[idx]))
    err = np.abs(b[idx] - line_boundary(normal, offset, dirs[idx]))
    if not np.isfinite(err).all():
        return normal, offset, np.inf, err
    return normal, offset, float(np.percentile(err, 90)), err


def fit_wall_segments(bon: np.ndarray, min_span: int = 10, tol_deg: float = 3.0,
                      max_dist: float = 15.0, jump_ratio: float = 0.35) -> List[Segment]:
    """Per-column boundary -> straight wall segments, in the pano's local frame.

    Split-and-merge in **boundary-angle space**:

    1. cut the column ring where the range jumps (an occlusion edge, not a wall);
    2. fit each run, and while its 90th-percentile angular error exceeds
       ``tol_deg``, split it at the worst column -- which is where the corner is;
    3. merge neighbours back together when their joint fit still passes.

    Two details that cost a rewrite each. The test is angular, because a metric
    tolerance fragments far walls and swallows corners on near ones (the same
    angular error becomes a radial error growing with r^2). And it is the 90th
    percentile, not the maximum: with per-column noise sigma, the max over a
    hundred columns exceeds any 2-sigma gate almost surely. Set ``tol_deg`` to
    about twice the network's boundary error (measured: 1.6 deg on ZInD, 6 deg
    on out-of-domain data).
    """
    n_cols = len(bon)
    b = np.asarray(bon, float)
    r = np.clip(bfloor_to_dist(b), 1e-3, max_dist)
    dirs = dir_of_theta(col_centers_theta(n_cols))
    pts = dirs * r[:, None]
    usable = r < max_dist * 0.999
    tol = math.radians(tol_deg)

    # ---- 1. runs of usable columns, cut at range jumps ------------------- #
    # rel_jump[j] compares column j with j+1, so a jump ends the run *after* j.
    # Rotating the ring so it starts just after a break keeps every run
    # contiguous; with no break at all the seam is handled by the merge below.
    rel_jump = np.abs(np.diff(r, append=r[:1])) / np.maximum(r, 1e-3)
    brk = np.flatnonzero((rel_jump > jump_ratio) | ~usable)
    order = np.arange(n_cols) if len(brk) == 0 else np.roll(np.arange(n_cols), -(int(brk[0]) + 1))

    uniq, cur = [], []
    for j in order:
        if not usable[j]:
            if len(cur) >= min_span:
                uniq.append(np.array(cur))
            cur = []
            continue
        cur.append(int(j))
        if rel_jump[j] > jump_ratio:
            if len(cur) >= min_span:
                uniq.append(np.array(cur))
            cur = []
    if len(cur) >= min_span:
        uniq.append(np.array(cur))

    # ---- 2. split at the corner until every piece is straight ------------ #
    pieces = []
    stack = list(uniq)
    while stack:
        idx = stack.pop()
        if len(idx) < min_span:
            continue
        normal, offset, p90, err = _fit_run(b, dirs, pts, idx, tol)
        if p90 <= tol:
            pieces.append((idx, normal, offset, err))
            continue
        if len(idx) < 2 * min_span:                        # too short to split further
            if p90 <= tol * 2.5:
                pieces.append((idx, normal, offset, err))
            continue
        # split at the worst column -- the corner -- but never so close to an end
        # that the split is illegal; clipping keeps the piece instead of dropping it
        k = int(np.clip(int(np.argmax(np.where(np.isfinite(err), err, -1))),
                        min_span, len(idx) - min_span))
        stack.append(idx[:k])
        stack.append(idx[k:])

    # ---- 3. merge neighbours that are really one wall -------------------- #
    pieces.sort(key=lambda p: p[0][0])
    merged = []
    for piece in pieces:
        if merged:
            prev = merged[-1]
            if (prev[0][-1] + 1) % n_cols == piece[0][0] % n_cols:
                idx = np.concatenate([prev[0], piece[0]])
                normal, offset, p90, err = _fit_run(b, dirs, pts, idx, tol)
                if p90 <= tol:
                    merged[-1] = (idx, normal, offset, err)
                    continue
        merged.append(piece)
    if len(merged) >= 2:                                   # across the ring seam
        a, z = merged[0], merged[-1]
        if a[0][0] == 0 and z[0][-1] == n_cols - 1:
            idx = np.concatenate([z[0], a[0]])
            normal, offset, p90, err = _fit_run(b, dirs, pts, idx, tol)
            if p90 <= tol:
                merged = [(idx, normal, offset, err)] + merged[1:-1]

    # Orient every wall so the camera -- the local origin -- is on its negative
    # side. Without a consistent orientation, association cannot tell one wall
    # from the parallel wall on the other side of the room.
    segs = []
    for idx, normal, offset, err in merged:
        if len(idx) < min_span:
            continue
        if offset < 0:
            normal, offset = -normal, -offset
        segs.append(Segment("", idx, normal, offset, float(np.sqrt(np.nanmean(err ** 2)))))
    return sorted(segs, key=lambda s: -s.span)


# --------------------------------------------------------------------------- #
# association across panoramas
# --------------------------------------------------------------------------- #
def _to_world(normal: np.ndarray, offset: float, theta: float, c: np.ndarray):
    """A line in a pano's local frame -> the same line in the scene frame."""
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    n_w = R @ normal
    return n_w, float(offset + n_w @ c)


def associate_walls(segments: Dict[str, List[Segment]], theta: Dict[str, float],
                    c: Dict[str, np.ndarray], angle_tol_deg: float = 12.0,
                    offset_tol: float = 0.35) -> List[dict]:
    """Group per-pano segments that are the same physical wall.

    Greedy agglomeration in the scene frame, largest segment first. Two segments
    merge when their normals and offsets agree; the tolerances are deliberately
    loose, because the poses feeding this are the thing being corrected.

    Orientation is kept, never canonicalised away. Forcing every line to a
    positive offset makes the two parallel walls of a room -- x = +3 and x = -3
    -- identical, and the optimiser then has to satisfy an impossible constraint
    by pushing cameras out of the building. It did exactly that: 18 units.
    A flipped normal is still allowed to match, but its offset flips with it,
    which is what separates those two walls (offsets 3 and -3, six apart).
    """
    items = []
    for pano, segs in segments.items():
        for s in segs:
            n_w, d_w = _to_world(s.normal, s.offset, theta[pano], c[pano])
            items.append({"pano": pano, "seg": s, "n": n_w, "d": d_w})
    items.sort(key=lambda it: -it["seg"].span)

    walls: List[dict] = []
    cos_tol = math.cos(math.radians(angle_tol_deg))
    for it in items:
        for w in walls:
            if abs(float(w["n"] @ it["n"])) < cos_tol:
                continue
            sign = 1.0 if float(w["n"] @ it["n"]) > 0 else -1.0
            if abs(sign * it["d"] - w["d"]) > offset_tol:
                continue
            if any(m["pano"] == it["pano"] for m in w["members"]):
                continue                        # one wall, one segment per pano
            k = len(w["members"])
            w["n"] = w["n"] + sign * it["n"] * (1.0 / (k + 1))
            w["n"] = w["n"] / np.linalg.norm(w["n"])
            w["d"] = (w["d"] * k + sign * it["d"]) / (k + 1)
            w["members"].append(it)
            break
        else:
            walls.append({"n": it["n"].copy(), "d": it["d"], "members": [it]})
    return walls


# --------------------------------------------------------------------------- #
# Atlanta prior
# --------------------------------------------------------------------------- #
def atlanta_directions(walls: Sequence[dict], n_dirs: int = 2, tol_deg: float = 20.0,
                       min_support: int = 120) -> List[float]:
    """Dominant wall directions, as angles in [0, pi/2).

    Atlanta rather than Manhattan: a building may hold several rectangular
    frames at odd angles to each other (a bay, an extension, a corridor), so
    instead of forcing one global cross this finds up to ``n_dirs`` orientation
    families. Walls are taken modulo 90 deg, because a wall and its
    perpendicular belong to the same family; support is the number of columns
    that observed them, so a long wall votes louder than a short one.

    Returns an empty list when nothing has enough support -- the caller should
    then simply not apply the prior rather than invent a direction.
    """
    if not walls:
        return []
    ang = np.array([math.atan2(w["n"][1], w["n"][0]) % (math.pi / 2) for w in walls])
    sup = np.array([float(sum(m["seg"].span for m in w["members"])) for w in walls])
    dirs: List[float] = []
    live = np.ones(len(ang), bool)
    tol = math.radians(tol_deg)
    for _ in range(max(n_dirs, 0)):
        if not live.any():
            break
        # circular mean over the quarter-turn, weighted by support
        best, best_w = None, -1.0
        for k in np.flatnonzero(live):
            d = np.abs((ang - ang[k] + math.pi / 4) % (math.pi / 2) - math.pi / 4)
            w = float(sup[(d < tol) & live].sum())
            if w > best_w:
                best, best_w = k, w
        if best is None or best_w < min_support:
            break
        d = np.abs((ang - ang[best] + math.pi / 4) % (math.pi / 2) - math.pi / 4)
        sel = (d < tol) & live
        off = (ang[sel] - ang[best] + math.pi / 4) % (math.pi / 2) - math.pi / 4
        dirs.append(float((ang[best] + np.average(off, weights=sup[sel])) % (math.pi / 2)))
        live &= ~sel
    return dirs


def _atlanta_pull(angles: np.ndarray, dirs: Sequence[float]) -> np.ndarray:
    """Signed angular distance from each wall angle to its nearest allowed one."""
    if not dirs:
        return np.zeros_like(angles)
    a = angles % (math.pi / 2)
    best = None
    for d in dirs:
        off = (a - d + math.pi / 4) % (math.pi / 2) - math.pi / 4
        best = off if best is None else np.where(np.abs(off) < np.abs(best), off, best)
    return best


# --------------------------------------------------------------------------- #
# bundle adjustment
# --------------------------------------------------------------------------- #
def wall_conditioning(walls: Sequence[dict], pano: Optional[str] = None) -> Tuple[int, float]:
    """(how many walls, how much their normals differ) for a pano or a scene.

    A wall line pins two degrees of freedom, but two *parallel* walls pin the
    same two: rotation and one translation axis, leaving sliding along them
    free. So what matters is not the count but the angular spread of the
    normals, taken modulo 180 deg because a line has no front.
    """
    sel = [w for w in walls
           if pano is None or any(m["seg"].pano == pano for m in w["members"])]
    if len(sel) < 2:
        return len(sel), 0.0
    a = np.array([math.atan2(w["n"][1], w["n"][0]) % math.pi for w in sel])
    d = np.abs(a[:, None] - a[None, :])
    d = np.minimum(d, math.pi - d)
    return len(sel), float(np.degrees(d.max()))


@dataclass
class BAResult:
    theta: Dict[str, float]
    c: Dict[str, np.ndarray]
    walls: List[dict]
    cost_before: float
    cost_after: float
    residual_deg_before: float
    residual_deg_after: float
    n_observations: int
    n_walls: int
    n_shared_walls: int
    iterations: int
    message: str = ""
    degenerate: bool = False
    normal_spread_deg: float = 0.0
    atlanta_deg: List[float] = field(default_factory=list)
    support: Dict[str, int] = field(default_factory=dict)        # observations per pano
    shared_walls: Dict[str, int] = field(default_factory=dict)   # shared walls per pano


def _residuals(params, panos, walls, obs, anchor_idx, anchor_pose, huber_deg, prior,
               atlanta=None):
    """Angular wall residual for every observation, plus the gauge anchor.

    The observed floor point is placed in the scene frame from the pose and the
    observed range, and its distance to the candidate wall line is converted to
    the boundary-angle error it corresponds to::

        e ~= (n . p_world - d) / (1 + r^2)      radians

    which is the first-order form of ``db/dr = -1/(1 + r^2)``. Writing it this
    way keeps the angular weighting that makes near walls count for more, while
    staying smooth everywhere -- an exact re-projection has to divide by
    ``n . direction``, which flips sign when a wall passes behind the camera and
    leaves a flat, gradientless penalty the optimiser happily walks into. It did:
    the first version drove the cameras 222 units away.
    """
    n_p = len(panos)
    th = params[:n_p]
    cc = params[n_p:3 * n_p].reshape(n_p, 2)
    wa = params[3 * n_p::2]
    wd = params[3 * n_p + 1::2]

    wi, pi = obs["wall_i"], obs["pano_i"]
    n = np.stack([np.cos(wa[wi]), np.sin(wa[wi])], axis=-1)
    d = wd[wi]
    t = th[pi]
    cos_t, sin_t = np.cos(t), np.sin(t)
    u = obs["dir"] * obs["r"][:, None]                 # observed point, pano frame
    px = cos_t * u[:, 0] - sin_t * u[:, 1] + cc[pi, 0]
    py = sin_t * u[:, 0] + cos_t * u[:, 1] + cc[pi, 1]
    e = np.degrees((n[:, 0] * px + n[:, 1] * py - d) / (1.0 + obs["r"] ** 2))

    a = np.abs(e)
    scaled = np.where(a <= huber_deg, e,
                      np.sign(e) * np.sqrt(np.maximum(2 * huber_deg * a - huber_deg ** 2, 0.0)))
    # A weak prior on every pose, towards where it started. Well-constrained
    # panoramas barely feel it; one that ends up with no observation at all --
    # every segment of it too short to survive the span guards -- is otherwise a
    # free parameter, and a free parameter in a least-squares problem wanders.
    # This one wandered 3465 units before the prior existed. Measured on the
    # synthetic scene: prior_pos 1.0 leaves position worse than it started
    # (0.249 -> 0.446), 3.0 holds it (0.239) while yaw still improves 6x.
    th0, c0_, w_yaw, w_pos = prior     # w_yaw / w_pos are per-panorama
    n_at = len(wa) if atlanta and atlanta[0] else 0
    res = np.empty(len(e) + 3 * n_p + n_at + 3)
    res[:len(e)] = scaled * obs["weight"]
    k = len(e)
    res[k:k + n_p] = np.asarray(w_yaw) * np.degrees(np.array(
        [float(wrap_pi(th[i] - th0[i])) for i in range(n_p)]))
    res[k + n_p:k + 3 * n_p] = (np.asarray(w_pos)[:, None] * (cc - c0_)).ravel()
    if n_at:
        # Atlanta: pull each wall towards the nearest dominant direction. Soft,
        # and weighted by how much evidence the wall has, so a well-observed
        # wall at a genuinely odd angle can refuse to snap.
        dirs, w_at, support = atlanta
        res[k + 3 * n_p:k + 3 * n_p + n_at] = (
            w_at * np.sqrt(support) * np.degrees(_atlanta_pull(wa, dirs)))
    res[-3] = 1e3 * float(wrap_pi(th[anchor_idx] - anchor_pose[0]))
    res[-2:] = 1e3 * (cc[anchor_idx] - anchor_pose[1])
    return res


def bundle_adjust(bon: Dict[str, np.ndarray], theta: Dict[str, float],
                  c: Dict[str, np.ndarray], n_cols: int,
                  rounds: int = 3, huber_deg: float = 3.0, min_members: int = 2,
                  max_dist: float = 15.0, min_seg_span: int = 30,
                  min_wall_span: int = 80, prior_yaw: float = 0.3,
                  prior_pos: float = 3.0, atlanta_dirs: int = 0,
                  atlanta_weight: float = 0.5, atlanta_tol_deg: float = 20.0,
                  min_walls: int = 2, min_normal_spread_deg: float = 20.0,
                  **fit_kw) -> BAResult:
    """Refine panorama poses and shared wall lines together.

    Only walls seen by at least ``min_members`` panoramas enter the problem: a
    wall one panorama can see constrains nothing about anybody's pose, and
    including it would let the optimiser explain away errors by moving walls.

    Short segments are dropped before association (``min_seg_span``), and a
    wall needs ``min_wall_span`` columns of total support to be believed. Both
    guards are load-bearing: a ten-column fragment of noise from one panorama
    and another from a second will happily associate into a wall that is not
    there, and the optimiser then drags a camera 2.5 units to satisfy it.
    """
    from scipy.optimize import least_squares

    panos = sorted(bon)
    idx = {p: i for i, p in enumerate(panos)}
    anchor = max(panos, key=lambda p: len(bon[p]))
    theta = dict(theta)
    c = {k: np.asarray(v, float).copy() for k, v in c.items()}
    anchor_pose = (theta[anchor], c[anchor].copy())

    th_grid = col_centers_theta(n_cols)
    dirs = dir_of_theta(th_grid)
    segments = {}
    for p in panos:
        segs = [s for s in fit_wall_segments(bon[p], max_dist=max_dist, **fit_kw)
                if s.span >= min_seg_span]
        for s in segs:
            s.pano = p
        segments[p] = segs

    degenerate = False
    frozen: set = set()
    cost_before = cost_after = float("nan")
    res_before = res_after = float("nan")
    walls: List[dict] = []
    n_obs = 0
    message = ""
    for rnd in range(rounds):
        walls = associate_walls(segments, theta, c)
        shared = [w for w in walls if len(w["members"]) >= min_members
                  and sum(m["seg"].span for m in w["members"]) >= min_wall_span]
        if not shared:
            message = "no wall was seen by two panoramas; nothing to bundle"
            break

        n_w_scene, spread = wall_conditioning(shared)
        if n_w_scene < min_walls or spread < 5.0:
            message = (f"degenerate: {n_w_scene} shared walls spanning {spread:.0f} deg "
                       f"of normal direction -- refusing to move the poses")
            degenerate = True
            break

        pano_i, wall_i, b_obs, dir_list, r_list, weight = [], [], [], [], [], []
        for wi, w in enumerate(shared):
            for m in w["members"]:
                s = m["seg"]
                r = bfloor_to_dist(bon[s.pano][s.cols])
                keep = (r > 1e-3) & (r < max_dist * 0.999)
                if keep.sum() < 3:
                    continue
                cols = s.cols[keep]
                pano_i.append(np.full(len(cols), idx[s.pano]))
                wall_i.append(np.full(len(cols), wi))
                b_obs.append(np.asarray(bon[s.pano])[cols])
                dir_list.append(dirs[cols])
                r_list.append(r[keep])
                weight.append(np.ones(keep.sum()))
        if not pano_i:
            message = "no usable observations after range filtering"
            break
        obs = {"pano_i": np.concatenate(pano_i), "wall_i": np.concatenate(wall_i),
               "b_obs": np.concatenate(b_obs), "dir": np.concatenate(dir_list),
               "r": np.concatenate(r_list), "weight": np.concatenate(weight)}
        n_obs = len(obs["pano_i"])
        support = {p: int((obs["pano_i"] == idx[p]).sum()) for p in panos}
        shared_per_pano = {p: 0 for p in panos}
        for w in shared:
            for m in w["members"]:
                shared_per_pano[m["seg"].pano] += 1

        p0 = np.concatenate([
            np.array([theta[p] for p in panos]),
            np.array([c[p] for p in panos]).ravel(),
            np.array([[math.atan2(w["n"][1], w["n"][0]), w["d"]] for w in shared]).ravel(),
        ])
        # Freeze any panorama whose own walls cannot constrain it. One wall, or
        # several parallel ones, leaves it free to slide; the optimiser then
        # trades a good pose for a slightly better fit to that one wall. On ZInD
        # this alone turned a 1.5 deg pose into a 10 deg one.
        w_yaw_v = np.full(len(panos), prior_yaw)
        w_pos_v = np.full(len(panos), prior_pos)
        # Graded, not binary. A single wall leaves a panorama free to slide
        # along it, so freeze it. Two parallel walls still pin rotation and one
        # axis, so only stiffen those -- freezing them outright throws away the
        # constraint that works (measured: yaw 6.9 -> 5.9 deg when frozen,
        # 6.9 -> 1.1 deg when merely stiffened).
        for i_p, p_ in enumerate(panos):
            n_w, sp = wall_conditioning(shared, p_)
            if n_w < 2:
                w_yaw_v[i_p] *= 200.0
                w_pos_v[i_p] *= 200.0
                frozen.add(p_)
            elif sp < min_normal_spread_deg:
                w_yaw_v[i_p] *= 10.0
                w_pos_v[i_p] *= 10.0
        prior = (np.array([theta[p] for p in panos]),
                 np.array([c[p] for p in panos]),
                 w_yaw_v, w_pos_v)
        atlanta = None
        if atlanta_dirs > 0:
            at_dirs = atlanta_directions(shared, atlanta_dirs, atlanta_tol_deg)
            at_support = np.array([float(sum(m["seg"].span for m in w["members"]))
                                   for w in shared])
            atlanta = (at_dirs, atlanta_weight, at_support) if at_dirs else None
        args = (panos, shared, obs, idx[anchor], anchor_pose, huber_deg, prior, atlanta)
        r0 = _residuals(p0, *args)
        n_e = len(obs["pano_i"])
        if rnd == 0:
            cost_before = float(0.5 * np.sum(r0[:n_e] ** 2))
            res_before = float(np.median(np.abs(r0[:n_e])))

        sol = least_squares(_residuals, p0, args=args, method="trf", loss="linear",
                            max_nfev=200 * (rnd + 1), x_scale="jac")
        message = sol.message
        n_p = len(panos)
        for p in panos:
            theta[p] = float(wrap_pi(sol.x[idx[p]]))
            c[p] = sol.x[n_p + 2 * idx[p]: n_p + 2 * idx[p] + 2].copy()
        for wi, w in enumerate(shared):
            a, d = sol.x[3 * n_p + 2 * wi], sol.x[3 * n_p + 2 * wi + 1]
            w["n"] = np.array([math.cos(a), math.sin(a)])
            w["d"] = float(d)
        cost_after = float(0.5 * np.sum(sol.fun[:n_e] ** 2))
        res_after = float(np.median(np.abs(sol.fun[:n_e])))

    shared = [w for w in walls if len(w["members"]) >= min_members
              and sum(m["seg"].span for m in w["members"]) >= min_wall_span]
    return BAResult(theta=theta, c=c, walls=shared, cost_before=cost_before,
                    cost_after=cost_after, residual_deg_before=res_before,
                    residual_deg_after=res_after, n_observations=n_obs,
                    n_walls=len(walls), n_shared_walls=len(shared),
                    iterations=rounds, message=message,
                    support=locals().get("support", {}),
                    shared_walls=locals().get("shared_per_pano", {}),
                    atlanta_deg=[round(math.degrees(d), 2)
                                 for d in (locals().get("atlanta") or ([],))[0]],
                    degenerate=degenerate,
                    normal_spread_deg=wall_conditioning(shared)[1])


def pose_outliers(theta0: Dict[str, float], c0: Dict[str, np.ndarray],
                  res: BAResult, t_thresh: float = 1.0,
                  yaw_thresh_deg: float = 20.0, min_shared: int = 1) -> List[dict]:
    """Panoramas worth doubting after the refinement.

    Two signals, and the second is the one that works. A large correction says
    the refinement disagreed with the input. But the failure this data actually
    has -- a pairwise pose flipped by 90 or 180 degrees -- does not produce a
    large correction: a flipped panorama's walls associate with nothing, so it
    has no observations, the prior holds it exactly where it was, and it moves
    zero. **Lack of wall support is what exposes it.**
    """
    out = []
    for p in res.theta:
        dt = float(np.linalg.norm(res.c[p] - np.asarray(c0[p], float)))
        dy = abs(math.degrees(float(wrap_pi(res.theta[p] - theta0[p]))))
        shared = res.shared_walls.get(p, 0)
        reasons = []
        if shared < min_shared:
            reasons.append(f"shares {shared} walls with the rest of the scene")
        if dt > t_thresh:
            reasons.append(f"moved {dt:.2f}")
        if dy > yaw_thresh_deg:
            reasons.append(f"rotated {dy:.1f} deg")
        if reasons:
            out.append({"pano": p, "moved": dt, "rotated_deg": dy,
                        "shared_walls": shared, "reasons": reasons})
    return sorted(out, key=lambda x: (x["shared_walls"], -x["moved"]))


def wall_polylines(res: BAResult, bon: Dict[str, np.ndarray], n_cols: int,
                   pad: float = 0.05) -> List[np.ndarray]:
    """Each shared wall as a drawable segment.

    The line itself is infinite; its extent comes from the columns that actually
    observed it, projected onto the wall direction in the scene frame.
    """
    dirs = dir_of_theta(col_centers_theta(n_cols))
    out = []
    for w in res.walls:
        n, d = np.asarray(w["n"], float), float(w["d"])
        along = np.array([-n[1], n[0]])
        ts = []
        for m in w["members"]:
            s = m["seg"]
            th, c = res.theta[s.pano], res.c[s.pano]
            R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
            r = bfloor_to_dist(np.asarray(bon[s.pano])[s.cols])
            pts = (R @ (dirs[s.cols] * r[:, None]).T).T + c
            ts += [float(along @ pts.min(0)), float(along @ pts.max(0))]
            ts += list(pts @ along)
        if not ts:
            continue
        lo, hi = min(ts) - pad, max(ts) + pad
        base = n * d
        out.append(np.stack([base + along * lo, base + along * hi]))
    return out
