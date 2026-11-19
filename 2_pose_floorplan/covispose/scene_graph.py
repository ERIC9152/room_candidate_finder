"""Assemble a floor plan from CoVisPose relative-pose predictions.

Input is a *scene* in the ``data/group/<scene>`` layout: a ``manifest.json``
listing panorama nodes and the edges between them (transcribed from the tour's
HOTSPOT links), plus the panoramas themselves.

    manifest -> pairwise relative poses (the network) -> global poses -> polygons

Conventions
-----------
The network predicts, for an ordered pair (A, B), the pose of A in B's frame::

    p_b = Rot(yaw_ab) p_a + t_ab

With a global pose (theta_k, c_k) per panorama mapping its local frame into the
scene frame (``p_world = Rot(theta_k) p_k + c_k``), that measurement becomes two
constraints, which is what :func:`global_poses` solves::

    theta_a - theta_b = yaw_ab
    c_a - c_b         = Rot(theta_b) t_ab

Units: the network's translations are in camera heights. This dataset puts the
camera at ``z0 = 50`` in world units (every ``layout/<pano>.json`` agrees), so
one camera height is 50 world units -- see ``UNITS_PER_CAMERA_HEIGHT``.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .geometry import bfloor_to_dist, col_centers_theta, dir_of_theta, wrap_pi

UNITS_PER_CAMERA_HEIGHT = 50.0


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
@dataclass
class Scene:
    scene_id: str
    scene_dir: str
    panos: Dict[str, str]                 # pano_id -> absolute image path
    edges: List[Tuple[str, str]]          # undirected, deduplicated
    room_idx: Dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.panos)


def load_scene(scene_dir: str) -> Scene:
    """Read ``manifest.json``. Handles both manifest variants in this dataset.

    Connectivity comes from ``edges_raw``: it is present in every manifest and
    is the direct transcription of the tour's hotspot links, whereas the
    per-node ``connections`` / ``neighbors`` keys differ between scenes.
    """
    scene_dir = os.path.abspath(scene_dir)
    with open(os.path.join(scene_dir, "manifest.json"), "r", encoding="utf-8") as f:
        m = json.load(f)

    panos, room_idx = {}, {}
    for n in m.get("nodes", []):
        pid = n["pano_id"]
        # Resolve against the scene directory: manifest paths are relative to a
        # repository root that need not be where the data now lives.
        p = os.path.join(scene_dir, "panos", pid + ".jpg")
        if os.path.isfile(p):
            panos[pid] = p
            room_idx[pid] = n.get("room_idx", -1)

    seen, edges = set(), []
    for e in m.get("edges_raw", []):
        a, b = e.get("src"), e.get("dst")
        if a == b or a not in panos or b not in panos:
            continue
        key = tuple(sorted((a, b)))
        if key not in seen:
            seen.add(key)
            edges.append((a, b))
    return Scene(m.get("scene_id", os.path.basename(scene_dir)), scene_dir,
                 panos, edges, room_idx)


# --------------------------------------------------------------------------- #
# global pose solve
# --------------------------------------------------------------------------- #
def _rot(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def spanning_tree(nodes: Sequence[str], edges: Sequence[dict]):
    """Greedy maximum-confidence spanning forest.

    Returns ``(used_edges, ignored_edges, component index per node)``. Edges are
    taken highest-confidence first, so the tree that seeds the solver leans on
    the pairs the network was most sure of.
    """
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    used, ignored = [], []
    for e in sorted(edges, key=lambda e: -e["confidence"]):
        ra, rb = find(e["a"]), find(e["b"])
        if ra == rb:
            ignored.append({"from": e["a"], "to": e["b"],
                            "reason": "loop_edge_not_in_tree"})
        else:
            parent[ra] = rb
            used.append(e)
    comp_id, comps = {}, {}
    for n in nodes:
        r = find(n)
        comps.setdefault(r, len(comps))
        comp_id[n] = comps[r]
    return used, ignored, comp_id


def _init_from_tree(nodes, used, comp_id):
    """Walk each component's tree for an initial (theta, c) per node."""
    adj: Dict[str, List] = {n: [] for n in nodes}
    for e in used:
        adj[e["a"]].append((e["b"], e, True))    # this node is the measurement's A
        adj[e["b"]].append((e["a"], e, False))

    theta = {n: 0.0 for n in nodes}
    c = {n: np.zeros(2) for n in nodes}
    roots: Dict[int, str] = {}
    seen = set()
    for n in sorted(nodes, key=lambda x: -len(adj[x])):
        if n in seen:
            continue
        roots.setdefault(comp_id[n], n)
        stack = [n]
        seen.add(n)
        while stack:
            u = stack.pop()
            for v, e, u_is_a in adj[u]:
                if v in seen:
                    continue
                if u_is_a:   # u == A, v == B
                    theta[v] = float(wrap_pi(theta[u] - e["yaw"]))
                    c[v] = c[u] - _rot(theta[v]) @ e["t"]
                else:        # u == B, v == A
                    theta[v] = float(wrap_pi(theta[u] + e["yaw"]))
                    c[v] = c[u] + _rot(theta[u]) @ e["t"]
                seen.add(v)
                stack.append(v)
    return theta, c, roots


def global_poses(nodes: Sequence[str], edges: Sequence[dict], refine: bool = True,
                 iters: int = 30):
    """Relative measurements -> one global pose per panorama.

    A maximum-confidence spanning tree gives the initial guess; ``refine`` then
    runs weighted least squares over **all** edges, including the loop closures
    the tree had to drop. Rotations are solved first (iteratively, because the
    residual wraps), then translations in closed form given those rotations.
    Each connected component is solved in its own gauge, root at the origin.
    """
    nodes = list(nodes)
    used, ignored, comp_id = spanning_tree(nodes, edges)
    theta, c, roots = _init_from_tree(nodes, used, comp_id)
    if not refine or not edges:
        return theta, c, used, ignored, comp_id, roots

    idx = {n: i for i, n in enumerate(nodes)}
    n_n = len(nodes)
    root_rows = [idx[r] for r in roots.values()]

    # ---- rotations: Gauss-Newton on wrapped residuals -------------------- #
    for _ in range(iters):
        A = np.zeros((len(edges) + len(root_rows), n_n))
        r = np.zeros(len(edges) + len(root_rows))
        for k, e in enumerate(edges):
            ia, ib, w = idx[e["a"]], idx[e["b"]], math.sqrt(e["confidence"])
            A[k, ia], A[k, ib] = w, -w
            r[k] = w * float(wrap_pi(e["yaw"] - (theta[e["a"]] - theta[e["b"]])))
        for j, i in enumerate(root_rows):          # gauge: root angle pinned
            A[len(edges) + j, i] = 1e3
        d = np.linalg.lstsq(A, r, rcond=None)[0]
        for n in nodes:
            theta[n] = float(wrap_pi(theta[n] + d[idx[n]]))
        if np.abs(d).max() < 1e-9:
            break

    # ---- translations: linear least squares given those rotations -------- #
    rows = 2 * len(edges) + 2 * len(root_rows)
    A = np.zeros((rows, 2 * n_n))
    b = np.zeros(rows)
    for k, e in enumerate(edges):
        ia, ib, w = idx[e["a"]], idx[e["b"]], math.sqrt(e["confidence"])
        for d_ in (0, 1):
            A[2 * k + d_, 2 * ia + d_] = w
            A[2 * k + d_, 2 * ib + d_] = -w
        b[2 * k:2 * k + 2] = w * (_rot(theta[e["b"]]) @ e["t"])
    for j, i in enumerate(root_rows):              # gauge: root position pinned
        A[2 * len(edges) + 2 * j, 2 * i] = 1e3
        A[2 * len(edges) + 2 * j + 1, 2 * i + 1] = 1e3
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    c = {n: sol[2 * idx[n]:2 * idx[n] + 2] for n in nodes}
    return theta, c, used, ignored, comp_id, roots


def edge_residuals(edges: Sequence[dict], theta: Dict[str, float],
                   c: Dict[str, np.ndarray]) -> List[dict]:
    """Per-edge disagreement between a measurement and the solved global poses.

    A large residual on an edge that the tree did not use is a loop that did not
    close; it is the honest place to look for a wrong pair.
    """
    out = []
    for e in edges:
        dyaw = float(wrap_pi(e["yaw"] - (theta[e["a"]] - theta[e["b"]])))
        dt = (c[e["a"]] - c[e["b"]]) - _rot(theta[e["b"]]) @ e["t"]
        out.append({"a": e["a"], "b": e["b"],
                    "yaw_residual_deg": math.degrees(dyaw),
                    "t_residual": float(np.linalg.norm(dt)),
                    "confidence": e["confidence"]})
    return out


# --------------------------------------------------------------------------- #
# polygons
# --------------------------------------------------------------------------- #
def boundary_to_polygon(bon: np.ndarray, max_dist: float = 20.0) -> np.ndarray:
    """Per-column floor boundary (HorizonNet ``bon``) -> local 2D room polygon.

    Returns (n_cols, 2) in camera heights. Columns whose implied distance runs
    past ``max_dist`` are clamped rather than dropped, so the ring stays closed.
    """
    r = np.clip(bfloor_to_dist(np.asarray(bon, dtype=np.float64)), 1e-3, max_dist)
    return dir_of_theta(col_centers_theta(len(r))) * r[:, None]


def simplify_polygon(poly: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker on a closed ring; uniform decimation if shapely is absent."""
    try:
        from shapely.geometry import Polygon
        p = Polygon(poly).simplify(tol, preserve_topology=True)
        if p.is_empty:
            raise ValueError("empty after simplify")
        xy = np.asarray(p.exterior.coords)[:-1]
        return xy if len(xy) >= 3 else np.asarray(poly)
    except Exception:  # noqa: BLE001
        step = max(len(poly) // 64, 1)
        return np.asarray(poly)[::step]


def to_world(poly: np.ndarray, theta: float, c: np.ndarray) -> np.ndarray:
    return (_rot(theta) @ np.asarray(poly).T).T + np.asarray(c)


def transform_matrix(theta: float, c: np.ndarray) -> List[List[float]]:
    """3x3 homogeneous local->world transform, matching ``world_transforms``."""
    r = _rot(theta)
    return [[float(r[0, 0]), float(r[0, 1]), float(c[0])],
            [float(r[1, 0]), float(r[1, 1]), float(c[1])],
            [0.0, 0.0, 1.0]]


# --------------------------------------------------------------------------- #
# comparison against whatever the existing pipeline produced
# --------------------------------------------------------------------------- #
def align_rigid(src: np.ndarray, dst: np.ndarray, allow_mirror: bool = True):
    """Best rigid (optionally mirrored) alignment of ``src`` onto ``dst``.

    Lets us compare our camera layout against the existing pipeline's without
    assuming the two share a gauge. Scale is *not* fitted -- both sides are in
    the same world units, so a scale error must show up as residual. Because a
    mirror is allowed and reported, this also answers whether our azimuth
    convention matches theirs.
    """
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    best = None
    for mirror in ((False, True) if allow_mirror else (False,)):
        s = src * np.array([-1.0, 1.0]) if mirror else src
        sc, dc = s.mean(0), dst.mean(0)
        s0, d0 = s - sc, dst - dc
        cross = float((s0[:, 0] * d0[:, 1] - s0[:, 1] * d0[:, 0]).sum())
        dot = float((s0 * d0).sum())
        a = math.atan2(cross, dot)
        rms = float(np.sqrt((((_rot(a) @ s0.T).T - d0) ** 2).sum(1).mean()))
        if best is None or rms < best["rms"]:
            best = {"rms": rms, "yaw_deg": math.degrees(a), "mirror": mirror,
                    "t": (dc - _rot(a) @ sc).tolist()}
    return best
