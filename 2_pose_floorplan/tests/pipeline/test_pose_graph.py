"""Checks for pipeline.pose_graph on synthetic room groups (CPU, no data needed).

Run with:  python -m pytest tests/pipeline  (or: python tests/pipeline/test_pose_graph.py)

Each synthetic pair carries floor-boundary correspondences P (camera a) / Q (camera b) that agree with the pair's
own predicted pose, so the pair-acceptance rule of pipeline.poses accepts it, as it does for real predictions.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pipeline import common as C  # noqa: E402
from pipeline import pose_graph as G  # noqa: E402
from pipeline import poses as P  # noqa: E402


def make_group(n, pairs, noise_yaw_deg=1.0, noise_t=0.06, outliers=(), seed=0):
    rng = np.random.default_rng(seed)
    gt = {f'c{i}': (float(rng.uniform(-math.pi, math.pi)), rng.uniform(-3, 3, 2), 'c0') for i in range(n)}
    edges = []
    for k, (i, j) in enumerate(pairs):
        a, b = f'c{i}', f'c{j}'
        y, t = P.rel(gt, a, b)
        y = y + math.radians(rng.normal(0, noise_yaw_deg)) + (math.radians(25) if k in outliers else 0)
        t = t + rng.normal(0, noise_t, 2)
        pts = rng.uniform(-3, 3, (200, 2))
        edges.append(dict(id=f'e{k}', a=a, b=b, P=pts, Q=pts @ C.rot(y).T + t, yaw=y, t=t, geo_yaw=y, geo_t=t,
                          duplicate=False))
    return sorted(gt), edges, gt


def cam_errors(poses, gt, nodes):
    out = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            out.append(C.err(P.rel(poses, a, b), P.rel(gt, a, b)))
    return np.array(out)


def test_tree_components_unchanged():
    nodes, edges, _ = make_group(4, [(0, 1), (1, 2), (2, 3)])
    cls, p1, tree, _, _, _ = P.solve(nodes, edges)
    p2, info = G.optimise(nodes, edges, cls, p1)
    assert info == []
    for u in nodes:
        assert p2[u][0] == p1[u][0] and np.array_equal(p2[u][1], p1[u][1])


def test_loops_reduce_error():
    full = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    better = 0
    for seed in range(10):
        nodes, edges, gt = make_group(6, full, seed=seed)
        cls, p1, _, _, _, _ = P.solve(nodes, edges)
        p2, info = G.optimise(nodes, edges, cls, p1)
        assert info and info[0]['optimised']
        better += cam_errors(p2, gt, nodes)[:, 1].mean() < cam_errors(p1, gt, nodes)[:, 1].mean()
    assert better >= 8


def test_outlier_pair_does_not_pull_the_loop():
    full = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    nodes, edges, gt = make_group(5, full, noise_yaw_deg=0.3, noise_t=0.02, outliers=(3,), seed=3)
    cls, p1, _, _, _, _ = P.solve(nodes, edges)
    assert cls['e3'][0] == 'full'                       # the wrong pair is accepted, as real wrong pairs can be
    p2, _ = G.optimise(nodes, edges, cls, p1)
    e = cam_errors(p2, gt, nodes)
    assert e[:, 0].max() < 2.0 and e[:, 1].max() < 0.15


def test_solve_graph_keeps_flags_and_tree():
    full = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    nodes, edges, _ = make_group(5, full, seed=1)
    a = P.solve(nodes, edges)
    b = G.solve_graph(nodes, edges)
    assert a[2] == b[2] and a[4] == b[4] and a[5] == b[5]   # tree, roots, flags
    assert len(b[3]) == len(edges) - (len(nodes) - 1)         # every non-tree accepted pair is reported


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f()
            print('ok', name)
