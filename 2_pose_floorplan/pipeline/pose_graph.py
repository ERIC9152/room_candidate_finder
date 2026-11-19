"""Step 3 in the docstring numbering, step 4 in the README (CPU, default in Run_Pipeline.ps1): pose-graph optimisation of the CoVisPose pair poses.

  python -m pipeline.pose_graph --run runs\\<name>          -> runs\\<name>\\poses_<time>\\  (same format as pipeline.poses)
  python -m pipeline.pose_graph --run runs\\<name> --eval   -> leave-one-pair-out comparison, spanning tree vs pose graph

Same pair acceptance (G rule + direction links) and the same maximum-weight spanning forest as pipeline.poses; the
forest is then only the starting point. Inside every connected component that has more accepted pairs than a tree
needs (= at least one loop), all accepted pairs are fitted together by robust least squares:

  residual of pair (a, b), predicted p_b = R(yaw) p_a + t, from the camera poses (yaw_u, t_u):
      yaw:  wrap((yaw_a - yaw_b) - yaw)                / SIGMA_YAW
      t:    (R(-yaw_b) (t_a - t_b) - t)                / SIGMA_T (full pair) or SIGMA_T_DIR (direction-only pair)
  each scaled by sqrt(weight), weight = 1 + f (full) or f (direction-only), the forest's weights;
  Huber loss, switching to linear at F_SCALE sigmas, so one bad pair cannot pull a whole loop.

The pair poses are the decoder poses (4-view TTA), the estimator the pair acceptance is built on; on ZInD test the
TTA decoder had yaw median 0.754 deg / t median 0.058 camera heights (training/PHASE5_NOTES.md), which sets
SIGMA_YAW / SIGMA_T. Components that are trees (no extra pair) are returned unchanged: there the forest already fits
every pair exactly. The gauge (forest root at yaw 0, t 0) is kept fixed; output roots are chosen as in pipeline.poses.

Camera flags (W, L, D, U) are computed exactly as in pipeline.poses (from the raw pair predictions); their error
rates for these optimised poses were measured on ZInD test (FLAG_STATS_GRAPH, written into every group file as
flag_stats). W still separates wrong cameras well; L much less than before, because the optimisation repairs most of
the loops it flags. `loops` lists the non-tree pairs' disagreement with the optimised poses. pose_graph_report.json
holds, per group, every accepted pair's disagreement with the forest poses and with the optimised poses.

ZInD test (scripts/eval_pose_graph_zind.py; 219 complete rooms with >= 3 panoramas, 1087 panoramas, 2294 posed camera
pairs; settings above fixed before the test): accuracy (0.3, 5 deg) 68.7% -> 76.2%, yaw mean 4.54 -> 3.36 deg,
t mean 0.297 -> 0.249, wrong cameras 7.05% -> 6.10%; on the 183 rooms with a loop the bootstrap 95% intervals of the
differences exclude 0 (README section 11.3).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import common as C  # noqa: E402
from pipeline import poses as P  # noqa: E402

SIGMA_YAW = math.radians(1.0)   # decoder yaw median 0.754 deg (ZInD test, TTA)
SIGMA_T = 0.06                  # decoder t median 0.058 camera heights
SIGMA_T_DIR = 0.60              # direction-only pairs: position may be off by ~0.6 camera heights
F_SCALE = 3.0                   # Huber threshold in sigmas (3 deg / 0.18 for a weight-1 pair)
RULE = ('G rule + direction links f=0.30; maximum-weight spanning forest as initial value; '
        'pose-graph optimisation of all accepted pairs per component (robust least squares, Huber)')
FLAG_STATS_GRAPH = dict(
    source='pose_graph_zind_test_v1: ZInD test, 219 complete rooms with >= 3 panoramas (158 houses), pose-graph poses; '
           'camera wrong = majority of its pairs > 10 deg or > 0.5 (scripts/eval_pose_graph_zind.py, 2026-11-19)',
    valid=dict(D=False, W=True, L=True, U=False),
    W=dict(flagged=0.2138, unflagged=0.0365, n=145), L=dict(flagged=0.0709, unflagged=0.0546, n=409))
_REPORT = []


def _accepted(edges, cls):
    out = []
    for e in edges:
        k, _, f = cls[e['id']]
        if k:
            out.append((e, k, f))
    return out


def optimise(nodes, edges, cls, p1):
    """p1 = forest poses {u: (yaw, t, forest_root)}. Returns new poses (same form) and per-component info."""
    from scipy.optimize import least_squares
    acc = _accepted(edges, cls)
    comps = {}
    for u in nodes:
        comps.setdefault(p1[u][2], []).append(u)
    out = {u: (p1[u][0], np.asarray(p1[u][1], float), p1[u][2]) for u in nodes}
    info = []
    for r, comp in comps.items():
        cs = set(comp)
        ce = [(e, k, f) for e, k, f in acc if e['a'] in cs]
        if len(comp) < 3 or len(ce) <= len(comp) - 1:
            continue                                   # a tree: the forest already fits every pair exactly
        free = sorted(u for u in comp if u != r)
        ix = {u: i for i, u in enumerate(free)}
        x0 = np.concatenate([[p1[u][0], *np.asarray(p1[u][1], float)] for u in free])
        E = [(e['a'], e['b'], float(e['yaw']), np.asarray(e['t'], float), math.sqrt(1 + f if k == 'full' else f),
              SIGMA_T if k == 'full' else SIGMA_T_DIR) for e, k, f in ce]

        def pose(x, u):
            if u == r:
                return 0., np.zeros(2)
            i = 3 * ix[u]
            return x[i], x[i + 1:i + 3]

        def res(x):
            out_r = []
            for a, b, y, t, w, st in E:
                ya, ta = pose(x, a)
                yb, tb = pose(x, b)
                dy = C.wrap((ya - yb) - y)
                dt = C.rot(-yb) @ (ta - tb) - t
                out_r += [w * dy / SIGMA_YAW, w * dt[0] / st, w * dt[1] / st]
            return np.asarray(out_r)

        sol = least_squares(res, x0, loss='huber', f_scale=F_SCALE, method='trf', max_nfev=2000)
        ok = bool(sol.success and np.isfinite(sol.x).all())
        if ok:
            for u in free:
                y, t = pose(sol.x, u)
                out[u] = (float(C.wrap(y)), np.array(t, float), r)
        info.append(dict(root=r, cameras=len(comp), pairs=len(ce), optimised=ok, nfev=int(sol.nfev),
                         cost_initial=float(0.5 * np.sum(res(x0) ** 2)), cost_final=float(sol.cost) if ok else None,
                         message=str(sol.message)))
    return out, info


def _pair_errors(poses, edges, cls):
    rows = []
    for e, k, f in _accepted(edges, cls):
        if poses[e['a']][2] != poses[e['b']][2]:
            continue
        dy, dt = C.err(P.rel(poses, e['a'], e['b']), (e['yaw'], e['t']))
        rows.append(dict(pair_id=e['id'], kind=k, yaw_deg=round(float(dy), 3), translation=round(float(dt), 4)))
    return rows


def solve_graph(nodes, edges):
    cls, p1, tree, _, roots, flags = P.solve(nodes, edges)
    p2, info = optimise(nodes, edges, cls, p1)
    loops = []
    for e, _, _ in _accepted(edges, cls):
        if e['id'] not in tree:
            dy, dt = C.err(P.rel(p2, e['a'], e['b']), (e['yaw'], e['t']))
            loops.append(dict(id=e['id'], yaw_deg=dy, translation=dt))
    _REPORT.append(dict(nodes=nodes, components=info, pairs_forest=_pair_errors(p1, edges, cls),
                        pairs_pose_graph=_pair_errors(p2, edges, cls)))
    return cls, p2, tree, loops, roots, flags


def run_pose_graph(run):
    _REPORT.clear()
    dest = P.poses(run, solver=solve_graph, rule=RULE, flag_stats=FLAG_STATS_GRAPH, method='pose_graph')
    m = C.read(C.local(run) / 'manifest.json')
    rep = []
    for g, r in zip(m['groups'], _REPORT):
        rep.append(dict(group=g['room_id'], **{k: v for k, v in r.items() if k != 'nodes'}))
    C.write(dest / 'pose_graph_report.json', dict(
        method='pose_graph', sigma_yaw_deg=math.degrees(SIGMA_YAW), sigma_t=SIGMA_T, sigma_t_direction=SIGMA_T_DIR,
        huber_f_scale=F_SCALE, note='pairs_*: each accepted pair vs the forest poses / the optimised poses (deg, camera heights)',
        groups=rep))
    s = C.read(dest / 'summary.json')
    s['components_optimised'] = sum(1 for r in rep for c in r['components'] if c['optimised'])
    C.write(dest / 'summary.json', s)
    print('POSE GRAPH: optimised components', s['components_optimised'])
    return dest


# ------------------------------------------------------------------ evaluation (no ground truth: leave one pair out)
def evaluate(run):
    """Every full pair that lies on a loop is held out in turn; the poses are solved from the remaining pairs (forest,
    and pose graph) and the held-out pair is predicted from them. Compared with the held-out pair's own decoder pose
    and its geometric-branch pose (an estimate the solve never sees)."""
    run = C.local(run)
    m = C.read(run / 'manifest.json')
    rows = []
    for g in m['groups']:
        if len(g['panorama_uuids']) < 3:
            continue
        edges, _ = P.load_group_edges(run, m, g)
        nodes = sorted(g['panorama_uuids'])
        cls = {e['id']: P.classify(e) for e in edges}
        for e in edges:
            if cls[e['id']][0] != 'full':
                continue
            rest = [x for x in edges if x['id'] != e['id']]
            c2, p1, _, _, _, _ = P.solve(nodes, rest)
            if p1[e['a']][2] != p1[e['b']][2]:
                continue                                       # not on a loop: nothing to predict it from
            p2, _ = optimise(nodes, rest, c2, p1)
            row = dict(group=g['room_id'], pair_id=e['id'])
            for name, ps in (('forest', p1), ('graph', p2)):
                pr = P.rel(ps, e['a'], e['b'])
                for ref, key in (('decoder', ('yaw', 't')), ('geometric', ('geo_yaw', 'geo_t'))):
                    dy, dt = C.err(pr, (e[key[0]], e[key[1]]))
                    row[f'{name}_vs_{ref}_yaw_deg'] = float(dy)
                    row[f'{name}_vs_{ref}_t'] = float(dt)
            rows.append(row)

    def stats(name, ref):
        y = np.array([r[f'{name}_vs_{ref}_yaw_deg'] for r in rows])
        t = np.array([r[f'{name}_vs_{ref}_t'] for r in rows])
        return dict(yaw_median=round(float(np.median(y)), 3), yaw_mean=round(float(y.mean()), 3),
                    t_median=round(float(np.median(t)), 4), t_mean=round(float(t.mean()), 4),
                    within_2deg_0p10=round(float(np.mean((y <= 2) & (t <= .10))), 3),
                    within_5deg_0p30=round(float(np.mean((y <= 5) & (t <= .30))), 3))
    out = dict(held_out_pairs=len(rows), groups=len({r['group'] for r in rows}),
               summary={f'{n}_vs_{ref}': stats(n, ref) for ref in ('decoder', 'geometric') for n in ('forest', 'graph')},
               better_vs_decoder=dict(
                   graph_better=sum(1 for r in rows if r['graph_vs_decoder_t'] < r['forest_vs_decoder_t'] - 1e-6),
                   forest_better=sum(1 for r in rows if r['forest_vs_decoder_t'] < r['graph_vs_decoder_t'] - 1e-6)),
               pairs=rows)
    dest = run / f'pose_graph_eval_{C.stamp()}.json'
    C.write(dest, out)
    print(json.dumps({k: v for k, v in out.items() if k != 'pairs'}, indent=1))
    print('EVAL: ' + str(dest))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--eval', action='store_true', help='leave-one-pair-out comparison only (writes no poses folder)')
    a = ap.parse_args()
    evaluate(a.run) if a.eval else run_pose_graph(a.run)
