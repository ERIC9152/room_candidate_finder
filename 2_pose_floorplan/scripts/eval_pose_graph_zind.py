"""Spanning forest (pipeline.poses) vs pose graph (pipeline.pose_graph), scored against ZInD ground truth.

  python scripts/eval_pose_graph_zind.py --run runs\\zind_test --gt evaluation\\zind_test_posegraph\\gt.json

Needs a run whose CoVisPose step is complete (prepare + manual_room_groups_v2.py run on the groups written by
scripts/make_zind_room_groups.py). CPU only. Both methods use the same accepted pairs, so they connect exactly the
same cameras; they differ only inside components with a loop. For every two cameras in the same component the
predicted relative pose p_v = R(yaw) p_u + t is compared with the ZInD one (t in camera heights of v).

Reported (overall and for components with a loop): yaw / translation errors, accuracy at (t, yaw) thresholds, the
share of "wrong" cameras (majority of the camera's pairs off by > 10 deg or > 0.5, the definition behind the
pose-flag statistics), whether the W / L flags still pick out wrong cameras, and a bootstrap over groups for the
graph - forest difference. Writes <run>\\pose_graph_zind_eval_<time>.json.
"""
import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from covispose.geometry import PanoPose, relative_pose  # noqa: E402
from pipeline import common as C  # noqa: E402
from pipeline import pose_graph as G  # noqa: E402
from pipeline import poses as P  # noqa: E402

THRESH = [(0.1, 1.0), (0.3, 5.0), (0.5, 10.0), (1.0, 20.0)]   # (camera heights, deg)
WRONG = (0.5, 10.0)


def gt_rel(gt, u, v):
    pose = lambda d: PanoPose(d['translation'], d['rotation'], d['scale'])
    t, y, _ = relative_pose(pose(gt[u]), pose(gt[v]))
    return y, t


def stats(rows, m):
    if not rows:
        return dict(n=0)
    y = np.array([r[m + '_yaw'] for r in rows])
    t = np.array([r[m + '_t'] for r in rows])
    out = dict(n=len(rows), yaw_median=round(float(np.median(y)), 3), yaw_mean=round(float(y.mean()), 3),
               t_median=round(float(np.median(t)), 4), t_mean=round(float(t.mean()), 4))
    for tt, yy in THRESH:
        out[f'acc_{tt}_{yy:g}deg'] = round(float(np.mean((t <= tt) & (y <= yy))), 4)
    return out


def camera_wrong(rows, m):
    per = defaultdict(list)
    for r in rows:
        bad = r[m + '_t'] > WRONG[0] or r[m + '_yaw'] > WRONG[1]
        per[(r['group'], r['u'])].append(bad)
        per[(r['group'], r['v'])].append(bad)
    return {k: sum(v) > len(v) / 2 for k, v in per.items()}


def evaluate(run, gt_path, boot=2000, seed=0):
    run = C.local(run)
    m = C.read(run / 'manifest.json')
    gt = C.read(gt_path)['panos']
    st = C.read(run / 'inference_status.json')
    if st.get('pending_or_error') or st.get('complete') != len(m['pairs']):
        raise SystemExit('CoVisPose step not complete for ' + str(run))
    rows, edges_rows, flags, per_group = [], [], {}, []
    for g in m['groups']:
        edges, _ = P.load_group_edges(run, m, g)
        nodes = sorted(g['panorama_uuids'])
        cls, p1, tree, _, _, fl = P.solve(nodes, edges)
        p2, info = G.optimise(nodes, edges, cls, p1)
        looped = {c['root'] for c in info}
        for u, f in fl.items():
            flags[(g['room_id'], u)] = f
        for e in edges:
            k = cls[e['id']][0]
            if k:
                gy, gtt = gt_rel(gt, e['a'], e['b'])
                dy, dt = C.err((e['yaw'], e['t']), (gy, gtt))
                edges_rows.append(dict(kind=k, yaw=dy, t=dt))
        grows = []
        for u, v in itertools.combinations(nodes, 2):
            if p1[u][2] != p1[v][2]:
                continue
            gy, gtt = gt_rel(gt, u, v)
            r = dict(group=g['room_id'], u=u, v=v, loop=p1[u][2] in looped)
            for name, ps in (('forest', p1), ('graph', p2)):
                r[name + '_yaw'], r[name + '_t'] = C.err(P.rel(ps, u, v), (gy, gtt))
            grows.append(r)
        rows += grows
        per_group.append(dict(group=g['room_id'], cameras=len(nodes), camera_pairs=len(nodes) * (len(nodes) - 1) // 2,
                              posed_pairs=len(grows), loop=bool(info), rows=grows))
    loop_rows = [r for r in rows if r['loop']]
    out = dict(run=run.name, groups=len(m['groups']), cameras=len(m['images']),
               camera_pairs=sum(g['camera_pairs'] for g in per_group), camera_pairs_posed=len(rows),
               groups_with_loop=sum(g['loop'] for g in per_group),
               accepted_pair_poses=dict(full=stats([dict(d_yaw=r['yaw'], d_t=r['t']) for r in edges_rows if r['kind'] == 'full'], 'd'),
                                        direction=stats([dict(d_yaw=r['yaw'], d_t=r['t']) for r in edges_rows if r['kind'] == 'dir'], 'd')),
               all={k: stats(rows, k) for k in ('forest', 'graph')},
               loop_components={k: stats(loop_rows, k) for k in ('forest', 'graph')})
    for name in ('forest', 'graph'):
        w = camera_wrong(rows, name)
        out[f'wrong_cameras_{name}'] = dict(n=len(w), wrong=sum(w.values()), share=round(sum(w.values()) / max(len(w), 1), 4))
        fs = {}
        for key in ('W', 'L', 'D'):
            fl_ = [w[k] for k in w if flags.get(k, {}).get(key)]
            un = [w[k] for k in w if not flags.get(k, {}).get(key)]
            fs[key] = dict(flagged=len(fl_), wrong_if_flagged=round(float(np.mean(fl_)), 4) if fl_ else None,
                           wrong_if_unflagged=round(float(np.mean(un)), 4) if un else None)
        out[f'flags_{name}'] = fs
    # paired bootstrap over groups that have a loop (the only ones where the two methods can differ)
    lg = [g for g in per_group if g['loop'] and g['rows']]
    if lg:
        rng = np.random.default_rng(seed)
        def metric(gs, mth):
            rr = [r for g in gs for r in g['rows']]
            t = np.array([r[mth + '_t'] for r in rr]); y = np.array([r[mth + '_yaw'] for r in rr])
            return t.mean(), y.mean(), np.mean((t <= .3) & (y <= 5)), np.mean((t <= .1) & (y <= 1))
        d = []
        for _ in range(boot):
            s = [lg[i] for i in rng.integers(0, len(lg), len(lg))]
            d.append(np.subtract(metric(s, 'graph'), metric(s, 'forest')))
        d = np.array(d)
        point = np.subtract(metric(lg, 'graph'), metric(lg, 'forest'))
        out['graph_minus_forest_loop_groups'] = {
            name: dict(diff=round(float(point[i]), 4), ci95=[round(float(x), 4) for x in np.percentile(d[:, i], [2.5, 97.5])])
            for i, name in enumerate(('t_mean', 'yaw_mean_deg', 'acc_0.3_5deg', 'acc_0.1_1deg'))}
        out['groups_better'] = dict(
            graph=sum(1 for g in lg if metric([g], 'graph')[0] < metric([g], 'forest')[0] - 1e-6),
            forest=sum(1 for g in lg if metric([g], 'forest')[0] < metric([g], 'graph')[0] - 1e-6))
    out['settings'] = dict(sigma_yaw_deg=math.degrees(G.SIGMA_YAW), sigma_t=G.SIGMA_T, sigma_t_direction=G.SIGMA_T_DIR,
                           huber_f_scale=G.F_SCALE)
    out['camera_pair_errors'] = rows
    dest = run / f'pose_graph_zind_eval_{C.stamp()}.json'
    C.write(dest, out)
    show = {k: v for k, v in out.items() if k != 'camera_pair_errors'}
    print(json.dumps(show, indent=1, ensure_ascii=False))
    print('EVAL: ' + str(dest))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--gt', required=True)
    a = ap.parse_args()
    evaluate(a.run, a.gt)
