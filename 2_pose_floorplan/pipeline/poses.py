"""Step 3 (CPU): CoVisPose predictions -> camera poses per room group (spanning forest only).

Run_Pipeline.ps1 runs pipeline.pose_graph instead, which uses the same rules and this forest as its starting point and
then optimises every component that has a loop; this module is kept as the -NoPoseGraph option.

  python -m pipeline.poses --run runs\\<name>          -> runs\\<name>\\poses_<time>\\

Rules (validated earlier on ZInD; numbers frozen here):
  * pair accepted ("full")     : >= 20 inlier correspondences (residual <= 0.30), inlier fraction >= 0.40, and the
                                 decoder pose agrees with the geometric-branch pose (<= 5 deg, <= 0.25)   [G rule]
  * direction-only link        : same, but inlier fraction in [0.30, 0.40); rotation is reliable, position may be
                                 off by ~0.6 camera heights                                           [direction links]
  * maximum-weight spanning forest (weights 1+f for full, f for direction links), no global optimisation;
  * display root of each component = camera with the most accepted pairs (ties: summed inlier fraction, then id);
  * pose flags: W weak edge on the path to the root (f < 0.50, or decoder/geometric disagreement > 2.5 deg or > 0.125),
                L camera on a loop whose residual is > 5 deg or > 0.30. On ZInD test, W-flagged cameras were wrong
                26.5% of the time and L-flagged 14.9% (unflagged 1.6-3.7%); D = position via a direction link.
Convention: p_B = R(yaw) p_A + t (native CoVisPose XY); camera pose p_root = R(yaw_u) p_u + t_u; camera height = 1.
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import common as C  # noqa: E402

RES, MIN_IN, MIN_FRAC, F_LOW = .30, 20, .40, .30
THETA, TAU = 5., .25
W_FRAC, W_YAW, W_T = .50, 2.5, .125
L_YAW, L_T = 5., .30
FLAG_STATS = dict(source='pose_flags_v1, ZInD test (40 houses); camera wrong = majority of its pairs > 10 deg or > 0.5',
                  valid=dict(D=False, W=True, L=True, U=False),
                  W=dict(flagged=0.2647, unflagged=0.0370, n=34), L=dict(flagged=0.1489, unflagged=0.0155, n=94))


def classify(e):
    P, Q = e['P'], e['Q']
    if len(P) == 0:
        return None, 0, 0.
    r = np.linalg.norm(P @ C.rot(e['yaw']).T + np.asarray(e['t']) - Q, axis=1)
    n = int((r <= RES).sum())
    f = n / len(r)
    dy, dt = C.err((e['yaw'], e['t']), (e['geo_yaw'], e['geo_t']))
    agree = dy <= THETA and dt <= TAU
    if n >= MIN_IN and agree and f >= MIN_FRAC:
        return 'full', n, f
    if n >= MIN_IN and agree and F_LOW <= f < MIN_FRAC:
        return 'dir', n, f
    return None, n, f


def inv(y, t):
    return -y, -C.rot(-y) @ np.asarray(t, float)


def rel(poses, a, b):
    ya, ta, _ = poses[a]
    yb, tb, _ = poses[b]
    return C.wrap(ya - yb), C.rot(-yb) @ (ta - tb)


def forest(nodes, edges, acc):
    """acc: {edge id: weight}. Returns poses {u: (yaw, t, root)}, tree ids, loops."""
    order = sorted((e for e in edges if e['id'] in acc), key=lambda e: (-acc[e['id']], e['id']))
    parent = {u: u for u in nodes}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    adj, tree = defaultdict(list), set()
    for e in order:
        a, b = e['a'], e['b']
        if find(a) == find(b):
            continue
        parent[find(a)] = find(b)
        tree.add(e['id'])
        adj[a].append((b, e['yaw'], np.asarray(e['t'], float)))
        adj[b].append((a, *inv(e['yaw'], e['t'])))
    poses = {}
    for root in sorted(nodes):
        if root in poses:
            continue
        poses[root] = (0., np.zeros(2), root)
        st = [root]
        while st:
            a = st.pop()
            ya, ta, r = poses[a]
            for b, y, t in sorted(adj[a], key=lambda x: x[0]):
                if b in poses:
                    continue
                poses[b] = (C.wrap(ya - y), ta - C.rot(ya - y) @ t, r)
                st.append(b)
    loops = []
    for e in edges:
        if e['id'] in acc and e['id'] not in tree:
            dy, dt = C.err(rel(poses, e['a'], e['b']), (e['yaw'], e['t']))
            loops.append(dict(id=e['id'], yaw_deg=dy, translation=dt))
    return poses, tree, loops


def solve(nodes, edges):
    """Spanning forest + display roots + flags. Returns cls, poses, tree, loops, roots {root: cams}, flags {u: ...}."""
    cls = {e['id']: classify(e) for e in edges}
    both = {k: (1 + v[2] if v[0] == 'full' else v[2]) for k, v in cls.items() if v[0]}
    p1, tree, loops = forest(nodes, edges, both)
    E = {e['id']: e for e in edges}
    info, deg, wsum = {}, Counter(), Counter()
    for e in edges:
        k, n, f = cls[e['id']]
        if not k:
            continue
        dy, dt = C.err((e['yaw'], e['t']), (e['geo_yaw'], e['geo_t']))
        info[e['id']] = dict(kind=k, frac=f, inliers=n, dyaw=dy, dt=dt, weak=k == 'full' and (f < W_FRAC or dy > W_YAW or dt > W_T))
        for x in (e['a'], e['b']):
            deg[x] += 1
            wsum[x] += f
    adj = defaultdict(list)
    for i in tree:
        adj[E[i]['a']].append((E[i]['b'], i))
        adj[E[i]['b']].append((E[i]['a'], i))
    comps = defaultdict(list)
    for u in nodes:
        comps[p1[u][2]].append(u)
    flags, roots = {}, {}
    for comp in comps.values():
        if len(comp) < 2:
            continue
        root = max(comp, key=lambda u: (deg[u], wsum[u], [-ord(ch) for ch in u]))
        roots[root] = sorted(comp)
        path = {root: ([root], [])}
        st = [root]
        while st:
            a = st.pop()
            for b, i in sorted(adj[a]):
                if b not in path:
                    path[b] = (path[a][0] + [b], path[a][1] + [i])
                    st.append(b)
        on_cycle, bad = set(), {}
        for l in loops:
            a, b = E[l['id']]['a'], E[l['id']]['b']
            if a not in path:
                continue
            pa, pb = path[a][0], path[b][0]
            k = 0
            while k < min(len(pa), len(pb)) and pa[k] == pb[k]:
                k += 1
            cyc = set(pa[k - 1:]) | set(pb[k - 1:])
            on_cycle |= cyc
            if l['yaw_deg'] > L_YAW or l['translation'] > L_T:
                for x in cyc:
                    if x not in bad or l['yaw_deg'] > bad[x]['yaw_deg']:
                        bad[x] = dict(l)
        for u in comp:
            if u == root:
                continue
            ep = path[u][1]
            weak = [dict(pair_id=i, **{k: round(v, 3) if isinstance(v, float) else v for k, v in info[i].items()}) for i in ep if info[i]['weak']]
            flags[u] = dict(root=root, D=any(info[i]['kind'] == 'dir' for i in ep), W=bool(weak), L=u in bad, U=u not in on_cycle,
                            path_edges=ep, weak_edges=weak, bad_loop=bad.get(u))
    return cls, p1, tree, loops, roots, flags


def reason(c, e, dy, dt):
    k, n, f = c
    if k:
        return {'full': 'accepted', 'dir': 'accepted_direction_only'}[k]
    if len(e['P']) == 0:
        return 'no_correspondences'
    if n < MIN_IN:
        return 'too_few_inliers'
    if f < F_LOW:
        return 'low_inlier_fraction'
    if dy > THETA or dt > TAU:
        return 'decoder_geometric_disagree'
    return 'rejected'


def load_group_edges(run, m, g):
    """Edges with correspondences + per-image floor boundaries for one group (verifies prediction hashes)."""
    from covispose.ransac import build_correspondences
    edges, bons = [], defaultdict(list)
    for p in [p for p in m['pairs'] if p['group_id'] == g['room_id']]:
        pj = C.read(run / 'predictions' / f"{p['pair_id']}.json")
        npz = run / 'predictions' / f"{p['pair_id']}.npz"
        if pj.get('status') != 'ok' or C.digest(npz) != pj['heads_sha256']:
            raise SystemExit('prediction missing or changed: ' + p['pair_id'])
        with np.load(npz) as z:
            A, B = build_correspondences(z['bon_a'], z['bon_b'], z['corr_a'], z['covis_a'] > .5, max_dist=10)
            q2, p2 = build_correspondences(z['bon_b'], z['bon_a'], z['corr_b'], z['covis_b'] > .5, max_dist=10)
            bons[p['a']].append(z['bon_a'].astype(float))
            bons[p['b']].append(z['bon_b'].astype(float))
        edges.append(dict(id=p['pair_id'], a=p['a'], b=p['b'], P=np.concatenate([A, p2]), Q=np.concatenate([B, q2]),
                          yaw=pj['yaw'], t=np.array(pj['t']), geo_yaw=pj['geo_yaw'], geo_t=np.array(pj['geo_t']),
                          duplicate=p['duplicate_content']))
    return edges, bons


RULE = 'G rule + direction links f=0.30; maximum-weight spanning forest; no global optimisation'


def poses(run, solver=None, rule=RULE, flag_stats=None, method='spanning_forest'):
    """solver: replaces solve() (pipeline.pose_graph passes its optimiser); default = spanning forest only.
    flag_stats: error rates of the flags for this method (default FLAG_STATS); written into every group file."""
    flag_stats = flag_stats or FLAG_STATS
    solver = solver or solve
    run = C.local(run)
    m = C.read(run / 'manifest.json')
    st = C.read(run / 'inference_status.json') if (run / 'inference_status.json').exists() else {}
    if st.get('pending_or_error') or st.get('complete') != len(m['pairs']):
        raise SystemExit('CoVisPose GPU step not complete for ' + str(run))
    dest = run / ('poses_' + C.stamp())
    (dest / 'groups').mkdir(parents=True)
    (dest / 'plots').mkdir()
    pair_rows, cam_rows, summ = [], [], []
    for g in m['groups']:
        edges, _ = load_group_edges(run, m, g)
        nodes = sorted(g['panorama_uuids'])
        cls, p1, tree, loops, roots, flags = solver(nodes, edges)
        root_of = {u: r for r, cs in roots.items() for u in cs}
        pairs_out = []
        for e in edges:
            dy, dt = C.err((e['yaw'], e['t']), (e['geo_yaw'], e['geo_t']))
            k, n, f = cls[e['id']]
            r = dict(pair_id=e['id'], a=e['a'], b=e['b'], status=reason(cls[e['id']], e, dy, dt), in_tree=e['id'] in tree,
                     inliers=n, inlier_fraction=round(f, 4), correspondences=len(e['P']), decoder_geometric_yaw_deg=round(dy, 2),
                     decoder_geometric_t=round(dt, 3), yaw=float(e['yaw']), t=[float(x) for x in e['t']], duplicate_content=e['duplicate'])
            pairs_out.append(r)
            pair_rows.append(dict(group=g['room_id'], **{k2: v for k2, v in r.items() if k2 not in ('yaw', 't')}))
        cams = {}
        for u in nodes:
            lab = m['images'][u].get('label', '')
            if u in root_of:
                r = root_of[u]
                yr, tr = p1[r][0], np.asarray(p1[r][1])
                yaw = float(C.wrap(p1[u][0] - yr))
                t = (C.rot(-yr) @ (np.asarray(p1[u][1]) - tr)).tolist()
            else:
                r, yaw, t = u, 0., [0., 0.]
            fl = flags.get(u, {})
            cams[u] = dict(label=lab, component_root=r, is_root=u == r, yaw=yaw, t=t,
                           flags=dict(direction_link=bool(fl.get('D')), weak_edge=bool(fl.get('W')), loop_not_closing=bool(fl.get('L')),
                                      unverified=bool(fl.get('U'))),
                           weak_edges=fl.get('weak_edges', []), bad_loop=fl.get('bad_loop'))
            cam_rows.append(dict(group=g['room_id'], house=g['task'], camera=u, label=lab, component_root=r, yaw_deg=round(math.degrees(yaw), 2),
                                 x=round(t[0], 4), y=round(t[1], 4), direction_link=int(bool(fl.get('D'))), weak_edge=int(bool(fl.get('W'))),
                                 loop_not_closing=int(bool(fl.get('L')))))
        comps = sorted([sorted(c) for c in {tuple(v) for v in roots.values()}], key=lambda c: (-len(c), c))
        comps += [[u] for u in nodes if u not in root_of]
        s = dict(group=g['room_id'], house=g['task'], n_images=len(nodes), pairs=len(edges),
                 accepted_full=sum(1 for e in edges if cls[e['id']][0] == 'full'),
                 accepted_direction=sum(1 for e in edges if cls[e['id']][0] == 'dir'),
                 component_sizes=[len(c) for c in comps], fully_connected=len(comps) == 1,
                 loop_max_yaw_deg=max((l['yaw_deg'] for l in loops), default=None),
                 loop_max_translation=max((l['translation'] for l in loops), default=None),
                 flagged_weak_or_loop=sum(1 for u in nodes if cams[u]['flags']['weak_edge'] or cams[u]['flags']['loop_not_closing']))
        summ.append(s)
        C.write(dest / 'groups' / f"{g['room_id']}.json", dict(
            group=g, summary=s, rule=rule,
            convention='p_root = R(yaw) p_u + t; camera height = 1; root = most-connected camera', method=method, flag_stats=flag_stats,
            components=comps, cameras=cams, tree_edges=sorted(tree), pairs=pairs_out, loops=loops, pose_verified=False))
        plot(dest / 'plots' / f"{g['room_id']}.png", g, cams, pairs_out, comps, s)
    C.csvwrite(dest / 'pairs.csv', pair_rows)
    C.csvwrite(dest / 'cameras.csv', cam_rows)
    C.csvwrite(dest / 'groups.csv', summ)
    tot = dict(method=method, groups=len(summ), images=sum(s['n_images'] for s in summ), pairs=len(pair_rows),
               accepted_full=sum(s['accepted_full'] for s in summ), accepted_direction=sum(s['accepted_direction'] for s in summ),
               fully_connected=sum(s['fully_connected'] for s in summ),
               fully_connected_2=f"{sum(s['fully_connected'] for s in summ if s['n_images'] == 2)}/{sum(1 for s in summ if s['n_images'] == 2)}",
               fully_connected_3plus=f"{sum(s['fully_connected'] for s in summ if s['n_images'] >= 3)}/{sum(1 for s in summ if s['n_images'] >= 3)}",
               cameras_flagged_weak_or_loop=sum(s['flagged_weak_or_loop'] for s in summ),
               pair_status=dict(Counter(r['status'] for r in pair_rows)))
    C.write(dest / 'summary.json', tot)
    print(json.dumps(tot, ensure_ascii=False))
    print('POSES: ' + str(dest))
    return dest


def plot(path, g, cams, pairs, comps, s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    C.mpl_fonts(matplotlib)
    big = [c for c in comps if len(c) > 1]
    fig, axes = plt.subplots(1, max(1, len(big)), figsize=(max(6.6, 5.4 * len(big)), 5.4), squeeze=False)
    for ax, comp in zip(axes[0], big or [comps[0]]):
        for p in pairs:
            if p['in_tree'] and p['a'] in comp:
                a, b = cams[p['a']]['t'], cams[p['b']]['t']
                d = p['status'] == 'accepted_direction_only'
                ax.plot([-a[0], -b[0]], [a[1], b[1]], '--' if d else '-', color='C1' if d else '0.4', lw=1.5 if d else 1)
        for u in comp:
            c = cams[u]
            x, y = c['t']
            yv = np.array([-math.sin(c['yaw']), math.cos(c['yaw'])])
            bad = c['flags']['weak_edge'] or c['flags']['loop_not_closing']
            ax.scatter(-x, y, s=60 if c['is_root'] else 36, marker='s' if c['is_root'] else 'o',
                       color='C1' if c['flags']['direction_link'] else 'k', zorder=3)
            if bad:
                ax.scatter(-x, y, s=260, facecolors='none', edgecolors='C3', linewidths=2, zorder=3)
            ax.arrow(-x, y, -.35 * yv[0], .35 * yv[1], head_width=.08, color='k', zorder=3)
            ax.annotate(f"{u[-6:]} {c['label'][:14]}", (-x, y), xytext=(5, 5), textcoords='offset points', fontsize=7)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=.3)
        ax.set_xlabel('camera heights (x = -native x)', fontsize=8)
    lm = s['loop_max_yaw_deg']
    fig.suptitle(f"{g['room_id']}\n{s['n_images']} panos | components {s['component_sizes']}" + (f" | max loop {lm:.1f}°" if lm is not None else '')
                 + '\nsquare = root, orange = direction-only link, red ring = weak edge / loop not closing', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    poses(ap.parse_args().run)
