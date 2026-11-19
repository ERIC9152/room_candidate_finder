"""Step 4 (CPU): floor plans per room group from the poses and the per-panorama outlines.

  python -m pipeline.plans --run runs\\<name>          (uses the newest poses_* folder; writes plans_<time>\\)

Outlines per panorama (camera frame, camera height = 1):
  CV  CoVisPose floor boundary, column-wise median over the panorama's pairs, simplified (0.05);
  HN  official HorizonNet floor corners (None if a corner lies beyond 10 camera heights, i.e. degenerate);
  MX  per column the farther of CV and HN, simplified.
Component plan: >= 2 panoramas -> union of HN outlines placed with the poses (MX where HN is missing/degenerate);
1 panorama -> MX. (Best combination found on the 36 Dollhouse groups with layout ground truth; see README.)
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
from pipeline.poses import load_group_edges  # noqa: E402
from covispose.geometry import col_centers_theta, dir_of_theta, bfloor_to_dist, polygon_to_column_distance  # noqa: E402

MAXD, SIMPLIFY = 10., .05
DIRS = dir_of_theta(col_centers_theta(1024))


def simplify(poly):
    from shapely.geometry import Polygon
    p = Polygon(poly).simplify(SIMPLIFY, preserve_topology=True)
    xy = np.asarray(p.exterior.coords)[:-1]
    return xy if len(xy) >= 3 else np.asarray(poly)


def hn_footprint(path):
    """HorizonNet JSON -> floor polygon (camera frame) or None if degenerate / unusable."""
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    uv = np.asarray(d['uv'], float)
    if len(uv) % 2 or len(uv) < 6:
        return None
    corners = []
    for k in range(0, len(uv), 2):
        a, b = uv[k], uv[k + 1]
        f = a if a[1] > b[1] else b
        if f[1] <= .5:
            return None
        corners.append(f)
    corners = sorted(corners, key=lambda c: c[0])
    th = np.array([(c[0] - .5) * 2 * math.pi for c in corners])
    r_raw = np.array([1 / math.tan((c[1] - .5) * math.pi) for c in corners])
    if (r_raw > MAXD).any():
        return None
    return dir_of_theta(th) * r_raw[:, None]


def coldist(poly):
    if poly is None:
        return None
    d = polygon_to_column_distance(np.asarray(poly, float), 1024)
    return np.where(np.isfinite(d), np.minimum(d, MAXD), np.nan)


def outlines(bons, hn_json):
    r_cv = np.median(np.clip(bfloor_to_dist(np.stack(bons)), 1e-3, MAXD), 0)
    hn = hn_footprint(hn_json) if hn_json.exists() else None
    r_hn = coldist(hn)
    r_mx = r_cv if r_hn is None else np.where(np.isfinite(r_hn), np.maximum(r_cv, r_hn), r_cv)
    return dict(CV=simplify(DIRS * r_cv[:, None]), HN=hn, MX=simplify(DIRS * r_mx[:, None]))


def placed(poly, cam):
    return np.asarray(poly, float) @ C.rot(cam['yaw']).T + np.asarray(cam['t'])


def rings(geom):
    from shapely.geometry import mapping
    g = mapping(geom)
    polys = [g['coordinates']] if g['type'] == 'Polygon' else g['coordinates'] if g['type'] == 'MultiPolygon' else []
    return [[[round(x, 4), round(y, 4)] for x, y in p[0][:-1]] for p in polys]


def plans(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    C.mpl_fonts(matplotlib)
    run = C.local(run)
    m = C.read(run / 'manifest.json')
    pdir = C.latest(run, 'poses')
    dest = run / ('plans_' + C.stamp())
    (dest / 'groups').mkdir(parents=True)
    (dest / 'plots').mkdir()
    index = []
    for g in m['groups']:
        gid = g['room_id']
        gp = C.read(pdir / 'groups' / f'{gid}.json')
        cams = gp['cameras']
        _, bons = load_group_edges(run, m, g)
        own = {u: outlines(bons[u], run / 'horizonnet' / f'{u}.json') for u in g['panorama_uuids']}
        comps_out = []
        for comp in gp['components']:
            single = len(comp) == 1
            used = {u: ('MX' if single or own[u]['HN'] is None else 'HN') for u in comp}
            plan = unary_union([Polygon(placed(own[u][used[u]], cams[u])).buffer(0) for u in comp])
            cvu = unary_union([Polygon(placed(own[u]['CV'], cams[u])).buffer(0) for u in comp])
            comps_out.append(dict(panoramas=comp, root=cams[comp[0]]['component_root'], method='MX (single panorama)' if single else 'HN union',
                                  outline_source=used, plan=rings(plan), area=float(plan.area), covispose_plan=rings(cvu),
                                  outlines={u: {k: (None if v is None else np.round(np.asarray(v), 4).tolist()) for k, v in own[u].items()} for u in comp}))
        C.write(dest / 'groups' / f'{gid}.json', dict(
            group=gid, task=g['task'], convention='camera heights; frame of the component root camera (CoVisPose native XY); plot x = -native x',
            poses=str(pdir.name), components=comps_out, pose_verified=False))
        fig, axes = plt.subplots(1, len(comps_out), figsize=(5.6 * len(comps_out), 5.4), squeeze=False)
        for ax, c in zip(axes[0], comps_out):
            for ring in c['plan']:
                x, y = np.asarray(ring + ring[:1]).T
                ax.fill(-x, y, facecolor='C0', alpha=.15)
                ax.plot(-x, y, color='C0', lw=2)
            for p in gp['pairs']:
                if p['in_tree'] and p['a'] in c['panoramas']:
                    a_, b_ = cams[p['a']]['t'], cams[p['b']]['t']
                    d = p['status'] == 'accepted_direction_only'
                    ax.plot([-a_[0], -b_[0]], [a_[1], b_[1]], '--' if d else ':', color='C1' if d else '0.3', lw=1.6 if d else 1)
            for u in c['panoramas']:
                x, y = cams[u]['t']
                fl = cams[u]['flags']
                ax.scatter(-x, y, s=46 if u == c['root'] else 30, marker='s' if u == c['root'] else 'o', color='C1' if fl['direction_link'] else 'k', zorder=5)
                if fl['weak_edge'] or fl['loop_not_closing']:
                    ax.scatter(-x, y, s=220, facecolors='none', edgecolors='C3', linewidths=2, zorder=5)
                d = C.rot(cams[u]['yaw']) @ np.array([0., 1.])
                ax.arrow(-x, y, -.3 * d[0], .3 * d[1], head_width=.08, color='k', zorder=5)
                ax.annotate(f"{u[-6:]} {cams[u]['label'][:16]}", (-x, y), xytext=(4, 4), textcoords='offset points', fontsize=7)
            ax.set_title(f"{c['method']} | {len(c['panoramas'])} pano(s)", fontsize=9)
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid(alpha=.3)
            ax.set_xlabel('camera heights (x = -native x)', fontsize=8)
        fig.suptitle(f"{gid} ({len(g['panorama_uuids'])} panos) | components {[len(c['panoramas']) for c in comps_out]}"
                     + '\norange = direction-only link, red ring = pose flag', fontsize=9)
        fig.tight_layout()
        fig.savefig(dest / 'plots' / f'{gid}.png', dpi=100)
        plt.close(fig)
        index.append(dict(group=gid, n=len(g['panorama_uuids']), components=[len(c['panoramas']) for c in comps_out],
                          hn_missing=[u for u in g['panorama_uuids'] if own[u]['HN'] is None]))
    C.write(dest / 'index.json', index)
    print('PLANS: ' + str(dest))
    return dest


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    plans(ap.parse_args().run)
