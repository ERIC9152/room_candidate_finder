"""Compare a run of the bundled sample (Verify.ps1) with the reference results in sample/reference/.

  python -m pipeline.verify --run runs\\<verify run>

Reference = the original Dollhouse results for the same 9 panoramas (3 groups): CoVisPose predictions of the
104-group run, HorizonNet results of room_stitch_mvp, and the poses/plans the pipeline computes from them.
Pass criteria (GPU arithmetic differs slightly between machines/drivers):
  CoVisPose pair poses: yaw <= 0.5 deg, translation <= 0.02 camera heights (decoder and geometric branch)
  HorizonNet: same number of corners, floor-corner positions within 0.05 camera heights
  poses: same components, spanning tree and flags; camera positions within 0.05, yaw within 1 deg
  plans: area within 5%
reference.json keeps one set of poses/plans per step-4 method: "groups" for the pose graph (pipeline.pose_graph, the
default) and "groups_spanning_forest" for -NoPoseGraph (pipeline.poses); the run's poses folder says which it used.
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
from pipeline.plans import hn_footprint  # noqa: E402

REF = ROOT / 'sample' / 'reference'
REF_KEY = {'pose_graph': 'groups', 'spanning_forest': 'groups_spanning_forest'}


def verify(run):
    run = C.local(run)
    m = C.read(run / 'manifest.json')
    ref = C.read(REF / 'reference.json')
    rows, ok = [], True

    def check(name, good, detail):
        nonlocal ok
        ok &= bool(good)
        rows.append((name, 'OK' if good else 'FAIL', detail))
    # CoVisPose predictions
    worst = [0., 0., 0., 0.]
    for p in m['pairs']:
        r = ref['pairs'][f"{p['a']}|{p['b']}"]
        pj = C.read(run / 'predictions' / f"{p['pair_id']}.json")
        for i, (y, t) in enumerate((('yaw', 't'), ('geo_yaw', 'geo_t'))):
            dy, dt = C.err((pj[y], pj[t]), (r[y], r[t]))
            worst[2 * i] = max(worst[2 * i], dy)
            worst[2 * i + 1] = max(worst[2 * i + 1], dt)
    check('CoVisPose pair poses', worst[0] <= .5 and worst[1] <= .02 and worst[2] <= .5 and worst[3] <= .02,
          f'{len(m["pairs"])} pairs; max diff decoder {worst[0]:.3f} deg / {worst[1]:.4f}, geometric {worst[2]:.3f} deg / {worst[3]:.4f}')
    # HorizonNet
    hw, bad = 0., []
    for u in m['images']:
        f, rf = run / 'horizonnet' / f'{u}.json', REF / 'horizonnet' / f'{u}.json'
        a, b = (hn_footprint(f) if f.exists() else None), hn_footprint(rf)
        if (a is None) != (b is None) or (a is not None and len(a) != len(b)):
            bad.append(u)
            continue
        if a is not None:
            hw = max(hw, float(np.abs(a - b).max()))
    check('HorizonNet outlines', not bad and hw <= .05, f'{len(m["images"])} panoramas; max corner diff {hw:.4f}' + (f'; differ: {bad}' if bad else ''))
    # poses and plans
    pdir, ldir = C.latest(run, 'poses'), C.latest(run, 'plans')
    for g in m['groups']:
        gid = g['room_id']
        gp = C.read(pdir / 'groups' / f'{gid}.json')
        rp = ref[REF_KEY[gp.get('method', 'spanning_forest')]][gid]
        same_c = sorted(map(sorted, gp['components'])) == sorted(map(sorted, rp['components']))
        tree = sorted(tuple(sorted((p['a'], p['b']))) for p in gp['pairs'] if p['in_tree'])
        same_t = tree == sorted(map(tuple, rp['tree']))
        fl = {u: c['flags'] for u, c in gp['cameras'].items()}
        same_f = fl == rp['flags']
        dpos = max((float(np.linalg.norm(np.subtract(c['t'], rp['cameras'][u]['t']))) for u, c in gp['cameras'].items()), default=0.)
        dyaw = max((abs(math.degrees(C.wrap(c['yaw'] - rp['cameras'][u]['yaw']))) for u, c in gp['cameras'].items()), default=0.)
        gl = C.read(ldir / 'groups' / f'{gid}.json')
        areas = [c['area'] for c in gl['components']]
        da = max(abs(a - b) / max(b, 1e-6) for a, b in zip(areas, rp['areas'])) if len(areas) == len(rp['areas']) else 9
        check(f"{gid} ({gp.get('method', 'spanning_forest')})", same_c and same_t and same_f and dpos <= .05 and dyaw <= 1 and da <= .05,
              f'components {"same" if same_c else "DIFFER"}, tree {"same" if same_t else "DIFFER"}, flags {"same" if same_f else "DIFFER"}, '
              f'max camera diff {dpos:.4f} / {dyaw:.3f} deg, plan area diff {da:.2%}')
    vw = run / 'viewer' / 'viewer.html'
    check('viewer', vw.exists() and len(list((run / 'viewer' / 'images').glob('*.jpg.js'))) == len(m['images']), str(vw))
    w = max(len(r[0]) for r in rows)
    for name, s, d in rows:
        print(f'{name:<{w}}  {s:4}  {d}')
    C.write(run / 'verify_result.json', dict(passed=bool(ok), checks=[dict(check=a, result=b, detail=c) for a, b, c in rows], at=C.stamp()))
    print('VERIFY PASSED' if ok else 'VERIFY FAILED')
    return ok


def make_reference(run, pairs=False):
    """Maintainer tool: update sample/reference/reference.json from a run built from the original results.
    Writes the poses/plans of the run's step-4 method ("groups" or "groups_spanning_forest") and keeps the rest;
    the CoVisPose pair poses are (re)written only when missing or with pairs=True (--pairs)."""
    run = C.local(run)
    m = C.read(run / 'manifest.json')
    pdir, ldir = C.latest(run, 'poses'), C.latest(run, 'plans')
    out = C.read(REF / 'reference.json') if (REF / 'reference.json').exists() else dict(note='original Dollhouse results for the sample panoramas')
    if pairs or not out.get('pairs'):
        out['pairs'] = {}
        for p in m['pairs']:
            pj = C.read(run / 'predictions' / f"{p['pair_id']}.json")
            out['pairs'][f"{p['a']}|{p['b']}"] = {k: pj[k] for k in ('yaw', 't', 'geo_yaw', 'geo_t')}
    key = None
    for g in m['groups']:
        gid = g['room_id']
        gp, gl = C.read(pdir / 'groups' / f'{gid}.json'), C.read(ldir / 'groups' / f'{gid}.json')
        key = REF_KEY[gp.get('method', 'spanning_forest')]
        out.setdefault(key, {})[gid] = dict(components=gp['components'], tree=sorted(tuple(sorted((p['a'], p['b']))) for p in gp['pairs'] if p['in_tree']),
                                  flags={u: c['flags'] for u, c in gp['cameras'].items()},
                                  cameras={u: dict(yaw=c['yaw'], t=c['t']) for u, c in gp['cameras'].items()},
                                  areas=[c['area'] for c in gl['components']])
    C.write(REF / 'reference.json', out)
    print('REFERENCE WRITTEN', REF / 'reference.json', key)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--make-reference', action='store_true')
    ap.add_argument('--pairs', action='store_true', help='with --make-reference: also rewrite the CoVisPose pair poses')
    a = ap.parse_args()
    if a.make_reference:
        make_reference(a.run, a.pairs)
    else:
        sys.exit(0 if verify(a.run) else 1)
