"""Post-hoc diagnostics (written after seeing results.json; descriptive only)."""
import json, sys, glob, os
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon
from shapely import affinity
sys.path.insert(0, '/home/claude/pkg/2_pose_floorplan/third_party/HorizonNet')
from misc import post_proc
W, H, CH = 1024, 512, -1.6
rows = json.load(open('out/rows.json', encoding='utf-8'))
G = '/home/claude/hneval/gtflat'; P = '/home/claude/hneval/runs/all/horizonnet'
def poly(c): return Polygon(post_proc.np_coor2xy(c[1::2], CH, W, H, floorW=1, floorH=1))
def iou(a, b):
    i = a.intersection(b).area; return i / (a.area + b.area - i)
for r in rows:
    gt = np.loadtxt(f"{G}/{r['scene']}/{r['pano']}.txt").reshape(-1, 2)
    dt = np.asarray(json.load(open(f"{P}/{r['pano']}.json"))['uv']) * [W, H]
    g, d = poly(gt), poly(dt)
    ss = np.exp(np.linspace(np.log(.4), np.log(2.5), 121))
    v = [iou(affinity.scale(d, s, s, origin=(0, 0)), g) for s in ss]
    k = int(np.argmax(v)); r['iou2d_scale_opt'] = float(v[k]); r['best_scale'] = float(ss[k])
    # camera-height-free: equal-area scaling
    r['iou2d_eqarea'] = iou(affinity.scale(d, (g.area/d.area)**.5, (g.area/d.area)**.5, origin=(0, 0)), g)
    # GT ceiling height (camera heights): consistency of the annotation
    r['gt_ceil_h'] = float(abs(post_proc.get_z1(gt[1::2, 1], gt[0::2, 1], CH, H).mean() - CH) / abs(CH))
    r['dt_ceil_h'] = float(abs(post_proc.get_z1(dt[1::2, 1], dt[0::2, 1], CH, H).mean() - CH) / abs(CH))
json.dump(rows, open('out/rows_diag.json', 'w', encoding='utf-8'), ensure_ascii=False)
m = lambda k, rs: float(np.mean([r[k] for r in rs]))
odd = [r for r in rows if r['gt_corners'] % 2]; even = [r for r in rows if not r['gt_corners'] % 2]
print(f"odd-corner GT: n={len(odd)} 2D {m('iou2d',odd):.3f} | even: n={len(even)} 2D {m('iou2d',even):.3f}")
from collections import Counter; print('gt corner counts', sorted(Counter(r['gt_corners'] for r in rows).items()))
print(f"all: 2D {m('iou2d',rows):.3f} | scale-optimal 2D {m('iou2d_scale_opt',rows):.3f} | equal-area 2D {m('iou2d_eqarea',rows):.3f}")
print('best scale median %.3f, IQR %.3f-%.3f' % tuple(np.quantile([r['best_scale'] for r in rows], [.5, .25, .75])))
# house share of signed wall error variance
by = defaultdict(list)
for r in rows: by[r['house']].append(np.log(r['best_scale']))
allv = np.array([x for v in by.values() for x in v]); within = np.array([x - np.mean(v) for v in by.values() for x in v])
print('log best-scale variance explained by house: %.2f' % (1 - within.var() / allv.var()))
print('house median best scale:', {h: round(float(np.exp(np.median(v))), 2) for h, v in sorted(by.items())})
print('ceiling height (camera heights): GT median %.3f, pred median %.3f' % (np.median([r['gt_ceil_h'] for r in rows]), np.median([r['dt_ceil_h'] for r in rows])))
byh = defaultdict(list)
for r in rows: byh[r['house']].append(r['gt_ceil_h'])
print('GT ceiling height per house (median):', {h: round(float(np.median(v)), 2) for h, v in sorted(byh.items())})
byd = defaultdict(list)
for r in rows: byd[r['house']].append(r['dt_ceil_h'])
print('pred ceiling height per house (median):', {h: round(float(np.median(v)), 2) for h, v in sorted(byd.items())})
