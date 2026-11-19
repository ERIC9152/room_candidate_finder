"""Score pipeline HorizonNet outlines against per-panorama layout_gt (see PLAN.md). CPU only.
python evaluate.py --pred <run>/horizonnet --gt <gt dir: <scene>/<pano>.txt, <scene>/rooms.csv> --hn <HorizonNet repo> --out <dir>"""
import argparse, csv, glob, json, math, os, re, sys
from collections import defaultdict
import numpy as np
from shapely.geometry import Polygon

ap = argparse.ArgumentParser()
ap.add_argument('--pred', required=True); ap.add_argument('--gt', required=True)
ap.add_argument('--hn', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
sys.path.insert(0, a.hn)
from misc import post_proc                     # official HorizonNet
from eval_general import layout_2_depth        # official HorizonNet
os.makedirs(a.out, exist_ok=True)
W, H, CH = 1024, 512, -1.6


def room_types(scene_dir):
    # rooms.csv of the Dollhouse annotation has Chinese column names: '圖檔' = image file, '房間可能性質' = likely room type,
    # '房間描述' = room description (its JSON-like text holds '房間性質' = room type). '未標' = not labelled.
    out = {}
    f = os.path.join(scene_dir, 'rooms.csv')
    if not os.path.exists(f):
        return out
    for row in csv.DictReader(open(f, encoding='utf-8-sig')):
        img = (row.get('圖檔') or '').strip()
        txt = (row.get('房間可能性質') or '') + ' ' + (row.get('房間描述') or '')
        m = re.search(r'房[問間]性質"?\s*:\s*"?([^",\n}]+)', txt)
        out[img[:-4]] = (m.group(1).strip() if m else (row.get('房間可能性質') or '').strip()) or '未標'
    return out


def floor_poly(cor):
    xy = post_proc.np_coor2xy(cor[1::2], CH, W, H, floorW=1, floorH=1)
    return Polygon(xy)


def col_dist(cor):
    """Distance (camera height 1) from the camera to the floor polygon along each of 1024 column azimuths."""
    poly = floor_poly(cor)
    if not poly.is_valid or poly.area <= 0:
        return None
    from shapely.geometry import LineString, Point
    b = poly.boundary; d = np.full(256, np.nan)
    for k, x in enumerate(np.arange(256) * 4 + 2):
        p = np.asarray(post_proc.np_coor2xy(np.array([[x, H * .75]]), CH, W, H, floorW=1, floorH=1))[0]
        far = tuple(p / np.hypot(*p) * 1e3)               # same azimuth convention as the official projection
        hit = LineString([(0, 0), far]).intersection(b)
        if hit.is_empty:
            continue
        g = getattr(hit, 'geoms', [hit])
        d[k] = min(Point(0, 0).distance(q) for q in g)
    return d / abs(CH)


rows = []
for gtf in sorted(glob.glob(os.path.join(a.gt, '*', '*.txt'))):
    scene = os.path.basename(os.path.dirname(gtf)); u = os.path.basename(gtf)[:-4]
    house = scene.split('_')[0]
    gt = np.loadtxt(gtf).reshape(-1, 2)
    r = dict(pano=u, scene=scene, house=house, gt_corners=len(gt) // 2)
    pf = os.path.join(a.pred, u + '.json')
    if not os.path.exists(pf):
        r.update(status='missing', iou2d=0., iou3d=0.); rows.append(r); continue
    d = json.load(open(pf, encoding='utf-8'))
    dt = np.asarray(d['uv'], float) * [W, H]
    r['dt_corners'] = len(dt) // 2
    gpoly, dpoly = floor_poly(gt), floor_poly(dt)
    if not gpoly.is_valid:
        r.update(status='gt_invalid'); rows.append(r); continue
    if not dpoly.is_valid or dpoly.area <= 0:
        r.update(status='pred_invalid', iou2d=0., iou3d=0.); rows.append(r); continue
    inter = dpoly.intersection(gpoly).area
    iou2d = inter / (dpoly.area + gpoly.area - inter)
    try:
        h_dt = abs(post_proc.get_z1(dt[1::2, 1], dt[0::2, 1], CH, H).mean() - CH)
        h_gt = abs(post_proc.get_z1(gt[1::2, 1], gt[0::2, 1], CH, H).mean() - CH)
        i3 = inter * min(h_dt, h_gt); iou3d = i3 / (dpoly.area * h_dt + gpoly.area * h_gt - i3)
    except Exception:
        iou3d = 0.
    dts = dt.copy(); dts[0::2, 0] = dts[1::2, 0]          # snap ceiling x to floor x for the depth metrics
    gd = layout_2_depth(gt, H, W)
    try:
        dd = layout_2_depth(dts, H, W)
    except Exception:
        dd = np.zeros_like(gd)
    rmse = float(((gd - dd) ** 2).mean() ** .5)
    with np.errstate(divide='ignore', invalid='ignore'):
        delta1 = float((np.maximum(gd / dd, dd / gd) < 1.25).mean())
    cg, cd = col_dist(gt), col_dist(dt)
    rel = np.nanmedian(np.abs(cd - cg) / cg) if cg is not None and cd is not None else np.nan
    signed = np.nanmedian((cd - cg) / cg) if cg is not None and cd is not None else np.nan
    r.update(status='ok', iou2d=float(iou2d), iou3d=float(iou3d), rmse=rmse, delta1=delta1,
             wall_rel_err=float(rel), wall_signed=float(signed),
             area_ratio=float(dpoly.area / gpoly.area))
    rows.append(r)

types = {}
for sd in glob.glob(os.path.join(a.gt, '*')):
    for k, v in room_types(sd).items():
        types[k] = v
for r in rows:
    r['room_type'] = types.get(r['pano'], '未標')

json.dump(rows, open(os.path.join(a.out, 'rows.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
ev = [r for r in rows if r['status'] != 'gt_invalid']


def boot(rs, key, n=1000, seed=0):
    by = defaultdict(list)
    for r in rs:
        by[r['house']].append(r[key])
    hs = sorted(by); rng = np.random.default_rng(seed)
    v = [np.mean([x for h in rng.choice(hs, len(hs)) for x in by[h]]) for _ in range(n)]
    return float(np.quantile(v, .025)), float(np.quantile(v, .975))


def summ(rs):
    ok = [r for r in rs if r['status'] == 'ok']
    s = dict(n=len(rs), n_ok=len(ok), houses=len({r['house'] for r in rs}),
             iou2d=float(np.mean([r['iou2d'] for r in rs])), iou3d=float(np.mean([r['iou3d'] for r in rs])))
    if ok:
        s.update(rmse=float(np.mean([r['rmse'] for r in ok])), delta1=float(np.mean([r['delta1'] for r in ok])),
                 wall_rel_err=float(np.nanmedian([r['wall_rel_err'] for r in ok])),
                 wall_signed=float(np.nanmedian([r['wall_signed'] for r in ok])),
                 iou2d_median=float(np.median([r['iou2d'] for r in rs])),
                 corners_match=float(np.mean([r['dt_corners'] == r['gt_corners'] for r in ok])))
    if s['houses'] >= 3:
        s['iou2d_ci'] = boot(rs, 'iou2d'); s['iou3d_ci'] = boot(rs, 'iou3d')
    return s


def cbin(n):
    return str(n) if n < 10 else '10+'


res = dict(overall=summ(ev),
           status={k: sum(r['status'] == k for r in rows) for k in ('ok', 'missing', 'pred_invalid', 'gt_invalid')},
           by_corners={k: summ([r for r in ev if cbin(r['gt_corners']) == k]) for k in ('4', '6', '8', '10+') if any(cbin(r['gt_corners']) == k for r in ev)},
           by_type={t: summ([r for r in ev if r['room_type'] == t]) for t in sorted({r['room_type'] for r in ev})},
           by_house={h: summ([r for r in ev if r['house'] == h]) for h in sorted({r['house'] for r in ev})},
           hist2d=np.histogram([r['iou2d'] for r in ev], bins=10, range=(0, 1))[0].tolist())
json.dump(res, open(os.path.join(a.out, 'results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res['overall'], ensure_ascii=False)); print(res['status'])
