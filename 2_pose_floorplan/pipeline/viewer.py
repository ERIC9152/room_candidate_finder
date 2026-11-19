"""Step 5 (CPU): interactive viewer (plan + walkable panorama) for a run.

  python -m pipeline.viewer --run runs\\<name> [--title "..."] [--lang en|zh]   -> runs\\<name>\\viewer\\viewer.html (+ images\\)

Open viewer.html in a browser (no server needed). Keep viewer.html and images\\ together when copying.
Modes: each panorama's outline in its own colour with the overlaps, the merged plan, the CoVisPose-only plan.
Red ring "!" = pose flag (weak edge / loop not closing), orange = position via a direction-only link.
Language: English by default (--lang, else the RCF_LANG environment variable); the page's language button switches.
"""
import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import common as C  # noqa: E402
from pipeline.poses import FLAG_STATS  # noqa: E402

TEMPLATE = Path(__file__).with_name('viewer_template.html')
PANO_SIZE = (4096, 2048)     # power of two: WebGL1 can only wrap (REPEAT) power-of-two textures


def overlap_rings(polys):
    from shapely.ops import unary_union
    inter = [polys[i].intersection(polys[j]) for i in range(len(polys)) for j in range(i + 1, len(polys))]
    ov = unary_union([x for x in inter if not x.is_empty and x.area > 1e-4]) if inter else None
    if ov is None or ov.is_empty:
        return []
    return [[[round(x, 4), round(y, 4)] for x, y in list(p.exterior.coords)[:-1]]
            for p in getattr(ov, 'geoms', [ov]) if p.geom_type == 'Polygon' and p.area > 1e-3]


def viewer(run, title=None, lang=None):
    from PIL import Image
    from shapely.geometry import Polygon
    run = C.local(run)
    m = C.read(run / 'manifest.json')
    pdir, ldir = C.latest(run, 'poses'), C.latest(run, 'plans')
    out = run / 'viewer'
    (out / 'images').mkdir(parents=True, exist_ok=True)
    data = []
    for g in m['groups']:
        gid = g['room_id']
        gp, gl = C.read(pdir / 'groups' / f'{gid}.json'), C.read(ldir / 'groups' / f'{gid}.json')
        cams, comps = {}, []
        dir_pairs = {p['pair_id'] for p in gp['pairs'] if p['status'] == 'accepted_direction_only'}
        for c in gl['components']:
            for u in c['panoramas']:
                pc, o = gp['cameras'][u], c['outlines'][u]
                own = o[c['outline_source'][u]]
                cams[u] = dict(label=pc['label'] or u[-6:], yaw=round(pc['yaw'], 5), t=[round(x, 4) for x in pc['t']],
                               unc=pc['flags']['direction_link'], src=c['outline_source'][u],
                               outline=None if own is None else np.round(np.asarray(own), 3).tolist(),
                               cvo=np.round(np.asarray(o['CV']), 3).tolist())
                if pc['flags']['weak_edge'] or pc['flags']['loop_not_closing']:
                    cams[u]['bad'] = dict(weak=[dict(frac=w['frac'], dyaw=w['dyaw'], dt=w['dt']) for w in pc['weak_edges']] if pc['flags']['weak_edge'] else [],
                                          loop=dict(yaw=round(pc['bad_loop']['yaw_deg'], 1), t=round(pc['bad_loop']['translation'], 2))
                                          if pc['flags']['loop_not_closing'] else None)
            edges = [dict(a=p['a'], b=p['b'], dir=p['pair_id'] in dir_pairs) for p in gp['pairs'] if p['in_tree'] and p['a'] in c['panoramas']]
            polys = [Polygon(np.asarray(cams[u]['outline']) @ C.rot(cams[u]['yaw']).T + np.asarray(cams[u]['t'])).buffer(0)
                     for u in c['panoramas'] if cams[u]['outline']]
            comps.append(dict(panos=c['panoramas'], root=c['root'], method=c['method'], plan=c['plan'], edges=edges, gt=None,
                              overlap=overlap_rings(polys), cvplan=c['covispose_plan']))
        im = Image.open(ldir / 'plots' / f'{gid}.png').convert('RGB')
        w = min(640, im.width)
        im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
        b = io.BytesIO()
        im.save(b, 'JPEG', quality=70, optimize=True)
        data.append(dict(id=gid, kind=g.get('space_kind', 'room'), n=len(g['panorama_uuids']), comps=comps, cams=cams, iou=None,
                         loop=gp['summary']['loop_max_yaw_deg'], dirlink=any(e['dir'] for c in comps for e in c['edges']),
                         labels=sorted({cams[u]['label'] for u in g['panorama_uuids'] if m['images'][u].get('label')}),
                         nbad=sum(1 for u in cams if cams[u].get('bad')),
                         thumb='data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()))
    n_new = 0
    for u, rec in m['images'].items():
        dst = out / 'images' / f'{u}.jpg.js'
        if dst.exists() or rec['status'] != 'ok':
            continue
        im = Image.open(C.local(rec['path'])).convert('RGB')
        if im.size != PANO_SIZE:
            im = im.resize(PANO_SIZE, Image.LANCZOS)
        b = io.BytesIO()
        im.save(b, 'JPEG', quality=82)
        detail = json.dumps({'path': f'images/{u}.jpg', 'data': 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode('ascii')})
        dst.write_text("window.dispatchEvent(new CustomEvent('floorplan-photo',{detail:" + detail + "}));", encoding='utf-8')
        n_new += 1
    # flag error rates and pose method as written by the poses step (older poses folders: spanning forest, FLAG_STATS)
    g0 = C.read(pdir / 'groups' / f"{m['groups'][0]['room_id']}.json") if m['groups'] else {}
    fs = g0.get('flag_stats') or FLAG_STATS
    method = g0.get('method', 'spanning_forest')
    stats = dict(valid=fs['valid'], W=fs['W'], L=fs['L'])
    lang = (lang or os.environ.get('RCF_LANG', 'en')).strip().lower()[:2]
    lang = lang if lang in ('en', 'zh') else 'en'
    html = (TEMPLATE.read_text(encoding='utf8').replace('__TITLE__', json.dumps(title, ensure_ascii=False))   # None -> per-language default
            .replace('__RUN__', json.dumps(run.name, ensure_ascii=False)).replace('__LANG__', json.dumps(lang))
            .replace('__IMG_BASE__', json.dumps('images/')).replace('__FLAGSTATS__', json.dumps(stats)).replace('__METHOD__', json.dumps(method))
            .replace('__DATA__', json.dumps(data, ensure_ascii=False)))
    (out / 'viewer.html').write_text(html, encoding='utf8')
    print(f'VIEWER: {out / "viewer.html"} ({len(html) / 1e6:.1f} MB, {n_new} panorama files written)')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--title', help='page title (default: "<run> room group plans", translated)')
    ap.add_argument('--lang', choices=('en', 'zh'), help='initial language (default: RCF_LANG, else en)')
    a = ap.parse_args()
    viewer(a.run, a.title, a.lang)
