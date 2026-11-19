"""Step 2 of the pipeline: CoVisPose relative poses for every pair of a prepared run folder.

    python scripts/manual_room_groups_v2.py run --run runs\\<name> [--device auto|cuda|cpu]

Speed: the image encoder works on each panorama separately, so its features are computed once per panorama
(plain + mirrored; the half-turn TTA view is an exact shift of the features) and reused for every pair of the
group; only the pair transformer runs per pair. The first pair is also computed the original way and compared;
if the two differ the run continues the original way. --no-feature-cache switches the reuse off.

Device: CUDA when available, otherwise the CPU (same results, slower: about 10 s per pair on a 2-core CPU,
measured on the bundled sample). --device or the environment variable RCF_DEVICE forces one of them.

The run folder (manifest.json, run_config.json) is written by `python -m pipeline.prepare`.
Inference: EMA weights, FP32, TF32 disabled, 4-view TTA (original, half-turn roll, mirror, mirror + half-turn);
the decoder pose AND the geometric-branch pose are stored per pair in predictions/.
Resumable: verified pairs are skipped, failed pairs are retried.
Before running, the frozen inputs are checked (manifest, source files, checkpoint and image hashes).

Convention: p_B = R(yaw) p_A + t, native CoVisPose XY, camera height = 1.

Path mapping (only when the stored Windows paths must be read on another OS):
MRG_PATH_MAP="C:\\data=>/mnt/data;..."

(The research-only `prepare` / `analyze` sub-commands of the original Dollhouse study were removed in the
delivered package; RULES / DIAG below are still recorded by pipeline.prepare.)
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KEYS = ('covis_a', 'covis_b', 'corr_a', 'corr_b', 'bon_a', 'bon_b')
RULES = dict(covis_threshold=.5, max_distance=10., residual=.30, min_inliers=20, min_fraction=.40)
DIAG = dict(pose_agreement_yaw_deg=10., pose_agreement_t=.5, ransac_threshold=.10, ransac_iters=500)
FROZEN_SOURCES = ['scripts/manual_room_groups_v2.py', 'scripts/run_scene_graph.py'] + \
    [f'covispose/{n}.py' for n in ('__init__', 'checkpoint', 'evaluate', 'model', 'geo_pose',
                                    'ransac', 'geometry', 'scene_graph')]

# ----------------------------------------------------------------------------- io
def local(p):
    """Map a stored (Windows) path to a readable local path via MRG_PATH_MAP."""
    s = str(p)
    for item in filter(None, os.environ.get('MRG_PATH_MAP', '').split(';')):
        src, dst = item.split('=>')
        if s.lower().replace('/', '\\').startswith(src.lower().replace('/', '\\')):
            return Path(dst + s[len(src):].replace('\\', '/'))
    return Path(s)


def read(p):
    return json.loads(local(p).read_text(encoding='utf-8-sig'))


def digest(p):
    h = hashlib.sha256()
    with local(p).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')


def write(p, data):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    os.replace(tmp, p)


def sources():
    return {s: digest(ROOT / s) for s in FROZEN_SOURCES}

# ------------------------------------------------------------------------ checks
def validate(run, check_images=True):
    cfg, m = read(run / 'run_config.json'), read(run / 'manifest.json')
    if digest(run / 'manifest.json') != cfg['manifest_sha256']:
        raise ValueError('Frozen manifest changed. Prepare a new version.')
    changed = [k for k, v in sources().items() if cfg['source_code'].get(k) != v]
    if changed:
        raise ValueError('Frozen source changed: ' + ', '.join(changed))
    if digest(cfg['checkpoint']) != cfg['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    if check_images:
        for im in m['images'].values():
            if im['status'] == 'ok' and digest(im['path']) != im['sha256']:
                raise ValueError('Image changed: ' + im['path'])
    return cfg, m


def completed(run, pair):
    import numpy as np
    p = run / 'predictions' / (pair['pair_id'] + '.json')
    if not p.exists():
        return False
    r = read(p)
    if r.get('status') != 'ok' or any(r.get(k) != pair[k] for k in ('pair_id', 'a', 'b', 'group_id')):
        return False
    dense = p.with_suffix('.npz')
    if not dense.exists() or digest(dense) != r['heads_sha256']:
        return False
    with np.load(dense, allow_pickle=False) as z:
        return all(k in z and np.isfinite(z[k]).all() for k in KEYS)

# ------------------------------------------------------------------------------ run
def pick_device(want='auto'):
    """'cuda' or 'cpu'. auto = CUDA when this torch can use it, otherwise the CPU (slow, same results)."""
    import torch
    want = (want or 'auto').lower()
    if want == 'auto':
        want = (os.environ.get('RCF_DEVICE') or 'auto').lower()
    if want not in ('auto', 'cuda', 'cpu'):
        raise SystemExit(f"Unknown device '{want}' (use auto, cuda or cpu).")
    has_cuda = torch.cuda.is_available()
    if want == 'cuda' and not has_cuda:
        raise SystemExit('CUDA was asked for (--device cuda / RCF_DEVICE=cuda) but this PyTorch cannot use a GPU.')
    device = 'cuda' if (want == 'cuda' or (want == 'auto' and has_cuda)) else 'cpu'
    if device == 'cpu':
        why = 'as requested' if want == 'cpu' else 'no usable CUDA GPU was found'
        print(f'CoVisPose runs on the CPU ({why}): {torch.get_num_threads()} threads; expect several seconds per pair (about 10 s on 2 cores).', flush=True)
    return device


def encode_views(model, img):
    """Column features of one panorama for the TTA views: {False: plain, True: mirrored}, each (1, C, n_cols)."""
    import torch
    n = model.cfg.n_cols
    return {False: model.column_encoder(img, n), True: model.column_encoder(torch.flip(img, dims=[3]), n)}


def forward_cols(model, cols_a, cols_b):
    """CoVisPose.forward from already-encoded column features (same operations after the column encoder)."""
    import torch
    from covispose.geo_pose import geometric_pose
    c, n = model.cfg, model.cfg.n_cols
    fused = model.norm(model.transformer(torch.cat([model.tokens(cols_a, 0), model.tokens(cols_b, 1)], dim=1)))
    a, b, pose = model.heads(fused[:, :n]), model.heads(fused[:, n:]), model.pose_decoder(fused)
    out = {'covis_logit_a': a['covis_logit'], 'covis_logit_b': b['covis_logit'], 'corr_vec_a': a['corr_vec'],
           'corr_vec_b': b['corr_vec'], 'bon_a': a['bon'], 'bon_b': b['bon'], 't': pose['t'], 'yaw_vec': pose['yaw_vec']}
    if c.geo_pose:
        geo = geometric_pose(out, c.geo_max_dist, c.geo_irls_iters, c.geo_irls_c)
        out.update({'geo_t': geo['t'], 'geo_yaw_vec': geo['yaw_vec'], 'geo_weight': geo['weight_sum']})
    return out


def forward_tta_cached(model, feats_a, feats_b, rolls=2, flip=True):
    """covispose.evaluate.forward_tta with the encoder output reused.

    A view is (mirror, roll A by da columns, roll B by db columns). The encoder pads circularly in azimuth, so
    rolling the image by whole feature strides rolls its column features by the same number of columns; the
    mirrored image is encoded separately (a CNN is not mirror-equivariant)."""
    import torch
    from covispose.evaluate import invert_view, tta_views
    n = model.cfg.n_cols
    views, acc = tta_views(n, rolls, flip), {}
    for f, da, db in views:
        o = forward_cols(model, torch.roll(feats_a[f], da, dims=2), torch.roll(feats_b[f], db, dims=2))
        o = invert_view({k: v.float() for k, v in o.items()}, n, f, da, db)
        for k, v in o.items():
            if k.endswith('vec'):
                v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            acc[k] = acc[k] + v if k in acc else v
    return {k: v / len(views) for k, v in acc.items()}


def cache_usable(model, img_w, rolls=2):
    """The roll views must be whole multiples of the encoder stride (32 px) for the feature shift to be exact."""
    from covispose.evaluate import tta_views
    n = model.cfg.n_cols
    px = img_w // n
    return img_w % n == 0 and all((d * px) % 32 == 0 for _, da, db in tta_views(n, rolls, True) for d in (da, db))


def same_prediction(o1, o2, yaw_deg=0.05, t_tol=0.002):
    """Do two forward_tta outputs give the same poses? (decoder and geometric branch)"""
    import math
    import torch
    for yk, tk in (('yaw_vec', 't'), ('geo_yaw_vec', 'geo_t')):
        if yk not in o1:
            continue
        y1, y2 = (torch.atan2(o[yk][0, 1], o[yk][0, 0]).item() for o in (o1, o2))
        dy = abs(math.degrees((y1 - y2 + math.pi) % (2 * math.pi) - math.pi))
        dt = (o1[tk][0] - o2[tk][0]).norm().item()
        if not (dy <= yaw_deg and dt <= t_tol):
            return False, f'{yk}: {dy:.4f} deg, {tk}: {dt:.5f}'
    return True, ''


def infer(args):
    import numpy as np
    import torch
    run = Path(args.run)
    cfg, m = validate(run)
    todo = [p for p in m['pairs'] if not completed(run, p)]
    print(f"Completed {len(m['pairs']) - len(todo)}/{len(m['pairs'])}; remaining {len(todo)}", flush=True)
    if todo:
        device = pick_device(args.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        from covispose.evaluate import load_model_from_ckpt, forward_tta
        from scripts.run_scene_graph import load_pano
        dest = run / 'predictions'
        dest.mkdir(exist_ok=True)
        lock = run / 'inference.lock'
        with lock.open('x', encoding='utf8') as f:   # one inference process per run folder
            f.write(str(os.getpid()))
        try:
            env = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda, device=device,
                       gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None,
                       cpu_threads=torch.get_num_threads(), platform=platform.platform(),
                       allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                       allow_tf32_cudnn=torch.backends.cudnn.allow_tf32, started=stamp())
            write(run / f'run_env_{env["started"]}.json', env)
            model, mc, blob = load_model_from_ckpt(cfg['checkpoint'], device, weights='ema')
            if blob.get('loaded_weights') != 'ema':
                raise RuntimeError('EMA weights not loaded')
            del blob
            model.eval()
            use_cache = not args.no_feature_cache and cache_usable(model, mc.img_w)
            checked = False            # the first pair is computed both ways before the cache is trusted
            print('Encoder features: ' + ('computed once per panorama and reused for every pair' if use_cache
                                          else 'recomputed for every pair (feature cache off)'), flush=True)
            cache, feats, t0 = {}, {}, time.time()
            with torch.inference_mode():
                for i, p in enumerate(todo, 1):
                    result, t1 = dict(p), time.time()
                    try:
                        ims = [m['images'][p[k]] for k in ('a', 'b')]
                        if any(im['status'] != 'ok' for im in ims):
                            raise ValueError('Input image failed preparation audit')
                        if p['group_id'] not in cache:
                            cache, feats = {p['group_id']: {}}, {}      # one group in memory at a time
                        gc = cache[p['group_id']]
                        for im in ims:
                            if im['id'] not in gc:
                                gc[im['id']] = load_pano(str(local(im['path'])), mc.img_h, mc.img_w)[None]
                        ia, ib = [gc[im['id']].to(device) for im in ims]
                        if use_cache:
                            for im, x in zip(ims, (ia, ib)):
                                if im['id'] not in feats:
                                    feats[im['id']] = encode_views(model, x)
                            o = forward_tta_cached(model, feats[ims[0]['id']], feats[ims[1]['id']], rolls=2, flip=True)
                            if not checked:
                                same, why = same_prediction(o, forward_tta(model, ia, ib, rolls=2, flip=True))
                                checked = True
                                if not same:
                                    print(f'WARNING: reused encoder features differ from the direct computation ({why}); '
                                          'feature cache switched off for this run.', flush=True)
                                    use_cache, feats = False, {}
                        if not use_cache:
                            o = forward_tta(model, ia, ib, rolls=2, flip=True)
                        arr = lambda k: o[k][0].float().cpu().numpy()
                        h = {}
                        for s in ('a', 'b'):
                            h['covis_' + s] = o['covis_logit_' + s][0].sigmoid().float().cpu().numpy()
                            v = arr('corr_vec_' + s)
                            h['corr_' + s] = np.arctan2(v[:, 1], v[:, 0])
                            h['bon_' + s] = arr('bon_' + s)
                        v, gv = arr('yaw_vec'), arr('geo_yaw_vec')
                        yaw, t = float(np.arctan2(v[1], v[0])), arr('t')
                        gyaw, gt = float(np.arctan2(gv[1], gv[0])), arr('geo_t')
                        if not all(np.isfinite(x).all() for x in list(h.values()) + [t, gt, [yaw, gyaw]]):
                            raise ValueError('Nonfinite prediction')
                        target = dest / (p['pair_id'] + '.npz')
                        tmp = target.with_suffix('.tmp')
                        with tmp.open('wb') as f:
                            np.savez_compressed(f, **h)
                        os.replace(tmp, target)
                        result.update(status='ok', yaw=yaw, t=t.tolist(), geo_yaw=gyaw, geo_t=gt.tolist(),
                                      geo_weight=float(arr('geo_weight').reshape(-1)[0]),
                                      heads_sha256=digest(target), seconds=round(time.time() - t1, 3))
                        del o, ia, ib
                    except Exception as exc:
                        result.update(status='error', reason=f'{type(exc).__name__}: {exc}')
                        feats = {}
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                    write(dest / (p['pair_id'] + '.json'), result)
                    eta = (time.time() - t0) / i * (len(todo) - i)
                    print(f"[{i}/{len(todo)}] {p['group_id']} {p['pair_id']}: {result['status']}  ETA {eta/60:.1f} min", flush=True)
        finally:
            lock.unlink(missing_ok=True)
    bad = [p['pair_id'] for p in m['pairs'] if not completed(run, p)]
    write(run / 'inference_status.json', dict(total=len(m['pairs']), complete=len(m['pairs']) - len(bad),
                                              pending_or_error=bad, updated=stamp()))
    if bad:
        raise SystemExit(f'{len(bad)} pairs failed or pending; rerun the same command. See predictions/*.json.')
    print('INFERENCE COMPLETE', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='action', required=True)
    r = sub.add_parser('run', help='inference for a run folder prepared by pipeline.prepare')
    r.add_argument('--run', required=True)
    r.add_argument('--no-feature-cache', action='store_true',
                   help='encode both panoramas again for every pair (the original, slower way)')
    r.add_argument('--device', default='auto', choices=('auto', 'cuda', 'cpu'),
                   help='auto (default): CUDA when available, else CPU; env RCF_DEVICE also sets it')
    args = ap.parse_args()
    infer(args)


if __name__ == '__main__':
    main()
