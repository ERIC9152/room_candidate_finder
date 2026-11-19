"""Step 2b (GPU): official HorizonNet layout for every panorama of a run (MP3D checkpoint, VP-aligned).

  python -m pipeline.horizonnet --run runs\\<name> --hn-python <HorizonNet env python.exe>

Same procedure as room_stitch_mvp/scripts/precompute_review_hn.py: HorizonNet's own preprocess.py (LSD vanishing
points, full 3D rotation) -> inference.py on the aligned image -> corners mapped back to the ORIGINAL image frame.
Writes runs\\<name>\\horizonnet\\<id>.json. Resumable: panoramas with a result are skipped.
The HorizonNet code runs in its own Python environment (Python 3.9, torch 1.12) as a subprocess.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import common as C  # noqa: E402
from pipeline import equirect as eq  # noqa: E402


# HorizonNet/misc/pano_lsd_align uses xyz = [cos v sin u, cos v cos u, sin v] (x, forward, up) with the same
# pixel->azimuth definition as equirect.py, i.e. theirs = [d_x, d_z, d_y].
def _ours_to_lsd(d):
    return np.asarray(d)[..., [0, 2, 1]]


def _lsd_to_ours(d):
    return np.asarray(d)[..., [0, 2, 1]]


def vp_rotation(vp):
    """R used by preprocess.py: aligned direction = R @ original direction (lsd axes)."""
    return np.linalg.inv(np.asarray(vp)[2::-1].T)


def rotation_diagnostics(R):
    up = np.array([0.0, 0.0, 1.0])
    tilt = float(np.degrees(np.arccos(np.clip(abs(R @ up @ up), -1, 1))))
    f = R @ np.array([0.0, 1.0, 0.0])
    return {"tilt_deg": tilt, "yaw_deg": float(np.degrees(np.arctan2(f[0], f[1]))), "det": float(np.linalg.det(R))}


def unalign(d, R):
    """HorizonNet result on the VP-aligned image -> original image frame."""
    uv = np.asarray(d["uv"], float)
    ratio = d["z0"] / -d["z1"]
    out, Rinv = [], np.linalg.inv(R)
    for i in range(0, len(uv), 2):
        (uc, vc), (uf, vf) = uv[i], uv[i + 1]
        if vc > vf:
            (uc, vc), (uf, vf) = (uf, vf), (uc, vc)
        xz = eq.floor_uv_to_xy(uf, vf, 1.0)
        for y in (ratio, -1.0):
            p_old = _lsd_to_ours(Rinv @ _ours_to_lsd(np.array([xz[0], y, xz[1]])))
            un, vn = eq.ray_to_uv(p_old)
            out.append([float(un), float(vn)])
    return {**d, "uv": out}


def run_cmd(cmd, cwd, log_path, tmp):
    env = dict(os.environ)
    env["TEMP"] = env["TMP"] = env["HN_ASCII_TMP"] = str(tmp)   # scipy etc. fail on non-ASCII TEMP paths (e.g. a Chinese user name)
    th = C.ROOT / 'models' / 'torch_home'      # ImageNet ResNet-50 that HorizonNet builds before loading its checkpoint (Setup.ps1)
    if th.is_dir():
        env["TORCH_HOME"] = str(th)            # otherwise the user's normal torch cache is used
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    Path(log_path).write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"{Path(cmd[1]).name} failed (rc={r.returncode}), see {log_path}:\n{r.stderr[-1500:]}")
    return dict(command=[str(c) for c in cmd], returncode=r.returncode, runtime_s=round(time.time() - t0, 2))


def horizonnet(run, hn_python, batch=40, no_cuda=False):
    run = C.local(run)
    m = C.read(run / 'manifest.json')
    out = run / 'horizonnet'
    out.mkdir(exist_ok=True)
    repo, ckpt = C.local(C.HN_REPO), C.local(C.HN_CHECKPOINT)
    tmp = C.ROOT / 'tmp'
    tmp.mkdir(exist_ok=True)
    if not no_cuda:       # HorizonNet's inference.py has no automatic fallback: ask its own torch
        probe = subprocess.run([hn_python, '-c', 'import torch; print(torch.cuda.is_available())'], capture_output=True, text=True)
        if probe.returncode == 0 and probe.stdout.strip() == 'False':
            print('HorizonNet: no usable CUDA GPU for this environment; running on the CPU (slower).')
            no_cuda = True
    if not str(tmp).isascii():
        print('WARNING: the package path contains non-ASCII characters; HorizonNet may fail. Move the folder to an ASCII path.')
    todo = [u for u, im in sorted(m['images'].items()) if im['status'] == 'ok' and not (out / f'{u}.json').exists()]
    print(f"HorizonNet: {len(m['images'])} panoramas, {len(todo)} to run")
    for k in range(0, len(todo), batch):
        chunk = todo[k:k + batch]
        work = out / '_work'
        if work.exists():
            shutil.rmtree(work)
        inp, pre, al = work / '_input', work / '_preprocessed', work / '_pass_aligned'
        for d in (inp, pre, al):
            d.mkdir(parents=True)
        for u in chunk:
            src = C.local(m['images'][u]['path'])
            try:
                os.link(src, inp / f'{u}.jpg')
            except OSError:
                shutil.copy2(src, inp / f'{u}.jpg')
        prov = dict(method='official HorizonNet preprocess.py (VP alignment) + inference.py, mapped back to the original frame',
                    checkpoint=C.HN_CHECKPOINT, checkpoint_sha256=C.digest(C.HN_CHECKPOINT), started=C.stamp(), passes={})
        prov['passes']['preprocess'] = run_cmd([hn_python, 'preprocess.py', '--img_glob', str(inp / '*.jpg'), '--output_dir', str(pre)],
                                               repo, work / 'hn_preprocess.log', tmp)
        prov['passes']['inference'] = run_cmd([hn_python, 'inference.py', '--pth', str(ckpt), '--img_glob', str(pre / '*_aligned_rgb.png'),
                                               '--output_dir', str(al), '--visualize'] + (['--no_cuda'] if no_cuda else []), repo, work / 'hn_inference.log', tmp)
        per = {}
        for u in chunk:
            info, vpf, f = {}, pre / f'{u}_VP.txt', al / f'{u}_aligned_rgb.json'
            if not vpf.exists():
                info['error'] = 'preprocess.py produced no VP file'
            elif not f.exists():
                info['error'] = 'inference.py produced no JSON for the aligned image'
            else:
                vp = np.loadtxt(vpf)
                R = vp_rotation(vp)
                info.update(vp=vp.tolist(), R=R.tolist(), **rotation_diagnostics(R))
                d = unalign(json.loads(f.read_text(encoding='utf-8')), R)
                d['room_stitch_alignment'] = {**info, 'note': 'uv mapped back to the ORIGINAL image frame'}
                (out / f'{u}.json').write_text(json.dumps(d), encoding='utf-8')
            per[u] = info
        prov['per_image'] = per
        C.write(out / f'provenance_{C.stamp()}_{k // batch:03d}.json', prov)
        shutil.rmtree(work)
        print(f"  batch {k // batch + 1}: {sum('error' not in v for v in per.values())}/{len(chunk)} ok")
    missing = [u for u, im in m['images'].items() if im['status'] == 'ok' and not (out / f'{u}.json').exists()]
    C.write(out / 'status.json', dict(total=len(m['images']), missing=missing, updated=C.stamp()))
    if missing:
        print(f'WARNING: no HorizonNet result for {len(missing)} panoramas (their CoVisPose outline is used instead): {missing[:5]}')
    print('HORIZONNET DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--hn-python', required=True)
    ap.add_argument('--no-cuda', action='store_true', help='run HorizonNet on the CPU (slow; for GPUs torch 1.12 does not support)')
    a = ap.parse_args()
    horizonnet(a.run, a.hn_python, no_cuda=a.no_cuda)
