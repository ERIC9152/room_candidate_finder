"""Shared helpers. Every path is resolved against the package root unless absolute."""
import csv
import hashlib
import json
import math
import ntpath
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # package root (contains pipeline/, covispose/, scripts/)
# CoVisPose weights, relative to ROOT. Inference only reads the EMA weights and the model config, so either file works:
#   epoch_015.pt      full training checkpoint (1.4 GB: raw + EMA weights, optimizer, scheduler, RNG state)
#   epoch_015_ema.pt  inference-only export of the same EMA weights (0.35 GB, tools: export_inference_weights.py)
# The full checkpoint is used when it is there; the export is the fallback (smaller package to hand over).
CHECKPOINT_FULL = 'models/covispose/epoch_015.pt'
CHECKPOINT_EMA = 'models/covispose/epoch_015_ema.pt'
CHECKPOINT = CHECKPOINT_EMA if ((ROOT / CHECKPOINT_EMA).exists() and not (ROOT / CHECKPOINT_FULL).exists()) else CHECKPOINT_FULL
HN_CHECKPOINT = 'models/horizonnet/resnet50_rnn__mp3d.pth'
HN_REPO = 'third_party/HorizonNet'


def local(p):
    """Map a stored path to a readable one. Relative paths are relative to the package root.
    MRG_PATH_MAP="C:/x=>/mnt/x;..." maps Windows paths when testing on another OS."""
    s = str(p)
    for item in filter(None, os.environ.get('MRG_PATH_MAP', '').split(';')):
        src, dst = item.split('=>')
        if s.lower().replace('/', '\\').startswith(src.lower().replace('/', '\\')):
            return Path(dst + s[len(src):].replace('\\', '/'))
    q = Path(s.replace('\\', '/')) if not ntpath.isabs(s) else Path(s)
    return q if q.is_absolute() or ntpath.isabs(s) else ROOT / q


def rel_or_abs(p):
    """Store paths inside the package as relative POSIX paths (portable); others unchanged."""
    s = str(p)
    try:
        r = Path(os.path.abspath(local(s))).relative_to(ROOT)
        return r.as_posix()
    except ValueError:
        return s


def read(p):
    return json.loads(local(p).read_text(encoding='utf-8-sig'))


def write(p, data):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    os.replace(tmp, p)


def digest(p):
    h = hashlib.sha256()
    with local(p).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')


def csvwrite(p, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r)) or ['empty']
    with Path(p).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def rot(y):
    c, s = math.cos(y), math.sin(y)
    return np.array([[c, -s], [s, c]])


def wrap(y):
    return (y + math.pi) % (2 * math.pi) - math.pi


def err(p, q):
    """(|yaw difference| in degrees, translation difference) between two relative poses (yaw, t)."""
    return abs(math.degrees(wrap(p[0] - q[0]))), float(np.linalg.norm(np.asarray(p[1], float) - np.asarray(q[1], float)))


def latest(run, prefix):
    c = sorted(Path(run).glob(prefix + '_*'))
    if not c:
        raise SystemExit(f'no {prefix}_* folder in {run}')
    return c[-1]


def mpl_fonts(matplotlib):
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for n in ('Microsoft JhengHei', 'Noto Sans CJK TC', 'Noto Sans CJK JP'):
        if n in names:
            matplotlib.rcParams['font.sans-serif'] = [n, 'DejaVu Sans']
            break
