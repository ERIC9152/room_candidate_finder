"""Step 1 (CPU): room_candidate_finder room_groups.json -> frozen run folder for the CoVisPose GPU step.

  python -m pipeline.prepare --groups <room_groups.json> [--scan-dir <finder scans\\dataset>] [--index <data_index.json>] --output runs\\<name>

The run folder (manifest.json, run_config.json) has exactly the format of manual_room_groups_v2, so the unchanged
`scripts/manual_room_groups_v2.py run --run <folder>` performs inference. Groups that contain a pair judged
different_room (or an export with conflicts) are rejected before anything runs on the GPU.
"""
import argparse
import hashlib
import itertools
import json
import ntpath
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import common as C  # noqa: E402
import scripts.manual_room_groups_v2 as MRG  # noqa: E402  (frozen GPU step; only its source hashes / rules are used here)


def join(*parts):
    s = [str(p).replace('/', '\\') for p in parts]
    out = ntpath.normpath(ntpath.join(*s))
    return out if ntpath.isabs(out) else out.replace('\\', '/')


def prepare(groups_path, output, scan_dir=None, index=None):
    from PIL import Image
    scan_dir = scan_dir or ntpath.dirname(str(groups_path).replace('/', '\\'))
    raw, fm = C.read(groups_path), C.read(join(scan_dir, 'manifest.json'))
    # prefer the root relative to the scan folder (portable when data\ and scans\ move together), else the absolute one
    root = fm['root'] if ntpath.isabs(fm.get('root') or '') else join(scan_dir, fm.get('root') or '.')
    if not C.local(root).is_dir() and fm.get('root_abs'):
        root = fm['root_abs']
    where = {(h, it['id']): (it['path'] if ntpath.isabs(it['path']) else join(root, it['path']))
             for h, items in fm['houses'].items() for it in items}
    groups, problems, seen = raw['groups'], [], set()
    if any(g.get('status') != 'human_verified' for g in groups):
        problems.append('a group is not human_verified')
    for g in groups:
        ms = g['panorama_uuids']
        if len(ms) < 2 or len(set(ms)) != len(ms) or seen.intersection(ms):
            problems.append('duplicate or invalid membership: ' + g['room_id'])
        seen.update(ms)
    if len({g['room_id'] for g in groups}) != len(groups):
        problems.append('duplicate group ids')
    neg = {frozenset((r['uuid_a'], r['uuid_b'])) for r in raw.get('pair_reviews', []) if r.get('relation') == 'different_room'}
    clash = [(g['room_id'], a, b) for g in groups for a, b in itertools.combinations(g['panorama_uuids'], 2) if frozenset((a, b)) in neg]
    if clash or raw.get('conflicts'):
        problems.append(f'different_room pair inside a group, fix it in the review tool first: {clash[:3] or raw["conflicts"][:3]}')
    if not groups:
        problems.append('no groups (nothing judged same_room yet?)')
    if problems:
        raise SystemExit('Input rejected:\n  ' + '\n  '.join(problems))
    labels = {}
    if index:
        idx = C.read(index)['panos']
        labels = {u: (idx.get(u) or {}).get('asteroom_name') or '' for u in seen}
    out = C.local(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'classification_source.json').write_bytes(C.local(groups_path).read_bytes())
    images, failures = {}, []
    for g in groups:
        for u in g['panorama_uuids']:
            path = where.get((g['task'], u))
            rec = dict(id=u, label=labels.get(u, ''), house=g['task'], status='input_error', path=path and C.rel_or_abs(path))
            try:
                if not path:
                    raise ValueError(f"panorama {u} is not in house {g['task']} of the scan manifest")
                with Image.open(C.local(path)) as im:
                    im.load()                          # full decode; truncated JPEGs raise
                    size = list(im.size)
                if size[0] != 2 * size[1]:
                    raise ValueError('not a 2:1 panorama')
                rec.update(status='ok', sha256=C.digest(path), size=size)
            except Exception as exc:
                rec['error'] = f'{type(exc).__name__}: {exc}'
                failures.append(u)
            images[u] = rec
    by_sha = defaultdict(list)
    for u, im in images.items():
        if im['status'] == 'ok':
            by_sha[im['sha256']].append(u)
    duplicates = [sorted(v) for v in by_sha.values() if len(v) > 1]
    dup_pairs = {frozenset(p) for d in duplicates for p in itertools.combinations(d, 2)}
    reviewed = {frozenset((r['uuid_a'], r['uuid_b'])) for r in raw.get('pair_reviews', []) if r.get('relation') == 'same_room'}
    pairs = []
    for g in sorted(groups, key=lambda g: g['room_id']):
        for a, b in itertools.combinations(sorted(g['panorama_uuids']), 2):
            pid = hashlib.sha256(('v2|' + g['room_id'] + '|' + a + '|' + b).encode()).hexdigest()[:24]
            pairs.append(dict(pair_id=pid, group_id=g['room_id'], a=a, b=b, directly_reviewed=frozenset((a, b)) in reviewed,
                              duplicate_content=frozenset((a, b)) in dup_pairs, blind_review=None))
    manifest = dict(groups=[dict(room_id=g['room_id'], task=g['task'], floor=g.get('floor', ''), space_kind=g.get('space_kind', 'room'),
                                 panorama_uuids=g['panorama_uuids'], status=g['status']) for g in sorted(groups, key=lambda g: g['room_id'])],
                    images=images, pairs=pairs, duplicate_images=duplicates)
    C.write(out / 'manifest.json', manifest)
    cfg = dict(stage='room_pose_pipeline (manual_room_groups_v2 compatible)', single_source=True, weights='ema', rolls=2, flip=True,
               precision='fp32; TF32 disabled; no autocast', convention='p_B=R(yaw)*p_A+t; native CoVisPose XY; camera height=1',
               checkpoint=C.CHECKPOINT, checkpoint_sha256=C.digest(C.CHECKPOINT), source_code=MRG.sources(),
               classification_source=str(groups_path), classification_sha256=C.digest(groups_path),
               scan_manifest=join(scan_dir, 'manifest.json'), index=index,
               manifest_sha256=C.digest(out / 'manifest.json'), rules=MRG.RULES, diagnostics=MRG.DIAG)
    C.write(out / 'run_config.json', cfg)
    gs = Counter(len(g['panorama_uuids']) for g in groups)
    audit = dict(groups=len(groups), houses=len({g['task'] for g in groups}), images=len(images), pairs=len(pairs),
                 input_errors={u: images[u]['error'] for u in failures}, group_sizes={str(k): gs[k] for k in sorted(gs)},
                 duplicate_images=duplicates, status='ready_for_gpu' if not failures else 'input_errors_preserved')
    C.write(out / 'preflight.json', audit)
    print(json.dumps(audit, ensure_ascii=False))
    print('PREPARED: ' + str(out))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--groups', required=True)
    ap.add_argument('--scan-dir')
    ap.add_argument('--index')
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    prepare(a.groups, a.output, a.scan_dir, a.index)
