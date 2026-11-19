"""ZInD split -> pipeline input for evaluating pose assembly against ground truth.

  python scripts/make_zind_room_groups.py --data-root C:\\zind_data --partition C:\\zind\\zind_partition.json
         --split test --out evaluation\\zind_test_posegraph [--min-panos 3] [--root-abs C:\\zind_data]

One group per ZInD complete room with >= --min-panos panoramas (primary, inside panoramas, as in training), so a group
is what the review app would export for that room. Writes, into --out:
  room_groups.json  groups (status human_verified) for pipeline.prepare
  manifest.json     scan manifest: root = ZInD data folder, houses = {house: [{id, path}]}
  index.json        {"panos": {id: {"asteroom_name": "<floor> <complete room> <pano>"}}} (labels in the outputs)
  gt.json           ground-truth ZInD floor_plan_transformation of every panorama {id: {translation, rotation, scale}}
Panorama ids are "<house>_<floor>_<partial room>_<pano>" (unique across houses).
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from covispose.zind_parse import load_house, load_partition  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', default=os.environ.get('ZIND_ROOT', r'C:\zind_data'))
    ap.add_argument('--partition', default=r'C:\zind\zind_partition.json')
    ap.add_argument('--split', default='test')
    ap.add_argument('--min-panos', type=int, default=3)
    ap.add_argument('--root-abs', help='data root as written into manifest.json (default: --data-root)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    houses, groups, index, gt = defaultdict(list), [], {}, {}
    for h in load_partition(a.partition)[a.split]:
        H = load_house(a.data_root, h)
        rooms = defaultdict(list)
        for p in H.panos.values():
            rooms[(p.floor_id, p.complete_room_id)].append(p)
        for (floor, room), ps in sorted(rooms.items()):
            if len(ps) < a.min_panos:
                continue
            ids = []
            for p in sorted(ps, key=lambda q: (q.room_id, q.pano_id)):
                u = f'{h}_{floor}_{p.room_id}_{p.pano_id}'
                rel = os.path.relpath(p.image_path, a.data_root).replace(os.sep, '/')
                houses[h].append(dict(id=u, path=rel, w=0, h=0))
                index[u] = dict(asteroom_name=f'{floor} {room} {p.pano_id}')
                gt[u] = dict(translation=p.pose.t.tolist(), rotation=p.pose.rot_deg, scale=p.pose.scale,
                             camera_height=p.camera_height)
                ids.append(u)
            groups.append(dict(room_id=f'zind_{h}_{floor}_{room}', task=h, floor=floor, room_label=room,
                               space_kind='room', panorama_uuids=ids, status='human_verified',
                               notes=f'ZInD {a.split} ground-truth complete room'))
    w = lambda name, d: (out / name).write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    w('room_groups.json', dict(schema_version=1, generator='make_zind_room_groups', groups=groups, conflicts=[], pair_reviews=[]))
    w('manifest.json', dict(root=a.root_abs or a.data_root, houses=houses, skipped=[],
                            note=f'ZInD {a.split}, complete rooms with >= {a.min_panos} panoramas'))
    w('index.json', dict(panos=index))
    w('gt.json', dict(source=f'ZInD {a.split} floor_plan_transformation', panos=gt))
    n = sum(len(g['panorama_uuids']) for g in groups)
    print(f'{len(groups)} groups, {n} panoramas, {sum(len(g["panorama_uuids"]) * (len(g["panorama_uuids"]) - 1) // 2 for g in groups)} pairs -> {out}')


if __name__ == '__main__':
    main()
