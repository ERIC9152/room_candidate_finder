"""Build train/val/test pano-pair indices from covis_scores.csv + zind_partition.json.

Every emitted pair is verified to resolve to two parsable panos with a layout
polygon and a pose, and to a pano JPG that exists on disk.

    python scripts/build_pair_index.py --out-dir training/pairs --houses-per-split 30
    python scripts/build_pair_index.py --out-dir training/pairs            # full ZInD

The paper's pair set (A1, A8): co-visibility down to 0.10, plus zero-overlap
negatives for a positive:negative ratio of about 2.5 --

    python scripts/build_pair_index.py --covis-min 0.10 --neg-ratio 0.4         --adj-per-house 40 --tag _c10neg

Negatives are pairs of panos in two complete rooms that share no door or
opening, with zero geometric co-visibility. Pairs in rooms that DO connect go
to a separate ``*_adjacent.csv`` for evaluation only (see covispose/pairs.py).
"""

import argparse
import os
import sys
from collections import Counter, defaultdict
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from covispose.geometry import relative_pose
from covispose.pairs import (
    DEFAULT_COVIS_MIN,
    DEFAULT_MIN_BASELINE,
    PairRef,
    read_covis_scores,
    room_connections,
    write_pair_index,
)
from covispose.targets import compute_pair_targets
from covispose.zind_parse import load_house, load_partition


def sample_cross_room(house, n_neg, n_adj, args, drop):
    """Negatives and door-adjacent pairs for one house, in a fixed random order."""
    if n_neg <= 0 and n_adj <= 0:
        return [], []
    conn = room_connections(house)
    cands = []
    for fid, panos in house.by_floor().items():
        panos = sorted((r for r in panos if os.path.isfile(r.image_path)), key=lambda r: r.key)
        for i, a in enumerate(panos):
            for b in panos[i + 1:]:
                if a.complete_room_id != b.complete_room_id:
                    cands.append((a, b))
    rng = np.random.default_rng(args.seed * 100003 + int(house.house_id))
    neg, adj = [], []
    for i in rng.permutation(len(cands)):
        if len(neg) >= n_neg and len(adj) >= n_adj:
            break
        a, b = cands[i]
        if rng.random() < 0.5:
            a, b = b, a  # the CSV's own pairs come in both orders too
        linked = (a.floor_id,) + tuple(sorted((a.complete_room_id, b.complete_room_id))) in conn
        if (linked and len(adj) >= n_adj) or (not linked and len(neg) >= n_neg):
            continue
        t, yaw, k = relative_pose(a.pose, b.pose)
        baseline = float(np.linalg.norm(t))
        if baseline < args.min_baseline or abs(k - 1.0) > 1e-3:
            drop["cross_room_degenerate"] += 1
            continue
        tg = compute_pair_targets(a, b, n_cols=args.n_cols)
        covis = 0.5 * float(tg.covis_a.mean() + tg.covis_b.mean())
        if not linked and covis > args.neg_max_covis:
            drop["neg_not_zero_covis"] += 1
            continue
        (adj if linked else neg).append(PairRef(
            house_id=house.house_id, floor_id=a.floor_id,
            room_a=a.room_id, pano_a=a.pano_id, room_b=b.room_id, pano_b=b.pano_id,
            covis_score=covis, baseline=baseline, dyaw_deg=float(np.degrees(yaw)),
            kind="adj" if linked else "neg",
        ))
    if len(neg) < n_neg:
        drop["neg_short"] += n_neg - len(neg)
    return neg, adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("ZIND_ROOT", r"C:\zind_data"))
    ap.add_argument("--covis-csv", default=r"C:\zind\covis_scores.csv")
    ap.add_argument("--partition", default=r"C:\zind\zind_partition.json")
    ap.add_argument("--out-dir", default="training/pairs")
    ap.add_argument("--covis-min", type=float, default=DEFAULT_COVIS_MIN)
    ap.add_argument("--min-baseline", type=float, default=DEFAULT_MIN_BASELINE)
    ap.add_argument("--layout", default="layout_visible")
    ap.add_argument(
        "--houses-per-split",
        type=int,
        default=0,
        help="0 = use every house in the official split; >0 = take the first N "
        "(deterministic subset used for the RTX 1070 prototype runs)",
    )
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    ap.add_argument("--neg-ratio", type=float, default=0.0,
                    help="negatives per positive, per house (0.4 = the paper's pos:neg 2.5)")
    ap.add_argument("--neg-max-covis", type=float, default=0.02,
                    help="a negative's geometric co-visibility must not exceed this")
    ap.add_argument("--adj-per-house", type=int, default=0,
                    help="door-adjacent pairs per house, written to *_adjacent.csv")
    ap.add_argument("--n-cols", type=int, default=1024,
                    help="column grid the negatives' co-visibility is checked on")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    partition = load_partition(args.partition)
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {}
    for split in ("train", "val", "test"):
        houses = sorted(partition[split])
        if args.houses_per_split > 0:
            houses = houses[: args.houses_per_split]

        raw = read_covis_scores(args.covis_csv, houses=houses, covis_min=args.covis_min)
        by_house = defaultdict(list)
        for p in raw:
            by_house[p.house_id].append(p)

        kept, drop = [], Counter()
        negs, adjs = [], []
        n_same_croom = 0
        for h in houses:
            if not by_house[h]:
                continue
            try:
                house = load_house(
                    args.data_root, h, layout_key=args.layout,
                    primary_only=False, inside_only=True,
                )
            except Exception as e:  # noqa: BLE001
                drop["house_unreadable"] += len(by_house[h])
                print(f"  [warn] house {h} unreadable: {e}")
                continue
            n_before = len(kept)
            for p in by_house[h]:
                ra, rb = house.panos.get(p.key_a), house.panos.get(p.key_b)
                if ra is None or rb is None:
                    drop["pano_missing_or_filtered"] += 1
                    continue
                if not (os.path.isfile(ra.image_path) and os.path.isfile(rb.image_path)):
                    drop["jpg_missing"] += 1
                    continue
                t, yaw, k = relative_pose(ra.pose, rb.pose)
                baseline = float(np.linalg.norm(t))
                if baseline < args.min_baseline:
                    drop["degenerate_baseline"] += 1
                    continue
                if abs(k - 1.0) > 1e-3:
                    # Would mean the two panos have different camera heights.
                    # Never observed in ZInD, but do not silently mis-label it.
                    drop["scale_ratio_not_1"] += 1
                    continue
                n_same_croom += ra.complete_room_id == rb.complete_room_id
                kept.append(
                    replace(p, baseline=baseline, dyaw_deg=float(np.degrees(yaw)))
                )
            n_neg = int(round(args.neg_ratio * (len(kept) - n_before)))
            ng, ad = sample_cross_room(house, n_neg, args.adj_per_house, args, drop)
            negs += ng
            adjs += ad

        name = f"{split}{args.tag}.csv"
        write_pair_index(os.path.join(args.out_dir, name), kept + negs)
        if args.adj_per_house > 0:
            write_pair_index(os.path.join(args.out_dir, f"{split}{args.tag}_adjacent.csv"), adjs)
        scores = np.array([p.covis_score for p in kept]) if kept else np.zeros(1)
        bl = np.array([p.baseline for p in kept]) if kept else np.zeros(1)
        summary[split] = (len(houses), len(raw), len(kept), dict(drop))
        print(
            f"{split:5s}: houses={len(houses):4d}  rows>={args.covis_min}: {len(raw):7d}"
            f"  kept: {len(kept):7d}  ({100.0*len(kept)/max(len(raw),1):.1f}%)"
            f"  covis mean={scores.mean():.3f}"
            f"  |t| p50={np.percentile(bl,50):.2f} p90={np.percentile(bl,90):.2f}"
            f"  same-complete-room={100.0*n_same_croom/max(len(kept),1):.1f}%"
        )
        if negs or adjs:
            print(f"        + negatives: {len(negs)}  (pos:neg = {len(kept)/max(len(negs),1):.2f})"
                  f"   door-adjacent (eval only): {len(adjs)}")
        if drop:
            print(f"        dropped: {dict(drop)}")
        print(f"        -> {os.path.join(args.out_dir, name)}")

    print("\ndone.")


if __name__ == "__main__":
    main()
