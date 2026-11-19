"""Phase-1 mandatory check: render ground-truth overlays for a few pairs.

    python scripts/verify_targets.py --pair-index training/pairs/train_c10neg.csv \
        --out-dir outputs/phase1_verify --n 5
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from covispose.pairs import read_pair_index
from covispose.targets import compute_pair_targets
from covispose.viz import plot_pair_targets
from covispose.zind_parse import load_house


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("ZIND_ROOT", r"C:\zind_data"))
    ap.add_argument("--pair-index", default="training/pairs/train_c10neg.csv")
    ap.add_argument("--out-dir", default="outputs/phase1_verify")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--n-cols", type=int, default=256)
    ap.add_argument("--layout", default="layout_visible")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--spread-covis",
        action="store_true",
        default=True,
        help="pick pairs spanning the covis_score range instead of at random",
    )
    args = ap.parse_args()

    pairs = read_pair_index(args.pair_index)
    print(f"{len(pairs)} pairs in {args.pair_index}")

    rng = np.random.default_rng(args.seed)
    if args.spread_covis:
        order = np.argsort([p.covis_score for p in pairs])
        picks = [pairs[order[int(q * (len(order) - 1))]]
                 for q in np.linspace(0.02, 0.98, args.n)]
    else:
        picks = [pairs[i] for i in rng.choice(len(pairs), args.n, replace=False)]

    os.makedirs(args.out_dir, exist_ok=True)
    cache = {}
    for i, p in enumerate(picks):
        if p.house_id not in cache:
            cache[p.house_id] = load_house(
                args.data_root, p.house_id, layout_key=args.layout,
                primary_only=False, inside_only=True,
            )
        house = cache[p.house_id]
        ra, rb = house.panos[p.key_a], house.panos[p.key_b]
        tg = compute_pair_targets(ra, rb, n_cols=args.n_cols)
        name = f"{i:02d}_{p.house_id}_{p.floor_id}_{p.room_a}_{p.pano_a}__{p.room_b}_{p.pano_b}.png"
        out = plot_pair_targets(ra, rb, tg, covis_score=p.covis_score,
                                out_path=os.path.join(args.out_dir, name))
        print(
            f"[{i}] {p.house_id} {p.key_a} <-> {p.key_b}\n"
            f"     zind covis={p.covis_score:.3f}  ours A={tg.covis_a.mean():.3f} B={tg.covis_b.mean():.3f}\n"
            f"     t=({tg.t_ab[0]:+.3f},{tg.t_ab[1]:+.3f}) |t|={np.linalg.norm(tg.t_ab):.3f} "
            f"yaw={np.degrees(tg.yaw_ab):+.1f}deg k={tg.scale_ab:.4f}\n"
            f"     -> {out}"
        )


if __name__ == "__main__":
    main()
