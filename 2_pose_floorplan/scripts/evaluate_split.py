"""Final evaluation of a checkpoint on one split.

Reports the three per-column heads and BOTH pose estimators (the conv decoder
and RANSAC over the predicted correspondences), overall and bucketed by
co-visibility and by baseline length.

Run it on val and on test and compare the two -- see docs/PHASE4_NOTES.md. The
val split is not fully independent any more (checkpoint selection used it), so
test is the honest number.

    python scripts/evaluate_split.py --split val
    python scripts/evaluate_split.py --split test
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from covispose.dataset import make_loader
from covispose.evaluate import (
    collect_predictions, head_metrics, load_model_from_ckpt, pose_errors,
    run_ransac_over_cache, summarize_pose,
)

# Chosen on the VAL split only (scripts/tune_ransac.py), see assumption A14b.
RANSAC_THRESHOLD = 0.30


def bucket_report(name, edges, values, errs_dec, errs_ran, keep):
    print(f"\n  by {name}:")
    print(f"    {'bucket':>14} {'n':>6} | {'dec t_med':>9} {'dec yaw':>8} {'dec acc':>8} | "
          f"{'ran t_med':>9} {'ran yaw':>8} {'ran acc':>8}")
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (values >= lo) & (values < hi) & keep
        if m.sum() < 10:
            continue
        d = summarize_pose(errs_dec["t_err"], errs_dec["yaw_err_deg"], m)
        r = summarize_pose(errs_ran["t_err"], errs_ran["yaw_err_deg"], m)
        print(f"    [{lo:5.2f},{hi:5.2f}) {m.sum():6d} | "
              f"{d['t_median']:9.4f} {d['yaw_median']:8.3f} {d['acc_0.5_10deg']:8.4f} | "
              f"{r['t_median']:9.4f} {r['yaw_median']:8.3f} {r['acc_0.5_10deg']:8.4f}")
        rows.append({"lo": lo, "hi": hi, "n": int(m.sum()), "decoder": d, "ransac": r})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/covispose/epoch_015.pt")
    ap.add_argument("--split", choices=["val", "test", "train"], default="test")
    ap.add_argument("--pair-index", default=None)
    ap.add_argument("--data-root", default=os.environ.get("ZIND_ROOT", r"C:\zind_data"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--n-pairs", type=int, default=0, help="0 = whole split")
    ap.add_argument("--ransac-threshold", type=float, default=RANSAC_THRESHOLD)
    ap.add_argument("--out-dir", default="outputs/eval")
    ap.add_argument("--weights", choices=["auto", "ema", "raw"], default="auto",
                    help="auto = EMA weights when the checkpoint has them")
    ap.add_argument("--tta-rolls", type=int, default=1,
                    help="average over this many column-rolled views (1 = off)")
    ap.add_argument("--tta-flip", action="store_true",
                    help="also average over mirrored views (only meaningful for "
                         "models trained with flip_prob > 0)")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (default fp32)")
    ap.add_argument("--tag", default="", help="suffix for the output json name")
    ap.add_argument("--dump", action="store_true",
                    help="also write per-pair predictions to <split><tag>_pairs.npz, so the "
                         "results can be re-scored under a different protocol later")
    args = ap.parse_args()

    index = args.pair_index or f"training/pairs/{args.split}_c10neg.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mcfg, blob = load_model_from_ckpt(args.ckpt, device, args.weights)
    ds_cfg = (blob.get("config") or {}).get("data", {})
    print(f"checkpoint : {args.ckpt} (epoch {blob.get('epoch')}, step {blob.get('global_step')}, "
          f"{blob['loaded_weights']} weights)")
    print(f"split      : {args.split}  ({index})")

    loader = make_loader(
        index, args.data_root, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, augment=False,
        pin_memory=False,  # see make_loader: pinned memory runs out over a long pass
        img_h=mcfg.img_h, img_w=mcfg.img_w, n_cols=mcfg.n_cols,
        subsample_n=args.n_pairs, subsample_seed=777,
        # decode exactly the way the model saw images in training
        fast_jpeg=bool(ds_cfg.get("fast_jpeg", False)),
    )
    n = len(loader.dataset)
    print(f"pairs      : {n}")
    print(f"ransac thr : {args.ransac_threshold}")
    print(f"TTA        : rolls={args.tta_rolls} flip={args.tta_flip}\n")

    print(f"collecting predictions ({'bf16' if args.amp else 'fp32'})...")
    P = collect_predictions(model, loader, device, amp=args.amp, progress_every=200,
                            tta_rolls=args.tta_rolls, tta_flip=args.tta_flip)

    print("\n=== per-column heads ===")
    hm = head_metrics(P)
    for k, v in hm.items():
        print(f"  {k:16s} {v:.4f}")

    # Pose is reported over the pairs it is defined for: not negatives (A19).
    pv = P.pose_valid
    n_neg = int((~pv).sum())
    if n_neg:
        frac = 0.5 * ((P.covis_a > 0.5).mean(1) + (P.covis_b > 0.5).mean(1))
        print(f"\n=== negatives ({n_neg} pairs: rooms that share nothing) ===")
        print(f"  predicted co-visible fraction: negatives mean {frac[~pv].mean():.4f} "
              f"median {np.median(frac[~pv]):.4f}  |  positives mean {frac[pv].mean():.4f}")
        for thr in (0.05, 0.10, 0.20):
            print(f"  negatives predicted > {thr:.2f} co-visible: {100*(frac[~pv] > thr).mean():5.1f} %")

    dec = pose_errors(P.t_pred, P.yaw_pred, P.gt_t, P.gt_yaw)
    d_all = summarize_pose(dec["t_err"], dec["yaw_err_deg"], pv)

    print(f"\nrunning RANSAC over {n} pairs...")
    R = run_ransac_over_cache(P, threshold=args.ransac_threshold,
                              progress_every=max(n // 10, 1))
    ran = pose_errors(R["t"], R["yaw"], P.gt_t, P.gt_yaw)
    r_all = summarize_pose(ran["t_err"], ran["yaw_err_deg"], R["ok"] & pv)
    r_all["solved_frac"] = float(R["ok"][pv].mean())
    d_all["solved_frac"] = 1.0

    estimators = [("pose decoder", d_all), (f"RANSAC @{args.ransac_threshold}", r_all)]
    g_all = None
    if P.geo_t is not None:
        geo = pose_errors(P.geo_t, P.geo_yaw, P.gt_t, P.gt_yaw)
        g_all = summarize_pose(geo["t_err"], geo["yaw_err_deg"], pv)
        estimators.append(("geometric (Procrustes)", g_all))

    print("\n=== pose ===")
    print(f"  {'':22s} {'t_mean':>8} {'t_med':>8} {'yaw_mean':>9} {'yaw_med':>8} "
          f"{'acc(.1,1)':>10} {'acc(.5,10)':>11} {'acc(1,20)':>10}")
    for label, s in estimators:
        print(f"  {label:22s} {s['t_mean']:8.4f} {s['t_median']:8.4f} {s['yaw_mean']:9.3f} "
              f"{s['yaw_median']:8.3f} {s['acc_0.1m_1deg']:10.4f} {s['acc_0.5_10deg']:11.4f} "
              f"{s['acc_1.0_20deg']:10.4f}")
    rok = R["ok"] & pv
    print(f"  RANSAC inliers: {R['n_inliers'][rok].mean():.1f} of "
          f"{R['n_proposed'][rok].mean():.1f} proposed "
          f"({100*R['n_inliers'][rok].mean()/max(R['n_proposed'][rok].mean(),1):.1f}%)")

    covis_rows = bucket_report("ZInD covis_score",
                               [0.1, 0.25, 0.3, 0.5, 0.7, 0.85, 0.95, 1.01],
                               P.covis_score, dec, ran, pv)
    base_rows = bucket_report("baseline |t| (camera heights)", [0.0, 1.0, 2.0, 3.0, 5.0, 100.0],
                              P.baseline, dec, ran, pv)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.split}{args.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "ckpt": args.ckpt, "split": args.split, "n_pairs": int(n),
            "weights": blob["loaded_weights"],
            "tta_rolls": args.tta_rolls, "tta_flip": args.tta_flip,
            "ransac_threshold": args.ransac_threshold,
            "heads": hm, "decoder": d_all, "ransac": r_all, "geometric": g_all,
            "ransac_mean_inliers": float(R["n_inliers"][rok].mean()),
            "ransac_mean_proposed": float(R["n_proposed"][rok].mean()),
            "n_negatives": n_neg,
            "by_covis": covis_rows, "by_baseline": base_rows,
        }, f, indent=2)
    print(f"\n-> {out}")

    if args.dump:
        # Everything needed to re-score under another protocol (the paper's
        # covis buckets, translation direction error, errors in metres) without
        # running the network again. Row i is loader.dataset.pairs[i].
        pairs = loader.dataset.pairs
        extra = {}
        if P.geo_t is not None:
            extra = {"geo_t": P.geo_t, "geo_yaw": P.geo_yaw}
        npz = os.path.join(args.out_dir, f"{args.split}{args.tag}_pairs.npz")
        np.savez_compressed(
            npz,
            t_pred=P.t_pred, yaw_pred=P.yaw_pred, gt_t=P.gt_t, gt_yaw=P.gt_yaw,
            ransac_t=R["t"], ransac_yaw=R["yaw"], ransac_ok=R["ok"],
            covis_score=P.covis_score, baseline=P.baseline,
            house_id=np.array([p.house_id for p in pairs]),
            floor_id=np.array([p.floor_id for p in pairs]),
            key_a=np.array([p.key_a for p in pairs]),
            key_b=np.array([p.key_b for p in pairs]),
            kind=np.array([p.kind for p in pairs]),
            covis_frac_pred=0.5 * ((P.covis_a > 0.5).mean(1) + (P.covis_b > 0.5).mean(1)),
            **extra)
        print(f"-> {npz}")


if __name__ == "__main__":
    main()
