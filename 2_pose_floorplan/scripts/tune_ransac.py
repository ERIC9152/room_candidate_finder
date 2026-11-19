"""Re-tune the RANSAC inlier threshold against PREDICTED inputs (assumption A14b).

The Phase-2 default (0.10) was chosen by feeding the estimator ground-truth
per-column values, where the residual noise floor is 0.015. The trained
network's boundary head has ~2.0 deg of error, so the reconstructed points carry
far more slack than that, and the GT-tuned threshold rejects almost everything.

**This tunes on the VALIDATION split only.** Tuning on test would make the
final test numbers meaningless.

    python scripts/tune_ransac.py --ckpt ckpt/proto_rtx1070/keep/epoch_005.pt
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
    collect_predictions, load_model_from_ckpt, pose_errors,
    run_ransac_over_cache, summarize_pose,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/covispose/epoch_015.pt")
    ap.add_argument("--pair-index", default="training/pairs/val_c10neg.csv",
                    help="MUST be the val split -- never tune on test")
    ap.add_argument("--data-root", default=os.environ.get("ZIND_ROOT", r"C:\zind_data"))
    ap.add_argument("--n-pairs", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default="outputs/ransac_tuning.json")
    args = ap.parse_args()

    assert "test" not in os.path.basename(args.pair_index), (
        "refusing to tune hyper-parameters on the test split"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mcfg, blob = load_model_from_ckpt(args.ckpt, device)
    ds_cfg = (blob.get("config") or {}).get("data", {})
    print(f"checkpoint : {args.ckpt} (epoch {blob.get('epoch')}, step {blob.get('global_step')})")

    loader = make_loader(
        args.pair_index, args.data_root, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, augment=False, pin_memory=False,
        img_h=mcfg.img_h, img_w=mcfg.img_w, n_cols=mcfg.n_cols,
        subsample_n=args.n_pairs, subsample_seed=4242,
        # decode exactly the way the model saw images in training
        fast_jpeg=bool(ds_cfg.get("fast_jpeg", False)),
    )
    print(f"tuning on  : {len(loader.dataset)} pairs from {args.pair_index}")

    print("running the network once and caching predictions...")
    P = collect_predictions(model, loader, device, amp=False, progress_every=40)
    print(f"cached {len(P)} pairs\n")

    dec = pose_errors(P.t_pred, P.yaw_pred, P.gt_t, P.gt_yaw)
    d = summarize_pose(dec["t_err"], dec["yaw_err_deg"])
    print(f"pose decoder (for reference): t_med={d['t_median']:.4f} "
          f"yaw_med={d['yaw_median']:.3f}deg  acc@(0.5,10deg)={d['acc_0.5_10deg']:.4f}\n")

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]
    covis_threshs = [0.5]

    print(f"{'thresh':>7} {'covis':>6} {'inliers':>8} {'inl%':>6} | "
          f"{'t_med':>7} {'yaw_med':>8} {'acc(.5,10)':>11} {'acc(1,20)':>10} {'solved':>7}")
    print("-" * 82)
    results = []
    for ct in covis_threshs:
        for th in thresholds:
            R = run_ransac_over_cache(P, threshold=th, covis_thresh=ct)
            e = pose_errors(R["t"], R["yaw"], P.gt_t, P.gt_yaw)
            s = summarize_pose(e["t_err"], e["yaw_err_deg"], R["ok"])
            inl = R["n_inliers"][R["ok"]].mean() if R["ok"].any() else float("nan")
            prop = R["n_proposed"][R["ok"]].mean() if R["ok"].any() else float("nan")
            frac = inl / max(prop, 1)
            print(f"{th:7.2f} {ct:6.2f} {inl:8.1f} {100*frac:5.1f}% | "
                  f"{s['t_median']:7.4f} {s['yaw_median']:8.3f} {s['acc_0.5_10deg']:11.4f} "
                  f"{s['acc_1.0_20deg']:10.4f} {s['solved_frac']:7.4f}")
            results.append({"threshold": th, "covis_thresh": ct,
                            "mean_inliers": float(inl), "mean_proposed": float(prop),
                            **s})

    best = max(results, key=lambda r: r["acc_0.5_10deg"])
    best_med = min(results, key=lambda r: r["yaw_median"])
    print(f"\nbest by acc@(0.5, 10deg) : threshold={best['threshold']} "
          f"-> {best['acc_0.5_10deg']:.4f}")
    print(f"best by median yaw error : threshold={best_med['threshold']} "
          f"-> {best_med['yaw_median']:.3f} deg")
    print(f"Phase-2 default was 0.10 : acc={next(r for r in results if r['threshold']==0.10)['acc_0.5_10deg']:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"ckpt": args.ckpt, "pair_index": args.pair_index,
                   "n_pairs": len(P), "decoder_reference": d,
                   "sweep": results, "best_by_acc": best}, f, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
