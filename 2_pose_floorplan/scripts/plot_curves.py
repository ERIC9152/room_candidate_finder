"""Plot per-head training / validation curves from a run's log.jsonl.

Deliberately one panel per head rather than a single total, because a falling
total can hide a head that is doing nothing.

    python scripts/plot_curves.py --run-dir ckpt/proto_rtx1070
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TERMS = [
    ("total", "total loss", None),
    ("covis", "co-visibility (BCE)", None),
    ("corr", "correspondence (circular L1, rad)", None),
    ("bon", "floor boundary (L1, rad)", None),
    ("pose_t", "translation (L1, camera heights)", None),
    ("pose_yaw", "yaw (circular L1, rad)", None),
]

METRICS = [
    ("covis_acc", "co-visibility accuracy", "covis_acc", True),
    ("corr_err_deg", "correspondence error (deg)", "corr_err_deg", False),
    ("bon_err_deg", "boundary error (deg)", "bon_err_deg", False),
    ("t_err_mean", "translation error (camera heights)", "t_err_mean", False),
    ("yaw_err_deg_mean", "yaw error (deg)", "yaw_err_deg_mean", False),
]


def read_log(path):
    train, quickval, epoch, baseline = [], [], [], None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            k = r.get("kind")
            if k == "train":
                train.append(r)
            elif k == "quickval":
                quickval.append(r)
            elif k == "epoch":
                epoch.append(r)
            elif k == "baseline":
                baseline = r
    return train, quickval, epoch, baseline


def smooth(y, w):
    if w <= 1 or len(y) < w:
        return y
    c = np.convolve(y, np.ones(w) / w, mode="valid")
    return np.concatenate([y[: w - 1], c])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="ckpt/proto_rtx1070")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smooth", type=int, default=101)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log_path = os.path.join(args.run_dir, "log.jsonl")
    train, quickval, epochs, baseline = read_log(log_path)
    print(f"{len(train)} train records, {len(quickval)} quickval, {len(epochs)} epoch")
    if not train:
        print("nothing to plot")
        return

    out = args.out or os.path.join(args.run_dir, "curves.png")
    ts = np.array([r["step"] for r in train])
    vs = np.array([r["step"] for r in quickval]) if quickval else np.array([])
    es = np.array([r["step"] for r in epochs]) if epochs else np.array([])

    # 6 loss terms + lr + grad norm + 5 metrics = 13 panels
    fig, axes = plt.subplots(5, 3, figsize=(17, 17))
    axes = axes.ravel()

    for i, (key, label, _) in enumerate(TERMS):
        ax = axes[i]
        y = np.array([r[f"train_{key}"] for r in train])
        ax.plot(ts, y, color="#c9c9c9", lw=0.6, label="train (raw)")
        ax.plot(ts, smooth(y, args.smooth), color="#1f77b4", lw=1.6, label="train (smoothed)")
        if quickval:
            ax.plot(vs, [r[f"val_{key}"] for r in quickval], "o-", color="#d62728",
                    ms=3, lw=1.2, label="val (periodic)")
        if epochs:
            ax.plot(es, [r[f"val_{key}"] for r in epochs], "s", color="#111111",
                    ms=8, label="val (epoch end)")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    # learning rate
    ax = axes[6]
    ax.plot(ts, [r["lr"] for r in train], color="#2ca02c")
    ax.set_title("learning rate", fontsize=10)
    ax.set_xlabel("step")
    ax.grid(alpha=0.25)

    # gradient norm
    ax = axes[7]
    gn = np.array([r.get("grad_norm", np.nan) for r in train])
    ax.plot(ts, gn, color="#c9c9c9", lw=0.5)
    ax.plot(ts, smooth(gn, args.smooth), color="#9467bd", lw=1.5)
    ax.set_yscale("log")
    ax.set_title("gradient norm (pre-clip)", fontsize=10)
    ax.set_xlabel("step")
    ax.grid(alpha=0.25)

    # interpretable metrics vs their trivial baselines
    for j, (key, label, mk, higher_better) in enumerate(METRICS):
        ax = axes[8 + j]
        if quickval:
            ax.plot(vs, [r.get(f"m_{mk}", np.nan) for r in quickval], "o-",
                    color="#d62728", ms=3, lw=1.2, label="val")
        if epochs:
            ax.plot(es, [r.get(f"m_{mk}", np.nan) for r in epochs], "s",
                    color="#111111", ms=8, label="val (epoch end)")
        if baseline is not None and mk in baseline:
            ax.axhline(baseline[mk], color="#ff7f0e", ls="--", lw=1.5,
                       label=f"trivial baseline ({baseline[mk]:.3g})")
        ax.set_title(label + ("  (higher better)" if higher_better else "  (lower better)"),
                     fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    for k in range(8 + len(METRICS), len(axes)):
        axes[k].axis("off")

    fig.suptitle(f"CoVisPose training curves -- {args.run_dir}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out, dpi=110)
    print(f"-> {out}")

    # a compact text summary too
    if epochs:
        print("\nepoch-end validation:")
        for r in epochs:
            print(f"  epoch {r['epoch']}: total={r['val_total']:.4f} "
                  f"covis={r['val_covis']:.4f} corr={r['val_corr']:.4f} "
                  f"bon={r['val_bon']:.4f} t={r['val_pose_t']:.4f} yaw={r['val_pose_yaw']:.4f}")
            print(f"           covis_acc={r.get('m_covis_acc', float('nan')):.4f} "
                  f"corr_err={r.get('m_corr_err_deg', float('nan')):.2f}d "
                  f"bon_err={r.get('m_bon_err_deg', float('nan')):.2f}d "
                  f"t_err={r.get('m_t_err_mean', float('nan')):.4f} "
                  f"yaw_err={r.get('m_yaw_err_deg_mean', float('nan')):.2f}d")
            if r.get("warnings"):
                for w in r["warnings"]:
                    print(f"           [WARN] {w}")


if __name__ == "__main__":
    main()
