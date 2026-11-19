"""Sanity check: the model must be able to overfit a handful of pairs.

This is the gate before any real training run.  If a network cannot drive the
loss to ~0 on 5 fixed examples, something is wrong with the wiring (a head not
connected, a target mis-shaped, a mask killing every gradient) and no amount of
data will fix it.

We assert on all four outputs separately, because a joint loss going down can
easily hide one dead head.

    python tests/test_overfit.py                # small config, CPU-friendly
    python tests/test_overfit.py --full-config  # the real RTX 1070 config
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from covispose.dataset import PairDataset  # noqa: E402
from covispose.losses import LossWeights, covispose_loss, covispose_metrics  # noqa: E402
from covispose.model import CoVisPoseConfig, build_model  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _requires import needs_zind  # noqa: E402


def run_overfit(
    pair_index: str,
    data_root: str,
    n_pairs: int = 5,
    steps: int = 300,
    lr: float = 3e-4,
    full_config: bool = False,
    device: str = None,
    verbose: bool = True,
    server_options: bool = False,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    if full_config:
        cfg = CoVisPoseConfig()
    elif server_options:
        # small, but with the attention pose decoder and the geometric branch
        cfg = CoVisPoseConfig(
            backbone="resnet18", pretrained=True, img_h=256, img_w=512,
            n_cols=256, d_model=64, n_heads=4, n_layers=2, dim_ff=128, dropout=0.0,
            pose_decoder="attn", geo_pose=True,
        )
    else:
        # small enough to be quick, big enough to exercise every code path
        cfg = CoVisPoseConfig(
            backbone="resnet18", pretrained=True, img_h=256, img_w=512,
            n_cols=128, d_model=64, n_heads=4, n_layers=2, dim_ff=128, dropout=0.0,
        )

    ds = PairDataset(
        pair_index, data_root,
        img_h=cfg.img_h, img_w=cfg.img_w, n_cols=cfg.n_cols,
        augment=False, max_pairs=n_pairs,
    )
    assert len(ds) == n_pairs, f"index only has {len(ds)} pairs"
    batch = torch.utils.data.dataloader.default_collate([ds[i] for i in range(n_pairs)])
    batch = {k: v.to(device) for k, v in batch.items()}

    model = build_model(cfg).to(device)
    model.train()
    # dropout off: we *want* to memorise these five pairs
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    w = LossWeights(geo_t=0.5, geo_yaw=0.5) if cfg.geo_pose else LossWeights()
    opt = torch.optim.AdamW(model.param_groups(lr, backbone_lr_mult=1.0), weight_decay=0.0)

    first, last = None, None
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(batch["img_a"], batch["img_b"])
        terms = covispose_loss(out, batch, w)
        terms["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step == 0:
            first = {k: v.item() for k, v in terms.items()}
            first_m = covispose_metrics(out, batch)
        if verbose and (step % max(steps // 10, 1) == 0 or step == steps - 1):
            m = covispose_metrics(out, batch)
            print(
                f"  step {step:4d}  total={terms['total'].item():7.4f}  "
                f"covis={terms['covis'].item():.4f} corr={terms['corr'].item():.4f} "
                f"bon={terms['bon'].item():.4f} t={terms['pose_t'].item():.4f} "
                f"yaw={terms['pose_yaw'].item():.4f}  |  "
                f"covis_acc={m['covis_acc']:.3f} corr_err={m['corr_err_deg']:5.1f}d "
                f"bon_err={m['bon_err_deg']:5.2f}d t_err={m['t_err_mean']:.3f} "
                f"yaw_err={m['yaw_err_deg_mean']:5.1f}d"
            )
        last = {k: v.item() for k, v in terms.items()}

    model.eval()
    with torch.no_grad():
        out = model(batch["img_a"], batch["img_b"])
        final_m = covispose_metrics(out, batch)
    dt = time.time() - t0
    if verbose:
        print(f"  {steps} steps in {dt:.1f}s ({1000*dt/steps:.0f} ms/step) on {device}")
    return first, last, first_m, final_m, cfg


@needs_zind
def test_overfit_five_pairs(pair_index=None, data_root=os.environ.get("ZIND_ROOT", r"C:\zind_data"),
                            full_config=False, steps=300, server_options=False):
    pair_index = pair_index or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "training", "pairs", "train_c10neg.csv",
    )
    first, last, first_m, m, cfg = run_overfit(
        pair_index, data_root, n_pairs=5, steps=steps, full_config=full_config,
        server_options=server_options,
    )

    print(f"\n  first-step loss: {first}")
    print(f"  last-step  loss: {last}")
    print(f"  final metrics  : { {k: round(v, 4) for k, v in m.items()} }")

    # Every term must have come down a long way -- this is what catches a head
    # that is silently disconnected from the graph.
    for k in ("covis", "corr", "bon", "pose_t", "pose_yaw"):
        assert last[k] < 0.5 * first[k], (
            f"loss term '{k}' did not halve: {first[k]:.4f} -> {last[k]:.4f} "
            "(head not learning?)"
        )
    assert last["total"] < 0.15 * first["total"], (first["total"], last["total"])

    # And the interpretable metrics must actually be good, not just smaller.
    assert m["covis_acc"] > 0.95, m["covis_acc"]
    assert m["corr_err_deg"] < 10.0, m["corr_err_deg"]
    assert m["bon_err_deg"] < 3.0, m["bon_err_deg"]
    assert m["t_err_mean"] < 0.25, m["t_err_mean"]
    assert m["yaw_err_deg_mean"] < 10.0, m["yaw_err_deg_mean"]
    if cfg.geo_pose:
        # the geometric pose must be learnable too, not just finite
        assert last["geo_t"] < 0.5 * first["geo_t"], (first["geo_t"], last["geo_t"])
        assert m["geo_acc_0.5_10"] > 0.5, m["geo_acc_0.5_10"]
    print("\n  ok  test_overfit_five_pairs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-index", default=None)
    ap.add_argument("--data-root", default=os.environ.get("ZIND_ROOT", r"C:\zind_data"))
    ap.add_argument("--full-config", action="store_true")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--server-options", action="store_true",
                    help="small model with the attn pose decoder and geo pose branch")
    a = ap.parse_args()
    test_overfit_five_pairs(a.pair_index, a.data_root, a.full_config, a.steps, a.server_options)
