"""Losses for the three per-column heads and the pose decoder.

Masking rules (these matter, and are easy to get wrong)
------------------------------------------------------
* **co-visibility**  : BCE over *every* column. This is the head that decides
  which columns are valid, so it must see the negatives too.
* **correspondence** : only defined where the column IS co-visible -- a column
  whose content the other pano cannot see has no corresponding azimuth. Those
  columns are masked out entirely; we never invent a target for them.
* **floor boundary** : over every column with a valid polygon hit, co-visible or
  not. The layout task is monocular, it does not depend on the other pano.

Angle handling
--------------
Correspondence and yaw are circular, so they are predicted as unnormalised
(cos, sin) pairs and scored with a circular L1, ``|wrap(pred - gt)|``, computed
through ``atan2`` so it stays differentiable and has no seam at +-pi.
The floor boundary is *not* circular -- it is an elevation confined to
(0, pi/2) -- so it gets a plain L1 in radians.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    """All five weights are engineering guesses -- see docs/ASSUMPTIONS.md A9.

    Rough scale of each raw term at initialisation, which is how these were
    picked: covis BCE ~0.69, corr ~1.57 rad (a uniformly random angle),
    bon ~0.25 rad, pose t ~2.0 camera heights, pose yaw ~1.57 rad.
    """

    covis: float = 1.0
    corr: float = 1.0
    bon: float = 2.0
    pose_t: float = 1.0
    pose_yaw: float = 1.0
    # Differentiable geometric pose (``model.geo_pose``). 0 disables the terms;
    # the training loop ramps them in, because Procrustes over random heads is
    # noise.
    geo_t: float = 0.0
    geo_yaw: float = 0.0
    # BCE ``pos_weight`` for co-visibility. 1.0 = the original unweighted BCE.
    # Assumption A14a is still open, so this is an ablation knob, not a fix.
    covis_pos_weight: float = 1.0


def wrap_pi(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def angle_from_vec(vec: torch.Tensor) -> torch.Tensor:
    """(..., 2) unnormalised (cos, sin) -> angle."""
    return torch.atan2(vec[..., 1], vec[..., 0])


def circular_l1(pred_vec: torch.Tensor, target_angle: torch.Tensor) -> torch.Tensor:
    """Elementwise |wrap(pred - target)| in radians. No reduction."""
    return wrap_pi(angle_from_vec(pred_vec) - target_angle).abs()


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``x`` over ``mask``; exactly 0 (and gradient-free) if empty."""
    # No ``.item()`` branch: that would force a host-device sync on every call.
    # With an empty mask the numerator is exactly 0 and so is its gradient.
    mask = mask.to(x.dtype)
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def _pose_valid(batch: Dict[str, torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    """Per-pair pose mask; all True for batches from indices without negatives."""
    if "pose_valid" in batch:
        return batch["pose_valid"].bool()
    return torch.ones(like.shape[0], dtype=torch.bool, device=like.device)


def covispose_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    w: LossWeights = LossWeights(),
) -> Dict[str, torch.Tensor]:
    """Full multi-task loss. Returns a dict with ``total`` plus each raw term.

    ``out``   : the model's forward() dict.
    ``batch`` : targets from :class:`covispose.dataset.PairDataset`.

    Under AMP the model's outputs arrive as fp16. We upcast them here before
    doing any trigonometry -- ``atan2`` on fp16 loses enough precision to matter
    for an angular regression, and the loss itself is cheap, so there is no
    reason to compute it in half precision.
    """
    out = {
        k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
        for k, v in out.items()
    }
    terms: Dict[str, torch.Tensor] = {}

    # ---- co-visibility: every column ------------------------------------- #
    covis_loss = 0.0
    pos_weight = None
    if w.covis_pos_weight != 1.0:
        pos_weight = out["covis_logit_a"].new_tensor(w.covis_pos_weight)
    for s in ("a", "b"):
        covis_loss = covis_loss + F.binary_cross_entropy_with_logits(
            out[f"covis_logit_{s}"], batch[f"covis_{s}"].to(out[f"covis_logit_{s}"].dtype),
            pos_weight=pos_weight,
        )
    terms["covis"] = covis_loss / 2

    # ---- correspondence: co-visible columns only ------------------------- #
    corr_loss = 0.0
    for s in ("a", "b"):
        # mask = GT co-visibility AND a valid boundary to correspond with
        mask = batch[f"covis_{s}"].bool() & batch[f"bon_valid_{s}"].bool()
        err = circular_l1(out[f"corr_vec_{s}"], batch[f"corr_{s}"])
        corr_loss = corr_loss + _masked_mean(err, mask)
    terms["corr"] = corr_loss / 2

    # ---- floor boundary: every column with a valid polygon hit ----------- #
    bon_loss = 0.0
    for s in ("a", "b"):
        err = (out[f"bon_{s}"] - batch[f"bon_{s}"]).abs()
        bon_loss = bon_loss + _masked_mean(err, batch[f"bon_valid_{s}"].bool())
    terms["bon"] = bon_loss / 2

    # ---- pose ------------------------------------------------------------- #
    # Not on negatives (A19): with nothing in common the pose is unobservable.
    pv = _pose_valid(batch, out["t"])
    terms["pose_t"] = _masked_mean((out["t"] - batch["t"]).abs().sum(-1), pv)
    terms["pose_yaw"] = _masked_mean(circular_l1(out["yaw_vec"], batch["yaw"]), pv)

    total = (
        w.covis * terms["covis"]
        + w.corr * terms["corr"]
        + w.bon * terms["bon"]
        + w.pose_t * terms["pose_t"]
        + w.pose_yaw * terms["pose_yaw"]
    )

    # ---- geometric pose (optional) --------------------------------------- #
    if "geo_t" in out:
        # Pairs whose predicted co-visibility is ~empty have no correspondences
        # to fit; their Procrustes output is undefined, so they are masked.
        ok = (out["geo_weight"] > 1e-3) & pv
        # Masking alone is not enough: a degenerate fit has yaw_vec = (0, 0),
        # and whether atan2's gradient there is 0 or NaN depends on the torch
        # version -- and 0 * NaN is NaN. Keep those rows out of the graph.
        yaw_vec = torch.where(ok.unsqueeze(-1), out["geo_yaw_vec"],
                              out["geo_yaw_vec"].new_tensor([1.0, 0.0]))
        terms["geo_t"] = _masked_mean((out["geo_t"] - batch["t"]).abs().sum(-1), ok)
        terms["geo_yaw"] = _masked_mean(circular_l1(yaw_vec, batch["yaw"]), ok)
        total = total + w.geo_t * terms["geo_t"] + w.geo_yaw * terms["geo_yaw"]

    terms["total"] = total
    return terms


@torch.no_grad()
def covispose_metrics(
    out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """Interpretable numbers for logging (not used for optimisation)."""
    out = {
        k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
        for k, v in out.items()
    }
    m: Dict[str, float] = {}

    accs, ious = [], []
    for s in ("a", "b"):
        pred = out[f"covis_logit_{s}"] > 0
        gt = batch[f"covis_{s}"].bool()
        accs.append((pred == gt).float().mean().item())
        inter = (pred & gt).sum().float()
        union = (pred | gt).sum().float().clamp(min=1)
        ious.append((inter / union).item())
    m["covis_acc"] = sum(accs) / 2
    m["covis_iou"] = sum(ious) / 2

    errs = []
    for s in ("a", "b"):
        mask = batch[f"covis_{s}"].bool() & batch[f"bon_valid_{s}"].bool()
        if mask.any():
            e = circular_l1(out[f"corr_vec_{s}"], batch[f"corr_{s}"])
            errs.append(math.degrees((e * mask).sum().item() / mask.sum().item()))
    m["corr_err_deg"] = sum(errs) / len(errs) if errs else float("nan")

    berrs = []
    for s in ("a", "b"):
        mask = batch[f"bon_valid_{s}"].bool()
        e = (out[f"bon_{s}"] - batch[f"bon_{s}"]).abs()
        berrs.append(math.degrees((e * mask).sum().item() / max(mask.sum().item(), 1)))
    m["bon_err_deg"] = sum(berrs) / 2

    # Pose over the pairs it is defined for; NaN (skipped by validate()) when a
    # batch happens to hold only negatives.
    pv = _pose_valid(batch, out["t"])
    nan = float("nan")
    t_err = torch.linalg.norm(out["t"] - batch["t"], dim=-1)[pv]
    yaw_err = circular_l1(out["yaw_vec"], batch["yaw"])[pv]
    has = t_err.numel() > 0
    m["t_err_mean"] = t_err.mean().item() if has else nan
    m["t_err_median"] = t_err.median().item() if has else nan
    m["yaw_err_deg_mean"] = math.degrees(yaw_err.mean().item()) if has else nan
    m["yaw_err_deg_median"] = math.degrees(yaw_err.median().item()) if has else nan
    # pose accuracy at the headline threshold -- the metric checkpoints are
    # compared on in docs/PHASE4_NOTES.md
    m["pose_acc_0.5_10"] = (((t_err < 0.5) & (yaw_err < math.radians(10.0))).float().mean().item()
                            if has else nan)

    if "geo_t" in out:
        g_t = torch.linalg.norm(out["geo_t"] - batch["t"], dim=-1)[pv]
        g_yaw = circular_l1(out["geo_yaw_vec"], batch["yaw"])[pv]
        m["geo_t_err_median"] = g_t.median().item() if has else nan
        m["geo_yaw_err_deg_median"] = math.degrees(g_yaw.median().item()) if has else nan
        m["geo_acc_0.5_10"] = (((g_t < 0.5) & (g_yaw < math.radians(10.0))).float().mean().item()
                               if has else nan)

    # Does the co-visibility head say "nothing in common" when that is true?
    # Mean predicted co-visible fraction, per pair, on each kind of pair (A16).
    if "is_neg" in batch:
        frac = 0.5 * ((out["covis_logit_a"] > 0).float().mean(-1)
                      + (out["covis_logit_b"] > 0).float().mean(-1))
        neg = batch["is_neg"].bool()
        m["covis_frac_neg"] = frac[neg].mean().item() if neg.any() else nan
        m["covis_frac_pos"] = frac[~neg].mean().item() if (~neg).any() else nan
    return m
