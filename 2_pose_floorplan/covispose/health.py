"""Training health monitoring: stop early rather than burn hours on a bad run.

Three separate questions, which need three separate mechanisms:

1. **Is it broken?**  NaN/Inf in the loss -> abort immediately.
2. **Is it diverging?**  A term's smoothed value climbing well above its own
   best and staying there -> abort.
3. **Is a head actually learning?**  This is the one a falling *total* loss
   hides, so we check each head against a **trivial baseline** computed from
   the validation data itself (predict the majority class / the mean / zero).
   A head no better than its trivial baseline is not learning, however nice
   the loss curve looks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

TERMS = ("covis", "corr", "bon", "pose_t", "pose_yaw")


@dataclass
class HealthReport:
    should_stop: bool = False
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_stop(self, msg: str) -> None:
        self.should_stop = True
        self.reasons.append(msg)

    def add_warn(self, msg: str) -> None:
        self.warnings.append(msg)


class HealthMonitor:
    """Per-term EMA tracking with NaN and divergence detection."""

    def __init__(
        self,
        ema_beta: float = 0.99,
        divergence_factor: float = 3.0,
        divergence_patience: int = 500,
        min_steps_before_divergence: int = 1000,
    ):
        self.beta = ema_beta
        self.divergence_factor = divergence_factor
        self.divergence_patience = divergence_patience
        self.min_steps = min_steps_before_divergence
        self.ema: Dict[str, float] = {}
        self.best: Dict[str, float] = {}
        self.bad_streak: Dict[str, int] = {}
        self.steps = 0

    def state_dict(self) -> dict:
        return {
            "ema": self.ema, "best": self.best,
            "bad_streak": self.bad_streak, "steps": self.steps,
        }

    def load_state_dict(self, d: dict) -> None:
        self.ema = dict(d.get("ema", {}))
        self.best = dict(d.get("best", {}))
        self.bad_streak = dict(d.get("bad_streak", {}))
        self.steps = int(d.get("steps", 0))

    def update(self, terms: Dict[str, float]) -> HealthReport:
        rep = HealthReport()
        self.steps += 1

        for k, v in terms.items():
            if not math.isfinite(v):
                rep.add_stop(f"loss term '{k}' is {v} at step {self.steps}")
                return rep

        for k in TERMS:
            if k not in terms:
                continue
            v = terms[k]
            self.ema[k] = v if k not in self.ema else self.beta * self.ema[k] + (1 - self.beta) * v
            e = self.ema[k]
            if k not in self.best or e < self.best[k]:
                self.best[k] = e
                self.bad_streak[k] = 0
            elif self.steps > self.min_steps and e > self.divergence_factor * max(self.best[k], 1e-6):
                self.bad_streak[k] = self.bad_streak.get(k, 0) + 1
                if self.bad_streak[k] >= self.divergence_patience:
                    rep.add_stop(
                        f"loss term '{k}' diverged: EMA {e:.4f} has been over "
                        f"{self.divergence_factor}x its best ({self.best[k]:.4f}) for "
                        f"{self.bad_streak[k]} steps"
                    )
            else:
                self.bad_streak[k] = 0
        return rep


@torch.no_grad()
def trivial_baselines(loader, device, max_batches: int = 64) -> Dict[str, float]:
    """What each head would score by predicting the dumbest possible thing.

    * covis     : always predict the majority class.
    * corr      : a uniformly random angle averages pi/2 = 90 deg of error.
    * bon       : predict the dataset's mean boundary everywhere.
    * pose_t    : predict zero translation.
    * pose_yaw  : predict zero rotation.

    A head that does not beat these is not learning, no matter what the loss
    curve does.
    """
    n_pos = n_tot = 0
    bon_vals, bon_masks = [], []
    t_norms, yaw_abs = [], []

    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        for s in ("a", "b"):
            c = b[f"covis_{s}"]
            n_pos += float(c.sum())
            n_tot += c.numel()
            bon_vals.append(b[f"bon_{s}"].flatten())
            bon_masks.append(b[f"bon_valid_{s}"].flatten())
        # same pairs the pose metrics are computed on (no negatives)
        pv = b["pose_valid"].bool() if "pose_valid" in b else slice(None)
        t_norms.append(torch.linalg.norm(b["t"], dim=-1)[pv])
        yaw_abs.append(b["yaw"].abs()[pv])

    bon = torch.cat(bon_vals)
    msk = torch.cat(bon_masks).bool()
    bon_mean = bon[msk].mean()

    pos_rate = n_pos / max(n_tot, 1)
    return {
        "covis_acc": max(pos_rate, 1 - pos_rate),
        "covis_pos_rate": pos_rate,
        "corr_err_deg": 90.0,
        "bon_err_deg": math.degrees((bon[msk] - bon_mean).abs().mean().item()),
        "t_err_mean": torch.cat(t_norms).mean().item(),
        "yaw_err_deg_mean": math.degrees(torch.cat(yaw_abs).mean().item()),
    }


def check_against_baselines(
    metrics: Dict[str, float], baseline: Dict[str, float], margin: float = 0.95
) -> HealthReport:
    """Flag any head that is not meaningfully beating its trivial baseline."""
    rep = HealthReport()

    if metrics["covis_acc"] <= baseline["covis_acc"]:
        rep.add_warn(
            f"co-visibility head is NOT beating the majority-class baseline: "
            f"acc {metrics['covis_acc']:.4f} vs {baseline['covis_acc']:.4f} "
            f"(positive rate {baseline['covis_pos_rate']:.3f})"
        )
    for key, label in (
        ("corr_err_deg", "correspondence"),
        ("bon_err_deg", "boundary"),
        ("t_err_mean", "translation"),
        ("yaw_err_deg_mean", "yaw"),
    ):
        m, b = metrics.get(key, float("nan")), baseline[key]
        if not math.isfinite(m) or m >= margin * b:
            rep.add_warn(
                f"{label} head is NOT beating its trivial baseline: "
                f"{key}={m:.4f} vs baseline {b:.4f}"
            )
    return rep


def vram_guard(budget_gib: float, label: str = "") -> Optional[str]:
    """Return a message if reserved VRAM has passed the budget.

    Windows/WDDM silently pages to host RAM instead of raising OOM, which turns
    a 1.2 s step into an 8.4 s step with no error anywhere. Catching this by
    measured memory is the only reliable way -- see docs/PHASE2_NOTES.md.
    """
    if not torch.cuda.is_available():
        return None
    resv = torch.cuda.max_memory_reserved() / 2**30
    if resv > budget_gib:
        return (
            f"VRAM budget exceeded{' (' + label + ')' if label else ''}: "
            f"{resv:.2f} GiB reserved > {budget_gib:.2f} GiB budget. On Windows this "
            f"does NOT raise OOM -- the driver pages to host RAM and the step time "
            f"explodes silently. Reduce batch size or resolution."
        )
    return None
