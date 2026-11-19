"""Exponential moving average of model weights.

For a dense regression target like relative pose, evaluating an EMA of the
weights is a nearly free improvement over the raw end-of-schedule weights: it
averages out the last steps' SGD noise without changing training at all.
Floating-point buffers (BatchNorm running statistics) are averaged like the
weights and integer buffers are copied -- timm's ``ModelEmaV2`` convention.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998, warmup_steps: int = 2000):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.updates = 0

    def current_decay(self) -> float:
        # Ramp the decay up so the EMA is not dominated by the random init for
        # the first ~1/(1-decay) steps (timm's exponential warmup; it stays
        # strictly below 1 for any warmup length).
        return self.decay * (1.0 - math.exp(-self.updates / max(self.warmup_steps, 1)))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.current_decay()
        ema_sd = self.module.state_dict()
        for k, v in model.state_dict().items():
            e = ema_sd[k]
            if e.dtype.is_floating_point:
                e.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                e.copy_(v)

    def state_dict(self) -> dict:
        return {"module": self.module.state_dict(), "updates": self.updates,
                "decay": self.decay, "warmup_steps": self.warmup_steps}

    def load_state_dict(self, d: dict) -> None:
        self.module.load_state_dict(d["module"])
        self.updates = int(d.get("updates", 0))
