"""Differentiable geometric pose from the per-column heads.

The RANSAC path (``ransac.py``) turns the three per-column predictions into
2D-2D floor-plane correspondences and fits a rigid transform *after* training,
so nothing ever tells the heads that their outputs have to be geometrically
consistent with each other. This module does the same fit as a closed-form,
weighted 2D Procrustes **inside** the graph, so a pose loss on its output
back-propagates into the boundary, correspondence and co-visibility heads.

Correspondences (same construction as ``ransac.build_correspondences``)
---------------------------------------------------------------------
From A's columns ``j``:   P = r_a[j] dir(theta_j)            (A's frame)
                          Q = r_b(phi_j) dir(phi_j)           (B's frame)
From B's columns ``j``:   Q = r_b[j] dir(theta_j)            (B's frame)
                          P = r_a(psi_j) dir(psi_j)           (A's frame)
with ``phi = corr_a``, ``psi = corr_b``, and ``Q ~= Rot(yaw) P + t``.

Weights: predicted co-visibility times ``1 / (1 + (r_P + r_Q) / 2)^2``. A fixed
angular error in the boundary or correspondence displaces a point by an amount
proportional to its range, so far points are down-weighted like an inverse
variance. Points beyond ``max_dist`` get zero weight.

Everything runs in float32 with autocast disabled: ``1/tan`` and ``atan2`` are
exactly the operations fp16 / bf16 get wrong.
"""

from __future__ import annotations

import math
from typing import Dict

import torch


def col_theta(n_cols: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Torch twin of ``geometry.col_centers_theta``."""
    j = torch.arange(n_cols, device=device, dtype=dtype)
    return ((j + 0.5) / n_cols - 0.5) * 2 * math.pi


def dir_of(theta: torch.Tensor) -> torch.Tensor:
    """Torch twin of ``geometry.dir_of_theta``: (...,) -> (..., 2)."""
    return torch.stack([-torch.sin(theta), torch.cos(theta)], dim=-1)


def bon_to_dist(bon: torch.Tensor, max_dist: float) -> torch.Tensor:
    """HorizonNet floor boundary -> radial distance, clamped to ``max_dist``."""
    lo = math.atan2(1.0, max_dist)
    return 1.0 / torch.tan(bon.clamp(lo, math.pi / 2 - 1e-3))


def sample_circular(values: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of a per-column signal at arbitrary azimuths.

    ``values`` (B, N) on the column grid, ``phi`` (B, M) radians. Differentiable
    in both ``values`` and ``phi``; wraps around the 360 deg seam.
    """
    n = values.shape[-1]
    u = (phi / (2 * math.pi) + 0.5) * n - 0.5
    u0 = torch.floor(u)
    w = u - u0
    i0 = u0.long().remainder(n)
    i1 = (i0 + 1).remainder(n)
    return values.gather(-1, i0) * (1 - w) + values.gather(-1, i1) * w


def weighted_procrustes_2d(P: torch.Tensor, Q: torch.Tensor, w: torch.Tensor,
                           eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    """Batched weighted rigid fit ``Q ~= Rot(yaw) P + t``.

    P, Q: (B, M, 2); w: (B, M) >= 0. Returns ``t`` (B, 2), ``yaw_vec`` (B, 2) as
    an unnormalised (cos, sin) pair -- the same form the decoder emits -- and
    ``weight_sum`` (B,) so callers can mask degenerate pairs.
    """
    ws = w.sum(-1)
    wn = w / ws.clamp(min=eps).unsqueeze(-1)
    pc = (wn.unsqueeze(-1) * P).sum(1)
    qc = (wn.unsqueeze(-1) * Q).sum(1)
    p0 = P - pc.unsqueeze(1)
    q0 = Q - qc.unsqueeze(1)
    cross = (wn * (p0[..., 0] * q0[..., 1] - p0[..., 1] * q0[..., 0])).sum(-1)
    dot = (wn * (p0 * q0).sum(-1)).sum(-1)
    norm = torch.sqrt(dot * dot + cross * cross + eps)
    c, s = dot / norm, cross / norm
    t = qc - torch.stack([c * pc[:, 0] - s * pc[:, 1], s * pc[:, 0] + c * pc[:, 1]], dim=-1)
    return {"t": t, "yaw_vec": torch.stack([dot, cross], dim=-1), "weight_sum": ws}


def _residuals(P: torch.Tensor, Q: torch.Tensor, fit: Dict[str, torch.Tensor]) -> torch.Tensor:
    yv = fit["yaw_vec"]
    nrm = yv.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    c, s = (yv[:, :1] / nrm), (yv[:, 1:] / nrm)
    RP = torch.stack([c * P[..., 0] - s * P[..., 1], s * P[..., 0] + c * P[..., 1]], dim=-1)
    return (RP + fit["t"].unsqueeze(1) - Q).norm(dim=-1)


def geometric_pose(out: Dict[str, torch.Tensor], max_dist: float = 10.0,
                   irls_iters: int = 2, irls_c: float = 0.3) -> Dict[str, torch.Tensor]:
    """Pose from the per-column head outputs of :class:`covispose.model.CoVisPose`.

    Plain weighted least squares has no outlier rejection, and even ground-truth
    head values carry outliers (the co-visibility label slack near occlusion
    edges, assumption A11). ``irls_iters`` rounds of Cauchy re-weighting,
    ``w / (1 + (residual / irls_c)^2)``, are the differentiable stand-in for
    RANSAC. Measured on ground truth (house 0000, 102 pairs, 256 columns):

    ============  =========  =========  ===========
    IRLS rounds   t median   t p90      yaw p90
    ============  =========  =========  ===========
    0             0.0553     0.184      2.55 deg
    2, c=0.2      0.0375     0.113      0.92 deg
    ============  =========  =========  ===========

    ``irls_c`` defaults to 0.3 rather than the GT optimum because predicted
    boundaries are noisier -- the same reason RANSAC's threshold went from 0.10
    to 0.30 (A14b); the GT sweep is flat from 0.1 to 0.4. The robust weights are
    detached: gradients flow through the fit, not through the outlier decision.
    """
    with torch.autocast(device_type=out["bon_a"].device.type, enabled=False):
        bon_a, bon_b = out["bon_a"].float(), out["bon_b"].float()
        p_a = torch.sigmoid(out["covis_logit_a"].float())
        p_b = torch.sigmoid(out["covis_logit_b"].float())
        phi = torch.atan2(out["corr_vec_a"][..., 1].float(), out["corr_vec_a"][..., 0].float())
        psi = torch.atan2(out["corr_vec_b"][..., 1].float(), out["corr_vec_b"][..., 0].float())

        n = bon_a.shape[-1]
        theta = col_theta(n, device=bon_a.device).expand_as(bon_a)
        r_a, r_b = bon_to_dist(bon_a, max_dist), bon_to_dist(bon_b, max_dist)

        # A's columns
        r_b_phi = sample_circular(r_b, phi)
        P1, Q1 = dir_of(theta) * r_a.unsqueeze(-1), dir_of(phi) * r_b_phi.unsqueeze(-1)
        # B's columns
        r_a_psi = sample_circular(r_a, psi)
        P2, Q2 = dir_of(psi) * r_a_psi.unsqueeze(-1), dir_of(theta) * r_b.unsqueeze(-1)

        P = torch.cat([P1, P2], dim=1)
        Q = torch.cat([Q1, Q2], dim=1)
        rp = torch.cat([r_a, r_a_psi], dim=1)
        rq = torch.cat([r_b_phi, r_b], dim=1)
        in_range = ((rp < max_dist * 0.999) & (rq < max_dist * 0.999)).float()
        w = torch.cat([p_a, p_b], dim=1) * in_range / (1.0 + 0.5 * (rp + rq)) ** 2
        fit = weighted_procrustes_2d(P, Q, w)
        for _ in range(irls_iters):
            res = _residuals(P, Q, fit).detach()
            fit = weighted_procrustes_2d(P, Q, w / (1.0 + (res / irls_c) ** 2))
        # report the pre-IRLS weight mass: it is what says "no correspondences"
        fit["weight_sum"] = w.sum(-1)
        return fit
