"""Shared evaluation machinery for Phase 4.

Runs a checkpoint over a split, collecting:
  * per-column head metrics (co-visibility, correspondence, boundary)
  * pose from the conv decoder
  * pose from RANSAC over the predicted correspondences
and caches the raw per-column predictions so that RANSAC hyper-parameters can
be swept **without** re-running the network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .checkpoint import torch_load
from .model import build_model, config_from_dict
from .ransac import pose_from_predictions


@dataclass
class CachedPredictions:
    """Per-pair network outputs plus ground truth, all numpy, all on CPU."""

    covis_a: np.ndarray      # (N, C) probabilities
    covis_b: np.ndarray
    corr_a: np.ndarray       # (N, C) radians
    corr_b: np.ndarray
    bon_a: np.ndarray        # (N, C) radians
    bon_b: np.ndarray
    t_pred: np.ndarray       # (N, 2)
    yaw_pred: np.ndarray     # (N,)
    # ground truth
    gt_covis_a: np.ndarray
    gt_covis_b: np.ndarray
    gt_corr_a: np.ndarray
    gt_corr_b: np.ndarray
    gt_bon_a: np.ndarray
    gt_bon_b: np.ndarray
    gt_bon_valid_a: np.ndarray
    gt_bon_valid_b: np.ndarray
    gt_t: np.ndarray
    gt_yaw: np.ndarray
    covis_score: np.ndarray
    baseline: np.ndarray
    # geometric (Procrustes) pose, only for models built with geo_pose=True
    geo_t: Optional[np.ndarray] = None
    geo_yaw: Optional[np.ndarray] = None
    # (N,) bool, True for negatives (pose undefined); None = no negatives
    is_neg: Optional[np.ndarray] = None

    @property
    def pose_valid(self) -> np.ndarray:
        return np.ones(len(self), bool) if self.is_neg is None else ~self.is_neg.astype(bool)

    def __len__(self) -> int:
        return len(self.yaw_pred)


def load_model_from_ckpt(ckpt_path: str, device: str = "cuda", weights: str = "auto"):
    """Rebuild the model a checkpoint was trained with and load its weights.

    ``weights``: ``"raw"`` = the optimised weights, ``"ema"`` = the EMA copy,
    ``"auto"`` = EMA when the checkpoint has one.
    """
    blob = torch_load(ckpt_path, map_location="cpu")
    cfg = blob.get("config") or {}
    model_cfg = config_from_dict(cfg.get("model", {}), pretrained=False, dropout=0.0,
                                 drop_path=0.0, backbone_drop_path=0.0,
                                 grad_checkpoint=False)
    model = build_model(model_cfg).to(device).eval()
    use_ema = blob.get("ema") is not None and weights in ("auto", "ema")
    if weights == "ema" and blob.get("ema") is None:
        raise ValueError(f"{ckpt_path} has no EMA weights")
    model.load_state_dict(blob["ema"]["module"] if use_ema else blob["model"])
    blob["loaded_weights"] = "ema" if use_ema else "raw"
    return model, model_cfg, blob


# --------------------------------------------------------------------------- #
# test-time augmentation
# --------------------------------------------------------------------------- #
_COL_KEYS = ("covis_logit", "corr_vec", "bon")


def _rotate_vec(v: torch.Tensor, a: float) -> torch.Tensor:
    """Add ``a`` radians to the angle an unnormalised (cos, sin) pair encodes."""
    c, s = math.cos(a), math.sin(a)
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], dim=-1)


def _mirror_vec(v: torch.Tensor) -> torch.Tensor:
    return torch.stack([v[..., 0], -v[..., 1]], dim=-1)


def invert_view(out: Dict[str, torch.Tensor], n_cols: int, flip: bool,
                da: int, db: int) -> Dict[str, torch.Tensor]:
    """Map outputs predicted on an augmented view back to the original frames.

    Exact inverse of :func:`covispose.dataset.augment_targets` (mirror, then
    roll A by ``da`` and B by ``db`` columns): unroll, then unmirror.
    """
    phi_a = 2 * math.pi * da / n_cols
    phi_b = 2 * math.pi * db / n_cols
    o = dict(out)
    for s, dc, phi_other in (("a", da, phi_b), ("b", db, phi_a)):
        for k in _COL_KEYS:
            o[f"{k}_{s}"] = torch.roll(o[f"{k}_{s}"], -dc, dims=1)
        o[f"corr_vec_{s}"] = _rotate_vec(o[f"corr_vec_{s}"], -phi_other)
    pose_keys = [("t", "yaw_vec")] + ([("geo_t", "geo_yaw_vec")] if "geo_t" in o else [])
    for tk, yk in pose_keys:
        o[tk] = _rotate_vec(o[tk], -phi_b)   # t lives in B's frame: same rotation law
        o[yk] = _rotate_vec(o[yk], -(phi_b - phi_a))
    if flip:
        for s in ("a", "b"):
            for k in _COL_KEYS:
                o[f"{k}_{s}"] = torch.flip(o[f"{k}_{s}"], dims=[1])
            o[f"corr_vec_{s}"] = _mirror_vec(o[f"corr_vec_{s}"])
        for tk, yk in pose_keys:
            o[tk] = torch.stack([-o[tk][..., 0], o[tk][..., 1]], dim=-1)
            o[yk] = _mirror_vec(o[yk])
    return o


def tta_views(n_cols: int, rolls: int, flip: bool) -> List[Tuple[bool, int, int]]:
    """(flip, da, db) per view; A and B get different offsets for diversity."""
    views = []
    for f in ([False, True] if flip else [False]):
        for k in range(max(rolls, 1)):
            views.append((f, (k * n_cols) // rolls, (((rolls - k) % rolls) * n_cols) // rolls))
    return views


def forward_tta(model, img_a: torch.Tensor, img_b: torch.Tensor, rolls: int = 1,
                flip: bool = False) -> Dict[str, torch.Tensor]:
    """Average predictions over mirrored / rolled copies of the pair.

    Rolls are whole output columns, so every view is an exact re-parameterisation
    of the same pair. Angles are averaged as unit vectors, never as raw radians.
    """
    n_cols = model.cfg.n_cols
    px = img_a.shape[-1] // n_cols
    acc: Dict[str, torch.Tensor] = {}
    views = tta_views(n_cols, rolls, flip)
    for f, da, db in views:
        a, b = (torch.flip(img_a, dims=[3]), torch.flip(img_b, dims=[3])) if f else (img_a, img_b)
        a, b = torch.roll(a, da * px, dims=3), torch.roll(b, db * px, dims=3)
        o = invert_view({k: v.float() for k, v in model(a, b).items()}, n_cols, f, da, db)
        for k, v in o.items():
            if k.endswith("vec"):
                v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            acc[k] = acc[k] + v if k in acc else v
    return {k: v / len(views) for k, v in acc.items()}


@torch.no_grad()
def collect_predictions(model, loader, device: str, amp: bool = False,
                        progress_every: int = 50, tta_rolls: int = 1,
                        tta_flip: bool = False) -> CachedPredictions:
    """Run the network over a loader and cache everything needed downstream.

    ``amp`` defaults to **False** for two reasons. First, torch 1.12's fused
    attention fast path (``torch._native_multi_head_attention``, taken only in
    eval mode) raises "expected scalar type Half but found Float" under
    autocast. Second, and more importantly, a final evaluation should not carry
    fp16's non-determinism (assumption A13) -- forward-only fp32 at batch 8 is
    well inside the memory budget, so there is nothing to gain from AMP here.
    """
    buf: Dict[str, List[np.ndarray]] = {k: [] for k in (
        "covis_a", "covis_b", "corr_a", "corr_b", "bon_a", "bon_b", "t_pred", "yaw_pred",
        "gt_covis_a", "gt_covis_b", "gt_corr_a", "gt_corr_b", "gt_bon_a", "gt_bon_b",
        "gt_bon_valid_a", "gt_bon_valid_b", "gt_t", "gt_yaw", "covis_score", "baseline")}
    if model.cfg.geo_pose:
        buf["geo_t"], buf["geo_yaw"] = [], []
    buf["is_neg"] = []

    n_batches = len(loader)
    for i, b in enumerate(loader):
        imgs_a = b["img_a"].to(device, non_blocking=True)
        imgs_b = b["img_b"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                            enabled=amp):
            out = forward_tta(model, imgs_a, imgs_b, tta_rolls, tta_flip)
        if model.cfg.geo_pose:
            buf["geo_t"].append(out["geo_t"].float().cpu().numpy())
            buf["geo_yaw"].append(model.decode_angle(out["geo_yaw_vec"].float()).cpu().numpy())

        buf["covis_a"].append(torch.sigmoid(out["covis_logit_a"].float()).cpu().numpy())
        buf["covis_b"].append(torch.sigmoid(out["covis_logit_b"].float()).cpu().numpy())
        buf["corr_a"].append(model.decode_angle(out["corr_vec_a"].float()).cpu().numpy())
        buf["corr_b"].append(model.decode_angle(out["corr_vec_b"].float()).cpu().numpy())
        buf["bon_a"].append(out["bon_a"].float().cpu().numpy())
        buf["bon_b"].append(out["bon_b"].float().cpu().numpy())
        buf["t_pred"].append(out["t"].float().cpu().numpy())
        buf["yaw_pred"].append(model.decode_angle(out["yaw_vec"].float()).cpu().numpy())

        for k_out, k_in in (("gt_covis_a", "covis_a"), ("gt_covis_b", "covis_b"),
                            ("gt_corr_a", "corr_a"), ("gt_corr_b", "corr_b"),
                            ("gt_bon_a", "bon_a"), ("gt_bon_b", "bon_b"),
                            ("gt_bon_valid_a", "bon_valid_a"),
                            ("gt_bon_valid_b", "bon_valid_b"),
                            ("gt_t", "t"), ("gt_yaw", "yaw"),
                            ("covis_score", "covis_score"), ("baseline", "baseline")):
            buf[k_out].append(b[k_in].numpy())
        buf["is_neg"].append(b["is_neg"].numpy() if "is_neg" in b
                             else np.zeros(len(b["t"]), bool))

        if progress_every and (i + 1) % progress_every == 0:
            print(f"    {i+1}/{n_batches} batches", flush=True)

    return CachedPredictions(**{k: np.concatenate(v, axis=0) for k, v in buf.items()})


def wrap_pi_np(a):
    return np.arctan2(np.sin(a), np.cos(a))


def head_metrics(P: CachedPredictions) -> Dict[str, float]:
    """Per-column head quality, over the whole split."""
    out: Dict[str, float] = {}

    accs, ious = [], []
    for s in ("a", "b"):
        pred = getattr(P, f"covis_{s}") > 0.5
        gt = getattr(P, f"gt_covis_{s}") > 0.5
        accs.append((pred == gt).mean())
        inter = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        ious.append(inter / max(union, 1))
    out["covis_acc"] = float(np.mean(accs))
    out["covis_iou"] = float(np.mean(ious))

    errs = []
    for s in ("a", "b"):
        m = (getattr(P, f"gt_covis_{s}") > 0.5) & getattr(P, f"gt_bon_valid_{s}").astype(bool)
        e = np.abs(wrap_pi_np(getattr(P, f"corr_{s}") - getattr(P, f"gt_corr_{s}")))
        if m.any():
            errs.append(np.degrees(e[m].mean()))
    out["corr_err_deg"] = float(np.mean(errs))

    berrs = []
    for s in ("a", "b"):
        m = getattr(P, f"gt_bon_valid_{s}").astype(bool)
        e = np.abs(getattr(P, f"bon_{s}") - getattr(P, f"gt_bon_{s}"))
        berrs.append(np.degrees(e[m].mean()))
    out["bon_err_deg"] = float(np.mean(berrs))
    return out


def pose_errors(t_pred: np.ndarray, yaw_pred: np.ndarray,
                gt_t: np.ndarray, gt_yaw: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "t_err": np.linalg.norm(t_pred - gt_t, axis=-1),
        "yaw_err_deg": np.abs(np.degrees(wrap_pi_np(yaw_pred - gt_yaw))),
    }


def summarize_pose(t_err: np.ndarray, yaw_err: np.ndarray,
                   valid: Optional[np.ndarray] = None) -> Dict[str, float]:
    if valid is None:
        valid = np.ones(len(t_err), bool)
    te, ye = t_err[valid], yaw_err[valid]
    if len(te) == 0:
        return {k: float("nan") for k in
                ("t_mean", "t_median", "yaw_mean", "yaw_median",
                 "acc_0.1m_1deg", "acc_0.5_10deg", "acc_1.0_20deg", "solved_frac")}
    return {
        "t_mean": float(te.mean()), "t_median": float(np.median(te)),
        "yaw_mean": float(ye.mean()), "yaw_median": float(np.median(ye)),
        # camera height ~1.5 m, so 0.1 camera heights ~ 15 cm
        "acc_0.1m_1deg": float(((te < 0.1) & (ye < 1.0)).mean()),
        "acc_0.5_10deg": float(((te < 0.5) & (ye < 10.0)).mean()),
        "acc_1.0_20deg": float(((te < 1.0) & (ye < 20.0)).mean()),
        "solved_frac": float(valid.mean()),
    }


def run_ransac_over_cache(P: CachedPredictions, threshold: float,
                          covis_thresh: float = 0.5, iters: int = 500,
                          min_points: int = 6, seed: int = 0,
                          progress_every: int = 0) -> Dict[str, np.ndarray]:
    """RANSAC pose for every cached pair. No GPU, no network re-run."""
    n = len(P)
    t_out = np.zeros((n, 2))
    yaw_out = np.zeros(n)
    ok = np.zeros(n, bool)
    n_inl = np.zeros(n, int)
    n_prop = np.zeros(n, int)

    for i in range(n):
        r = pose_from_predictions(
            P.bon_a[i], P.bon_b[i], P.corr_a[i], P.covis_a[i],
            covis_thresh=covis_thresh, threshold=threshold,
            iters=iters, min_points=min_points, seed=seed,
        )
        t_out[i], yaw_out[i], ok[i] = r.t, r.yaw, r.ok
        n_inl[i] = int(r.inliers.sum())
        n_prop[i] = r.n_proposed
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    ransac {i+1}/{n}", flush=True)

    return {"t": t_out, "yaw": yaw_out, "ok": ok,
            "n_inliers": n_inl, "n_proposed": n_prop}
