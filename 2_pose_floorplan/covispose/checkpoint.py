"""Checkpointing that can genuinely resume a 9-hour epoch.

Saves everything that makes a resumed run continue the *same* trajectory:
model, optimizer, LR scheduler, AMP scaler, epoch/step counters, the health
monitor's state, and the RNG states for torch / numpy / python.

Two kinds of file:
* ``last.pt``            -- rewritten every ``save_every_steps``; this is the
                            interruption-resilience one.
* ``epoch_XXX.pt``       -- one per finished epoch, kept for the last N epochs.
"""

from __future__ import annotations

import glob
import os
import random
import shutil
from typing import Any, Dict, Optional

import numpy as np
import torch


def torch_load(path: str, map_location="cpu") -> Dict[str, Any]:
    """``torch.load`` for our own checkpoints.

    torch >= 2.6 defaults to ``weights_only=True``, which refuses the numpy RNG
    state and config dict stored here. These files are written by this repo, so
    full unpickling is the intended behaviour; older torch has no such argument.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _atomic_save(obj: Dict[str, Any], path: str) -> None:
    """Write to a temp file then replace, so an interrupted save cannot leave a
    truncated checkpoint where a valid one used to be."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _fix_adam_step_device(optimizer) -> None:
    """Move Adam/AdamW's ``step`` counters back to CPU after a resume.

    ``Optimizer.load_state_dict`` relocates every floating-point state tensor to
    the parameter's device. In torch 1.12 ``step`` is itself a float tensor, so
    it ends up on CUDA and the next ``.step()`` trips
    ``assert not step_t.is_cuda, "If capturable=False, state_steps should not be
    CUDA tensors."``  Resuming any AdamW run on GPU hits this.
    """
    for state in optimizer.state.values():
        s = state.get("step")
        if torch.is_tensor(s) and s.is_cuda:
            state["step"] = s.cpu()


def save_checkpoint(
    path: str,
    *,
    model,
    optimizer,
    scheduler=None,
    scaler=None,
    epoch: int = 0,
    global_step: int = 0,
    health=None,
    config: Optional[dict] = None,
    extra: Optional[dict] = None,
    ema=None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    blob = {
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "health": health.state_dict() if health is not None else None,
        "config": config,
        "rng": {
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "extra": extra or {},
    }
    _atomic_save(blob, path)


def load_checkpoint(
    path: str,
    *,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    health=None,
    map_location="cpu",
    restore_rng: bool = True,
    ema=None,
) -> Dict[str, Any]:
    blob = torch_load(path, map_location=map_location)
    model.load_state_dict(blob["model"])
    if ema is not None:
        if blob.get("ema") is not None:
            ema.load_state_dict(blob["ema"])
        else:  # resuming a run that had no EMA yet: start it from the weights
            ema.module.load_state_dict(blob["model"])
    if optimizer is not None and blob.get("optimizer") is not None:
        optimizer.load_state_dict(blob["optimizer"])
        _fix_adam_step_device(optimizer)
    if scheduler is not None and blob.get("scheduler") is not None:
        scheduler.load_state_dict(blob["scheduler"])
    if scaler is not None and blob.get("scaler") is not None:
        scaler.load_state_dict(blob["scaler"])
    if health is not None and blob.get("health") is not None:
        health.load_state_dict(blob["health"])
    if restore_rng and blob.get("rng"):
        rng = blob["rng"]
        try:
            torch.set_rng_state(rng["torch"].cpu() if torch.is_tensor(rng["torch"]) else rng["torch"])
            if rng.get("torch_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() if torch.is_tensor(s) else s
                                              for s in rng["torch_cuda"]])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        except Exception as e:  # noqa: BLE001
            print(f"[warn] could not restore RNG state ({e}); continuing with fresh RNG")
    return {
        "epoch": blob.get("epoch", 0),
        "global_step": blob.get("global_step", 0),
        "config": blob.get("config"),
        "extra": blob.get("extra", {}),
    }


def find_resume(ckpt_dir: str, mode: str = "auto") -> Optional[str]:
    """``auto`` -> last.pt if present, else the newest epoch_*.pt, else None."""
    if mode in (None, "", "none", False):
        return None
    if mode not in ("auto", True, "last"):
        return mode if os.path.isfile(mode) else None
    last = os.path.join(ckpt_dir, "last.pt")
    if os.path.isfile(last):
        return last
    eps = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
    return eps[-1] if eps else None


def prune_epoch_checkpoints(ckpt_dir: str, keep_last: int) -> None:
    if keep_last <= 0:
        return
    eps = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
    for p in eps[:-keep_last]:
        try:
            os.remove(p)
        except OSError:
            pass
