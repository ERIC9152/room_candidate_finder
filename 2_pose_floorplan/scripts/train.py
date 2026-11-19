"""CoVisPose training loop.

Single GPU:
    python scripts/train.py --config configs/server_a5000.yaml

Several GPUs on one machine (DDP):
    torchrun --standalone --nproc_per_node=3 scripts/train.py --config configs/server_a5000.yaml

Design points that matter for this review cycle:

* **Per-head logging.** Every loss term and every interpretable metric is
  written to a JSONL log per step, so the review can plot each head separately
  rather than staring at a single total.
* **Interruptible.** ``last.pt`` is rewritten every ``save_every_steps``, and a
  snapshot is kept per finished epoch. Resume restores optimizer, scheduler,
  AMP scaler, EMA, counters, health state and RNG.
* **Stops itself when something is wrong.** NaN or a diverging term aborts;
  heads that fail to beat a trivial baseline raise a loud warning at every
  validation point. Under DDP the loss terms are averaged across ranks before
  the health check, so every rank reaches the same stop decision and none is
  left hanging in a collective.
* **VRAM guard.** Reserved memory is checked against a budget after warmup,
  because Windows pages silently instead of raising OOM.
* **Checkpoint selection on a fixed score.** ``best.pt`` is chosen on the
  validation loss under the config's *full* loss weights, evaluated with the
  EMA weights when EMA is on -- not on whatever ramped weights happened to be
  active at the time.
"""

import argparse
import contextlib
import copy
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from covispose.checkpoint import (
    find_resume, load_checkpoint, prune_epoch_checkpoints, save_checkpoint, torch_load,
)
from covispose.dataset import make_loader
from covispose.ema import ModelEMA
from covispose.health import (
    HealthMonitor, check_against_baselines, trivial_baselines, vram_guard,
)
from covispose.losses import LossWeights, covispose_loss, covispose_metrics
from covispose.model import build_model, config_from_dict, count_params

BASE_TERMS = ("covis", "corr", "bon", "pose_t", "pose_yaw")


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def _deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path):
    """YAML config with optional ``_base_: other.yaml`` inheritance (relative path)."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base:
        cfg = _deep_merge(load_config(os.path.join(os.path.dirname(path), base)), cfg)
    return cfg


# --------------------------------------------------------------------------- #
# distributed helpers
# --------------------------------------------------------------------------- #
def setup_distributed():
    """-> (rank, world_size, local_rank). Plain single-process run if not under torchrun."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        backend = "nccl"
    else:
        backend = "gloo"  # CPU smoke tests of the DDP path
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=60))
    return rank, world, local


def all_reduce_(t, op=None):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=op or dist.ReduceOp.SUM)
    return t


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


class JsonlLogger:
    def __init__(self, path, enabled=True):
        self.f = None
        if enabled:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self.f = open(path, "a", encoding="utf-8")

    def log(self, rec):
        if self.f:
            self.f.write(json.dumps(rec) + "\n")
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


# --------------------------------------------------------------------------- #
# schedule
# --------------------------------------------------------------------------- #
def build_scheduler(opt, cfg, steps_per_epoch):
    """Linear warmup then cosine, over the FULL planned horizon.

    The horizon is ``train.epochs`` even when this run only executes one epoch
    (``--max-epochs 1``) -- otherwise the LR would anneal to zero at the end of
    epoch 1 and a later continuation would restart from a cold schedule.
    ``min_lr_ratio`` keeps a floor under the cosine, so the last epoch still
    learns something instead of running at ~1e-11.
    """
    total = max(int(cfg["train"]["epochs"]) * steps_per_epoch, 1)
    warmup = int(cfg["train"].get("warmup_steps", 500))
    floor = float(cfg["train"].get("min_lr_ratio", 0.0))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        p = (step - warmup) / max(total - warmup, 1)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda), total


def ramped_weights(full: LossWeights, step: int, start: int, ramp: int) -> LossWeights:
    """Geometric pose terms ramp linearly in from ``start`` over ``ramp`` steps."""
    if step < start:
        f = 0.0
    else:
        f = min((step - start) / max(ramp, 1), 1.0)
    w = copy.copy(full)
    w.geo_t, w.geo_yaw = full.geo_t * f, full.geo_yaw * f
    return w


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(model, loader, weights, device, amp_ctx, max_batches=0):
    """Loss terms and metrics summed over this rank's shard, then all-reduced."""
    was_training = model.training
    model.eval()
    sums, msums, mcount = {}, {}, {}
    n = 0
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        with amp_ctx():
            out = model(b["img_a"], b["img_b"])
        terms = covispose_loss(out, b, weights)
        mets = covispose_metrics(out, b)
        bs = b["img_a"].shape[0]
        for k, v in terms.items():
            sums[k] = sums.get(k, 0.0) + float(v) * bs
        for k, v in mets.items():
            # register every key even when NaN, so all ranks reduce the same vector
            msums.setdefault(k, 0.0)
            mcount.setdefault(k, 0)
            if math.isfinite(v):
                msums[k] += v * bs
                mcount[k] += bs
        n += bs
    model.train(was_training)

    # Every rank has the same keys (same model, same loss config), so a fixed
    # sorted order lines the tensors up across ranks.
    lk, mk = sorted(sums), sorted(msums)
    vec = torch.tensor([sums[k] for k in lk] + [msums[k] for k in mk]
                       + [mcount[k] for k in mk] + [n], dtype=torch.float64, device=device)
    all_reduce_(vec)
    vec = vec.tolist()
    n_all = vec[-1]
    loss = {k: vec[i] / max(n_all, 1) for i, k in enumerate(lk)}
    met = {k: vec[len(lk) + i] / max(vec[len(lk) + len(mk) + i], 1) for i, k in enumerate(mk)}
    return loss, met, int(n_all)


def fmt_metrics(m):
    s = (f"covis_acc={m.get('covis_acc', float('nan')):.4f} "
         f"iou={m.get('covis_iou', float('nan')):.4f} "
         f"corr={m.get('corr_err_deg', float('nan')):6.2f}d "
         f"bon={m.get('bon_err_deg', float('nan')):5.2f}d "
         f"t={m.get('t_err_mean', float('nan')):.4f} "
         f"yaw={m.get('yaw_err_deg_mean', float('nan')):6.2f}d")
    if "pose_acc_0.5_10" in m:
        s += f" acc={m['pose_acc_0.5_10']:.4f}"
    if "geo_acc_0.5_10" in m:
        s += f" geo_acc={m['geo_acc_0.5_10']:.4f}"
    if "covis_frac_neg" in m:
        s += f" covis_frac neg/pos={m['covis_frac_neg']:.3f}/{m['covis_frac_pos']:.3f}"
    return s


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/configs/server_c10neg.yaml")
    ap.add_argument("--max-epochs", type=int, default=0,
                    help="run at most this many epochs THIS invocation (0 = all "
                         "remaining). The LR schedule horizon stays train.epochs.")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--val-subsample", type=int, default=None,
                    help="pairs used for the end-of-epoch validation (0 = full split; "
                         "default: train.val_subsample, else 2000)")
    ap.add_argument("--quickval-subsample", type=int, default=512,
                    help="pairs used for the periodic in-epoch validation")
    ap.add_argument("--quickval-every", type=int, default=2000, help="optimizer steps")
    ap.add_argument("--log-every", type=int, default=25, help="optimizer steps")
    ap.add_argument("--save-every-steps", type=int, default=500)
    ap.add_argument("--limit-steps", type=int, default=0, help="debug: stop early")
    ap.add_argument("--data-root", default=None, help="override data.root")
    args = ap.parse_args()

    rank, world, local_rank = setup_distributed()
    main_proc = rank == 0
    say = print if main_proc else (lambda *a, **k: None)

    cfg = load_config(args.config)
    mcfg, tcfg, dcfg = cfg["model"], cfg["train"], cfg["data"]
    if args.data_root or os.environ.get("ZIND_ROOT"):
        dcfg["root"] = args.data_root or os.environ["ZIND_ROOT"]
    out_dir = args.out_dir or cfg["ckpt"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    seed = int(tcfg.get("seed", 0))
    # Different augmentation / dropout streams per rank; DDP broadcasts rank 0's
    # initial weights, so model init does not need identical seeds.
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    torch.backends.cudnn.benchmark = True

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    use_cuda = device.startswith("cuda")
    if use_cuda and tcfg.get("tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    amp = bool(tcfg.get("amp", True)) and use_cuda
    amp_dtype = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                 "float16": torch.float16, "fp16": torch.float16}[str(tcfg.get("amp_dtype", "float16"))]
    if amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        say("[warn] bf16 not supported on this GPU, falling back to fp16")
        amp_dtype = torch.float16

    def amp_ctx():
        return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp)

    budget = float(cfg.get("eval", {}).get("vram_budget_gib", 7.0))
    channels_last = bool(tcfg.get("channels_last", False)) and use_cuda
    accum = max(int(tcfg.get("grad_accum", 1)), 1)

    say(f"device   : {device} ({torch.cuda.get_device_name(local_rank) if use_cuda else '-'})"
        f"  world={world}")
    if world > 1 and main_proc:
        for r in range(torch.cuda.device_count()):
            print(f"           cuda:{r} {torch.cuda.get_device_name(r)}")
    say(f"torch    : {torch.__version__}  amp={amp} ({amp_dtype})  channels_last={channels_last}")
    say(f"config   : {args.config}")
    say(f"out dir  : {out_dir}")

    # ---- data ------------------------------------------------------------ #
    common = dict(img_h=mcfg["img_h"], img_w=mcfg["img_w"], n_cols=mcfg["n_cols"],
                  layout_key=dcfg.get("layout", "layout_visible"),
                  fast_jpeg=bool(dcfg.get("fast_jpeg", False)))
    nw = int(dcfg.get("num_workers", 4))
    prefetch = int(dcfg.get("prefetch_factor", 2))
    # WSL2 limits page-locked host memory; pinning can fail or thrash there.
    common["pin_memory"] = bool(dcfg.get("pin_memory", True))
    train_loader = make_loader(
        dcfg["train_index"], dcfg["root"], batch_size=tcfg["batch_size"], shuffle=True,
        num_workers=nw, augment=dcfg.get("augment", True), drop_last=True,
        prefetch_factor=prefetch, distributed=world > 1, seed=seed,
        flip_prob=float(dcfg.get("flip_prob", 0.0)),
        max_pairs=int(dcfg.get("max_train_pairs", 0)),  # debug / smoke tests only
        **common,
    )
    val_n = args.val_subsample if args.val_subsample is not None else int(tcfg.get("val_subsample", 2000))
    val_bs = int(tcfg.get("val_batch_size", tcfg["batch_size"]))
    val_loader = make_loader(
        dcfg["val_index"], dcfg["root"], batch_size=val_bs, shuffle=False,
        num_workers=max(nw // 2, 1), augment=False,
        subsample_n=val_n, subsample_seed=1234, shard=(rank, world), **common,
    )
    quickval_loader = make_loader(
        dcfg["val_index"], dcfg["root"], batch_size=val_bs, shuffle=False,
        num_workers=max(nw // 4, 1), augment=False,
        subsample_n=args.quickval_subsample, subsample_seed=99, shard=(rank, world), **common,
    )
    steps_per_epoch = len(train_loader) // accum
    say(f"train    : {len(train_loader.dataset)} pairs, {steps_per_epoch} optimizer steps/epoch "
        f"@ batch {tcfg['batch_size']} x {world} GPU x accum {accum} "
        f"= {tcfg['batch_size'] * world * accum} pairs/step")
    say(f"val      : {len(val_loader.dataset) * world} pairs (epoch-end, ~{val_n or 'full'}), "
        f"{len(quickval_loader.dataset) * world} (periodic)")

    # ---- model ----------------------------------------------------------- #
    model_cfg = config_from_dict(mcfg)
    core = build_model(model_cfg).to(device)
    if channels_last:
        core.column_encoder.to(memory_format=torch.channels_last)
    say(f"model    : {model_cfg}")
    say(f"params   : { {k: round(v, 2) for k, v in count_params(core).items()} }")

    full_weights = LossWeights(**cfg["loss"])
    if (full_weights.geo_t or full_weights.geo_yaw) and not model_cfg.geo_pose:
        raise ValueError("loss.geo_* > 0 needs model.geo_pose: true")
    geo_start = int(tcfg.get("geo_start_steps", 0))
    geo_ramp = int(tcfg.get("geo_ramp_steps", 1))
    say(f"loss w   : {full_weights}  (geo terms ramp in over steps "
        f"{geo_start}..{geo_start + geo_ramp})")

    wd = float(tcfg.get("weight_decay", 1e-4))
    groups = core.param_groups(
        float(tcfg["lr"]), float(tcfg.get("backbone_lr_mult", 0.1)),
        weight_decay=wd if tcfg.get("no_decay_norms", False) else None,
    )
    opt_kw = {"weight_decay": wd, "betas": tuple(tcfg.get("betas", (0.9, 0.999)))}
    if use_cuda and tcfg.get("fused_adamw", False):
        opt_kw["fused"] = True
    opt = torch.optim.AdamW(groups, **opt_kw)
    scheduler, total_sched_steps = build_scheduler(opt, cfg, steps_per_epoch)
    # Loss scaling is only needed for fp16; bf16 has fp32's range.
    use_scaler = amp and amp_dtype == torch.float16
    scaler = (torch.amp.GradScaler("cuda", enabled=use_scaler) if hasattr(torch, "amp")
              and hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=use_scaler))
    health = HealthMonitor()

    ema = None
    if float(tcfg.get("ema_decay", 0.0)) > 0:
        ema = ModelEMA(core, float(tcfg["ema_decay"]), int(tcfg.get("ema_warmup_steps", 2000)))

    # ---- resume ---------------------------------------------------------- #
    start_epoch, global_step = 0, 0
    resume_path = find_resume(out_dir, args.resume or cfg["ckpt"].get("resume", "auto"))
    if resume_path:
        info = load_checkpoint(resume_path, model=core, optimizer=opt, scheduler=scheduler,
                               scaler=scaler, health=health, map_location=device, ema=ema)
        start_epoch, global_step = info["epoch"], info["global_step"]
        say(f"resumed  : {resume_path} (epoch {start_epoch}, step {global_step})")
    else:
        say("resumed  : no checkpoint found, starting fresh")

    model = core
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            core, device_ids=[local_rank] if use_cuda else None,
            output_device=local_rank if use_cuda else None,
            gradient_as_bucket_view=True, static_graph=bool(tcfg.get("static_graph", False)),
        )
    if tcfg.get("compile", False):
        model = torch.compile(model)

    logger = JsonlLogger(os.path.join(out_dir, "log.jsonl"), enabled=main_proc)

    # ---- trivial baselines, so we can tell a dead head from a hard one --- #
    baseline = None
    if main_proc:
        print("\ncomputing trivial baselines on the validation subset...")
        baseline = trivial_baselines(quickval_loader, device)
        print(f"baseline : {fmt_metrics(baseline)}  "
              f"(covis positive rate {baseline['covis_pos_rate']:.3f})")
        logger.log({"kind": "baseline", **baseline})

    planned = tcfg["epochs"] if not args.max_epochs else min(
        start_epoch + args.max_epochs, tcfg["epochs"])
    say(f"\nrunning epochs {start_epoch} .. {planned-1} "
        f"(LR schedule horizon = {tcfg['epochs']} epochs = {total_sched_steps} steps)\n")

    def ckpt_kw(epoch, extra=None):
        return dict(model=core, optimizer=opt, scheduler=scheduler, scaler=scaler,
                    epoch=epoch, global_step=global_step, health=health, config=cfg,
                    ema=ema, extra=extra)

    model.train()
    stop_all = False       # any early exit, including the debug --limit-steps
    aborted = False        # specifically a health / VRAM failure
    t_start = time.time()
    # Track the best epoch explicitly and keep a dedicated copy. The best epoch
    # is not necessarily the last, and relying on the per-epoch snapshots means
    # relying on the retention policy not having pruned the winner.
    best_score, best_epoch = float("inf"), -1
    _best_path = os.path.join(out_dir, "best.pt")
    if main_proc and os.path.isfile(_best_path):
        try:
            _x = torch_load(_best_path, map_location="cpu").get("extra", {})
            best_score = float(_x.get("score", _x.get("val_loss", {}).get("total", float("inf"))))
            best_epoch = int(_x.get("best_epoch", -1))
            print(f"best so far: epoch {best_epoch}, score {best_score:.4f} (from best.pt)")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] could not read {_best_path} ({e}); best tracking restarts")

    grad_clip = float(tcfg.get("grad_clip", 5.0))
    for epoch in range(start_epoch, planned):
        if world > 1:
            train_loader.sampler.set_epoch(epoch)
        ep_t0 = time.time()
        run, run_n, n_clipped = {}, 0, 0
        opt.zero_grad(set_to_none=True)
        weights = ramped_weights(full_weights, global_step, geo_start, geo_ramp)

        for it, batch in enumerate(train_loader):
            if it >= steps_per_epoch * accum:  # drop an incomplete accumulation tail
                break
            if args.limit_steps and global_step >= args.limit_steps:
                stop_all = True
                say(f"[stop] --limit-steps {args.limit_steps} reached")
                if main_proc:
                    save_checkpoint(os.path.join(out_dir, "last.pt"), **ckpt_kw(epoch))
                break

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if channels_last:
                for k in ("img_a", "img_b"):
                    batch[k] = batch[k].contiguous(memory_format=torch.channels_last)

            is_step = (it + 1) % accum == 0
            sync_ctx = model.no_sync if (world > 1 and not is_step) else contextlib.nullcontext
            with sync_ctx():
                with amp_ctx():
                    out = model(batch["img_a"], batch["img_b"])
                terms = covispose_loss(out, batch, weights)
                scaler.scale(terms["total"] / accum).backward()
            if not is_step:
                continue

            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(core.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            if ema is not None:
                ema.update(core)
            weights = ramped_weights(full_weights, global_step, geo_start, geo_ramp)

            # one sync per step: average the terms across ranks so every rank's
            # health monitor sees the same numbers and makes the same decision
            keys = sorted(terms)
            vec = torch.stack([terms[k].detach().float() for k in keys] + [gnorm.detach().float()])
            all_reduce_(vec)
            vec = (vec / world).tolist()
            tvals = dict(zip(keys, vec[:-1]))
            gn = vec[-1]
            n_clipped += int(gn > grad_clip)

            rep = health.update(tvals)
            for k, v in tvals.items():
                run[k] = run.get(k, 0.0) + v
            run_n += 1

            if rep.should_stop:
                say("\n" + "=" * 70)
                say("TRAINING STOPPED -- health check failed:")
                for r in rep.reasons:
                    say("  * " + r)
                say("=" * 70)
                logger.log({"kind": "abort", "step": global_step, "reasons": rep.reasons})
                if main_proc:
                    save_checkpoint(os.path.join(out_dir, "abort.pt"), **ckpt_kw(epoch))
                stop_all = aborted = True
                break

            if global_step % args.log_every == 0:
                logger.log({"kind": "train", "epoch": epoch, "step": global_step,
                            "lr": scheduler.get_last_lr()[-1], "grad_norm": gn,
                            **{f"train_{k}": v for k, v in tvals.items()}})

            if global_step % 200 == 0:
                done = (it + 1) // accum
                ep_el = time.time() - ep_t0
                eta = timedelta(seconds=int((steps_per_epoch - done) * ep_el / max(done, 1)))
                avg = {k: v / max(run_n, 1) for k, v in run.items()}
                geo = (f" geo_t={avg['geo_t']:.4f} geo_yaw={avg['geo_yaw']:.4f}"
                       if "geo_t" in avg else "")
                say(f"  e{epoch} {done:6d}/{steps_per_epoch}  "
                    f"total={avg['total']:.4f} covis={avg['covis']:.4f} "
                    f"corr={avg['corr']:.4f} bon={avg['bon']:.4f} "
                    f"t={avg['pose_t']:.4f} yaw={avg['pose_yaw']:.4f}{geo}  "
                    f"lr={scheduler.get_last_lr()[-1]:.2e} gn={gn:.2f} "
                    f"clip={n_clipped / max(run_n, 1):.0%}  "
                    f"{ep_el / done * 1000:.0f}ms/step "
                    f"{tcfg['batch_size'] * world * accum * done / ep_el:.1f}pairs/s  eta={eta}")
                run, run_n, n_clipped = {}, 0, 0

            # VRAM guard, once the steady state is reached
            if use_cuda and global_step in (50, 200):
                total_gib = torch.cuda.get_device_properties(local_rank).total_memory / 2**30
                msg = vram_guard(min(budget, 0.98 * total_gib), f"rank {rank} step {global_step}")
                flag = torch.tensor([1.0 if msg else 0.0], device=device)
                all_reduce_(flag)
                if flag.item() > 0:
                    say("\n" + "=" * 70)
                    say("TRAINING STOPPED -- " + (msg or "VRAM budget exceeded on another rank"))
                    say("=" * 70)
                    logger.log({"kind": "abort", "step": global_step, "reasons": [msg or "vram"]})
                    stop_all = aborted = True
                    break
                say(f"  [vram] rank0 {torch.cuda.max_memory_reserved(local_rank) / 2**30:.2f} "
                    f"GiB reserved (budget {min(budget, 0.98 * total_gib):.2f})")

            if args.quickval_every and global_step % args.quickval_every == 0:
                vl, vm, vn = validate(core, quickval_loader, weights, device, amp_ctx)
                if main_proc:
                    brep = check_against_baselines(vm, baseline)
                    print(f"  [quickval @ {global_step}] total={vl['total']:.4f} "
                          f"covis={vl['covis']:.4f} corr={vl['corr']:.4f} bon={vl['bon']:.4f} "
                          f"t={vl['pose_t']:.4f} yaw={vl['pose_yaw']:.4f} | {fmt_metrics(vm)}")
                    for w_ in brep.warnings:
                        print(f"  [WARN] {w_}")
                    logger.log({"kind": "quickval", "epoch": epoch, "step": global_step,
                                **{f"val_{k}": v for k, v in vl.items()},
                                **{f"m_{k}": v for k, v in vm.items()},
                                "warnings": brep.warnings})

            if main_proc and args.save_every_steps and global_step % args.save_every_steps == 0:
                save_checkpoint(os.path.join(out_dir, "last.pt"), **ckpt_kw(epoch))

        if stop_all:
            break

        # ---- end of epoch ------------------------------------------------ #
        # Score = val loss under the FULL weights, on the EMA weights if any.
        vl, vm, vn = validate(core, val_loader, full_weights, device, amp_ctx)
        evl, evm = None, None
        if ema is not None:
            evl, evm, _ = validate(ema.module, val_loader, full_weights, device, amp_ctx)
        score = (evl or vl)["total"]
        ep_dt = time.time() - ep_t0

        if main_proc:
            brep = check_against_baselines(evm or vm, baseline)
            print("\n" + "-" * 70)
            print(f"epoch {epoch} done in {timedelta(seconds=int(ep_dt))}  ({vn} val pairs)")
            for label, l_, m_ in (("raw", vl, vm), ("ema", evl, evm)):
                if l_ is None:
                    continue
                geo = (f" geo_t={l_['geo_t']:.4f} geo_yaw={l_['geo_yaw']:.4f}"
                       if "geo_t" in l_ else "")
                print(f"  {label} loss : total={l_['total']:.4f} covis={l_['covis']:.4f} "
                      f"corr={l_['corr']:.4f} bon={l_['bon']:.4f} "
                      f"t={l_['pose_t']:.4f} yaw={l_['pose_yaw']:.4f}{geo}")
                print(f"  {label} met  : {fmt_metrics(m_)}")
            print(f"  baseline : {fmt_metrics(baseline)}")
            for w_ in brep.warnings:
                print(f"  [WARN] {w_}")
            print("-" * 70 + "\n")

            rec = {"kind": "epoch", "epoch": epoch, "step": global_step, "seconds": ep_dt,
                   "score": score,
                   **{f"val_{k}": v for k, v in vl.items()},
                   **{f"m_{k}": v for k, v in vm.items()},
                   "warnings": brep.warnings}
            if evl is not None:
                rec.update({f"ema_val_{k}": v for k, v in evl.items()})
                rec.update({f"ema_m_{k}": v for k, v in evm.items()})
            logger.log(rec)

            extra = {"val_loss": vl, "val_metrics": vm, "ema_val_loss": evl,
                     "ema_val_metrics": evm, "score": score}
            save_checkpoint(os.path.join(out_dir, f"epoch_{epoch:03d}.pt"),
                            **ckpt_kw(epoch + 1, extra))
            save_checkpoint(os.path.join(out_dir, "last.pt"), **ckpt_kw(epoch + 1))

            if score < best_score:
                best_score, best_epoch = score, epoch
                save_checkpoint(_best_path, **ckpt_kw(epoch + 1, {**extra, "best_epoch": epoch}))
                print(f"  new best score={best_score:.4f} (epoch {best_epoch}) -> best.pt")
            else:
                print(f"  score {score:.4f} did not beat epoch {best_epoch} "
                      f"({best_score:.4f}); best.pt unchanged")

            prune_epoch_checkpoints(out_dir, int(cfg["ckpt"].get("keep_last", 3)))
        barrier()

    logger.close()
    say(f"finished at {datetime.now().isoformat(timespec='seconds')}, "
        f"total {timedelta(seconds=int(time.time() - t_start))}")
    if use_cuda:
        say(f"peak VRAM reserved (rank 0): {torch.cuda.max_memory_reserved(local_rank) / 2**30:.2f} GiB")
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    return 1 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
