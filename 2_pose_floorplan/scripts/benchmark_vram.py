"""Measure what actually fits and how fast it runs on this GPU.

Everything here is measured, not estimated: peak VRAM, ms/step and the AMP
speed/memory delta for a grid of (backbone, resolution, batch size).

    python scripts/benchmark_vram.py
    python scripts/benchmark_vram.py --steps 12 --data-throughput
"""

import argparse
import gc
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from covispose.losses import LossWeights, covispose_loss
from covispose.model import CoVisPoseConfig, build_model, count_params


def synth_batch(bs, cfg, device):
    n = cfg.n_cols
    return {
        "img_a": torch.rand(bs, 3, cfg.img_h, cfg.img_w, device=device),
        "img_b": torch.rand(bs, 3, cfg.img_h, cfg.img_w, device=device),
        "bon_a": torch.rand(bs, n, device=device) * 0.8 + 0.2,
        "bon_b": torch.rand(bs, n, device=device) * 0.8 + 0.2,
        "bon_valid_a": torch.ones(bs, n, dtype=torch.bool, device=device),
        "bon_valid_b": torch.ones(bs, n, dtype=torch.bool, device=device),
        "covis_a": (torch.rand(bs, n, device=device) > 0.4).float(),
        "covis_b": (torch.rand(bs, n, device=device) > 0.4).float(),
        "corr_a": (torch.rand(bs, n, device=device) - 0.5) * 6.28,
        "corr_b": (torch.rand(bs, n, device=device) - 0.5) * 6.28,
        "t": torch.randn(bs, 2, device=device),
        "yaw": (torch.rand(bs, device=device) - 0.5) * 6.28,
    }


def bench(backbone, img_h, img_w, n_cols, bs, amp, steps, device, accum=1):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg = CoVisPoseConfig(
        backbone=backbone, pretrained=False, img_h=img_h, img_w=img_w, n_cols=n_cols
    )
    try:
        model = build_model(cfg).to(device).train()
        opt = torch.optim.AdamW(model.param_groups(1e-4), weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        w = LossWeights()
        batch = synth_batch(bs, cfg, device)

        for i in range(3):  # warmup
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                loss = covispose_loss(model(batch["img_a"], batch["img_b"]), batch, w)["total"]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for i in range(steps):
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                loss = covispose_loss(model(batch["img_a"], batch["img_b"]), batch, w)["total"]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / steps

        peak_alloc = torch.cuda.max_memory_allocated() / 2**30
        peak_resv = torch.cuda.max_memory_reserved() / 2**30
        params = count_params(model)["total_M"]
        res = dict(ok=True, ms=dt * 1000, alloc=peak_alloc, resv=peak_resv,
                   params=params, pairs_s=bs / dt)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        res = dict(ok=False, ms=float("nan"), alloc=float("nan"),
                   resv=float("nan"), params=float("nan"), pairs_s=float("nan"))
    finally:
        for v in ("model", "opt", "batch", "scaler"):
            if v in dir():
                pass
        try:
            del model, opt, batch, scaler
        except Exception:  # noqa: BLE001
            pass
        gc.collect()
        torch.cuda.empty_cache()
    return res


def bench_config(config_path, batch_sizes, steps, device):
    """Measure a training config as the training loop will run it.

    Same model options, same autocast dtype, channels_last and loss weights as
    ``scripts/train.py``, on synthetic batches. Stops at the first OOM.
    """
    from covispose.ema import ModelEMA
    from covispose.model import config_from_dict
    from scripts.train import load_config  # noqa: E402

    cfg = load_config(config_path)
    tcfg = cfg["train"]
    mcfg = config_from_dict(cfg["model"], pretrained=False)
    dtype = torch.bfloat16 if str(tcfg.get("amp_dtype", "")).startswith("bf") else torch.float16
    cl = bool(tcfg.get("channels_last", False))
    torch.backends.cuda.matmul.allow_tf32 = bool(tcfg.get("tf32", True))
    torch.backends.cudnn.allow_tf32 = bool(tcfg.get("tf32", True))
    torch.backends.cudnn.benchmark = True
    w = LossWeights(**cfg["loss"])
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"config {config_path}: {mcfg}\n")
    print(f"{'bs':>3s} | {'peak alloc':>10s} {'reserved':>9s} | {'ms/step':>8s} {'pairs/s':>8s}")
    for bs in batch_sizes:
        torch.cuda.empty_cache()
        try:
            model = build_model(mcfg).to(device).train()
            if cl:
                model.column_encoder.to(memory_format=torch.channels_last)
            ema = ModelEMA(model) if float(tcfg.get("ema_decay", 0)) > 0 else None
            opt = torch.optim.AdamW(model.param_groups(1e-4, weight_decay=0.05))
            batch = synth_batch(bs, mcfg, device)
            if cl:
                for k in ("img_a", "img_b"):
                    batch[k] = batch[k].contiguous(memory_format=torch.channels_last)

            def step():
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=dtype):
                    out = model(batch["img_a"], batch["img_b"])
                covispose_loss(out, batch, w)["total"].backward()
                opt.step()
                if ema is not None:
                    ema.update(model)

            for _ in range(3):
                step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(steps):
                step()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / steps
            print(f"{bs:3d} | {torch.cuda.max_memory_allocated() / 2**30:8.2f}GiB "
                  f"{torch.cuda.max_memory_reserved() / 2**30:7.2f}GiB | "
                  f"{dt * 1000:7.0f} {bs / dt:8.2f}   (card {total:.0f} GiB)")
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"{bs:3d} | OOM")
            break
        finally:
            model = opt = batch = ema = None
            gc.collect()
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="benchmark this training config instead of the 1070 grid")
    ap.add_argument("--batch-sizes", default="4,6,8,10,12,16")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument(
        "--budget-gib", type=float, default=7.0,
        help="treat a config as not fitting once reserved memory passes this; "
             "leaves headroom on an 8 GiB card for the desktop compositor",
    )
    ap.add_argument("--data-throughput", action="store_true")
    ap.add_argument("--pair-index", default="training/pairs/train_c10neg.csv")
    ap.add_argument("--data-root", default=os.environ.get("ZIND_ROOT", r"C:\zind_data"))
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA device"
    device = "cuda"
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    cap = torch.cuda.get_device_capability(0)
    print(f"device: {name}  total VRAM {total:.2f} GiB  compute {cap[0]}.{cap[1]}")
    print(f"torch {torch.__version__}   steps per measurement: {args.steps}\n")

    if args.config:
        bench_config(args.config, [int(x) for x in args.batch_sizes.split(",")],
                     args.steps, device)
        if args.data_throughput:
            from scripts.train import load_config
            c = load_config(args.config)
            data_throughput(args, c["model"]["img_h"], c["model"]["img_w"],
                            c["model"]["n_cols"], bool(c["data"].get("fast_jpeg", False)),
                            workers=(4, 8, 16, 32))
        return

    # Windows/WDDM does NOT raise OOM when a config exceeds VRAM -- the driver
    # silently pages to host RAM and the step time explodes (we measured 8.4 s
    # vs 1.2 s for the same config). So we cut a config off by measured memory,
    # not by waiting for an exception, and skip larger batches once one spills.
    budget = args.budget_gib
    grid = []
    for backbone in ("resnet34", "resnet18", "resnet50"):
        for (h, w, nc) in ((512, 1024, 256), (256, 512, 128)):
            for amp in (True, False):
                for bs in (1, 2, 4, 6, 8, 12, 16):
                    grid.append((backbone, h, w, nc, bs, amp))

    print(f"{'backbone':9s} {'res':10s} {'bs':>2s} {'amp':>5s} | "
          f"{'peak alloc':>10s} {'reserved':>9s} | {'ms/step':>8s} {'pairs/s':>8s}")
    print("-" * 80)
    results = {}
    spilled = set()
    for backbone, h, w, nc, bs, amp in grid:
        tag = f"{h}x{w}"
        series = (backbone, h, w, amp)
        if series in spilled:
            print(f"{backbone:9s} {tag:10s} {bs:2d} {str(amp):>5s} | "
                  f"{'skipped':>10s} {'':>9s} | (a smaller batch already exceeded budget)")
            continue

        r = bench(backbone, h, w, nc, bs, amp, args.steps, device)
        results[(backbone, h, w, bs, amp)] = r
        if not r["ok"]:
            print(f"{backbone:9s} {tag:10s} {bs:2d} {str(amp):>5s} | {'OOM':>10s}")
            spilled.add(series)
            continue

        over = r["resv"] > budget
        flag = "  <-- SPILLS to host RAM" if over else ""
        print(f"{backbone:9s} {tag:10s} {bs:2d} {str(amp):>5s} | "
              f"{r['alloc']:8.2f}GiB {r['resv']:7.2f}GiB | "
              f"{r['ms']:7.0f} {r['pairs_s']:8.2f}{flag}")
        if over:
            spilled.add(series)

    # ---- AMP verdict on this GPU ---------------------------------------- #
    print("\nAMP effect (same config, amp=True vs amp=False):")
    for key in sorted({(b, h, w, bs) for (b, h, w, bs, _) in results}):
        b, h, w, bs = key
        on, off = results.get((b, h, w, bs, True)), results.get((b, h, w, bs, False))
        if not (on and off and on["ok"] and off["ok"]):
            continue
        print(f"  {b:9s} {h}x{w} bs={bs}: memory {off['alloc']/on['alloc']:.2f}x lower, "
              f"speed {off['ms']/on['ms']:.2f}x faster")

    if args.data_throughput:
        data_throughput(args)


def data_throughput(args, img_h=512, img_w=1024, n_cols=256, fast_jpeg=False,
                    workers=(0, 2, 4)):
    from covispose.dataset import make_loader
    print(f"\ndata loader throughput (pairs/s, includes JPEG decode + target build; "
          f"{img_h}x{img_w}, {n_cols} cols, fast_jpeg={fast_jpeg}):")
    for nw in workers:
        dl = make_loader(args.pair_index, args.data_root, batch_size=8, shuffle=True,
                         num_workers=nw, augment=True, img_h=img_h, img_w=img_w,
                         n_cols=n_cols, fast_jpeg=fast_jpeg, flip_prob=0.5)
        it = iter(dl)
        next(it)
        t0 = time.perf_counter()
        n = 0
        for _ in range(max(10, 2 * nw)):
            b = next(it)
            n += b["img_a"].shape[0]
        dt = time.perf_counter() - t0
        print(f"  num_workers={nw}: {n/dt:6.2f} pairs/s")
        del dl, it


if __name__ == "__main__":
    main()
