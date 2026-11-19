"""Resume must continue the *same* trajectory, not just load weights.

A 9-hour epoch will get interrupted. The thing that makes an interruption
harmless is that optimizer moments, the LR schedule position, the AMP scaler
and the step counter all come back -- reloading only the weights silently
restarts the optimizer and changes the run.

    python tests/test_checkpoint.py
"""

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from covispose.checkpoint import find_resume, load_checkpoint, save_checkpoint  # noqa: E402
from covispose.health import HealthMonitor  # noqa: E402
from covispose.model import CoVisPoseConfig, build_model  # noqa: E402

SMALL = CoVisPoseConfig(
    backbone="resnet18", pretrained=False, img_h=128, img_w=256,
    n_cols=64, d_model=32, n_heads=4, n_layers=2, dim_ff=64, dropout=0.0,
)


def _make(device, amp):
    torch.manual_seed(0)
    m = build_model(SMALL).to(device)
    o = torch.optim.AdamW(m.param_groups(1e-3), weight_decay=1e-4)
    s = torch.optim.lr_scheduler.LambdaLR(o, lambda i: min((i + 1) / 10.0, 1.0))
    sc = torch.cuda.amp.GradScaler(enabled=amp)
    h = HealthMonitor()
    return m, o, s, sc, h


def _step(m, o, s, sc, device, seed, amp):
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.rand(2, 3, SMALL.img_h, SMALL.img_w, generator=g).to(device)
    b = torch.rand(2, 3, SMALL.img_h, SMALL.img_w, generator=g).to(device)
    o.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=amp):
        out = m(a, b)
        loss = sum(v.float().pow(2).mean() for v in out.values())
    sc.scale(loss).backward()
    sc.step(o)
    sc.update()
    s.step()
    return float(loss)


def test_resume_restores_state_exactly():
    """Everything that defines the trajectory must come back **bit-exact**.

    This is the real correctness test, and it is exact because it compares the
    restored state against the saved state directly -- no forward or backward
    pass is involved, so GPU non-determinism cannot muddy it.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "last.pt")

    mA, oA, sA, scA, hA = _make(device, amp)
    for i in range(3):
        _step(mA, oA, sA, scA, device, 100 + i, amp)
    snap_model = {k: v.detach().clone() for k, v in mA.state_dict().items()}
    snap_opt = {i: {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
                for i, st in enumerate(oA.state.values())}
    save_checkpoint(path, model=mA, optimizer=oA, scheduler=sA, scaler=scA,
                    epoch=1, global_step=3, health=hA, config={"x": 1})

    mB, oB, sB, scB, hB = _make(device, amp)
    with torch.no_grad():  # perturb, so a no-op load could not pass
        for p in mB.parameters():
            p.add_(1.0)
    info = load_checkpoint(path, model=mB, optimizer=oB, scheduler=sB, scaler=scB,
                           health=hB, map_location=device)

    assert info["epoch"] == 1 and info["global_step"] == 3, info
    assert info["config"] == {"x": 1}

    worst = max((v.float() - snap_model[k].float()).abs().max().item()
                for k, v in mB.state_dict().items())
    assert worst == 0.0, f"model state not restored exactly: {worst}"

    # optimizer moments -- exactly what a weights-only reload silently loses
    stB = list(oB.state.values())
    assert len(stB) == len(snap_opt) and len(stB) > 0
    wm = 0.0
    for i, st in enumerate(stB):
        for key in ("exp_avg", "exp_avg_sq"):
            wm = max(wm, (st[key] - snap_opt[i][key]).abs().max().item())
        assert float(st["step"]) == float(snap_opt[i]["step"])
    assert wm == 0.0, f"optimizer moments not restored exactly: {wm}"

    assert sA.last_epoch == sB.last_epoch == 3
    assert sA.get_last_lr() == sB.get_last_lr(), (sA.get_last_lr(), sB.get_last_lr())
    assert scA.get_scale() == scB.get_scale()
    assert hB.steps == hA.steps and hB.ema == hA.ema
    print(f"  ok  test_resume_restores_state_exactly "
          f"(model/optimizer restored bit-exact, lr {sB.get_last_lr()[-1]:.3e}, "
          f"scaler {scB.get_scale()})")


def _max_param_delta(m1, m2):
    return max((p1 - p2).abs().max().item()
               for (_, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()))


def test_resume_continues_identically():
    """End-to-end: an interrupted run must continue like an uninterrupted one.

    A fixed tolerance is the wrong instrument here, because the GPU is not
    deterministic: cuDNN's convolution backward uses atomics, so two identical
    backward passes disagree by ~6e-7 without AMP and ~1e-3 with it (measured),
    and Adam amplifies that over subsequent steps.

    So the test calibrates itself. It measures the noise floor directly, by
    resuming *twice* from the same checkpoint through the identical code path,
    then requires the uninterrupted-vs-resumed difference to be of the same
    order. That distinguishes "resume is broken" from "the GPU is noisy",
    which a hardcoded threshold cannot. See docs/ASSUMPTIONS.md A13.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = False  # keeps the noise floor small enough for the test to have teeth
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "last.pt")

    def continue_from_checkpoint():
        m, o, s, sc, h = _make(device, amp)
        with torch.no_grad():  # perturb, so a no-op load could not pass
            for p in m.parameters():
                p.add_(1.0)
        load_checkpoint(path, model=m, optimizer=o, scheduler=s, scaler=sc,
                        health=h, map_location=device)
        for i in range(3, 6):
            _step(m, o, s, sc, device, 100 + i, amp)
        return m, s

    # uninterrupted reference
    mA, oA, sA, scA, hA = _make(device, amp)
    for i in range(3):
        _step(mA, oA, sA, scA, device, 100 + i, amp)
    save_checkpoint(path, model=mA, optimizer=oA, scheduler=sA, scaler=scA,
                    epoch=1, global_step=3, health=hA)
    for i in range(3, 6):
        _step(mA, oA, sA, scA, device, 100 + i, amp)

    mB, sB = continue_from_checkpoint()
    mC, _ = continue_from_checkpoint()

    noise_floor = _max_param_delta(mB, mC)   # same code path twice = pure GPU noise
    resume_err = _max_param_delta(mA, mB)    # uninterrupted vs resumed

    tol = max(5.0 * noise_floor, 1e-6)
    assert resume_err <= tol, (
        f"resumed run diverged from the uninterrupted one by {resume_err:.3e}, "
        f"which is more than {tol:.3e} (5x the measured GPU noise floor "
        f"{noise_floor:.3e}) -- that is a real checkpoint bug, not GPU noise"
    )
    assert sA.get_last_lr() == sB.get_last_lr()
    print(f"  ok  test_resume_continues_identically "
          f"(resume error {resume_err:.2e} vs GPU noise floor {noise_floor:.2e})")


def test_adamw_step_stays_on_cpu_after_resume():
    """torch 1.12 asserts if AdamW's `step` state lands on CUDA after a load."""
    if not torch.cuda.is_available():
        print("  skip test_adamw_step_stays_on_cpu_after_resume (no CUDA)")
        return
    device = "cuda"
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "last.pt")

    m, o, s, sc, h = _make(device, amp=True)
    _step(m, o, s, sc, device, 7, amp=True)
    save_checkpoint(path, model=m, optimizer=o, scheduler=s, scaler=sc,
                    epoch=0, global_step=1, health=h)

    m2, o2, s2, sc2, h2 = _make(device, amp=True)
    load_checkpoint(path, model=m2, optimizer=o2, scheduler=s2, scaler=sc2,
                    health=h2, map_location=device)
    for st in o2.state.values():
        if torch.is_tensor(st.get("step")):
            assert not st["step"].is_cuda, "AdamW step tensor must stay on CPU"
    _step(m2, o2, s2, sc2, device, 8, amp=True)  # would assert without the fix
    print("  ok  test_adamw_step_stays_on_cpu_after_resume")


def test_ema_is_saved_and_restored():
    from covispose.ema import ModelEMA

    device = "cpu"
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "last.pt")
    m, o, s, sc, h = _make(device, amp=False)
    ema = ModelEMA(m, decay=0.99, warmup_steps=1)
    for i in range(3):
        _step(m, o, s, sc, device, 200 + i, amp=False)
        ema.update(m)
    # the EMA must actually have moved off the initial weights, and not onto m
    p0 = next(ema.module.parameters())
    assert not torch.equal(p0, next(m.parameters()))
    save_checkpoint(path, model=m, optimizer=o, scheduler=s, scaler=sc, epoch=0,
                    global_step=3, health=h, ema=ema)

    m2, o2, s2, sc2, h2 = _make(device, amp=False)
    ema2 = ModelEMA(m2, decay=0.99, warmup_steps=1)
    load_checkpoint(path, model=m2, optimizer=o2, scheduler=s2, scaler=sc2, health=h2, ema=ema2)
    for (k, a), (_, b) in zip(ema.module.state_dict().items(), ema2.module.state_dict().items()):
        assert torch.equal(a, b), k
    assert ema2.updates == 3

    # a checkpoint written without EMA seeds the EMA from the weights
    save_checkpoint(path, model=m, optimizer=o, scheduler=s, scaler=sc, epoch=0, global_step=3)
    ema3 = ModelEMA(_make(device, amp=False)[0])
    load_checkpoint(path, model=m2, ema=ema3)
    assert torch.equal(next(ema3.module.parameters()), next(m.parameters()))
    print("  ok  test_ema_is_saved_and_restored")


def test_find_resume():
    tmp = tempfile.mkdtemp()
    assert find_resume(tmp, "auto") is None
    open(os.path.join(tmp, "epoch_000.pt"), "w").close()
    open(os.path.join(tmp, "epoch_001.pt"), "w").close()
    assert find_resume(tmp, "auto").endswith("epoch_001.pt")
    open(os.path.join(tmp, "last.pt"), "w").close()
    assert find_resume(tmp, "auto").endswith("last.pt")
    assert find_resume(tmp, "none") is None
    print("  ok  test_find_resume")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{len(fns)} checkpoint checks passed")
