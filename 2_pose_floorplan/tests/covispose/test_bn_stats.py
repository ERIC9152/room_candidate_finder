"""Regression test: constructing the model must not disturb pretrained weights.

The model runs two probe forward passes at construction time (to discover the
backbone's channel widths and the column-feature width). If either runs in
train mode, BatchNorm updates its running statistics towards the statistics of
the all-zero probe input -- 10 % of the way, at the default momentum -- which
silently degrades the ImageNet initialisation before training starts.
"""

import os
import sys

import torch
import torchvision.models as tvm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from covispose.model import CoVisPoseConfig, build_model  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _requires import needs_download  # noqa: E402


@needs_download
def test_construction_preserves_pretrained_bn_stats():
    ref = tvm.resnet34(weights="IMAGENET1K_V1")
    ref_bn = {
        n: (m.running_mean.clone(), m.running_var.clone())
        for n, m in ref.named_modules()
        if isinstance(m, torch.nn.BatchNorm2d)
    }

    model = build_model(CoVisPoseConfig(backbone="resnet34", pretrained=True))
    got_bn = {
        n: (m.running_mean, m.running_var)
        for n, m in model.column_encoder.encoder.net.named_modules()
        if isinstance(m, torch.nn.BatchNorm2d)
    }

    checked = 0
    for name, (mean, var) in ref_bn.items():
        if name not in got_bn:
            continue
        gm, gv = got_bn[name]
        assert torch.allclose(mean, gm, atol=1e-6), f"{name}: running_mean drifted"
        assert torch.allclose(var, gv, atol=1e-6), f"{name}: running_var drifted"
        checked += 1
    assert checked >= 30, f"only compared {checked} BatchNorm layers"
    print(f"  ok  test_construction_preserves_pretrained_bn_stats ({checked} BN layers intact)")


@needs_download
def test_construction_preserves_pretrained_conv_weights():
    ref = tvm.resnet34(weights="IMAGENET1K_V1")
    model = build_model(CoVisPoseConfig(backbone="resnet34", pretrained=True))
    got = dict(model.column_encoder.encoder.net.named_parameters())
    checked = 0
    for name, p in ref.named_parameters():
        if name.startswith("fc."):
            continue
        # wrap_lr_pad wraps convs in a Sequential, so 'layer1.0.conv1.weight'
        # becomes 'layer1.0.conv1.1.weight'
        cand = [name, name.replace(".weight", ".1.weight").replace(".bias", ".1.bias")]
        match = next((c for c in cand if c in got), None)
        assert match is not None, f"missing {name}"
        assert torch.allclose(p, got[match], atol=1e-6), f"{name}: weights differ"
        checked += 1
    assert checked >= 100, checked
    print(f"  ok  test_construction_preserves_pretrained_conv_weights ({checked} tensors intact)")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{len(fns)} pretrained-weight checks passed")
