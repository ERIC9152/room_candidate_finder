"""CoVisPose model: HorizonNet-style column encoder + pair Transformer + heads.

Shape of the thing
------------------
    image pair          2 x (B, 3, H, W)          H=512, W=1024 by default
      -> ResNet-34 (circularly padded), 4 scales
      -> HorizonNet height compression            2 x (B, C, n_cols)
      -> linear proj + azimuth PE + pano-id emb   2 x (B, n_cols, d_model)
      -> concat into one sequence                 (B, 2*n_cols, d_model)
      -> Transformer encoder (self-attention across BOTH panos)
      -> split back                               2 x (B, n_cols, d_model)
      -> 3 per-column heads     : co-visibility, correspondence, floor boundary
      -> pose decoder           : (dx, dy, dyaw), 1D conv or attention pooling
      -> optional geometric pose: weighted Procrustes over the column heads

Everything that had to be shrunk for an 8 GB GTX 1070 is marked `RTX1070` in
the comments and listed in README.md. The server-scale options (defaults off,
so the 1070 model is unchanged) are described in docs/SERVER_OPTIMIZATION.md.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.models as tvm

from .geo_pose import geometric_pose

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# circular padding (straight from HorizonNet -- essential for 360 wraparound)
# --------------------------------------------------------------------------- #
class LRPad(nn.Module):
    """Pad left/right with each other's content instead of zeros."""

    def __init__(self, padding: int = 1):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        p = self.padding
        return torch.cat([x[..., -p:], x, x[..., :p]], dim=-1)


def wrap_lr_pad(net: nn.Module) -> None:
    """Replace horizontal padding of every Conv2d / MaxPool2d with circular padding.

    HorizonNet's version of this only patches ``Conv2d``, which leaves the ResNet
    stem's ``MaxPool2d`` zero-padded and so leaves a seam at the 360 deg
    wraparound.  Patching it too takes the column encoder from ~5 % to <1e-6
    relative error under a roll of the input
    (``tests/test_model.py::test_column_encoder_is_circularly_equivariant``).
    """
    for name, m in list(net.named_modules()):
        if isinstance(m, nn.Conv2d):
            pad_h, pad_w = m.padding if isinstance(m.padding, tuple) else (m.padding, m.padding)
            if pad_w == 0:
                continue
            m.padding = (pad_h, 0)
        elif isinstance(m, nn.MaxPool2d):
            pad_h, pad_w = m.padding if isinstance(m.padding, tuple) else (m.padding, m.padding)
            if pad_w == 0:
                continue
            m.padding = (pad_h, 0)
        else:
            continue
        parts = name.split(".")
        root = functools.reduce(lambda o, i: getattr(o, i), [net] + parts[:-1])
        setattr(root, parts[-1], nn.Sequential(LRPad(int(pad_w)), m))


# --------------------------------------------------------------------------- #
# encoder
# --------------------------------------------------------------------------- #
class ResnetEncoder(nn.Module):
    """Multi-scale CNN trunk, returning the 4 stage outputs (1/4 .. 1/32).

    Despite the (checkpoint-compatible) name this also wraps torchvision's
    ConvNeXt, whose ``features`` stages sit at the same four strides, so the
    HorizonNet height compression downstream is unchanged.
    """

    def __init__(self, backbone: str = "resnet34", pretrained: bool = True,
                 weights: Optional[str] = None, drop_path: float = 0.0):
        super().__init__()
        w = (weights or "IMAGENET1K_V1") if pretrained else None
        self.is_convnext = backbone.startswith("convnext")
        if self.is_convnext:
            self.net = getattr(tvm, backbone)(weights=w, stochastic_depth_prob=drop_path)
            del self.net.classifier, self.net.avgpool
        else:
            self.net = getattr(tvm, backbone)(weights=w)
            del self.net.fc, self.net.avgpool

    def forward(self, x) -> List[torch.Tensor]:
        n = self.net
        if self.is_convnext:
            feats = []
            for i, stage in enumerate(n.features):
                x = stage(x)
                if i % 2 == 1:  # stem, s1, down, s2, down, s3, down, s4
                    feats.append(x)
            return feats
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        f1 = n.layer1(x)
        f2 = n.layer2(f1)
        f3 = n.layer3(f2)
        f4 = n.layer4(f3)
        return [f1, f2, f3, f4]

    @property
    def channels(self) -> Tuple[int, int, int, int]:
        # eval() matters: a probe forward in train mode would push the ImageNet
        # BatchNorm running stats 10 % of the way towards the statistics of an
        # all-zero input (momentum 0.1), silently degrading the pretrained
        # weights before training even starts.
        was_training = self.training
        self.eval()
        with torch.no_grad():
            out = tuple(f.shape[1] for f in self.forward(torch.zeros(1, 3, 64, 64)))
        self.train(was_training)
        return out


class ConvCompressH(nn.Module):
    """Halve feature height (HorizonNet's building block)."""

    def __init__(self, in_c: int, out_c: int, ks: int = 3):
        super().__init__()
        assert ks % 2 == 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, ks, stride=(2, 1), padding=ks // 2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class GlobalHeightConv(nn.Module):
    """Squeeze one scale down in height by 16x and resample to a common width."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.layer = nn.Sequential(
            ConvCompressH(in_c, in_c // 2),
            ConvCompressH(in_c // 2, in_c // 2),
            ConvCompressH(in_c // 2, in_c // 4),
            ConvCompressH(in_c // 4, out_c),
        )

    def forward(self, x, out_w: int):
        x = self.layer(x)
        # Circular upsample to out_w, exactly as HorizonNet does it: wrap one
        # column onto each side, interpolate, then crop the wrapped part off.
        assert out_w % x.shape[3] == 0, (
            f"out_w={out_w} must be a multiple of this scale's width {x.shape[3]}"
        )
        factor = out_w // x.shape[3]
        x = torch.cat([x[..., -1:], x, x[..., :1]], dim=-1)
        x = F.interpolate(
            x, size=(x.shape[2], out_w + 2 * factor), mode="bilinear", align_corners=False
        )
        return x[..., factor:-factor]


class ColumnEncoder(nn.Module):
    """Image -> per-column feature sequence (B, C_out, n_cols).

    This is HorizonNet's ``GlobalHeightStage``: every scale is compressed to
    height 1 and all four are concatenated, giving one token per image column
    that has seen the full vertical extent of the panorama.
    """

    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        out_scale: int = 8,
        img_h: int = 512,
        img_w: int = 1024,
        n_cols: int = 256,
        weights: Optional[str] = None,
        backbone_drop_path: float = 0.0,
    ):
        super().__init__()
        self.encoder = ResnetEncoder(backbone, pretrained, weights, backbone_drop_path)
        c1, c2, c3, c4 = self.encoder.channels
        self.cs = (c1, c2, c3, c4)
        self.out_scale = out_scale
        self.ghc = nn.ModuleList(
            [GlobalHeightConv(c, max(c // out_scale, 1)) for c in self.cs]
        )
        self.register_buffer("x_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("x_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        wrap_lr_pad(self)

        # HorizonNet's closed form -- (c1*8 + c2*4 + c3*2 + c4)/out_scale -- only
        # holds at H=512, where the four compressed heights come out as 8/4/2/1.
        # At other input heights they clamp at 1, so measure it instead.
        self._out_channels = self._probe_out_channels(img_h, img_w, n_cols)

    def _probe_out_channels(self, img_h: int, img_w: int, n_cols: int) -> int:
        was_training = self.training
        self.eval()  # keep BatchNorm running stats untouched by the probe
        with torch.no_grad():
            c = self.forward(torch.zeros(1, 3, img_h, img_w), n_cols).shape[1]
        self.train(was_training)
        return int(c)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, img: torch.Tensor, n_cols: int) -> torch.Tensor:
        x = (img - self.x_mean) / self.x_std
        feats = self.encoder(x)
        b = x.shape[0]
        return torch.cat(
            [g(f, n_cols).reshape(b, -1, n_cols) for g, f in zip(self.ghc, feats)], dim=1
        )


# --------------------------------------------------------------------------- #
# positional encoding
# --------------------------------------------------------------------------- #
def azimuth_encoding(n_cols: int, dim: int, device=None, dtype=None) -> torch.Tensor:
    """Sinusoidal encoding of the column *azimuth*, not the column index.

    Frequencies are the integers 1..dim/2, so the encoding is exactly periodic
    over the 360 deg sequence -- an index-based encoding is not, and would put a
    fake discontinuity at the wraparound. With dim/2 <= n_cols/2 this is a
    truncated DFT basis over the column grid: complete, orthogonal, no aliasing.
    Returns (n_cols, dim).
    """
    assert dim % 2 == 0, "azimuth encoding needs an even dim"
    j = torch.arange(n_cols, device=device, dtype=torch.float32)
    theta = ((j + 0.5) / n_cols - 0.5) * 2 * math.pi  # matches geometry.col_centers_theta

    # Never go past Nyquist for this column grid: frequencies above n_cols/2
    # alias onto lower ones and would just duplicate channels. If d_model asks
    # for more channels than the grid can carry, the surplus is left at zero.
    n_freq = min(dim // 2, n_cols // 2)
    k = torch.arange(1, n_freq + 1, device=device, dtype=torch.float32)
    ang = theta[:, None] * k[None, :]
    pe = torch.zeros(n_cols, dim, device=device, dtype=torch.float32)
    pe[:, :n_freq] = torch.sin(ang)
    pe[:, n_freq:2 * n_freq] = torch.cos(ang)
    if dtype is not None:
        pe = pe.to(dtype)
    return pe


# --------------------------------------------------------------------------- #
# pair transformer
# --------------------------------------------------------------------------- #
def drop_path(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Stochastic depth: drop the whole residual branch per sample."""
    if p == 0.0 or not training:
        return x
    keep = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(1.0 - p)
    return x * keep / (1.0 - p)


class PairEncoderLayer(nn.Module):
    """Pre-norm transformer block with stochastic depth.

    Submodule names deliberately match ``nn.TransformerEncoderLayer`` so that
    checkpoints trained with the stock layer (the RTX 1070 run) load unchanged.
    ``nn.MultiheadAttention`` dispatches to ``scaled_dot_product_attention``
    (flash / memory-efficient kernels) on torch >= 2.0 when no attention weights
    are requested, which is the case here.
    """

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float,
                 drop_path: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.drop_path = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.self_attn(h, h, h, need_weights=False)[0]
        x = x + drop_path(self.dropout1(h), self.drop_path, self.training)
        h = self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(x)))))
        return x + drop_path(self.dropout2(h), self.drop_path, self.training)


class PairTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float,
                 n_layers: int, drop_path: float = 0.0, grad_checkpoint: bool = False):
        super().__init__()
        # linearly increasing stochastic depth, the usual ViT / DeiT schedule
        rates = [drop_path * i / max(n_layers - 1, 1) for i in range(n_layers)]
        self.layers = nn.ModuleList(
            [PairEncoderLayer(d_model, nhead, dim_ff, dropout, r) for r in rates]
        )
        self.grad_checkpoint = grad_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.grad_checkpoint and self.training and x.requires_grad:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #
class ColumnHeads(nn.Module):
    """Three per-column outputs from the fused per-column features."""

    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()

        def mlp(out_dim, final_bias=None):
            layers = [
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, out_dim),
            ]
            if final_bias is not None:
                nn.init.zeros_(layers[-1].weight)
                with torch.no_grad():
                    layers[-1].bias.copy_(torch.as_tensor(final_bias))
            return nn.Sequential(*layers)

        self.covis = mlp(1)
        # correspondence is predicted as an unnormalised (cos, sin) pair so that
        # the +-pi wraparound is not a discontinuity in the output space.
        self.corr = mlp(2)
        # floor boundary in HorizonNet 'bon' radians; init at atan2(1, 2) which
        # is a wall two camera heights away -- roughly the dataset median.
        self.bon = mlp(1, final_bias=[math.atan2(1.0, 2.0)])

    def forward(self, tok: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "covis_logit": self.covis(tok).squeeze(-1),  # (B, n_cols)
            "corr_vec": self.corr(tok),                  # (B, n_cols, 2)
            "bon": self.bon(tok).squeeze(-1),            # (B, n_cols)
        }


class PoseDecoder(nn.Module):
    """Lightweight 1D conv decoder: fused column sequence -> (dx, dy, dyaw).

    Runs over the *concatenated* A|B sequence, because the target (A's pose in
    B's frame) is asymmetric in the pair order and the decoder must be able to
    tell the two apart.
    """

    def __init__(self, d_model: int, width: int = 128, pool: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, width, 5, stride=2, padding=2), nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv1d(width, width, 5, stride=2, padding=2), nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv1d(width, width, 3, stride=2, padding=1), nn.GroupNorm(8, width), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(pool)
        self.head = nn.Sequential(
            nn.Linear(width * pool, 256),
            nn.GELU(),
            nn.Linear(256, 4),  # (dx, dy, cos_yaw, sin_yaw)
        )
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            # start at identity rotation, zero translation
            self.head[-1].bias.copy_(torch.tensor([0.0, 0.0, 1.0, 0.0]))

    def forward(self, seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        # seq: (B, 2*n_cols, d_model)
        x = self.conv(seq.transpose(1, 2))
        x = self.pool(x).flatten(1)
        out = self.head(x)
        return {"t": out[:, :2], "yaw_vec": out[:, 2:4]}


class AttnPoseDecoder(nn.Module):
    """Learned queries cross-attend to the fused A|B sequence -> (dx, dy, dyaw).

    The conv decoder above treats the concatenated sequence as one open 1D
    signal, so its receptive fields straddle the artificial A|B junction and its
    zero padding breaks the 360 deg wraparound of both panoramas. Attention
    pooling has neither problem: every query sees every column of both panos,
    and which pano a token came from is carried by the pano-id embedding that
    was added before the transformer. The queries are concatenated in a fixed
    order, so the pair order (A in B's frame) stays distinguishable.
    """

    def __init__(self, d_model: int, n_heads: int = 8, n_queries: int = 4,
                 n_layers: int = 2, dim_ff: int = 1024, hidden: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "norm_q": nn.LayerNorm(d_model),
                "norm_kv": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "norm_ff": nn.LayerNorm(d_model),
                "ff": nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(),
                                    nn.Linear(dim_ff, d_model)),
            }))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * n_queries, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),  # (dx, dy, cos_yaw, sin_yaw)
        )
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.tensor([0.0, 0.0, 1.0, 0.0]))

    def forward(self, seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        q = self.queries.unsqueeze(0).expand(seq.shape[0], -1, -1).to(seq.dtype)
        for blk in self.blocks:
            kv = blk["norm_kv"](seq)
            q = q + blk["attn"](blk["norm_q"](q), kv, kv, need_weights=False)[0]
            q = q + blk["ff"](blk["norm_ff"](q))
        out = self.head(self.norm(q).flatten(1))
        return {"t": out[:, :2], "yaw_vec": out[:, 2:4]}


# --------------------------------------------------------------------------- #
# full model
# --------------------------------------------------------------------------- #
@dataclass
class CoVisPoseConfig:
    backbone: str = "resnet34"       # RTX1070: paper-scale would be ResNet-50+
    pretrained: bool = True
    img_h: int = 512
    img_w: int = 1024
    n_cols: int = 256                # RTX1070: W/4, the HorizonNet step_cols=4 grid
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dim_ff: int = 1024
    dropout: float = 0.1
    out_scale: int = 8
    pose_width: int = 128
    head_hidden: int = 128
    # ---- server-scale options; the defaults reproduce the RTX 1070 model ---- #
    weights: Optional[str] = None     # torchvision weight tag, e.g. IMAGENET1K_V2
    drop_path: float = 0.0            # stochastic depth in the pair transformer
    backbone_drop_path: float = 0.0   # ConvNeXt only
    pose_decoder: str = "conv"        # "conv" (original) | "attn"
    pose_queries: int = 4
    pose_layers: int = 2
    geo_pose: bool = False            # differentiable Procrustes pose branch
    geo_max_dist: float = 10.0
    geo_irls_iters: int = 2
    geo_irls_c: float = 0.3
    grad_checkpoint: bool = False     # transformer layers only


def config_from_dict(d: Optional[dict], **overrides) -> CoVisPoseConfig:
    """Build a config from a YAML ``model:`` block, ignoring unknown keys."""
    names = {f.name for f in dataclasses.fields(CoVisPoseConfig)}
    kw = {k: v for k, v in (d or {}).items() if k in names}
    kw.update(overrides)
    return CoVisPoseConfig(**kw)


class CoVisPose(nn.Module):
    def __init__(self, cfg: CoVisPoseConfig):
        super().__init__()
        self.cfg = cfg
        self.column_encoder = ColumnEncoder(
            cfg.backbone, cfg.pretrained, cfg.out_scale,
            img_h=cfg.img_h, img_w=cfg.img_w, n_cols=cfg.n_cols,
            weights=cfg.weights, backbone_drop_path=cfg.backbone_drop_path,
        )

        self.proj = nn.Sequential(
            nn.Linear(self.column_encoder.out_channels, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        # which of the two panoramas a token came from
        self.pano_emb = nn.Parameter(torch.zeros(2, cfg.d_model))
        nn.init.normal_(self.pano_emb, std=0.02)
        self.register_buffer(
            "pos_enc", azimuth_encoding(cfg.n_cols, cfg.d_model), persistent=False
        )

        self.transformer = PairTransformer(
            cfg.d_model, cfg.n_heads, cfg.dim_ff, cfg.dropout, cfg.n_layers,
            drop_path=cfg.drop_path, grad_checkpoint=cfg.grad_checkpoint,
        )
        self.norm = nn.LayerNorm(cfg.d_model)

        self.heads = ColumnHeads(cfg.d_model, cfg.head_hidden)
        if cfg.pose_decoder == "conv":
            self.pose_decoder = PoseDecoder(cfg.d_model, cfg.pose_width)
        elif cfg.pose_decoder == "attn":
            self.pose_decoder = AttnPoseDecoder(
                cfg.d_model, cfg.n_heads, cfg.pose_queries, cfg.pose_layers,
                dim_ff=cfg.dim_ff, hidden=2 * cfg.head_hidden,
            )
        else:
            raise ValueError(f"unknown pose_decoder {cfg.pose_decoder!r}")

    # -- pieces, exposed so tests can poke at them ------------------------- #
    def tokens(self, cols: torch.Tensor, which: int) -> torch.Tensor:
        tok = self.proj(cols.transpose(1, 2))                  # (B, n_cols, d)
        return tok + self.pos_enc.to(tok.dtype) + self.pano_emb[which].to(tok.dtype)

    def encode(self, img: torch.Tensor, which: int) -> torch.Tensor:
        return self.tokens(self.column_encoder(img, self.cfg.n_cols), which)

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> Dict[str, torch.Tensor]:
        n, b = self.cfg.n_cols, img_a.shape[0]
        # One backbone call over both panoramas: a single 2B batch is faster
        # than two B batches, and the weights are shared anyway.
        cols = self.column_encoder(torch.cat([img_a, img_b], dim=0), n)
        seq = torch.cat([self.tokens(cols[:b], 0), self.tokens(cols[b:], 1)], dim=1)
        fused = self.norm(self.transformer(seq))

        tok_a, tok_b = fused[:, :n], fused[:, n:]
        out_a = self.heads(tok_a)
        out_b = self.heads(tok_b)
        pose = self.pose_decoder(fused)

        out = {
            "covis_logit_a": out_a["covis_logit"], "covis_logit_b": out_b["covis_logit"],
            "corr_vec_a": out_a["corr_vec"], "corr_vec_b": out_b["corr_vec"],
            "bon_a": out_a["bon"], "bon_b": out_b["bon"],
            "t": pose["t"], "yaw_vec": pose["yaw_vec"],
        }
        if self.cfg.geo_pose:
            geo = geometric_pose(out, self.cfg.geo_max_dist, self.cfg.geo_irls_iters,
                                 self.cfg.geo_irls_c)
            out.update({"geo_t": geo["t"], "geo_yaw_vec": geo["yaw_vec"],
                        "geo_weight": geo["weight_sum"]})
        return out

    # -- convenience -------------------------------------------------------- #
    @staticmethod
    def decode_angle(vec: torch.Tensor) -> torch.Tensor:
        """(..., 2) unnormalised (cos, sin) -> angle in [-pi, pi)."""
        return torch.atan2(vec[..., 1], vec[..., 0])

    def param_groups(self, lr: float, backbone_lr_mult: float = 0.1,
                     weight_decay: Optional[float] = None):
        """Backbone gets a lower LR since it starts from ImageNet weights.

        With ``weight_decay`` given, norms, biases, embeddings and the pose
        queries are split into their own zero-decay groups (the standard AdamW
        recipe); without it the original two groups are returned unchanged, so
        the optimizer state of older checkpoints still lines up.
        """
        # (is_backbone, no_decay) -> params, in a fixed backbone-first order
        keys = [(True, False), (True, True), (False, False), (False, True)]
        groups = {k: [] for k in keys}
        for n_, p in self.named_parameters():
            if not p.requires_grad:
                continue
            bb = n_.startswith("column_encoder.encoder.")
            no_decay = weight_decay is not None and (
                p.ndim <= 1 or n_.endswith("pano_emb") or n_.endswith("queries"))
            groups[(bb, no_decay)].append(p)
        out = []
        for bb, no_decay in keys:
            if weight_decay is None and no_decay:
                continue
            g = {"params": groups[(bb, no_decay)], "lr": lr * (backbone_lr_mult if bb else 1.0)}
            if weight_decay is not None:
                g["weight_decay"] = 0.0 if no_decay else weight_decay
            out.append(g)
        return out


def build_model(cfg: Optional[CoVisPoseConfig] = None, **kw) -> CoVisPose:
    if cfg is None:
        cfg = CoVisPoseConfig(**kw)
    return CoVisPose(cfg)


def count_params(m: nn.Module) -> Dict[str, float]:
    def n(mod):
        return sum(p.numel() for p in mod.parameters()) / 1e6

    return {
        "total_M": n(m),
        "backbone_M": n(m.column_encoder.encoder),
        "height_compress_M": n(m.column_encoder.ghc),
        "transformer_M": n(m.transformer),
        "heads_M": n(m.heads),
        "pose_decoder_M": n(m.pose_decoder),
    }
