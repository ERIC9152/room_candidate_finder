"""Torch Dataset over the pano-pair index.

Each item is one pair: two images plus the per-column targets built on the fly
by :func:`covispose.targets.compute_pair_targets` (a couple of ray casts, well
under a millisecond, so there is nothing to precompute).

Augmentation notes
------------------
* **Horizontal roll** is the natural augmentation for equirectangular panoramas
  and is free: rolling pano A by ``d`` columns rotates its local frame by
  ``-theta_d``, which we apply exactly to the pose and correspondence targets
  rather than approximating. Rolls are in whole *output* columns so the targets
  stay exact.
* **Pair order swap** turns (A, B) into (B, A) with the inverse pose.
* **Horizontal mirror**, applied to *both* panoramas together, reflects the
  whole scene through the camera's y-axis. A reflected room is still a valid
  room and a reflected rigid motion is still rigid: azimuths negate, columns
  reverse, ``t -> (-tx, ty)`` and ``yaw -> -yaw`` (``M Rot(a) M = Rot(-a)`` for
  ``M = diag(-1, 1)``). Mirroring only one pano would not be a rigid transform.
* Photometric jitter (gamma) touches images only.

No vertical flip and no pano stretch: both would change the floor-wall geometry
in ways the pose target would have to follow, and stretch in particular is not
a rigid transform of the floor plane.

The geometric augmentations live in :func:`augment_targets` so the test-time
augmentation in ``evaluate.py`` can be checked against exactly the same
bookkeeping (``tests/test_model.py::test_tta_inverse_matches_dataset``).
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.utils.data as tud
from PIL import Image

from .geometry import wrap_pi
from .pairs import PairRef, read_pair_index
from .targets import compute_pair_targets
from .zind_parse import HouseRecord, load_house


def _load_image(path: str, h: int, w: int, fast_jpeg: bool = False) -> np.ndarray:
    with Image.open(path) as im:
        if fast_jpeg:
            # Let libjpeg decode straight at 1/2, 1/4 or 1/8 scale in the DCT
            # domain. ZInD panos are 2048x1024, so at 1024x512 this skips 3/4 of
            # the decode work -- JPEG decode is what bounds loader throughput.
            im.draft("RGB", (w, h))
        im = im.convert("RGB")
        if im.size != (w, h):
            im = im.resize((w, h), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def augment_targets(d: Dict[str, np.ndarray], n_cols: int, flip: bool,
                    da: int, db: int) -> Dict[str, np.ndarray]:
    """Mirror both panos (optional), then roll A by ``da`` and B by ``db`` columns.

    ``d`` holds the per-column arrays ``{bon,val,cov,corr}_{a,b}`` plus ``t``
    (2,) and ``yaw``; images are handled by the caller. Returns a new dict.
    """
    d = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    if flip:
        for s in ("a", "b"):
            for k in ("bon", "val", "cov", "corr"):
                d[f"{k}_{s}"] = d[f"{k}_{s}"][::-1].copy()
            d[f"corr_{s}"] = wrap_pi(-d[f"corr_{s}"]).astype(np.float32)
        d["t"] = np.array([-d["t"][0], d["t"][1]])
        d["yaw"] = float(wrap_pi(-d["yaw"]))

    for s, dc in (("a", da), ("b", db)):
        for k in ("bon", "val", "cov", "corr"):
            d[f"{k}_{s}"] = np.roll(d[f"{k}_{s}"], dc)
    # Rolling pano X right by d columns rotates its local frame by -phi_d.
    phi_a = 2 * math.pi * da / n_cols
    phi_b = 2 * math.pi * db / n_cols
    # correspondence values live in the *other* pano's frame
    d["corr_a"] = wrap_pi(d["corr_a"] + phi_b).astype(np.float32)
    d["corr_b"] = wrap_pi(d["corr_b"] + phi_a).astype(np.float32)
    # pose: p_b = Rot(yaw) p_a + t, expressed in the rolled frames
    c, s_ = math.cos(phi_b), math.sin(phi_b)
    t = d["t"]
    d["t"] = np.array([c * t[0] - s_ * t[1], s_ * t[0] + c * t[1]])
    d["yaw"] = float(wrap_pi(d["yaw"] + phi_b - phi_a))
    return d


class PairDataset(tud.Dataset):
    def __init__(
        self,
        pair_index: str,
        data_root: str,
        img_h: int = 512,
        img_w: int = 1024,
        n_cols: int = 256,
        layout_key: str = "layout_visible",
        augment: bool = False,
        gamma_aug: bool = True,
        max_pairs: int = 0,
        subsample_n: int = 0,
        subsample_seed: int = 0,
        house_cache_size: int = 64,
        seed: int = 0,
        flip_prob: float = 0.0,
        fast_jpeg: bool = False,
        shard: Optional[tuple] = None,
    ):
        assert img_w % n_cols == 0, (
            f"img_w={img_w} must be a multiple of n_cols={n_cols} so that a roll "
            "augmentation of whole output columns is also a whole number of pixels"
        )
        self.pairs: List[PairRef] = read_pair_index(pair_index)
        if subsample_n > 0 and subsample_n < len(self.pairs):
            # Random, not the first N: the index is ordered by house, so a head
            # slice would evaluate on a handful of homes and tell us nothing
            # about generalisation across the split.
            idx = np.random.default_rng(subsample_seed).choice(
                len(self.pairs), subsample_n, replace=False
            )
            self.pairs = [self.pairs[i] for i in sorted(idx)]
        if max_pairs > 0:
            self.pairs = self.pairs[:max_pairs]
        if shard is not None:
            # (rank, world_size): disjoint, unpadded slices for distributed
            # evaluation -- every pair is scored exactly once across ranks.
            rank, world = shard
            self.pairs = self.pairs[rank::world]
        self.flip_prob = flip_prob
        self.fast_jpeg = fast_jpeg
        self.data_root = data_root
        self.img_h, self.img_w, self.n_cols = img_h, img_w, n_cols
        self.layout_key = layout_key
        self.augment = augment
        self.gamma_aug = gamma_aug
        self.seed = seed
        self._cache: Dict[str, HouseRecord] = {}
        self._cache_order: List[str] = []
        self._cache_size = house_cache_size

    def __len__(self) -> int:
        return len(self.pairs)

    def _house(self, house_id: str) -> HouseRecord:
        if house_id not in self._cache:
            self._cache[house_id] = load_house(
                self.data_root, house_id, layout_key=self.layout_key,
                primary_only=False, inside_only=True,
            )
            self._cache_order.append(house_id)
            while len(self._cache_order) > self._cache_size:
                self._cache.pop(self._cache_order.pop(0), None)
        return self._cache[house_id]

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p = self.pairs[idx]
        house = self._house(p.house_id)
        rec_a, rec_b = house.panos[p.key_a], house.panos[p.key_b]

        if self.augment and np.random.rand() < 0.5:
            rec_a, rec_b = rec_b, rec_a

        tg = compute_pair_targets(rec_a, rec_b, n_cols=self.n_cols)

        img_a = _load_image(rec_a.image_path, self.img_h, self.img_w, self.fast_jpeg)
        img_b = _load_image(rec_b.image_path, self.img_h, self.img_w, self.fast_jpeg)

        d = {
            "bon_a": tg.bon_a, "bon_b": tg.bon_b,
            "val_a": tg.bon_valid_a, "val_b": tg.bon_valid_b,
            "cov_a": tg.covis_a, "cov_b": tg.covis_b,
            "corr_a": tg.corr_a2b, "corr_b": tg.corr_b2a,
            "t": tg.t_ab.copy(), "yaw": tg.yaw_ab,
        }

        if self.augment:
            flip = self.flip_prob > 0 and np.random.rand() < self.flip_prob
            # roll each pano independently, in whole output columns
            da = int(np.random.randint(self.n_cols))
            db = int(np.random.randint(self.n_cols))
            d = augment_targets(d, self.n_cols, flip, da, db)
            px = self.img_w // self.n_cols
            if flip:
                img_a, img_b = img_a[:, ::-1], img_b[:, ::-1]
            img_a = np.roll(img_a, da * px, axis=1)
            img_b = np.roll(img_b, db * px, axis=1)

            if self.gamma_aug:
                img_a = self._gamma(img_a)
                img_b = self._gamma(img_b)
        else:
            d = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}

        bon_a, bon_b, val_a, val_b = d["bon_a"], d["bon_b"], d["val_a"], d["val_b"]
        cov_a, cov_b, corr_a, corr_b = d["cov_a"], d["cov_b"], d["corr_a"], d["corr_b"]
        t, yaw = d["t"], d["yaw"]

        to_img = lambda a: torch.from_numpy(  # noqa: E731
            np.ascontiguousarray(a.transpose(2, 0, 1))
        ).float().div_(255.0)

        return {
            "img_a": to_img(img_a),
            "img_b": to_img(img_b),
            "bon_a": torch.from_numpy(bon_a).float(),
            "bon_b": torch.from_numpy(bon_b).float(),
            "bon_valid_a": torch.from_numpy(val_a).bool(),
            "bon_valid_b": torch.from_numpy(val_b).bool(),
            "covis_a": torch.from_numpy(cov_a).float(),
            "covis_b": torch.from_numpy(cov_b).float(),
            "corr_a": torch.from_numpy(corr_a).float(),
            "corr_b": torch.from_numpy(corr_b).float(),
            "t": torch.from_numpy(np.asarray(t, dtype=np.float32)),
            "yaw": torch.tensor(float(yaw), dtype=torch.float32),
            "covis_score": torch.tensor(p.covis_score, dtype=torch.float32),
            "baseline": torch.tensor(p.baseline, dtype=torch.float32),
            # Negatives (two rooms that share nothing) have an exact pose that no
            # image evidence can recover; it is not supervised (A19).
            "pose_valid": torch.tensor(p.kind != "neg"),
            "is_neg": torch.tensor(p.kind == "neg"),
            "idx": torch.tensor(idx, dtype=torch.long),
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _gamma(img: np.ndarray) -> np.ndarray:
        g = np.random.uniform(1.0, 2.0)
        if np.random.randint(2) == 0:
            g = 1.0 / g
        lut = (np.linspace(0, 1, 256) ** g * 255.0).astype(np.uint8)
        return lut[img]


def _worker_init(worker_id: int) -> None:
    """Give every worker its own numpy stream.

    PyTorch reseeds ``random`` and ``torch`` per worker but **not** ``numpy``,
    so without this every worker would draw the identical sequence of roll
    offsets and gammas -- the augmentation would silently lose most of its
    variety.  ``initial_seed()`` already differs per worker and per epoch.
    """
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)


def make_loader(
    pair_index: str,
    data_root: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
    augment: bool = False,
    drop_last: bool = False,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    distributed: bool = False,
    seed: int = 0,
    **ds_kw,
) -> tud.DataLoader:
    """Build a DataLoader over the pair index.

    ``pin_memory`` is worth it for training, where the GPU is the bottleneck,
    but page-locked memory is a scarce resource: at 512x1024 each batch of 8
    pairs is ~100 MB, and 4 workers prefetching 2 batches each keeps ~800 MB
    pinned. Over a long evaluation pass that allocation fails with
    ``RuntimeError: CUDA error: out of memory`` raised from the pin-memory
    thread -- which looks like a GPU OOM but is not one. Pass
    ``pin_memory=False`` for evaluation, where the throughput does not matter.
    """
    ds = PairDataset(pair_index, data_root, augment=augment, **ds_kw)
    kw = {}
    if num_workers > 0:
        kw["prefetch_factor"] = prefetch_factor
        kw["persistent_workers"] = True
        kw["worker_init_fn"] = _worker_init
    if distributed:
        # Training only: each rank draws a disjoint shuffled slice. Call
        # ``loader.sampler.set_epoch(e)`` every epoch or every epoch repeats.
        kw["sampler"] = tud.distributed.DistributedSampler(
            ds, shuffle=shuffle, drop_last=drop_last, seed=seed)
        shuffle = False
    return tud.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **kw,
    )
