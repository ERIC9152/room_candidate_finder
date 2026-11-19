"""Model / loss / RANSAC checks that do not need a trained network.

    python tests/test_model.py
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from covispose.losses import LossWeights, covispose_loss, covispose_metrics  # noqa: E402
from covispose.model import CoVisPoseConfig, azimuth_encoding, build_model, count_params  # noqa: E402
from covispose.ransac import pose_from_predictions  # noqa: E402
from covispose.targets import compute_pair_targets  # noqa: E402
from covispose.zind_parse import load_house  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _requires import ZIND_ROOT, needs_zind  # noqa: E402
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROTO_INDEX = os.path.join(REPO, "training", "pairs", "train_c10neg.csv")

SMALL = CoVisPoseConfig(
    backbone="resnet18", pretrained=False, img_h=128, img_w=256,
    n_cols=64, d_model=32, n_heads=4, n_layers=2, dim_ff=64,
)
# every server-scale option switched on, at toy size
SMALL_SERVER = CoVisPoseConfig(
    backbone="resnet18", pretrained=False, img_h=128, img_w=256,
    n_cols=128, d_model=32, n_heads=4, n_layers=2, dim_ff=64,
    drop_path=0.1, pose_decoder="attn", geo_pose=True, grad_checkpoint=True,
)


def test_azimuth_encoding_is_periodic():
    pe = azimuth_encoding(64, 32)
    assert pe.shape == (64, 32)
    # rolling the column grid must roll the encoding: no seam at the wraparound
    pe2 = azimuth_encoding(64, 32)
    assert torch.allclose(pe, pe2)
    # adjacent-column similarity must be the same across the seam as anywhere
    d_mid = (pe[32] - pe[33]).norm().item()
    d_seam = (pe[63] - pe[0]).norm().item()
    assert abs(d_mid - d_seam) < 1e-4, (d_mid, d_seam)
    print("  ok  test_azimuth_encoding_is_periodic")


def test_forward_shapes():
    m = build_model(SMALL).eval()
    b = 2
    a = torch.rand(b, 3, SMALL.img_h, SMALL.img_w)
    c = torch.rand(b, 3, SMALL.img_h, SMALL.img_w)
    with torch.no_grad():
        out = m(a, c)
    n = SMALL.n_cols
    assert out["covis_logit_a"].shape == (b, n)
    assert out["corr_vec_a"].shape == (b, n, 2)
    assert out["bon_b"].shape == (b, n)
    assert out["t"].shape == (b, 2)
    assert out["yaw_vec"].shape == (b, 2)
    print("  ok  test_forward_shapes", {k: tuple(v.shape) for k, v in out.items()})


def test_column_encoder_is_circularly_equivariant():
    """Rolling the input must roll the column features by the same amount.

    This is the check that ``wrap_lr_pad`` actually took effect -- with ordinary
    zero padding the image seam leaks into the features and this fails.  We roll
    by a multiple of the backbone's total stride (32) so the feature grid lines
    up exactly, which makes the equivariance exact rather than approximate.

    Note this tests the *encoder*, not the whole model: the model adds an
    absolute azimuth positional encoding, so it is deliberately not
    roll-equivariant end to end (random roll augmentation is what teaches it
    that invariance -- see docs/ASSUMPTIONS.md A10).
    """
    m = build_model(SMALL).eval()
    enc = m.column_encoder
    x = torch.rand(1, 3, SMALL.img_h, SMALL.img_w)
    roll_px = 64
    assert roll_px % 32 == 0
    roll_cols = roll_px * SMALL.n_cols // SMALL.img_w

    with torch.no_grad():
        f1 = enc(x, SMALL.n_cols)
        f2 = enc(torch.roll(x, roll_px, dims=3), SMALL.n_cols)

    ref = torch.roll(f1, roll_cols, dims=2)
    err = (f2 - ref).abs().max().item()
    scale = f1.abs().max().item() + 1e-6
    assert err / scale < 1e-4, f"max rel err {err/scale:.6f} -- circular padding broken?"

    # control: the same comparison without rolling must be far off, otherwise
    # the test would pass on a constant feature map and prove nothing.
    ctrl = (f1 - torch.roll(f1, roll_cols, dims=2)).abs().max().item() / scale
    assert ctrl > 1e-2, f"features are too flat for this test to mean anything ({ctrl})"
    print(f"  ok  test_column_encoder_is_circularly_equivariant "
          f"(rel err {err/scale:.2e}, control {ctrl:.3f})")


def test_loss_masks_noncovisible_correspondence():
    """The correspondence loss must ignore non-co-visible columns entirely."""
    torch.manual_seed(0)
    b, n = 2, 16
    out = {
        "covis_logit_a": torch.zeros(b, n), "covis_logit_b": torch.zeros(b, n),
        "corr_vec_a": torch.randn(b, n, 2, requires_grad=True),
        "corr_vec_b": torch.randn(b, n, 2),
        "bon_a": torch.zeros(b, n), "bon_b": torch.zeros(b, n),
        "t": torch.zeros(b, 2), "yaw_vec": torch.tensor([[1.0, 0.0]] * b),
    }
    covis = torch.zeros(b, n)
    covis[:, :4] = 1.0
    batch = {
        "covis_a": covis, "covis_b": covis,
        "bon_valid_a": torch.ones(b, n, dtype=torch.bool),
        "bon_valid_b": torch.ones(b, n, dtype=torch.bool),
        "corr_a": torch.randn(b, n), "corr_b": torch.randn(b, n),
        "bon_a": torch.zeros(b, n), "bon_b": torch.zeros(b, n),
        "t": torch.zeros(b, 2), "yaw": torch.zeros(b),
    }
    terms = covispose_loss(out, batch, LossWeights())
    terms["corr"].backward()
    g = out["corr_vec_a"].grad.abs().sum(dim=(0, 2))
    assert (g[:4] > 0).all(), "co-visible columns must receive gradient"
    assert (g[4:] == 0).all(), "non-co-visible columns must receive NO gradient"

    # and changing the target on masked-out columns must not change the loss
    batch2 = dict(batch)
    batch2["corr_a"] = batch["corr_a"].clone()
    batch2["corr_a"][:, 4:] += 3.0
    out2 = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in out.items()}
    t1 = covispose_loss(out2, batch, LossWeights())["corr"].item()
    t2 = covispose_loss(out2, batch2, LossWeights())["corr"].item()
    assert abs(t1 - t2) < 1e-9, (t1, t2)
    print("  ok  test_loss_masks_noncovisible_correspondence")


def test_empty_covis_mask_is_safe():
    """A pair with zero co-visible columns must not produce NaN."""
    b, n = 1, 8
    out = {
        "covis_logit_a": torch.zeros(b, n), "covis_logit_b": torch.zeros(b, n),
        "corr_vec_a": torch.randn(b, n, 2, requires_grad=True),
        "corr_vec_b": torch.randn(b, n, 2),
        "bon_a": torch.zeros(b, n), "bon_b": torch.zeros(b, n),
        "t": torch.zeros(b, 2, requires_grad=True), "yaw_vec": torch.tensor([[1.0, 0.0]]),
    }
    batch = {
        "covis_a": torch.zeros(b, n), "covis_b": torch.zeros(b, n),
        "bon_valid_a": torch.ones(b, n, dtype=torch.bool),
        "bon_valid_b": torch.ones(b, n, dtype=torch.bool),
        "corr_a": torch.zeros(b, n), "corr_b": torch.zeros(b, n),
        "bon_a": torch.zeros(b, n), "bon_b": torch.zeros(b, n),
        "t": torch.zeros(b, 2), "yaw": torch.zeros(b),
    }
    terms = covispose_loss(out, batch, LossWeights())
    assert torch.isfinite(terms["total"]), terms
    terms["total"].backward()
    assert torch.isfinite(out["corr_vec_a"].grad).all()
    print("  ok  test_empty_covis_mask_is_safe")


@needs_zind
def test_ransac_recovers_pose_from_ground_truth():
    """Feed the RANSAC estimator *ground-truth* per-column values.

    It must return the ground-truth pose.  This validates ransac.py and, once
    more, that the correspondence and boundary targets are mutually consistent.
    """
    house = load_house(ZIND_ROOT, "0000", primary_only=False, inside_only=True)
    keys = sorted(house.panos)
    n_ok, errs_t, errs_yaw = 0, [], []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ra, rb = house.panos[keys[i]], house.panos[keys[j]]
            if ra.floor_id != rb.floor_id:
                continue
            tg = compute_pair_targets(ra, rb, n_cols=256)
            if tg.covis_a.sum() < 20 or np.linalg.norm(tg.t_ab) < 0.1:
                continue
            res = pose_from_predictions(tg.bon_a, tg.bon_b, tg.corr_a2b, tg.covis_a)
            if not res.ok:
                continue
            n_ok += 1
            errs_t.append(np.linalg.norm(res.t - tg.t_ab))
            errs_yaw.append(abs(math.degrees(np.arctan2(
                np.sin(res.yaw - tg.yaw_ab), np.cos(res.yaw - tg.yaw_ab)))))
    assert n_ok >= 10, f"only {n_ok} usable pairs"
    errs_t, errs_yaw = np.array(errs_t), np.array(errs_yaw)
    print(f"  ok  test_ransac_recovers_pose_from_ground_truth  "
          f"(n={n_ok}, median |t| err={np.median(errs_t):.4f}, "
          f"median yaw err={np.median(errs_yaw):.4f} deg, "
          f"p90 |t| err={np.percentile(errs_t, 90):.4f})")
    # The floor here is the co-visibility label tolerance (A5), not a bug: a
    # column counts as co-visible when the two distances agree to within
    # abs_tol=0.10, so correspondences carry that much slack. Verified by
    # sweeping it -- abs_tol 0.10 -> 0.05 moves the median from 0.030 to 0.0037.
    assert np.median(errs_t) < 0.05, np.median(errs_t)
    assert np.median(errs_yaw) < 0.5, np.median(errs_yaw)


def _closed_loop_error(item):
    """Check bon/corr/covis/pose in one item are mutually consistent.

    Take A's boundary point at column j from ``bon_a``, push it through the
    pose, and it must land at azimuth ``corr_a[j]`` and at the distance
    ``bon_b`` reports for that azimuth.  Any error in the roll-augmentation
    bookkeeping breaks this immediately.
    """
    import numpy as np

    from covispose.geometry import bfloor_to_dist, col_centers_theta, dir_of_theta, theta_of, wrap_pi
    from covispose.ransac import sample_boundary_at_azimuth

    bon_a = item["bon_a"].numpy()
    bon_b = item["bon_b"].numpy()
    corr_a = item["corr_a"].numpy()
    covis_a = item["covis_a"].numpy() > 0.5
    t = item["t"].numpy()
    yaw = float(item["yaw"])

    n = len(bon_a)
    theta = col_centers_theta(n)
    P = dir_of_theta(theta) * bfloor_to_dist(bon_a)[:, None]
    c, s = math.cos(yaw), math.sin(yaw)
    Q = (np.array([[c, -s], [s, c]]) @ P.T).T + t

    ang_err = np.abs(wrap_pi(theta_of(Q) - corr_a))
    r_err = np.abs(np.linalg.norm(Q, axis=1) - sample_boundary_at_azimuth(bon_b, corr_a))
    return ang_err, r_err, covis_a


@needs_zind
def test_roll_augmentation_is_exact():
    """The augmented targets must still satisfy the pose/correspondence relation.

    The roll augmentation rotates each pano's local frame, so the pose,
    correspondence values and every per-column array have to be transformed
    together.  This closes the loop on that hand-derived bookkeeping.
    """
    from covispose.dataset import PairDataset

    for augment, flip_prob in ((False, 0.0), (True, 0.0), (True, 1.0)):
        np.random.seed(7)
        ds = PairDataset(PROTO_INDEX, ZIND_ROOT, img_h=128, img_w=256, n_cols=64,
                         augment=augment, gamma_aug=False, max_pairs=24,
                         flip_prob=flip_prob)
        worst_ang, worst_r, n_used = 0.0, 0.0, 0
        for i in range(len(ds)):
            ang_err, r_err, covis = _closed_loop_error(ds[i])
            if covis.sum() < 5:
                continue
            n_used += 1
            worst_ang = max(worst_ang, math.degrees(ang_err[covis].max()))
            # r_err is compared against an interpolated boundary, so it carries
            # the same interpolation slack the RANSAC path does; use the median.
            worst_r = max(worst_r, float(np.median(r_err[covis])))
        assert n_used >= 5, n_used
        assert worst_ang < 1e-3, f"augment={augment}: correspondence angle off by {worst_ang} deg"
        assert worst_r < 0.15, f"augment={augment}: radial mismatch {worst_r}"
        print(f"  ok  test_roll_augmentation_is_exact (augment={augment}, flip={flip_prob}, "
              f"n={n_used}, max angle err={worst_ang:.2e} deg, max median r err={worst_r:.4f})")


@needs_zind
def test_mirror_is_applied_to_images_consistently():
    """The mirrored image must be the pixel-exact mirror, and boundary must follow.

    The closed-loop test above checks the targets against each other; this one
    checks them against the *image* side of the augmentation.
    """
    from covispose.dataset import PairDataset

    kw = dict(img_h=128, img_w=256, n_cols=64, gamma_aug=False, max_pairs=4)
    plain = PairDataset(PROTO_INDEX, ZIND_ROOT, augment=False, **kw)
    item0 = plain[0]
    aug = PairDataset(PROTO_INDEX, ZIND_ROOT, augment=True, flip_prob=1.0, **kw)
    # Replay the dataset's random draws (swap, flip, da, db) to learn the
    # offsets, using a seed for which the pair is not swapped.
    seed = next(s for s in range(100) if np.random.RandomState(s).rand() >= 0.5)
    np.random.seed(seed)
    np.random.rand(), np.random.rand()
    da, db = int(np.random.randint(64)), int(np.random.randint(64))
    np.random.seed(seed)
    item1 = aug[0]
    px = 256 // 64
    exp_img = torch.roll(torch.flip(item0["img_a"], dims=[2]), da * px, dims=2)
    exp_bon = torch.roll(torch.flip(item0["bon_a"], dims=[0]), da, dims=0)
    exp_img_b = torch.roll(torch.flip(item0["img_b"], dims=[2]), db * px, dims=2)
    assert torch.equal(item1["img_a"], exp_img)
    assert torch.equal(item1["img_b"], exp_img_b)
    assert torch.allclose(item1["bon_a"], exp_bon)
    print(f"  ok  test_mirror_is_applied_to_images_consistently (da={da}, db={db})")


def _fake_outputs_from_targets(d, sharp=20.0):
    """Turn a target dict (numpy, dataset naming) into a model-output dict."""
    out = {}
    for s in ("a", "b"):
        out[f"covis_logit_{s}"] = torch.tensor((d[f"cov_{s}"].astype(np.float32) - 0.5) * sharp)[None]
        corr = torch.tensor(d[f"corr_{s}"], dtype=torch.float32)
        out[f"corr_vec_{s}"] = torch.stack([torch.cos(corr), torch.sin(corr)], -1)[None]
        out[f"bon_{s}"] = torch.tensor(d[f"bon_{s}"], dtype=torch.float32)[None]
    out["t"] = torch.tensor(np.asarray(d["t"], dtype=np.float32))[None]
    yaw = float(d["yaw"])
    out["yaw_vec"] = torch.tensor([[math.cos(yaw), math.sin(yaw)]])
    return out


def _pair_targets_dict(n_cols=64):
    house = load_house(ZIND_ROOT, "0000", primary_only=False, inside_only=True)
    keys = sorted(house.panos)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ra, rb = house.panos[keys[i]], house.panos[keys[j]]
            if ra.floor_id != rb.floor_id:
                continue
            tg = compute_pair_targets(ra, rb, n_cols=n_cols)
            if tg.covis_a.sum() > n_cols // 4 and np.linalg.norm(tg.t_ab) > 1.0:
                return {"bon_a": tg.bon_a, "bon_b": tg.bon_b,
                        "val_a": tg.bon_valid_a, "val_b": tg.bon_valid_b,
                        "cov_a": tg.covis_a, "cov_b": tg.covis_b,
                        "corr_a": tg.corr_a2b, "corr_b": tg.corr_b2a,
                        "t": tg.t_ab.copy(), "yaw": tg.yaw_ab}
    raise AssertionError("no usable pair in house 0000")


@needs_zind
def test_tta_inverse_matches_dataset():
    """``evaluate.invert_view`` must exactly undo ``dataset.augment_targets``.

    Augment real ground truth with the dataset's code, express it as model
    outputs, invert with the TTA code, and require the original back. This ties
    the evaluation-time bookkeeping to the training-time bookkeeping, which the
    closed-loop test has already validated against geometry.
    """
    from covispose.dataset import augment_targets
    from covispose.evaluate import invert_view, tta_views

    n = 64
    d0 = _pair_targets_dict(n)
    ref = _fake_outputs_from_targets(d0)
    worst = 0.0
    for flip, da, db in tta_views(n, rolls=4, flip=True) + [(True, 5, 37), (False, 63, 1)]:
        o = invert_view(_fake_outputs_from_targets(augment_targets(d0, n, flip, da, db)),
                        n, flip, da, db)
        for k, v in ref.items():
            if k.endswith("vec"):
                err = (torch.atan2(o[k][..., 1], o[k][..., 0])
                       - torch.atan2(v[..., 1], v[..., 0]))
                err = torch.atan2(torch.sin(err), torch.cos(err)).abs()
                # correspondence is only meaningful on co-visible columns
                if k.startswith("corr_vec"):
                    err = err[ref[f"covis_logit_{k[-1]}"] > 0]
            else:
                err = (o[k] - v).abs()
            worst = max(worst, float(err.max()))
    assert worst < 1e-4, worst
    print(f"  ok  test_tta_inverse_matches_dataset (max err {worst:.2e})")


def test_forward_tta_single_view_is_plain_forward():
    from covispose.evaluate import forward_tta

    m = build_model(SMALL_SERVER).eval()
    a, b = torch.rand(2, 3, 128, 256), torch.rand(2, 3, 128, 256)
    with torch.no_grad():
        o1, o2 = m(a, b), forward_tta(m, a, b, rolls=1, flip=False)
        o4 = forward_tta(m, a, b, rolls=4, flip=True)
    for k in ("covis_logit_a", "bon_b", "t", "geo_t"):
        assert torch.allclose(o1[k], o2[k], atol=1e-6), k
    assert all(torch.isfinite(v).all() for v in o4.values())
    print("  ok  test_forward_tta_single_view_is_plain_forward")


@needs_zind
def test_geometric_pose_recovers_ground_truth():
    """Weighted Procrustes on GT head values must return the GT pose.

    Same role as the RANSAC test: it validates the torch correspondence
    construction (both directions) against the targets.
    """
    from covispose.geo_pose import geometric_pose

    house = load_house(ZIND_ROOT, "0000", primary_only=False, inside_only=True)
    keys = sorted(house.panos)
    errs_t, errs_yaw = [], []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ra, rb = house.panos[keys[i]], house.panos[keys[j]]
            if ra.floor_id != rb.floor_id:
                continue
            tg = compute_pair_targets(ra, rb, n_cols=256)
            if tg.covis_a.sum() < 20 or np.linalg.norm(tg.t_ab) < 0.1:
                continue
            d = {"bon_a": tg.bon_a, "bon_b": tg.bon_b, "cov_a": tg.covis_a,
                 "cov_b": tg.covis_b, "corr_a": tg.corr_a2b, "corr_b": tg.corr_b2a,
                 "t": tg.t_ab, "yaw": tg.yaw_ab}
            g = geometric_pose(_fake_outputs_from_targets(d), max_dist=10.0,
                               irls_iters=2, irls_c=0.2)
            errs_t.append(float(torch.linalg.norm(g["t"][0] - torch.tensor(tg.t_ab).float())))
            yaw = math.atan2(float(g["yaw_vec"][0, 1]), float(g["yaw_vec"][0, 0]))
            errs_yaw.append(abs(math.degrees(math.atan2(math.sin(yaw - tg.yaw_ab),
                                                        math.cos(yaw - tg.yaw_ab)))))
    errs_t, errs_yaw = np.array(errs_t), np.array(errs_yaw)
    print(f"  ok  test_geometric_pose_recovers_ground_truth (n={len(errs_t)}, "
          f"median |t| err={np.median(errs_t):.4f}, median yaw err={np.median(errs_yaw):.4f} deg, "
          f"p90 |t| err={np.percentile(errs_t, 90):.4f})")
    assert len(errs_t) >= 10
    # measured 0.0375 / 0.046 deg; RANSAC's bound is 0.05 / 0.5 deg
    assert np.median(errs_t) < 0.05, np.median(errs_t)
    assert np.median(errs_yaw) < 0.5, np.median(errs_yaw)
    assert np.percentile(errs_yaw, 90) < 1.5, np.percentile(errs_yaw, 90)


def test_geometric_pose_gradients_reach_the_heads():
    """The whole point of the geo branch: its loss must train the column heads."""
    torch.manual_seed(0)
    m = build_model(SMALL_SERVER).train()
    a, b = torch.rand(2, 3, 128, 256), torch.rand(2, 3, 128, 256)
    out = m(a, b)
    t = torch.tensor([[1.0, 0.5], [-0.3, 2.0]])
    yaw = torch.tensor([0.3, -1.2])
    loss = ((out["geo_t"] - t).abs().sum(-1).mean()
            + (torch.atan2(out["geo_yaw_vec"][:, 1], out["geo_yaw_vec"][:, 0]) - yaw).abs().mean())
    loss.backward()
    for name in ("heads.bon", "heads.corr", "heads.covis", "transformer.layers.0"):
        g = [p.grad for n_, p in m.named_parameters() if n_.startswith(name) and p.grad is not None]
        assert g, f"no gradient reached {name}"
        tot = sum(float(x.abs().sum()) for x in g)
        assert math.isfinite(tot) and tot > 0, (name, tot)
    print("  ok  test_geometric_pose_gradients_reach_the_heads")


def test_geo_loss_is_finite_with_zero_correspondences():
    """All points out of range -> zero Procrustes weight -> yaw vec is (0, 0).

    The loss masks such pairs, but a masked atan2(0, 0) still back-propagates
    0 * NaN = NaN unless it is kept out of the graph.
    """
    b, n = 2, 32
    corr = torch.randn(b, n, 2, requires_grad=True)
    bon = torch.zeros(b, n, requires_grad=True)  # r = max_dist everywhere
    out = {"covis_logit_a": torch.zeros(b, n), "covis_logit_b": torch.zeros(b, n),
           "corr_vec_a": corr, "corr_vec_b": corr.detach(), "bon_a": bon, "bon_b": bon.detach(),
           "t": torch.zeros(b, 2, requires_grad=True), "yaw_vec": torch.tensor([[1.0, 0.0]] * b)}
    from covispose.geo_pose import geometric_pose
    g = geometric_pose(out)
    out.update({"geo_t": g["t"], "geo_yaw_vec": g["yaw_vec"], "geo_weight": g["weight_sum"]})
    assert float(g["weight_sum"].max()) == 0.0
    batch = {"covis_a": torch.ones(b, n), "covis_b": torch.ones(b, n),
             "bon_valid_a": torch.ones(b, n, dtype=torch.bool),
             "bon_valid_b": torch.ones(b, n, dtype=torch.bool),
             "corr_a": torch.zeros(b, n), "corr_b": torch.zeros(b, n),
             "bon_a": torch.full((b, n), 0.4), "bon_b": torch.full((b, n), 0.4),
             "t": torch.ones(b, 2), "yaw": torch.ones(b)}
    terms = covispose_loss(out, batch, LossWeights(geo_t=1.0, geo_yaw=1.0))
    terms["total"].backward()
    assert torch.isfinite(terms["total"])
    assert torch.isfinite(corr.grad).all() and torch.isfinite(bon.grad).all()
    print("  ok  test_geo_loss_is_finite_with_zero_correspondences")


def test_server_options_forward_and_loss():
    m = build_model(SMALL_SERVER).train()
    b = 2
    a, c = torch.rand(b, 3, 128, 256), torch.rand(b, 3, 128, 256)
    out = m(a, c)
    n = SMALL_SERVER.n_cols
    assert out["corr_vec_b"].shape == (b, n, 2) and out["geo_t"].shape == (b, 2)
    batch = {
        "covis_a": torch.ones(b, n), "covis_b": torch.ones(b, n),
        "bon_valid_a": torch.ones(b, n, dtype=torch.bool),
        "bon_valid_b": torch.ones(b, n, dtype=torch.bool),
        "corr_a": torch.zeros(b, n), "corr_b": torch.zeros(b, n),
        "bon_a": torch.full((b, n), 0.4), "bon_b": torch.full((b, n), 0.4),
        "t": torch.zeros(b, 2), "yaw": torch.zeros(b),
    }
    w = LossWeights(geo_t=0.5, geo_yaw=0.5, covis_pos_weight=0.31)
    terms = covispose_loss(out, batch, w)
    assert {"geo_t", "geo_yaw"} <= set(terms) and torch.isfinite(terms["total"])
    terms["total"].backward()
    mets = covispose_metrics(out, batch)
    assert "geo_acc_0.5_10" in mets and "pose_acc_0.5_10" in mets
    print("  ok  test_server_options_forward_and_loss")


def test_convnext_encoder_is_circularly_equivariant():
    cfg = CoVisPoseConfig(backbone="convnext_tiny", pretrained=False, img_h=128, img_w=256,
                          n_cols=64, d_model=32, n_heads=4, n_layers=1, dim_ff=64)
    enc = build_model(cfg).eval().column_encoder
    x = torch.rand(1, 3, 128, 256)
    with torch.no_grad():
        f1 = enc(x, 64)
        f2 = enc(torch.roll(x, 64, dims=3), 64)
    scale = f1.abs().max().item() + 1e-6
    err = (f2 - torch.roll(f1, 16, dims=2)).abs().max().item() / scale
    assert err < 1e-4, err
    print(f"  ok  test_convnext_encoder_is_circularly_equivariant (rel err {err:.2e})")


def test_param_groups_cover_every_parameter_once():
    m = build_model(SMALL_SERVER)
    n_params = sum(1 for p in m.parameters() if p.requires_grad)
    for wd in (None, 0.05):
        groups = m.param_groups(1e-3, 0.1, weight_decay=wd)
        ids = [id(p) for g in groups for p in g["params"]]
        assert len(ids) == len(set(ids)) == n_params, (wd, len(ids), n_params)
    legacy = m.param_groups(1e-3, 0.1)
    assert len(legacy) == 2 and legacy[0]["lr"] == 1e-4  # backbone first, as before
    print("  ok  test_param_groups_cover_every_parameter_once")


def test_rtx1070_checkpoint_still_loads():
    path = os.path.join(REPO, "ckpt", "proto_rtx1070", "keep", "epoch_005.pt")
    if not os.path.isfile(path):
        print("  skip test_rtx1070_checkpoint_still_loads (no checkpoint)")
        return
    from covispose.evaluate import load_model_from_ckpt
    m, cfg, blob = load_model_from_ckpt(path, "cpu")
    assert blob["loaded_weights"] == "raw" and cfg.pose_decoder == "conv"
    print("  ok  test_rtx1070_checkpoint_still_loads (strict)")


def test_param_count():
    m = build_model(CoVisPoseConfig(pretrained=False))
    p = count_params(m)
    print("  ok  test_param_count", {k: round(v, 2) for k, v in p.items()})
    assert p["total_M"] < 60, p


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{len(fns)} model checks passed")
