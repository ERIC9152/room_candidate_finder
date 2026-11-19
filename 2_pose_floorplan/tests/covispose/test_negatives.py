"""Cross-room pairs: the back-face rule (A17), room connectivity (A18) and
pose masking on negatives (A19).

Synthetic geometry, no ZInD needed -- except the last test, which checks the
fixed co-visibility against Zillow's own scores on real pairs.

    python -m pytest tests/test_negatives.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from covispose.geometry import PanoPose, ray_cast_polygon
from covispose.pairs import PairRef, read_pair_index, room_connections, write_pair_index
from covispose.targets import compute_pair_targets
from covispose.zind_parse import HouseRecord, PanoRecord

ZIND_ROOT = os.environ.get("ZIND_ROOT", r"C:\zind_data")
COVIS_CSV = os.environ.get("ZIND_COVIS_CSV", r"C:\zind\covis_scores.csv")


def _square(cx, cy, half):
    return np.array([[cx - half, cy - half], [cx + half, cy - half],
                     [cx + half, cy + half], [cx - half, cy + half]], float)


def _pano(name, croom, centre, room_global, doors=(), floor="floor_01"):
    """A pano at ``centre`` (global) in a room given in global coordinates.

    Pose: identity rotation, scale 1, so local = global - centre.
    """
    c = np.asarray(centre, float)
    return PanoRecord(
        house_id="9999", floor_id=floor, complete_room_id=croom, room_id=croom.replace("complete", "partial"),
        pano_id=name, image_path="", pose=PanoPose(c, 0.0, 1.0),
        vertices=room_global - c, camera_height=1.0, ceiling_height=1.6,
        is_primary=True, is_inside=True, layout_source="layout_visible",
        doors=np.asarray([np.asarray(d, float) - c for d in doors]).reshape(-1, 2, 2),
    )


# Two 4 x 4 rooms side by side, sharing the wall x = 0 -- modelled the way
# ZInD has it, as two faces 0.05 apart (a wall's thickness), well inside the
# 0.10 distance tolerance.
ROOM_L = _square(-2.025, 0.0, 2.0)   # right face at x = -0.025
ROOM_R = _square(2.025, 0.0, 2.0)    # left face at x = +0.025


def test_ray_cast_reports_hit_edge():
    t, edge, s = ray_cast_polygon(np.zeros(2), np.array([[1.0, 0.0], [0.0, 1.0]]),
                                  _square(0, 0, 1), return_edge=True)
    np.testing.assert_allclose(t, [1.0, 1.0])
    assert list(edge) == [1, 2]          # +x wall, then +y wall
    np.testing.assert_allclose(s, [0.5, 0.5])
    # unchanged default
    np.testing.assert_allclose(ray_cast_polygon(np.zeros(2), np.array([[1.0, 0.0]]),
                                                _square(0, 0, 1)), [1.0])


def test_shared_wall_seen_from_both_sides_is_not_covisible():
    a = _pano("pano_1", "complete_room_01", (-1.0, 0.3), ROOM_L)
    b = _pano("pano_2", "complete_room_02", (1.0, -0.2), ROOM_R)
    tg = compute_pair_targets(a, b, n_cols=512)
    assert tg.covis_a.sum() == 0 and tg.covis_b.sum() == 0


def test_same_room_covisibility_is_untouched():
    a = _pano("pano_1", "complete_room_01", (-2.5, 0.4), ROOM_L)
    b = _pano("pano_2", "complete_room_01", (-1.4, -0.5), ROOM_L)
    tg = compute_pair_targets(a, b, n_cols=512)
    # Convex room, both cameras inside: every wall point is seen by both.
    assert tg.covis_a.mean() > 0.97 and tg.covis_b.mean() > 0.97


def test_room_connections_from_doors():
    door_l = [[-0.025, -0.4], [-0.025, 0.4]]   # the two faces of one doorway
    door_r = [[0.025, 0.4], [0.025, -0.4]]
    far = _square(20.0, 0.0, 2.0)
    house = HouseRecord("9999")
    for p in (_pano("pano_1", "complete_room_01", (-1.0, 0.0), ROOM_L, doors=[door_l]),
              _pano("pano_2", "complete_room_02", (1.0, 0.0), ROOM_R, doors=[door_r]),
              _pano("pano_3", "complete_room_03", (20.0, 0.0), far,
                    doors=[[[18.0, -0.4], [18.0, 0.4]]])):
        house.panos[p.key] = p
    conn = room_connections(house)
    assert conn == {("floor_01", "complete_room_01", "complete_room_02")}


def test_pair_kind_roundtrip_and_old_indices(tmp_path):
    rows = [PairRef("0001", "floor_01", "r1", "p1", "r2", "p2", 0.0, 3.0, 10.0, kind="neg"),
            PairRef("0001", "floor_01", "r1", "p1", "r1", "p3", 0.6, 1.0, 5.0)]
    path = tmp_path / "x.csv"
    write_pair_index(str(path), rows)
    assert [p.kind for p in read_pair_index(str(path))] == ["neg", "pos"]
    # an index written before negatives existed has no "kind" column
    old = tmp_path / "old.csv"
    old.write_text("house_id,floor_id,room_a,pano_a,room_b,pano_b,covis_score,baseline,dyaw_deg\n"
                   "0001,floor_01,r1,p1,r1,p3,0.6,1.0,5.0\n")
    assert read_pair_index(str(old))[0].kind == "pos"


def test_loss_and_metrics_ignore_pose_on_negatives():
    torch = pytest.importorskip("torch")
    from covispose.losses import LossWeights, covispose_loss, covispose_metrics

    n, c = 4, 16
    g = torch.Generator().manual_seed(0)
    out = {f"covis_logit_{s}": torch.randn(n, c, generator=g) for s in "ab"}
    out.update({f"corr_vec_{s}": torch.randn(n, c, 2, generator=g) for s in "ab"})
    out.update({f"bon_{s}": torch.rand(n, c, generator=g) for s in "ab"})
    out["t"] = torch.randn(n, 2, generator=g, requires_grad=True)
    out["yaw_vec"] = torch.randn(n, 2, generator=g)
    batch = {f"covis_{s}": (torch.rand(n, c, generator=g) > 0.5).float() for s in "ab"}
    batch.update({f"corr_{s}": torch.rand(n, c, generator=g) for s in "ab"})
    batch.update({f"bon_{s}": torch.rand(n, c, generator=g) for s in "ab"})
    batch.update({f"bon_valid_{s}": torch.ones(n, c, dtype=torch.bool) for s in "ab"})
    batch["t"] = torch.randn(n, 2, generator=g)
    batch["yaw"] = torch.rand(n, generator=g)
    batch["pose_valid"] = torch.tensor([True, True, False, False])
    batch["is_neg"] = ~batch["pose_valid"]

    terms = covispose_loss(out, batch, LossWeights())
    terms["pose_t"].backward()
    # no gradient reaches the negatives' pose
    assert out["t"].grad[2:].abs().sum() == 0 and out["t"].grad[:2].abs().sum() > 0
    # and the value is the mean over positives only
    expect = (out["t"][:2] - batch["t"][:2]).abs().sum(-1).mean()
    assert torch.isclose(terms["pose_t"], expect)

    m = covispose_metrics({k: v.detach() for k, v in out.items()}, batch)
    t_err = torch.linalg.norm(out["t"].detach() - batch["t"], dim=-1)
    assert m["t_err_mean"] == pytest.approx(t_err[:2].mean().item(), rel=1e-6)
    assert 0.0 <= m["covis_frac_neg"] <= 1.0

    # a batch of only negatives: no pose loss, NaN pose metrics (skipped by validate)
    batch["pose_valid"][:] = False
    batch["is_neg"][:] = True
    assert covispose_loss(out, batch, LossWeights())["pose_t"].item() == 0.0
    assert np.isnan(covispose_metrics({k: v.detach() for k, v in out.items()}, batch)["t_err_mean"])


@pytest.mark.skipif(not (os.path.isdir(ZIND_ROOT) and os.path.isfile(COVIS_CSV)),
                    reason="needs ZInD")
def test_fixed_covisibility_agrees_with_zillow():
    """The back-face rule moved our co-visibility *towards* Zillow's own
    score (r 0.9953 -> 0.9987 on 1 457 pairs, see A17); keep it there."""
    from covispose.pairs import read_covis_scores
    from covispose.zind_parse import load_house

    houses = ["%04d" % i for i in range(6)]
    rows = read_covis_scores(COVIS_CSV, houses=houses, covis_min=0.0)
    rng = np.random.default_rng(0)
    rows = [rows[i] for i in rng.choice(len(rows), min(300, len(rows)), replace=False)]
    cache, ours, theirs = {}, [], []
    for p in rows:
        if p.house_id not in cache:
            cache[p.house_id] = load_house(ZIND_ROOT, p.house_id, primary_only=False)
        a, b = cache[p.house_id].panos.get(p.key_a), cache[p.house_id].panos.get(p.key_b)
        if a is None or b is None:
            continue
        tg = compute_pair_targets(a, b, n_cols=256)
        ours.append(0.5 * (tg.covis_a.mean() + tg.covis_b.mean()))
        theirs.append(p.covis_score)
    ours, theirs = np.array(ours), np.array(theirs)
    assert np.corrcoef(ours, theirs)[0, 1] > 0.997
    assert abs((ours - theirs).mean()) < 0.01
