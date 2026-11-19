# Phase 0 + Phase 1 notes

## Phase 0 — environment

| item | value |
|---|---|
| GPU | NVIDIA GeForce GTX 1070, 8 GB, compute capability **6.1** (Pascal) |
| driver / runtime | 560.94 / CUDA 12.6 reported by `nvidia-smi` |
| PyTorch | 1.12.0+cu113, `torch.cuda.is_available() == True` |
| torchvision | 0.13.0+cu113 |
| Python | 3.10.11, venv at `C:\HorizonNet\HorizonNet\venv` (reused) |

Added to that venv for this project: `matplotlib==3.5.3`, `pyyaml`. Nothing was
removed or upgraded.

> **Note on AMP.** The task brief asks for `torch.cuda.amp`. Pascal (sm_61) has
> **no tensor cores**, and its native FP16 arithmetic throughput is 1/64 of FP32.
> AMP on a 1070 therefore buys **memory**, not speed — and can be slightly slower
> per step. We still use it, because memory is the binding constraint at 8 GB,
> but we should not expect a speedup. This is measured again in Phase 3.

### HorizonNet reference

The official repo is already checked out at `C:\HorizonNet\HorizonNet`. What we
reuse (conventions and design, re-implemented rather than vendored):

* `model.py::Resnet` — multi-scale ResNet feature extraction (4 blocks, 1/4 … 1/32).
* `model.py::GlobalHeightConv` / `GlobalHeightStage` — the **height-compression**
  trick: each block is squeezed to height 1 by strided `(2,1)` convs and resized
  to a common width, then all four scales are concatenated into one 1D
  per-column feature sequence. This is exactly the "per image-column feature
  sequence" CoVisPose needs.
* `model.py::LR_PAD` / `wrap_lr_pad` — circular (left-right) padding on every
  conv so the 360° wraparound is seamless. Essential and easy to forget.
* `dataset.py::cor_2_1d` — the `bon` boundary parameterisation,
  `b = ((v + 0.5)/H − 0.5)·π`, which we adopt verbatim (see assumption A7).
* Input resolution convention **512 × 1024**.

---

## Phase 1 — data pipeline

### What was built

| file | role |
|---|---|
| `covispose/geometry.py` | conventions, ray casting, relative pose |
| `covispose/zind_parse.py` | `zind_data.json` → flat `PanoRecord`s |
| `covispose/pairs.py` | `covis_scores.csv` → filtered pair index |
| `covispose/targets.py` | per-column boundary / correspondence / co-visibility |
| `covispose/viz.py` | ground-truth overlay figures |
| `tests/test_geometry.py` | 8 self-checks, cross-validated against ZInD's own code |
| `scripts/build_pair_index.py` | writes `data/pairs/{train,val,test}*.csv` |
| `scripts/verify_targets.py` | the mandatory Phase-1 visual check |
| `scripts/analyze_pairs.py` | the dataset analysis that produced the numbers below |

### Ground-truth derivation

ZInD gives each pano a `floor_plan_transformation`
`p_global = s·Rot(θ)·p_local + t`. For panos A, B on one floor:

```
p_B = k · Rot(Δyaw) · p_A + t_AB
Δyaw = θ_A − θ_B          k = s_A / s_B          t_AB = Rot(−θ_B)(t_A − t_B)/s_B
```

`t_AB` is A's camera centre in B's frame, in units of B's camera height.
This is checked against composing through the global frame in
`test_relative_pose_against_global` (50 random poses, max error < 1e-9).

Per column *j* of A: cast a ray at azimuth `θ_j` against A's `layout_visible`
polygon → distance `r_j` → boundary `b_j = atan2(1, r_j)` (HorizonNet form).
Map the boundary point into B's frame → its azimuth is the **angular
correspondence** target, and the distance test of assumption A5 gives the
**co-visibility** target.

### Bugs found and fixed during Phase 1

1. **Ray/segment sign error** in `ray_cast_polygon` — the segment parameter was
   computed as `s = −(a×d)/(d×e)` instead of `s = +(a×d)/(d×e)`. Symptom: 33 % of
   columns reported "no polygon hit" even though the camera was verified
   (via shapely) to be strictly inside the polygon. After the fix: **0.00 %**.
2. **Over-aggressive pano filtering** — `primary_only=True` dropped the
   `covis_scores.csv` join rate to 10.2 %. See assumption A2.
3. A flawed *test* (random polygons with angular gaps > π are not star-shaped
   about the origin, so "every ray hits" was the wrong expectation) — fixed in
   the test, not the code.

### Validation results

* `tests/test_geometry.py` — **8/8 pass**, including two that cross-check our
  azimuth/elevation conventions directly against ZInD's
  `code/transformations.py` (max disagreement 0.499 px at W=1024, exactly the
  known `W` vs `W−1` half-pixel offset).
* **Co-visibility vs Zillow's published score: r = 0.994** over 6 794 pairs from
  25 houses. Bucketed agreement:

  | `covis_score` bucket | n | our mean per-column co-visibility |
  |---|---|---|
  | [0.00, 0.10) | 1656 | 0.077 |
  | [0.10, 0.20) | 784 | 0.182 |
  | [0.20, 0.35) | 618 | 0.304 |
  | [0.35, 0.50) | 496 | 0.452 |
  | [0.50, 0.65) | 494 | 0.600 |
  | [0.65, 0.80) | 588 | 0.749 |
  | [0.80, 1.01] | 2158 | 0.940 |

* On an inspected cross-room pair, co-visible boundary distances agreed to
  `0.0000` — the two panos' annotations literally share wall vertices in the
  global frame, so the relative pose is exact, not approximate.

### Dataset facts established (these are not in any paper)

* `covis_scores.csv` has 458 103 rows over all 1 575 houses; **100 % of rows are
  pairs within a single `complete_room`** (checked on every joinable row of the
  first 25 houses). So ZInD's co-visibility pairs are intra-complete-room
  wide-baseline pairs, not arbitrary cross-room pairs.
* `home_id` in the CSV is unpadded (`"0"`, `"1574"`); folders are 4-digit.
* `camera_height` is `1` for every pano in ZInD.
* `floor_plan_transformation.scale` is **not** constant within a house (3 of the
  first 40 houses vary it), so it must be read per pano — but the *ratio*
  between two panos of a pair is 1.0000 for 99.4 % of pairs (assumption A6).
* Baseline distribution of the kept training pairs (camera heights, 1 ≈ 1.5 m):
  p10 0.76, p25 1.23, **p50 1.97**, p75 3.09, p90 4.30, p99 7.69. Genuinely
  wide-baseline. `|Δyaw|` median is 86°, and 15 % of pairs are within 5° of
  aligned.

### Prototype index (30 houses per split)

```
train: houses=30  rows>=0.3: 6212  kept: 5968 (96.1%)  covis mean=0.761  |t| p50=1.98
val  : houses=30  rows>=0.3: 6822  kept: 6654 (97.5%)  covis mean=0.790  |t| p50=2.04
test : houses=30  rows>=0.3: 5060  kept: 4812 (95.1%)  covis mean=0.769  |t| p50=1.98
```

Drop reasons: `pano_missing_or_filtered` (referenced pano is `is_inside=False`
or has no layout), `degenerate_baseline` (A3), `scale_ratio_not_1` (A6).

### Visual check

`outputs/phase1_verify/*.png`, five pairs spanning `covis_score` 0.32 → 1.00.
Each figure shows both panos with the GT floor-wall boundary drawn on the image
(co-visible span highlighted), a hue bar encoding the angular correspondence, a
top-down view with both polygons brought into B's frame, and the correspondence
curve. The boundary tracks the visible floor-wall line, and the non-co-visible
spans land exactly on the geometry only one camera can see (e.g. a hallway
visible from B but occluded from A).
