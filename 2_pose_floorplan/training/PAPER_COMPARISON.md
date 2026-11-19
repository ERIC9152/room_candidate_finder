# This reimplementation vs the CoVisPose paper

The README used to say the paper had "no public preprint, only the paywalled
ECCV version", and every design decision in `ASSUMPTIONS.md` was written on
that basis. **That was wrong**: ECCV 2022 is open access through the European
Computer Vision Association, and the paper is at
<https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920610.pdf>.
This file checks the guesses against it, then compares results under the
paper's own protocol.

---

## 1. Method: what the paper actually does, and what we guessed

| | paper | RTX 1070 run | server run | verdict |
|---|---|---|---|---|
| backbone | ResNet-50 + HorizonNet HCM | ResNet-34 | ResNet-50 | server matches |
| column tokens | **256**, each 1024-dim | 256, 256-dim | 1024, 384-dim | — |
| per-column outputs | **1024** — each token emits **4 columns** through one FC layer | 256 | 1024, one per token | the paper reaches 1 024 outputs from 256 tokens, far cheaper than our 1 024 tokens |
| transformer | 6 layers, 8 heads, FFN 2048 | 6 × 8, FFN 1024 | 8 × 8, FFN 1536 | — |
| position | fixed sinusoidal + learnable per-image segment embedding | same idea (A10) | same | **guessed right** |
| per-column head | a single FC layer | small MLP | small MLP | — |
| pose decoder | 6-layer 1D CNN (k3, BN, ReLU, halving channels) **applied to each image separately**, then concatenated → FC | one CNN over the concatenated A\|B sequence | attention pooling | the paper avoids the A\|B seam our 1070 decoder straddled |
| pose output | 5 numbers: unit rotation vector, unit translation direction, scale (ReLU) | (dx, dy, cos, sin) | same | 3 DoF either way (A6 **right** about DoF); the paper decouples direction from magnitude |
| column losses | L1 boundary + L1 correspondence + **0.25 · BCE** co-visibility | 1 / 1 / 2 weights (A9) | same | guessed, differently |
| pose loss | **MSE** on unit rotation vector (β = 3e-3) and on t·s (β = 6e-2) | L1 + circular L1 | same | different loss family |
| optimiser, schedule | lr 1e-4, **30 epochs**, 4 × V100, mixed precision, best by val loss sum | 3e-4, 6 epochs | 3e-4, 16 epochs, EMA | — |
| augmentation | random rotation, pair swap | roll, swap, gamma | + mirror | ours is a superset |
| world assumption | upright camera, fixed height, **Atlanta world** | upright, fixed height | same | — |

### Two differences that are not details

**Pairs across doorways (A8).** The paper did not stop at ZInD's intra-room
pairs. Its §4.4: co-visibility "cannot be confidently extended across doorways"
without knowing whether doors are open, so they **labelled 5 000 doors, trained
a door-state classifier, and used it to label the rest** — explicitly "to allow
extension of our method to whole floors". Our A8 kept every pair inside one
`complete_room`, reasoning that different rooms have near-zero overlap.

That single decision is why this model fails on the virtual-tour data, whose
hotspot graph is 100 % cross-room (`PHASE5_NOTES.md`). It was never a limit of
the method; it was a choice about the training pairs, and the paper made the
other one.

**Negatives and a lower overlap floor (A1).** The paper keeps pairs with
co-visibility ≥ **0.10**, not 0.30, and adds pairs with **zero** co-visibility
sampled at 10 %, for a positive:negative ratio of about 2.5. We have no
negatives at all — which is exactly why our co-visibility head, on
out-of-distribution pairs, reports its prior (0.765) instead of "nothing in
common" (A16). The paper then *uses* the co-visibility score as a confidence for
precision–recall over unordered panoramas, including negatives; our head could
not support that use.

---

## 2. Results under the paper's protocol

Server model (`best.pt`, epoch 15, EMA), ZInD test split, 26 267 pairs, single
forward pass, re-scored with `scripts/compare_with_paper.py`: rotation error,
**translation-direction** error, translation error **in metres**, fraction under
2.5° and 0.5 m, stratified by ground-truth co-visibility exactly as the paper's
Table 1. Metres use ZInD's own `scale_meters_per_coordinate` and each
panorama's pose scale; 17 of the 158 test houses have no metric scale in ZInD
at all and 13 more lack it on some floor, so metric numbers cover 83 % of
pairs (median 1.43 m per camera height).

### The headline number is misleading

| whole test set, direct | rot mean | rot median | rot < 2.5° | tdir median | m median | < 0.5 m |
|---|---|---|---|---|---|---|
| paper (Table 2, pairs ≥ 0.10 covis) | 4.33° | 0.97° | 80.2 % | 1.39° | 0.14 | 88.7 % |
| ours (pairs ≥ 0.30 covis) | **3.05°** | **0.87°** | **86.2 %** | **0.96°** | **0.09** | **94.2 %** |

That looks like a win on every column. **It is not**: our test pairs start at
co-visibility 0.30 and the paper's at 0.10, and the 10–25 % bucket is by far
its hardest. Comparing whole-set averages compares pair distributions, not
models.

### Like for like, bucket by bucket

Rotation error (the paper's first metric), direct regression:

| co-visibility | paper mean / median / < 2.5° | ours mean / median / < 2.5° | median gap |
|---|---|---|---|
| 75–100 % | 1.27 / 0.56 / 97.9 % | 1.74 / 0.71 / 92.9 % | +27 % |
| 50–75 % | 1.48 / 0.73 / 94.7 % | 3.27 / 1.01 / 83.6 % | +38 % |
| 25–50 % (ours: 30–50 only) | 3.47 / 1.03 / 83.9 % | 7.24 / 1.59 / 66.5 % | +54 % |
| 10–25 % | 8.90 / 2.10 / 55.9 % | *no pairs* | — |

**The paper is better on rotation in every bucket, and the gap widens as
overlap falls** — even though our 25–50 bucket excludes its harder 25–30 %
part. On translation the picture is different:

| co-visibility | translation direction, median | metres, median | < 0.5 m |
|---|---|---|---|
| 75–100 % | paper 1.15° / **ours 0.90°** | 0.09 / **0.07** | 98.9 / 98.5 % |
| 50–75 % | 1.09° / **0.91°** | **0.10** / 0.11 | **96.9** / 91.9 % |
| 25–50 % | 1.24° / 1.26° | **0.14** / 0.18 | **92.5** / 82.0 % |

Translation *direction* is as good or better at every overlap; metric
translation matches at high overlap and falls behind below it. RANSAC tells the
same story as direct regression. Our geometric (Procrustes) estimator, which
the paper does not have, is marginally the best of our three on rotation in
every bucket.

### Per-column accuracy (paper §5.5)

| | paper | ours (300 test pairs) |
|---|---|---|
| correspondence error < 5° | 73.1 % of columns | 87.9 % |
| floor boundary error < 2.5° | 83.1 % | 88.2 % |

The correspondence figure flatters us for the same reason the whole-set pose
figure does -- low-overlap pairs, where correspondence is hard, are not in our
test set. The boundary is monocular and far less sensitive to which pairs are
drawn, so 88.2 vs 83.1 % is the more meaningful of the two.

---

## 3. Reading it

Where the pairs are easy, this reimplementation is roughly on par with the
paper -- worse on rotation by about a tenth of a degree, better on translation
direction. Where overlap drops, it falls behind, and the rotation gap grows
from +27 % to +54 % in the median.

The shape of that gap points at one cause more than any other: **the training
pairs**. The paper trained on co-visibility down to 0.10, on zero-overlap
negatives, and on pairs extended through open doors. Ours stopped at 0.30,
within one room, with no negatives. Our 30–50 % bucket is the very edge of what
this model has ever seen; for the paper it is the middle of its distribution.
The gap grows exactly where our training data runs out.

Other differences could contribute and are not separated here -- the pose loss
(MSE on unit vectors vs our circular L1), the pose decoder (per-image CNN vs our
attention pooling), 30 epochs vs 16, token width 1024 vs 384. But the pair set is
the one difference that predicts the *shape* of the gap, and it is also the one
that explains the tour-data failure (§1). Matching the paper's pair
construction -- 0.10 floor, negatives, door-crossing pairs -- is therefore the
single change most likely to close both.

What this comparison does **not** show is that the server run's architectural
changes were worth it: we have no run with the paper's architecture on our
pairs, or ours on theirs. `configs/server_legacy_arch.yaml` would answer part of
that.

Raw numbers: `RESULTS/02_evaluation/paper_comparison.json`.

---

## 4. Same pair floor: the old model on a 0.10 test set

The comparison above had to leave the paper's hardest bucket empty. A test set
built the paper's way (`test_c10neg.csv`: co-visibility >= 0.10, plus 13 194
zero-overlap negatives; A17-A19) fills it, and makes the whole-set numbers
comparable for the first time. Same model, same weights, nothing retrained:

| whole test set, direct, pairs >= 0.10 | rot mean | rot median | rot < 2.5° | tdir median | m median | < 0.5 m |
|---|---|---|---|---|---|---|
| paper (Table 2) | **4.33°** | **0.97°** | **80.2 %** | 1.39° | 0.14 | **88.7 %** |
| server model | 8.38° | 1.16° | 71.8 % | **1.23°** | **0.12** | 82.4 % |

With the pair distributions matched, the "win" of section 2 is gone: the paper
is ahead on rotation and on the 0.5 m rate. The bucket the old model never
trained on is where it falls apart:

| co-visibility 10-25 %, direct | rot mean | rot median | rot < 2.5° | m median | < 0.5 m |
|---|---|---|---|---|---|
| paper | 8.90° | 2.10° | 55.9 % | 0.27 | 73.6 % |
| server model | 29.18° | 9.28° | 19.0 % | 0.87 | 35.2 % |

A 4.4x larger median rotation error. The 25-30 % band, now included in the
25-50 bucket, pulls that bucket's rotation median from 1.59° to 1.78°.

**On the negatives the co-visibility head is useless**: it calls a median of
**97 %** of the columns co-visible on pairs of rooms that share nothing, and
calls more than 20 % co-visible on 99.8 % of them. On 6 208 door-adjacent pairs
(`test_c10neg_adjacent.csv`, exact pose, evaluation only) pose is chance --
yaw median **94.2°**, under 10° for 7.5 % -- while the head reports a median of
83.5 % overlap, uncorrelated with the error (r = 0.056). That is the tour-data
failure (A16) reproduced inside ZInD, where the ground truth is exact.

These are the numbers `configs/server_c10neg.yaml` -- same model, the paper's
pair set -- is trained to beat. Raw: `RESULTS/02_evaluation/*old_c10neg*`,
`*old_adjacent*`, `paper_comparison_old_c10neg.json`.


---

## 5. Retrained on the paper's pair set (`configs/server_c10neg.yaml`)

Same architecture and schedule as the server run; only the pairs changed
(co-visibility >= 0.10 + negatives, A17-A19). 16 epochs, 46 h 08 m;
`best.pt` = epoch 13 on validation loss (epochs 14 and 15 were 0.002 behind).
Evaluated on the same test files as section 4, pair for pair.

### Under the paper's protocol (direct regression)

| | rot mean | rot median | rot < 2.5° | tdir median | m median | < 0.5 m |
|---|---|---|---|---|---|---|
| paper, pairs >= 0.10 | **4.33°** | **0.97°** | **80.2 %** | 1.39° | 0.14 | **88.7 %** |
| old model | 8.38° | 1.16° | 71.8 % | 1.23° | 0.12 | 82.4 % |
| **new model** | 5.31° | 1.26° | 72.4 % | 1.24° | **0.12** | 86.1 % |

| co-vis 10-25 % | rot mean | rot median | rot < 2.5° | < 0.5 m |
|---|---|---|---|---|
| paper | **8.90°** | **2.10°** | **55.9 %** | **73.6 %** |
| old model | 29.18° | 9.28° | 19.0 % | 35.2 % |
| new model | 14.54° | 4.15° | 34.8 % | 54.9 % |

### What changed, pair for pair (34 077 identical positives)

| co-visibility | yaw median old -> new | > 10° old -> new | new better on |
|---|---|---|---|
| 10-25 % | 9.28° -> **4.15°** | 48.2 % -> **26.5 %** | 70 % of pairs |
| 25-50 % | 1.78° -> 1.79° | 11.4 % -> 9.1 % | 52 % |
| 50-75 % | **1.01°** -> 1.17° | 3.0 % -> 2.8 % | 45 % |
| 75-100 % | **0.71°** -> 0.85° | 0.9 % -> 1.0 % | 44 % |

Gross failures (> 45°) halve, 5.13 % -> 2.67 %; mean rotation error drops 37 %.
**But precision on easy pairs regressed**: +16-20 % median at >= 50 % overlap,
and on the old >= 0.30 test set acc@(0.1 m, 1°) fell 0.4455 -> 0.3841 while
acc@(0.5, 10°) held (0.9549 -> 0.9565). Correspondence error, measured on the
same (A17-fixed) targets, rose 4.50° -> 5.57°. The pair set traded high-overlap
precision for low-overlap robustness. Checkpoint selection is **not** the
cause: epoch 15 (end of the anneal) is better than the selected epoch 13
everywhere, but only by 2-4 % -- 75-100 % yaw median 0.85° -> 0.83° against the
old model's 0.71°; on the >= 0.30 test set acc@(0.1 m, 1°) 0.3841 -> 0.3947
against 0.4455. Exposure is not the cause either: both runs saw the same
>= 0.30 pairs for the same 16 epochs. What remains is interference -- the shared
network now also has to place low-overlap pairs and reject empty ones -- and it
costs about a sixth of the median rotation precision on easy pairs. Epoch 15 is
the checkpoint of record (`RESULTS/02_evaluation/*e15_*`).

### The co-visibility head now says "nothing in common"

| | old | new |
|---|---|---|
| negatives, predicted co-visible fraction (median) | 0.975 | **0.000** |
| negatives called > 5 % co-visible | 100 % | **1.2 %** |
| door-adjacent pairs, predicted fraction (median) | 0.835 | **0.000** |

On door-adjacent pairs pose is still chance (yaw median 89.9°) -- nothing was
trained to cross a doorway -- but the head now *knows*: RANSAC refuses 96.3 %
of them for lack of co-visible columns, and where the head does see overlap
the pose is mostly right:

| new model, adjacent pairs, predicted co-visible | n | yaw median | < 10° |
|---|---|---|---|
| < 2 % | 6 008 | 90.3° | 6.4 % |
| 2-10 % | 110 | 97.6° | 11.8 % |
| 10-30 % | 52 | 59.3° | 23.1 % |
| >= 30 % | 38 | **5.8°** | **57.9 %** |

So the head is now usable as a gate: drop edges it calls empty, keep the rest.
That is the A16 failure fixed in the direction that matters for the tour data
-- refusal instead of confident nonsense. Producing poses *through* doorways
still needs door-crossing positives (stage 2).

Raw: `RESULTS/02_evaluation/*new_*`, `paper_comparison_new_c10neg.json`;
training log `RESULTS/01_training/server_c10neg/`.
