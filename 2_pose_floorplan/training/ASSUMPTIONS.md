# Assumptions log ("this is where I guessed")

The CoVisPose paper (Hutchcroft et al., ECCV 2022) has no public preprint — only
the paywalled ECCV version. Every design decision below is therefore **my
engineering judgement, not something read off the paper**. Each entry records
*what* was decided, *why*, and *how to change it*.

Entries are append-only; later phases add to the bottom.

---

## A1. Pair source = ZInD's `covis_scores.csv`, threshold 0.30

* **Decision**: training pairs come from `covis_scores.csv` rows with
  `covis_score >= 0.30`.
* **Why**: the paper says pairs are "wide baseline" and co-visible, but does not
  publish the sampling rule. `covis_scores.csv` is the only official
  co-visibility signal shipped with ZInD, and our own per-column co-visibility
  ground truth reproduces its score with **r = 0.994** (6 794 pairs, 25 houses),
  so the two definitions agree. Below ≈0.2 the pair shares so little of the room
  that relative pose is often under-constrained — that is label noise, not a hard
  example. 0.30 keeps ~61 % of all rows.
* **Full-CSV distribution**: p25 0.135, p50 0.497, p75 0.876 (458 103 rows).
* **Change it**: `--covis-min` on `scripts/build_pair_index.py`.
* **Revisited (paper checked, `docs/PAPER_COMPARISON.md`)**: the paper's floor
  is **0.10**, and "too little overlap is label noise" turned out to be the wrong
  call -- per co-visibility bucket, our rotation gap to the paper widens from
  +27 % to +54 % exactly as overlap drops toward the edge of our pair set.
  `configs/server_c10neg.yaml` trains on the paper's floor; 0.10 keeps 80.0 %
  of the CSV (366 339 rows) against 61.3 % at 0.30. The r = 0.994 above was
  measured before the A17 fix; after it, r = **0.9987**.

## A2. Secondary panos are kept, `is_inside=False` panos are dropped

* **Decision**: `primary_only=False`, `inside_only=True`.
* **Why**: `covis_scores.csv` references secondary panos heavily. Filtering to
  primary panos only (which is what HorizonNet's own ZInD prep does, for layout
  *quality*) drops the join rate from **96.4 % to 10.2 %**. Secondary panos still
  carry their own `floor_plan_transformation` and `layout_visible`, which is
  everything CoVisPose needs. Panos flagged `is_inside=False` were annotated from
  outside the room and their layout is unreliable, so they go.
* **Change it**: `primary_only` / `inside_only` in `zind_parse.load_house`.

## A3. Minimum baseline 0.10 camera heights

* **Decision**: drop pairs with `|t| < 0.10` (≈15 cm).
* **Why**: ZInD contains pano pairs shot from essentially the same spot
  (`|t| ≈ 0.005`, `covis_score ≈ 0.99`). Their translation *direction* is
  undefined, so they are pure noise for a pose regressor. Only ~0.9 % of rows.
* **Change it**: `--min-baseline`.

## A4. Layout source = `layout_visible`

* **Decision**: per-column boundary ground truth comes from `layout_visible`,
  falling back to `layout_raw` then `layout_complete`.
* **Why**: the paper's layout head predicts the floor-wall boundary *as seen in
  the image*, and `layout_visible` is the annotation that excludes occluded
  geometry. It is also what HorizonNet's own ZInD preparation uses. Measured:
  with `layout_visible`, **0.00 %** of columns fail to hit the polygon, vs 11 %
  for `layout_raw` (whose camera is sometimes outside its own polygon).
  Note `layout_visible` legitimately extends *through* doorways into adjacent
  space — verified numerically, and that is what makes cross-room co-visibility
  work.
* **Change it**: `--layout`.

## A5. Co-visibility ground-truth rule

* **Decision**: column *j* of pano A is co-visible with B iff the floor-wall
  boundary point A sees at column *j*, mapped into B's frame, has radial
  distance matching B's own boundary distance along that azimuth, within
  `max(0.10, 0.05 · r)` camera heights.
* **Why**: the paper defines co-visibility as "is this column's content also
  visible in the other image" but publishes no rule. This one needs nothing but
  the two per-pano layouts, and it handles occlusion for free — a point hidden
  behind one of B's own walls gets a smaller `r_B` and is correctly rejected.
  **Validation**: mean per-column co-visibility under this rule correlates
  **r = 0.994** with Zillow's published `covis_score`, and on inspected pairs the
  agreeing distances match to `0.0000` (the two panos' annotations share
  identical wall vertices in the global frame).
* **Change it**: `DEFAULT_ABS_TOL` / `DEFAULT_REL_TOL` in `covispose/targets.py`.

## A6. Relative pose is 3-DoF, not 4 — and the word "scale" means two things

* **Decision**: the pose target is `(Δx, Δy, Δyaw)` only. No scale output.

* **The naming collision.** "Scale" appears in two unrelated roles here, and
  conflating them is what produced the original 4-DoF brief:

  1. **`floor_plan_transformation.scale`** — a ZInD annotation field. It converts
     *one* pano's local coordinates into the shared floor-plan coordinate system.
     It is a property of the annotation, **not part of a camera pose**, and it
     never appears in a relative pose between two panos except through the ratio
     `k = scale_A / scale_B`.
  2. **"pose regression *with scale*"** in the paper's abstract — this describes
     the *output being metric*. Classical epipolar geometry from two views
     recovers the translation **direction** only; its magnitude is unrecoverable.
     The paper's point is that their network regresses a translation that
     carries true metric magnitude. It is **not** a fourth degree of freedom.

* **Why 3-DoF is right here**: ZInD normalises every pano's local frame by its
  own camera height (`camera_height == 1` for every pano in the dataset), so our
  GT translation is already expressed in metric units of camera height
  (1 unit ≈ 1.5 m). `(Δx, Δy)` therefore *already carries* the "scale" the paper
  is talking about — predicting a separate scalar would be predicting nothing.
  Measured: `k = scale_A / scale_B` is **exactly 1.0000 for 99.4 % of pairs**
  (18 000+ pairs); the 0.6 % where `k ≠ 1` would mean the two panos had different
  camera heights, and they are dropped at index time (`scale_ratio_not_1`)
  rather than silently mis-labelled.

* **Change it**: re-enable a 4th output in the pose head and stop dropping those
  pairs — only meaningful if you switch to a dataset where camera height varies
  between the two views of a pair.

## A8. Pairs stay intra-`complete_room` — that is the problem setting, not a shortcut

* **Decision**: do not synthesise extra cross-room pairs. Use ZInD's official
  pairs as-is.
* **Why**: we measured that 100 % of `covis_scores.csv` rows are pairs within a
  single `complete_room`, and that is not an arbitrary restriction on Zillow's
  part. A `complete_room` *is* one physical space that had to be shot from
  several positions and split into `partial_room`s because of occlusion, then
  merged back. So the official pairs already are "the same room from very
  different viewpoints" — exactly the wide-baseline setting the paper targets.
  Two genuinely different rooms (kitchen vs bedroom) have near-zero visual
  overlap, so their co-visibility target would be ~0 everywhere and carry no
  training signal for correspondence or pose.
* **Change it**: generate cross-room pairs and score them with
  `targets.compute_pair_targets`; only worth doing if a later experiment needs
  hard negatives or a cross-room deployment scenario.
* **Revisited**: both conditions came true. The tour data is a cross-room
  deployment (A16), and the paper turns out to train with zero-overlap negatives
  *and* door-crossing positives. Negatives are now generated (A18, A19);
  door-crossing positives are not, because they need door open/closed states
  that ZInD does not annotate -- the paper labelled 5 000 doors and trained a
  classifier for exactly this. The "co-visibility ~0, no signal" argument above
  was also wrong in a way that mattered: scored naively, 15 % of cross-room
  pairs came out *above* 0.10 (A17).

## A9. Loss weights

* **Decision**: `covis 1.0, corr 1.0, bon 2.0, pose_t 1.0, pose_yaw 1.0`.
* **Why**: the paper publishes no weights. These were picked so no term
  dominates at initialisation, using the measured starting magnitudes:
  covis BCE ≈ 0.68, corr ≈ 1.64 rad (a uniformly random angle averages π/2),
  bon ≈ 0.36 rad, pose_t ≈ 1.92 camera heights, pose_yaw ≈ 2.25 rad. The
  boundary term is the smallest and the only purely monocular one, so it gets
  2.0 to keep the layout head from being drowned out.
* **Change it**: `LossWeights` in `covispose/losses.py`, or the `loss:` block of
  the config. A principled alternative worth trying later is learned
  uncertainty weighting (Kendall et al.), which removes the guess entirely.

## A10. The model is deliberately *not* roll-equivariant

* **Decision**: add an **absolute** azimuth positional encoding, and teach
  rotation robustness with random roll augmentation instead of building it in.
* **Why**: the correspondence head has to emit an azimuth *in the other pano's
  frame*, which is an absolute quantity, so the transformer needs absolute
  position. The column encoder itself *is* exactly roll-equivariant (verified:
  relative error 0.00e+00 under a 64-px roll), and each pano is rolled
  independently during training with the pose and correspondence targets
  transformed to match exactly (verified: 6e-5 deg closed-loop error).
* **Note**: the positional encoding uses sin/cos of **integer** multiples of the
  azimuth, capped at the column grid's Nyquist frequency. Integer multiples make
  it exactly periodic over 360°, so there is no artificial discontinuity at the
  wraparound the way an index-based encoding would have.
* **Change it**: swap `azimuth_encoding` for a relative-position attention bias.

## A11. RANSAC inlier threshold 0.10, matched to the co-visibility tolerance

* **Decision**: `ransac_rigid(threshold=0.10)`, the same value as
  `targets.DEFAULT_ABS_TOL`.
* **Why**: measured, not guessed. Feeding the RANSAC estimator *ground-truth*
  per-column values (house 0000, 102 pairs) gives a median |t| error of **0.030**
  camera heights at threshold 0.25 but **0.015** at 0.10, with the same number of
  pairs solved. The residual is not a bug and does not come from column
  discretisation — it is the co-visibility label slack: a column counts as
  co-visible when the two boundary distances agree to within `abs_tol`, so the
  correspondences carry that much error into the least-squares refit. Sweeping
  `abs_tol` 0.10 → 0.05 moves the median from 0.0300 to 0.0037 (8×), while
  sweeping `n_cols` 128 → 1024 changes nothing.
* **Consequence**: ~0.015 camera heights (≈2 cm) is the **noise floor** of the
  RANSAC pose path under this labelling. The direct pose decoder has no such
  floor. Worth remembering when comparing the two estimators in Phase 4.

## A12. Circular padding is applied to MaxPool2d as well as Conv2d

* **Decision**: extend HorizonNet's `wrap_lr_pad` to patch `nn.MaxPool2d`.
* **Why**: HorizonNet's version only patches `Conv2d`, which leaves the ResNet
  stem's max-pool zero-padded and so leaves a seam at the 360° wraparound.
  Measured: patching it takes the column encoder from **4.9 % to 0.0 %** relative
  error under a roll of the input. This is a deliberate (small) deviation from
  the reference implementation.

## A13. Runs are not bitwise reproducible, and that is AMP's doing

* **Fact, measured**: two *identical* backward passes, same process, same
  weights, same input, disagree by **1.1e-3 with AMP** and **5.8e-7 without**.
  cuDNN's convolution backward accumulates with atomics, so the summation order
  varies run to run. Adam then amplifies that over subsequent steps.
* **Consequence**: "resume produces bitwise-identical results" is not
  achievable while AMP is on, and any test asserting a fixed small tolerance is
  really just measuring GPU noise. `tests/test_checkpoint.py` therefore does two
  things instead: it asserts the restored state is **bit-exact** (no forward
  pass involved, so it is a clean check), and it calibrates the end-to-end
  continuation test against a **measured** noise floor obtained by resuming
  twice through the same code path. Measured result: resume error 2.02e-4 vs
  noise floor 2.02e-4 — identical, i.e. resume contributes no error at all.
* **Change it**: `torch.use_deterministic_algorithms(True)` plus
  `cudnn.benchmark=False` would make it reproducible, at a speed cost we cannot
  afford on this card.

## A14. Two deferred decisions (open, deliberately not acted on yet)

Both were identified after epoch 0 and both are the kind of thing that would be
tuned against a moving target if done early. Recorded here so they are not
forgotten, with the trigger condition that should make us act.

**A14a — co-visibility class imbalance. UNRESOLVED, and deliberately left so.**
The positive rate is 76.5 %, so accuracy is nearly uninformative (an
all-positive predictor scores 0.7652) and **IoU is the metric to watch**. Over
the 6-epoch run IoU went 0.8046 → 0.8516 → 0.8725 → 0.8847 → 0.8886 → 0.8895,
i.e. gains of +0.0470, +0.0209, +0.0123, +0.0038, **+0.0009**. It flattened,
which was the trigger we had set for trying class-balanced BCE.

**But this run cannot support that conclusion.** The flattening coincides with
the cosine LR annealing to ~0, and *every* head flattened at the same time by a
similar factor (correspondence −0.8 %, boundary −1.0 %, translation −1.8 %, yaw
−0.8 % in the final epoch). Nothing here distinguishes "class imbalance caps
this head" from "the learning rate went to zero".

*Status*: **not fixed, and not diagnosed.** Settling it needs a controlled
second run — same schedule, `pos_weight` enabled — which is another ~47 h on a
single GTX 1070 and was judged not worth it for this project. Anyone reporting
on this head should say it is an open question rather than imply it was
addressed.
*If someone does run it*: `pos_weight = (1 - p) / p` with p = 0.765 gives ≈0.31,
or use a `covis_score`-stratified sampler. **Not** a bigger model — nothing in
the evidence points at capacity.

**A14b — RANSAC inlier threshold. RESOLVED: 0.10 → 0.30.**
`threshold=0.10` was set in A11 by feeding the estimator ground-truth per-column
values, where the residual noise floor is 0.015. The trained network's
reconstructed points inherit the boundary head's ~2° error, so that threshold
was far too tight.

Swept on **1 500 validation pairs only** (never test) with the epoch-5
checkpoint, `scripts/tune_ransac.py`:

| threshold | inlier rate | t median | yaw median | acc@(0.5, 10°) |
|---|---|---|---|---|
| 0.05 | 20.5 % | 0.1140 | 2.138° | 0.7727 |
| **0.10 (old)** | 40.7 % | 0.1034 | 1.814° | 0.8040 |
| 0.15 | 55.2 % | 0.0981 | 1.748° | 0.8340 |
| 0.20 | 65.0 % | 0.1006 | **1.661°** | 0.8453 |
| **0.30 (chosen)** | 76.5 % | 0.1032 | 1.720° | 0.8560 |
| 0.40 | 82.6 % | 0.1041 | 1.797° | 0.8607 |
| 0.75 | 91.4 % | 0.1172 | 1.833° | **0.8633** |
| 1.00 | 94.0 % | 0.1217 | 1.914° | 0.8567 |

**Chose 0.30, not the argmax.** The accuracy surface is flat from 0.30 to 1.00
(0.856–0.863); picking 0.75 because it scored 0.0073 higher on 1 500 pairs
would be fitting noise. 0.30 sits in the middle of the plateau where all three
criteria are near their best, and it keeps the estimator *actually robust* — at
threshold 1.00 the inlier rate is 94 %, meaning nothing is being rejected and
RANSAC has degenerated into plain least squares.

**What the flat surface tells us**: outliers are *not* the dominant error
source. Widening the gate 3× barely changes accuracy, so what limits the RANSAC
path is the accuracy of the boundary and correspondence heads that generate the
points, not contamination. Re-tuning bought ~5 points of accuracy
(0.8040 → 0.8560) and nothing more will come from the threshold.

## A15. Server-scale decisions (unmeasured until the server run)

* **Decision**: ResNet-50 (V2 weights), 1 024 columns, 8 × d384 transformer
  with drop-path 0.1, attention pose decoder, geometric Procrustes pose loss at
  weight 0.5 (ramped in over the second epoch), mirror augmentation, AdamW
  wd 0.05, 16 epochs with a 1 % LR floor, EMA 0.9998, lr 5e-4 at 24 pairs/step.
* **Why**: each is argued in `docs/SERVER_OPTIMIZATION.md`. The ones that are
  pure guesses, with no evidence from this dataset yet: the geo loss weight and
  ramp, drop-path 0.1, wd 0.05, the epoch count, and the √-scaled LR.
* **Verified before the run**: correctness only — mirror exactness, TTA inverse,
  Procrustes recovering GT pose (median |t| 0.0375 with 2 IRLS rounds, vs 0.0553
  without), bit-identical outputs for the RTX 1070 checkpoint, CPU training +
  resume + 2-process DDP.
* **Measured after the run** (`docs/PHASE5_NOTES.md`): every test metric improved
  over the 1070 run, e.g. yaw median 1.671° → 0.869°, and the two weakest
  buckets gained ~18 points each. **Two of the guesses were wrong or unneeded**:
  lr 5e-4 was unstable and had to drop to 3e-4 (the √-scaling argument does not
  hold here), and 16 epochs was one or two more than needed — epochs 12–15
  bought 0.009 of validation score between them.
* **Still unattributed**: which of the simultaneous changes produced the gain.
  `configs/server_legacy_arch.yaml` is the control that would split recipe from
  architecture; it has not been run.
* **Change it**: `configs/server_a5000.yaml`.

## A16. The co-visibility head is not calibrated outside the training distribution

* **Fact, measured**: run over 370 cross-room panorama pairs from a virtual-tour
  dataset, the head's median output is **0.765** — the ZInD positive rate
  (76.5 %) to three decimals — and its correlation with the actual pose error is
  **+0.08**. It is reporting its prior, not a judgement about overlap.
* **Why**: every training pair is intra-`complete_room` (A8), so the head has
  never seen a pair with almost nothing in common and was never asked to say so.
* **Consequence**: `covis_mean` must not be used as a confidence signal on
  out-of-distribution pairs — including for weighting edges in a pose graph,
  which is what `scripts/run_scene_graph.py` does and why its confidences are
  unreliable there.
* **Change it**: train with cross-room pairs mixed in (A8 describes how to
  synthesise them with exact ground truth), which supplies the near-zero-overlap
  negatives the head lacks.
* **Resolved for the model trained with negatives** (`configs/server_c10neg.yaml`,
  A19). On the same 370 tour edges its co-visible fraction separates good
  edges from bad ones: the 142 it calls >= 10 % co-visible have yaw median
  **2.6°** against the existing pipeline (92 % within 10°), the 228 it drops
  84.7° (6 %). The old head ranks them backwards -- its 142 most confident edges
  are *worse* (60.3°) than the rest (19.5°). `run_scene_graph.py
  --min-covis-frac` uses this as a gate; see `docs/PHASE5_NOTES.md`.

## A17. A wall seen from both sides is not co-visible

* **Fact, measured**: the co-visibility rule (A1's definition: radial distances
  agree within `max(0.10, 5 %)`) cannot tell the two faces of one wall apart. In
  ZInD's floor plan a wall between two rooms is a few centimetres thick, inside
  that tolerance, so on 1 630 "co-visible" columns of cross-room pairs, **98 %**
  had the two cameras on opposite sides of the wall. 15 % of cross-room pairs
  scored above 0.10 for this reason alone.
* **Decision**: a column is co-visible only if the other camera faces the hit
  wall from the same side as this one (`targets._same_side`; at a corner, either
  wall meeting there may qualify). Two cameras that genuinely see one wall point
  are both on its inner side, so the rule can only remove false positives.
* **It also touched intra-room pairs**: 3.0 % of co-visible columns changed, 93 %
  of them in pairs of *different partial rooms* -- interior partitions seen from
  both sides inside one complete room. Every measure of agreement with Zillow's
  own score improved (1 457 pairs): r **0.9953 -> 0.9987**, mean |diff|
  0.0208 -> 0.0067, bias +0.0187 -> +0.0022. The change is largest at low
  overlap (-0.036 on average for Zillow scores 0.10-0.30) -- the band the
  `c10neg` run adds.
* **Consequence for comparisons**: the server run's co-visibility and
  correspondence targets carried this bias; its pose targets did not (they come
  from the floor plan). So pose comparisons across the two runs are clean, and
  head metrics are measured against slightly different targets.
* **Test**: `tests/test_negatives.py` (fails with 170 false columns without the rule).

## A18. Rooms connect through annotated doors and openings

* **Decision**: two complete rooms "connect" when a door or opening annotated in
  one lies within 0.5 camera heights of one annotated in the other
  (`pairs.room_connections`).
* **Why**: ZInD annotates every door and opening per pano, as two end points on
  the wall plus (z_bottom, z_top) -- measured: end points lie on the pano's own
  polygon (max 0.0018 camera heights), median door width 0.59 camera heights
  (~0.85 m), bottom at z = -0.996 (the floor). The two annotations of one doorway
  are its two faces, a wall's thickness apart; two different doorways are at
  least a door's width apart. 97 % of complete rooms find at least one
  connection; 52 % of cross-room pano pairs are in connected rooms.
* **What it is for**: a pair in connected rooms may genuinely share content
  through an open door, and ZInD does not say which doors are open. So those
  pairs are never used as negatives. They go to `*_adjacent.csv` instead: exact
  pose, unknown co-visibility -- the ZInD stand-in for the tour data, evaluation
  only.

## A19. Negatives: rooms that share nothing, pose not supervised

* **Decision**: a negative is two panos on one floor in complete rooms that do
  not connect (A18), with geometric co-visibility <= 0.02 after A17 (the residue
  is corner rounding: max 0.010 measured). Sampled per house at 0.4 per positive
  -- the paper's pos:neg of ~2.5. They supervise the co-visibility head (all
  zeros), and the boundary head as usual. Correspondence is masked by
  co-visibility, so it gets nothing automatically.
* **Pose is masked on negatives** -- the direct pose loss, the geometric
  (Procrustes) loss, and all pose metrics. The ground truth exists (the floor
  plan gives it exactly) but no image evidence can recover it, so regressing it
  would teach the decoder to guess from priors. The paper does not say what it
  does here.
* **Not done**: hard negatives in *connected* rooms with the door known to be
  closed. Without door states that cannot be told apart from an open door.
* **Change it**: `--neg-ratio`, `--neg-max-covis` on `build_pair_index.py`;
  `pose_valid` in `dataset.py`.

## A7. Column/row ↔ angle convention

* **Decision**: HorizonNet's convention —
  `θ_j = ((j+0.5)/W − 0.5)·2π`, `b_i = ((i+0.5)/H − 0.5)·π`, floor boundary
  `b = atan2(1, r) > 0`.
* **Why**: ZInD's own `sphere_to_pixel` divides by `W−1` / `H−1` instead of
  `W` / `H`. The two differ by **0.499 px at W=1024** (measured in
  `tests/test_geometry.py`). We follow HorizonNet because the network's column
  grid and the `bon` regression target are defined that way, and reusing
  HorizonNet's parameterisation lets its pretrained layout conventions carry
  over. Both conventions are cross-checked against each other in the tests.
