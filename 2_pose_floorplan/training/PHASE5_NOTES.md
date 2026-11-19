# Phase 5 notes — the server run

`configs/server_a5000.yaml`, 3 GPUs (2× RTX A5000 24 GB + 1× RTX A6000 48 GB)
under DDP, **16 epochs in 17 h 56 m**, peak VRAM 16.33 GiB of 24, zero NaN /
divergence / spill events. Checkpoint `ckpt/server_r50_c1024/best.pt`
(epoch 15, EMA weights) — the validation minimum *and* the last epoch.

What changed relative to the RTX 1070 run, and why, is
[`SERVER_OPTIMIZATION.md`](SERVER_OPTIMIZATION.md). This file is what happened
when it ran.

---

## Headline: the test split

Full official test split, 26 267 pairs, fp32, EMA weights, single forward pass.

| | **RTX 1070** | **server** | change |
|---|---|---|---|
| co-visibility IoU | 0.8887 | **0.9276** | +0.0390 |
| co-visibility accuracy | 0.9086 | **0.9418** | +0.0332 |
| correspondence | 7.55° | **3.94°** | **−48 %** |
| floor boundary | 1.89° | **1.64°** | −13 % |
| decoder, t median | 0.0944 | **0.0616** | −35 % (14 cm → **9.2 cm**) |
| decoder, t mean | 0.2241 | **0.1428** | −36 % |
| decoder, yaw median | 1.671° | **0.869°** | **−48 %** |
| decoder, yaw mean | 5.25° | **3.05°** | −42 % |
| decoder, acc@(0.1, 1°) | 0.2406 | **0.4455** | **+20.5 pts** |
| decoder, acc@(0.5, 10°) | 0.8911 | **0.9549** | +6.4 pts |
| decoder, acc@(1.0, 20°) | 0.9538 | **0.9754** | +2.2 pts |
| RANSAC @0.30, acc@(0.5, 10°) | 0.8498 | **0.9464** | +9.7 pts |

With 8-view test-time augmentation the server model reaches t median **0.0585**
(≈ 8.8 cm), yaw median **0.754°** and acc@(0.1, 1°) **0.4996** — see §
"Validation-only decisions" for why TTA only moves the strict metric.

Translation is in camera heights, 1 unit ≈ 1.5 m.

### The two weakest buckets of the 1070 run improved most

Phase 4 singled out low overlap and long baselines. Both gained ~18 points:

| bucket | 1070 | server |
|---|---|---|
| `covis_score` [0.30, 0.50) | 0.6821 | **0.8622** |
| baseline \|t\| [5, 100) camera heights | 0.5218 | **0.7052** |

### Three pose estimators (test)

| estimator | t median | yaw median | acc@(0.1, 1°) | acc@(0.5, 10°) |
|---|---|---|---|---|
| conv→attention decoder | 0.0616 | 0.869° | 0.4455 | **0.9549** |
| RANSAC @0.30 | 0.0635 | 0.869° | 0.4406 | 0.9464 |
| **geometric (Procrustes)** | **0.0600** | **0.804°** | **0.4726** | 0.9540 |

The differentiable geometric branch is the most *precise* estimator — it wins
the strict threshold by 2.7 points and the medians — while the decoder wins the
loose threshold, i.e. it produces fewer catastrophic failures. That split is the
expected one: the geometric fit reads the per-column heads, so it inherits both
their precision and their occasional gross errors.

RANSAC gained the most of the three (+9.7 points). That is Phase 4's A14b
conclusion confirmed from the other direction: what limited RANSAC was never
outlier contamination but the accuracy of the heads generating its points, and
those heads got much better.

Raw results: `outputs/eval/{val,test}.json` on the server.

---

## The run itself

### An aborted first attempt: lr 5e-4 was too high

The config's first lr came from √-scaling the 1070 run's 3e-4 to the 3× larger
batch. It destabilised as soon as warmup ended:

| step | lr | total | bon | grad norm | clipped |
|---|---|---|---|---|---|
| 1400 | 3.5e-4 | 6.64 | **0.049** | 5.9 | 40 % |
| 1800 | 4.5e-4 | 6.72 | 0.066 | **39.0** | 94 % |
| 2600 | 5.0e-4 | 6.71 | 0.112 | **138.7** | 100 % |
| 3800 | 5.0e-4 | 6.69 | 0.112 | **172.8** | 100 % |

The boundary head *regressed* to twice its best, gradient norms spiked into the
hundreds, and co-visibility sat at the majority class (0.7651) through step
4 200. Restarted at **lr 3e-4 with warmup 4 000**, which held: grad norms stayed
at 10–25 for the whole run, the same regime the 1070 run trained in.

Cost of the abort: 35 minutes. The evidence is kept in the config comment so
the next person does not re-derive 5e-4 from the same scaling argument.

### Per-epoch (EMA weights, full 24 912-pair validation split)

| epoch | train | val | EMA score | IoU | corr | bon | t med | yaw med | acc(.5,10) | geo acc |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4.998 | 6.126 | 5.8288 | 0.7786 | 52.94 | 4.07 | 1.0636 | 63.580 | 0.0184 | 0.0942 |
| 1 | 2.844 | 2.687 | 2.8116 | 0.7967 | 25.18 | 3.88 | 0.4436 | 11.592 | 0.3285 | 0.4487 |
| 2 | 1.923 | 1.935 | 1.8909 | 0.8213 | 16.79 | 3.59 | 0.2625 | 6.407 | 0.5847 | 0.6715 |
| 3 | 1.458 | 1.552 | 1.4587 | 0.8388 | 13.14 | 3.09 | 0.1823 | 4.355 | 0.7426 | 0.7891 |
| 4 | 1.214 | 1.349 | 1.2032 | 0.8542 | 10.78 | 2.60 | 0.1409 | 3.262 | 0.8241 | 0.8562 |
| 5 | 0.996 | 1.115 | 1.0306 | 0.8675 | 9.21 | 2.30 | 0.1153 | 2.504 | 0.8824 | 0.8968 |
| 6 | 0.761 | 0.949 | 0.8869 | 0.8875 | 7.90 | 2.11 | 0.0982 | 1.966 | 0.9198 | 0.9255 |
| 7 | 0.663 | 0.845 | 0.7698 | 0.9047 | 6.13 | 1.98 | 0.0856 | 1.533 | 0.9394 | 0.9402 |
| 8 | 0.542 | 0.755 | 0.6961 | 0.9143 | 4.93 | 1.90 | 0.0781 | 1.306 | 0.9496 | 0.9485 |
| 9 | 0.490 | 0.706 | 0.6524 | 0.9198 | 4.42 | 1.82 | 0.0726 | 1.140 | 0.9555 | 0.9544 |
| 10 | 0.423 | 0.665 | 0.6229 | 0.9235 | 4.13 | 1.77 | 0.0689 | 1.055 | 0.9583 | 0.9565 |
| 11 | 0.387 | 0.644 | 0.6095 | 0.9258 | 3.97 | 1.74 | 0.0674 | 0.964 | 0.9597 | 0.9575 |
| 12 | 0.350 | 0.621 | 0.6072 | 0.9269 | 3.90 | 1.72 | 0.0669 | 0.927 | 0.9595 | 0.9579 |
| 13 | 0.322 | 0.610 | 0.6008 | 0.9279 | 3.83 | 1.71 | 0.0663 | 0.891 | 0.9605 | 0.9579 |
| 14 | 0.313 | 0.604 | 0.5995 | 0.9282 | 3.80 | 1.71 | 0.0663 | 0.884 | 0.9602 | 0.9586 |
| **15** | **0.292** | **0.598** | **0.5979** | **0.9283** | **3.79** | **1.70** | **0.0664** | **0.871** | **0.9603** | **0.9589** |

**Validation improved every single epoch**, so `best.pt` is the last one. The
train/val gap widened from 0.19 (epoch 6) to 0.31 (epoch 15) without
validation ever regressing — the same pattern the 1070 run showed, and the
reason no early stopping was needed on top of best-checkpoint selection.

Two things the curve says about the schedule: epochs 12–15 bought 0.009 of
score between them, so 16 epochs was, if anything, one or two more than needed;
and the geometric estimator tracked the decoder to within 0.002 accuracy from
epoch 9 onwards, having *led* it by 12 points at epoch 1.

### Measured on this hardware

| | |
|---|---|
| throughput | 58 pairs/s (3 GPUs, 411–420 ms/step at 24 pairs/step) |
| epoch | 66–68 min including full validation of raw **and** EMA weights |
| batch | 8 per GPU; 13.3 GiB reserved, and throughput is flat past 8 (23.1 vs 23.7 pairs/s at 12) |
| loader | 6 workers/rank; one loader alone delivers 141 pairs/s at 4 workers vs the 23 pairs/s a GPU consumes |
| peak VRAM | 16.33 GiB of 24 |

A5000-only figures; the A6000 in the same DDP group is bounded by them.

### EMA

EMA was worse than the raw weights while the model moved fast (epoch 1: corr
25.18° vs 22.67°) and better once it settled, converging to a dead heat at the
end (0.5979 vs 0.5983). Its value here was mostly in giving checkpoint
selection a smoother score, not in the final number.

---

## Validation-only decisions

Both were made on the val split and only then applied to test, as in Phase 4.

**RANSAC threshold: 0.30 again.** Swept on 1 500 val pairs. The accuracy surface
is flat from 0.30 to 1.00 (0.956–0.961); 0.40 scored 0.5 points higher, which is
inside the noise of 1 500 pairs, so the middle of the plateau was kept — where
the inlier rate is 90 % and the estimator is still rejecting something.

**Test-time augmentation: worth it for precision.** On 4 000 val pairs, 4 column
rolls × mirror (8 views):

| | no TTA | TTA |
|---|---|---|
| correspondence | 3.756° | **3.534°** |
| boundary | 1.728° | **1.672°** |
| decoder yaw median | 0.881° | **0.757°** |
| decoder acc@(0.1, 1°) | 0.4343 | **0.4925** |
| decoder acc@(0.5, 10°) | 0.9607 | 0.9630 |

The loose threshold is saturated, so TTA buys almost nothing there; the strict
one gains 5.8 points.

**On test, TTA delivered what val said it would** (26 267 pairs, 4 rolls ×
mirror = 8 forward passes per pair, ~90 min vs ~11 min):

| test | single pass | **8-view TTA** | change |
|---|---|---|---|
| co-visibility IoU | 0.9276 | **0.9314** | +0.0038 |
| correspondence | 3.94° | **3.68°** | −6.5 % |
| floor boundary | 1.64° | **1.58°** | −3.2 % |
| decoder, t median | 0.0616 | **0.0585** | −5.0 % (≈ 8.8 cm) |
| decoder, yaw median | 0.869° | **0.754°** | −13.2 % |
| decoder, yaw mean | 3.05° | **2.91°** | −4.8 % |
| decoder, acc@(0.1, 1°) | 0.4455 | **0.4996** | **+5.4 pts** |
| decoder, acc@(0.5, 10°) | 0.9549 | 0.9548 | −0.0001 |
| decoder, acc@(1.0, 20°) | 0.9754 | 0.9753 | −0.0001 |

The predicted +5.8 points on the strict threshold came in at +5.4, and the loose
thresholds moved by 0.01 points, exactly as the val sweep implied. **TTA is a
precision tool here, not an accuracy one**: it sharpens pairs the model already
gets roughly right and does nothing for the ones it gets wrong.

With TTA the decoder also overtakes the geometric branch on the strict
threshold (0.4996 vs 0.4977) — averaging over views removes precisely the
angular noise the geometric fit was better at suppressing:

| estimator (TTA, test) | t median | yaw median | acc@(0.1, 1°) | acc@(0.5, 10°) |
|---|---|---|---|---|
| **attention decoder** | 0.0585 | **0.754°** | **0.4996** | **0.9548** |
| RANSAC @0.30 | 0.0618 | 0.835° | 0.4582 | 0.9468 |
| geometric (Procrustes) | **0.0583** | 0.762° | 0.4977 | 0.9538 |

Whether the 8× inference cost is worth 5 points of strict accuracy is an
application decision; the single-pass numbers are the ones to quote for
throughput-bound use.

---

## Deployment test: a different dataset, and what it exposed

The checkpoint was also run over a 42-scene virtual-tour dataset
(`scripts/run_scene_graph.py`), whose panorama graph comes from the tour's
HOTSPOT links. Result: **it does not transfer**, for a reason this repo had
already written down.

* **Every one of the 744 hotspot links joins two different rooms.** The training
  pairs are 100 % intra-`complete_room` (assumption A8). This is exactly the
  case A8 says nothing here should be read as evidence for.
* **The co-visibility head is uninformative out of distribution.** Its median
  output on those pairs is 0.765 — the ZInD positive rate to three decimals. Its
  correlation with actual pose error is **+0.08**, so it cannot be used as a
  confidence signal there.
* **Translation survives better than rotation.** Against the tour's own
  geometry: translation direction median 13.8°, magnitude error 0.50 camera
  heights with no scale bias (best global scale 1.00) — but yaw median 44.9°,
  and the errors cluster at multiples of 90° (error mod 90° has median 9.3°,
  52 % within 10°). Rectangular rooms look similar under a 90° turn, and nothing
  in training taught the model to break that tie across a doorway.
* Scene-level verdicts: 14 usable, 15 marginal, 13 unreliable
  (`scripts/validate_scenes.py`).

One convention finding worth keeping: that dataset's panoramas run the opposite
way round in azimuth, and **yaw alone cannot detect it** — yaw negates under
either mirror axis. It took the translation direction to identify which mirror
was right (negate y: hotspot direction error 30.3° vs 149.7° for negating x).

The obvious next experiment is the one A8 already describes: synthesise
cross-room pairs in ZInD — the floor-plan poses make exact ground truth
available — and retrain with them mixed in. That would also give the
co-visibility head the near-zero-overlap negatives it has never seen.


---

## Tour data again, with the model trained on negatives

`ckpt/server_c10neg/epoch_015.pt` (trained on co-visibility >= 0.10 plus
zero-overlap negatives, `docs/PAPER_COMPARISON.md` section 5), over the same 42
scenes and 370 hotspot edges, with a co-visibility gate:

    python scripts/run_scene_graph.py --ckpt ckpt/server_c10neg/epoch_015.pt         --min-covis-frac 0.10 --out-name covispose_c10neg --summary covispose_c10neg_scenes.json
    python scripts/compare_tour_runs.py --runs covispose,covispose_old_notta,covispose_c10neg

No TTA, so the old model was re-run without TTA too (`covispose_old_notta`) to
match; the original TTA run stays in `covispose/`. Each edge's relative pose is
scored before any solve, against the two references `validate_scenes.py` uses.

| 370 edges | hotspot direction median / < 30° | yaw vs existing pipeline median / < 10° (n = 129) |
|---|---|---|
| old model, TTA | 30.3° / 49.6 % | 44.9° / 38.0 % |
| old model, no TTA | 30.1° / 50.0 % | 40.9° / 37.2 % |
| new model, no TTA | 30.6° / 49.3 % | 35.2° / 40.3 % |

**Over all edges the new model is no better** -- expected: two thirds of these
edges cross a doorway, and neither model was trained to do that.

**What changed is that it knows which edges those are:**

| new model | edges | hotspot direction median / < 30° | yaw median / < 10° |
|---|---|---|---|
| kept (co-visible >= 10 %) | 142 | **13.0°** / 73.9 % | **2.6°** / **92.2 %** (n = 51) |
| dropped | 228 | 57.3° / 34.0 % | 84.7° / 6.4 % (n = 78) |
| *old model, its 142 most confident* | 142 | 50.0° / 34.9 % | 60.3° / 26.3 % (n = 38) |

217 of the 370 edges get < 2 % predicted co-visibility, and their poses are
noise (yaw median 84.7°). Above 10 % the poses are good. The old model's
co-visibility ranks edges backwards. On the 142 kept edges, all three runs are
about equally accurate (yaw 2.2-2.8°) -- the gain is selection, not better
poses.

**The cost is connectivity.** With 142 of 370 edges, no scene stays in one
piece: the largest component holds a median 31 % of a scene's panoramas, and 6
scenes keep no edge at all. Each component is solved in its own frame and the
floor plans draw them side by side (`<scene>/covispose_c10neg/floorplan.png`).
The plans are *correct in pieces* instead of *wrong in one piece*.

Joining the pieces needs exactly the edges that are dropped: pairs that see
each other through a doorway. That is the door-crossing positive the paper
trained with and this model never saw (A8, stage 2).
