# Door-aware wedge: correcting the GeoVerify wedge rule with SAM3 door-opening masks

Recorded 2026-10-27. Data and GT as in `RESULTS.md` (60 houses, 7,985 same-floor pairs, GT after the blind review). Chinese version: [RESULTS_open_wedge_ZH.md](RESULTS_open_wedge_ZH.md).

## Method

1. Re-ran the GeoVerify geometric step on the CPU from the RoMa matches in `work/roma_cache_direct`, keeping the per-point inliers
   (`scripts/recompute_inliers_cpu.py` → `work/inliers_direct/`; 1,973 pairs with a pose and ≥ 30 inliers, 1 pair whose inlier count differs from the original scan).
2. SAM3 ran the "open doorway" prompt on every panorama (`sam3-pano-seg/scripts/run_samroom_openings.ps1`,
   pano-vote, threshold 0.0; same settings as `group_wdo_masks_alt/frame_opendoorway`).
3. An inlier is dropped when it lies inside a door mask (dilated by about 0.7°) on side A or B; with ≥ 30 inliers left the 80% wedge is recomputed and both sides ≥ threshold means same room.
   (`scripts/eval_open_wedge.py`, rule name "去門楔形 or dil4", i.e. door-aware wedge, dilation 4.)

## Choosing the prompt (17 houses, 1,747 pairs, leave-one-house-out pair F1)

| Door mask | F1 |
|---|---:|
| No door mask (current wedge) | 0.592 |
| SAM3 union of door frame + open doorway + doorway + glass door | 0.597 |
| SAM3 glass door | 0.604 |
| SAM3 interior architectural door | 0.613 |
| SAM3 open doorway + doorway + door frame | 0.617 |
| SAM3 door frame | 0.620 |
| **SAM3 open doorway** | **0.652** |
| Sub-project 1 door semantics (bounding boxes, only 22 houses) | 0.660 |

"door"-type prompts mostly mark closed door leaves and pictures on the wall, so they remove same-room inliers; only "open doorway" marks the openings one can see through.

## Results on 60 houses (open doorway)

### Pair level

| Rule | Threshold | P | R | F1 | FP | Hard-negative FP rate |
|---|---:|---:|---:|---:|---:|---:|
| Current wedge | 40° | 0.490 | 0.589 | 0.535 | 257 | 0.158 |
| Current wedge | 45° | 0.507 | 0.538 | 0.522 | 219 | 0.135 |
| **Door-aware wedge** | **40°** | **0.559** | **0.584** | **0.571** | **194** | **0.122** |

- Leave-one-house-out: current F1 0.535 → door-aware **0.569** (P 0.490 → 0.557, R 0.589 → 0.582); 59 of the 60 folds chose "door-aware wedge, dil4 @ 40°", so the choice is stable.
- Precision at equal recall: at R ≥ 0.5, 0.507 → 0.572; at R ≥ 0.7, 0.382 → 0.445; at R ≥ 0.8, 0.308 → 0.356.
- Compared with the current 40°: 71 false positives removed (37 of them hotspot-adjacent rooms behind a door), 7 new false positives; 6 same-room pairs lost, 4 gained.
- Per-house F1: 29 houses better, 6 worse, 25 unchanged. The worse ones mostly have only 1–2 same-room pairs.

### Cluster level (connected components)

| Rule | Threshold | Cluster P | Cluster R | Cluster F1 | Rooms recovered exactly | Predicted rooms |
|---|---:|---:|---:|---:|---:|---:|
| Current wedge | 45° (current default) | 0.383 | 0.615 | 0.472 | 64/148 | 194 |
| Current wedge | 40° | 0.355 | 0.675 | 0.466 | 67/148 | 197 |
| Door-aware wedge | 35° | — | — | 0.513 | 66/148 | — |
| **Door-aware wedge** | **40°** | **0.421** | **0.631** | **0.505** | **70/148** | **179** |
| Door-aware wedge | 45° | — | — | 0.510 | 70/148 | — |

## Conclusions

- Door semantics should be used after GeoVerify, to drop inliers "seen through a doorway", not as a connectivity pre-filter.
- The SAM3 prompt is decisive: only "open doorway" works.
- Recommended new default: door-aware wedge ≥ 40° (inliers ≥ 30). The review list has about as many candidates as the current 45° (about 440 vs 445 pairs), while precision rises from about 0.51 to 0.56 and recall from about 0.54 to 0.58.

## Limitations

- The threshold and the dilation were chosen on the same data; the leave-one-house-out figure (F1 0.569) is the more reliable one.
- Open spaces (living/dining/kitchen connected) have no doorway; the method cannot handle them and they remain the main source of false positives.
- New data needs one SAM3 run per panorama (about 3.3 s per panorama on an RTX GPU).

## Artefacts (same_room_benchmark project, not in this package)

- `scripts/recompute_inliers_cpu.py`, `scripts/cache_open_masks.py`, `scripts/eval_open_wedge.py`, `scripts/sam3_gate_experiment.py`
- `work/inliers_direct/`, `work/open_masks_*/`, `work/open_wedge_sam3_adj.json`, `work/open_wedge_60.txt`
- SAM3 masks: `C:\sam3-pano-seg\outputs\samroom_opendoorway\` (828 panoramas) + `group_wdo_masks_alt\frame_opendoorway\` (404 panoramas)

## Clustering accuracy (added 2026-10-27)

`scripts/cluster_open_wedge.py`: uses the scores of `room_candidate_finder/scans/dollhouse` directly with `finder.scores.passes` (the candidates the review app actually lists) and the evaluation of `cluster_metrics.py` (60 houses, 1,232 panoramas, 148 GT rooms, unweighted). The plain wedge at 45° matches level 2 of `RESULTS.md` above.

### Automatic clustering (every listed candidate taken as same room, connected components)

| Rule | Candidate pairs | Cluster P | Cluster R | Cluster F1 | Exact / predicted rooms | Room F1 | B-cubed F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plain wedge 45° (old default) | 446 | 0.383 | 0.615 | 0.472 | 64/194 | 0.374 | 0.768 |
| Plain wedge 40° | 500 | 0.355 | 0.675 | 0.466 | 67/197 | 0.388 | 0.756 |
| Door-aware wedge 35° | 498 | 0.411 | 0.720 | **0.523** | 70/187 | 0.418 | 0.781 |
| **Door-aware wedge 40° (new default)** | 433 | 0.423 | 0.637 | 0.508 | **73/184** | 0.440 | 0.791 |
| Door-aware wedge 45° | 387 | 0.451 | 0.599 | 0.514 | 72/178 | **0.442** | **0.799** |

Leave-one-house-out (threshold chosen by the cluster F1 of the other 59 houses): plain wedge F1 0.439, 61/148 rooms exact; door-aware wedge F1 **0.511**, **69/148** exact (59 folds chose 35°).
Error types: merged predicted rooms (containing ≥ 2 GT rooms) 107 → 89 (door-aware 40°); split GT rooms 57 → 46.

### Upper bound after review (the reviewer judges every listed pair correctly)

| Rule | Pairs to review | Cluster R | Exact / predicted rooms | Room F1 | B-cubed F1 |
|---|---:|---:|---:|---:|---:|
| Plain wedge 45° | 446 | 0.605 | 90/119 | 0.674 | 0.872 |
| Door-aware wedge 40° | 433 | 0.635 | 101/128 | 0.732 | 0.891 |
| Plain wedge 30° | 660 | 0.782 | 115/133 | 0.819 | 0.934 |
| Door-aware wedge 30° | 570 | 0.782 | 114/131 | 0.817 | 0.933 |
| Plain wedge 20° | 933 | 0.931 | 131/137 | 0.919 | 0.971 |
| Door-aware wedge 20° | 772 | 0.903 | 128/140 | 0.889 | 0.965 |

- For the same number of reviewed pairs the door-aware wedge recovers more rooms (about 440 pairs: 90 → 101 rooms); to reach the same cluster recall it needs about 13% fewer reviews (0.78: 660 → 570 pairs).
- Without manual review, at most about half of the rooms are recovered exactly (73/148) and the precision of the predicted rooms is about 0.4, so automatic clustering still needs a person to confirm.
