# Same-room decision comparison: results

Recorded 2026-10-23. Data: 60 Dollhouse houses, 1,232 indoor panoramas, 7,985 same-floor pairs. GT: manual groups + blind review (148 groups, 410 panoramas). Evaluation standard: `EVALUATION_STANDARD.md`. Chinese version: [RESULTS_ZH.md](RESULTS_ZH.md).

## Decision

Adopted the **relaxed GeoVerify wedge rule** as the same-room classifier: a pose found, E inliers ≥ 30, co-visible wedge ≥ 45° on both sides (GitHub main b589709, `verify_pair` direct mode, default parameters). The wedge threshold can be set between 15° and 60°; the expected precision/recall is in the calibration table below.

## Level 1: same-room decision (all pairs)

| Rule | P | R | F1 | TP | FP | FN | Hard-negative FP rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| CoVisPose strict rule | 0.754 | 0.205 | 0.322 | 86 | 28 | 334 | 0.028 |
| CoVisPose loose rule | 0.652 | 0.455 | 0.536 | 191 | 102 | 229 | 0.083 |
| CoVisPose covis_frac ≥ 0.05 | 0.357 | 0.845 | 0.502 | 355 | 638 | 65 | 0.260 |
| CoVisPose covis_frac ≥ 0.10 | 0.408 | 0.821 | 0.545 | 345 | 501 | 75 | 0.236 |
| CoVisPose covis_frac ≥ 0.25 | 0.470 | 0.746 | 0.576 | 313 | 354 | 107 | 0.191 |
| CoVisPose covis_frac ≥ 0.40 | 0.486 | 0.666 | 0.562 | 280 | 296 | 140 | 0.165 |
| CoVisPose covis_frac ≥ 0.60 | 0.535 | 0.553 | 0.544 | 232 | 202 | 188 | 0.123 |
| CoVisPose covis_frac ≥ 0.80 | 0.579 | 0.395 | 0.470 | 166 | 121 | 254 | 0.072 |
| GeoVerify wedge rule (≥ 60°) | 0.530 | 0.400 | 0.456 | 168 | 149 | 252 | 0.101 |
| GeoVerify relaxed wedge ≥ 45° | 0.507 | 0.538 | 0.522 | 226 | 219 | 194 | 0.135 |
| GeoVerify relaxed wedge ≥ 30° | 0.414 | 0.667 | 0.511 | 280 | 397 | 140 | 0.239 |
| GeoVerify door rule (22 houses) | 0.335 | 0.839 | 0.479 | 119 | 236 | 23 | 0.205 |
| GeoVerify floor n_inlier ≥ 300 * | 0.216 | 0.756 | 0.336 | 331 | 1202 | 107 | 0.671 |
| GeoVerify floor n_inlier ≥ 500 * | 0.247 | 0.678 | 0.363 | 297 | 903 | 141 | 0.581 |
| GeoVerify floor n_inlier ≥ 750 * | 0.267 | 0.575 | 0.365 | 252 | 692 | 186 | 0.467 |
| GeoVerify floor n_inlier ≥ 1000 * | 0.276 | 0.454 | 0.344 | 199 | 522 | 239 | 0.374 |
| GeoVerify floor n_inlier ≥ 1500 * | 0.289 | 0.286 | 0.288 | 125 | 308 | 313 | 0.236 |

\* floor has scores only on stratified-sampled pairs; weighted estimates. The door rule covers only 22 houses.

## Level 2: clustering (connected components)

| Rule | Cluster P | Cluster R | Cluster F1 | Room P | Room R | Room F1 | Exact / predicted rooms |
|---|---:|---:|---:|---:|---:|---:|---|
| CoVisPose strict rule | 0.78 | 0.19 | 0.30 | 0.44 | 0.28 | 0.34 | 41/93 |
| CoVisPose loose rule | 0.60 | 0.51 | 0.55 | 0.44 | 0.41 | 0.42 | 61/140 |
| CoVisPose covis_frac ≥ 0.05 | 0.22 | 0.95 | 0.36 | 0.33 | 0.32 | 0.33 | 48/146 |
| CoVisPose covis_frac ≥ 0.10 | 0.26 | 0.94 | 0.41 | 0.34 | 0.36 | 0.35 | 54/158 |
| CoVisPose covis_frac ≥ 0.25 | 0.33 | 0.90 | 0.48 | 0.38 | 0.43 | 0.41 | 64/167 |
| CoVisPose covis_frac ≥ 0.40 | 0.37 | 0.87 | 0.52 | 0.40 | 0.46 | 0.43 | 68/168 |
| CoVisPose covis_frac ≥ 0.60 | 0.44 | 0.72 | 0.54 | 0.41 | 0.46 | 0.43 | 68/166 |
| CoVisPose covis_frac ≥ 0.80 | 0.43 | 0.44 | 0.43 | 0.38 | 0.36 | 0.37 | 53/140 |
| GeoVerify wedge rule (≥ 60°) | 0.41 | 0.47 | 0.44 | 0.34 | 0.37 | 0.35 | 55/164 |
| GeoVerify relaxed wedge ≥ 45° | 0.38 | 0.61 | 0.47 | 0.33 | 0.43 | 0.37 | 64/194 |
| GeoVerify relaxed wedge ≥ 30° | 0.27 | 0.78 | 0.40 | 0.26 | 0.36 | 0.30 | 53/207 |
| GeoVerify door rule (22 houses) | 0.14 | 0.92 | 0.25 | 0.18 | 0.18 | 0.18 | 11/62 |

## Calibration table of the adopted classifier (inliers ≥ 30, varying wedge threshold)

| Both wedges ≥ | P | R | F1 | Pairs judged same | Hard-negative FP rate |
|---:|---:|---:|---:|---:|---:|
| 0° | 0.193 | 0.925 | 0.319 | 2014 | 0.687 |
| 5° | 0.216 | 0.920 | 0.350 | 1787 | 0.649 |
| 10° | 0.260 | 0.885 | 0.402 | 1427 | 0.565 |
| 15° | 0.308 | 0.841 | 0.451 | 1147 | 0.452 |
| 20° | 0.347 | 0.769 | 0.478 | 931 | 0.371 |
| 25° | 0.382 | 0.719 | 0.499 | 789 | 0.300 |
| 30° | 0.414 | 0.667 | 0.511 | 677 | 0.239 |
| 35° | 0.460 | 0.632 | 0.532 | 577 | 0.193 |
| 40° | 0.490 | 0.589 | 0.535 | 504 | 0.158 |
| 45° | 0.507 | 0.538 | 0.522 | 445 | 0.135 |
| 50° | 0.521 | 0.495 | 0.507 | 399 | 0.121 |
| 55° | 0.542 | 0.445 | 0.489 | 345 | 0.104 |
| 60° | 0.530 | 0.400 | 0.456 | 317 | 0.101 |
| 65° | 0.528 | 0.355 | 0.424 | 282 | 0.088 |
| 70° | 0.523 | 0.321 | 0.398 | 258 | 0.081 |
| 75° | 0.509 | 0.276 | 0.358 | 228 | 0.074 |
| 80° | 0.514 | 0.260 | 0.345 | 212 | 0.069 |
| 85° | 0.495 | 0.221 | 0.306 | 188 | 0.067 |
| 90° | 0.488 | 0.198 | 0.281 | 170 | 0.063 |

## Blind review

- 683 pairs reviewed (judged same room by some method, originally labelled negative); 97 confirmed same room. Pairs flagged only by floor 750 were sampled (p = 0.29).
- Control pairs: all 25 originally labelled different room agreed; 8 of the 25 originally labelled same room were judged different room (the original annotator ym and the reviewer differ on hallways, entrances and open spaces).
- The groups went from 104 to 148; 30 pairs judged different room in the review still fall into one group through chaining.

## Known limitations

- The threshold was chosen on the same data, so the numbers may be optimistic; confirm with leave-one-house-out cross-validation before adopting.
- Pairs that no method judged same room were not checked by a person, so recall may be slightly overestimated.
- Above a 45° wedge the precision stays around 0.5; the main errors come from open spaces and differing annotation standards.
