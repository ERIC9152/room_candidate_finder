# HorizonNet outline evaluation on 404 panoramas

Evaluated 2026-11-05. Chinese version: [REPORT_ZH.md](REPORT_ZH.md). The scoring plan was written before looking at any result: [PLAN.md](PLAN.md) (English translation; the pre-registered original is [PLAN_ZH.md](PLAN_ZH.md), sha256 `6151bd7c…`). Parts marked "post-hoc diagnosis" were added after seeing the main result and are descriptive only.

## 1. What was done

| Item | Details |
|---|---|
| Prediction | The package's own `pipeline/horizonnet.py` (official `preprocess.py` vanishing-point alignment → `inference.py`, MP3D weights → corners mapped back to the original image), the same code as the production pipeline |
| Input | 404 Dollhouse panoramas, originals 8192×4096, resized to 1024×512 with BICUBIC (what `preprocess.py` does internally; identical result) |
| Ground truth | `room-stitching-systemV3\wall-solver-point-edge\data\group\*\layout_gt\*.txt` (latest version; compared with V3 `data\group`, 15 panoramas were re-annotated in September), 22 houses, 42 floors, each panorama with its own room outline (PLAN.md says "24 houses"; a typo that does not affect the scoring) |
| Run | Cloud CPU (2 cores). Spot check on one panorama: the corners differ by at most 0.02 px from the 10/26 production result on an RTX 3060, so this evaluates the production output |
| Metrics | 2D IoU, 3D IoU, depth RMSE and δ1 from HorizonNet's official `eval_general.py`, plus the per-column relative wall-distance error |
| Statistics | 1,000 bootstrap resamples by house, 95% intervals |

Compared with the earlier evaluation: the 10/21 run covered only 94 panoramas and used the "stitched group ground truth", with a mean IoU of 0.567. This run compares each panorama with its own ground truth and covers all 404. The definitions differ, so the numbers cannot be compared directly.

## 2. Main results

All 404 inferences succeeded; no missing or invalid outlines.

| Metric | This run (Dollhouse, 404) | Official HorizonNet: MatterportLayout | Official HorizonNet: ZInD |
|---|---:|---:|---:|
| **2D IoU** | **51.4%** (95% interval 46.7–56.5; median 52.3%) | 81.71% | 90.44% |
| 3D IoU | 44.9% (39.2–51.0) | 79.11% | 88.59% |
| Depth RMSE | 0.779 | 0.197 | 0.123 |
| δ1 | 0.586 | 0.929 | 0.957 |
| Per-column relative wall-distance error (median) | 21.9% (signed +3.7%) | — | — |
| Same corner count as the ground truth | 15.8% | — | — |

The official figures are from Table 2 of the DOPNet paper (arXiv 2312.16268).

2D IoU distribution (bins of 0.1, from 0–0.1 to 0.9–1.0): 16, 22, 50, 70, 32, 54, 50, 45, 54, 11 panoramas.

## 3. Post-hoc diagnosis: the ground truth has house-wide vertical offsets

### 3.1 Finding

The error direction is very consistent within a house: for example 58498 is too close everywhere, 59868 too far. For each panorama we computed "the overall scale of the predicted outline that best fits the ground truth":

- **48% of the variance of the best scale is explained by "which house".**
- The ground-truth ceiling height (in camera heights) is 2.79 in 58498 and 2.61 in 58546, while most other houses are at 1.7–2.1. HorizonNet predicts 1.49 and 1.77 for these two houses.

### 3.2 Looking at the images

Ground truth (green) and HorizonNet (magenta) wall–ceiling and wall–floor boundaries drawn on the panoramas:

| House | Best scale | Observation |
|---|---:|---|
| 58472 (`fig/58472_b06d-4fab.png`) | 1.00 | The two nearly coincide, 2D IoU 0.95 |
| 58498 (`fig/58498_e0d0-47c7.png`, `58498_a2ed-4ea4.png`) | 1.62 | **Magenta follows the real boundaries; both green lines are pulled towards the horizon and the floor line lies on the wall. The ground truth is wrong** |
| 59868 (`fig/59868_e86d-478d.png`) | 0.71 | The opposite: the green floor line lies on the carpet and the ceiling line on the ceiling, so the ground-truth room is too small; magenta follows the visible skirting |
| 58599 (`fig/58599_9524-492e.png`) | 0.71 | The right half as in 59868; on the left HorizonNet is blocked by a sofa and meets the opening to the dining room, so neither is accurate |

The azimuths (horizontal positions) of the ground-truth corners are usually right; the error is mainly vertical, i.e. the overall scale. This is what one expects when "the ground truth was cut from a floor plan and projected back into the image with an assumed camera height" and that height was wrong.

### 3.3 Shape only (scale removed)

| Metric | Value |
|---|---:|
| Absolute 2D IoU (main metric) | 0.514 |
| **2D IoU after the best scale (shape)** | **0.668** (95% interval 0.639–0.702; median 0.772) |
| 2D IoU after area matching (shape, another way) | 0.648 |
| Absolute 2D IoU without the 7 houses whose best scale is outside 0.8–1.25 | 0.580 (95% interval 0.545–0.627; 15 houses, 255 panoramas) |

The 7 excluded houses: 58498, 58546, 58553, 58558, 58599, 59863, 59868. This cut is descriptive only and does not replace correcting the ground truth.

### 3.4 By room type and ground-truth corner count (≥ 8 panoramas)

| Room type | Panoramas | Absolute 2D IoU | Shape 2D IoU |
|---|---:|---:|---:|
| Bedroom | 85 | 0.611 | 0.783 |
| Garage | 26 | 0.636 | 0.759 |
| Entrance | 11 | 0.418 | 0.722 |
| Storage | 23 | 0.515 | 0.703 |
| Bathroom | 67 | 0.538 | 0.700 |
| Uncertain | 24 | 0.586 | 0.679 |
| Laundry | 13 | 0.525 | 0.679 |
| Living room | 86 | 0.397 | 0.540 |
| Kitchen | 27 | 0.454 | 0.539 |
| Hallway | 12 | 0.390 | 0.501 |
| Dining room | 8 | 0.241 | 0.483 |

(Room types come from the Chinese labels in `rooms.csv`; `results.json` keeps the original labels.)

| Ground-truth corners | Panoramas | Absolute 2D IoU | Shape 2D IoU |
|---|---:|---:|---:|
| 4 | 41 | 0.389 | 0.506 |
| 6 | 114 | 0.538 | 0.684 |
| 8 | 45 | 0.561 | 0.757 |
| 10+ | 23 | 0.467 | 0.617 |
| Odd (5, 7, 9…: non-Manhattan or clipped ground-truth pieces) | 181 | 0.520 | 0.680 |

HorizonNet outputs only Manhattan rooms (even corner counts). For 181 of the 404 panoramas the ground truth has an odd number of corners, so the counts can never match.

### 3.5 HorizonNet's own failures

The worst panoramas after removing scale (`fig/worst_*.png`):

- **Open living room** (58582, `worst_d9fa-4081.png`): the outline cuts through the TV cabinet and sofa and draws a room that does not exist.
- **Small space seen from the doorway** (58533 bathroom, `worst_8a6f-4f2e.png`): the outline goes through the doorway into the hallway and kitchen outside.

Both are the known limitations in README §18 (open spaces, through doors). Living rooms, kitchens, dining rooms and hallways reach a shape IoU of only 0.48–0.54; bedrooms and garages 0.76–0.78.

## 4. Conclusions

1. **Shape**: HorizonNet's outlines are broadly right for ordinary rooms (bedrooms, garages, bathrooms, storage), shape IoU 0.70–0.78, and clearly worse for open living rooms, kitchens, dining rooms and hallways (about 0.5). The overall shape IoU is 0.668, still below the official 0.817 on MatterportLayout, mainly because Dollhouse has a lot of furniture and open space, and 45% of the ground truth is not Manhattan.
2. **Absolute scale**: the absolute 2D IoU of 0.514 is an **underestimate**. For at least three houses (58498, 59868, 58599) a visual check confirmed house-wide vertical offsets in the ground truth; judging by the scale factor, up to 7 houses may be affected.
3. **Impact on earlier results**: this `layout_gt` was also the reference of several earlier evaluations, including the 10/21 stitched IoU of 0.567, the PSMNet comparison and the choice of the plan rule (union for several panoramas, farther-of for one). All of them may be affected by the same offsets.
4. **Recommendation**: correct the vertical position of the ground truth of these 7 houses first (e.g. re-annotate from the visible skirting, or re-project with the right camera height), then re-run this evaluation and the comparisons above.

## 5. Speed (same run)

| Stage | Per panorama | Environment |
|---|---:|---|
| Vanishing-point alignment (`preprocess.py`) | 6.64 s | cloud 2-core CPU, 404 panoramas in 2,682 s |
| Inference (`inference.py`, CPU mode) | 1.26 s | same, 510 s in total (batches of 40, including model loading) |

For comparison, the 10/26 record on an RTX 3060 (13 panoramas): preprocessing 5.4 s and inference 1.3 s per panorama (13 panoramas sharing the model loading). Preprocessing is CPU-bound and does not get faster on a GPU; for large batches, GPU inference should be well under 1 s per panorama, but it was not measured on a GPU in this run.

## 6. Files and how to reproduce

| File | Contents |
|---|---|
| `PLAN.md`, `PLAN_ZH.md` | scoring plan (written beforehand; English translation and Chinese original) |
| `evaluate.py` | main scoring |
| `diag.py` | post-hoc diagnosis (best scale, area matching, ceiling height) |
| `overlay.py` | draws the ground-truth and predicted boundaries on the panoramas |
| `results.json`, `diag_summary.json`, `rows.json` | summary, diagnosis summary, per-panorama results |
| `fig\` | figures cited above |
| `offline_sitecustomize.py` | only needed offline, where the ImageNet weights cannot be downloaded: lets HorizonNet build the model without downloading (the weights are overwritten by the checkpoint, so results are unaffected) |

```bash
# inside 2_pose_floorplan: create a run with pipeline.prepare first, or write manifest.json yourself (images: {id: {path, status: "ok"}})
python -m pipeline.horizonnet --run <run> --hn-python <python of the HorizonNet environment> [--no-cuda]
python evaluation/horizonnet_404/evaluate.py --pred <run>/horizonnet --gt <ground-truth folder: <floor>/<pano>.txt and rooms.csv> --hn third_party/HorizonNet --out <output>
```
