# 2_pose_floorplan code overview

This document outlines every program in `2_pose_floorplan\`: where it sits in the flow, its main functions and what they do. Usage, input and output formats are in the root [README.md](../README.md) (§5, §6.3, §8, §9). Chinese version: [CODE_OVERVIEW_ZH.md](CODE_OVERVIEW_ZH.md).

This folder reads the room groups exported by `1_room_finder` and produces, in order, CoVisPose relative poses, HorizonNet room outlines, camera poses, merged floor plans and an interactive web page (steps 1–6). It also holds the CoVisPose training and evaluation code.

> The docstrings in the code keep the older numbering (prepare = Step 1, CoVisPose = Step 2, HorizonNet = Step 2b, poses = Step 3, plans = Step 4, viewer = Step 5). This document and the README number the steps 1–6.

---

## 1. Call structure

```
Run_Pipeline.ps1 (main environment; HorizonNet in its own environment)
  1 python -m pipeline.prepare      groups file + scan manifest → runs\<name>\ frozen inputs
  ─ Assert-GpuFree (tools\_env.ps1)
  2 python scripts\manual_room_groups_v2.py run   CoVisPose → predictions\
       └─ covispose.evaluate (load_model_from_ckpt, forward_tta), covispose.model, covispose.geo_pose
  3 python -m pipeline.horizonnet   → subprocess: preprocess.py, inference.py in the HorizonNet env → horizonnet\
  4 python -m pipeline.pose_graph   → poses_<time>\     (default; spanning tree from pipeline.poses, then pose-graph optimisation)
    or python -m pipeline.poses     → poses_<time>\     (-NoPoseGraph: spanning tree only; uses covispose.ransac.build_correspondences)
  5 python -m pipeline.plans        → plans_<time>\     (uses covispose.geometry, shapely)
  6 python -m pipeline.viewer       → viewer\viewer.html, images\

Verify.ps1 = Run_Pipeline.ps1 -Groups sample\room_groups.json -ScanDir sample\scan -Index sample\data_index_sample.json
             -Name _verify_<time>, then python -m pipeline.verify
```

---

## 2. Launch scripts

| File | What it does |
|---|---|
| `Run_Pipeline.ps1` | Parameters `-Groups`, `-Name`, `-ScanDir`, `-Index`, `-Title`, `-Cpu`, `-HnCpu`, `-NoPoseGraph` (step 4 with `pipeline.poses` instead of `pipeline.pose_graph`). Resolves the paths against the current folder, then changes to `2_pose_floorplan`. Runs prepare when `runs\<Name>\manifest.json` does not exist, otherwise resumes. Then runs steps 2–6 in order. CoVisPose is guarded by `inference.lock`; its console output is written to `covispose_gpu_<time>.log` by `Tee-Object`. Stops at the first failing step; after a fix, the same command continues |
| `Verify.ps1` | Calls `Run_Pipeline.ps1` on the 9 panoramas / 3 groups of `sample\` (`sample\room_groups.json`, `sample\scan`, `sample\data_index_sample.json`), then compares with `sample\reference\` through `pipeline.verify`. Prints `Pipeline failed on the sample.` when the pipeline fails |

---

## 3. `pipeline\`: steps 1–6

### 3.1 `common.py`: shared helpers

| Name | What it does |
|---|---|
| `CHECKPOINT`, `CHECKPOINT_FULL`, `CHECKPOINT_EMA`, `HN_CHECKPOINT` | Weight locations; the EMA export is used automatically when it is the only CoVisPose file |
| `local(p)`, `rel_or_abs(p)` | On reading, relative paths resolve against the package root; on writing, paths inside the package are stored relative, so a moved folder still works |
| `read`, `write`, `csvwrite` | JSON I/O (UTF-8); CSV written as UTF-8 with BOM |
| `digest(path)`, `stamp()` | sha256; time stamp (folder names) |
| `rot(yaw)`, `wrap(a)`, `err(p, q)` | 2D rotation matrix; angle wrapped to [−π, π); difference between two relative poses (degrees, translation) |
| `latest(run, prefix)` | Newest `poses_*` / `plans_*` folder |
| `mpl_fonts(matplotlib)` | Chinese-capable fonts for matplotlib |

### 3.2 `prepare.py`: step 1, freeze the inputs (CPU)

| Function | What it does |
|---|---|
| `prepare(groups, output, scan_dir, index)` | ① Reads the groups file and the scan `manifest.json` (`root` first, then `root_abs`); ② checks: status `human_verified`, at least 2 panoramas per group, no repeated members or `room_id`, no `different_room` pair or `conflicts` inside a group; otherwise prints `Input rejected` and stops; ③ decodes each image fully with Pillow, checks exact 2:1, computes sha256; ④ finds identical images, generates every pair of each group and its `pair_id`; ⑤ writes `classification_source.json`, `manifest.json`, `run_config.json` (weight hash, code hashes from `manual_room_groups_v2.sources()`, `RULES`), `preflight.json` |
| `join(*parts)` | Path joining that works for Windows and Linux paths |

### 3.3 `scripts\manual_room_groups_v2.py`: step 2, CoVisPose inference (GPU/CPU)

Although it lives in `scripts\`, this is part of the pipeline and a "frozen source": its hash and those of 8 files of `covispose\` (`FROZEN_SOURCES`) are recorded in `run_config.json`.

| Function | What it does |
|---|---|
| `validate(run)` | Before running: manifest, frozen code, weights and image hashes must equal those of prepare, otherwise stop |
| `completed(run, pair)` | Whether a pair already has a complete result with matching hashes; used for resuming |
| `pick_device(choice)` | `auto` / `cuda` / `cpu` (also `RCF_DEVICE`); auto uses CUDA when available, otherwise the CPU |
| `encode_views(model, img)` | Runs the image encoder once per panorama, giving plain and mirrored column features |
| `cache_usable(...)` | Checks that the half-turn TTA roll is a whole multiple of the encoder stride (32 px), so the features can be shifted exactly |
| `forward_cols(model, fa, fb)` | Runs the rest of CoVisPose (pair Transformer, heads, pose decoder, geometric branch) on encoded column features |
| `forward_tta_cached(...)` | The same 4-view TTA as `covispose.evaluate.forward_tta`, reusing the features |
| `same_prediction(x, y)` | Compares cached and uncached results (yaw ≤ 0.05°, translation ≤ 0.002) |
| `infer(args)` | Main loop: loads the EMA weights (`load_model_from_ckpt`), FP32, TF32 off; TTA for every pair, the first pair also computed without the cache and compared (falls back to no cache on a mismatch); writes `predictions\<pair_id>.json` (decoder pose and the `geometric_pose` branch), `.npz` (co-visibility, correspondence, wall–floor boundary), `inference_status.json`, `run_env_<time>.json` |
| `sources()`, `local`, `read`, `write`, `digest`, `stamp` | Frozen-code hashes and I/O helpers (`MRG_PATH_MAP` maps Windows paths for another OS) |
| `main()` | Sub-command `run --run <folder> [--device] [--no-feature-cache]` |

### 3.4 `horizonnet.py`: step 3, room outlines (GPU/CPU)

HorizonNet runs as a subprocess in its own Python 3.9 environment; this module prepares the inputs and converts the coordinates.

| Function | What it does |
|---|---|
| `horizonnet(run, hn_python, no_cuda)` | Copies panoramas without a result into a work folder (as `<id>.jpg`) and sets `TEMP`/`TMP`/`HN_ASCII_TMP` to an ASCII path (scipy and others fail with a Chinese user name); runs the official `preprocess.py` (LSD vanishing points, 3D rotation alignment) and `inference.py --pth <MP3D weights>`; maps the result back with `unalign` into `horizonnet\<id>.json`. Adds `--no_cuda` when the HorizonNet environment has no CUDA; resumable |
| `vp_rotation(vp)` | Rebuilds the rotation R used by `preprocess.py` from the vanishing points |
| `unalign(...)` | Corners on the aligned image → rays → Rᵀ → uv in the original image (uses `equirect.floor_uv_to_xy`, `ray_to_uv`) |
| `rotation_diagnostics(R)` | Tilt, yaw and determinant for the provenance record |
| `_ours_to_lsd`, `_lsd_to_ours` | Axis conversion between this pipeline and HorizonNet/LSD |
| `run_cmd(cmd, cwd)` | Runs a subprocess, records command, duration and output into `provenance_*.json` |

### 3.5 `equirect.py`: equirectangular geometry

`uv_to_angles`, `angles_to_uv`, `uv_to_ray`, `ray_to_uv` (pixel ↔ azimuth/elevation ↔ 3D ray); `floor_uv_to_xy`, `ceiling_uv_to_xy`, `xy_to_floor_uv`, `xy_to_ceiling_uv` (pixel ↔ point on the floor/ceiling plane); `ceiling_ratio_from_pair` (ceiling height ratio from the ceiling and floor pixel of one corner column); `circ_du`, `split_at_seam` (the left/right seam of the panorama).

### 3.6 `poses.py`: step 4, camera poses (CPU)

| Function | What it does |
|---|---|
| `load_group_edges(run, m, g)` | Reads every `predictions` file of a group (hashes checked), turns the per-column outputs into matched 2D point sets P, Q with `covispose.ransac.build_correspondences`, and collects each panorama's wall–floor boundaries |
| `classify(e)` | Residuals of P, Q under the decoder pose; counts inliers (residual ≤ 0.30) and the inlier fraction f; checks that decoder and geometric branch agree (≤ 5°, ≤ 0.25). Result: `full` (≥ 20 points and f ≥ 0.40), `dir` (f 0.30–0.40) or rejected |
| `forest(nodes, edges, acc)` | Maximum-weight spanning forest (weight 1 + f for full links, f for direction links); propagates each camera's `(yaw, t)` along the tree; accepted links outside the tree become loops |
| `solve(nodes, edges)` | Calls `classify` and `forest`; picks the camera with the most links as the origin; computes the W (weak edge), L (loop not closing > 5° or > 0.30) and D (direction link) flags and the loop residuals |
| `inv`, `rel` | Pose inversion; relative pose of two cameras |
| `reason(c, e, dy, dt)` | Status text per pair (`accepted`, `low_inlier_fraction`, `decoder_geometric_disagree`, …) |
| `poses(run, solver, rule, flag_stats, method)` | Checks that step 2 is complete, solves every group, writes `groups\*.json`, `cameras.csv`, `pairs.csv`, `groups.csv`, `summary.json`. Defaults: `solve`, the tree rule text, `FLAG_STATS`, `spanning_forest`; `pose_graph` passes its own |
| `plot(...)` | matplotlib plot of cameras and links (bird's-eye view, (−x, y)) |
| `FLAG_STATS` | Error rates of the flags on ZInD test for the tree poses (`pose_flags_v1`), written to the output and shown by the viewer |

### 3.6b `pose_graph.py`: step 4, pose-graph optimisation (CPU, default)

Same acceptance rules and spanning forest as `poses.py`; the forest is the starting point. Settings and the ZInD test result: README §11.3.

| Name | What it does |
|---|---|
| `optimise(nodes, edges, cls, p1)` | For every component with more accepted links than a tree needs: robust least squares over all accepted links (`scipy.optimize.least_squares`, Huber, `F_SCALE` = 3). Unknowns: (yaw, x, y) of every camera except the forest root. Residual per link: yaw / `SIGMA_YAW` (1°), translation / `SIGMA_T` (0.06; `SIGMA_T_DIR` 0.6 for direction links), × √weight. Components without a loop are returned unchanged; a failed fit keeps the tree poses. Returns the poses and per-component info (cost before/after, evaluations) |
| `solve_graph(nodes, edges)` | `poses.solve` (acceptance, tree, roots, flags) + `optimise`; `loops` = non-tree links against the optimised poses. Drop-in replacement for `poses.solve` |
| `run_pose_graph(run)` | `poses.poses(run, solver=solve_graph, rule=RULE, flag_stats=FLAG_STATS_GRAPH, method='pose_graph')`, then `pose_graph_report.json` and `components_optimised` in `summary.json` |
| `evaluate(run)` (`--eval`) | Without ground truth: each full link on a loop is left out, predicted from the other links by the tree and by the pose graph, and compared with its own decoder and geometric-branch poses; writes `pose_graph_eval_<time>.json` |
| `FLAG_STATS_GRAPH` | Error rates of the flags on ZInD test for the pose-graph poses (`pose_graph_zind_test_v1`) |

### 3.7 `plans.py`: step 5, floor plans (CPU)

| Function | What it does |
|---|---|
| `outlines(bons, hn_json)` | Three outlines per panorama: CV (column-wise median of the pairs' wall–floor boundaries, `bfloor_to_dist`), HN (`hn_footprint`), MX (per column the farther of the two) |
| `hn_footprint(path)` | HorizonNet `uv` → floor polygon in the camera frame; None when a corner lies beyond 10 camera heights or the data are incomplete |
| `coldist(poly)` | Polygon → radial distance for 1024 columns (`covispose.geometry.polygon_to_column_distance`) |
| `simplify(poly)` | shapely simplification (tolerance 0.05) |
| `placed(poly, cam)` | Places an outline in the origin frame with the camera pose |
| `rings(geom)` | shapely geometry → lists of outer vertices (holes dropped) |
| `plans(run)` | Reads the newest `poses_*`; components with ≥ 2 panoramas take the union of HN outlines (`unary_union`, MX where HN is unusable), a single panorama takes MX; also the CV-only union; writes `groups\*.json`, `index.json`, `plots\*.png` |

### 3.8 `viewer.py`: step 6, interactive page (CPU)

| Function | What it does |
|---|---|
| `viewer(run, title, lang)` | Reads the newest poses and plans (the pose method and the flag error rates come from the poses folder's `method` and `flag_stats`) and embeds cameras, outlines, plans, flags and thumbnails (JPEG quality 70) in one `viewer.html`; panoramas are resized to 4096×2048 (quality 82) and stored as base64 in `images\<id>.jpg.js`, loaded on demand. English and Chinese interface |
| `overlap_rings(polys)` | Overlaps between the panoramas' outlines (shapely), shown in the overlap mode |

### 3.9 `verify.py`, `envcheck.py`

| Function | What it does |
|---|---|
| `verify.verify(run)` | Compares with `sample\reference\reference.json`: CoVisPose yaw/translation per pair, HorizonNet corner counts and positions (via `plans.hn_footprint`), poses (components, tree, flags, camera positions), plan areas; writes `verify_result.json`. Poses and plans are compared with `groups` (pose graph) or `groups_spanning_forest` (tree), chosen by the run's `method` |
| `verify.make_reference(run, pairs)` | Maintainer tool (`--make-reference [--pairs]`): writes the poses and plans of the run's method into the reference and keeps the other method; the CoVisPose pair poses are rewritten only with `--pairs` (after changing weights) |
| `envcheck.main()` | Checks that the main environment's packages import, whether CUDA is available (only a note when not), and the sha256 of both weights |

---

## 4. `covispose\`: the CoVisPose model (shared by inference and training)

### 4.1 Modules used by the pipeline or training

| Module | Main functions / classes | What it does |
|---|---|---|
| `model.py` | `CoVisPose`, `build_model`, `config_from_dict`, `CoVisPoseConfig`, `count_params` | The full model: `ColumnEncoder` (ResNet-50 + HorizonNet-style height compression `ConvCompressH` / `GlobalHeightConv`, circular left/right padding with `LRPad` / `wrap_lr_pad`) → 1024 column features per panorama + `azimuth_encoding` → `PairTransformer` (`PairEncoderLayer` with `drop_path`) fuses both → `ColumnHeads` (co-visibility, correspondence azimuth, wall–floor boundary) + `AttnPoseDecoder` (4 queries cross-attend, output dx, dy, dyaw; the older `PoseDecoder` is kept) + geometric branch |
| `geo_pose.py` | `geometric_pose`, `weighted_procrustes_2d`, `bon_to_dist`, `sample_circular` | Differentiable geometric pose: per-column outputs → two 2D point sets → weighted Procrustes (2 IRLS iterations) for yaw and t |
| `geometry.py` | `col_centers_theta`, `theta_to_col`, `dir_of_theta`, `theta_of`, `bfloor_to_dist`, `dist_to_bfloor`, `wrap_pi`, `relative_pose`, `transform_points`, `ray_cast_polygon`, `polygon_to_column_distance`, `PanoPose` | Basic geometry: column ↔ azimuth, boundary ↔ distance, ZInD poses, ray–polygon intersection (the coordinate conventions of README §9 are defined here) |
| `ransac.py` | `build_correspondences`, `ransac_rigid`, `kabsch_2d`, `pose_from_predictions`, `sample_boundary_at_azimuth` | Per-column outputs → matched points → RANSAC rigid registration; `classify` in step 4 uses `build_correspondences` |
| `evaluate.py` | `load_model_from_ckpt`, `forward_tta`, `tta_views`, `invert_view`, `collect_predictions`, `CachedPredictions`, `head_metrics`, `pose_errors`, `summarize_pose`, `run_ransac_over_cache` | Rebuilds the model from the settings stored in a checkpoint and loads EMA or raw weights; 4-view TTA and its inversion; evaluation metrics |
| `checkpoint.py` | `save_checkpoint`, `load_checkpoint`, `find_resume`, `torch_load`, `prune_epoch_checkpoints` | Full resumable checkpoints (model, EMA, optimizer, scheduler, AMP scaler, RNG state), atomic writes |
| `ema.py` | `ModelEMA` | Exponential moving average of the weights (0.9998) |
| `losses.py` | `covispose_loss`, `covispose_metrics`, `LossWeights`, `circular_l1` | Multi-task loss: co-visibility, correspondence, wall–floor boundary, decoder pose, geometric pose |
| `dataset.py` | `PairDataset`, `make_loader`, `augment_targets` | ZInD pair dataset; horizontal roll, pair swap, mirroring, gamma augmentation |
| `targets.py` | `compute_pair_targets`, `PairTargets` | Per-column ground truth from the ZInD annotations (boundary, co-visibility, correspondence) |
| `pairs.py` | `read_covis_scores`, `write_pair_index`, `read_pair_index`, `room_connections`, `PairRef` | Builds and reads the pair lists from `covis_scores.csv` |
| `zind_parse.py` | `load_house`, `load_partition`, `PanoRecord`, `HouseRecord`, `house_scale_meters` | Parses ZInD `zind_data.json` |
| `health.py` | `HealthMonitor`, `check_against_baselines`, `trivial_baselines`, `vram_guard` | Training monitor: stops early on NaN, divergence, not beating trivial baselines, or too much GPU memory |
| `scene_graph.py` | `spanning_tree`, `global_poses`, `edge_residuals`, `boundary_to_polygon`, `to_world`, `align_rigid` | Older scene assembly (used by `run_scene_graph.py`; one of the frozen sources) |
| `viz.py` | `plot_pair_targets`, `plot_prediction_vs_gt` | Ground-truth and prediction plots (used by `verify_targets.py`) |

### 4.2 Research experiments (not used by the pipeline or training)

`corners.py` (`CornerNet` wall corners), `corner_fit.py` (polygon fitting with fixed corner bearings), `floor_boundary.py` (`FloorBoundaryNet`, single-panorama boundary), `joint_layout.py` (`JointLayoutNet`, LGT-Net-style layout), `layout_integration.py` (`LayoutAdapter`, connects those layout models to fixed poses), `manhattan_fit.py` (Manhattan room fitting), `wall_ba.py` (wall fitting and multi-view bundle adjustment). Kept for reference; changing them does not affect the pipeline and does not block resuming older runs.

---

## 5. `scripts\`: training and evaluation

| File | Main functions | What it does |
|---|---|---|
| `train.py` | `main`, `load_config` (with `_base_` inheritance), `setup_distributed`, `build_scheduler` (warm-up + cosine), `ramped_weights` (geometric terms ramped in), `validate`, `JsonlLogger` | Single-GPU or torchrun multi-GPU training; bf16 AMP, EMA, automatic resume; full validation every epoch, `log.jsonl` |
| `build_pair_index.py` | `main`, `sample_cross_room` | Builds the train/val/test pair lists from ZInD (positives, zero-co-visibility negatives, adjacent-room negatives) |
| `evaluate_split.py` | `main`, `bucket_report` | Evaluates a checkpoint on one split (per-column metrics, three pose sources, buckets by co-visibility) |
| `tune_ransac.py` | `main` | Re-tunes the RANSAC inlier threshold on model predictions |
| `benchmark_vram.py` | `bench`, `bench_config`, `data_throughput` | Measures GPU memory, step time and data throughput |
| `verify_targets.py` | `main` | Draws ground-truth overlays for a few pairs to check the data pipeline |
| `plot_curves.py` | `read_log`, `smooth`, `main` | Training and validation curves from `log.jsonl` |
| `run_scene_graph.py` | `load_pano`, `predict_edges`, `edge_confidence`, `run_scene`, `plot_floorplan` | Older whole-house assembly. The pipeline uses only `load_pano` (reads and resizes to 512×1024); give `--ckpt` when running it by hand |
| `make_zind_room_groups.py` | `main` | ZInD split → pipeline input: one group per complete room with ≥ `--min-panos` panoramas; writes `room_groups.json`, `manifest.json`, `index.json`, `gt.json` (ZInD poses). Used for `evaluation\zind_test_posegraph\` |
| `eval_pose_graph_zind.py` | `evaluate`, `stats`, `camera_wrong`, `gt_rel` | Tree vs pose graph against ZInD ground truth on a run of those groups: errors and accuracy of every camera pair, wrong cameras, flag error rates per method, bootstrap over rooms; writes `pose_graph_zind_eval_<time>.json` |

---

## 6. External code and libraries

| Source | Parts used | Purpose |
|---|---|---|
| PyTorch (main environment, torch 2.10 cu130) | model, inference, training | CoVisPose |
| torchvision | `resnet50(IMAGENET1K_V2)` | CoVisPose encoder backbone |
| Official HorizonNet code (`third_party\HorizonNet`, separate Python 3.9 environment) | `preprocess.py`, `inference.py` (training: `train.py`) | Vanishing-point alignment, room outlines |
| numpy | — | Geometry, RANSAC, statistics |
| scipy | `optimize.least_squares` | `pipeline\pose_graph.py` (step 4) and the experiments of §4.2 |
| shapely | `Polygon`, `unary_union`, `simplify`, `intersection` | Outline unions, overlaps, simplification |
| matplotlib | — | PNG plots of `poses` and `plans` |
| Pillow | `Image.open`, resizing, JPEG encoding | Image checks, viewer images |
| PyYAML | — | Training configuration files |

The official HorizonNet code is unchanged.

---

## 7. Tests

`tests\covispose\` (`python -m pytest tests\covispose`):

| File | Covers |
|---|---|
| `test_geometry.py` | Ray–polygon intersection, boundary ↔ distance, azimuth and elevation matching ZInD, relative pose and its inverse, per-column distance |
| `test_model.py` | Periodic azimuth encoding, output shapes, circular equivariance of the encoder, loss masks, RANSAC and geometric branch recovering ground truth, roll and mirror augmentation, TTA inversion, parameter count, older checkpoints still loading |
| `test_checkpoint.py` | Exact state after resuming and identical continuation, AdamW step on the CPU, EMA saving, `find_resume` |
| `test_negatives.py` | A shared wall seen from both sides is not co-visible, rooms connected by doors, pair-list I/O, no pose loss on negatives, co-visibility agreeing with Zillow's scores |
| `test_bn_stats.py`, `test_overfit.py` | Pretrained ImageNet BatchNorm statistics and weights kept after building the model; over-fitting 5 pairs (the model can learn; needs ZInD) |
| `_requires.py` | Skips tests when ZInD data, ZInD's own code or the network are missing |

`tests\pipeline\test_pose_graph.py` (`python -m pytest tests\pipeline`, no data needed): a tree component is left unchanged; on noisy synthetic rooms the pose graph is closer to the truth than the tree; one wrong but accepted link does not pull the loop; the tree, roots and flags are the same as `poses.solve`.
