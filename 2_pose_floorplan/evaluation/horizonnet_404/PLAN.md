# HorizonNet outline evaluation: scoring plan (written before looking at any result)

English translation, made on 2026-11-09. The pre-registered original is the Chinese [PLAN_ZH.md](PLAN_ZH.md) (sha256 `6151bd7c…`); if the two differ, the Chinese original is authoritative.

- Prediction: the package's `2_pose_floorplan/pipeline/horizonnet.py` (official preprocess.py vanishing-point alignment + inference.py, MP3D weights, corners mapped back to the original image coordinates), run on the CPU; input is the original image resized to 1024×512 with BICUBIC (the same as inside preprocess.py).
- Ground truth: room-stitching-systemV3/wall-solver-point-edge/data/group/*/layout_gt/*.txt (latest version, 404 panoramas, 42 floors, 24 houses), each panorama's own room outline in 1024×512 pixel coordinates.
- Metrics (per panorama, same definitions as HorizonNet's official eval_general.py):
  1. 2D IoU: IoU of the polygons formed by projecting the floor corners onto the ground (main metric).
  2. 3D IoU: 2D intersection × the lower ceiling height; ceiling height from the official post_proc.get_z1.
  3. Depth RMSE, δ1 (official layout_2_depth; the predicted ceiling-corner x is snapped to the floor-corner x, a difference of about 1 px).
  4. Per-column relative wall-distance error (median).
- Missing prediction or invalid polygon: IoU counted as 0, and the count listed separately.
- Statistics: overall mean; grouped by ground-truth corner count (4, 6, 8, 10+), room type (rooms.csv) and floor; 95% intervals from 1,000 bootstrap resamples by house.
- References: official MatterportLayout results (2D 81.71, 3D 79.11, RMSE 0.197, δ1 0.929); the 2026-10-21 evaluation (stitched group ground truth, 94 panoramas, IoU_J 0.567): list why it differs, do not compare the numbers directly.
