# -*- coding: utf-8 -*-
"""Paths and the default classifier. Everything is relative to this folder so the whole
room_candidate_finder folder can be copied to another computer."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# GeoVerify code used for scanning (GitHub main export shipped in third_party/; has SOURCE_COMMIT.txt)
GEOVERIFY_MAIN = os.environ.get("RCF_GEOVERIFY_MAIN", os.path.join(ROOT, "third_party", "geoverify_main"))
DATA_DIR = os.path.join(ROOT, "data")       # datasets: data/<dataset>/<house>/*.jpg
SCANS_DIR = os.path.join(ROOT, "scans")     # scan outputs: scans/<dataset>/

# RoMa v2 downloads its weights (and the DINOv3 code) through torch.hub on first use. Keep them inside the package
# (models/torch_home) so a copy of the package does not download them again. An explicit TORCH_HOME wins; and when the
# package has no copy yet but this computer's default torch cache does, that cache keeps being used (no new download;
# Setup.ps1 copies it into the package).
TORCH_HOME = os.path.join(ROOT, "models", "torch_home")
_ROMA_REL = os.path.join("hub", "checkpoints", "romav2.0.1.pt")
if "TORCH_HOME" not in os.environ and (
        os.path.isfile(os.path.join(TORCH_HOME, _ROMA_REL))
        or not os.path.isfile(os.path.join(os.path.expanduser("~"), ".cache", "torch", _ROMA_REL))):
    os.environ["TORCH_HOME"] = TORCH_HOME

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
# folders inside a house folder that hold derived images, not source panoramas
EXCLUDE_DIRS = ("bi_layout_predictions", "horizonnet_predictions", "semantic", "preview", "thumbs")

# Default classifier chosen in same_room_benchmark (RESULTS.md, 2026-10-23)
DEFAULT_WEDGE_DEG = 45.0
DEFAULT_MIN_INLIERS = 30
CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_dollhouse.json")

# Door-aware ("open") wedge, same_room_benchmark/RESULTS_open_wedge.md (2026-10-27):
# GeoVerify inliers that fall inside a SAM3 "open doorway" mask in A or B are dropped (seen through an
# opening = another room), the 80% wedge is recomputed from the rest. Needs openings/ + open_scores.jsonl.
DEFAULT_OPEN_WEDGE_DEG = 40.0
OPEN_MASK_W, OPEN_MASK_H = 2048, 1024     # masks are compared at this size
OPEN_DILATE_PX = 4                        # mask dilation at 2048 px width: 4 px (0.7 deg) sideways, 10 px up/down
OPEN_PROMPT = "open doorway"              # SAM3 prompt (best of 8 prompt sets in the benchmark)
# SAM3 project used by Start_Openings.ps1 (env RCF_SAM3_ROOT overrides)
SAM3_ROOT = os.environ.get("RCF_SAM3_ROOT", os.path.join(ROOT, "third_party", "sam3-pano-seg"))
