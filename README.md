# room_candidate_finder: same-room screening → camera poses → floor plans

This is the only document of the package. It replaces the earlier `README`, `NOTICE`, `SPEC` and `OUTPUT_FORMAT` files, which were merged here on 2026-11-17. Chinese version: [README_ZH.md](README_ZH.md).

The package turns a set of 360° panoramas into a camera layout and a floor plan for each room. The whole folder can be copied to another computer and run there, because every path is relative to this folder.

**Contents**

| § | Topic |
|---|---|
| [0](#0-read-this-before-handing-over-or-using-the-package) | Read this first: hand-over essentials |
| [1](#1-overview) | Overview, flow, models |
| [2](#2-folder-layout) | Folder layout |
| [3](#3-environment) | Hardware, environments, SAM3 |
| [4](#4-installation-and-verification) | Installation and verification (including after copying) |
| [5](#5-usage) | Commands and options |
| [6](#6-input-formats) | **Input formats** of both folders |
| [7](#7-output-formats-1_room_finder) | **Output formats** of `1_room_finder` |
| [8](#8-output-formats-2_pose_floorplan) | **Output formats** of `2_pose_floorplan` |
| [9](#9-units-coordinates-and-the-birds-eye-view) | Units, coordinates and the bird's-eye view |
| [10](#10-speed) | Speed |
| [11](#11-algorithms-and-parameters) | Algorithms and parameters |
| [12](#12-models-and-weights) | Models and weights |
| [13](#13-training-covispose) | Training CoVisPose |
| [14](#14-training-or-fine-tuning-horizonnet) | Training or fine-tuning HorizonNet |
| [15](#15-geoverify-and-roma-v2) | GeoVerify and RoMa v2 |
| [16](#16-tests-and-verification-records) | Tests and verification records |
| [17](#17-licences) | Licences |
| [18](#18-known-conditions-limitations-and-open-items) | Known conditions, limitations, open items |
| [19](#19-change-history) | Change history |
| [20](#20-other-documents) | Other documents |

---

## 0. Read this before handing over or using the package

**Getting it from GitHub**

The Git repository holds the code, documents, the 9 sample panoramas, the training pair lists and the result files. These are **not** in it:

| Not in the repository | Size | How to get it |
|---|---|---|
| `2_pose_floorplan\models\covispose\epoch_015.pt` | 1.41 GB | Release `v1.0` asset (see `models\covispose\README.md`) |
| `2_pose_floorplan\models\horizonnet\resnet50_rnn__mp3d.pth` | 327 MB | Release `v1.0` asset |
| RoMa v2 weights, DINOv3 code, ImageNet ResNet-50 (`models\torch_home\`) | 1.2 GB | downloaded by `Setup.ps1` |
| Python environments | — | built by `Setup.ps1` |
| SAM3 (`1_room_finder\third_party\sam3-pano-seg\`) | about 9 GB | not distributed; without it the review uses the plain wedge (§3.3) |
| Dollhouse panoramas (`1_room_finder\data\dollhouse`) and the viewer's panorama files (`runs\*\viewer\images`) | about 14 GB | not distributed; the scans, review records and the 104-group results are included, `viewer.html` shows the plans but not the panoramas |
| ZInD ground truth `evaluation\zind_test_posegraph\gt.json` and the ZInD predictions | — | regenerated from your own ZInD copy by `Run_ZInD_PoseGraph_Eval.ps1` (§11.3) |

Steps: `git clone` to a path with ASCII letters only → download the two release assets into the folders above (`gh release download v1.0 --repo ERIC9152/room_candidate_finder --dir <folder> --pattern <file>`, or the release page) → `Setup.ps1` → `2_pose_floorplan\Verify.ps1`.

**Using it on another computer**

1. Put the folder on a path with **ASCII letters only**, for example `C:\room_candidate_finder`. HorizonNet and the SAM3 environment fail on a path with Chinese characters.
2. After copying or moving the folder, run `Setup.ps1` once, then `2_pose_floorplan\Verify.ps1` (§4). A Windows Python virtual environment remembers the absolute path it was created at, so the environments in a copied folder cannot start; `Setup.ps1` detects this and rebuilds them. `VERIFY PASSED` means the computer is ready.
3. `Setup.ps1` needs network access: about 10–30 minutes and 8–10 GB of downloads. Afterwards the package runs offline.
4. GPU: the scan (step A) and SAM3 need an NVIDIA GPU with a driver that supports CUDA 13.0. Steps 1–6 fall back to the CPU when there is no GPU. RTX 50 cards need `-HnCpu` for HorizonNet (§18).

**Licences** (§17)

5. The CoVisPose weights and `training\pairs\` come from ZInD and are for **academic, non-commercial use only**.
6. SAM3 is under the Meta SAM License. `1_room_finder\third_party\sam3-pano-seg\third_party\sam3\LICENSE` **must stay in the package** when it is handed over.
7. The Dollhouse panoramas (`1_room_finder\data\dollhouse`, about 15 GB) are bundled. Check with the data provider's terms that the recipient may receive them, or delete `1_room_finder\data\dollhouse` and `1_room_finder\scans\dollhouse`.

**Reading the outputs** (§6–§9)

8. **Angles come in two units.** JSON angles without a `_deg` suffix (`yaw`, `geo_yaw`, `corr_*`, `bon_*`) are **radians**. Fields with a `_deg` suffix, `yaw_deg` in the CSV files, every wedge in `1_room_finder` and the GeoVerify azimuths are **degrees** (§9.1).
9. **Lengths are in camera heights**: the camera-to-floor distance is 1. Multiply by the real camera height to get metres (about 1.5 m in ZInD; unknown for Dollhouse). Areas are in camera heights².
10. **Coordinates are CoVisPose native coordinates; plotted with x to the right and y up they come out mirrored.** For a bird's-eye view, plot **(−x, y)**. The pipeline's PNG files and the viewer already do this (§9.3).
11. **Poses**: relative pose `p_B = R(yaw) · p_A + t`; camera pose `p_root = R(yaw) · p_u + t`. GeoVerify's 3D pose is a direction only (`t` is a unit vector), and the pipeline uses GeoVerify only to decide "same room".
12. **There is no pose ground truth.** Every pose and plan file has `pose_verified: false`; a red "!" means "worth checking", not "known to be wrong".
13. **Dollhouse has two group files**: `room_groups_dollhouse104.json` is the input of the 104-group results; `room_groups_test_20261026.json` is only a 6-group test export (§7.10).
14. **A run folder freezes its inputs and code hashes.** After `scripts\manual_room_groups_v2.py` or the `covispose\` code changes, older runs cannot be resumed; prepare again under another `-Name`.
15. `viewer\viewer.html` must stay next to `viewer\images\`; `covispose_gpu_*.log` is UTF-16 (written by PowerShell 5.1).

---

## 1. Overview

The input is a set of 360° equirectangular panoramas, one folder per house. The output is, for each room, the relative camera positions, a merged floor plan, and an interactive web page in which you can walk around the plan and look at the panoramas.

```
panorama folder
   │
   ▼  1_room_finder
   A. same-room candidates (GeoVerify + RoMa v2, GPU)
   │   └─ A'. SAM3 door openings → door-aware wedge (bundled, optional)
   B. manual review (review app, judge each pair: same room / different room)
   │
   ▼  room_groups.json (+ the scan manifest.json next to it)
   │
   ▼  2_pose_floorplan
   1. prepare (freeze the inputs)        CPU
   2. CoVisPose relative poses           GPU (CPU fallback)
   3. HorizonNet room outlines           GPU (CPU fallback)
   4. poses (acceptance, pose graph)     CPU
   5. plans (merged floor plans)         CPU
   6. viewer (interactive page)          CPU
```

| Step | What it does | Main program | Hardware |
|---|---|---|---|
| A. Same-room candidates | Geometric verification of every pair of panoramas in a house: inlier count and the co-visible wedge angle on both sides | `finder/scan.py` (GeoVerify + RoMa v2) | GPU |
| A'. Door openings (optional) | SAM3 "open doorway" masks → drop inliers seen through an opening → door-aware wedge | `Start_Openings.ps1`, `finder/openings.py` | GPU |
| B. Manual review | Lists candidates above a wedge threshold; a person judges same room / different room; same-room pairs are chained into room groups and exported as `room_groups.json` | `finder/app.py` (Tkinter) | — |
| 1. prepare | Reads the groups and images, checks every image, freezes the inputs (image sha256, pair list, weight and code hashes); rejects a group that contains a "different room" decision | `pipeline/prepare.py` | CPU |
| 2. CoVisPose | Relative pose and per-column outputs for every pair inside a group | `scripts/manual_room_groups_v2.py run` | GPU; CPU when none |
| 3. HorizonNet | Room outline of each panorama | `pipeline/horizonnet.py` → `third_party/HorizonNet` | GPU; CPU possible |
| 4. poses | Acceptance rules, spanning tree as the starting point, pose-graph optimisation of every component with a loop (§11.3), origin, pose flags | `pipeline/pose_graph.py` (`-NoPoseGraph`: `pipeline/poses.py`, tree only) | CPU |
| 5. plans | Merged floor plan | `pipeline/plans.py` | CPU |
| 6. viewer | Interactive web page | `pipeline/viewer.py` | CPU |

`2_pose_floorplan\Run_Pipeline.ps1` runs steps 1–6 in one go and can be interrupted and resumed.

**Models**

| Model | Use | Weights | Trained in this project |
|---|---|---|---|
| CoVisPose (own re-implementation of the ECCV 2022 architecture) | Two panoramas → co-visibility, per-column correspondence, wall–floor boundary, relative translation and yaw | `models/covispose/epoch_015.pt` | **Yes** (ZInD, §13) |
| HorizonNet (official) | One panorama → room outline | `models/horizonnet/resnet50_rnn__mp3d.pth` | No, official MP3D weights (fine-tuning: §14) |
| RoMa v2 (used by GeoVerify) | Dense feature matching | `1_room_finder\models\torch_home` (downloaded by `Setup.ps1`) | No |
| SAM3 (sam3-pano-seg) | Door-opening segmentation ("open doorway") | `third_party/sam3-pano-seg/checkpoints/sam3.pt` | No |

---

## 2. Folder layout

The package has two sub-folders, numbered in run order. They are connected only through the `room_groups.json` exported by the review app and the scan `manifest.json` next to it; neither imports Python code from the other. There is one main environment: it lives under `1_room_finder`, and `2_pose_floorplan` uses it too.

```
room_candidate_finder\
├─ README.md, README_ZH.md                                     this document (English / Chinese)
├─ Setup.ps1                                                   builds every environment (shared by both sub-folders)
├─ tools\_env.ps1                                              shared functions: find environments, find SAM3, check the GPU
│
├─ 1_room_finder\          ── same-room screening (steps A, A', B)
│   ├─ CODE_OVERVIEW.md, CODE_OVERVIEW_ZH.md                    programs and functions of this folder
│   ├─ Import_Dataset.ps1                                       import + scan (+ door openings)
│   ├─ Start_Openings.ps1                                       door-opening detection (also called by Import_Dataset.ps1)
│   ├─ Start_Review.bat                                         review app
│   ├─ finder\             scan, door-aware wedge, review app; config.py = default thresholds;
│   │                      calibration_dollhouse.json = P/R table; i18n.py = English/Chinese interface text
│   ├─ tools\              fix_sam3_env.ps1, recompute_inliers.py, import_openings.py
│   ├─ models\torch_home\                                       RoMa v2 weights and DINOv3 code (TORCH_HOME)
│   ├─ third_party\geoverify_main\                              GeoVerify (commit b5897093); the main venv is built here
│   ├─ third_party\sam3-pano-seg\                               SAM3: ready environment .runtime\, weights, source, sam3_pano (about 9 GB)
│   ├─ scans\dollhouse\                                         scan results, door masks, review records of the 60 Dollhouse houses
│   ├─ data\                                                    imported panoramas (data\dollhouse: the Dollhouse dataset, about 15 GB)
│   ├─ tests\test_core.py
│   └─ docs\REVIEW_APP.md, docs\benchmark\                      review-app guide, same-room classifier comparison
│
└─ 2_pose_floorplan\       ── relative poses and floor plans (steps 1–6)
    ├─ CODE_OVERVIEW.md, CODE_OVERVIEW_ZH.md                    programs and functions of this folder
    ├─ Run_Pipeline.ps1  Verify.ps1                             steps 1–6, sample check
    ├─ requirements_horizonnet.txt  requirements_main_extra.txt  constraints_main.txt
    ├─ pipeline\           steps 1–6 (step 4: pose_graph.py, default; poses.py, tree only), verify.py (sample comparison), envcheck.py (environment check)
    ├─ covispose\          CoVisPose model, dataset, losses, geometry, evaluation (shared by inference and training)
    ├─ scripts\
    │   ├─ manual_room_groups_v2.py, run_scene_graph.py        inference (step 2)
    │   ├─ make_zind_room_groups.py, eval_pose_graph_zind.py    ZInD evaluation of step 4 (§11.3)
    │   └─ train.py, build_pair_index.py, evaluate_split.py,
    │      tune_ransac.py, benchmark_vram.py, verify_targets.py, plot_curves.py   training and evaluation
    ├─ models\covispose\epoch_015.pt                            1.41 GB (or epoch_015_ema.pt, 0.35 GB, §12)
    ├─ models\horizonnet\resnet50_rnn__mp3d.pth                 327 MB
    ├─ models\torch_home\                                       ImageNet ResNet-50 (used to build HorizonNet)
    ├─ third_party\HorizonNet\                                  official HorizonNet code (with train.py); the HorizonNet venv is built here
    ├─ training\           configs, ZInD pair lists (44 MB), training environment, original log, design notes
    ├─ evaluation\horizonnet_404\                               HorizonNet outline evaluation on 404 panoramas
    ├─ evaluation\zind_test_posegraph\                          ZInD test input and launcher of the step-4 evaluation (§11.3)
    ├─ sample\                                                  9 panoramas, scan manifest, room_groups.json, reference results
    ├─ runs\                                                    output of Run_Pipeline.ps1
    └─ tests\covispose\, tests\pipeline\
```

Below, paths for steps A, A' and B are relative to `1_room_finder\`, and paths for steps 1–6, CoVisPose, HorizonNet and training are relative to `2_pose_floorplan\`, unless stated otherwise.

---

## 3. Environment

### 3.1 Hardware and operating system

| Item | Requirement |
|---|---|
| Operating system | Windows 10/11 64-bit (the scripts are PowerShell 5.1 and .bat) |
| GPU | NVIDIA. The main environment uses the CUDA 13.0 build of PyTorch and needs a driver that supports it; GeoVerify was verified on Ada (RTX 40, sm_89) and Blackwell (RTX 50, sm_120). Older architectures (Pascal, Volta, …) are not supported by cu130 |
| GPU memory | One program on the GPU at a time; CoVisPose inference is FP32 with 88 M parameters, 8 GB or more recommended |
| Disk | Package without Dollhouse about 11 GB (about 9 GB of it is the bundled SAM3 environment and weights); Dollhouse panoramas about 15 GB more (whole folder about 26 GB); the environments built by `Setup.ps1` another 8–10 GB |
| Path | **ASCII letters only** (`Setup.ps1` warns) |
| Network | Needed while building the environments; offline afterwards |

### 3.2 The three inference environments (built inside the package)

| Environment | Location | Used for | Versions |
|---|---|---|---|
| Main | `1_room_finder\third_party\geoverify_main\venv` | Step A, the wedge computation of A', steps 1, 2, 4, 5, 6 | Python 3.11 (uv), torch 2.10.0+cu130, numpy 1.26.4, opencv-python 4.11.0.86, scikit-learn 1.8.0, pillow 12.0.0, tqdm 4.70.0, romav2 (GitHub source); plus shapely ≥ 2.0, matplotlib ≥ 3.8, scipy ≥ 1.11 (numpy pinned to 1.26.4 by `constraints_main.txt`) |
| HorizonNet | `2_pose_floorplan\third_party\HorizonNet\venv` | Step 3 | Python 3.9, torch 1.12.0+cu113, torchvision 0.13.0+cu113, numpy 1.21.0, opencv-python 4.7.0.72, pylsd-nova 1.2.0, scipy 1.7.3, scikit-learn 1.0.2, Shapely 1.8.5, pillow 11.3.0, matplotlib 3.5.3 (`requirements_horizonnet.txt`) |
| Review | `1_room_finder\.venv` | Step B | Any Python 3.9 or newer with Tkinter, plus pillow ≥ 9 |

Environment variables (set before running a script):

| Variable | Meaning |
|---|---|
| `RCF_MAIN_PYTHON`, `RCF_HN_PYTHON`, `RCF_REVIEW_PYTHON` | Use an existing python.exe for the main / HorizonNet / review environment instead of the package's own |
| `RCF_SAM3_ROOT` | sam3-pano-seg folder (default `1_room_finder\third_party\sam3-pano-seg`) |
| `RCF_GEOVERIFY_MAIN` | GeoVerify code (default `1_room_finder\third_party\geoverify_main`) |
| `RCF_DEVICE` | `cpu` forces CoVisPose onto the CPU (default: CUDA when available) |
| `RCF_LANG` | Interface language of the review app and the viewer: `en` (default) or `zh` |
| `TORCH_HOME` | Left alone when set; otherwise `finder\config.py` points it at `1_room_finder\models\torch_home` |
| `ZIND_ROOT`, `ZIND_CODE` | ZInD data and ZInD's own code, for training and the tests that need them |

### 3.3 SAM3 environment (bundled, ready to use)

| Location (in `1_room_finder\third_party\sam3-pano-seg\`) | Contents |
|---|---|
| `.runtime\python312\` | Python 3.12.10 (base interpreter) |
| `.runtime\sam3-env\` | Virtual environment: torch 2.7.1+cu126, torchvision 0.22.1+cu126 etc.; `sam3` and `sam3_pano` installed in editable mode |
| `checkpoints\sam3.pt` | SAM3 weights (3.45 GB) |
| `third_party\sam3\` | SAM3 source and its licence file `LICENSE` |
| `sam3_pano\` | Panorama segmentation; the pipeline runs `python -m sam3_pano.seg.unified --categories door --door-prompt-override "open doorway" --pano-vote --mask-only` |

- Nothing has to be installed, but the virtual environment remembers absolute paths (`home` in `pyvenv.cfg` and two `__editable___*_finder.py` files). `tools\fix_sam3_env.ps1` rewrites them for the current location; `Setup.ps1` and `Start_Openings.ps1` call it automatically, and it does nothing when the paths are already right.
- About 3–4 s per panorama (RTX GPU); loading the model the first time takes about 1–2 minutes.
- `-Sam3Root` / `RCF_SAM3_ROOT` can point to a sam3-pano-seg elsewhere. If `third_party\sam3-pano-seg` is deleted (saves about 9 GB), the scan skips door-opening detection and the review app uses the plain wedge (45°).

---

## 4. Installation and verification

From the package root:

```powershell
powershell -ExecutionPolicy Bypass -File Setup.ps1                     # builds every environment (network, about 10–30 minutes)
powershell -ExecutionPolicy Bypass -File 2_pose_floorplan\Verify.ps1   # steps 1–6 on the 9 bundled sample panoramas, compared with the reference (about 2–4 minutes)
```

`Setup.ps1` does, in order:

1. **Main environment**: runs GeoVerify's `1_room_finder\third_party\geoverify_main\environments\setup_env.ps1` (uv installs Python 3.11, torch 2.10.0 cu130, the core packages and romav2, then re-pins versions), then adds `2_pose_floorplan\requirements_main_extra.txt`. That installer contains Chinese text in UTF-8 without a BOM, which Windows PowerShell 5.1 cannot parse on a Chinese-locale system, so `Setup.ps1` runs a temporary copy with a BOM (the original is unchanged). RoMa v2 weights are copied from this computer's cache or downloaded into `1_room_finder\models\torch_home`.
2. **HorizonNet environment**: `py -3.9 -m venv`, installs `requirements_horizonnet.txt`, downloads the ImageNet ResNet-50 into `2_pose_floorplan\models\torch_home`. Without Python 3.9 on the computer, uv downloads one (`uv venv --python 3.9 --seed`).
3. **Review environment**: creates `1_room_finder\.venv` with the first of `py -3.11`, `py -3`, `python` that works, installs `pillow>=9`, checks Tkinter and Pillow. If no Python with Tkinter exists, it stops: install Python 3 from python.org (keep "tcl/tk and IDLE") and run `Setup.ps1` again.
4. **SAM3**: `1_room_finder\tools\fix_sam3_env.ps1`, then an import check (`import torch, sam3, sam3_pano`).
5. **Checks**: `python -m pipeline.envcheck` (packages, CUDA, sha256 of the weights) and an import test of the HorizonNet environment.

An environment that exists but does not start (for example one copied from another computer) is rebuilt automatically; there is nothing to delete by hand.

`Verify.ps1` runs steps 1–6 on the 9 panoramas / 3 groups of `sample\` and compares with the original Dollhouse results (`pipeline\verify.py`). Pass criteria: CoVisPose yaw ≤ 0.5° and translation ≤ 0.02 per pair; HorizonNet same number of corners and floor corners within 0.05; poses with the same components, spanning tree and flags, camera positions within 0.05 and yaw within 1°; plan areas within 5%. The result is written to `2_pose_floorplan\runs\_verify_<time>\verify_result.json` (`passed`, `checks`, `at`); every run makes a new folder, which can be deleted afterwards.

Before a GPU step, `tools\_env.ps1` (`Assert-GpuFree`) asks for confirmation when **another Python program** is using the GPU (answer `y` to continue anyway). Desktop programs that `nvidia-smi` lists on Windows are ignored. Without `nvidia-smi` the check is skipped with a note.

---

## 5. Usage

All commands are run from the package root.

```powershell
# A (+A') import and scan; resumable (run the same command again to continue)
powershell -ExecutionPolicy Bypass -File 1_room_finder\Import_Dataset.ps1 -Source D:\new_panos -Name new_panos
#   -NoCopy      panoramas already in 1_room_finder\data\new_panos: scan only (leave out -Source)
#   -Limit 5     stop after 5 new pairs (trial run)
#   -NoOpenings  skip SAM3 door-opening detection

# A' add door-opening detection later (-Scan is relative to 1_room_finder; -Sam3Root optional)
powershell -ExecutionPolicy Bypass -File 1_room_finder\Start_Openings.ps1 -Scan scans\new_panos

# B review: judge each pair, then press "Export room groups JSON" (without an argument a folder picker opens)
1_room_finder\Start_Review.bat 1_room_finder\scans\new_panos

# 1–6 poses, plans, viewer; resumable
powershell -ExecutionPolicy Bypass -File 2_pose_floorplan\Run_Pipeline.ps1 -Groups 1_room_finder\scans\new_panos\room_groups.json -Name new_panos
```

Then open `2_pose_floorplan\runs\new_panos\viewer\viewer.html`.

`Run_Pipeline.ps1` options:

| Option | Meaning |
|---|---|
| `-Groups` (required) | Room-groups file (§6.3); relative to the current folder |
| `-Name` (required) | Run name; output in `2_pose_floorplan\runs\<Name>`. An existing run with `manifest.json` is resumed |
| `-ScanDir` | Scan folder holding `manifest.json`, when the groups file is not inside it |
| `-Index` | Dollhouse `data_index.json`, only used for room names |
| `-Title` | Title of the viewer page |
| `-Cpu` | CoVisPose and HorizonNet on the CPU, no GPU check |
| `-HnCpu` | Only HorizonNet on the CPU (RTX 50 series) |
| `-NoPoseGraph` | Step 4 with the spanning tree only (`pipeline.poses`, the method before 2026-11-19). Default: pose-graph optimisation (`pipeline.pose_graph`, §11.3) |

Per-point inliers for the review screen are stored by the current scan. For an older scan, recompute them (GPU, about 3 s per pair, resumable), from `1_room_finder` with the main environment:

```powershell
third_party\geoverify_main\venv\Scripts\python.exe tools\recompute_inliers.py --scan scans\dollhouse [--house NAME] [--reviewed] [--groups <room_groups.json>]
```

**Language**: the review app and the viewer are in English by default. Press the language button, set `RCF_LANG=zh` before starting, or use `python -m finder.app --lang zh` / `python -m pipeline.viewer --run <run> --lang zh`. The viewer remembers the last choice in the browser.

**Rerunning the 104 Dollhouse groups** (§16.2):

```powershell
powershell -ExecutionPolicy Bypass -File 2_pose_floorplan\Run_Pipeline.ps1 -Groups 1_room_finder\scans\dollhouse\room_groups_dollhouse104.json -ScanDir 1_room_finder\scans\dollhouse -Index 1_room_finder\scans\dollhouse\data_index.json -Name dollhouse104 -Title "Dollhouse 104 groups"
```

---

## 6. Input formats

### 6.1 Step A (`Import_Dataset.ps1`, `finder.scan`): panorama folder

```
<source folder>\
├─ <house 1>\            one sub-folder per house; the folder name becomes the house name
│   ├─ pano_001.jpg
│   └─ floor2\pano_002.png      deeper sub-folders are read too
└─ <house 2>\ ...
```

| Rule | Details (`finder\dataset.py`, `finder\config.py`) |
|---|---|
| File types | `.jpg`, `.jpeg`, `.png` (case-insensitive) |
| Skipped folders | `bi_layout_predictions`, `horizonnet_predictions`, `semantic`, `preview`, `thumbs` |
| Projection | Equirectangular. Width / height must be 2 within ±0.02, otherwise the file is skipped (`not 2:1`). **Step 1 later requires width = exactly 2 × height** (§6.3), so use exact 2:1 images |
| Panorama id | File name without extension. Same name, different content in one house → `_2`, `_3`, … appended |
| Duplicates | Same content (first MiB + file size) inside a house → kept once, the others recorded in `skipped` |
| Houses | A house needs at least 2 panoramas. If the source folder has no sub-folders with images, the folder itself is one house |
| Pairs | Every pair inside a house (n panoramas → n(n−1)/2 pairs); floors are not distinguished |
| Unreadable files | Skipped with the reason in `manifest.json` → `skipped` |

`-Source` is copied to `1_room_finder\data\<Name>` (robocopy); `-NoCopy` scans `data\<Name>` as it is. Output: `1_room_finder\scans\<Name>\` (§7).

### 6.2 Steps A' and B: the scan folder

`Start_Openings.ps1 -Scan` and `Start_Review.bat` take a scan folder written by step A. It must contain `manifest.json`; the panoramas are found through its `root` / `root_abs` (§7.1), so `data\<Name>` must still exist. The review app also reads `scores.jsonl`, `pairs\` (or `pairs_inliers\`), and when present `open_scores.jsonl` and `decisions.jsonl`.

### 6.3 Step 1 (`Run_Pipeline.ps1 -Groups`): room-groups file

The export of the review app (§7.9) or any JSON with the following fields. `room_groups_dollhouse104.json` (`room_classification.json` format with more fields) is accepted the same way. Checked by `pipeline\prepare.py`:

| Field | Required | Rule |
|---|---|---|
| `groups` | yes | Not empty |
| `groups[].room_id` | yes | Unique |
| `groups[].task` | yes | House name; must equal a house key in the scan `manifest.json` |
| `groups[].panorama_uuids` | yes | At least 2, no repeats, each panorama in only one group; each must be an `id` of that house in the scan manifest |
| `groups[].status` | yes | Must be `human_verified` |
| `groups[].floor`, `space_kind` | no | Copied (defaults `""`, `room`) |
| `pair_reviews[]` | no | `uuid_a`, `uuid_b`, `relation`. A `different_room` pair inside one group rejects the whole input; `same_room` marks the pair `directly_reviewed` |
| `conflicts` | no | Not empty → whole input rejected |

Other inputs of step 1:

| Input | Default | Rule |
|---|---|---|
| Scan `manifest.json` | the folder of the groups file; otherwise `-ScanDir` | `houses[house][].id` and `path`; `root` (relative to the scan folder) is tried first, then `root_abs` |
| Images | — | Must decode completely and be exactly 2:1; failures are recorded per image (`status: input_error`) in `manifest.json` and `preflight.json` |
| `-Index` | none | JSON `{"panos": {"<uuid>": {"asteroom_name": "<room name>"}}}`; only the room name is used |

Rejections are printed as `Input rejected:` before anything runs on the GPU; fix them in the review app and export again.

### 6.4 Training inputs

CoVisPose training needs ZInD (§13.2): `zind_data\<house>\zind_data.json`, `panos\*.jpg`, plus the pair lists `training\pairs\*.csv` with columns `house_id, floor_id, room_a, pano_a, room_b, pano_b, covis_score, baseline, dyaw_deg, kind`. HorizonNet training data format: §14.

---

## 7. Output formats: `1_room_finder`

Location: `1_room_finder\scans\<dataset name>\`.

### 7.1 `manifest.json`: the dataset

| Field | Meaning |
|---|---|
| `root` | Panorama folder, relative to the scan folder (e.g. `../../data/dollhouse`) |
| `root_abs` | Absolute path used when `root` is not found |
| `geoverify_commit`, `mode`, `params` | GeoVerify version, mode (always `direct`), parameters |
| `houses` | `{house: [{id, path, w, h}]}`; `id` = panorama id (§6.1), `path` relative to `root`, `w`/`h` = image size in pixels |
| `skipped` | Skipped files: `[{path, reason}]` |
| `total_pairs`, `created` | Number of pairs, creation time |

The Dollhouse scan was imported from an earlier experiment: `w` and `h` are 0 and a `note` field says where it came from.

### 7.2 `scores.jsonl`: GeoVerify summary per pair

One line per pair: `{"house", "a", "b", "score": {...}}`. When a pair appears more than once, the last line wins.

| `score` field | Unit | Meaning |
|---|---|---|
| `has_pose` | — | GeoVerify found a pose |
| `n_inlier` | points | Essential-matrix inliers |
| `inlier_rate` | 0–1 | Inliers / matches used |
| `wedge_a`, `wedge_b` | degrees | Co-visible wedge: narrowest arc holding 80% of the inlier azimuths |
| `wedge_min` | degrees | The smaller of the two; this is what the classifier compares |
| `n_regions` | count | Co-visible regions |

### 7.3 `pairs\<house>\<a>__<b>.json`: full GeoVerify output

GeoVerify's `verify_pair` result as it is (`schema_version` = `wp1-step3-v2`); all fields in `third_party\geoverify_main\API.md` and `pose_regions\README.md`. What matters for hand-over:

- Pixel coordinates are in the **original image size** (`img_size_a`, `img_size_b`, e.g. 8192 × 4096).
- Azimuth: `az = (x / W) × 360 − 180` (degrees); centre = 0, increasing to the right.
- `pose.R`, `pose.t`: 3D relative pose `x_b = R · x_a + t`; **`t` is a unit vector**; `cam_b_in_a` is the direction of camera B in A's frame. Not used by the pipeline.
- `quality.az_wedge_a_deg` / `az_wedge_b_deg`: the wedges of `scores.jsonl` (degrees).
- `regions`: co-visible regions; polygon vertices in original pixels.
- New scans include `inliers`; the Dollhouse pairs imported from the earlier experiment do not.

### 7.4 `pairs_inliers\<house>\<a>__<b>.json`: recomputed, with inliers

Written by `tools\recompute_inliers.py`; same format as §7.3 plus `inliers`: `[x_a, y_a, x_b, y_b, conf]` (original pixels; at most 2000 points). The review app and the door-opening step read these first. `consistency.jsonl` records whether the recomputation agrees with the original scan.

### 7.5 `roma_cache\*.npz`: raw RoMa matches

`matches`: N × 5, `[x_a, y_a, x_b, y_b, conf]` in original pixels. Only saves RoMa inference when recomputing; for Dollhouse only the 301 recomputed pairs have a cache.

### 7.6 `openings\<house>\<id>_mask.png`: door-opening masks

SAM3 "open doorway" result, 2048 × 1024, 1-bit, white = opening (a doorway you can see through). Stored **without dilation**; dilated when scoring (4 px left/right, 10 px up/down at 2048 px width).

### 7.7 `open_scores.jsonl`, `open_scores_meta.json`: door-aware wedge

One line per pair: `{"house", "a", "b", "open": {...}}`, only for pairs with a pose and ≥ 30 inliers.

| `open` field | Unit | Meaning |
|---|---|---|
| `n_keep` | points | Inliers left after removing those inside an opening (scaled to the full inlier count) |
| `door_frac` | 0–1 | Share of inliers inside either mask |
| `wedge_a`, `wedge_b`, `wedge_min` | degrees | Wedges recomputed from the remaining inliers |

`open_scores_meta.json`: prompt, mask size, dilation, counts.

### 7.8 `decisions.jsonl`: manual review record

Append-only, one line per decision; for one pair (either order of `a`, `b`) the last line wins.

| Field | Meaning |
|---|---|
| `house`, `a`, `b` | The pair |
| `relation` | `same_room`, `different_room`, `uncertain`; `null` = earlier decision cleared |
| `note`, `reviewer`, `at` | Note, reviewer, time |
| `score` | Scores at the time (as §7.2, including `open`) |
| `thresholds` | Review-app threshold and mode at the time |

### 7.9 `room_groups.json`: exported room groups (input of `2_pose_floorplan`)

Written by the review app's "Export room groups JSON"; the file name is up to the user.

| Field | Meaning |
|---|---|
| `schema_version` | 1 |
| `generator` | `room_candidate_finder` |
| `classifier` | Rule text, thresholds and GeoVerify version at export time |
| `exported_at` | Export time |
| `groups` | Per group: `room_id` (`<house>_room_001`, …), `task` (house), `floor` (empty), `room_label` (empty), `space_kind` (always `room`), `panorama_uuids`, `status` (`human_verified`), `notes`, `source_pairs`, `created_at` |
| `conflicts` | Pairs judged "different room" inside one group; when not empty, step 1 rejects the whole input |
| `pair_reviews` | Every pair's decision (`pair_id`, `uuid_a`, `uuid_b`, `relation`, …) |

A group is a connected component of "same room" decisions with at least 2 panoramas; a panorama on its own is in no group.

### 7.10 Dollhouse-only files (`scans\dollhouse\`)

| File | Meaning |
|---|---|
| `room_groups_dollhouse104.json` | **Input of the 104-group results** (43 houses, 278 panoramas, 301 pairs). The manual classification of `room_stitch_mvp` (`room_classification.json` format: more fields than §7.9, `room_id` = `task_<number>_<floor>_room_<n>`); `Run_Pipeline.ps1` takes it directly |
| `data_index.json` | Dollhouse label index; the pipeline reads only `panos[uuid].asteroom_name`. Other fields hold paths of the development computer and are not used |
| `room_groups_test_20261026.json` | Test export of the review app on 2026-10-26: 6 groups, 10 decisions, plain wedge 20°. **Not** the 104 groups |

### 7.11 Other files

| File | Meaning |
|---|---|
| `status.json`, `scan.log`, `RUNNING.lock` | Scan progress, log, lock of a running scan (delete it by hand only if no scan is running) |
| `openings.log`, `openings_todo.txt` | Door-opening log; the staging folders `openings_stage\`, `openings_raw\` are deleted when done |

---

## 8. Output formats: `2_pose_floorplan`

Location: `2_pose_floorplan\runs\<name>\`. Steps 4 and 5 create a new time-stamped folder on every run and never overwrite an older one; later steps read the newest.

### 8.1 Frozen inputs (step 1)

| File | Content |
|---|---|
| `classification_source.json` | Exact copy of the input groups file |
| `manifest.json` | `groups` (room_id, task, floor, space_kind, panorama_uuids, status); `images` (per image `id`, `label`, `house`, `status`, `path`, `sha256`, `size` [width, height], `error` when failed); `pairs`; `duplicate_images` (byte-identical images) |
| `run_config.json` | Inference settings (EMA, TTA, FP32), weights path and sha256, hashes of the frozen code, acceptance-rule parameters, hashes of the inputs |
| `preflight.json` | Groups, houses, images, pairs, group sizes, input errors, `status` (`ready_for_gpu` or `input_errors_preserved`) |

Each entry of `pairs`: `pair_id` (first 24 hex digits of `sha256("v2|" + room_id + "|" + a + "|" + b)`, `a` and `b` in alphabetical order), `group_id`, `a`, `b`, `directly_reviewed`, `duplicate_content`, `blind_review` (reserved).

### 8.2 CoVisPose results (step 2): `predictions\`

`<pair_id>.json`:

| Field | Unit | Meaning |
|---|---|---|
| `status` | — | `ok` or `error` (with `reason`) |
| `yaw` | **radians** | Decoder relative rotation |
| `t` | camera heights | Decoder relative translation [x, y]; position of camera A in B's frame |
| `geo_yaw`, `geo_t` | radians, camera heights | Pose of the geometric branch (Procrustes on the per-column correspondences) |
| `geo_weight` | — | Total correspondence weight of the geometric branch; about 0 = no usable correspondences |
| `heads_sha256` | — | Hash of the matching `.npz` (checked on resume) |
| `seconds` | s | Inference time; with the feature cache the first pair that uses a panorama is slower |

Convention: `p_B = R(yaw) · p_A + t`, averaged over 4 TTA views.

`<pair_id>.npz`: six float32 arrays of length 1024; column j has azimuth `θ_j = ((j + 0.5) / 1024 − 0.5) × 2π` (column 0 = left edge of the image).

| Array | Unit | Meaning |
|---|---|---|
| `covis_a`, `covis_b` | 0–1 | Probability that the column's content is visible in the other panorama |
| `corr_a`, `corr_b` | radians, [−π, π) | Azimuth of the column's content in the other panorama |
| `bon_a`, `bon_b` | radians | Depression angle b (> 0) of the floor–wall boundary; distance = 1 / tan(b) camera heights |

`inference_status.json`: completed and failed pairs. `run_env_<time>.json`: Python, torch, CUDA, device used (`cuda`/`cpu`), GPU name, CPU threads. `covispose_gpu_<time>.log`: console output of step 2 (UTF-16).

### 8.3 HorizonNet results (step 3): `horizonnet\`

`<panorama>.json`:

| Field | Meaning |
|---|---|
| `uv` | Room corners in **normalised image coordinates** (0–1, u right, v down), mapped back to the **original** image; two points per corner, ceiling (v < 0.5) and floor (v > 0.5) |
| `z0`, `z1` | HorizonNet internal heights: ceiling fixed at 50, floor negative; `z0 / −z1` = ceiling-to-camera ÷ camera-to-floor |
| `room_stitch_alignment` | Vanishing-point alignment: `vp`, rotation `R`, `tilt_deg`, `yaw_deg`, `det` |

`provenance_<time>_<batch>.json`: commands, weights hash, per-image alignment diagnostics. `status.json`: panoramas without a result (these use the CoVisPose outline instead).

### 8.4 Camera poses (step 4): `poses_<time>\`

**`groups\<group>.json`**

| Field | Meaning |
|---|---|
| `convention` | `p_root = R(yaw) p_u + t; camera height = 1; root = most-connected camera` |
| `method` | `pose_graph` (default) or `spanning_forest` (`-NoPoseGraph`). Folders written before 2026-11-19 have no `method` and are `spanning_forest` |
| `rule` | Acceptance rule and solver (G rule + direction links f = 0.30; maximum-weight spanning forest; then pose-graph optimisation, or none with `-NoPoseGraph`) |
| `components` | Connected components (panorama uuids) |
| `cameras` | Per camera: `label`, `component_root`, `is_root`, `yaw` (**radians**), `t` (camera heights, native coordinates), `flags`, `weak_edges`, `bad_loop` |
| `tree_edges` | Pairs used by the spanning tree |
| `pairs` | Per pair: `status`, `in_tree`, `inliers`, `inlier_fraction`, `correspondences`, `decoder_geometric_yaw_deg` (**degrees**), `decoder_geometric_t`, `yaw` (**radians**), `t`, `duplicate_content` |
| `loops` | Accepted links outside the tree: `yaw_deg` (**degrees**), `translation` (camera heights) = difference between the link's prediction and the final camera poses (tree poses, or optimised poses with the pose graph) |
| `flag_stats` | Error rates of the flags on ZInD test for this method (`pose_graph_zind_test_v1` for the pose graph, `pose_flags_v1` for the tree; §11.3); the viewer shows these numbers |
| `summary`, `pose_verified` | Group summary; always `false` |

Pair `status`: `accepted` (full link), `accepted_direction_only` (direction link: rotation reliable, position may be off by about 0.6 camera heights), `low_inlier_fraction`, `too_few_inliers`, `no_correspondences`, `decoder_geometric_disagree`.

Camera `flags`: `direction_link` (a direction link on the path), `weak_edge` (a weak link on the path), `loop_not_closing` (a loop through it misses by > 5° or > 0.30), `unverified` (on no loop; information only). The first three show as orange or a red "!" in the viewer.

**CSV (UTF-8 with BOM)**

| File | Columns |
|---|---|
| `cameras.csv` | group, house, camera, label, component_root, `yaw_deg` (**degrees**), `x`, `y` (native coordinates), direction_link, weak_edge, loop_not_closing |
| `pairs.csv` | group, pair_id, a, b, status, in_tree, inliers, inlier_fraction, correspondences, decoder_geometric_yaw_deg, decoder_geometric_t, duplicate_content (no poses) |
| `groups.csv` | group, house, n_images, pairs, accepted_full, accepted_direction, component_sizes, fully_connected, loop_max_yaw_deg, loop_max_translation, flagged_weak_or_loop |

`summary.json`: totals over all groups, with `method` (and, for the pose graph, `components_optimised`). `plots\<group>.png`: camera positions (drawn as (−x, y), a bird's-eye view).

`pose_graph_report.json` (pose graph only): the settings (`sigma_yaw_deg`, `sigma_t`, `sigma_t_direction`, `huber_f_scale`) and, per group, `components` (each optimised component: `root`, `cameras`, `pairs`, `optimised`, `nfev`, `cost_initial`, `cost_final`, `message`), `pairs_forest` and `pairs_pose_graph` (every accepted pair's `yaw_deg` / `translation` difference from the tree poses and from the optimised poses). Camera `flags` are computed from the raw pair predictions and are the same for both methods.

### 8.5 Floor plans (step 5): `plans_<time>\`

**`groups\<group>.json`** → `components`, per component:

| Field | Meaning |
|---|---|
| `panoramas`, `root` | Panoramas of the component, origin camera |
| `method` | `HN union` (≥ 2 panoramas: union of HorizonNet outlines) or `MX (single panorama)` |
| `outline_source` | Outline used in `plan` per panorama: `HN` or `MX` |
| `plan` | Merged plan: list of polygons, each the outer vertices `[[x, y], …]` (first point not repeated; **holes dropped**), origin-camera frame, camera heights |
| `area` | Area of `plan`, camera heights² (holes subtracted) |
| `covispose_plan` | Union of the CoVisPose outlines only (for comparison) |
| `outlines` | Three outlines per panorama, **each in its own camera frame** (not placed): `CV` (CoVisPose floor boundary, median over the pairs), `HN` (HorizonNet), `MX` (per column the farther of the two); `null` when unavailable |

To place an outline in the origin frame: `p_root = R(yaw) · p + t`, with that camera's `yaw` and `t` from `poses`.

`index.json`: per group `n`, component sizes, `hn_missing`. `plots\<group>.png`: plan (bird's-eye view).

### 8.6 Interactive viewer (step 6): `viewer\`

| File | Meaning |
|---|---|
| `viewer.html` | One web page; data and thumbnails (JPEG quality 70) are embedded; no server needed |
| `images\<panorama>.jpg.js` | Panorama (4096 × 2048, JPEG quality 82, base64), loaded by the page |

### 8.7 Verification

`verify_result.json` (`Verify.ps1`): result of each check and `passed`.

---

## 9. Units, coordinates and the bird's-eye view

### 9.1 Angle units

| Unit | Fields |
|---|---|
| radians | `yaw`, `geo_yaw` in `predictions\*.json`; `corr_*`, `bon_*` in `predictions\*.npz`; `cameras.*.yaw` and `pairs[].yaw` in `poses_*\groups\*.json` |
| degrees | Every field named `…_deg`; `yaw_deg` in `cameras.csv`; `loops[].yaw_deg`; every wedge of `1_room_finder`; GeoVerify `az_*`, `rot_angle_deg` |

### 9.2 Coordinate definitions

- **Panorama column and azimuth**: column j (width W) has azimuth θ = ((j + 0.5) / W − 0.5) × 2π; centre = 0, increasing to the right. GeoVerify uses the same direction in degrees.
- **Floor coordinates of one panorama** (CoVisPose native, same as ZInD): azimuth θ points along `(−sin θ, cos θ)`. The image centre is +y; things in the right half of the image have x < 0.
- **Relative pose** (`predictions`, `pairs` in `poses`): `p_B = R(yaw) · p_A + t`, `R(yaw) = [[cos, −sin], [sin, cos]]`; `t` is the position of camera A in B's frame.
- **Camera pose** (`cameras` in `poses`): `p_root = R(yaw) · p_u + t`; `t` is the position of camera u in the origin camera's frame, and the camera's forward direction there is `(−sin yaw, cos yaw)`. The origin is the camera with the most links in its component; there is no absolute scale or orientation.
- **Plans**: origin-camera frame, the same as `poses`.

### 9.3 Draw the bird's-eye view as (−x, y)

Native coordinates drawn with x to the right and y up give a mirrored picture. The correct bird's-eye view is (−x, y); the PNG files of `poses` and `plans` and the viewer all draw it that way.

How this was checked (2026-11-17):

1. **Derivation**: standing at the camera and facing the image centre (+y), something on the right of the image has x < 0 in native coordinates; only (−x, y) puts it on the right.
2. **Against ZInD's own floor plan**: the room outlines of ZInD house 0000, transformed with ZInD's own transformation, line up exactly with the floor plan drawn by Zillow (`floor_plans\floor_01.png`) with x to the right and y down. That is (x, −y), which differs from (−x, y) by a 180° rotation; neither is mirrored.

Assumption: the panoramas are ordinary equirectangular images (towards the right of the image = clockwise seen from above). ZInD is; Dollhouse has no independent floor plan to compare with, but CoVisPose, HorizonNet and the viewer use the same azimuth definition throughout, so the pipeline is self-consistent.

```python
import json, numpy as np, matplotlib.pyplot as plt
g = json.load(open(r'runs\<name>\plans_<time>\groups\<group>.json', encoding='utf-8'))
for comp in g['components']:
    for ring in comp['plan']:
        p = np.array(ring + ring[:1])
        plt.plot(-p[:, 0], p[:, 1])          # bird's-eye view: (-x, y)
plt.gca().set_aspect('equal'); plt.show()
```

---

## 10. Speed

All numbers come from logged runs. On a different GPU the time scales roughly with the inverse of GPU performance.

| Step | Unit | Time per unit | Measured on | Source |
|---|---|---|---|---|
| A. GeoVerify scan | per pair | **median 2.59 s**, mean 2.80 s (RoMa v2 2.54 s + verification 0.05 s); in practice **about 3 s** | RTX 4060, torch 2.10 cu130, 120 pairs | `1_room_finder\third_party\geoverify_main\environments\timing_torch210_n120.md` |
| A. Model loading | per scan | 6.4 s | same | same |
| A'. SAM3 door openings | per panorama | **median 3.19 s**, p90 3.75 s; first model load 1–2 min | RTX 3060, 828 panoramas | sam3-pano-seg `outputs\samroom_opendoorway.log` |
| A'. Door-aware wedge | per scan | seconds to tens of seconds (CPU) | — | — |
| 1. prepare | per run | about 7 s (mostly the sha256 of the weights) | 2-core CPU | packaging check 2026-11-05 |
| 2. CoVisPose | per pair | **about 0.48 s** (4-view TTA, FP32); loading about 7 s | RTX 3060; 301 pairs in 144 s (before the feature cache) | `C:\covispose\outputs\manual_room_groups_v2\` |
| 2. CoVisPose, ZInD test | 2,536 pairs | about 8.5 min (about 0.2 s per pair: rooms of 3–11 panoramas reuse the encoder features) | RTX 3060 | 2026-11-19, `runs\zind_test_posegraph` |
| 2. CoVisPose (CPU) | per pair | about 10 s without the feature cache; with it 2.8 s per pair + about 1.5 s once per panorama | 2-core CPU, sample | packaging checks 2026-11-05 and 2026-11-15 |
| 3. HorizonNet | per panorama | about 6.7 s (vanishing-point alignment about 5.4 s, mostly CPU, + inference about 1.3 s) | RTX 3060, 13 panoramas | `provenance_*.json` |
| 3. HorizonNet (CPU) | per panorama | preprocessing 6.64 s + inference 1.26 s | 2-core CPU, 404 panoramas | `evaluation\horizonnet_404\REPORT.md` §5 |
| 4. poses / 5. plans | per run | 1.7 s each (sample) | 2-core CPU | packaging check |
| 6. viewer | per run | 5.7 s (9 panoramas, about 0.6 s each) | 2-core CPU | packaging check |

Estimating a batch:

- Step A: n panoramas in a house → n(n−1)/2 pairs × about 3 s. 30 panoramas → 435 pairs → about 19–22 min. Dollhouse 60 houses, 7,985 pairs → about 6–7 h.
- Step A': about 3.2 s per panorama plus 1–2 min loading.
- Step 2: only pairs inside a group, k(k−1)/2 per group of k. Dollhouse 104 groups, 301 pairs → about 2.4 min on a GPU.
- Step 3: about 6.7–7.9 s per panorama. Dollhouse 278 panoramas → about 31–37 min.
- Steps 1, 4, 5, 6: usually under a minute.
- Only one GPU step at a time; a scan and the pipeline cannot run together.

Training speed: 3 GPUs (2× A5000 + A6000), about 36.8 pairs/s; one epoch about 4 h 13 min; 16 epochs 46 h 07 min.

---

## 11. Algorithms and parameters

### 11.1 Same-room candidate classifier (steps A, A')

Based on `1_room_finder\docs\benchmark\RESULTS.md` (2026-10-23) and `RESULTS_open_wedge.md` (2026-10-27). Data: Dollhouse, 60 houses, 1,232 indoor panoramas, 7,985 same-floor pairs; ground truth = manual groups plus a blind review (148 groups, 410 panoramas).

| Rule | Definition | Default threshold (`finder\config.py`) |
|---|---|---|
| Plain wedge | GeoVerify `verify_pair`, direct mode, default parameters: pose found, E inliers ≥ 30, 80% co-visible wedge ≥ threshold on both sides | 45° |
| Door-aware wedge | Inliers inside a SAM3 "open doorway" mask on side A or B are dropped (masks dilated 4 px ≈ 0.7° horizontally, 10 px vertically at 2048×1024); with ≥ 30 inliers left the 80% wedge is recomputed and must be ≥ threshold on both sides | 40° |

| Rule | Threshold | Precision | Recall | F1 | Candidate pairs |
|---|---:|---:|---:|---:|---:|
| Plain wedge | 45° | 0.507 | 0.538 | 0.522 | 445 |
| Door-aware wedge | 40° | 0.559 | 0.584 | 0.571 | 439 |

Leave-one-house-out cross-validation: F1 0.535 → 0.569. P/R at other thresholds: `1_room_finder\docs\REVIEW_APP.md` §5; the review app shows the expected values from `finder\calibration_dollhouse.json`.

### 11.2 CoVisPose inference (step 2)

- Weights: the **EMA weights** of epoch 15 of `server_c10neg` (last epoch of the cosine schedule).
- FP32, TF32 off; 4-view TTA (original, half-turn, mirrored, mirrored + half-turn), averaged after mapping the coordinates back.
- Input 512 × 1024; pose convention `p_B = R(yaw) p_A + t`, camera height = 1.
- Self-check parameters of the per-column correspondence (`RULES`): co-visibility threshold 0.5, maximum distance 10, residual ≤ 0.30, inliers ≥ 20, inlier fraction ≥ 0.40.
- Feature cache: each panorama is encoded once (plain + mirrored; the half-turn view is a shift of the features) and reused for every pair of its group. The first pair of each run is also computed without the cache and compared (yaw ≤ 0.05°, translation ≤ 0.002); on a mismatch the run continues without the cache. `--no-feature-cache` switches it off.
- Device: `--device auto|cuda|cpu` (default auto), `RCF_DEVICE`, `Run_Pipeline.ps1 -Cpu`.

### 11.3 Acceptance rules, pose graph and flags (step 4)

- **Accepted**: inliers ≥ 20 (residual ≤ 0.30), inlier fraction ≥ 0.40, and the decoder and geometric branches agree (≤ 5°, ≤ 0.25).
- **Direction link**: inlier fraction 0.30–0.40 and the two branches agree. The direction is reliable; the position may be off by about 0.6 camera heights (orange in the viewer).
- **Spanning tree**: maximum-weight spanning forest of the accepted links (weight 1 + f for full links, f for direction links); the poses are chained along it. The origin is the camera with the most links. With `-NoPoseGraph` this is the result (the method before 2026-11-19).
- **Pose-graph optimisation** (default since 2026-11-19, `pipeline\pose_graph.py`): the tree is the starting point. In every component with more accepted links than a tree needs (at least one loop), all accepted links are fitted together by robust least squares (`scipy.optimize.least_squares`, Huber loss). Residual of a link (a, b) with predicted pose (yaw, t): `wrap((yaw_a − yaw_b) − yaw) / 1°` and `(R(−yaw_b)(t_a − t_b) − t) / 0.06` (direction links: / 0.6), each × √weight with the tree's weights; the Huber loss turns linear at 3σ, so one wrong link cannot pull a whole loop. The pair poses are the decoder poses (4-view TTA), the estimator the acceptance rule is built on; 1° and 0.06 are about its median errors on ZInD test (`training\PHASE5_NOTES.md`). Components without a loop keep the tree poses exactly. No retraining: the step only uses the saved predictions.
- **Red "!" flag**: a weak edge on the path to the origin, or a loop that does not close (> 5° or > 0.30). Computed from the raw pair predictions, so the same cameras are flagged with both methods; a flag means "worth checking", not "certainly wrong".

**Evaluation on ZInD test (2026-11-19)**: every complete room with ≥ 3 panoramas in the ZInD test split (219 rooms, 158 houses, 1,087 panoramas, 2,536 pairs; input `evaluation\zind_test_posegraph\`, made by `scripts\make_zind_room_groups.py`), run through `prepare` and the unchanged CoVisPose step (RTX 3060), then `scripts\eval_pose_graph_zind.py`. The settings above were fixed before this test. Every two cameras that end up in the same component (2,294 camera pairs, the same for both methods) are compared with the ZInD relative pose; 183 rooms have a loop.

| Metric | Spanning tree | Pose graph |
|---|---|---|
| yaw median / mean | 1.96° / 4.54° | **1.49° / 3.36°** |
| translation median / mean (camera heights) | 0.148 / 0.297 | **0.132 / 0.249** |
| accuracy ≤ 0.1, 1° | 17.7% | 19.8% |
| accuracy ≤ 0.3, 5° | 68.7% | **76.2%** |
| accuracy ≤ 0.5, 10° | 87.0% | 89.7% |
| accuracy ≤ 1.0, 20° | 94.7% | 95.7% |
| wrong cameras (most of its pairs > 0.5 or > 10°) | 7.05% (74 / 1,050) | **6.10% (64 / 1,050)** |

- Rooms with a loop, bootstrap over rooms (pose graph − tree, 95% interval): translation mean −0.050 [−0.088, −0.023]; yaw mean −1.25° [−2.11°, −0.60°]; accuracy ≤ 0.3, 5° +7.9 points [+4.2, +11.4]; accuracy ≤ 0.1, 1° +2.1 points [−0.5, +4.7] (not significant). 116 rooms better, 67 worse.
- The gain grows with the room (accuracy ≤ 0.3, 5°): 84.1% → 86.4% with 3 panoramas, 72.7% → 79.3% with 5, 65.5% → 74.7% with 7 or more.
- Not every room improves: 125 camera pairs went from wrong to right and 64 from right to wrong (wrong = > 0.5 or > 10°).
- Flags (wrong if flagged / if not flagged): W 22.8% / 4.5% with the tree, 21.4% / 3.7% with the pose graph; L 10.3% / 5.0% → 7.1% / 5.5%; D (21 cameras) 14.3% → 28.6%. W still separates wrong cameras; L much less, because the optimisation repairs most of the loops it flags. The pose-graph outputs carry these numbers (`flag_stats`).
- Accepted pair poses on this set: yaw median 1.16°, translation median 0.093 (1,475 full links). This is worse than the `PHASE5_NOTES.md` numbers because the set has every pair of a room, also those with little overlap; it does not affect the comparison.
- Result per camera pair: `runs\zind_test_posegraph\pose_graph_zind_eval_20261119_173053.json`. To repeat: `powershell -ExecutionPolicy Bypass -File 2_pose_floorplan\evaluation\zind_test_posegraph\Run_ZInD_PoseGraph_Eval.ps1` (needs ZInD in `C:\zind_data`; about 10 min on an RTX 3060). For other settings, tune on the validation split (`make_zind_room_groups.py --split val`), not on test.

**Dollhouse 104 groups** (no ground truth): 22 components have a loop; 65 of 278 cameras move (median 0.031 camera heights / 0.73°; largest 0.93 / 23.7° in `task_69734_floor2_room_001`, whose loop Laundry–Pantry–Kitchen 2–Kitchen contradicts itself). A leave-one-pair-out check (each link on a loop predicted from the others) showed no clear difference, as expected with few, small loops. The bundled `runs\dollhouse104\` was made with the spanning tree; to get the pose-graph version, run `python -m pipeline.pose_graph`, `pipeline.plans` and `pipeline.viewer` on a copy of that run (the trial run itself was moved out of the package).

### 11.4 HorizonNet (step 3)

Official `preprocess.py` (LSD vanishing points, full 3D rotation alignment) → `inference.py` (MP3D weights) on the aligned image → corners mapped back to original-image coordinates. `pipeline\horizonnet.py` uses an ASCII-only temporary folder (`HN_ASCII_TMP`) and adds `--no_cuda` when the HorizonNet environment has no CUDA.

Accuracy on 404 Dollhouse panoramas (2026-11-05, `evaluation\horizonnet_404\REPORT.md`):

| Metric | Dollhouse, 404 panoramas | Official: MatterportLayout |
|---|---:|---:|
| 2D IoU (against each panorama's own `layout_gt`) | 51.4% (95% interval 46.7–56.5) | 81.71% |
| 3D IoU | 44.9% | 79.11% |
| 2D IoU with the scale factored out (shape) | 66.8% (63.9–70.2; median 77.2%) | — |
| Per-column relative wall-distance error (median) | 21.9% | — |

- Bedrooms, garages, bathrooms, storage rooms: shape IoU 0.70–0.78; open living rooms, kitchens, dining rooms, hallways about 0.5. Typical failures: the outline cuts through furniture or runs through a doorway into the next space.
- The absolute IoU is an underestimate: `layout_gt` has house-wide vertical offsets. In three houses (58498, 59868, 58599) a visual check confirmed wrong ground-truth wall lines; up to 7 houses may be affected. Evaluations based on this ground truth (for example the plan rule of §11.5) may be affected too.
- Corner post-processing is sensitive to floating-point differences: on another GPU or driver, individual panoramas can get a different number of corners (§16.2).

### 11.5 Floor plans (step 5)

Groups with several panoramas take the union of the HorizonNet outlines; a single panorama takes, per column, the farther of CoVisPose and HorizonNet. The rule was chosen by comparing several methods on the 36 Dollhouse groups with ground truth. The union tends to be too large and may leak through openings into the next room.

---

## 12. Models and weights

| File (in `2_pose_floorplan`) | sha256 | Origin | Training data and licence |
|---|---|---|---|
| `models\covispose\epoch_015.pt` | `d2daeb2af602a61c0c0d610c12d1f721fcc2b578fd59a45252cb295095775bed` | Trained in this project (`training\configs\server_c10neg.yaml`); raw weights, EMA, AdamW state, model settings | ZInD: **academic, non-commercial only** |
| `models\covispose\epoch_015_ema.pt` (optional, 0.35 GB) | `f87c3122…2436c` | EMA export of the same epoch (`C:\covispose\weights\covispose_c10neg_epoch015_ema.pt`) | Same |
| `models\horizonnet\resnet50_rnn__mp3d.pth` | `bdef2e337133f8f2937f99e7643cbef7be56a745a1fe3dad00d855bdcadd0c96` | Official HorizonNet release | Matterport3D layout annotations; Matterport3D and HorizonNet licences |

- When `epoch_015.pt` is absent and `epoch_015_ema.pt` is present, the pipeline uses the EMA export (`pipeline\common.py`). The full checkpoint is needed only to resume training, to use the raw (non-EMA) weights, or to export again.
- `CKPT_SHA` (both hashes) and `HN_SHA` in `pipeline\envcheck.py` are checked during Setup and before each run.

---

## 13. Training CoVisPose

### 13.1 Original training environment (`training\env\`)

| Item | Value |
|---|---|
| Operating system | Ubuntu 20.04 (Linux 5.15) |
| GPU | 2× RTX A5000 24 GB, 1× RTX A6000 48 GB (Ampere) |
| Driver / CUDA | 570.181 / CUDA 12.8, cuDNN 9.19 |
| Python | 3.11.16 (conda) |
| Main packages | torch 2.11.0+cu128, torchvision 0.26.0+cu128, numpy 1.26.4, scipy 1.17.1, pillow 12.3.0, PyYAML 6.0.3, shapely 2.1.2, matplotlib 3.11.2, tqdm 4.70.1 (full list `training\env\pip_freeze.txt`; strictest rebuild `conda_explicit.txt`) |
| Time and resources | 16 epochs, 46 h 07 min; 16,456 steps per epoch; peak 16.56 GiB on rank 0 |

```bash
# from the 2_pose_floorplan folder (Linux + NVIDIA)
conda create -y -p ~/envs/covispose python=3.11
conda activate ~/envs/covispose
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r training/env/requirements-train.txt
# identical versions: pip install -r training/env/pip_freeze.txt
```

This is the training environment. Inference on Windows uses the main environment (§3.2); both read the same `covispose\` code and checkpoint format.

### 13.2 Data (ZInD, not included)

Apply for a download at https://github.com/zillow/zind (academic, non-commercial licence).

| File | Use | Location |
|---|---|---|
| `zind_data\<4-digit house id>\zind_data.json`, `panos\*.jpg` | Panoramas and annotations | `ZIND_ROOT` or `--data-root` |
| `covis_scores.csv` | Co-visibility scores of panorama pairs (from ZInD) | `--covis-csv` |
| `zind_partition.json` | Official train/val/test split | `--partition` |

The pair lists of the original training are included (`training\pairs\`):

| File | Pairs |
|---|---:|
| `train_c10neg.csv` | 394,963 |
| `val_c10neg.csv` | 44,852 |
| `test_c10neg.csv` | 47,271 |
| `test_c10neg_adjacent.csv` | 6,208 |

Columns: `house_id, floor_id, room_a, pano_a, room_b, pano_b, covis_score, baseline, dyaw_deg, kind` (`kind` = `pos` or a zero-co-visibility negative). To rebuild:

```bash
python scripts/build_pair_index.py --data-root $ZIND_ROOT --covis-csv <covis_scores.csv> --partition <zind_partition.json> \
    --covis-min 0.10 --neg-ratio 0.4 --adj-per-house 40 --tag _c10neg      # written to training/pairs/
```

Rules: co-visibility ≥ 0.10, baseline ≥ 0.10 camera heights, positives to negatives 2.5:1, plus 40 adjacent negatives per house; every positive lies inside one `complete_room`.

### 13.3 Training commands (from `2_pose_floorplan`)

```bash
export ZIND_ROOT=/path/to/zind_data
python scripts/benchmark_vram.py --config training/configs/server_c10neg.yaml --data-throughput   # on a new machine
torchrun --standalone --nproc_per_node=3 scripts/train.py --config training/configs/server_c10neg.yaml   # 3 GPUs (original)
python scripts/train.py --config training/configs/server_c10neg.yaml                                    # 1 GPU
python scripts/train.py --config training/configs/smoke_cpu.yaml                                        # CPU smoke test (still needs ZInD)
```

Checkpoints go to `2_pose_floorplan/ckpt/server_c10neg/`; `resume: auto` continues after an interruption. Other options: `--max-epochs`, `--out-dir`, `--resume`, `--val-subsample`, `--quickval-every`, `--save-every-steps`, `--limit-steps`, `--data-root`.

### 13.4 Main settings (`training\configs\server_c10neg.yaml`)

| Category | Value |
|---|---|
| Model | ResNet-50 (IMAGENET1K_V2, downloaded), input 512×1024, 1024 column tokens per panorama, d_model 384, 8-layer Transformer, 8 heads, dim_ff 1536, drop_path 0.1; attention pose decoder (4 queries, 2 layers) + differentiable weighted Procrustes branch (2 IRLS iterations, c = 0.3); 88.2 M parameters |
| Loss weights | covis 1.0, corr 1.0, bon 2.0, pose_t 1.0, pose_yaw 1.0, geo_t 0.5, geo_yaw 0.5 (geometric terms ramped in between steps 9,000 and 18,000) |
| Optimisation | AdamW, lr 3e-4 (backbone ×0.1), weight decay 0.05 (none on norm/bias/embedding), gradient clipping 5.0, 4,000 warm-up steps, cosine decay to 1%, EMA 0.9998 |
| Batch | 8 pairs per GPU × 3 = 24 pairs per step; bf16 AMP, TF32, channels_last, fused AdamW |
| Augmentation | exact horizontal rotation, pair swap, gamma, 50% mirroring of both images; `fast_jpeg` halves the size in the DCT domain |
| Validation | full val split each epoch (raw and EMA); quickval every 2,000 steps |

- lr 5e-4 (square-root scaling with the GPU count) was unstable and was set back to 3e-4 (`training\SERVER_OPTIMIZATION.md`).
- bf16 needs Ampere or newer; on older GPUs change `amp_dtype` and turn off `tf32` and `fused_adamw`.
- On one GPU the effective batch drops from 24 to 8; adjust the lr and the number of steps.

### 13.5 Choosing the checkpoint and evaluating

- `best.pt` (lowest EMA val loss) picked epoch 13; **the pipeline uses epoch 15**, which is better on high-co-visibility pairs (`training\PAPER_COMPARISON.md`).
- Validation of epoch 15 (EMA, 44,852 pairs): co-visibility accuracy 0.940, IoU 0.626, correspondence error 15.9°, boundary error 1.71°, translation error 0.248, yaw error 4.69°.
- Test (`test_c10neg`, 47,271 pairs):

| Pose source | Median translation | Median yaw | Accuracy (0.5, 10°) |
|---|---:|---:|---:|
| Pose decoder | 0.085 | 1.23° | 0.888 |
| RANSAC @0.3 | 0.084 | 1.09° | 0.899 |
| Geometric branch (Procrustes) | 0.082 | 1.08° | 0.890 |

```bash
python scripts/evaluate_split.py --ckpt ckpt/server_c10neg/epoch_015.pt --split test \
    --pair-index training/pairs/test_c10neg.csv --data-root $ZIND_ROOT --weights ema --batch-size 16 --dump
python scripts/tune_ransac.py --ckpt <checkpoint> --pair-index training/pairs/val_c10neg.csv    # re-tune RANSAC for a new model
```

`run_scene_graph.py`, `tune_ransac.py` and `plot_curves.py` default to training folders of the development computer; give `--ckpt models\covispose\epoch_015.pt` when running them by hand.

### 13.6 Putting new weights into the pipeline

1. Put the new checkpoint (a full file saved by `train.py`, including `ema`) at `models\covispose\epoch_015.pt`, or change `CHECKPOINT` in `pipeline\common.py`.
2. Update `CKPT_SHA` in `pipeline\envcheck.py`.
3. `sample\reference\` was made with the old weights, so the CoVisPose comparison of `Verify.ps1` will fail; this is expected. Run the sample with the new weights, check the result, then update the reference.
4. The thresholds of §11.2 and §11.3 were set with epoch 15; re-check them.

### 13.7 Reproducibility

With AMP on, the cuDNN convolution backward pass uses atomics, so training is not bit-reproducible (about 1e-3 between identical passes); resuming is exact (`training\ASSUMPTIONS.md` A13). The original log in `training\reference_run\` can be used to compare training curves.

---

## 14. Training or fine-tuning HorizonNet

This project uses the official MP3D weights and did not retrain HorizonNet. To train or fine-tune, use the official code in `third_party\HorizonNet\`.

- **Environment**: the HorizonNet environment plus `pip install tensorboardX`. Official tested versions are PyTorch 1.8.1 / Python 3.7.6; the bundled torch 1.12.0 / Python 3.9 runs inference. Train on Linux if possible (`train.py` lacks the non-ASCII temporary-path handling of `inference.py`).
- **Data format** (`README_PREPARE_DATASET.md`): `<dataset>/img/*.png` (1024×512, aligned to the vanishing points) and `<dataset>/label_cor/*.txt` (one `x y` per line: ceiling and floor coordinates of each corner in turn). ZInD: `misc/zind_prepare_dataset.py`; Structured3D: `README_ST3D.md`.
- **Training** (from `2_pose_floorplan\third_party\HorizonNet`):

```bash
python train.py --id my_model --train_root_dir data/my/train --valid_root_dir data/my/valid --epochs 50
python train.py --id my_ft --pth ../../models/horizonnet/resnet50_rnn__mp3d.pth --train_root_dir ... --valid_root_dir ... --epochs 50
```

Options: `--batch_size_train` (8), `--lr` (1e-4), `--epochs` (300), `--backbone` (resnet50), `--freeze_earlier_blocks`, `--no_cuda`, `--multi_gpu`. Evaluation: `eval_general.py`, `eval_cuboid.py`.

- **Into the pipeline**: place it in `models\horizonnet\`, change `HN_CHECKPOINT` in `pipeline\common.py` and `HN_SHA` in `pipeline\envcheck.py`, and update the Verify reference.

---

## 15. GeoVerify and RoMa v2

- Code: `1_room_finder\third_party\geoverify_main`, GitHub main, commit `b5897093ce2f7d98f3540b93c5c0f93b87e83a5e` (`SOURCE_COMMIT.txt`). Only `verify_pair` in direct mode with default parameters is used.
- RoMa v2 is a pretrained model, not trained here. Its weights and the DINOv3 code are kept in `1_room_finder\models\torch_home`.
- The environment is built by `environments\setup_env.ps1` (see `environments\README.md`).

---

## 16. Tests and verification records

### 16.1 Tests

| Test | Command (main environment) | Needs |
|---|---|---|
| finder core | in `1_room_finder`: `python -m pytest tests\test_core.py` | no GPU, no window |
| CoVisPose unit tests | in `2_pose_floorplan`: `python -m pytest tests\covispose` | some tests need `ZIND_ROOT`, ZInD's own code (`transformations`, `ZIND_CODE`) or network for the ImageNet weights; they are skipped when missing (`tests\covispose\_requires.py`) |
| Environment | in `2_pose_floorplan`: `python -m pipeline.envcheck` | — (no CUDA only gives a note) |
| Pose graph (step 4) | in `2_pose_floorplan`: `python -m pytest tests\pipeline` | no data; synthetic groups |
| End to end | `2_pose_floorplan\Verify.ps1` | GPU (or `-Cpu` path) |
| Step 4 against ground truth | `2_pose_floorplan\evaluation\zind_test_posegraph\Run_ZInD_PoseGraph_Eval.ps1` | ZInD in `C:\zind_data`, GPU (§11.3) |

### 16.2 Verification records

| Date | What | Result |
|---|---|---|
| 2026-11-05 | Packaging check (cloud CPU): finder tests; CoVisPose tests; sample steps 1, 2, 4, 5, 6 on the CPU with the HorizonNet reference substituted; PowerShell syntax; weight hashes | finder 8/8; VERIFY PASSED, 12 pairs differ by 0.000°/0.0000; all scripts parse, hashes match |
| 2026-11-15 | `Setup.ps1` + `Verify.ps1` on the development computer (RTX 3060, environments present) | VERIFY PASSED |
| 2026-11-15 | Linux, no GPU, no ZInD, EMA export only: sample on the CPU; `tests\covispose` | VERIFY PASSED (difference 0); 30 passed, 11 skipped, 0 failed |
| 2026-11-15 | Feature cache, CPU | Bit-identical to the run without the cache; VERIFY PASSED |
| 2026-11-17 | Package `prepare` on `room_groups_dollhouse104.json` | Inputs identical to the original experiment: 301 pair ids, sha256 of all 278 images, labels, 104 group memberships |
| 2026-11-17 | Poses and plans recomputed from the original CoVisPose and HorizonNet results | 104/104 groups identical (camera difference 0, area difference 0%, same flags) |
| 2026-11-17 | Full GPU rerun of the 104 groups (RTX 3060, feature cache on) | CoVisPose 301/301 pairs difference 0; poses and plans 104/104 groups within the `Verify.ps1` tolerances; HorizonNet 277/278 panoramas the same. The exception (`08dded67…`) sits on a corner post-processing threshold: 10 corners became 8, plan area of its group −0.5%. Judged a floating-point borderline case, not a portability problem |
| After the feature cache was added | `Verify.ps1` on the development computer, GPU, feature cache on | VERIFY PASSED |
| 2026-11-19 | `pipeline.poses` (the tree path, unchanged by the pose-graph change) recomputed on the 104 groups | Identical to the 2026-11-17 output (largest difference 4e-16, floating point) |
| 2026-11-19 | Pose graph vs spanning tree on ZInD test against ground truth (219 rooms, 2,536 pairs, RTX 3060) | Accuracy ≤ 0.3, 5°: 68.7% → 76.2%; details §11.3 |
| 2026-11-19 | `Verify.ps1` reference: pose-graph poses and plans added (`groups`), tree reference kept (`groups_spanning_forest`); checked on Linux CPU with the CoVisPose and HorizonNet results of the 2026-11-15 GPU run | Pose-graph sample run and the 2026-11-15 tree run both VERIFY PASSED |
| 2026-11-19 | `tests\pipeline\test_pose_graph.py` | 4/4 passed |

The 104-group check script and reference run are outside the package: `C:\covispose\scripts\check_rcf_dollhouse104.py`, `C:\covispose\outputs\rcf_dollhouse104_check\`.

---

## 17. Licences

| Item | Origin | Restriction |
|---|---|---|
| `2_pose_floorplan\models\covispose\*.pt`, `2_pose_floorplan\training\pairs\` | Trained on / generated from ZInD (Zillow Indoor Dataset) | **Academic, non-commercial research only** (ZInD licence) |
| `2_pose_floorplan\models\horizonnet\resnet50_rnn__mp3d.pth` | Official HorizonNet, Matterport3D layout annotations | Matterport3D and HorizonNet licences |
| SAM3 code and weights in `1_room_finder\third_party\sam3-pano-seg\` | Meta | SAM License: redistribution allowed; `third_party\sam3\LICENSE` must be included; research published with SAM3 results must state that SAM Materials were used |
| Dollhouse dataset (`1_room_finder\data\dollhouse`) | — | As set by the data provider |

The recipient must know that the CoVisPose weights are for academic, non-commercial use only. Commercial use needs a separate licence, or retraining on other data (§13). The public GitHub repository and its release carry the same restriction: the CoVisPose weights and `training\pairs` are derived from ZInD and are subject to the ZInD Terms of Use (https://bridgedataoutput.com/zillowterms); the ZInD data itself (images, annotations, `gt.json`) is not redistributed.

---

## 18. Known conditions, limitations and open items

**Open items (not handled by the scripts)**

| Item | Details | What to do |
|---|---|---|
| Large package | Dollhouse panoramas about 15 GB; whole folder about 26 GB without the environments | Delete `1_room_finder\data\dollhouse` and `1_room_finder\scans\dollhouse` when not needed |
| Paths only reached on another computer | Rebuilding an environment that does not start, getting Python 3.9 through uv, skipping the GPU check without `nvidia-smi` have not been run yet | Confirm with `Setup.ps1` + `Verify.ps1` on the target computer |
| GPU check change of 2026-11-17 | `Assert-GpuFree` now reports only Python programs; checked by reading, a run on Windows is pending | `Verify.ps1` should run without a `(y/N)` prompt |
| Step A in the current layout | Since the split into two sub-folders (2026-11-05), import, scan and door openings have not been run | Try `Import_Dataset.ps1 -Limit 5` first |
| Manual scripts with development-computer paths | `run_scene_graph.py`, `tune_ransac.py`, `plot_curves.py` | Give `--ckpt models\covispose\epoch_015.pt` |
| ZInD not included | Training and 11 tests need it; it may not be redistributed | §13.2 |
| `Verify.ps1` with the pose graph not yet run on Windows | The new reference was checked with the Linux CPU path (§16.2) | Run `Verify.ps1` once on the target computer |

Backups of earlier versions of the changed files are on the development computer in `C:\HorizonNet\_待刪除\room_candidate_finder_舊設定\` (logged in `C:\HorizonNet\_待刪除\搬移紀錄.tsv`).

**Environment**

1. The path must be ASCII-only.
2. CoVisPose uses CUDA by default and falls back to the CPU (same results). The GeoVerify scan (step A) and SAM3 need a GPU; without a suitable GPU only steps 1–6 can run, on an existing groups file.
3. The HorizonNet environment is torch 1.12 + CUDA 11.3, up to RTX 40. On RTX 50 use `-HnCpu` (slower).
4. The main environment needs a driver that supports CUDA 13.0.
5. One program at a time on the GPU. `Run_Pipeline.ps1` checks this; for `tools\recompute_inliers.py` check it yourself.
6. `Setup.ps1` needs network access; the review app needs a Python 3 with Tkinter.
7. After copying or moving the folder, run `Setup.ps1` once (environments and SAM3 paths). If `1_room_finder\third_party\sam3-pano-seg` is deleted, the plain wedge is used.

**Data**

8. Panoramas must be exactly 2:1 equirectangular (§6.1); all panoramas of a house are paired.
9. `scans\dollhouse\manifest.json` looks in `data\dollhouse` first and reads `root_abs` (`C:/datasets`) only when that is missing.
10. Exported groups are all `room`; open spaces and floors are not distinguished.

**Method**

11. **Adjacent rooms seen through a doorway cannot be posed**: CoVisPose had no "visible through a door" positives in training (every positive lies inside one `complete_room`, A8); on ZInD test, 814 of 834 door-connected pairs were predicted as zero co-visibility. The pipeline therefore works per same-room group and does not stitch whole floors.
12. Most remaining errors of the same-room classifier come from open spaces (living room, dining room and kitchen connected); there is no doorway, so the door-aware wedge does not help.
13. No pose ground truth for your data: the outputs only reflect internal consistency. Step 4 (spanning tree vs pose graph) was measured against ground truth on ZInD test only (§11.3); the pose graph is better on average but can be worse in a single room, mostly where a loop contradicts itself (such cameras keep the `loop_not_closing` flag).
14. Plans use the union of outlines: too large, may leak through openings.
15. The P/R table of the wedge thresholds comes from Dollhouse; treat it as a guide on very different data.
16. ZInD is unfurnished and is CoVisPose's training source, so ZInD numbers are optimistic; for furnished data use the Dollhouse results.
17. HorizonNet shape IoU about 0.67 on Dollhouse, about 0.5 in open spaces; numbers based on `layout_gt` are only indicative until it is corrected.

**Code**

18. `scripts\manual_room_groups_v2.py` contains only `run`. Its source and the `covispose\` code are hashed into `run_config.json`; a run prepared before a code change cannot be resumed (§0 item 14).
19. ZInD default paths in the training and evaluation scripts (`C:\zind_data` etc.) are only defaults; set `ZIND_ROOT` or pass arguments.

---

## 19. Change history

- **Integration**: the folder originally held only the same-room scan and the review app. Added CoVisPose (code and epoch-15 weights), HorizonNet (official code and MP3D weights), steps 1–6, the sample and the verification, from the 2026-10-26 room_pose_pipeline package, plus the CoVisPose training code, settings, pair lists and environment specification. This folder became the root; every script path was adjusted.
- **2026-11-05**: split into `1_room_finder\` and `2_pose_floorplan\`; the root keeps `Setup.ps1`, `tools\_env.ps1` and the documents. `Run_Pipeline.ps1` resolves `-Groups`, `-ScanDir`, `-Index` against the current folder. Settings unrelated to the current configuration (backup files, benchmark import tools, fallback environments outside the package, the old SAM3 location, unused settings files) were moved to `C:\HorizonNet\_待刪除\room_candidate_finder_舊設定\`. SAM3 became a bundled trimmed copy. HorizonNet's temporary folder default and the training pair-list path (`training\pairs`) changed.
- **2026-11-09**: English as the primary language with a Chinese option (review app, viewer, documents `*_ZH.md`). Launchers merged (`Start_Scan.ps1` → `Import_Dataset.ps1 -NoCopy`; setup scripts → `Setup.ps1`; `Start_Recompute_Inliers.ps1` removed). `manual_room_groups_v2.py` keeps only `run`. Third-party code unchanged.
- **2026-11-15 (portability)**: `Get-MainPy` / `Get-HnPy` / `Get-ReviewPy` test-run the environments; `Setup.ps1` rebuilds one that does not start; `Start_Review.bat` checks its environment. RoMa v2 weights and DINOv3 code moved into `1_room_finder\models\torch_home`. uv fallback for Python 3.9; GPU check skipped without `nvidia-smi`. Dollhouse panoramas copied back into `data\dollhouse`.
- **2026-11-15 (CPU fallback, EMA export, tests)**: `--device auto|cuda|cpu`, `RCF_DEVICE`, `Run_Pipeline.ps1 -Cpu`; HorizonNet `--no_cuda` added automatically; `envcheck` only notes a missing CUDA. The EMA export alone is accepted. Tests needing ZInD or the network are skipped.
- **2026-11-15 (feature cache)**: step 2 encodes each panorama once (§11.2). CPU: 108 s → 55 s on the sample; bit-identical output.
- **2026-11-17**: Dollhouse group files renamed (`room_groups_dollhouse104.json` added; the 6-group test export renamed `room_groups_test_20261026.json`; `data_index.json` added). Output formats documented and checked against the code; the bird's-eye view convention verified. `Assert-GpuFree` reports only Python programs (Windows desktop programs listed by `nvidia-smi` no longer cause a prompt). README, NOTICE, SPEC and OUTPUT_FORMAT merged into this document.
- **2026-11-19 (pose graph)**: step 4 optimises every component with a loop (`pipeline\pose_graph.py`, §11.3) instead of using the spanning tree alone; `Run_Pipeline.ps1 -NoPoseGraph` keeps the old method. `pipeline\poses.py` takes the solver, rule text, flag statistics and method as arguments (default output unchanged) and writes `method`; the viewer reads the method and the flag statistics from the poses folder. `Verify.ps1` reference holds both methods (`pipeline\verify.py` picks by `method`; `--make-reference` keeps the other method and the pair poses, `--pairs` rewrites them). Added `scripts\make_zind_room_groups.py`, `scripts\eval_pose_graph_zind.py`, `evaluation\zind_test_posegraph\`, `tests\pipeline\test_pose_graph.py`. Models and step 2 unchanged; existing runs can be resumed.
- **2026-11-19 (GitHub)**: `.gitignore` / `.gitattributes` for the public repository (environments, weights, SAM3, panoramas except the sample, ZInD ground truth and predictions are left out; §0 "Getting it from GitHub"); download notes in `models\covispose\` and `models\horizonnet\`; `envcheck` says where to get a missing checkpoint; `Run_ZInD_PoseGraph_Eval.ps1` regenerates `gt.json` from ZInD when missing (`-ZindData`, `-Partition`; checked byte-identical to the file used on 2026-11-19).

Backups of each change are in `C:\HorizonNet\_待刪除\room_candidate_finder_舊設定\` on the development computer.

---

## 20. Other documents

| Document | Contents |
|---|---|
| `1_room_finder\CODE_OVERVIEW.md`, `2_pose_floorplan\CODE_OVERVIEW.md` | Every program of the folder: role in the flow, main functions, external libraries, tests |
| `1_room_finder\docs\REVIEW_APP.md` | Review app: operation, choosing the threshold, FAQ |
| `1_room_finder\docs\benchmark\RESULTS.md`, `RESULTS_open_wedge.md`, `EVALUATION_STANDARD.md` | Same-room classifier comparison, door-aware wedge, evaluation standard |
| `2_pose_floorplan\evaluation\horizonnet_404\REPORT.md`, `PLAN.md` | HorizonNet outline evaluation on 404 panoramas and its pre-registered plan |
| `2_pose_floorplan\runs\zind_test_posegraph\pose_graph_zind_eval_*.json` | Step-4 evaluation on ZInD test: summary and the error of every camera pair for both methods (§11.3) |
| `2_pose_floorplan\training\SERVER_OPTIMIZATION.md` | Every change to the server training settings and why |
| `2_pose_floorplan\training\PAPER_COMPARISON.md` | Comparison with the CoVisPose paper, why epoch 15 |
| `2_pose_floorplan\training\ASSUMPTIONS.md` | Design assumptions A1–A18 |
| `2_pose_floorplan\training\PHASE1_NOTES.md`, `PHASE5_NOTES.md` | ZInD data pipeline, server training results |
| `2_pose_floorplan\third_party\HorizonNet\README*.md` | Official HorizonNet documentation |
| `1_room_finder\third_party\geoverify_main\README.md`, `API.md` | GeoVerify documentation |

Chinese versions (`*_ZH.md`) exist for CODE_OVERVIEW, REVIEW_APP, the benchmark documents, the HorizonNet report and plan, and `training\env\HOW_TO_RECREATE`. The training notes are English only.
