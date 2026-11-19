# 1_room_finder code overview

This document outlines every program in `1_room_finder\`: where it sits in the flow, its main functions and what they do. Usage, input and output formats are in the root [README.md](../README.md) (§5–§7). Chinese version: [CODE_OVERVIEW_ZH.md](CODE_OVERVIEW_ZH.md).

This folder does the same-room screening: geometric verification of every pair of panoramas (step A), a door-aware correction of the co-visible wedge with SAM3 opening masks (step A'), and manual review with export of the room groups (step B).

---

## 1. Call structure

```
Import_Dataset.ps1 ─┬─ robocopy                     copy the panoramas to data\<name>
  (steps A + A')     ├─ python -m finder.scan       GeoVerify on every pair → scans\<name>\
                    └─ Start_Openings.ps1          (runs automatically when SAM3 is present)

Start_Openings.ps1 ─┬─ python -m finder.openings --list-todo   panoramas without a mask
  (step A')          ├─ tools\fix_sam3_env.ps1                  re-point the SAM3 environment
                    ├─ sam3_pano.seg.unified (SAM3 env)         "open doorway" segmentation
                    ├─ tools\import_openings.py                 shrink masks to 2048×1024 into openings\
                    └─ python -m finder.openings               door-aware wedge → open_scores.jsonl

Start_Review.bat ──── python -m finder.app         review app (Tkinter)
  (step B)               └─ finder.model → finder.scores / finder.grouping / finder.i18n

By hand: tools\recompute_inliers.py               store per-point inliers for an older scan
```

Every PowerShell script first loads the root `tools\_env.ps1`, which finds and test-runs the environments (`Get-MainPy`, `Get-ReviewPy`), finds SAM3 (`Get-Sam3Root`) and checks whether the GPU is busy (`Assert-GpuFree`).

---

## 2. Launch scripts

| File | What it does |
|---|---|
| `Import_Dataset.ps1` | Parameters `-Source`, `-Name`, `-NoCopy`, `-Limit`, `-NoOpenings`. Copies the source folder to `data\<Name>` with robocopy, runs `finder.scan --root data\<Name> --out scans\<Name>` in the main environment, then calls `Start_Openings.ps1` when SAM3 is found |
| `Start_Openings.ps1` | Parameters `-Scan`, `-Sam3Root`. Five steps: ① `finder.openings --list-todo` lists panoramas without a mask; ② hard-links them (copies if that fails) into `openings_stage\` as `<house>__<id>`; ③ runs `sam3_pano.seg.unified --categories door --door-prompt-override "open doorway" --pano-vote --mask-only --skip-existing` in the SAM3 environment; ④ `tools\import_openings.py` imports the masks, then the staging folders are deleted; ⑤ `finder.openings` computes the door-aware wedge |
| `Start_Review.bat` | Uses `RCF_REVIEW_PYTHON` or `.venv`, checks that `tkinter` and `PIL` import, then runs `python -m finder.app [--scan <folder>]`. Asks for `Setup.ps1` when the environment is missing or does not start |

---

## 3. The `finder\` package

### 3.1 `config.py`: paths and defaults

No functions, only constants:

- Paths: `ROOT`, `GEOVERIFY_MAIN` (`RCF_GEOVERIFY_MAIN` overrides), `DATA_DIR`, `SCANS_DIR`, `SAM3_ROOT` (`RCF_SAM3_ROOT` overrides).
- `TORCH_HOME`: when not set, points at `models\torch_home` so the RoMa v2 weights travel with the folder. If the package has no copy yet but this computer's default cache has one, that cache is used.
- Scan: `IMAGE_EXTS`, `EXCLUDE_DIRS`.
- Classifier: `DEFAULT_WEDGE_DEG = 45`, `DEFAULT_MIN_INLIERS = 30`, `DEFAULT_OPEN_WEDGE_DEG = 40`, `CALIBRATION_FILE`.
- Opening masks: `OPEN_MASK_W/H = 2048×1024`, `OPEN_DILATE_PX = 4`, `OPEN_PROMPT = "open doorway"`.

### 3.2 `dataset.py`: houses and panoramas

| Function | What it does |
|---|---|
| `scan_houses(root)` | Each sub-folder is a house; without sub-folders the folder itself is one house. Keeps images with a width/height ratio of 2 ± 0.02, removes duplicates by content hash, appends `_2`, `_3` to equal names with different content. Returns `{house: [{id, path, w, h}]}` and `skipped` with reasons; houses with fewer than 2 panoramas are skipped |
| `_images_under(folder)` | Lists image files recursively, skipping `EXCLUDE_DIRS` |
| `_sha(path)` | Hash of the first MiB plus the file size, used to spot duplicates |
| `all_pairs(items)` | Every unordered pair in a fixed order |

### 3.3 `scan.py`: GeoVerify scan (step A, GPU)

| Function | What it does |
|---|---|
| `_import_geoverify()` | Imports `roma_geo` from `config.GEOVERIFY_MAIN`, checks it is the package's copy, reads `SOURCE_COMMIT.txt` |
| `main()` | Arguments `--root`, `--out`, `--house`, `--limit`. ① Creates `RUNNING.lock` so two scans cannot run together; ② `dataset.scan_houses` → `manifest.json` (`root` stored relative, plus `root_abs`); ③ `roma_geo.get_matcher("roma_v2", …, cache)` loads RoMa v2; ④ for each pair `roma_geo.verify_pair(a, b, mode="direct", keep_inliers=True)`, written to `pairs\<house>\<a>__<b>.json` (temporary file, then replaced), and its summary from `scores.summarize` appended to `scores.jsonl`; ⑤ errors go to `errors.jsonl`, `status.json` is updated every 25 pairs. Pairs already in `scores.jsonl` are skipped, so the scan can be resumed |

### 3.4 `scores.py`: score summary and classifier

| Function / class | What it does |
|---|---|
| `summarize(g)` | `verify_pair` result → `has_pose`, `n_inlier`, `inlier_rate`, `wedge_a/b`, `wedge_min`, `n_regions` |
| `load_scores(scan)` | Reads `scores.jsonl`, last line per pair wins → `{house: {(a, b): summary}}` |
| `uses_open(s, mode)` | Whether this pair is judged with the door-aware wedge (open mode and an `open` score present) |
| `active_wedge(s, mode)` | The wedge compared in this mode (plain or door-aware) |
| `passes(s, wedge_deg, min_inliers, mode)` | Classifier: pose found, inliers ≥ threshold, both wedges ≥ threshold |
| `Calibration` | Reads `calibration_dollhouse.json`; `expected()` returns the expected precision/recall of the nearest row for a slider position |

### 3.5 `openings.py`: door-aware wedge (step A')

| Function | What it does |
|---|---|
| `wedge_width(az, frac=0.8)` | Narrowest arc (degrees) holding 80% of the azimuths, same as GeoVerify's `az_wedge` |
| `load_mask(path)` | Opening mask → bool array; None when the file does not exist |
| `dilate(mask, r=4)` | Dilation: r px left/right (across the seam) and cumulative vertical shifts 1…r, i.e. r(r+1)/2 = 10 px up/down. Must stay as calibrated, or the calibration table no longer applies |
| `open_wedge(inliers, size, mask_a, mask_b, n_inlier)` | Scales the inliers to 2048×1024, drops those inside the A or B mask, recomputes `wedge_a/b/min`, `n_keep`, `door_frac` from the rest (GeoVerify keeps at most 2000 points; `n_keep` is scaled back with `n_inlier`) |
| `_pair_inliers(scan, house, a, b)` | Inliers of one pair: `pairs_inliers\` first, then `pairs\` or `--npz-dir`; handles files stored as `b__a` |
| `compute(scan)` | Door-aware wedge for every pair that can pass (pose, ≥ 30 inliers) → `open_scores.jsonl`, `open_scores_meta.json` |
| `list_todo(scan, out)` | Panoramas without a mask (`house <TAB> id <TAB> absolute path`), for `Start_Openings.ps1` |
| `mask_path(scan, house, id)` | Mask location `openings\<house>\<id>_mask.png` |

### 3.6 `grouping.py`: decisions → room groups

| Function / class | What it does |
|---|---|
| `pair_key(a, b)` | Order-independent pair key |
| `Decisions` | Manages `decisions.jsonl` (append-only, latest wins); `set()` writes a decision, `get()` reads one |
| `build_groups(house, panos, decisions)` | Union-find over `same_room` decisions into groups of ≥ 2; `different_room` pairs inside one group are returned as `conflicts` |
| `export_groups(path, houses, decisions, thresholds, source_commit)` | Writes `room_groups.json` (`groups`, `conflicts`, `pair_reviews`, `classifier`, …) |

### 3.7 `model.py`: review state without a user interface

`ReviewModel` combines the scan results, thresholds and decisions; `app.py` only displays them.

| Method | What it does |
|---|---|
| `set_mode(mode)`, `thresholds()` | Switch between the plain wedge (`plain`) and the door-aware wedge (`open`); the threshold jumps to that mode's default |
| `candidates()` | Pairs passing the current thresholds, widest narrower wedge first |
| `scanned_pairs()`, `expected_pairs()`, `progress()` | Scan and review progress |
| `panos_without_candidates()` | Panoramas without any candidate pair |
| `expected_quality()` | Expected P/R of the current threshold (`Calibration`) |
| `decide(a, b, relation, note)` | Records a decision |
| `groups()`, `export(path)` | Current room groups; JSON export |
| `pair_detail(a, b)` | GeoVerify result of one pair for the inlier overlay (A and B swapped by `_swap` when needed) |
| `opening_mask_path(house, id)`, `image_path(house, id)` | Mask location; panorama location (relative `root` first, `root_abs` as fallback) |

### 3.8 `app.py`: review app (Tkinter)

Class `App` builds the window; `main()` parses `--scan`, `--reviewer`, `--lang`. Main methods:

| Method | What it does |
|---|---|
| `_build()` | Layout: top bar with house selector, door-aware checkbox, wedge and inlier thresholds, export and language buttons; on the left the candidate list (Treeview) and the group list; on the right the canvases of panoramas A and B and the decision buttons. Keys: S same room, D different room, U uncertain, C clear, Down/N next, Up/P previous |
| `_mode_changed()`, `_thresholds_changed()`, `refresh()` | Re-filter the candidates when the mode or a threshold changes |
| `_render()`, `_schedule_render()` | Fit each panorama into its canvas at 2:1 without cropping |
| `_overlay()`, `_through_opening()`, `_load_mask()`, `_overlay_text()` | Draw GeoVerify inliers (same colour = matching points in A and B), mark those inside an opening, show the scores |
| `zoom(k)` | Show A or B as large as the screen allows (Esc or click closes) |
| `decide()`, `move()` | Write a decision (same / different / uncertain / clear), go to the previous or next pair |
| `render_groups()`, `export()` | Show the current groups; export `room_groups.json` |
| `toggle_lang()` | Switch English/Chinese and rebuild the window (decisions are already on disk) |

### 3.9 `i18n.py`: interface text

`set_lang()` and `get_lang()` set and read the language (default from `RCF_LANG`); `T(key)` returns a string in the current language (all text is in the `STRINGS` table, English and Chinese per key); `rel_text(relation)` turns a decision into display text.

---

## 4. `tools\`

| File | What it does |
|---|---|
| `fix_sam3_env.ps1` | Rewrites the absolute paths in the bundled SAM3 virtual environment (`home` in `pyvenv.cfg`, two `__editable___*_finder.py`) to the current location; does nothing when they already match. Called by `Setup.ps1` and `Start_Openings.ps1` |
| `import_openings.py` | `main()`: shrinks SAM3's full-size masks to 2048×1024 1-bit PNG in `openings\<house>\<id>_mask.png` (`--src` may be repeated, `--budget-s` limits the run time) |
| `recompute_inliers.py` | `main()`: re-runs `roma_geo.verify_pair(..., keep_inliers=True)` on candidate pairs of an existing scan, writes `pairs_inliers\`, and records in `consistency.jsonl` whether the result matches the original. Options `--house`, `--min-wedge`, `--min-inliers`, `--limit`, `--groups`, `--reviewed` |

---

## 5. External code and libraries

| Source | Parts used | Purpose |
|---|---|---|
| GeoVerify (`third_party\geoverify_main`, `roma_geo`) | `get_matcher("roma_v2", …)`, `verify_pair(..., mode="direct")` | RoMa v2 dense matching, essential-matrix verification, co-visible wedges and regions |
| RoMa v2 (through GeoVerify) | weights in `models\torch_home` | Feature matching |
| sam3-pano-seg (`third_party\sam3-pano-seg`) | `python -m sam3_pano.seg.unified` (in its own Python 3.12 environment) | "open doorway" segmentation of panoramas |
| numpy | — | Azimuths, wedges, mask operations |
| Pillow | `Image.open`, resizing, mask I/O | Image sizes, panorama display, mask conversion |
| tkinter | — | Review app interface |

The third-party code (GeoVerify, sam3-pano-seg) is unchanged.

---

## 6. Tests

`tests\test_core.py` (`python -m pytest tests\test_core.py`, no GPU or window) tests the core on synthetic data:

| Test | Covers |
|---|---|
| `test_scan_houses` | Finding houses and panoramas, skipping non-2:1 images, duplicates, numbered equal names |
| `test_classifier`, `test_open_classifier` | Plain and door-aware wedge rules |
| `test_calibration_lookup` | P/R calibration lookup |
| `test_model_filter_groups_export_resume` | Candidate filtering, decisions, groups, export, resuming after reopening |
| `test_open_mode_in_model`, `test_open_wedge_and_compute` | Door-aware mode and the computation of `open_scores.jsonl` |
| `test_moved_folder_still_works` | Relative paths still work after the whole folder is moved |
