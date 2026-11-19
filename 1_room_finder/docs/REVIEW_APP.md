# Review app and same-room scan: user guide

This tool takes a set of panoramas, automatically finds pairs that are "probably the same room", lets a person set the screening threshold by confidence and confirm the pairs one by one, and finally turns the confirmed pairs into room groups and exports them.

This document covers the scan and the review app of `1_room_finder\` (same-room screening). Installation, the full flow and the specification are in `README.md` at the package root (input and output formats: §6–§9). Chinese version: [REVIEW_APP_ZH.md](REVIEW_APP_ZH.md).

**Commands and paths below are relative to `1_room_finder\`** (`cd 1_room_finder` first); `Setup.ps1` is in the package root.

---

## 1. Quick start

1. Run `Setup.ps1` once in the package root (builds the scan and review environments, see README.md §4).
2. Double-click `Start_Review.bat`.
3. In the dialog, choose a dataset folder under `scans\` (for example the bundled `scans\dollhouse`; it needs the Dollhouse panoramas, see section 8).
4. Follow the instructions at the top of the window (details in section 4).

The app starts in English. The **中文** button at the top right switches to Chinese (and **English** switches back); to start in Chinese, set `RCF_LANG=zh` before `Start_Review.bat`, or run `python -m finder.app --lang zh`.

---

## 2. Files

| Location | Contents |
|---|---|
| `Start_Review.bat` | opens the review app |
| `Import_Dataset.ps1` | imports new panoramas and scans them (`-NoCopy`: scan only, data already in `data\`) |
| `Start_Openings.ps1` | SAM3 door-opening detection and door-aware wedge |
| `tools\recompute_inliers.py` | recomputes the per-point inliers of candidate pairs for the review screen (manual, section 9) |
| `finder\` | the program; `finder\config.py` holds the default thresholds; `finder\i18n.py` the English/Chinese texts |
| `third_party\geoverify_main\` | GeoVerify geometric verification (GitHub main, version b589709) |
| `scans\dollhouse\` | finished Dollhouse scan results and review records |

---

## 3. Environments

Built in one go by `Setup.ps1` in the package root: the review app uses `1_room_finder\.venv\` (Python + Tkinter + Pillow, no GPU needed), the scan uses `1_room_finder\third_party\geoverify_main\venv\` (Python 3.11 + PyTorch CUDA 13.0 + RoMa v2, NVIDIA GPU needed). To use environments elsewhere, point `RCF_REVIEW_PYTHON` and `RCF_MAIN_PYTHON` at their python.exe (README.md §3.2).

---

## 4. Using the review app

### 4.1 Opening

- Double-click `Start_Review.bat` and choose a dataset folder under `scans\` (it must contain `manifest.json`).
- Or name it directly: `Start_Review.bat scans\dollhouse`

### 4.2 Screen layout

| Area | Contents |
|---|---|
| Light-blue bar at the top | instructions |
| Toolbar (two rows) | House menu, "Door-aware wedge" switch, wedge slider, inlier threshold, Export button, language button; second row: "Hide judged", "Show inliers", expected precision/recall |
| List on the left | candidate pairs at the current thresholds (Panorama A, Panorama B, narrower wedge, inliers, door-inlier share, decision) |
| Bottom left | current room groups of this house and conflict warnings |
| Right | the two panoramas of the selected pair (A above, B below) and the decision buttons |
| Status bar | candidates, judged pairs, panoramas without candidates, scanned pairs |

### 4.3 Steps

1. **Pick a house**: the "House" menu at the top left.
2. **Set the thresholds** (the defaults are fine to start with, see section 5):
   - **Door-aware wedge (SAM3 openings)** (on by default when door data exists): inliers seen through a door opening do not count toward the wedge, which cuts "next room seen through a door" mistakes. Switching moves the slider to the default of that mode (door-aware 40°, plain 45°). On the panoramas, cyan marks the SAM3 door openings and grey hollow dots the excluded inliers.
   - **Both wedges >=** (default 40°; 45° without door data): the main confidence threshold. Left = more candidates, more same-room pairs found but more mistakes; right = fewer but more reliable candidates.
   - **inliers >=** (default 30): a secondary threshold; raising it removes a few more mistakes, usually leave it.
   - Pairs below the thresholds are not listed. A panorama with no candidate at all is treated as a single-panorama room.
3. **Judge each pair**: click a row in the list; the two panoramas appear on the right; decide with the table below. The next pair is selected automatically.

| Key | Button | Meaning |
|---|---|---|
| S | Same room | the two panoramas are the same room |
| D | Different room | not the same room (e.g. the next room seen through a door) |
| U | Uncertain | cannot tell |
| C | Clear | remove the decision for this pair |
| Down or N | Next | move to the next pair |
| Up or P | Previous | move to the previous pair |

   Click a panorama (or press A / B) to enlarge it; I toggles the inlier marks; drag the divider between the two panoramas to resize them.

4. **Room groups**: same-room pairs are chained automatically. If A–B and B–C are the same room, A, B and C become one room; not every pair needs a decision. If two panoramas judged "different room" end up in one group through other decisions, the bottom left shows the conflict in red; check the decisions involved.
5. **Only undecided pairs**: tick "Hide judged".
6. **Saving**: every decision is written to `scans\<dataset>\decisions.jsonl` immediately; there is nothing to save by hand, and reopening the app continues where you left off.
7. **Export**: press "Export room groups JSON" at the top right and choose where to save (default `scans\<dataset>\room_groups.json`). The export covers every house.

### 4.4 Judging tips

- Same room: the same walls, windows, ceiling or floor pattern are visible, and both cameras are inside that space.
- Different room: each camera sees the other only through a door or opening, or the two cameras are clearly in different spaces.
- Open spaces (e.g. no wall between living and dining room), hallways and entrances invite disagreement. Agree on a rule within the team first, and use "Uncertain" when you cannot tell.

---

## 5. Choosing the threshold

Measured on the 60 Dollhouse houses (inliers ≥ 30). P = share of listed candidates that really are the same room; R = share of all same-room pairs that are listed.

**Door-aware wedge** (default, needs door data)

| Both wedges ≥ | Precision | Recall | Candidate pairs | Use when |
|---:|---:|---:|---:|---|
| 20° | 0.41 | 0.75 | 773 | you want to miss as little as possible and accept many more candidates |
| 30° | 0.47 | 0.65 | 576 | leaning towards completeness |
| **40° (default)** | **0.56** | **0.58** | **439** | balanced |
| 50° | 0.58 | 0.47 | 341 | only the surest pairs |

**Plain wedge** (no door data, or "Door-aware wedge" unticked)

| Both wedges ≥ | Precision | Recall | Candidate pairs | Use when |
|---:|---:|---:|---:|---|
| 15° | 0.31 | 0.84 | 1147 | miss as little as possible |
| 30° | 0.41 | 0.67 | 677 | leaning towards completeness |
| **45° (default)** | **0.51** | **0.54** | **445** | balanced |
| 60° | 0.53 | 0.40 | 317 | only the surest pairs |

- The door-aware wedge at 40° lists about as many candidates as the plain wedge at 45°, with both higher precision and higher recall; the share of door-adjacent rooms wrongly listed as the same room drops from 0.13 to 0.12 (from 0.16 to 0.12 compared with the plain wedge at 40°). Leave-one-house-out cross-validation: F1 0.535 → 0.569. Details: `docs\benchmark\RESULTS_open_wedge.md`.
- Most remaining errors come from open spaces (living/dining/kitchen connected); there is no doorway there, so the door-aware wedge does not help.
- In practice: go through everything once at the default threshold, then lower the threshold, tick "Hide judged" and look at the newly listed candidates.
- The two tables come from Dollhouse; on very different data treat them as a rough guide.

## 6. Importing new panoramas

### 6.1 Input format

One folder with one sub-folder per house, holding that house's panoramas:

```
D:\new_panos\
    HouseA\  001.jpg  002.jpg  ...
    HouseB\  a.jpg    b.jpg    ...
```

- jpg, jpeg and png are supported; the aspect ratio must be 2:1 (equirectangular panoramas).
- Images in deeper sub-folders are read too; folders named `bi_layout_predictions`, `horizonnet_predictions`, `semantic`, `preview` or `thumbs` are skipped.
- Exact duplicate images are kept once; files that are not 2:1 or cannot be read are skipped and listed under `skipped` in `manifest.json`.
- All panoramas of a house are paired with each other (floors are not distinguished).

### 6.2 Import and scan

```powershell
# try 5 pairs first to check the environment
powershell -ExecutionPolicy Bypass -File Import_Dataset.ps1 -Source D:\new_panos -Name new_panos -Limit 5
# full scan
powershell -ExecutionPolicy Bypass -File Import_Dataset.ps1 -Source D:\new_panos -Name new_panos
```

It copies the data into `data\new_panos\`, scans into `scans\new_panos\`, then runs `Start_Openings.ps1` automatically: the bundled SAM3 (sam3-pano-seg, README.md §3.3) detects "open doorway" openings in each panorama and the door-aware wedges are computed. Then open `scans\new_panos` with `Start_Review.bat`.

- Without sam3-pano-seg, door-opening detection is skipped and the review app uses the plain wedge. SAM3 is bundled in `third_party\sam3-pano-seg`; another copy can be given with `-Sam3Root` or the `RCF_SAM3_ROOT` environment variable.
- To skip door-opening detection add `-NoOpenings`. To add it later:

```powershell
powershell -ExecutionPolicy Bypass -File Start_Openings.ps1 -Scan scans\new_panos
```

When the panoramas are already in `data\new_panos\`, scan without copying:

```powershell
powershell -ExecutionPolicy Bypass -File Import_Dataset.ps1 -Name new_panos -NoCopy
```

### 6.3 How long it takes

- About 3 s per pair. A house with n panoramas has n×(n−1)/2 pairs, e.g. 30 panoramas = 435 pairs, about 22 minutes.
- Door-opening detection takes about 3–4 s per panorama; loading SAM3 the first time takes about 1–2 minutes.
- After an interruption (shutdown, window closed) run the same command again; it resumes and finished pairs are not recomputed.
- Progress is written to `scans\<dataset>\status.json` and `scan.log`.

---

## 7. Output files

| File (in `scans\<dataset>\`) | Contents |
|---|---|
| `manifest.json` | houses and panoramas (paths relative to the dataset), skipped files, GeoVerify version and parameters |
| `scores.jsonl` | one line per pair: house, both panoramas, pose found or not, inliers, both wedges |
| `pairs\<house>\<A>__<B>.json` | full geometric-verification result per pair (pose, co-visible regions, inliers, etc.) |
| `openings\<house>\<panorama>_mask.png` | SAM3 door mask (2048×1024 black and white) |
| `open_scores.jsonl` | door-aware wedge per pair: remaining inliers, door-inlier share, both recomputed wedges |
| `decisions.jsonl` | review decisions (one line appended per decision; the latest one counts) |
| `room_groups.json` | exported room groups |

Contents of `room_groups.json` (also the input of `2_pose_floorplan\Run_Pipeline.ps1 -Groups`, see the root README):

- `groups`: each room group; `panorama_uuids` is the list of panorama IDs, `task` the house name, `source_pairs` the same-room pairs that form the group.
- `pair_reviews`: the decision for each pair (`same_room`, `different_room`, `uncertain`), reviewer and time.
- `conflicts`: groups that contain a pair judged different rooms.
- `classifier`: the rule in use at export (`mode`: `open` door-aware wedge / `plain` plain wedge), the thresholds and the GeoVerify version.

---

## 8. Troubleshooting

| Problem | What to do |
|---|---|
| `Start_Review.bat` says "Python environment not found" | Run `Setup.ps1` in the package root |
| PowerShell refuses to run scripts | Run them as `powershell -ExecutionPolicy Bypass -File <script>` |
| The review screen says "Cannot read the image" | The data under `data\` was moved or deleted. Move `scans\` and `data\` together, keeping their relative position, or put the dataset's absolute path in `root_abs` of `scans\<dataset>\manifest.json`. `scans\dollhouse` needs the Dollhouse dataset in `data\dollhouse`, or `root_abs` pointing to it (currently `C:/datasets`) |
| The scan says "Another scan may be running" | Make sure no other scan is running, then delete `scans\<dataset>\RUNNING.lock` |
| The scan says the main environment was not found | Run `Setup.ps1` in the package root |
| "Door-aware wedge" is greyed out | This scan has no door data; run `Start_Openings.ps1` (section 6.2) |
| A house has no candidates | Move the wedge slider to the left; if there are still none, the panoramas of that house hardly see each other |
| `scans\dollhouse` shows fewer scanned pairs than all pairs | Normal: that result only covers same-floor pairs |
| No inlier marks on the panoramas, only regions | The scan predates per-point inliers; recompute them (section 9) |

---

## 9. Advanced

- Automated tests: `third_party\geoverify_main\venv\Scripts\python.exe -m pytest tests\test_core.py` (no GPU or window needed).
- Default thresholds and paths: `finder\config.py`. The GeoVerify location can be set with `RCF_GEOVERIFY_MAIN`, the SAM3 location with `RCF_SAM3_ROOT`.
- The door-aware wedge is computed in `finder\openings.py` (needs numpy; run it with the main environment). `python -m finder.openings --scan <scan folder> [--npz-dir <per-point inlier folder>]` recomputes only `open_scores.jsonl`.
- Per-point inliers for older scans (GPU, about 3 s per pair, resumable; nothing else may use the GPU meanwhile):
  `third_party\geoverify_main\venv\Scripts\python.exe tools\recompute_inliers.py --scan scans\dollhouse` (candidate pairs at `--min-wedge 15 --min-inliers 30`), add `--reviewed` for every judged pair, `--groups <room_groups.json>` for every pair inside those groups, `--house NAME` to limit to one house. The results go to `scans\<dataset>\pairs_inliers\`; `scores.jsonl`, the decisions and `pairs\` are not changed.
- `scans\dollhouse` was converted from the results of the same-room classifier comparison (door masks imported by `tools\import_openings.py`); the choice of classifier and the full comparison are in `docs\benchmark\RESULTS.md`.
