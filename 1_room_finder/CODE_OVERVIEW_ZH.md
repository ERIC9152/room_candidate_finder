# 1_room_finder 程式說明

本文件概述 `1_room_finder\` 的每支程式：它在流程中的位置、主要函式與功能。使用方式、輸入與輸出格式見根目錄的 [README_ZH.md](../README_ZH.md)（§5–§7）。英文版：[CODE_OVERVIEW.md](CODE_OVERVIEW.md)。

這個資料夾負責「篩選相同房間」：全景兩兩做幾何驗證（步驟 A），用 SAM3 門洞遮罩修正共視楔形（步驟 A'），再由人工審查並匯出房間群組（步驟 B）。

---

## 1. 呼叫關係

```
Import_Dataset.ps1 ─┬─ robocopy                     複製全景到 data\<名稱>
  （步驟 A＋A'）      ├─ python -m finder.scan       每一對做 GeoVerify → scans\<名稱>\
                    └─ Start_Openings.ps1          （有 SAM3 時自動接著做）

Start_Openings.ps1 ─┬─ python -m finder.openings --list-todo   列出還沒有遮罩的全景
  （步驟 A'）         ├─ tools\fix_sam3_env.ps1                  SAM3 環境改指到目前位置
                    ├─ sam3_pano.seg.unified（SAM3 環境）       「open doorway」門洞分割
                    ├─ tools\import_openings.py                 遮罩縮成 2048×1024 存進 openings\
                    └─ python -m finder.openings               計算去門楔形 → open_scores.jsonl

Start_Review.bat ──── python -m finder.app         審查程式（Tkinter）
  （步驟 B）             └─ finder.model → finder.scores / finder.grouping / finder.i18n

手動：tools\recompute_inliers.py                  為舊掃描補存逐點內點
```

所有 PowerShell 腳本都先載入根目錄的 `tools\_env.ps1`，用它尋找並試跑環境（`Get-MainPy`、`Get-ReviewPy`）、尋找 SAM3（`Get-Sam3Root`），以及檢查 GPU 是否被占用（`Assert-GpuFree`）。

---

## 2. 啟動腳本

| 檔案 | 功能 |
|---|---|
| `Import_Dataset.ps1` | 參數 `-Source`、`-Name`、`-NoCopy`、`-Limit`、`-NoOpenings`。以 robocopy 把來源資料夾複製到 `data\<Name>`，再用主環境執行 `finder.scan --root data\<Name> --out scans\<Name>`；找得到 SAM3 時接著呼叫 `Start_Openings.ps1` |
| `Start_Openings.ps1` | 參數 `-Scan`、`-Sam3Root`。分 5 步：① `finder.openings --list-todo` 列出缺遮罩的全景；② 以 hard link（不行時複製）把它們放進 `openings_stage\`，檔名用 `<房屋>__<id>`；③ 用 SAM3 環境執行 `sam3_pano.seg.unified --categories door --door-prompt-override "open doorway" --pano-vote --mask-only --skip-existing`；④ `tools\import_openings.py` 匯入遮罩，再刪掉暫存資料夾；⑤ `finder.openings` 計算去門楔形 |
| `Start_Review.bat` | 依序找 `RCF_REVIEW_PYTHON` 或 `.venv`，先確認能匯入 `tkinter`、`PIL`，再執行 `python -m finder.app [--scan <資料夾>]`。找不到環境或環境無法啟動時，提示先執行 `Setup.ps1` |

---

## 3. `finder\` 套件

### 3.1 `config.py`：路徑與預設值

沒有函式，只放常數：

- 路徑：`ROOT`、`GEOVERIFY_MAIN`（可用 `RCF_GEOVERIFY_MAIN` 覆寫）、`DATA_DIR`、`SCANS_DIR`、`SAM3_ROOT`（可用 `RCF_SAM3_ROOT` 覆寫）。
- `TORCH_HOME`：沒有另外設定時，指向 `models\torch_home`，讓 RoMa v2 權重跟著資料夾走。資料夾內還沒有、但本機快取已有時，沿用本機快取。
- 掃描：`IMAGE_EXTS`、`EXCLUDE_DIRS`。
- 分類器：`DEFAULT_WEDGE_DEG = 45`、`DEFAULT_MIN_INLIERS = 30`、`DEFAULT_OPEN_WEDGE_DEG = 40`、`CALIBRATION_FILE`。
- 門洞遮罩：`OPEN_MASK_W/H = 2048×1024`、`OPEN_DILATE_PX = 4`、`OPEN_PROMPT = "open doorway"`。

### 3.2 `dataset.py`：找房屋與全景

| 函式 | 功能 |
|---|---|
| `scan_houses(root)` | 每個子資料夾是一棟房子；沒有子資料夾時整個資料夾是一棟。只留寬高比 2 ± 0.02 的影像，以內容雜湊去重，同名不同內容的加 `_2`、`_3`。回傳 `{房屋: [{id, path, w, h}]}` 與 `skipped`（含原因）；少於 2 張的房屋略過 |
| `_images_under(folder)` | 遞迴列出影像檔，跳過 `EXCLUDE_DIRS` |
| `_sha(path)` | 前 1 MiB＋檔案大小的雜湊，用來判斷重複 |
| `all_pairs(items)` | 依固定順序產生所有無序配對 |

### 3.3 `scan.py`：GeoVerify 掃描（步驟 A，GPU）

| 函式 | 功能 |
|---|---|
| `_import_geoverify()` | 從 `config.GEOVERIFY_MAIN` 匯入 `roma_geo`，確認匯入的是包內那一份，並讀取 `SOURCE_COMMIT.txt` |
| `main()` | 參數 `--root`、`--out`、`--house`、`--limit`。① 建立 `RUNNING.lock`，防止同時跑兩個掃描；② `dataset.scan_houses` → 寫 `manifest.json`（`root` 存相對路徑，另有 `root_abs`）；③ `roma_geo.get_matcher("roma_v2", …, cache)` 載入 RoMa v2；④ 每一對呼叫 `roma_geo.verify_pair(a, b, mode="direct", keep_inliers=True)`，結果寫入 `pairs\<房屋>\<a>__<b>.json`（先寫暫存檔再取代），摘要經 `scores.summarize` 追加到 `scores.jsonl`；⑤ 錯誤寫 `errors.jsonl`，每 25 對更新一次 `status.json`。已在 `scores.jsonl` 的配對會略過，所以可以中斷續跑 |

### 3.4 `scores.py`：分數摘要與分類規則

| 函式／類別 | 功能 |
|---|---|
| `summarize(g)` | `verify_pair` 結果 → `has_pose`、`n_inlier`、`inlier_rate`、`wedge_a/b`、`wedge_min`、`n_regions` |
| `load_scores(scan)` | 讀 `scores.jsonl`，同一對以最後一行為準 → `{房屋: {(a, b): 摘要}}` |
| `uses_open(s, mode)` | 這一對是否用去門楔形判斷（open 模式且有 `open` 分數） |
| `active_wedge(s, mode)` | 依模式取用來比較的楔形（一般或去門） |
| `passes(s, wedge_deg, min_inliers, mode)` | 分類規則：有位姿、內點 ≥ 門檻、兩側楔形都 ≥ 門檻 |
| `Calibration` | 讀 `calibration_dollhouse.json`；`expected()` 依滑桿位置查最近的一列，給出預估的 Precision／Recall |

### 3.5 `openings.py`：去門楔形（步驟 A'）

| 函式 | 功能 |
|---|---|
| `wedge_width(az, frac=0.8)` | 包住 80% 方位角的最窄圓弧（度），與 GeoVerify 的 `az_wedge` 相同 |
| `load_mask(path)` | 讀門洞遮罩 → bool 陣列；檔案不存在時回傳 None |
| `dilate(mask, r=4)` | 外擴遮罩：左右 r px（跨接縫），上下累積位移 1…r，合計 r(r+1)/2 = 10 px。必須與當初校準的做法相同，否則校準表不再適用 |
| `open_wedge(inliers, size, mask_a, mask_b, n_inlier)` | 把內點座標換算到 2048×1024，剔除落在 A 或 B 遮罩內的點，再用剩下的點重算 `wedge_a/b/min`、`n_keep`、`door_frac`（GeoVerify 最多保留 2000 點，`n_keep` 依 `n_inlier` 換算回完整數量） |
| `_pair_inliers(scan, house, a, b)` | 取一對的內點：優先 `pairs_inliers\`，其次 `pairs\` 或 `--npz-dir`；能處理以 `b__a` 順序存放的檔案 |
| `compute(scan)` | 對所有可能通過規則的配對（有位姿、內點 ≥ 30）計算去門楔形，寫 `open_scores.jsonl` 與 `open_scores_meta.json` |
| `list_todo(scan, out)` | 列出還沒有遮罩的全景（`房屋 <TAB> id <TAB> 絕對路徑`），給 `Start_Openings.ps1` 使用 |
| `mask_path(scan, house, id)` | 遮罩檔位置 `openings\<房屋>\<id>_mask.png` |

### 3.6 `grouping.py`：判定 → 房間群組

| 函式／類別 | 功能 |
|---|---|
| `pair_key(a, b)` | 不分順序的配對鍵 |
| `Decisions` | 管理 `decisions.jsonl`（只增不改，最後一筆為準）；`set()` 寫入一筆判定，`get()` 讀取 |
| `build_groups(house, panos, decisions)` | 以 union-find 把 `same_room` 判定串成群組（至少 2 張）；同一群組內有 `different_room` 的配對列為 `conflicts` |
| `export_groups(path, houses, decisions, thresholds, source_commit)` | 寫出 `room_groups.json`（`groups`、`conflicts`、`pair_reviews`、`classifier` 等） |

### 3.7 `model.py`：審查狀態（不含介面）

`ReviewModel` 把掃描結果、門檻與判定整合在一起，`app.py` 只負責顯示。

| 方法 | 功能 |
|---|---|
| `set_mode(mode)`、`thresholds()` | 切換一般楔形（`plain`）或去門楔形（`open`）；切換時門檻改為該模式的預設值 |
| `candidates()` | 通過目前門檻的配對，窄側楔形最寬的排在前面 |
| `scanned_pairs()`、`expected_pairs()`、`progress()` | 掃描與審查進度 |
| `panos_without_candidates()` | 沒有任何候選配對的全景 |
| `expected_quality()` | 目前門檻的預估 P/R（`Calibration`） |
| `decide(a, b, relation, note)` | 記錄一筆判定 |
| `groups()`、`export(path)` | 目前的房間群組；匯出 JSON |
| `pair_detail(a, b)` | 讀取一對的 GeoVerify 結果，給內點疊圖使用（必要時用 `_swap` 交換 A、B） |
| `opening_mask_path(house, id)`、`image_path(house, id)` | 遮罩位置；全景位置（先用相對的 `root`，找不到時用 `root_abs`） |

### 3.8 `app.py`：審查程式（Tkinter）

`App` 類別建立視窗，`main()` 解析 `--scan`、`--reviewer`、`--lang`。主要方法：

| 方法 | 功能 |
|---|---|
| `_build()` | 建立版面：上方是房屋選單、去門模式勾選、楔形與內點門檻、匯出與語言按鈕；左側是候選清單（Treeview）與群組清單；右側是 A、B 兩張全景畫布與判定按鈕。快捷鍵：S 同房、D 不同房、U 不確定、C 清除、↓／N 下一對、↑／P 上一對 |
| `_mode_changed()`、`_thresholds_changed()`、`refresh()` | 模式或門檻改變時，重新篩選候選清單 |
| `_render()`、`_schedule_render()` | 以不裁切的 2:1 比例把全景縮放到畫布中 |
| `_overlay()`、`_through_opening()`、`_load_mask()`、`_overlay_text()` | 疊上 GeoVerify 內點（同色代表 A、B 中的對應點），落在門洞內的點另外標示，並顯示分數文字 |
| `zoom(k)` | 全螢幕放大 A 或 B（Esc 或點擊關閉） |
| `decide()`、`move()` | 寫入判定（同房／不同房／不確定／清除），移到上一對或下一對 |
| `render_groups()`、`export()` | 顯示目前的群組；匯出 `room_groups.json` |
| `toggle_lang()` | 切換英文／中文，重建視窗（判定已存在檔案裡，不會遺失） |

### 3.9 `i18n.py`：介面文字

`set_lang()`、`get_lang()` 設定與讀取語言（預設來自 `RCF_LANG`）；`T(key)` 取得目前語言的字串（所有文字都在 `STRINGS` 表中，每個鍵有英文、中文兩個版本）；`rel_text(relation)` 把判定轉成顯示文字。

---

## 4. `tools\`

| 檔案 | 功能 |
|---|---|
| `fix_sam3_env.ps1` | 把內含 SAM3 虛擬環境中的絕對路徑（`pyvenv.cfg` 的 `home`、兩個 `__editable___*_finder.py`）改成目前位置；路徑已正確時不做任何事。由 `Setup.ps1`、`Start_Openings.ps1` 呼叫 |
| `import_openings.py` | `main()`：把 SAM3 輸出的全尺寸遮罩縮成 2048×1024、1 位元 PNG，存到 `openings\<房屋>\<id>_mask.png`（`--src` 可給多個來源，`--budget-s` 可限制執行時間） |
| `recompute_inliers.py` | `main()`：對既有掃描的候選配對重跑 `roma_geo.verify_pair(..., keep_inliers=True)`，結果寫入 `pairs_inliers\`，並在 `consistency.jsonl` 記錄是否與原結果一致。選項 `--house`、`--min-wedge`、`--min-inliers`、`--limit`、`--groups`、`--reviewed` |

---

## 5. 使用的外部程式與函式庫

| 來源 | 使用的部分 | 用途 |
|---|---|---|
| GeoVerify（`third_party\geoverify_main`，`roma_geo`） | `get_matcher("roma_v2", …)`、`verify_pair(..., mode="direct")` | RoMa v2 稠密匹配、本質矩陣幾何驗證、共視楔形與共視區域 |
| RoMa v2（經由 GeoVerify） | 權重在 `models\torch_home` | 特徵匹配 |
| sam3-pano-seg（`third_party\sam3-pano-seg`） | `python -m sam3_pano.seg.unified`（在自己的 Python 3.12 環境中執行） | 全景「open doorway」分割 |
| numpy | — | 方位角、楔形、遮罩運算 |
| Pillow | `Image.open`、縮放、遮罩讀寫 | 讀影像大小、顯示全景、遮罩轉換 |
| tkinter | — | 審查程式介面 |

第三方程式（GeoVerify、sam3-pano-seg）沒有修改。

---

## 6. 測試

`tests\test_core.py`（`python -m pytest tests\test_core.py`，不需要 GPU 與視窗）以假資料測試核心功能：

| 測試 | 內容 |
|---|---|
| `test_scan_houses` | 房屋與全景的搜尋、非 2:1 略過、重複去除、同名編號 |
| `test_classifier`、`test_open_classifier` | 一般與去門楔形的分類規則 |
| `test_calibration_lookup` | P/R 校準表查詢 |
| `test_model_filter_groups_export_resume` | 候選篩選、判定、群組、匯出、重新開啟後續審 |
| `test_open_mode_in_model`、`test_open_wedge_and_compute` | 去門模式與 `open_scores.jsonl` 計算 |
| `test_moved_folder_still_works` | 整個資料夾搬動後，相對路徑仍然有效 |
