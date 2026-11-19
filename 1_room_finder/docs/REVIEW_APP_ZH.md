# 審查程式與找同房掃描：操作說明

這個工具從一批全景圖中，自動找出「可能是同一間房」的全景配對，讓人依信心度調整篩選門檻、逐對確認，最後把確認結果整理成房間群組並匯出。

本文件說明 `1_room_finder\`（篩選相同房間）的掃描與審查程式。環境安裝、完整流程與規格見工具包根目錄的 `README_ZH.md`（輸入與輸出格式見 §6–§9）。英文版：[REVIEW_APP.md](REVIEW_APP.md)。

**以下指令與路徑都以 `1_room_finder\` 為準**（先 `cd 1_room_finder`）；`Setup.ps1` 在工具包根目錄。

---

## 1. 快速開始

1. 先在工具包根目錄執行一次 `Setup.ps1`（建立掃描、審查等環境，見 README_ZH.md §4）。
2. 雙擊 `Start_Review.bat`。
3. 在跳出的視窗中選擇 `scans\` 底下的資料集資料夾（例如附帶的 `scans\dollhouse`；它需要 Dollhouse 全景圖，見第 8 節）。
4. 依視窗最上方的操作說明開始審查（詳細說明見第 4 節）。

程式預設為英文介面，按右上角的「中文」按鈕切換為中文（再按「English」切回）；要一開始就是中文，在執行 `Start_Review.bat` 前設定 `RCF_LANG=zh`，或用 `python -m finder.app --lang zh`。

---

## 2. 相關檔案

| 位置 | 內容 |
|---|---|
| `Start_Review.bat` | 開啟審查程式 |
| `Import_Dataset.ps1` | 匯入新的全景資料並掃描（`-NoCopy`：資料已在 `data\` 底下時只掃描） |
| `Start_Openings.ps1` | SAM3 門洞偵測與去門楔形 |
| `tools\recompute_inliers.py` | 重算候選對的逐點內點，讓審查畫面標示內點（手動執行，見第 9 節） |
| `finder\` | 程式本體；`finder\config.py` 是預設門檻；`finder\i18n.py` 是英文／中文介面文字 |
| `third_party\geoverify_main\` | GeoVerify 幾何驗證程式（GitHub main，版本 b589709） |
| `scans\dollhouse\` | Dollhouse 已完成的掃描結果與審查紀錄 |

---

## 3. 環境

由工具包根目錄的 `Setup.ps1` 一次建立：審查程式用 `1_room_finder\.venv\`（Python＋Tkinter＋Pillow，不需要 GPU），掃描用 `1_room_finder\third_party\geoverify_main\venv\`（Python 3.11＋PyTorch CUDA 13.0＋RoMa v2，需要 NVIDIA GPU）。要沿用其他位置的環境，用環境變數 `RCF_REVIEW_PYTHON`、`RCF_MAIN_PYTHON` 指定 python.exe（README_ZH.md §3.2）。

---

## 4. 審查程式操作

### 4.1 開啟

- 雙擊 `Start_Review.bat`，選擇 `scans\` 底下的資料集資料夾（裡面要有 `manifest.json`）。
- 也可以直接指定：`Start_Review.bat scans\dollhouse`

### 4.2 畫面配置

| 區域 | 內容 |
|---|---|
| 最上方淺藍色列 | 操作說明 |
| 上方工具列（兩列） | 房子選單、「去門楔形」切換、楔形門檻滑桿、內點門檻、匯出按鈕、語言按鈕；第二列：「隱藏已判定」、「標示內點」、預估 precision／recall |
| 左側清單 | 目前門檻下的候選配對（全景 A、全景 B、較窄楔形角度、內點數、門洞內點比例、判定） |
| 左下方 | 這棟房子目前的房間群組與矛盾提示 |
| 右側 | 選中配對的兩張全景（上 A、下 B）與判定按鈕 |
| 最下方狀態列 | 候選數、已判定數、沒有候選的全景數、已掃描配對數 |

### 4.3 操作步驟

1. **選房子**：左上角「房子」下拉選單。
2. **設定門檻**（可先用預設值，見第 5 節）：
   - **去門楔形（SAM3 門洞）**（有門洞資料時預設勾選）：隔著門洞看到的內點不算進楔形，可減少「隔著門看到隔壁房間」的誤判。切換時滑桿會跳到該模式的預設值（去門楔形 40°、一般楔形 45°）。右側全景上，青色是 SAM3 偵測到的門洞，灰色空心點是被排除的內點。
   - **兩側楔形 ≥**（預設 40°；沒有門洞資料時 45°）：主要的信心門檻。往左拉，候選變多，較容易找到同房配對，但誤判也多；往右拉，候選變少但較可靠。
   - **內點 ≥**（預設 30）：次要門檻，調高可再減少一些誤判，通常不用動。
   - 低於門檻的配對不會出現在清單中。完全沒有候選的全景，視為單張房間。
3. **逐對判斷**：點左邊清單的一列，右邊顯示兩張全景，依下表判定。按完會自動跳到下一對。

| 按鍵 | 按鈕 | 意思 |
|---|---|---|
| S | 同房 | 兩張是同一間房 |
| D | 不同房 | 不是同一間房（例如隔著門看到隔壁） |
| U | 不確定 | 看不出來 |
| C | 清除 | 取消這一對的判定 |
| ↓ 或 N | 下一組 | 移到下一對 |
| ↑ 或 P | 上一組 | 移到上一對 |

   點全景（或按 A、B）可放大；I 開關內點標示；拖曳兩張全景之間的分隔線可調整上下大小。

4. **看房間群組**：判定為「同房」的配對會自動串起來。例如 A–B 同房、B–C 同房，A、B、C 就會歸成同一間，不需要每一對都判。如果某兩張被判「不同房」，卻因為其他判定被串在同一群組，左下方會以紅字標示矛盾，請回頭檢查相關判定。
5. **只看沒判過的**：勾選「隱藏已判定」。
6. **存檔**：每次判定都會立即存到 `scans\<資料集>\decisions.jsonl`，不用手動存檔；關掉程式再打開會接著做。
7. **匯出**：按右上角「匯出房間群組 JSON」，選擇存檔位置（預設 `scans\<資料集>\room_groups.json`）。匯出內容包含所有房子。

### 4.4 判斷時的建議

- 同一間房：看得到相同的牆面、窗戶、天花板或地板樣式，且兩台相機都在這個空間裡。
- 不同房：只有透過門口、開口才看得到對方，或兩台相機明顯在不同空間。
- 開放式空間（例如客廳和餐廳之間沒有牆）、走廊、玄關容易有不同看法。團隊內請先約定好標準，判不出來就用「不確定」。

---

## 5. 門檻怎麼選

下表是在 Dollhouse 60 棟上量得的結果（內點 ≥ 30）。P 是「列出的候選中真的同房的比例」，R 是「所有同房配對中被列出的比例」。

**去門楔形**（預設，需要門洞資料）

| 兩側楔形 ≥ | Precision | Recall | 候選對數 | 適合情況 |
|---:|---:|---:|---:|---|
| 20° | 0.41 | 0.75 | 773 | 想盡量不漏，願意多看很多候選 |
| 30° | 0.47 | 0.65 | 576 | 偏向找全 |
| **40°（預設）** | **0.56** | **0.58** | **439** | 平衡 |
| 50° | 0.58 | 0.47 | 341 | 只看最有把握的 |

**一般楔形**（沒有門洞資料，或取消勾選「去門楔形」時）

| 兩側楔形 ≥ | Precision | Recall | 候選對數 | 適合情況 |
|---:|---:|---:|---:|---|
| 15° | 0.31 | 0.84 | 1147 | 想盡量不漏 |
| 30° | 0.41 | 0.67 | 677 | 偏向找全 |
| **45°（預設）** | **0.51** | **0.54** | **445** | 平衡 |
| 60° | 0.53 | 0.40 | 317 | 只看最有把握的 |

- 去門楔形 40° 和一般楔形 45° 的候選數差不多，但 precision 和 recall 都比較高；隔門相鄰房間被誤判同房的比例由 0.13 降到 0.12，和一般楔形 40° 相比由 0.16 降到 0.12。留一棟房子交叉驗證：F1 0.535 → 0.569。細節見 `docs\benchmark\RESULTS_open_wedge_ZH.md`。
- 主要剩下的錯誤來自開放式空間（客廳／餐廳／廚房相連），這種情況沒有門洞，去門楔形也幫不上。
- 實際做法：先用預設門檻看完一輪，再把門檻往下調，勾選「隱藏已判定」，補看新出現的候選。
- 這兩張表來自 Dollhouse，換到風格差很多的資料時只能當參考。

## 6. 匯入新的全景資料

### 6.1 資料格式

一個資料夾，每個子資料夾是一棟房子，裡面放這棟的全景圖：

```
D:\new_panos\
    HouseA\  001.jpg  002.jpg  ...
    HouseB\  a.jpg    b.jpg    ...
```

- 支援 jpg、jpeg、png，長寬必須是 2:1（等距長方投影全景）。
- 子資料夾裡的圖也會讀到；名為 `bi_layout_predictions`、`horizonnet_predictions`、`semantic`、`preview`、`thumbs` 的資料夾會略過。
- 內容完全相同的重複圖片只保留一張；不是 2:1 或讀不到的檔案會跳過，並記錄在 `manifest.json` 的 `skipped`。
- 同一棟房子的所有全景會兩兩配對（不分樓層）。

### 6.2 匯入並掃描

```powershell
# 先試 5 對，確認環境正常
powershell -ExecutionPolicy Bypass -File Import_Dataset.ps1 -Source D:\new_panos -Name new_panos -Limit 5
# 正式掃描
powershell -ExecutionPolicy Bypass -File Import_Dataset.ps1 -Source D:\new_panos -Name new_panos
```

它會把資料複製到 `data\new_panos\`，再掃描到 `scans\new_panos\`，接著自動執行 `Start_Openings.ps1`：用內含的 SAM3（sam3-pano-seg，見 README_ZH.md §3.3）對每張全景偵測「open doorway」（門洞），再算出去門楔形。完成後，用 `Start_Review.bat` 開啟 `scans\new_panos` 審查。

- 找不到 sam3-pano-seg 時會略過門洞偵測，審查程式改用一般楔形。SAM3 已內含在 `third_party\sam3-pano-seg`，或用 `-Sam3Root`、環境變數 `RCF_SAM3_ROOT` 指定。
- 不想跑門洞偵測：加 `-NoOpenings`。之後要補時再執行：

```powershell
powershell -ExecutionPolicy Bypass -File Start_Openings.ps1 -Scan scans\new_panos
```

全景已經放在 `data\new_panos\` 時，可以不複製、只掃描：

```powershell
powershell -ExecutionPolicy Bypass -File Import_Dataset.ps1 -Name new_panos -NoCopy
```

### 6.3 需要多久

- 每對約 3 秒。一棟 n 張全景有 n×(n−1)/2 對，例如 30 張是 435 對，約 22 分鐘。
- 門洞偵測每張全景約 3–4 秒，第一次載入 SAM3 模型約需 1–2 分鐘。
- 掃描中斷（關機、關視窗）後，重跑同一個指令會從斷點接續，已完成的配對不會重算。
- 進度寫在 `scans\<資料集>\status.json` 與 `scan.log`。

---

## 7. 輸出檔案

| 檔案（在 `scans\<資料集>\`） | 內容 |
|---|---|
| `manifest.json` | 房子、全景清單（路徑相對於資料集）、跳過的檔案、GeoVerify 版本與參數 |
| `scores.jsonl` | 每對一行：房子、兩張全景、是否有位姿、內點數、兩側楔形角度 |
| `pairs\<房子>\<A>__<B>.json` | 每對完整的幾何驗證結果（位姿、共視區域、內點等） |
| `openings\<房子>\<全景>_mask.png` | SAM3 門洞 mask（2048×1024 黑白圖） |
| `open_scores.jsonl` | 每對的去門楔形：剩餘內點數、門洞內點比例、兩側重算後的楔形 |
| `decisions.jsonl` | 審查判定紀錄（每次判定追加一行，以最新一筆為準） |
| `room_groups.json` | 匯出的房間群組 |

`room_groups.json` 的內容（也是 `2_pose_floorplan\Run_Pipeline.ps1 -Groups` 的輸入，見工具包根目錄的 README）：

- `groups`：每個房間群組，`panorama_uuids` 為全景 ID 清單，`task` 為房子名稱，`source_pairs` 為構成群組的同房配對。
- `pair_reviews`：每一對的判定（`same_room`、`different_room`、`uncertain`）、審查者與時間。
- `conflicts`：群組內含有被判不同房的配對。
- `classifier`：匯出當下使用的規則（`mode`：`open` 去門楔形／`plain` 一般楔形）、門檻與 GeoVerify 版本。

---

## 8. 常見問題

| 狀況 | 處理方式 |
|---|---|
| `Start_Review.bat` 顯示 "Python environment not found" | 執行工具包根目錄的 `Setup.ps1` |
| PowerShell 不允許執行腳本 | 用 `powershell -ExecutionPolicy Bypass -File <腳本>` 的方式執行 |
| 審查畫面顯示「無法讀取影像」 | `data\` 底下的資料被移動或刪除。`scans\` 與 `data\` 必須一起搬，保持相對位置；或在 `scans\<資料集>\manifest.json` 的 `root_abs` 填入資料集的絕對路徑。`scans\dollhouse` 需要把 Dollhouse 資料集放到 `data\dollhouse`，或保留 `root_abs` 指向它（目前是 `C:/datasets`） |
| 掃描顯示 "Another scan may be running" | 確認沒有其他掃描在跑，再刪除 `scans\<資料集>\RUNNING.lock` |
| 掃描顯示「找不到主環境」 | 執行工具包根目錄的 `Setup.ps1` |
| 「去門楔形」是灰色、不能勾選 | 這份掃描沒有門洞資料，執行 `Start_Openings.ps1`（第 6.2 節） |
| 某棟房子沒有任何候選 | 把楔形門檻往左調；仍然沒有，代表這棟的全景彼此幾乎沒有共視 |
| `scans\dollhouse` 的「已掃描配對」比全部配對少 | 正常。這份結果只計算了同樓層的配對 |
| 全景上沒有內點標示，只有共視區域 | 這份掃描比逐點內點功能早；依第 9 節重算 |

---

## 9. 進階

- 自動測試：`third_party\geoverify_main\venv\Scripts\python.exe -m pytest tests\test_core.py`（不需 GPU 與視窗）。
- 預設門檻與路徑：`finder\config.py`。GeoVerify 程式位置可用環境變數 `RCF_GEOVERIFY_MAIN` 指定，SAM3 位置用 `RCF_SAM3_ROOT`。
- 去門楔形的計算在 `finder\openings.py`（需要 numpy，用主環境執行）。`python -m finder.openings --scan <掃描資料夾> [--npz-dir <逐點內點資料夾>]` 可以只重算 `open_scores.jsonl`。
- 較舊掃描的逐點內點（GPU，每對約 3 秒，可中斷續跑；執行時 GPU 不能有其他程式）：
  `third_party\geoverify_main\venv\Scripts\python.exe tools\recompute_inliers.py --scan scans\dollhouse`（`--min-wedge 15 --min-inliers 30` 下的候選對），加 `--reviewed` 重算所有判定過的配對、`--groups <room_groups.json>` 重算這些群組內的所有配對、`--house 名稱` 只算一棟。結果寫到 `scans\<資料集>\pairs_inliers\`，不會改動 `scores.jsonl`、判定紀錄與 `pairs\`。
- `scans\dollhouse` 是從同房分類器比較實驗的結果轉出的（門洞 mask 由 `tools\import_openings.py` 匯入）；分類器的選用依據與完整比較數字見 `docs\benchmark\RESULTS_ZH.md`。
