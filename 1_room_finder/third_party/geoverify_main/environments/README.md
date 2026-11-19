# environments/：環境建置與驗收

## 目標配置

| 項目 | 版本 |
|---|---|
| OS | Windows 11 64-bit / Linux |
| Python | 3.11 |
| torch | 2.10.0+cu130 |
| torchvision | 0.25.0+cu130 |
| numpy | 1.26.4 |
| opencv-python | 4.11.0.86 |
| scikit-learn | 1.8.0 |
| pillow | 12.0.0 |
| romav2 | 2.0.1（GitHub 源碼安裝） |

支援 Ada（sm_89）與 Blackwell（sm_120）世代的 GPU。不安裝 `fused-local-corr`，
所有效能數字皆為未加速狀態下量得。

## 建置

Windows：

```powershell
cd environments
.\setup_env.ps1
```

Linux / Kaggle：

```bash
bash environments/setup_env.sh
```

兩者是同一份配置的兩個平台版本，修改其一時另一個要同步。

依序：安裝 uv → 準備 Python 3.11 → 建立 `../venv`（可重複執行）→ 從 cu130 index
安裝 torch → 安裝釘版核心套件 → 安裝 romav2 → 重新釘版 → 執行驗收。

## 驗收

```powershell
..\venv\Scripts\python.exe environments\check_env.py
```

檢查版本比對（Python 只比 major.minor）、GPU arch 涵蓋、三項實跑驗證
（CUDA kernel、romav2 匯入、`pose_regions --selftest`）。

同一份腳本可在其他機器執行以確認環境一致：arch 一項顯示各自 GPU 的
compute capability，其餘欄位應完全相同。

端到端功能測試已移入測試模組：

```powershell
..\venv\Scripts\python.exe tests\smoke_test.py
```

經 `roma_geo.verify_pair()` 跑完整核心路徑，與入庫的參考 JSON 逐欄對照，並檢查演算法
參數一致性與對外契約未洩漏內部欄位。判準為兩邊都有位姿、共視區域數相同、方位角差
小於 5°。`tests/run_all.py` 會在偵測到 GPU 與資料集時一併執行，用法見
[`tests/README.md`](../tests/README.md)。

## 效能基準

```powershell
..\venv\Scripts\python.exe environments\local_timing_bench.py --n 120 --tag <標籤>
```

從 `_summary.csv` 母體中決定性等距抽樣，排除前 2 對暖機，關閉 RoMa 快取。
相同的 `--n` 會抽到相同的 pair，不同環境的結果可逐對相比。

量測時關閉其他 GPU／CPU 負載。**幾何驗證時間**是判斷結果可信與否的對照變因：
該階段為純 CPU numpy，其中位數明顯偏離既有基準時，匹配時間亦不可信。

## 跨環境一致性檢查

```bash
python environments/cross_env_check.py --n 30 --tag <環境標籤>
```

在目前環境重跑一批配對，與 `Output/roma_regions/_summary.csv` 的既有結果逐對比較。
判準為位姿一致率 ≥85% 且方位角差中位 ≤5°。決定性等距抽樣，相同 `--n` 在任何環境
都取到同一批配對。

## Kaggle

把 `kaggle_run.py` 的整份內容貼進一個 Notebook 儲存格執行，不需要先 clone。
貼上後改頂端的 CONFIG 區塊：

```python
REPO_URL    = "https://github.com/<owner>/<repo>.git"
REPO_BRANCH = "main"
SECRET_NAME = "GITHUB_TOKEN"   # 私有 repo 用；公開 repo 設為 ""
N_PAIRS     = 30
TAG         = "kaggle"
SKIP_SETUP  = False            # 環境已建好時設 True
FULL_CHAIN  = True             # False 則只跑前四步
WORKERS     = "auto"           # auto = GPU 張數
TIME_BUDGET_H = 8.0
```

repo 已在本機時也可直接當一般腳本執行，此時沿用既有目錄不重新 clone。

⚠ **GPU 選擇**：需選 T4（sm_75）或 L4（sm_89）；Tesla P100（sm_60）與 V100（sm_70）
不適用，腳本會在安裝前檢查並中止。

### 執行步驟

前四步為快速關卡（數分鐘）：

| 步驟 | 腳本 | 判準 |
|---|---|---|
| 1 路徑設定 | `config.py` | 各項路徑解析得到、且存在 |
| 2 環境驗收 | `environments/check_env.py` | 三項實跑（CUDA kernel／romav2／幾何 selftest）皆過。版本對照僅供參考 |
| 3 端到端煙霧測試 | `tests/smoke_test.py` | 固定一對真實影像跑完整流程，與入庫的參考結果一致 |
| 4 跨環境一致性檢查 | `environments/cross_env_check.py` | 位姿有無一致率 ≥85%，且強證據配對的方位角差中位 ≤5° |

2 驗的是「跑不跑得動」，3 驗的是「跑出來對不對」，4 驗的是「整批的結論穩不穩」。

後六步為完整鏈條（`FULL_CHAIN = True`）：

| 步驟 | 腳本 | 產出 |
|---|---|---|
| 5 從連通矩陣挑配對 | `environments/select_pairs.py --force` | `Output/roma_regions/_pairs.csv` |
| 6 批次匹配 | `environments/batch_match.py` | `roma_match/cache/*.npz` + `roma_scores.npz` |
| 7 幾何驗證批次 | `pose_regions/batch_pose_from_cache.py` | `Output/roma_regions/*.json` + `_summary.csv` |
| 8 排序評估 | `pose_regions/covis_ranking_eval.py` | `Output/roma_regions/_ranking_eval.csv` |
| 9 工具房屋資料 | `stitch_tool/prepare_house_data.py` | `stitch_tool/data/*.json` |
| 10 工具語意門資料 | `stitch_tool/prepare_semantic_doors.py` | `stitch_tool/semantic/*.json` |

第 5 步依 `ROMA_GEO_MATRICES`／`matrices/<版本>/`（現行 `candidate_val`，同學 val 10 房）
與門檻 `PAIR_THRESHOLD`（logit ≥ 0.0 ⇔ 機率 ≥ 0.5）寫出配對清單。必須在匹配前跑，
且失敗即中止；否則 `batch_match` 會退回舊 `_summary.csv` 快照（約 7,069 對）。

第 6 步跑完 `_pairs.csv` 的全部配對，特性：

- **多 GPU**：`WORKERS = "auto"` 時每張卡一個 worker，由 `ParallelScorer` 分派。
  單張卡走序列路徑。
- **可續跑**：已有對應 npz 的配對直接略過。`TIME_BUDGET_H` 逼近工作階段時限即停止並
  保留已完成的部分，重跑接續（需把上一輪的 `roma_cache.zip` 掛回來或留在 working
  目錄）。第 7 步的 `--resume` 同理。
- 順帶重算 `roma_scores.npz`，供 `prepare_house_data.py` 補上 `roma_conf_*` 欄位。

第 4 步關快取（強制每對重算），第 6 步開快取寫檔。

**資料集自動偵測**：掃 `/kaggle/input` 底下的 `DollhouseTask_*` 找出根目錄，
透過 `ROMA_GEO_DATASETS` 交給 `config.py` 解析。

私有 repo 的權杖從 Kaggle Secrets 取得，寫進輸出前會遮蔽。

各步驟的完整輸出寫進 `/kaggle/working/roma_geo_run.log`，儲存格只留每步末尾
`TAIL_LINES` 行與結論。

### 產出

從 Kaggle 的 Output 分頁下載：

| 檔案 | 內容 | 條件 |
|---|---|---|
| `roma_geo_run.log` | 完整執行輸出 | 一律 |
| `roma_geo_results.zip` | 各步驟的 CSV／MD 報告 | 一律 |
| `roma_cache.zip` | RoMa 匹配快取 `*.npz` | `FULL_CHAIN = True` |
| `roma_regions.zip` | 共視區域 per-pair JSON | `FULL_CHAIN = True` |
| `stitch_tool_data.zip` | 工具的房屋與語意門資料 | `FULL_CHAIN = True` |
