# tests/：測試與交付物檢查

一道指令跑完本專案可在當前機器上執行的全部檢查：

```powershell
..\venv\Scripts\python.exe tests\run_all.py
```

離開碼 0 表示沒有任何一項失敗。

## 三層結構

分層的依據是**前提條件**。缺少前提的層級回報「略過」並說明原因，不視為失敗。本模組
會交付給環境各異的對象，多數環境不會具備 GPU 或完整資料集。

| 層 | 前提 | 內容 | 耗時 |
|---|---|---|---|
| 1 單元測試 | 無 | 純函式、對外契約、schema、決定性 | 約 5 秒 |
| 2 交付物檢查 | 磁碟上已有產出 | 資料本身是否符合契約 | 數秒 |
| 3 端到端 | GPU 與資料集 | 實跑一對真實影像，與參考結果比對 | 約 15 秒 |

常用選項：

```powershell
..\venv\Scripts\python.exe tests\run_all.py --fast            # 只跑第 1 層
..\venv\Scripts\python.exe tests\run_all.py -k azimuth        # 只跑名稱含該字串的單元測試
..\venv\Scripts\python.exe tests\run_all.py --deliverables-all # 交付物改為全量檢查
..\venv\Scripts\python.exe tests\run_all.py --full            # 併入環境驗收與跨環境一致性
```

`--full` 另有一項差別：缺少前提時視為失敗。發布前要求所有層級都實際跑過時使用。

## 第 1 層：單元測試

不需要 GPU、資料集、快取或任何既有產出，因此任何一台機器隨時可跑。也可單獨執行：

```powershell
..\venv\Scripts\python.exe -m unittest discover -s tests
..\venv\Scripts\python.exe tests\test_azimuth.py
```

| 檔案 | 涵蓋範圍 |
|---|---|
| `test_azimuth.py` | 方位角規約。四處各自實作的換算須指向同一方向；框跨接縫、規約字串與實作相符 |
| `test_geometry_core.py` | 方位統計、凸包、單側區域幾何、DBSCAN。重點在跨接縫與退化輸入 |
| `test_pose_estimation.py` | RANSAC 本質矩陣、位姿分解、退化偵測。以合成資料的已知位姿為準 |
| `test_analyze_pair.py` | 幾何驗證層的契約：schema 完整性、自適應 conf 階梯、旗標定義、決定性 |
| `test_public_contract.py` | 兩條對外輸出路徑的欄位裁切；並比對對外入口與批次腳本的預設參數是否仍相同 |
| `test_ingest.py` | 上游交界：機率無損、連通分數對稱、方位不得翻面 |
| `test_config.py` | 六項路徑的解析優先序，以及三處環境變數名的一致性 |
| `test_cache_naming.py` | 快取檔名約定、資料集探索、多樓層 GT 的載入順序 |
| `test_stitch_bundle.py` | 工具載入對外交付檔的相容層：欄位對映、無真值標示、共視區域的側別判定 |
| `test_pipeline.py` | 步驟獨立呼叫、可選輸入缺席時的降級標記、未見過的房子、設定由呼叫端指定 |
| `test_semantic_correction.py` | 語意門矯正規則、矯正不改寫相對位姿、交付檔版本欄 |
| `test_package_isolation.py` | 匯入本套件不蓋住使用端的 config／pipeline，也不加任何目錄到搜尋路徑 |
| `test_az_source_guard.py` | az_source 缺席只在連通矩陣缺席時容許 |
| `test_match_mode_defaults.py` | 各進入點匹配模式預設均為 direct |
| `test_import_from_root.py` | 專案根目錄未改 sys.path 時仍能 `import roma_geo` 取得對外 API |
| `test_missing_torch.py` | 缺少 torch 時 verify_pair 等進入點指出缺少什麼與安裝依據 |
| `test_addendum5.py` | 追加5：模組搜尋路徑、產出路徑、部分匹配失敗、降級方位契約、區域開關、快取檔名、彙總累積、輸出精度 |
| `test_addendum6.py` | 追加6：降級保留全部配對、無方位為空值、有矩陣時配對集合不變 |
| `test_addendum7.py` | 追加7：彙總寫入原子性、配對清單為本次範圍、受限來源、路徑優先序、失敗契約 |
| `test_addendum8.py` | 追加8：交付檔名稱驗證與原子寫入、配對清單覆寫、安裝路徑、匯入邊界、失敗契約、測試自備資料集 |
| `test_addendum9.py` | 追加9：名稱守衛在過濾之後、本次清單與批次來源分開、檢查工具路徑與略過、逐對產物降級 |
| `test_addendum10.py` | 追加10：相對位姿逐對判定、測試自備名稱樣本、搜尋路徑敘述 |

期望值一律由**規約或獨立算法**推得，不從受測程式碼反推。合成資料以指定的 R、t 正向
投影出兩側觀測，受測程式碼須解回同一組 R、t；`wedge_width` 另以不同的列舉方式暴力
算一遍對照。照著實作寫的測試只能鎖住現況，無法發現其中的錯誤。

`_support.py` 提供合成資料產生器與環境變數暫時覆寫，並在匯入時把專案根目錄補進
`sys.path`，因此每個測試檔的第一個匯入都是它。

## 第 2 層：交付物檢查

驗證的是資料。用途有二：批次跑完後確認產出可交付；收到別處（其他機器、
其他版本、上游）產生的檔案時，先確認格式相符再接。

```powershell
..\venv\Scripts\python.exe tests\check_deliverables.py          # 各類抽驗 200 份
..\venv\Scripts\python.exe tests\check_deliverables.py --all    # 全量
..\venv\Scripts\python.exe tests\check_deliverables.py --regions_dir <路徑>
```

| 類別 | 來源 | 主要判準 |
|---|---|---|
| per-pair JSON | `Output/roma_regions/` | schema 版本、位姿與旗標三者一致、內點率自洽、區域占比與上限 |
| `_summary.csv` | 同上 | 欄位與批次程式宣告的完全相同 |
| 連通性矩陣 | `matrices/<版本>/` | `(N,N,3)` 與影像清單對齊、連通分數對稱、方位為單位長或未填 |
| house JSON | `stitch_tool/data/` | 索引在範圍內、方位規約公式一致、`az_source` 有值時僅為 `geo` 或 `coarse`；連通矩陣缺席時方位可為空值 |
| 對外交付子集 | `Output/public_houses/` | 不含內部欄位、`has_geometry` 為布林、共視區域兩側齊備；降級時方位可為空值 |

抽樣為決定性等距抽樣，同一批檔案在任何機器上都取到同一組。來源目錄不存在時該類回報
略過，其餘照常檢查。各路徑皆依 `config.py` 解析，可用環境變數改指他處。

## 第 3 層：端到端

`smoke_test.py` 經 `roma_geo.verify_pair()` 跑完整核心路徑，與入庫的參考 JSON 逐欄對照。
判準為兩邊都有位姿、共視區域數相同、方位角差小於 5°。

參考結果 `smoke_ref_58553.json` 存放於本目錄。不可置於 `Output/roma_regions/`：那是批次
的輸出目錄，跑一次全量就會把錨點覆蓋成當下環境的結果，此後本測試變成與自身比對而
永遠通過。

## 與 environments/ 的關係

`environments/check_env.py` 驗的是「這台機器跑不跑得動」，`cross_env_check.py` 驗的是
「整批結論穩不穩」，兩者的對象是環境，且分別由安裝腳本與 Kaggle 流程直接
呼叫，故留在原處。`run_all.py --full` 會一併執行，使單一進入點仍涵蓋得到。

## 相依套件

只用 Python 標準函式庫的 `unittest`，不需要額外安裝任何測試框架
（`environments/requirements-core.txt` 無須改動）。測試本身用到的 `numpy` 與 `pillow`
已是核心相依。習慣 pytest 的人可直接 `pytest tests/`，它能執行 `unittest.TestCase`。
