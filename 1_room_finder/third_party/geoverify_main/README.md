# Roma_Geo

輸入一對 ERP 全景圖，輸出經幾何驗證的共視區域、通道方位角與可信度。

## 開始使用

先建立環境。本專案用 uv，虛擬環境裡沒有 pip。Windows：

```powershell
cd environments
.\setup_env.ps1
```

Linux：

```bash
bash environments/setup_env.sh
```

相依版本釘在 `environments/requirements-core.txt`。torch 由安裝腳本從 CUDA 13 索引安裝，不寫入打包設定。

把本套件裝進該虛擬環境（在專案根目錄）。
整房流程步驟依賴標註工具與挑配對腳本，須以編輯模式安裝並保留專案原始碼樹。
非編輯模式安裝時，這些步驟會以 `RomaGeoError` 指出原因。單對驗證只需套件本身。

```powershell
uv pip install -e . --python venv\Scripts\python.exe
```

單對：

```python
from roma_geo import verify_pair, to_public

res = verify_pair("a.jpg", "b.jpg")
pub = to_public(res)
```

同房判別吃兩張圖，幾何結果先讀產出目錄，沒有再跑上面這段並寫回。用法見 `src/roma_geo/same_room/README.md`。

尚未建立環境時，第一次呼叫會指出缺少哪些相依，以及相依版本在 `environments/requirements-core.txt`。

呼叫端指定影像路徑即可。不必先切換工作目錄，也不必設定環境變數。快取目錄可用 `cache_dir=` 傳入。
匯入 `roma_geo` 不修改模組搜尋路徑，呼叫整房流程步驟也不修改：專案根目錄、`src`、`stitch_tool`、`environments` 都不會進到呼叫端的搜尋路徑。步驟用到的專案內腳本以檔案路徑載入，模組名帶 `roma_geo._project.` 前綴，因此呼叫端自己的 `config`、`pipeline` 等同名模組不會被搶走。

整間房：

```python
from roma_geo.config import Settings
from roma_geo.pipeline import run_house

s = Settings(datasets="資料集根目錄", output="產出根目錄", cache="快取目錄")
result = run_house("Office_A", settings=s)
```

語意矯正由呼叫端開啟：`Settings(..., semantic_correct=True)`。預設關閉。
矯正只作用於 `az_source` 為 `geo` 的方位：規則要用共視區域判斷門是否包住共視，上游來源的配對沒有共視區域，方位維持原值。

參數與欄位見 [API.md](API.md)。

## 可選輸入

上游語意圖、上游連通矩陣、標註真值、相對位姿的逐對產物均可缺席。缺席時流程繼續，並在回傳的 `degradations` 寫明缺了哪一項、改用什麼。

| 缺席項目 | 降級 | 如何辨識 |
|---|---|---|
| 上游連通矩陣 | 該房全部影像兩兩配對 | `degradations` 含 `upstream_connectivity`；房子資料帶 `has_connectivity_matrix=false` |
| 標註真值 | `gt_*` 為空值；排序評估略過 | `degradations` 含 `ground_truth`；沒有標註時 `gt_connected` 為 `null`，標註存在且該對不在相鄰清單時為 `false` |
| 上游語意圖 | 語意門步驟略過 | `degradations` 含 `semantic_maps` |
| 相對位姿的逐對產物 | 其餘欄位照常輸出；有 `has_geometry` 的配對不帶 R/t | `degradations` 含 `pair_jsons`；交付物檢查判失敗（整批缺席）；命令列結束狀態 2 |

方位來源寫在 `az_source`：`geo` 為幾何驗證，`coarse` 為上游連通方向。沒有連通矩陣時仍寫出該房全部配對；取不到方位的配對，方位為空值、來源欄位缺席。連通矩陣是否缺席寫在 `has_connectivity_matrix`。

## 對外契約

單對結果經 `to_public()` 裁切；整房連接圖經 `export_public.house_to_public()` 裁切。欄位名稱與語意見 API.md。下列符號的輸出形狀與語意固定：

- `export_public.PAIR_FIELDS`、`HOUSE_FIELDS`
- `roma_geo.PUBLIC_FIELDS`、`PUBLIC_REGION_FIELDS`、`PUBLIC_REGION_SIDE_FIELDS`
- `roma_geo.to_public()`、`roma_geo.region_to_public()`

## 限制

- 相對位姿的尺度未知。
- `flags.no_covisibility` 表示這一對湊不出足夠的對極內點，兩張圖仍可能相連。
- RoMa 匹配需要 GPU。幾何驗證在 CPU 上執行。
- 逐步呼叫幾何驗證時請傳明確的配對清單，流程沿用清單裡的房子名與影像名。命令列掃描快取時若檔名無法唯一回推，會列出略過的檔與原因。
- 房屋目錄含 `_RESTRICTED` 標記檔時，對外匯出整批停下，此次已寫出的交付檔會刪除。

未見過的房子只要有全景影像即可進入流程，不要求既有資料集才有的欄位或檔名格式。

## 環境

| 項目 | 版本 |
|---|---|
| Python | 3.11 |
| torch | 2.10.0+cu130 |
| numpy | 1.26.4 |
| opencv-python | 4.11.0 |
| scikit-learn | 1.8.0 |
| pillow | 12.0.0 |

```powershell
cd environments
.\setup_env.ps1
```

`environments/check_env.py` 實跑三項：CUDA kernel、romav2 匯入、幾何核心 selftest。三項都過即代表這個環境可用。上表是開發時的參考配置；版本不同時該腳本標為 `DIFF`，結論仍以實跑為準。

## 測試

```powershell
venv\Scripts\python.exe tests\run_all.py
```

分三層：單元測試、交付物檢查、端到端煙霧測試。缺少前提的層級回報略過。離開碼 0 表示沒有任何一項失敗。各層判準見 `tests/README.md`。

## 路徑與門檻

呼叫端以 `config.Settings(...)` 傳入。未傳入的項目才讀環境變數或專案預設。
只指定產出根目錄（建構參數或 `ROMA_GEO_OUTPUT`）時，寫入目標（共視區域、對外交付、內部房子 JSON、語意門輸出）落在該目錄底下，不會寫進安裝來源。連通矩陣是輸入，不隨產出根目錄變動。

| 項目 | 環境變數 | 預設 |
|---|---|---|
| 全景資料集 | `ROMA_GEO_DATASETS` | `datasets/` |
| 門語意圖 | （跟隨資料集） | `<datasets>/semantic/` |
| 連通性矩陣 | `ROMA_GEO_MATRICES` | `matrices/<版本>/`（不隨產出根目錄變動） |
| 上游連通 JSON | `ROMA_GEO_CONNECTIONS` | 無（缺席則略過轉矩陣） |
| 挑配對門檻 | `ROMA_GEO_PAIR_THRESHOLD` | `config.PAIR_THRESHOLD`（現值 0.0） |
| 產出根目錄 | `ROMA_GEO_OUTPUT` | `Output/` |
| 匹配快取 | `ROMA_GEO_CACHE` | `roma_match/cache/` |
| 內部房子 JSON | `ROMA_GEO_HOUSE_DATA` | 已指定產出根目錄時為 `<output>/house_data`，否則 `stitch_tool/data/` |
| 語意門輸出 | `ROMA_GEO_SEMANTIC_OUT` | 已指定產出根目錄時為 `<output>/semantic`，否則 `stitch_tool/semantic/` |
| 對外交付目錄 | `ROMA_GEO_PUBLIC_HOUSES` | `<output>/public_houses` |

```bash
python config.py
```

印出目前每一項解析到哪裡。匹配快取可重新產生，不屬於交付內容。

挑配對門檻的單位由矩陣決定，本模組不做 sigmoid。換上游來源時重新校準；`select_pairs.py --inspect` 列出分數範圍與各門檻選出的對數，不寫檔。

## 處理步驟

1. 上游連通輸出（可選）轉成連通矩陣。
2. 連通矩陣或影像清單產生待驗證的配對。
3. RoMa 密集匹配，寫入快取。
4. 由匹配點估計共視區域與通道方位。
5. 組成每間房的連接圖。語意門為可選。
6. 裁成對外交付子集。

單對 `verify_pair()` 覆蓋步驟 3 與 4。

方向規則：correspondence 足夠時採用幾何驗證；不足且上游有方向時採用上游方向，並以 `az_source` 標明。

## 模組一覽

| 模組 | 職責 |
|---|---|
| `roma_geo` | 單對入口：`verify_pair` / `verify_pairs` / `to_public` |
| `roma_geo.same_room` | 同房判別：`by_wedge` / `by_door` |
| `pipeline` | 逐步呼叫與整房流程 |
| `config.Settings` | 路徑與門檻，由呼叫端指定 |
| `ingest_connections` | 上游連通 JSON 轉矩陣 |
| `export_public` | 整房對外子集 |
| `core_roma_match` | RoMa 匹配的唯一實作入口 |
| `roma_match` | 匹配封裝與快取 |
| `pose_regions` | 幾何驗證（純 numpy） |
| `semantic_doors` | 門語意圖抽取 |
| `stitch_tool` | 標註與檢視 |
| `environments` | 建置、驗收、挑配對、批次匹配 |
| `tests` | 測試總入口 |

連通矩陣格式見 `matrices/README.md`。方位解碼見 `upstream_direction.py`。
