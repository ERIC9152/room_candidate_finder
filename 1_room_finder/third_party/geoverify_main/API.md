# API 與工具規格

本文件描述 Roma_Geo 的程式介面與隨附工具。模組總覽與環境需求見 [README.md](README.md)。

## 開始使用

先依 `README.md` 用 uv 建立環境，再於專案根目錄執行 `uv pip install -e . --python venv\Scripts\python.exe`。相依版本見 `environments/requirements-core.txt`。整房流程步驟需要專案原始碼樹，故須編輯模式安裝；非編輯模式安裝時，這些步驟以 `RomaGeoError` 指出原因。單對驗證只需套件本身。

單對：

```python
from roma_geo import verify_pair, to_public

res = verify_pair("a.jpg", "b.jpg")
pub = to_public(res)
```

整間房：

```python
from roma_geo.config import Settings
from roma_geo.pipeline import run_house

s = Settings(datasets="資料集根目錄", output="產出根目錄", cache="快取目錄")
result = run_house("Office_A", settings=s)
```

`Settings` 的每一項都可由呼叫端傳入。未傳入的項目才讀環境變數或專案預設。逐步呼叫見第 2 節。

安裝後以 `roma_geo` 套件命名空間取用。
命令列維持 `python roma_geo.py a.jpg b.jpg`，或 `python -m roma_geo`。

---

## 0. 整房交付檔的兩項輸出

交付檔頂層帶 `schema_version`。現行值為 `house-v2`。沒有該欄的檔是變更前的格式，既有欄位仍可讀。

### 輸出一：通道方位

欄位：

| 欄位 | 存在條件 | 說明 |
|---|---|---|
| `az_ij` / `az_ji` | 每一對都有欄位 | 建議通道方位（度，0~360）。開啟語意矯正且規則命中時，改為門中心。沒有連通矩陣且沒有幾何方位時為 `null` |
| `az_source` | 有方位值時 | `geo` 來自相對位姿，`coarse` 來自上游關注位置。方位為空值時此欄缺席。語意矯正不改此欄 |
| `semantic_corrected` | 每一對 | 布林。此對的輸出一是否被語意門中心改寫 |

來源標示與是否矯正是兩個獨立欄位。未開啟語意矯正時，`semantic_corrected` 一律為 `false`。

開啟語意矯正時，輸出一與「由輸出二自行推算的方位」會不一致。輸出一是最終建議方向，輸出二是原始幾何。這是設計上的區分。

矯正規則：取與共視區域重疊度最高的門，且共視區域有八成以上落在該門範圍內時才改用門中心。
此規則需要共視區域才能判斷，因此只作用於 `az_source` 為 `geo` 的方位。上游來源（`coarse`）的配對沒有共視區域，方位維持原值。

開關：`Settings(semantic_correct=True)` 或 `export_public_houses(..., semantic_correct=True)`。預設關閉。語意圖缺席時流程照常跑完，矯正標記為否。

`with_regions=False` 只讓共視區域不出現在交付檔。相對位姿與語意矯正仍按各自的條件執行。

沒有連通矩陣時，該房影像的兩兩配對全部寫入。取不到方位的配對，`az_ij` / `az_ji` 為 `null`，且沒有 `az_source`。有連通矩陣時，每一對都有方位值與來源。連通矩陣是否缺席寫在房層的 `has_connectivity_matrix`。

### 輸出二：相對位姿

| 欄位 | 存在條件 | 說明 |
|---|---|---|
| `R` | 僅在 `has_geometry` 為真 | 3×3 旋轉矩陣。`x_j = R · x_i + t` |
| `t` | 僅在 `has_geometry` 為真 | 位移方向，長度為 1 |

沒有跑匹配、或對應點不足以解出位姿時，`R` 與 `t` 缺席。不以零值或單位矩陣填充。

位移向量長度為 1，且跨對之間沒有共用尺度。使用端無法直接以此拼出平面圖。語意矯正不改寫輸出二。

---

## 1. 單對幾何驗證

入口為套件 `roma_geo`。

```python
from roma_geo import verify_pair, verify_pairs, to_public, get_matcher
```

### 1.1 `verify_pair(img_a, img_b, **kwargs) -> dict`

對一組 ERP 全景圖做幾何驗證，回傳完整結果。

**輸入**

| 參數 | 型別 | 預設 | 說明 |
|---|---|---|---|
| `img_a`, `img_b` | `str` | 必填 | ERP 全景圖路徑。兩張圖尺寸可不同 |
| `matcher` | `RoMaMatcher` | `None` | 重用既有 matcher；`None` 時使用模組層單例 |
| `mode` | `str` | `"direct"` | `direct`（生產路徑）或 `core`（RoMaScan 分塊，約貴 5.6 倍） |
| `model_type` | `str` | `"roma_v2"` | 匹配模型版本。目前只支援 `roma_v2` |
| `device` | `str` | `None` | `None` 時自動選 cuda／cpu |
| `use_cache` | `bool` | `True` | 是否讀寫匹配快取的 npz。純加速，不影響結果 |
| `cache_dir` | `str` | `None` | 快取目錄。`None` 時依環境變數 `ROMA_GEO_CACHE` 再回落到 `roma_match/cache/`。本函式除此之外不讀任何環境設定 |
| `house` | `str` | `None` | 房子名；`None` 時從路徑推導 |
| `keep_inliers` | `bool` | `True` | `False` 時不回傳 `inliers`（可省下 JSON 體積大宗） |

另可覆寫下列演算法參數（一般不需要，見 §1.5）：
`conf_min` `retry_conf` `max_pts` `thresh_deg` `iters` `seed` `sampler` `lat_max`。
傳入未定義的參數會 `raise TypeError`。

**輸出**：`dict`。欄位全文見
[`pose_regions/README.md`](pose_regions/README.md) 的「per-pair JSON schema」。主要欄位：

| 欄位 | 型別 | 說明 |
|---|---|---|
| `schema_version` | `str` | 產出格式識別字串，供批次續跑判斷是否需重算 |
| `house` / `img_a` / `img_b` | `str` | 房子名與兩張圖的檔名（不含副檔名） |
| `img_size_a` / `img_size_b` | `[W, H]` | 原圖尺寸，每間房可以不同（5376×2688 ~ 8192×4096） |
| `params` | `dict` | 實際使用的演算法參數 |
| `n` | `dict` | `raw` / `conf` / `used` / `inlier` / `lat_drop` 各階段點數 |
| `conf_used` | `float\|null` | 自適應後實際採用的 conf 門檻（0.65／0.3／null） |
| `inlier_rate` | `float\|null` | `n.inlier / n.used` |
| `pose` | `dict\|null` | `R` `t` `cam_b_in_a` `az_ab_deg` `az_ba_deg` `rot_angle_deg` `baseline_txz` |
| `quality` | `dict` | 對極殘差、純旋轉殘差、方位楔形寬度與覆蓋度 |
| `flags` | `dict` | `low_matches` `no_covisibility` `degenerate_rot_only` `doorway_wedge` |
| `regions` | `list` | 共視區域，依點數降冪 |
| `inliers` | `list` | 內點明細 `[x_a, y_a, x_b, y_b, conf]`，最多 2000 點 |
| `timing` | `dict` | `match_s` / `verify_s` / `total_s` |

`pose` 為 `null` 且 `flags.no_covisibility` 為 `true` 代表找不到足夠的共視證據。
兩張圖仍可能相連，只是視野不重疊或對應點不足。

**方位角規約**：`az(x) = ((x / W) * 360 - 180) mod 360`，0° = 影像中央，±180° = 接縫。
`pose.az_ab_deg` 的範圍是 `(-180, 180]`。

**位姿規約**：`x_b = R · x_a + t`，`t` 為單位向量（尺度未知），
`cam_b_in_a = -Rᵀt` = B 相機在 A 座標系中的方向。

### 1.2 `to_public(res) -> dict`

把 `verify_pair()` 的結果裁成對外契約。
捨去 `R` / `t` / `cam_b_in_a` / `inliers` / `params` / `n` / `flags` 等內部量。
`quality` 只取兩側的共視楔形寬。

```json
{
  "schema_version": "wp1-step3-v2",
  "house": "DollhouseTask_58553",
  "img_a": "08ddb590-a6df-409d-850f-e45f59efa7a0",
  "img_b": "08ddb590-ad9f-4647-8dab-ccfd0b14c8da",
  "img_size_a": [8192, 4096],
  "img_size_b": [8192, 4096],
  "conf_used": 0.65,
  "inlier_rate": 0.6769,
  "az_ab_deg": 178.6,
  "az_ba_deg": -174.61,
  "az_wedge_a_deg": 111.1,
  "az_wedge_b_deg": 156.4,
  "no_covisibility": false,
  "regions": [
    {
      "n": 499,
      "frac_of_inliers": 0.841,
      "a": { "polygon": [[3318.8, 2907.9], [3370.7, 1370.0], "…"],
             "area_frac": 0.0475, "az_range": [-34.2, 9.0] },
      "b": { "polygon": [[7859.2, 1676.8], [7879.7, 1600.0], "…"],
             "area_frac": 0.015,  "az_range": [165.4, 192.6] }
    }
  ]
}
```

`az_wedge_a_deg` / `az_wedge_b_deg` 為共視楔形寬：把該側內點的方位角排成一圈之後，包住其中 80% 所需的最窄圓弧寬度。
窄代表共視證據集中在單一方向，位姿的條件較差。
判斷單一結果是否可信時取兩側較小者。
無位姿時為 `null`。

`regions[].a` / `.b` 為同一塊實體空間分別在 A、B 兩張圖上的凸包多邊形，
座標為原圖 ERP 像素。跨接縫的區域其 `x` 可能超出 `[0, W)` 以保持多邊形連續，
繪製時自行取模。`area_frac` 為多邊形面積佔全圖比例（ERP 像素面積，極區有失真）。

### 1.3 `verify_pairs(pairs, progress=False, **kwargs)`

多組圖的產生器版本，全程重用同一個 matcher。

```python
for res in verify_pairs([(a1, b1), (a2, b2)], progress=True):
    ...
```

建構 matcher 需載入 torch、CUDA 與模型權重（約 6.4 秒），多對處理用這個或自行傳入
`matcher=` 以重用。逐對 `yield`。

### 1.4 `get_matcher(model_type, device, use_cache, cache_dir) -> RoMaMatcher`

取得模組層共用的 matcher 單例。`conf_threshold` 固定 0.1，由幾何驗證層
自行做自適應分層（0.65 → 0.3）。

單例以參數組合為鍵（含 `cache_dir`），同一行程內可並存多組設定。

### 1.5 演算法參數

`roma_geo.DEFAULTS` 必須與批次幾何驗證腳本的 argparse 預設逐項相同。
`Output/roma_regions/` 的既有產出皆以這組參數產生。

| 參數 | 值 | 說明 |
|---|---|---|
| `conf_min` | 0.65 | 主要 conf 門檻 |
| `retry_conf` | 0.3 | 點數不足時的降級門檻 |
| `max_pts` | 4000 | 超過即抽樣，加速 RANSAC |
| `thresh_deg` | 0.5 | 對極內點角度門檻 |
| `iters` | 2000 | RANSAC 迭代數 |
| `seed` | 0 | RANSAC 亂數種子 |
| `sampler` | `stratified` | conf 漸進池 + 方位分桶 |
| `lat_max` | 60.0 | 天底／天頂前濾角 |

`tests/smoke_test.py` 內含這組參數與參考 JSON 的逐項比對，`tests/test_public_contract.py`
另以語法樹比對本表與批次腳本的 CLI 預設是否仍然逐項相同。

### 1.6 CLI

```bash
python roma_geo.py a.jpg b.jpg --public --out result.json
```

`--mode` `--conf_min` `--no_cache` `--public` `--out`。省略 `--out` 時輸出到 stdout。

### 1.7 效能

單對端到端以 RoMa 匹配為主，幾何驗證佔比很小；模型建構為一次性成本，
多對處理務必重用 matcher。驗證時間呈雙峰：有位姿的對較久，無共視證據的對提早退出。

逐項數字與外推耗時見 `environments/timing_torch210_n120.md`（由
`environments/local_timing_bench.py` 產生，換機器請重跑）。

---

## 2. 逐步呼叫

每一個處理步驟都是函式，輸入與輸出由參數給定。不依賴前一步在全域狀態或固定位置留下的隱含資料。
匯入 `roma_geo` 不修改模組搜尋路徑，呼叫流程步驟也不修改：專案根目錄、`src`、`stitch_tool`、`environments` 都不會進到呼叫端的搜尋路徑。步驟用到的專案內腳本以檔案路徑載入，模組名帶 `roma_geo._project.` 前綴，因此呼叫端自己的 `config`、`pipeline` 等同名模組不會被搶走。

```python
from roma_geo.config import Settings
from roma_geo.pipeline import (
    ingest_connections, select_pairs, match_pairs, estimate_poses,
    evaluate_ranking, prepare_house, prepare_semantic, export_public_houses,
    run_house,
)

s = Settings(datasets="...", output="...", cache="...", matrices="...")
```

失敗時丟出 `errors.RomaGeoError`，訊息含「輸入」與「條件不成立」。

每個步驟回傳 `StepResult`：`outputs` 為產出路徑或計數，`degradations` 為缺席的可選輸入。
`run_house(house, settings, matcher=None)` 依序呼叫下列步驟，並把降級紀錄去重後放在 `PipelineResult.degradations`。

| 步驟 | 函式 | 主要輸入 | 主要輸出 |
|---|---|---|---|
| 連通輸出轉矩陣 | `ingest_connections` | 上游 JSON 目錄 | 連通矩陣目錄 |
| 挑配對 | `select_pairs` | 連通矩陣或影像目錄 | `_step_pairs.csv`（本次範圍） |
| 匹配 | `match_pairs` | 配對清單、影像、matcher | 匹配快取 npz |
| 幾何驗證 | `estimate_poses` | 匹配快取、影像 | 逐對 JSON、`_summary.csv` |
| 排序評估 | `evaluate_ranking` | `_summary.csv`、標註真值 | `_ranking_eval.csv`（無真值則略過） |
| 組房 | `prepare_house` | 影像，可加矩陣與 `_summary.csv` | 內部房子 JSON |
| 語意門 | `prepare_semantic` | 語意圖 | 語意門 JSON（無語意圖則略過） |
| 對外匯出 | `export_public_houses` | 內部房子 JSON | 對外交付子集 |

任取兩個相鄰步驟，只要把上一步的產出路徑傳給下一步即可單獨呼叫。匹配步驟可傳入 `matcher=`，測試或無 GPU 時注入替身。

`match_pairs` 需要 GPU（或呼叫端提供的 matcher）。其餘步驟為 CPU。
`mode` 只接受 `direct` 與 `core`。部分配對失敗時，`outputs` 含 `n_fail`、`failures`、`ok_pairs`，後續幾何驗證只處理成功的配對。

### 2.1 `config.Settings`

| 參數 | 對應 | 未傳入時 |
|---|---|---|
| `datasets` | 全景資料集根目錄 | `ROMA_GEO_DATASETS` 或專案內 `datasets/` |
| `matrices` | 連通矩陣目錄 | `ROMA_GEO_MATRICES` 或專案內 `matrices/<版本>/`。不隨 `output` 變動 |
| `output` | 產出根目錄 | `ROMA_GEO_OUTPUT` 或專案內 `Output/` |
| `cache` | 匹配快取 | `ROMA_GEO_CACHE` 或 `roma_match/cache/` |
| `pair_threshold` | 挑配對門檻 | `ROMA_GEO_PAIR_THRESHOLD` 或 `PAIR_THRESHOLD` |
| `house_data` | 內部房子 JSON 目錄 | `ROMA_GEO_HOUSE_DATA`，否則已指定產出根目錄（建構參數或 `ROMA_GEO_OUTPUT`）時為 `<output>/house_data`，再否則專案內 `stitch_tool/data/` |
| `semantic` | 語意圖根目錄 | `<datasets>/semantic/` |
| `semantic_out` | 語意門 JSON 目錄 | `ROMA_GEO_SEMANTIC_OUT`，否則已指定產出根目錄（建構參數或 `ROMA_GEO_OUTPUT`）時為 `<output>/semantic`，再否則專案內 `stitch_tool/semantic/` |
| `public_houses` | 對外交付目錄 | `ROMA_GEO_PUBLIC_HOUSES` 或 `<output>/public_houses` |
| `connections` | 上游連通 JSON 目錄 | `ROMA_GEO_CONNECTIONS`；未指定則缺席，略過轉矩陣 |
| `semantic_correct` | 是否對輸出一做語意矯正 | `False` |

優先序一律為：建構參數，其次環境變數，最後預設值。輸入類路徑（資料集、連通矩陣、語意圖、上游連通 JSON）不隨產出根目錄變動。相對路徑以行程工作目錄為基準轉成絕對路徑。語意門目錄未指定時視為缺席，不組相對路徑。寫入目標（共視區域、對外交付、內部房子 JSON、語意門輸出）的預設位置跟隨解析後的產出根目錄，不論該值來自建構參數或環境變數。

逐步呼叫的本次配對清單預設為 `_step_pairs.csv`：每次覆寫，只含本次選出的配對，是下一步的輸入範圍。批次流程的穩定配對來源是 `_pairs.csv`：由命令列挑配對腳本寫入（既有檔須加 `--force`），`batch_match.py` 見到該檔存在就沿用。兩個預設位置分開，逐步呼叫不會改寫批次來源。彙總檔 `_summary.csv` 是產出帳本：同一路徑多次呼叫時累積，不覆寫其他房子。欄位不符時彙總檔以 `RomaGeoError` 中止，原檔不變。零位元組的既有彙總檔視為沒有既有內容。

### 2.2 可選輸入缺席時的回傳

`Degradation.input_name` 取值：

| 值 | 含義 |
|---|---|
| `upstream_connectivity` | 沒有上游連通矩陣（或連通 JSON） |
| `ground_truth` | 沒有 HOTSPOT 標註 |
| `semantic_maps` | 沒有門語意圖 |
| `matching` | 匹配步驟有部分配對失敗 |
| `pair_jsons` | 相對位姿整批缺席（每一對有 has_geometry 的配對都沒有逐對產物）。命令列結束狀態為 2。交付物檢查判失敗。部分缺漏、或產物檔在卻沒有位姿，則中止 |

對應的降級行為見 README「可選輸入」。

---

## 3. 工具

下列指令以專案根目錄為工作目錄時可用預設相對路徑；以 `Settings` 或 CLI 參數指定路徑時，工作目錄不限。`<py>` 代表 `venv\Scripts\python.exe`。

### 3.1 標註工具

瀏覽器介面，用於檢視配對、共視區域遮罩、裁切對與通道方位標示。

```bash
<py> stitch_tool/prepare_house_data.py
<py> stitch_tool/server.py --port 8010
```

開 http://localhost:8010。

HTTP 介面：

| 路由 | 說明 |
|---|---|
| `GET /api/houses` | 房子清單 |
| `GET /api/house/<house>` | 單一房子的配對與方位資料 |
| `GET /api/regions/<house>?a=<圖A>&b=<圖B>` | 共視區域 JSON（命中反序檔時回 `swap: true`） |
| `GET /api/semantic_doors/<house>` | 語意門抽取結果（無語意圖回 404） |
| `GET /api/state/<house>` · `POST /api/state/<house>` | 人工標註進度存讀 |
| `POST /api/roma_match/<house>` | 即時 RoMa 比對（延遲載入模型） |
| `GET /images/<house>/<檔名>` | ERP 影像 |

操作見 [`stitch_tool/使用說明.md`](stitch_tool/使用說明.md)。

### 3.2 組房資料

把連通性矩陣（若有）與幾何驗證結果整理成前端可讀的房子 JSON。沒有矩陣時改由影像清單與幾何驗證結果組房。

- 輸入：連通矩陣目錄、ERP 圖、可選的 HOTSPOT 標註、可選的幾何驗證 CSV
- 輸出：內部房子 JSON（預設 `stitch_tool/data/<house>.json`）與 `_index.json`
- 命令列在 `roma_scores.npz` 存在時會把 `roma_*` 分數寫進內部 JSON，供標註工具排序。`pipeline.prepare_house` 不讀該檔，對應欄位為空值。對外匯出會裁掉 `roma_*`
- 矩陣路徑優先序：`--oof_dir` > 環境變數 `ROMA_GEO_MATRICES` > `config.MATRICES_VERSION`
- 有位姿的 pair 會以幾何驗證的 `az_ab` / `az_ba` 覆寫粗篩方位，並記於 `az_source`

### 3.3 語意門轉檔

把門語意圖抽取結果轉存成工具可讀的格式。

```bash
<py> stitch_tool/prepare_semantic_doors.py
<py> stitch_tool/prepare_semantic_doors.py --refresh
```

輸出 `stitch_tool/semantic/<house>.json`。只有具語意圖的房子會產出，其餘前端顯示 404。

### 3.4 批次幾何驗證

讀 RoMa 匹配快取，產生逐對 JSON 與彙總 CSV。

```bash
<py> pose_regions/batch_pose_from_cache.py --out Output/roma_regions
```

輸入為快取目錄裡符合
`{house}_{imgA}_{imgB}_direct_roma_v2_std0.1_face512.npz` 命名的檔案。
標註真值讀資料集裡的 `*_HOTSPOT.json`；沒有該檔時 GT 欄位為空值。

程式介面為 `pose_regions.batch_pose_from_cache.run_pose_batch(...)`，路徑全部由參數傳入。

### 3.5 單對診斷

輸出內點疊圖與俯視圖。

```bash
<py> pose_regions/estimate_relative_pose.py --cache_npz <npz> --out pose_out
<py> pose_regions/estimate_relative_pose.py --selftest
```

### 3.6 共視區域疊圖

把共視區域 JSON 畫成疊圖。見 `pose_regions/render_region_overlay.py`。

### 3.7 環境工具

| 腳本 | 用途 |
|---|---|
| `setup_env.ps1` | 建立 venv 並安裝全部相依，結束時自動驗收 |
| `check_env.py` | 三項實跑驗證（CUDA kernel／romav2 匯入／幾何 selftest）決定通過與否 |
| `cross_env_check.py` | 跨環境一致性檢查，與既有 `_summary.csv` 逐對比較 |
| `local_timing_bench.py` | 單對耗時基準，決定性抽樣，可跨環境比較 |

用法見 [`environments/README.md`](environments/README.md)。

### 3.8 測試

```bash
venv/Scripts/python.exe tests/run_all.py
```

單一進入點，分三層：單元測試（無前提）、交付物檢查（掃描既有產出是否符合契約）、
端到端煙霧測試（需 GPU 與資料集，缺少時自動略過）。用法與各層判準見
[`tests/README.md`](tests/README.md)。
