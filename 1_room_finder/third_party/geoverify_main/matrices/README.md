# 連通性矩陣

連通性粗篩的輸出。本專案據此決定哪些配對送進 RoMa 幾何驗證，
並在標註工具中顯示粗篩的連通機率。

## 上游交界點

粗篩模型把矩陣寫到某個目錄，本專案以 `ROMA_GEO_MATRICES` 指到同一個目錄。
兩邊只透過**檔案**往來，不互相 import：

```
<上游粗篩模型>  →  <某目錄>/<house>.npy + <house>_images.json
                          │
      $env:ROMA_GEO_MATRICES = "<某目錄>"
                          ▼
  select_pairs.py       →  Output/roma_regions/_pairs.csv     ← 挑要算哪些對
  batch_match.py        →  roma_match/cache/*.npz
  batch_pose_from_cache →  Output/roma_regions/*.json + _summary.csv
  covis_ranking_eval    →  _ranking_eval.csv
  prepare_house_data    →  stitch_tool/data/*.json            ← 工具顯示粗篩機率
```


## 版本並存

一個子目錄一個版本，可以同時存在：

```
matrices/
├── candidate_val/           ← 目前使用（.npy 直接置於版本目錄，無子目錄）
├── <版本>/                  ← 亦可含 refined_matrices/（正式流程）與 oof_matrices/（covis_ranking_eval）子目錄
└── <版本>/
```

`config.matrices_root("refined")` / `matrices_root("oof")` 解析到對應子目錄。
版本目錄底下沒有子目錄、`.npy` 直接置於其中時，回退為版本目錄本身。

切換版本，優先序由高到低：

```bash
python stitch_tool/prepare_house_data.py --oof_dir matrices/<版本>
```

```powershell
$env:ROMA_GEO_MATRICES = "D:\somewhere\matrices_v2"
```

或改 `config.py` 的 `MATRICES_VERSION`。

檢查目前解析到哪一版：

```bash
python config.py
```

## 格式契約

新來源必須符合這份契約，否則 `prepare_house_data.py` 會 skip 或算錯。每間房三個檔：

| 檔名 | 形狀 | 內容 |
|---|---|---|
| `<house>.npy` | `(N, N, 3)` | ch0 = 連通分數 · ch1 = sin(az) · ch2 = cos(az) |
| `<house>_images.json` | 長度 N | 影像檔名（**含副檔名**），順序與矩陣的 row/col 對齊 |
| `<house>_doormap.npy` | `(N, 8)` | 每張圖的門方位 8-bin aux logits。**現行上游已不再產出**，消費端已移除；舉出僅為舊目錄容錯 |

`<house>` 必須與 `datasets/` 底下的資料夾名逐字一致（例：`DollhouseTask_58553`、
`DollhouseTask_58472_NoOutdoor`，其中 `_NoOutdoor` 後綴時有時無）。

### ch0 的單位

ch0 的單位由矩陣自己決定，本專案不假設它是 logit 或機率，也不做 sigmoid 轉換。
挑配對的門檻直接與 ch0 比較，因此上游改為輸出機率時只需把門檻換成 0~1 之值。

⚠ **換來源必須重新校準門檻**，不同模型的分數尺度不可互相沿用。不確定新來源的
尺度時，`select_pairs.py --inspect` 會列出分數範圍與各門檻選出的配對數。

### ch0 是有向的

`M[i,j,0]` 與 `M[j,i,0]` 不保證相等；`select_pairs.py --inspect` 會列出實際的不對稱程度。
配對本身無方向，因此挑配對取兩向平均。各消費端的取法：

| 消費端 | 取法 |
|---|---|
| `environments/select_pairs.py` | 兩向平均，`--pool {mean,max,min,upper}` 可換 |
| `pose_regions/covis_ranking_eval.py` | 兩向平均 `0.5*(M[i,j]+M[j,i])` |
| `stitch_tool/prepare_house_data.py` | 只取上三角 `M[i,j]`，作為工具顯示的 `logit` / `prob` |

### 方位角規約

本專案：`az(x) = ((x / W) * 360 - 180) mod 360`，**0° = 影像中央**，±180° = 接縫。

⚠ 子計畫一的 `segmentation/aligner.py` 用 `az = (x / W) * 360`（0° = 影像左緣），
**與本專案差 180°**。接新來源前先確認對方採用哪一種規約。

### 方位通道：型態可換

方向有兩個來源，優先序固定：

| `az_source` | 來源 | 何時採用 |
|---|---|---|
| `geo` | 幾何驗證的 `az_ab_deg` / `az_ba_deg`（`_summary.csv`） | RoMa 有足夠對應且解得出位姿 |
| `coarse` | 上游粗篩的方向輸出 | 其餘所有情況 |

`coarse` 目前取自矩陣的 `ch1/ch2`。新來源若不提供方向，這兩個欄位仍要在、填 0 即可。

上游粗篩之後將由子計畫的模型整體取代，其方向輸出型態未定（方位角或影像上的框）。
方向的解讀集中在 **[`upstream_direction.py`](../upstream_direction.py)**：`from_sincos`
處理現行 `ch1/ch2`，`from_box` 處理框（取中心 x，含跨接縫）。換上游時在該模組新增
一個解碼函式即可，`prepare_house_data.py` 的呼叫端不必更動；讀檔層待格式確定後補上。

⚠ 上游若給的是框，RoMa 對應不足時可用框中心當門口位置，但**必須標記來源**：
那是模型預測，與幾何驗證過的共視區域不同，不可混用。

## 目前版本：candidate_val

| 項目 | 值 |
|---|---|
| 每間房 | `<house>.npy`、`<house>_images.json` |
| ch0 單位 | logit（由上游機率轉入：logit = log(p/(1-p))） |
| 預設門檻 | `config.PAIR_THRESHOLD` = 0.0（機率 ≥ 0.5） |

房數、配對數、分數範圍與各門檻選出的對數以 `select_pairs.py --inspect` 為準。

`oof_matrices/` 只有 `pose_regions/covis_ranking_eval.py` 使用。正式流程只讀
`refined_matrices/`，新版本不提供 oof 也能接，排序評估報告會少一項對照。
