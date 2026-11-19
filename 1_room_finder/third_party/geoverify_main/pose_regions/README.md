# pose_regions：相對位姿與共視區域(RoMa 匹配的幾何驗證層)

從 RoMa 的 ERP 全景匹配點估計兩張圖之間的相對位姿(R、t 方向;尺度未知),
並以對極幾何過濾不合理匹配、萃取共視區域多邊形。不需要 GT pose / 深度 / layout。

可直接匯入:`from pose_regions import analyze_pair`。核心為純 numpy、匯入零副作用、
與 cwd 無關,Kaggle 或即時匹配流程皆可使用。公開 API 見 `__init__.py`。

## 檔案

| 檔案 | 用途 |
|---|---|
| `estimate_relative_pose.py` | 單對主程式:三種輸入 → RANSAC essential → R/t、內點 overlay、top-down 圖 |
| `batch_pose_from_cache.py` | 全量批次:快取內每一對 → per-pair JSON + `_summary.csv` + 報告 |
| `spike_cache_consistency.py` | 驗證 cache npz 的座標約定(可重跑) |
| `spike_cache_consistency_report.md` | 上述驗證的執行結果 |
| `covis_ranking_eval.py` | 共視/幾何訊號當**連通性排序分數**的判別力與上限評估;產生 `_ranking_eval.csv` |
| `covis_ranking_report.md` | 上述評估結果 |
| `render_region_overlay.py` | 把共視區域畫在 A/B 兩張圖上(抽樣或 on-demand) |

## 三種輸入

```bash
# A) 快取 npz(推薦;免 GPU、免匯出。圖自動從 datasets/ 找,--img_a/--img_b 可省略)
venv/Scripts/python.exe pose_regions/estimate_relative_pose.py \
    --cache_npz roma_match/cache/DollhouseTask_..._face512.npz --out pose_out

# B) 單對 live 跑 RoMa(需 GPU venv)
python estimate_relative_pose.py --match --img_a a.jpg --img_b b.jpg --mode core --conf 0.65

# C) 匹配點 CSV(欄位:x_a, y_a, x_b, y_b, confidence)
python estimate_relative_pose.py --csv a_to_b.csv --img_a a.jpg --img_b b.jpg

# 幾何核心自我驗證(合成資料;目前 PASS)
python estimate_relative_pose.py --selftest
```

`--cache_npz` 補充:
- `--img_a/--img_b` 給反了會自動交換座標(同 `roma_match/matcher.py` 讀快取的 is_swapped 行為),R/t 隨之取逆(已驗證:R 恰為轉置)。
- `--conf` 未指定時預設 **0.65**(cache 底床 0.1 的低段大多是雜訊)。

## cache npz 座標約定

`roma_match/cache/` 的檔名全數符合:

```
{house}_{imgA}_{imgB}_direct_roma_v2_std0.1_face512.npz
```

- 內容:`matches` (K,5) = **(x_a, y_a, x_b, y_b, conf)**,座標 = **原圖 ERP 像素**(A 圖前兩欄、B 圖後兩欄)。
- 座標為**連續值**(RoMaV2 內部 512×512 warp 雙線性取樣後放大回原圖;非格點量化)。
- conf 底床 **0.1**(存完整結果供不同門檻共享;上層讀取時再過濾)。
- **原圖尺寸依房子而異**(7 種:5376×2688 ~ 8192×4096)。換算球面射線時必須用
 **該房實際 W×H**,不能寫死 8192×4096。

此約定由 `spike_cache_consistency.py` 驗證(座標範圍、最近鄰距離、對極交叉驗證、
全量超界掃描四項),結果見 `spike_cache_consistency_report.md`。

## RANSAC 取樣策略(`--sampler`,單對與批次共用)

8 點法每次迭代抽 8 點 minimal sample(E 只有 5 自由度,存在 5 點法,但線性 8 點法在點數
充足時是標準務實選擇;輸入是單位球面射線,天然良態,不需 Hartley 正規化)。兩種取樣:

- **`stratified`(預設)**:(1) **conf 漸進池**(PROSAC-lite)。點按 RoMa conf 降冪排序,
  取樣池從 top 25% 逐塊(每 256 迭代)放寬到 100%,高 conf 點先被抽到;
  (2) **方位分桶**。每個 sample 的 8 點在 A 側方位 12 桶(30°)間輪替,提升空間分散度/
  條件數;只在「有點的桶」之間輪替,門洞 pair 支撐窄不會被強迫散開。
  conf 只影響「先抽誰」;內點判定與最終 refit 都對全部點做,不被 conf 偏見污染。
- **`uniform`**:完全均勻隨機(對照/重現舊數據用)。

兩種取樣結束後都做 **LO 式迭代 refit**:全內點重擬合 → 重算內點,循環至內點集不再變大
(只在變好時接受,最多 3 輪)。

**預設 `stratified`。** 兩者耗時相同;`uniform` 的內點「數」略多,但那是 RANSAC 自家的
目標函數,外部檢核(方位誤差 vs GT hotspot)是 `stratified` 明顯較好,故以後者為準。
`--selftest` 對兩種取樣都通過。

## nadir 前濾(`--lat_filter`,批次限定)

天底腳架帶(lat < −60°)在每張全景圖上長得一樣,會產生系統性的跨圖假匹配。這種匹配過得了對極驗證但沒有語意。`--lat_filter 60` 在 RANSAC 前丟掉「任一側
|lat| > 60°」的匹配點(天底與天頂對稱處理),丟掉的點數記入 JSON 的 `n.lat_drop`、
參數記入 `params.lat_max`。

**預設 60;設 0 關閉。** 開啟後方位精度不降反升,退化旗標的誤報減半,代價是位姿
覆蓋率下降(流失的多為假證據 pair)。採用依據與全量 A/B 的數字見 `nadir_ab_report.md`。

## `Output/roma_regions/` 使用限制

逐 pair 的 JSON 與 `_summary.csv` 由 `batch_pose_from_cache.py` 產生,
數量、覆蓋率與各項分佈見同批產出的 `batch_pose_report.md`。

- **逐對數值不跨執行重現。** 匹配階段的取樣帶有隨機性,同一台機器重跑兩次的差異
  與換一張 GPU 相當(實測:兩者的 az_ab 差中位皆約 1°、位姿有無一致率皆約 90%)。
  差異集中在 `conf_used=0.3` 的撈回層與低 inlier 的 pair;`n_inlier ≥ 400` 者
  差中位 0.45°、幾乎不翻面。**引用覆蓋率或逐對數值時必須註明是哪一次的產出。**
- **`n_regions` 只能當統計量,不能當 per-pair 的穩定標籤。** DBSCAN 對點集微擾敏感,
  同一對在不同次執行可能落在不同的區域數;整體分佈則穩定。需要穩定的 per-pair 依據
  時,改用主共視區域的中心方位。
- 聚合結論不受上述變異影響:排序 AP、FP@R90、覆蓋率與對 GT 的方位誤差,
  重跑之間的變動都在小數點後第三位的量級。

## 輸出與判讀

- `inliers_overlay.jpg`:A|B 並排,綠=對極內點、紅=外點(半透明)。
- `topdown.png`:內點三角化俯視散點 + 兩相機位置(尺度自由)。
- stdout:內點率、R、t(單位向量)、`median rot-only err`
  (內點若能被純旋轉解釋 → 疑似近零基線,t 方向不可靠,會印警告)。

## 全量批次 + 品質指標(`batch_pose_from_cache.py`)

```bash
venv/Scripts/python.exe pose_regions/batch_pose_from_cache.py             # 全量
venv/Scripts/python.exe pose_regions/batch_pose_from_cache.py --limit 20  # 冒煙
# --resume 跳過已有 JSON 續跑;--conf/--max_pts/--thresh_deg/--iters 可調
```

輸出三份:
1. `Output/roma_regions/{house}_{imgA}_{imgB}.json`：per-pair 結果(下方 schema;**不入 git**,約 20 分鐘可重生)
2. `Output/roma_regions/_summary.csv`：每 pair 一列指標總表(入 git,供驗收與出圖)
3. `Output/roma_regions/batch_pose_report.md`：耗時與分佈自動報告(耗時、inlier 率
   分佈、GT 方位交叉誤差、旗標統計;跟著 `--out` 走,故與該批的 `_summary.csv` 同批)

### 自適應 conf(`--retry_conf`,預設 0.3)

基準 conf 0.65 需 ≥30 點才進 RANSAC;**點數 <30 或內點 <8 時自動降到 0.3 重試**
(需 ≥8 點),實際門檻記入 `conf_used`(0.65 / 0.3 / null)。兩級都湊不出 8 個內點
才標 `no_covisibility`。動機:穿門窄楔形 pair 在 0.65 常只剩 <20 點(位姿覆蓋率是
拼接可用性的關鍵)。

⚠ **欄位門檻說明**:`flags.low_matches` 記錄基準層(conf 0.65)點數是否 <30;
`n.conf` 與 `n.used` 則記錄最終採用層(`conf_used`)的點數,`n.used` 另外套了
`max_pts` 上限(超過即抽樣,故 `n.used` ≤ `n.conf`,`inlier_rate` 的分母是 `n.used`)。
因此降級重試時會出現「`low_matches=true` 但 `n.conf` ≥30」,此為預期現象。

**⚠ 撈回層品質重尾**:0.3 級的方位誤差分佈,尾巴遠比 0.65 級長。下游必須用
`conf_used` + inlier 率 + 旗標一起 gate,**0.3 級位姿不可直接當 0.65 級用**。
各層的覆蓋率與誤差分位數見 `batch_pose_report.md`。

語意提醒:`no_covisibility` = 無共視證據,**不是**「不連通」。
幾何驗證驗的是共視 + 幾何一致,不是可通行。

### per-pair JSON schema

`schema_version` 欄位仍會寫入,供 `batch_pose_from_cache.py --resume` 判斷既有 JSON
是否需要重算,不必當成對外的版本協商機制。

```jsonc
{
  "schema_version": "wp1-step3-v2",
  "house": "DollhouseTask_58472_NoOutdoor",
  "img_a": "<uuid>", "img_b": "<uuid>",        // cache 檔名的 A/B 順序
  "img_size_a": [8192, 4096], "img_size_b": [8192, 4096],   // 原圖 W,H(per-house 不同!)
  "params": {"conf_min": 0.65, "retry_conf": 0.3, "max_pts": 4000, "thresh_deg": 0.5,
             "ransac_iters": 2000, "seed": 0, "sampler": "stratified",
             "lat_max": 60.0},                  // nadir 前濾角(v2 起;0/缺 = 未濾)
  "n": {"raw": 8146, "conf": 2134, "used": 2134, "inlier": 1474,
        "lat_drop": 312},                       // ⚠ n.conf 記的是 conf_used 點數（非固定 conf_min 級）；前濾丟點數(v2 起)
  "conf_used": 0.65,                            // 自適應實際門檻:0.65 / 0.3 / null(無位姿)
  "inlier_rate": 0.6907,                        // n.inlier / n.used;無位姿時 pose=null
  "pose": {                                     // null = low_matches 或內點 <8
    "R": [[...3x3...]],                         // bearing 約定:x_b 幀 = R·x_a 幀 + t(尺度自由)
    "t": [x, y, z],                             // 單位向量;= A 相機在 B 幀的方向
    "cam_b_in_a": [x, y, z],                    // -R^T t = B 相機在 A 幀的方向
    "az_ab_deg": -139.1,                        // A 幀中望向 B 的方位角(ERP 經度,0=圖中央,±180=接縫)
    "az_ba_deg": 40.2,                          // B 幀中望向 A 的方位角
    "rot_angle_deg": 12.3, "baseline_txz": 0.998
  },
  "quality": {
    "sampson_med_deg": 0.11, "sampson_p90_deg": 0.32,   // 內點對極(重投影)殘差統計
    "rot_only_med_deg": 20.6,                   // 內點被純旋轉解釋的中位殘差(近零基線偵測)
    "az_wedge_a_deg": 109.3, "az_wedge_b_deg": 95.0,    // 包住 80% 內點方位的最窄圓弧
    "az_cov_a": 0.42, "az_cov_b": 0.39          // 方位覆蓋度:有內點的 10° bin 佔比
  },
  "flags": {
    "low_matches": false,          // conf_min(0.65) 級點數 < 30(品質註記；撈回級會出現 low_matches=true 但 n.conf>=30，非矛盾)
    "no_covisibility": false,      // 兩級 conf 都湊不出 ≥8 內點(=無共視證據,非「不連通」)
    "degenerate_rot_only": false,  // rot_only_med < 1.5×thresh_deg → t 方向不可靠
    "doorway_wedge": false         // 非退化 且 inlier_rate≥0.3 且 n_inlier≥30 且兩側楔形≤75°
  },
  "regions": [                     // 共視區域:內點 6D 聯合 DBSCAN 聚類,依 n 降冪
    {"n": 1347, "frac_of_inliers": 0.965, "mean_conf": 0.872,
     "a": {"polygon": [[x,y], ...],   // A 側凸包頂點,原圖像素;過縫 pair x 可能超出 [0,W)
                                       // 以保多邊形連續,繪製時自行 mod W
           "area_frac": 0.2148,        // 多邊形面積 / (W*H)(ERP 像素面積,極區有失真)
           "az_range": [-81.7, 141.1], "az_width_deg": 222.8},   // 方位範圍(全跨度)
     "b": { /* B 側同構 */ }}
  ],                               // 聚類參數:eps=兩側各約12°(鏈接距離)、min_pts=max(8, 2%);
                                   // 門 pair 預期單一門洞區,同房/大環預期大覆蓋或多區。
                                   // ⚠ 區域「代表方位」請用 pose.az_ab,勿用區域中心(已測負結果)
  "gt": {"label": 1, "same_room": true, "az_ab_deg": -132.19, "az_ba_deg": 43.5},
                                   // label:HOTSPOT.json 的 ToIDName 推導;same_room:同檔的
                                   // (floor,Roomidx) 相同;az:GT hotspot 方位(直接相鄰才有)。
                                   // ⚠ hotspot 語意 = 「通道位置」,只有同房 pair 才近似指向對方
                                   // 相機;但直接相鄰 pair 的 B 通常就在通道正後方,兩語意方位
                                   // 高度重合,故 err_az_*
                                   // 當一致性檢核,非嚴格位姿精度。僅供驗收,非計算輸入
  "inliers": [[x_a, y_a, x_b, y_b, conf], ...], // 原圖像素,0.1px 取整,最多 2000 點(seed 0 抽樣)
  "time_ms": 833.7
}
```

`_summary.csv` 欄位 = 上述指標攤平(`house,img_a,img_b,n_*,inlier_rate,conf_used,az_*,sampson_*,rot_only_med_deg,az_wedge_*,az_cov_*,flag_*,n_regions,region1_frac,region1_az_width_a,gt_label,same_room,gt_az_*,err_az_*,time_ms`);
`err_az_ab` = 位姿方位與 GT hotspot 方位的圓角誤差;各組的實際數值見 `batch_pose_report.md`。

⚠ **判斷 pair 好壞請綜合 inlier 率、旗標與 az 一致性,不要單看 inlier 率。**
LO refit 會讓假配對的虛假模型也多納入一些門檻邊緣點,壓縮真假之間的 inlier 率差距
(仍清楚分離,但比未 refit 時小)。

## overlay 渲染器(`render_region_overlay.py`)

```bash
# 單對 on-demand
venv/Scripts/python.exe pose_regions/render_region_overlay.py --json Output/roma_regions/XXX.json
# 驗收抽樣批次:6 類(doorway/同房/跨房/0.3撈回/多區/過縫)各 N 張,決定性選擇
venv/Scripts/python.exe pose_regions/render_region_overlay.py --sample 3
```

輸出 `Output/roma_regions/overlays/*.jpg`(gitignore;A 上 B 下堆疊、region 兩側同色
半透明 + 邊框 + 編號、az_ab/az_ba 黃線、GT hotspot 青色虛線、資訊列)。
**overlay 不全量產出**,只做抽樣或 on-demand：全量產出的磁碟用量不合理。

**過縫繪製約定**(已在本渲染器驗證):polygon x 可能超出 [0,W) 以保連續;
繪製時同一多邊形平移 −W / 0 / +W 各畫一次,超界自然裁掉即可。

**⚠ 主區重心落在天底帶(仰角 < −60°,腳架與地面盲區)者不可直接採信**：該帶在各張
全景圖上外觀相同,是系統性假匹配熱點。`--lat_filter`(預設 60,見上方章節)已在 RANSAC
前濾掉這些點;渲染器與 `/api/regions` 另會對殘餘者標示警示。

## 共視訊號當排序分數:判別力與上限(`covis_ranking_eval.py`,2026-07-26)

```bash
python pose_regions/covis_ranking_eval.py          # 約 1 分鐘,純 CPU
```

回答「幾何驗證後的共視證據能不能把真正連通的 pair 排到前面」。
族群 = `_summary.csv` 的全部列(粗篩存活者),GT 用 `datasets/*_HOTSPOT.json` 完整鄰接重建
(不用 `metadata/*.csv`,那份只覆蓋 68%),另算 BFS topology distance。輸出
`covis_ranking_report.md` + `Output/roma_regions/_ranking_eval.csv`(每 pair 一列,
含 coarse logit：矩陣本身在 gitignore 內,這份 CSV 是唯一入庫的分數快照)。

結論摘要(數字與完整表格見 `covis_ranking_report.md`):

- 幾何 `n_inlier` 單獨的判別力高於未經驗證的 RoMa 匹配數;與 coarse 分數組合後再有
  提升,且免訓練的 rank 平均幾乎追平學習式組合,因此增益不是擬合假象。
- **上限沒被打破**:增益幾乎都來自距離較遠的負例。在「看得到、走不到」的相鄰房間
  這道牆上,幾何與 coarse 量的是同一件事,殘餘誤差被濃縮到牆上而不是穿過它。
- **覆蓋率是硬限制**:部分正例拿不到 inlier 或位姿,因此幾何分數
 **只能當特徵,不能當獨立過濾器**。
- ⚠ 此族群本來就由 coarse filter 挑出,coarse 的 AP 被截斷,不可與全矩陣評測相比,
  也不可讀成「幾何贏過模型」。

## 已知限制

- **沒有跨 pair 共用的尺度**:每對 `|t| = 1`,pair 之間的平移長度不可互相比較。
  真實尺度依分工由子計畫三提供。
- **per-house 圖尺寸不同**(7 種,5376×2688 ~ 8192×4096),任何 x/W 換算都必須用
  該圖自己的寬。
- 效能:live 重跑 3~10 s/對(RTX 4060);走 cache 的批次路徑為純 CPU,毫秒級/對。
