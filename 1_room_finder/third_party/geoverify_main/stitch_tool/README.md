# 拼接標註工具 (Stitch Labeler)

把 block_match_v2 的「連通性矩陣」變成可人工複核的 3D 全景導航工具。
站在一張全景圖裡，另一張圖的方向用圈圈標出；點圈圈就穿門走過去，完成的連接變色但不消失，
路線像 Google Map 一路成形。模型在 Recall90 下 precision 僅 ~18%，靠人直接定性把誤報刷掉。

## 快速開始

工具載入「房子產出包（bundle）」，不再回頭讀完整資料集樹；使用時手動指定要看哪些房子。

```bash
# 1) 產生每間房子的資料檔 data/*.json
python stitch_tool/prepare_house_data.py

# 2) 打包成產出包（--houses 可只挑幾間）
python stitch_tool/pack_bundle.py --out bundles/demo

# 3) 啟動伺服器
python stitch_tool/server.py --bundle bundles/demo    # http://localhost:8010
```

瀏覽器開 http://localhost:8010 ，左上下拉選房子。

### 直接載入對外交付檔

包內的房子資料也可以換成 `export_public.py` 產出的**對外交付檔**。下游實際收到的那一份：

```bash
python export_public.py                                       # 產生交付檔
python stitch_tool/pack_bundle.py --from_public --out bundles/public_demo
python stitch_tool/server.py --bundle bundles/public_demo
```

包內只有交付檔與影像，沒有真值也沒有門語意。工具能不能把它渲染出來，即為「交付內容是否
足夠」的直接檢驗：連接圖、通道方位、幾何證據強度、共視區域都畫得出來就代表下游拿得到的
資訊已經夠用，缺什麼會當場看得出來。介面上會顯示「無 GT 真值」，因為交付檔本來就不含真值。

## 操作
- 左鍵拖曳旋轉視角、滾輪縮放。
- 佇列依連通機率由高到低。目前候選會把你放進 **i 圖**，並在預測門位放 **琥珀色圈**、自動轉過去面對它。
- **Y 連通**：圈變 **綠**（永久保留）。點綠圈即可**穿門**跳到對方圖，對方圖的反向門顯示為 **藍圈**，可再跳回。
- **N 否決（批量否決流程，新）**：按 N 進入否決模式並自動切入 ERP。點任一圈圈後，畫面右側浮出
  「**同方向候選**」清單。把該圈方位所在聚類(±7°)裡所有待處理配對列出（每列 `i–j · prob% · az°`，
  預設全勾），頂部可「全選／全不選」，底部按「**否決所選(N筆)**」一次否決（單次存檔、單次重繪、一則
  總數提示）。面板關閉後仍留在否決模式，可繼續點下一個圈。只有一筆時也走同一面板（一致）。
- **拖曳圈圈**：把門位校正到最準的方位（水平＋垂直），確認時存校正後角度。
- **U 未連接通道（放圈推薦流程，新）**：按 U 進入未連接通道模式並**自動切入 ERP 對比視圖**。在某面板放一個
  **橘圈**後，右側浮出**推薦面板**：把該圖「看向對方」的方位與圈方位差 <25° 的待處理配對依 prob 降序列成
  「候選 k/N：img X · prob% · Δaz°」，並自動把**對面面板**切到候選那張圖、在對面對應方位畫一個**琥珀虛線預覽圈**
  幫你直接看圖判斷。用「上一個／下一個」瀏覽（對面面板連動換圖與預覽）；按「**就是這個(連通)**」即建立綠色
  確認邊（此側用你放的橘圈方位、對側優先用你放的橘圈否則用候選對面方位），按「**否決此候選**」則否決並自動跳
  下一個。看完 / 0 筆時提示「無更多推薦，可手動換對面圖」。回到**手動雙圈**流程：上下兩面板各放一個橘圈對準
  通道，按動作列「**連接**」鈕建立綠色確認邊。**孤立照片**（模型完全沒給任何 pair 的圖）也能這樣人工連起來。
  右鍵點橘圈可刪除；**退出模式時未被連接的橘圈會全部清除**、推薦面板一併關閉。（模式期間「回 3D」兩顆鈕暫時隱藏，退出後恢復。）
- **A 此圖全連接**：把目前圖所有剩餘候選一次淘汰。
- 確認某門後，其他「同一圖、同聚類(±7°)」的候選會自動淘汰；但若附近有你放的橘圈則保留。
- 已**否決/淘汰**的配對從佇列點回去時，3D 與 ERP 都會畫一個**紅色**圈（與橘色「未連接」語意分開）；按 Y 可重新確認連通。
- **右下路線圖（力導向佈局，新）**：節點=圖、綠邊=已確認。佈局改用**力導向**。已確認邊當彈簧拉近相連圖、
  節點間庫倫斥力避免重疊、**孤立圖被推到外圈**（一眼看出誰還沒連，正是收尾工作流的重點）。著色：有確認邊=綠、
  孤立=灰、目前圖高亮（3D=琥珀；ERP 上圖=藍、下圖=橘）。佈局**確定性**（圓形初始＋固定迭代，無隨機），
  且僅在確認邊變動時重算、換圖時只重繪，不會亂跳。**點節點直接跳到該圖**（3D 模式跳單一檢視器；ERP 模式顯示
  **上下兩個路線圖**，各自點節點換該面板的圖）。「▣ 路線圖」鈕可收合/展開（狀態隨存檔記憶）。
- **路線圖放大彈窗（新）**：路線圖標題列的「**⤢ 放大**」或**雙擊小圖**，開一個約 70% 螢幕的 modal：放大力導向
  圖、每個節點標 img 編號、hover 顯檔名、**點節點跳圖**（3D 模式跳 3D；ERP 模式**左鍵=設為上/左圖、右鍵=設為
  下/右圖**，modal 內有提示）。底部一排**未連接圖片快速入口**（孤立節點做成 chip，點擊直接前往）方便收尾孤立照片。
  Esc 或點背景關閉。
- **ERP 回 3D**：ERP 模式中，動作列的單顆「ERP 對比」鈕換成**兩顆**。「回上圖 3D／回下圖 3D」
  （左右排列時為「回左圖／回右圖」），點了就離開 ERP 回到該面板那張圖的 3D 檢視。E 鍵仍可切換 ERP。
- 進度自動存到伺服器 `state/<房子>.json`，換房子/重開會自動載回。快捷鍵 Y/N/U/R/A/E、Esc 前往下一對。

## 門顯示層 (v2)：大房間/戶外不用自己數門
先把模型已知的門方位畫出來，你只補模型漏掉的。左側「門顯示層」三個一鍵開關（狀態隨進度存讀）：
- **配對預測門**：把所有含目前圖的候選依方位做緊聚類（±7°）成「門」，一個門一個淡琥珀空心圈；
  低分雜訊用機率門檻藏掉（例：某圖 22 個預測→聚成 3 個門）。點淡圈可跳去複核那一對。
- **羅盤條**：底部 360° 尺，每個門一刻度（綠=已連/琥珀=預測未連/橘=手動），
  高亮目前視野、點刻度轉過去；標題顯示「偵測到 N 個門（M 已連）」。
- **場景參考圈**：在 3D 全景裡直接畫出上述門圈。
- **淘汰收緊**：確認某門後只淘汰「同一聚類(±7°)」的冗餘候選；不同門永不互相淘汰
  （未確認門聚類自動當保護槽，取代舊的 ±20° 規則，根治近門誤殺）。兩門距離<7° 想分開時放一個手動橘圈即可。
- **UI 狀態**：`STATE.ui` 除三個門顯示層開關外，另有 `minimap`（路線圖顯示/隱藏，預設 true）。
  舊存檔缺此欄位時 `normalizeState` 自動補預設，向後相容；state schema 其餘不變。

## 幾何/資料原理
- 連通機率：`config.matrices_root('refined')` 解析到的 `<house>.npy` 的 ch0，`prob=sigmoid(ch0)`。
  ch0 是有向的，此處取上三角 `M[i,j]`；格式契約見 `matrices/README.md`。
- 門方位（粗篩方向頭，coarse）：同矩陣 ch1/ch2 = `(sin, cos)`，`az(i→j)=atan2(ch1,ch2)`
  （已對 localization CSV 驗證一致）。
- **通道方位來源（2026-08-05 起）**：每 pair 的 `az_ij/az_ji` 優先採用 RoMa 幾何驗證的
  `az_ab_deg/az_ba_deg`（join `Output/roma_regions/_summary.csv`，該對有位姿才替換、mod 360
  換算；方向對映依檔名判斷 summary 的 img_a/img_b 順序是否與 (i,j) 相同）。連通性粗篩模型
  不輸出通道方位（只有連通 logit＋通道所在視角影像），未來整合後方位一律由幾何驗證提供，故
  改吃 az_ab/az_ba 取代粗篩方向頭。沒有位姿（no_covisibility／未跑 RoMa）的 pair 保留 coarse
  方位。每 pair 新增 `az_source`（`"geo"` 或 `"coarse"`）與備份欄位
  `az_coarse_ij/az_coarse_ji`（QA 對照用，值恆等於舊版 coarse 輸出）；`gt_az_ij/gt_az_ji` 不受影響。
  取代的依據是同批 pair 的逐向配對比較，幾何方位在中位與 p90 都明顯優於粗篩方向頭。
  前端零改動即透明生效（ERP 門圈／走一遍／裁切對／
  頂帶皆直接讀 az_ij/az_ji）；人工 ROI 校正、已確認邊的鎖定方位優先權不變。
- ERP↔方位：`az(x)=((x/W)*360-180)%360`；反解貼圖水平比例 `u=((az+180)%360)/360`。
  已用實圖驗證圈圈會落在真正的門/通道上（非鏡像）。

## 佇列排序（機率 / 幾何驗證）
左側佇列排序下拉有兩個模式：
- **機率**（預設）：伺服器已依連通機率排好序，前端不重排。
- **幾何驗證**（2026-08-05 起，取代原「RoMa 分數」）：依 `geo_n_inlier`（幾何驗證內點數）
  降冪排序、次鍵為機率；門檻拉條比對 `geo_inlier_rate`（0~1，天然吻合拉條刻度）。
  取幾何驗證內點數，判別力比較見 `pose_regions/covis_ranking_report.md`。
- **資料源**：`prepare_house_data.py` join `Output/roma_regions/_ranking_eval.csv`（已入 git），
  附加 `geo_n_inlier`／`geo_inlier_rate`／`geo_n_regions`／`geo_conf_used`／`geo_posed`。
  join key 為 (house, {img_a,img_b} 去副檔名集合)，pair 無序比對。**三層語意**：
  `geo_posed=null`＝此對不在 CSV（未過粗篩或沒跑 RoMa，佇列沉底）；
  `geo_posed=0`＝跑了但幾何驗證無法定位姿（`geo_n_inlier=0`，自然墊底，但仍在「有值」組內，
  排在 null 組之上）；`geo_posed=1`＝正常值。
- 舊 `roma_match_count`／`roma_conf_sum`／`roma_conf_mean`（`roma_scores.npz` 聚合，未經幾何驗證）
  欄位與載入邏輯保留不動，供向後相容／除錯對照；`STATE.romaThreshold` 欄位名沿用（語意已從
  RoMa 平均信心改為幾何驗證 inlier_rate 門檻，避免舊 state 檔不相容）。
- **重生資料**：`venv/Scripts/python.exe stitch_tool/prepare_house_data.py`（全房子；
  `_ranking_eval.csv`／`_summary.csv` 不存在時對應欄位全 null／保留 coarse 方位，不會炸）。
- **RoMa 即時比對（2026-08-05 停用）**：ERP 內原本可對目前兩張圖跑 RoMa 即時比對畫匹配線的
  「RoMa 比對」按鈕與信心輸入已從 UI 移除（共識區域取代 raw RoMa 匹配顯示）。
  `js/roma.js` 與後端 `/api/roma_match`、離線 cache 保留不動（`main.js` 的 `initRoma()` 呼叫
  已註解，零檔案刪除）。

## 檔案結構
- `prepare_house_data.py`：由 oof 矩陣＋HOTSPOT 產生 `data/<house>.json`（機率排序的 pairs、
  az_ij/az_ji［預設 join 幾何驗證 az_ab/az_ba，見上方「幾何/資料原理」］、`geo_*` 幾何驗證排序
  欄位［見上方「佇列排序」］、GT對照）。
- `server.py`：本機伺服器（房子清單／房子資料／ERP 影像／進度存讀）。
- `index.html`：three.js 前端外殼（DOM 與樣式）；以 `<script type="module" src="js/main.js">` 載入模組化前端，three.js r128 仍由 CDN classic script 提供全域 `THREE`。
- `SPEC.md`：前端規格。
- `data/`：房子資料檔。`state/`：人工標註進度。

### 前端模組 `js/`（ES modules，事件驅動）
單體 `app.js` 已拆為下列模組。低層（config/events/geometry/clustering/state/api/util）不依賴高層；跨層互動一律透過事件匯流排或由 `main.js` 組裝，避免循環依賴。
- `config.js`：常數（球體/標記尺寸、顏色、聚類角度 CLUSTER_DELTA、UI 預設）。
- `events.js`：極簡 pub/sub 事件匯流排（`on`/`emit`）。事件：`house-loaded`／`state-changed`／`view-changed`／`frame`／`resize`／`rebuild-markers`／`reject-confirmed`／`reset-filter`／`apply-ui-toggles`／`erp-reset-az`／`erp-derived`（ERP 面板衍生完成→刷新動作可用性與狀態列）／`apply-minimap-vis`（ERP 開關時切換單/雙路線圖）。動作只 emit，UI 元件訂閱重繪。
- `geometry.js`：方位↔球面幾何純函式（`azToPos`/`posToAz`/`azToLon`/`angDiff`，逐字照抄 SPEC）。
- `clustering.js`：門聚類純函式（`doorCandidates`/`circMean`/`clusterDoors`），3D 與 ERP 共用。
- `layout.js`：路線圖**力導向佈局**純函式（`forceLayout(N,edges,W,H,iter,init)→positions`、`isIsolated`）。確定性、不觸碰 DOM，小圖／彈窗／ERP 雙圖共用同一份 cache（`minimap.js` 內正規化後投影到各 canvas）。可 node 測試。
- `recommend.js`：未連接通道**放圈推薦**純函式（`recommendForCircle(imgIdx,az,hd,STATE,statusFn,windowDeg)→依 prob 降序候選`、`RECO_WINDOW=25`）。只依角度窗＋prob，排除非 pending。可 node 測試。
- `state.js`：標註狀態 `STATE`、house 資料、狀態判定（`pairKey`/`pairStatus`…）、debounce 存檔。schema 與 server API 不變。
- `api.js`：伺服器 API 封裝（`fetchHouses`/`fetchHouseData`/`fetchState`/`saveState`）。
- `nav.js`：導覽狀態（`nav.curImage`/`curPair`/`viewFrom`/`reciprocalKey`/`erpTopImg`/`erpBotImg`）與導覽入口（`gotoCurrentPair`/`jumpToPair`/`peekToPair`/`walkThroughDoor`…）。
- `scene3d.js`：three.js 場景/相機/raycast/pointer 事件/材質快取/標記建立/`buildMarkersForImage`/`buildDoorLayer`/`loadImageView`。
- `modes.js`：互斥模式狀態機 `mode:'none'|'unconn'|'reject'|'noRoi'|'allConn'` + 模式旗標 UI + `erpAutoEntered` 自動進/出 ERP 邏輯。
- `dispatch.js`：唯一圈圈點擊分派入口 `handleCircleClick(targetInfo)`，3D 與 ERP 共用。否決模式點圈→開批量否決面板；未連接模式與導覽照舊。
- `actions.js`：標註動作（Y/N/A、否決、無 ROI 淘汰、全連接、重新否決、reset 房屋）。含批量否決 helper `rejectPairs(list)`（單次存檔＋單次重繪＋總數提示，含把已確認者改否決）與推薦連通 helper `confirmRecommendedCandidate(...)`。單發 `rejectMarkerPair` 保留為內部 helper。
- `erp.js`：ERP 對比視圖全部（上下/左右並排、標門、拖曳校正、換圖）＋**未連接推薦面板**（`openRecoPanel`/`recommendForCircle` 呈現、對面預覽圈、連通/否決/手動後備、`initRecoPanel` 綁定）。
  **本對主標記**：每張 ERP 圖 **頂帶＋底帶**（DOM，與 marker.js 同視覺：白＋深底、
  ±15°、「往 img N」）；**僅水平拖曳**改 az。hover／拖曳時該圖內 **垂直虛線**連頂↔底
  （非跨圖）。**區域遮罩**改為 ERP 內開關（`btn-erp-regions`），canvas 疊在目前對比圖上，
  不再是獨立 viewmode（`mode-regions.js` 檔保留、index 已停用註冊）。
- `roma.js`：RoMa 即時比對（點按鈕對目前 ERP 兩張圖跑 RoMa、畫匹配線 overlay）。
  **2026-08-05 停用**（共識區域取代 raw RoMa 顯示；`index.html` 的按鈕／
  `roma-conf-input`、`main.js` 的 `initRoma()` 呼叫已移除/註解，檔案與後端 `/api/roma_match`、
  離線 cache 保留不動，見「佇列排序」段落）。
- `batch-panel.js`：**批量否決浮動面板**：`collectClusterPending(imgIdx,az)` 取同聚類全部 pending、`openBatchReject`/`initBatchPanel` 呈現與綁定，勾選後走 `actions.rejectPairs`。
- `minimap.js`／`compass.js`／`queue.js`：路線圖（力導向＋放大彈窗，`openMapModal`）／羅盤條／候選佇列與統計（皆訂閱事件自動重繪）。
- `info.js`：中央資訊面板。`ui.js`：按鈕綁定與快捷鍵，含集中式 `refreshActionAvailability()`（依模式/ERP/衍生配對統一設定 Y/N 停用與原因，訂閱 `view-changed`/`state-changed`/`erp-derived`）。`util.js`：toast/loading/小工具。
- `main.js`：組裝與啟動（boot）：初始化場景與訂閱者（含 `initBatchPanel`／`initRecoPanel` 綁定兩個浮動面板按鈕）、綁定 UI、載入房子。

換模型：`prepare_house_data.py --oof_dir .../oof_matrices/<其他exp>` 即可指向別的實驗輸出。

## 檢視模式插件系統

「檢視模式」＝主畫面呈現方式，與標註互動模式（modes.js）互相獨立。架構：

- `js/viewmodes/registry.js`：註冊表：`registerViewMode(plugin)`／`setViewMode(id)`／互斥切換；
  `ownButton:true` 的插件按鈕由 registry 自動生成在 `#viewmode-btns`。
  互斥規則：進入任一模式先退出其它；**ERP 被外部開啟（標註模式自動帶入/舊按鈕/E 鍵）時，
  其它 overlay 模式自動讓路**：既有標註流程完全不變。
- `js/viewmodes/index.js`：**唯一的增/停用修改點：一行 import＝一個模式**；
  註解該行＝停用該模式（按鈕一併消失），其餘模式不受影響。
- 插件檔（每模式一檔）：
  - `mode-3d.js`／`mode-erp.js`：既有 3D 球面與 2D ERP 包裝（委派 `toggleErp`，行為不變、沿用舊按鈕）。
  - `mode-crop.js`：**模式 A 透視裁切對**：沿 az_ij/az_ji 做 gnomonic 90° 投影，兩張 pinhole 並排
    （從 A 看向 B／從 B 看向 A）＋中間連接帶。解決「門後有門」ERP 誤讀。
  - `mode-morph.js`：**模式 C 動態 morph**：`/api/roma_match`（conf 0.3，命中離線 cache 秒回）取稀疏
    匹配 → 投到兩張裁切圖 → IDW 網格位移場 → 三角網格 warp ＋ 交叉淡化來回播放，右下顯示即時 FPS
    （實測 draw 成本 1.85ms/幀 ≈ 540fps 裕度，vsync 鎖 60）。真連通像「走過去」，假連通扭曲穿幫；
    標籤同時顯示 GT 連通與否，方便真/假對照示範。
- `mode-walk.js`：**走一遍**：時間軸
  `holdDoor(朝通道) → turnA(轉主區) → holdStart → fade(300ms 直跳) → holdIn`；
  固定 FOV 48；`END_WITH_LOOKBACK=false` 常數可恢復舊回看（非 UI）。
- 共用 helper（非插件）：`gnomonic.js`（ERP⇄透視投影、ERP ImageData 快取、圓形平均）、
  `pairsel.js`（目前配對推導，與 renderERP 同款邏輯）、`marker.js`（通道頂帶 canvas 繪製）。
  插件之間不得互相 import。

新增一個模式：寫 `js/viewmodes/mode-xxx.js`（實作 `{id,label,ownButton,isActive,enter,exit,refresh}`
並自行 `registerViewMode`），然後在 `js/viewmodes/index.js` 加一行 import。

### 共視區域遮罩（ERP 疊層，非獨立模式）

- **入口**：動作列「區域遮罩」鈕（`btn-erp-regions`）。開時若未在 ERP 會自動切入 ERP，
  把 多邊形 **疊在目前上下/左右兩張 ERP 圖上**（同色＝同一塊實體空間＋R 編號）。
  再按一次關閉。不再跳到另一個全螢幕 viewmode。
- 實作在 `erp.js`（`toggleErpRegions`／`.erp-reg-cv`）；`mode-regions.js` 檔保留備查、
  `viewmodes/index.js` 已註解停用註冊。
- `GET /api/regions/<house>?a=&b=`：讀 `Output/roma_regions/…`，`swap:true` 時對調 a/b。
- 過縫：polygon x 可超出 [0,W)，繪製時 −W/0/+W 各畫一次。
- server.py 對 js/html/css 送 `Cache-Control: no-store`。
