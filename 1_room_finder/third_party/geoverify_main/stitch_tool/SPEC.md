# 拼接標註工具 前端規格 (index.html)

單一自足檔案 `stitch_tool/index.html`，用 three.js r128
(`https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js`)。
不得使用 localStorage/sessionStorage 以外的持久化（進度一律存伺服器）。

## 目標
人工複核「哪些環(全景圖)真實連通」。模型在 Recall90 門檻下 precision 僅 ~18%，誤報多；
人能直接定性。工具讓使用者「站在一張全景圖裡」，把另一張圖的方向用圈圈標出，點圈圈即可
穿過門走到那張圖，完成的連接變色但不消失，路線像 Google Map 一路成形。

## 伺服器 API（server.py 已完成）
- `GET /api/houses` → `[{house,n_images,n_pairs,n_gt_connected_pairs}]`
- `GET /api/house/<house>` →
  `{house,n_images,width,height,images[N],pairs[],az_convention}`
  - `pairs[k]={i,j,logit,prob,az_ij,az_ji,gt_connected,gt_az_ij,gt_az_ji}`，已依 `prob` 由高到低排序。
    角度單位為度 [0,360)。`az_ij` = 在 **i 圖** 中看向 j 的門方位；`az_ji` = 在 j 圖看向 i。
    `gt_*` 僅供除錯對照（可用開關顯示），非必要。
- `GET /images/<house>/<file>` → ERP 全景 jpg（等距柱狀 equirectangular）。
- `GET /api/state/<house>` → 已存標註（或 `{}`）。
- `POST /api/state/<house>`（JSON body）→ 存標註。

## 方位 ↔ 球面幾何（務必照抄）
ERP 規約：`u = ((az + 180) mod 360) / 360`，u∈[0,1] 為全景水平比例(0=左緣,1=右緣)；
v∈[0,1] 垂直(0=上,1=下)。門標記用 v=0.5（水平線）。

球面用 `THREE.SphereGeometry(500,60,40)` 且 `geometry.scale(-1,1,1)`（由內往外看）。
貼圖點 (u,v) 對應的 3D 座標（把標記放半徑 R=480，略在球內才看得到）：
```
X = R * Math.cos(2*Math.PI*u) * Math.sin(Math.PI*v)
Y = R * Math.cos(Math.PI*v)
Z = R * Math.sin(2*Math.PI*u) * Math.sin(Math.PI*v)
```
此式正好落在該像素上，與相機規約無關。要「面向某門方位 az」時（沿用 panorama_viewer 的
lon/lat 相機）：`lon = (az + 180) mod 360`，`lat = 0`。點擊球面回推方位：對命中點 P，
`u = (atan2(P.z, P.x) / (2π)) mod 1`，`az = (u*360 - 180) mod 360`。

相機互動沿用 panorama_viewer：左鍵拖曳改 lon/lat、滾輪改 fov(10~100)。
（可直接參考 `../panorama_viewer/index.html` 的 init/onPointer*/update。）

## 版面
- 左側欄：房子下拉選單；候選佇列（顯示 `i–j prob% 狀態點`，依機率排序，可捲動；點一列可跳到該對）；
  統計（已確認/已否決/已淘汰/剩餘）；機率門檻滑桿（預設 0.5，僅影響佇列顯示與自動淘汰參考）；
  GT 對照開關；圖例。
- 中央：three.js 全景檢視器 + 資訊面板（目前圖名、目前 pair、prob）。
- 底部動作列 + 快捷鍵。

## 核心流程
1. 選房子 → 讀 house data 與 state。佇列 = pairs 中「未否決/未淘汰」者，機率高→低。
2. 目前 pair = 佇列第一個未處理者。進入其視角：載入 **i 圖** 全景，於 `az_ij` 放一個
   **候選色(琥珀)** 圈圈；相機自動轉到面向它，使用者立刻看到預測門位。
3. 動作（按鈕＋鍵盤）：
   - **是/連通 (Y)**：確認。圈圈轉 **確認色(綠)** 並永久保留。記錄 i 圖在(可能被拖曳校正後的)
     方位往 j 的通道，及 j 圖 `az_ji` 的互為反向通道。之後可點該圈跳到 j 圖。
   - **否 (N)**：否決，前進下一候選。
   - **拖曳校正**：候選/確認圈圈可被拖動（至少水平、最好含垂直）到真正門位，方位即時更新；
     確認時存校正後方位。
   - **點圈跳視角**：點「確認色」圈圈 → 載入對方圖全景並面向反向圈圈（走過門）。對方圖同樣有
     可拖曳/確認的反向圈圈，可再跳回來。
   - **未連接通道 (U)**：切換此模式後，點球面在目前圖放一個 **橘色** 圈圈＝「我看到這裡有門但還沒連」。
     依圖持久保存。用來避免「兩門很近就被誤淘汰」。
   - **此圖通道全連接 (A)**：把目前圖標記完成 → 佇列中所有含此圖的候選一次淘汰。
   - **自動淘汰＋安全弁**：確認 i 圖某方位 az* 的通道後，其他含 i 且 `|az_ik - az*|<Δ(預設20°)`
     的待處理候選自動淘汰；但若該近似方位處有使用者放的「未連接通道」橘圈，則保留不淘汰。
4. 路線 / 進度：右下小地圖(2D canvas)畫節點(圖)與已確認邊(綠)，隨確認逐步成形（環狀或簡單力導向佈局即可）。
5. 每次變更 debounce 後 `POST /api/state/<house>`；換房子時載回。

### state schema（存伺服器）
```json
{
  "confirmed":  [{"i":20,"j":22,"az_i":129.3,"az_j":89.0}],
  "rejected":   [[21,22]],
  "eliminated": [[20,21]],
  "extra_channels": {"20":[41.0,300.0]},
  "complete_images": [20],
  "threshold": 0.5
}
```

## 顏色圖例
候選=琥珀 `#f4b400`；確認=綠 `#12b886`；反向目標=藍 `#4c8dff`；未連接(使用者)=橘 `#ff7a45`；
已淘汰=灰/隱藏。

## 交付優先序
先做到「可實跑的核心」：檢視器＋佇列＋是/否＋圈圈定位＋點圈跳圖＋存讀。再做：拖曳校正、
未連接通道、全連接淘汰＋安全弁、路線小地圖、GT 對照、快捷鍵與打磨。
測試房子：`DollhouseTask_58472_NoOutdoor`（23 張，資料已生成）。

## 增強：門聚類層 + 羅盤條 + 場景參考圈（v2）
問題：大房間/戶外難以人工數出有幾個 ROI。作法：先把模型已知的門方位畫出來，使用者只補漏掉的。
避免滿螢幕圈圈＝聚類 + 門檻過濾；避免誤合併＝聚類只用於顯示/計數，淘汰只在同聚類內。

- **門候選（current image 視角）** 兩個來源，各一鍵開關（狀態存 `STATE.ui`）：
  - 配對聚類：所有含 imgIdx 的 pair，取在 imgIdx 看向對方的方位（i==imgIdx→az_ij，否則 az_ji），
    帶 prob/target/status。用**環狀單鏈聚類、CLUSTER_DELTA≈7°**併成門；cluster.bestProb=成員最大prob，
    status=有已確認成員→confirmed，否則 pending。只顯示 bestProb≥門檻或 confirmed/使用者的。
- **羅盤條**（底部 360° 尺）：每個門一刻度，顏色 綠=已確認/琥珀=預測未確認/橘=手動補；
  目前相機 lon 對應的 FOV 視窗高亮；點刻度→`lon=azToLon(az)` 轉過去。標題顯示「偵測到 N 個門（M 已連）」。
- **場景參考圈**：每個顯示中的門聚類在 repAz(v=0.5) 畫環；目前 pair 的門最醒目（琥珀實心大），
  其餘 pending 淡描邊、confirmed 綠、手動 橘。點 pending 參考圈可跳去複核該 target 的 pair；confirmed 走門。
- **淘汰收緊**：確認 pair 的門 az* 後，只淘汰「落在同一聚類(距 az*<CLUSTER_DELTA)」的其它待定候選；
  不同聚類永不互相淘汰。未確認門聚類**自動當保護槽**（等同舊橘圈作用）。手動橘圈退為「模型完全漏掉」補洞；
  若兩門距離<CLUSTER_DELTA 想分開，放一個手動橘圈即強制切成獨立保護槽。
- 幾何/顏色/存讀沿用 v1；toggles 存入 STATE.ui 並隨 state 存讀。
