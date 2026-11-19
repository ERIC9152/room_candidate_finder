/* =========================================================================
   config.js：全域常數（幾何尺寸、顏色、聚類角度、UI 預設）
   低層模組：不 import 任何其它模組。所有數值逐字沿用原 app.js。
   ========================================================================= */

/** 球體半徑（three.js SphereGeometry） */
export const SPHERE_R = 500;
/** 標記放置半徑（略在球內才看得到） */
export const MARKER_R = 480;
/** 標記 sprite 預設像素尺寸 */
export const MARKER_SIZE = 26;

/** 顏色圖例（沿用 SPEC 與原 app.js）。red = 已否決/淘汰配對的圈。muted = 非候選/停用灰。*/
export const COL = {
  amber:'#f4b400', green:'#12b886', blue:'#4c8dff', orange:'#ff7a45', gt:'#ff00ff',
  refdoor:'#f4b400', red:'#fa5252', muted:'#93a4b8'
};

/** 路線圖節點著色（力導向）。connected=有確認邊、isolated=孤立、hiTop/hiBot=ERP 雙圖各自高亮、hi3d=3D 高亮。*/
export const MAP_COL = {
  connected:'#12b886',      // 有確認邊（綠系）
  isolated:'#61708a',        // 孤立（灰）
  edge:'#12b886',            // 已確認邊線
  hi3d:'#f4b400',            // 3D 目前圖高亮（琥珀）
  hiTop:'#4c8dff',           // ERP 上/左圖高亮（藍）
  hiBot:'#ff7a45',           // ERP 下/右圖高亮（橘）
  label:'#c3d0de'            // 節點編號文字
};

/** deg: 橘圈落在候選方位附近時保護其免於淘汰 */
export const UNCONN_KEEP = 18;
/** deg: 把 extra-channel 對齊到某候選方位 */
export const UNCONN_SNAP = 12;
/** deg: 門聚類 / 同門淘汰的緊角度（分得開的近門不會被併 or 誤殺） */
export const CLUSTER_DELTA = 7;
/** 門顯示層開關預設值。minimap=路線圖顯示/隱藏（預設 true，舊存檔向後相容）。
 * semanticAssist=語意輔助（用子計畫一門語意圖校正通道方位），預設關、舊存檔向後相容。*/
export const UI_DEFAULT = { pairwise:true, compass:true, rings:true, minimap:true, semanticAssist:false };
