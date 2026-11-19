/* =========================================================================
   erp.js：ERP 對比視圖（上下/左右兩張平面全景並排，標門、可拖曳、可換圖）
   結構解耦：ERP 與 nav.curPair 徹底解耦。
     - nav.erpTopImg/erpBotImg 為 ERP 顯示唯一真相，永遠有效（pairing.ensureErpPanes 保證）。
     - 「目前配對」為衍生值（pairing.derivePairForPanes）：
         kind='pair'      -> 主標記可拖曳、Y/N 可用
         kind='confirmed' -> 綠色主標記、Y 停用、N=撤銷確認
         kind='none'      -> 不畫主標記、Y/N 停用（但面板/箭頭/路線圖/放圈全部照常可用）
       衍生結果同步寫回 nav.curPair（找不到 pair 時為 null，UI 不因此癱瘓）。
     - renderERP/setupErpPane/erpDoorList/drawErpMarkers 一律無條件執行（不再檢查 curPair）。
     - 換圖箭頭移出圖面到「面板控制列」(#erp-ctrl-top/bot)；overlay 內不再有 .erp-arrow-btn。

   共享狀態：erpMode / erpAzI / erpAzJ / erpHoriz / _derived（本輪面板衍生結果快取）。
   ERP 圈圈點擊統一走 dispatch.handleCircleClick（source:'erp'）。
   依賴：config / geometry / clustering / state / nav / pairing / dispatch / scene3d / actions / modes / util / events。
   ========================================================================= */

import { COL, CLUSTER_DELTA } from './config.js';
import { angDiff } from './geometry.js';
import { clusterDoors } from './clustering.js';
import {
  STATE, getHouseData, getHouseName, pairKey, pairStatus, scheduleSave
} from './state.js';
import { nav } from './nav.js';
import { ensureErpPanes, derivePairForPanes } from './pairing.js';
import { handleCircleClick } from './dispatch.js';
import { loadImageView, buildMarkersForImage } from './scene3d.js';
import { rejectConfirmed, rejectPairs, confirmRecommendedCandidate } from './actions.js';
import { recommendForCircle } from './recommend.js';
import { getMode, updateModeFlags, clearErpAutoEntered } from './modes.js';
import { toast, round1, statusLabel } from './util.js';
import { emit, on } from './events.js';
import { fetchRegions } from './api.js';
import { REG_PALETTE } from './viewmodes/regions-paint.js';
import { getPairOverride } from './semantic.js';

// ---------- ERP 狀態 ----------
let erpMode = false;
let erpAzI = null, erpAzJ = null;
let erpHoriz = false;                 // 上下 vs 左右排列
const SHOW_ERP_MAIN_LINE = false;     // 設 true 顯示本對的垂直參考線

/**
 * 本輪面板衍生結果快取：{ top, bot, kind, pair }。
 * renderERP 內算一次，drawErpMarkers/erpDoorList 直接讀，避免重覆推導與對 nav.curPair 的散落依賴。
 */
let _derived = { top:-1, bot:-1, kind:'none', pair:null };

/**
 * 目前 pair 的語意校正結果快照（getPairOverride() 回傳值），供 info.js 顯示來源標籤。
 * 每次 renderERP 開頭重設為 null，只有語意輔助開啟且 pair 未確認時才會由 applySemanticOverride 填入
 *：所以永遠對應「目前」pair，不會有切換 pair 後殘留舊資料的問題。
 */
let _semanticInfo = null;
/** @returns {object|null} 目前 pair 的語意校正結果（未開啟輔助 / 已確認 / 無 pair 時為 null）。*/
export function getSemanticInfo(){ return _semanticInfo; }
let _semanticLoggedKey = '';   // 避免同一 pair 重覆 render 時 console 洗版（見 applySemanticOverride）

/**
 * unconn 放圈推薦面板狀態。
 *   active：面板是否啟用
 *   srcImg：放橘圈的圖（此側）
 *   srcWhich：該圖所在面板 'top'|'bot'
 *   az：橘圈方位
 *   list：recommendForCircle 排序候選陣列
 *   idx：目前瀏覽到第幾個候選
 *   previewOther：目前為預覽而在對側面板暫畫的方位（琥珀預覽圈），null=無
 */
let _reco = { active:false, srcImg:-1, srcWhich:'top', az:0, list:[], idx:0, previewOther:null };

let _panePreview = null;   // {which:'top'|'bot', az:number} | null
export function setPanePreview(which, az){ _panePreview = {which, az}; if(isErpOn()) renderERP(); }
export function clearPanePreview(){ if(_panePreview){ _panePreview = null; if(isErpOn()) renderERP(); } }

let _paneFocus = 'top';   // 'top' | 'bot'
export function getPaneFocus(){ return _paneFocus; }
export function setPaneFocus(which){ _paneFocus = (which==='bot') ? 'bot' : 'top'; emit('pane-focus-changed'); refreshFocusUi(); }

export function refreshFocusUi(){
  const topCtrl = document.getElementById('erp-ctrl-top');
  const botCtrl = document.getElementById('erp-ctrl-bot');
  if(topCtrl && botCtrl){
    topCtrl.classList.toggle('focused', _paneFocus === 'top');
    botCtrl.classList.toggle('focused', _paneFocus === 'bot');
  }
  const fTopBtn = document.getElementById('focus-top');
  const fBotBtn = document.getElementById('focus-bot');
  if(fTopBtn && fBotBtn){
    fTopBtn.classList.toggle('on', _paneFocus === 'top');
    fBotBtn.classList.toggle('on', _paneFocus === 'bot');
    if(erpHoriz){
      fTopBtn.textContent = '左圖';
      fBotBtn.textContent = '右圖';
    } else {
      fTopBtn.textContent = '上圖';
      fBotBtn.textContent = '下圖';
    }
  }
}

/** @returns {boolean} ERP 是否開啟 */
export function isErpOn(){ return erpMode; }
/** @returns {number|null} 目前 i 圖拖曳方位（供 actYes 存校正角） */
export function getErpAzI(){ return erpAzI; }
/** @returns {number|null} 目前 j 圖拖曳方位 */
export function getErpAzJ(){ return erpAzJ; }
/** @returns {{top:number,bot:number,kind:string,pair:object|null}} 目前面板衍生結果（供其它模組查詢，如可用性刷新）。*/
export function getErpDerived(){ return _derived; }

// =========================================================================
//  TOGGLE
// =========================================================================
/** 切換 ERP 對比模式。關閉時清除「自動帶入」旗標。*/
export function toggleErp(){
  erpMode = !erpMode;
  if(erpMode){
    _paneFocus = 'top';
  } else {
    _passageHoverWhich = null;
    _passageDragWhich = null;
    if(!_erpRegOn){ /* keep flag; canvases cleared with ERP hide */ }
  }
  const el = document.getElementById('erp-mode');
  if(el) el.style.display = erpMode ? 'flex' : 'none';
  const btn = document.getElementById('btn-erp');
  if(btn) btn.classList.toggle('on', erpMode);   // ERP 中高亮；改用兩顆回 3D 鈕退出，故不再隱藏 E 鈕
  const layoutBtn = document.getElementById('btn-erp-layout');
  if(layoutBtn) layoutBtn.style.display = erpMode ? 'inline-block' : 'none';
  if(!erpMode) clearErpAutoEntered();    // 手動或自動關閉都清旗標
  updateModeFlags();
  updateErpBackButtons();                // 更新兩顆回 3D 鈕與連接鈕
  emit('apply-minimap-vis');             // ERP 開/關時切換單/雙路線圖顯示
  if(erpMode) renderERP();
  emit('view-changed');                  // 刷新動作列可用性（Y/N/箭頭語意等）
}

/** 切換 ERP 上下 / 左右排列。*/
export function toggleErpLayout(){
  erpHoriz = !erpHoriz;
  const el = document.getElementById('erp-mode');
  if(el) el.classList.toggle('horiz-layout', erpHoriz);
  const btn = document.getElementById('btn-erp-layout');
  if(btn) btn.textContent = erpHoriz ? 'ERP 上下排列' : 'ERP 左右排列';
  updateErpBackButtons();                // 連動改「回上/下」或「回左/右」字樣
  emit('apply-minimap-vis');             // 直/橫排標籤（上/下 vs 左/右）連動
  if(erpMode){ setTimeout(renderERP, 20); }
}

/** @returns {boolean} ERP 是否為左右排列（橫排）。供路線圖標籤用。*/
export function isErpHoriz(){ return erpHoriz; }

// =========================================================================
//  ERP 模式下「回 3D」兩顆鈕；unconn 模式的「連接」鈕
// =========================================================================
/**
 * 依 ERP 開關 / 排列 / 目前模式，更新動作列上：
 *   - 兩顆「回上/下(左/右)圖 3D」鈕（ERP 開、且非 unconn 模式時顯示）
 *   - 「連接」鈕（僅 unconn 模式顯示）
 */
export function updateErpBackButtons(){
  const seg = document.getElementById('pane-focus-seg');
  const bBack = document.getElementById('btn-back-3d');
  const bConn = document.getElementById('btn-connect');
  const inUnconn = getMode() === 'unconn';
  if(seg){
    seg.style.display = erpMode ? 'inline-flex' : 'none';
  }
  refreshFocusUi();
  // 在 ERP 模式下隱藏 3D 模式的「上一張/下一張」按鈕，避免與面板焦點鈕混淆
  const bPrevImg = document.getElementById('btn-prev-img');
  const bNextImg = document.getElementById('btn-next-img');
  if(bPrevImg){ bPrevImg.style.display = erpMode ? 'none' : 'inline-block'; }
  if(bNextImg){ bNextImg.style.display = erpMode ? 'none' : 'inline-block'; }
  // 單顆回 3D 鈕：ERP 開 且 非 unconn 模式時顯示
  if(bBack){
    bBack.style.display = (erpMode && !inUnconn) ? 'inline-block' : 'none';
  }
  // 連接鈕：僅 unconn 模式（自然要在 ERP 中）顯示
  if(bConn){ bConn.style.display = (erpMode && inUnconn) ? 'inline-block' : 'none'; }
}

/**
 * 離開 ERP 回到某面板的 3D 檢視。
 * 保持衍生的 nav.curPair 不變，nav.curImage 設為該面板圖並 loadImageView。
 * @param {'top'|'bot'} which
 */
export function exitErpToImage(which){
  const target = (which === 'top') ? nav.erpTopImg : nav.erpBotImg;
  // 關閉 ERP（沿用 toggleErp 的 DOM/旗標處理）
  if(erpMode){ toggleErp(); }
  if(typeof target === 'number' && target >= 0){
    nav.curImage = target;
    loadImageView(target, nav.curPair);   // 保持 curPair；loadImageView 會 emit view-changed
  }
}

// =========================================================================
//  RENDER（無條件執行；ERP 與 curPair 解耦）
// =========================================================================
/** 內部旗標：避免 renderERP -> emit('erp-derived') -> 其它訂閱者回頭呼叫 renderERP 造成遞迴。*/
let _renderingErp = false;

/**
 * 重繪整個 ERP（兩面板）。無條件執行（不再檢查 curPair）。
 * 流程：保證面板圖有效 -> 衍生配對 -> 同步 nav.curPair -> 設定拖曳方位 -> setup 兩面板。
 * 無 curPair 時一切照常（只是不畫主標記、Y/N 停用）。
 * 完成後 emit('erp-derived')：由 ui/info 訂閱刷新動作列可用性與狀態列文字（不 emit view-changed，避免遞迴）。
 */
export function renderERP(){
  if(!erpMode) return;
  if(_renderingErp) return;              // 防重入
  _renderingErp = true;
  try {
    const hd = getHouseData();
    if(!hd){                             // 尚未載入房子：清標籤，等 house-loaded 後再繪
      setPaneLabel('top', ''); setPaneLabel('bot', '');
      _derived = { top:-1, bot:-1, kind:'none', pair:null };
      return;
    }
    // 1) 面板圖唯一真相：永遠有效且互異
    const { top, bot } = ensureErpPanes(nav, hd);
    // 2) 衍生配對語意，同步寫回 nav.curPair（none -> null，但 UI 不癱瘓）
    const d = derivePairForPanes(top, bot, hd, STATE);
    _derived = { top, bot, kind:d.kind, pair:d.pair };
    nav.curPair = d.pair;                // 衍生結果即 curPair（找不到 pair 時 null）

    // 3) 主標記拖曳方位：pair -> 用 az_ij/az_ji（若已確認取確認角，人工 ROI 校正優先於語意輔助
    //：這條不變）；none -> 無主標記。語意輔助開啟時，未確認的 pair 再讓語意門校正
    //    對兩側各自獨立覆寫（見 applySemanticOverride）。
    if(d.pair){
      const cc = STATE.confirmed.find(x => pairKey(x.i, x.j) === pairKey(d.pair.i, d.pair.j));
      if(cc){ erpAzI = cc.az_i; erpAzJ = cc.az_j; _semanticInfo = null; }
      else {
        erpAzI = d.pair.az_ij; erpAzJ = d.pair.az_ji;
        _semanticInfo = null;
        if(STATE.ui.semanticAssist) applySemanticOverride(d.pair);
      }
    } else {
      erpAzI = null; erpAzJ = null; _semanticInfo = null;
    }

    // 4) 兩面板無條件 setup（各自有圖、控制列、參考門圈）
    if(!_passageDragWhich) _passageHoverWhich = null;
    setupErpPane('top', top);
    setupErpPane('bot', bot);
    // 5) 區域遮罩疊在 ERP overlay 上（非獨立模式）
    updateErpRegionOverlay();
  } finally {
    _renderingErp = false;
  }
  emit('erp-derived');                    // 刷新動作列可用性 + 狀態列文字（不觸發 view-changed 遞迴）
}

/**
 * 語意輔助。對未確認的目前 pair，兩側各自獨立判斷是否用門語意圖校正通道方位。
 * 資料（house 門清單 / 該對共視區域）可能還沒抓過：getPairOverride 這時回 null，本次
 * render 先顯示原始 az_ab；資料回來後的 onReady 回呼會在「仍是同一 pair」時重新 renderERP
 * 一次，讓覆寫生效。與 updateErpRegionOverlay 的 fetch-then-rerender 手法一致。
 * @param {{i:number,j:number,az_ij:number,az_ji:number}} pair
 */
function applySemanticOverride(pair){
  const house = getHouseName();
  const hd = getHouseData();
  const pKey = pairKey(pair.i, pair.j);
  const result = getPairOverride(house, hd, pair, () => {
    if(erpMode && _derived.pair && pairKey(_derived.pair.i, _derived.pair.j) === pKey) renderERP();
  });
  if(!result) return;                     // 資料仍在抓取中
  _semanticInfo = result;
  if(result.noSemanticForHouse) return;
  const triggeredI = result.i && result.i.triggered;
  const triggeredJ = result.j && result.j.triggered;
  if(triggeredI) erpAzI = result.i.az;
  if(triggeredJ) erpAzJ = result.j.az;
  // 觸發時記錄一則 console 訊息（哪一側、用了哪道門、包住比例）供目視抽查誤觸發。
  // 去重鍵含 house 與兩側實際採用的角度：同一對重覆 render 不重覆記，換房子同 index 的
  // 別對不會被誤擋，角度真的變了（換門/資料更新）則照記一次。
  const logKey = [house, pKey,
                  triggeredI ? result.i.doorAz.toFixed(1) : '-',
                  triggeredJ ? result.j.doorAz.toFixed(1) : '-'].join('|');
  if((triggeredI || triggeredJ) && _semanticLoggedKey !== logKey){
    _semanticLoggedKey = logKey;
    if(triggeredI){
      console.log('[語意輔助] ' + house + ' img ' + pair.i + ' 側：az_ij ' + pair.az_ij.toFixed(1) +
        '° → 門 ' + result.i.doorAz.toFixed(1) + '°（共視區包住 ' +
        (result.i.coverFrac * 100).toFixed(0) + '%）');
    }
    if(triggeredJ){
      console.log('[語意輔助] ' + house + ' img ' + pair.j + ' 側：az_ji ' + pair.az_ji.toFixed(1) +
        '° → 門 ' + result.j.doorAz.toFixed(1) + '°（共視區包住 ' +
        (result.j.coverFrac * 100).toFixed(0) + '%）');
    }
  }
}

/** 設定某面板控制列標籤（img N · 檔名前8 · 狀態徽章）。內容為 innerHTML。*/
function setPaneLabel(which, html){
  const el = document.getElementById(which === 'top' ? 'erp-label-top' : 'erp-label-bot');
  if(el) el.innerHTML = html;
}

/**
 * 對 (top,bot) 一張面板圖，算出它與另一面板圖的「該對狀態」徽章文字/顏色。
 * pending 琥珀 / confirmed 綠 / rejected 紅 / 非候選 灰。
 * @param {number} imgIdx 本面板圖
 * @param {number} otherIdx 對側面板圖
 * @returns {{label:string, color:string, status:string}}
 */
function paneBadge(imgIdx, otherIdx){
  const hd = getHouseData();
  if(hd && hd.pairs){
    const pr = hd.pairs.find(p => pairKey(p.i, p.j) === pairKey(imgIdx, otherIdx));
    if(pr){
      const st = pairStatus(pr);
      const col = st === 'confirmed' ? COL.green : st === 'pending' ? COL.amber : COL.red;
      const pct = (pr.prob != null) ? ' ' + (pr.prob * 100).toFixed(0) + '%' : '';
      return { label:statusLabel(st) + pct, color:col, status:st };
    }
  }
  // 非 hd.pairs：可能是人工確認邊
  const cc = STATE.confirmed.find(x => pairKey(x.i, x.j) === pairKey(imgIdx, otherIdx));
  if(cc){ return { label:'已確認(人工)', color:COL.green, status:'confirmed' }; }
  return { label:'非候選', color:COL.muted || '#93a4b8', status:'none' };
}

/**
 * 設定一個面板（尺寸、控制列標籤+徽章、圖片、點擊放橘圈、載入後畫標記）。
 * 無條件執行（不再依賴 nav.curPair）。
 * @param {'top'|'bot'} which
 * @param {number} imgIdx
 */
export function setupErpPane(which, imgIdx){
  const hd = getHouseData();
  const img = document.getElementById(which === 'top' ? 'erp-img-top' : 'erp-img-bot');
  const lab = document.getElementById(which === 'top' ? 'erp-label-top' : 'erp-label-bot');
  const inner = document.getElementById(which === 'top' ? 'erp-inner-top' : 'erp-inner-bot');
  if(!img || !lab || !inner || !hd) return;

  // 依 erp-pane 尺寸算最大 2:1 內框
  const pane = inner.parentElement;
  if(pane){
    const pW = pane.clientWidth, pH = pane.clientHeight;
    if(pW && pH){
      let w = pW, h = pW / 2;
      if(h > pH){ h = pH; w = pH * 2; }
      inner.style.width = w + 'px';
      inner.style.height = h + 'px';
    }
  }

  // 點面板：allConn 模式 -> 該圖全連接；unconn 模式 -> 放橘圈。控制列在 inner 外，天然不冒泡。
  inner.onpointerdown = function(e){
    if(e.button !== 0) return;
    // 保險：點到任何 button/圈時放行（不當成放圈/全連接）。控制列已移到 inner 外，此僅防 overlay 圈。
    if(e.target.closest('button, .erp-dot, .erp-main-dot')) return;
    const mode = getMode();
    if(mode === 'allConn'){
      e.preventDefault(); e.stopPropagation();
      handleCircleClick({ role:'pane', source:'erp', srcImg:imgIdx, which:which, az:0 });
      return;
    }
    if(mode !== 'unconn') return;
    e.preventDefault(); e.stopPropagation();
    const r = inner.getBoundingClientRect();
    const x = e.clientX - r.left;
    const az = (x / r.width * 360 - 180 + 360) % 360;
    const key = String(imgIdx);
    if(!STATE.extra_channels[key]) STATE.extra_channels[key] = [];
    STATE.extra_channels[key].push(round1(az));
    drawErpMarkers(which, imgIdx);
    scheduleSave();
    emit('state-changed');
    toast('已在此方位放置未連接通道標記 @' + az.toFixed(0) + '°');
    // 放圈後開推薦面板（列同方向 pending 候選，可瀏覽/連通/否決/手動後備）
    openRecoPanel(imgIdx, which, round1(az));
  };
  inner.oncontextmenu = function(e){ e.preventDefault(); };

  // ---- 控制列標籤：img N · 檔名前8 · 對狀態徽章（不再疊圖，資訊整合於控制列） ----
  const other = (which === 'top') ? nav.erpBotImg : nav.erpTopImg;
  const bd = paneBadge(imgIdx, other);
  const name = hd.images[imgIdx];
  const displayName = name.length > 12 ? '…' + name.slice(-12) : name;
  lab.innerHTML =
    '<span class="erp-imgname">img <b>' + imgIdx + '</b> · ' +
    displayName + '</span>' +
    '<span class="erp-badge" style="border-color:' + bd.color + ';color:' + bd.color + '">' +
    bd.label + '</span>';

  const url = '/images/' + encodeURIComponent(getHouseName()) + '/' + encodeURIComponent(hd.images[imgIdx]);
  if(img.getAttribute('data-src') !== url){
    img.setAttribute('data-src', url);
    img.onload = function(){ drawErpMarkers(which, imgIdx); };
    img.src = url;
  } else {
    drawErpMarkers(which, imgIdx);
  }
}

/**
 * 該圖要畫的參考門（排除本對的門，本對另用可拖曳主標記）。
 * 去掉對 nav.curPair 的依賴：pairAz 由 _derived（面板衍生對）決定，
 *   衍生為 none 時 pairAz=null，對應排除邏輯自動跳過。
 * @param {number} imgIdx
 * @returns {Array<object>}
 */
export function erpDoorList(imgIdx){
  const hd = getHouseData();
  if(!hd) return [];
  const thEl = document.getElementById('threshold');
  const th = thEl ? +thEl.value : 0.5;
  const out = [];
  // 本對主標記方位：僅當面板衍生出配對，且 imgIdx 為該對一端時才有值（否則 null -> 不排除）
  const dp = _derived.pair;
  const pairAz = (dp && imgIdx === dp.i) ? erpAzI : ((dp && imgIdx === dp.j) ? erpAzJ : null);

  if(STATE.ui.pairwise){
    clusterDoors(imgIdx, hd, STATE, pairStatus, pairKey).forEach(cl => {
      if(cl.status !== 'confirmed' && cl.bestProb < th) return;
      if(pairAz != null && cl.status !== 'confirmed' && angDiff(cl.repAz, pairAz) < CLUSTER_DELTA) return;
      out.push({
        az: cl.repAz,
        status: cl.status === 'confirmed' ? 'confirmed' : 'refdoor',
        targetImg: cl.repTarget,
        i: imgIdx
      });
    });
  }
  // 人工連接的確認邊若 hd.pairs 裡不存在（孤立照片），clusterDoors 抓不到 -> 直接迭代 STATE.confirmed 補綠圈。
  STATE.confirmed.forEach(c => {
    let az = null, tgt = null;
    if(c.i === imgIdx){ az = c.az_i; tgt = c.j; }
    else if(c.j === imgIdx){ az = c.az_j; tgt = c.i; }
    if(az == null) return;
    if(out.some(o => angDiff(o.az, az) < CLUSTER_DELTA)) return;      // 已由聚類/本對主標記涵蓋
    if(dp && pairKey(c.i, c.j) === pairKey(dp.i, dp.j)) return;       // 本對主標記另外畫（僅排除同一對，不憑角度）
    out.push({ az:az, status:'confirmed', targetImg:tgt, i:imgIdx });
  });
  (STATE.extra_channels[imgIdx] || STATE.extra_channels[String(imgIdx)] || []).forEach(a => {
    out.push({ az:a, status:'extra', i:imgIdx });
  });
  const gtEl = document.getElementById('gt-toggle');
  if(gtEl && gtEl.checked && dp){
    const gaz = (imgIdx === dp.i) ? dp.gt_az_ij : (imgIdx === dp.j ? dp.gt_az_ji : null);
    if(gaz != null){ out.push({ az:gaz, status:'gt', i:imgIdx }); }
  }
  return out;
}

// =========================================================================
//  本對主標記：每圖頂帶＋底帶；同圖垂直虛線（非跨面板）
//  視覺語言對齊 marker.js；DOM 版便於命中測試與水平拖曳。
// =========================================================================
const BAND_HALF_DEG = 15;
const BAND_FRAC = 0.055;
const BAND_MIN_PX = 20;
let _passageHoverWhich = null;   // 'top'|'bot'|null
let _passageDragWhich = null;

function setPassageLinkVisible(ov, show, xPx){
  if(!ov) return;
  let v = ov.querySelector('.erp-passage-vlink');
  if(!v){
    v = document.createElement('div');
    v.className = 'erp-passage-vlink';
    ov.appendChild(v);
  }
  if(xPx != null) v.style.left = xPx + 'px';
  v.classList.toggle('show', !!show);
}

function refreshPassageLinks(){
  ['top', 'bot'].forEach(which => {
    const ov = document.getElementById(which === 'top' ? 'erp-ov-top' : 'erp-ov-bot');
    if(!ov) return;
    const show = (_passageHoverWhich === which || _passageDragWhich === which);
    const band = ov.querySelector('.erp-main-dot');
    const x = band ? parseFloat(band.style.left) || 0 : 0;
    setPassageLinkVisible(ov, show, x);
  });
}

// ---- ERP 區域遮罩疊層（非獨立 viewmode；畫在目前 ERP 面板上）----
let _erpRegOn = false;
let _erpRegToken = 0;
let _erpRegCache = null;   // { key, sides } | null
let _erpRegNote = '';

export function isErpRegionsOn(){ return _erpRegOn; }

export function toggleErpRegions(){
  _erpRegOn = !_erpRegOn;
  const btn = document.getElementById('btn-erp-regions');
  if(btn){
    btn.classList.toggle('on', _erpRegOn);
    btn.textContent = '區域遮罩';
  }
  if(!erpMode && _erpRegOn){
    // 開遮罩時若還不在 ERP，自動切入 ERP 對比（不另開獨立模式）
    toggleErp();
  } else if(erpMode){
    renderERP();
  }
}

function clearErpRegCanvases(){
  ['top', 'bot'].forEach(which => {
    const ov = document.getElementById(which === 'top' ? 'erp-ov-top' : 'erp-ov-bot');
    if(!ov) return;
    const cv = ov.querySelector('.erp-reg-cv');
    if(cv){
      const ctx = cv.getContext('2d');
      if(ctx) ctx.clearRect(0, 0, cv.width, cv.height);
    }
  });
}

/** 在單一 overlay 上畫該側 region（ERP 平面座標，過縫 ±W）。*/
function paintErpRegionsOn(ov, items, srcW, srcH){
  if(!ov || !items || !items.length || !srcW || !srcH) return;
  let cv = ov.querySelector('.erp-reg-cv');
  if(!cv){
    cv = document.createElement('canvas');
    cv.className = 'erp-reg-cv';
    ov.insertBefore(cv, ov.firstChild);
  }
  const W = ov.clientWidth, H = ov.clientHeight;
  if(!W || !H) return;
  if(cv.width !== W || cv.height !== H){ cv.width = W; cv.height = H; }
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  items.forEach((item, idx) => {
    const poly = item.poly;
    if(!poly || poly.length < 3) return;
    const col = item.color || REG_PALETTE[idx % REG_PALETTE.length];
    for(const dx of [-srcW, 0, srcW]){
      const pts = poly.map(([x, y]) => [(x + dx) / srcW * W, y / srcH * H]);
      if(Math.max(...pts.map(p => p[0])) < 0 || Math.min(...pts.map(p => p[0])) > W) continue;
      ctx.beginPath();
      pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
      ctx.closePath();
      ctx.fillStyle = col + '4d';
      ctx.fill();
      ctx.strokeStyle = col;
      ctx.lineWidth = 2;
      ctx.stroke();
      const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
      const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
      if(cx >= 0 && cx < W){
        ctx.fillStyle = col;
        ctx.font = 'bold 16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('R' + (item.k != null ? item.k : idx), cx, cy);
      }
    }
  });
}

/** 依目前面板圖載入並疊區域遮罩。*/
function updateErpRegionOverlay(){
  if(!_erpRegOn || !erpMode){
    clearErpRegCanvases();
    return;
  }
  const hd = getHouseData();
  const house = getHouseName();
  if(!hd || !house) return;
  const top = _derived.top, bot = _derived.bot;
  if(top < 0 || bot < 0) return;
  const fa = hd.images[top], fb = hd.images[bot];
  if(!fa || !fb) return;
  const key = house + '|' + top + '|' + bot;
  const tk = ++_erpRegToken;

  /** pack.i = top 面板、pack.j = bot 面板（依 fetch(top,bot) 的 swap 已對齊）。*/
  const apply = (pack) => {
    if(tk !== _erpRegToken || !_erpRegOn) return;
    paintErpRegionsOn(document.getElementById('erp-ov-top'), pack.i, pack.Wi, pack.Hi);
    paintErpRegionsOn(document.getElementById('erp-ov-bot'), pack.j, pack.Wj, pack.Hj);
  };

  if(_erpRegCache && _erpRegCache.key === key){
    apply(_erpRegCache.sides);
    return;
  }
  fetchRegions(house, fa, fb).then(rj => {
    if(tk !== _erpRegToken) return;
    if(!rj || rj.error){
      _erpRegCache = null;
      _erpRegNote = (rj && rj.error) || '無區域';
      clearErpRegCanvases();
      return;
    }
    // fetch(top,bot)：swap=false → top 用 a、bot 用 b；swap=true → 檔為 bot_top，top 用 b
    const keyTop = rj.swap ? 'b' : 'a';
    const keyBot = rj.swap ? 'a' : 'b';
    const [Wt, Ht] = keyTop === 'a' ? (rj.img_size_a || [0, 0]) : (rj.img_size_b || [0, 0]);
    const [Wb, Hb] = keyBot === 'a' ? (rj.img_size_a || [0, 0]) : (rj.img_size_b || [0, 0]);
    const regs = rj.regions || [];
    const topItems = [], botItems = [];
    regs.forEach((reg, k) => {
      const col = REG_PALETTE[k % REG_PALETTE.length];
      const pt = reg[keyTop] && reg[keyTop].polygon;
      const pb = reg[keyBot] && reg[keyBot].polygon;
      if(pt && pt.length >= 3) topItems.push({ poly:pt, color:col, k });
      if(pb && pb.length >= 3) botItems.push({ poly:pb, color:col, k });
    });
    const pack = { i:topItems, j:botItems, Wi:Wt, Hi:Ht, Wj:Wb, Hj:Hb };
    _erpRegCache = { key, sides: pack };
    _erpRegNote = regs.length + ' 區';
    apply(pack);
  }).catch(() => {
    if(tk === _erpRegToken){
      _erpRegCache = null;
      clearErpRegCanvases();
    }
  });
}

/**
 * 在面板 overlay 畫參考門圈與本對可拖曳主標記（通道頂帶＝連通道方位）。無條件執行（不再檢查 curPair）。
 * 圈圈左鍵 -> dispatch.handleCircleClick；右鍵 -> 刪橘圈 / 重新否決確認門。
 * 本對主標記僅在面板衍生出配對（_derived.kind==='pair'|'confirmed'）時才畫。
 * @param {'top'|'bot'} which
 * @param {number} imgIdx
 */
export function drawErpMarkers(which, imgIdx){
  const ov = document.getElementById(which === 'top' ? 'erp-ov-top' : 'erp-ov-bot');
  if(!ov) return;
  ov.innerHTML = '';
  const W = ov.clientWidth, H = ov.clientHeight; if(!W || !H) return;
  const xOf = (az) => ((az + 180) % 360) / 360 * W;
  const colOf = { confirmed:COL.green, extra:COL.orange, refdoor:COL.refdoor, pending:COL.amber, gt:COL.gt };

  erpDoorList(imgIdx).forEach(d => {
    const c = colOf[d.status] || COL.amber;
    const dt = document.createElement('div'); dt.className = 'erp-dot';
    dt.style.left = xOf(d.az) + 'px'; dt.style.top = (H / 2) + 'px'; dt.style.borderColor = c;
    if(d.status === 'confirmed') dt.style.zIndex = '11';  // 已確認門優先於本對主標記(z-index 10)顯示，避免被蓋住看不見

    if(d.status === 'gt'){
      dt.style.pointerEvents = 'none';
      dt.style.opacity = '0.9';
      dt.style.cursor = 'default';
      dt.title = 'GT 基準門方位';
    } else {
      dt.style.opacity = '0.7';
      dt.style.cursor = 'pointer';
      dt.title = d.status === 'confirmed' ? '已確認門：左鍵穿門 / 右鍵重新否決'
        : d.status === 'refdoor' ? '預測門：左鍵跳去複核'
        : d.status === 'extra' ? '未連接通道：左鍵找預測目標 / 右鍵刪除' : '';

      dt.onpointerdown = function(e){
        if(e.button !== 0) return;           // 左鍵
        e.preventDefault();
        e.stopPropagation();                 // 避免觸發面板 onpointerdown（會多放一個標記）
        handleCircleClick({
          role: d.status,                    // 'confirmed'|'refdoor'|'extra'
          az: d.az,
          i: imgIdx,
          j: (d.status === 'confirmed') ? d.targetImg : undefined,
          targetImg: d.targetImg,
          srcImg: imgIdx,
          which: which,
          source: 'erp'
        });
        if(erpMode){ renderERP(); }
      };

      dt.oncontextmenu = function(e){
        e.preventDefault();
        e.stopPropagation();
        if(d.status === 'extra'){
          const key = String(imgIdx);
          if(STATE.extra_channels[key]){
            STATE.extra_channels[key] = STATE.extra_channels[key].filter(az => Math.abs(az - d.az) > 0.1);
            if(STATE.extra_channels[key].length === 0){ delete STATE.extra_channels[key]; }
            toast('已刪除此方位之未連接通道標記');
            scheduleSave();
            drawErpMarkers(which, imgIdx);
            emit('state-changed');
          }
        } else if(d.status === 'confirmed'){
          const ok = confirm('您確定要重新否決/刪除此連通門嗎？');
          if(ok){
            rejectConfirmed(imgIdx, d.targetImg);
            toast('已重新否決此連通門');
            if(erpMode){ renderERP(); }
          }
        }
      };
    }
    ov.appendChild(dt);
  });

  // ---- unconn 推薦「對側預覽圈」（琥珀虛線、不可點）----
  // 使用者在此側放橘圈後，推薦面板把對側面板切到候選對面圖，並在對面「看回來的方位」畫琥珀預覽圈，
  // 讓使用者直接看圖判斷是否為同一通道。此圈僅視覺提示，pointer-events 關閉。
  const rp = getRecoPreview();
  if(rp && rp.which === which && rp.az != null){
    const pv = document.createElement('div');
    pv.className = 'erp-dot';
    pv.style.left = xOf(rp.az) + 'px';
    pv.style.top = (H / 2) + 'px';
    pv.style.borderColor = COL.amber;
    pv.style.borderStyle = 'dashed';
    pv.style.background = 'rgba(244, 180, 0, 0.18)';
    pv.style.pointerEvents = 'none';
    pv.style.opacity = '0.85';
    pv.title = '推薦候選在此圖看回來的預估方位';
    ov.appendChild(pv);
  }

  // ---- 任務 2 新增：通用面板預覽圈（與 _reco 並存、樣式一致） ----
  const pp = _panePreview;
  if(pp && pp.which === which && pp.az != null){
    const pv = document.createElement('div');
    pv.className = 'erp-dot';
    pv.style.left = xOf(pp.az) + 'px';
    pv.style.top = (H / 2) + 'px';
    pv.style.borderColor = COL.amber;
    pv.style.borderStyle = 'dashed';
    pv.style.background = 'rgba(244, 180, 0, 0.18)';
    pv.style.pointerEvents = 'none';
    pv.style.opacity = '0.85';
    pv.title = '批量否決選中項在此圖看回來的方位';
    ov.appendChild(pv);
  }

  // ---- 本對主標記：同圖頂帶＋底帶，hover/拖曳時同圖垂直虛線 ----
  const dp = _derived.pair;
  if(!dp){
    setPassageLinkVisible(ov, false);
    return;
  }
  if(imgIdx !== dp.i && imgIdx !== dp.j) return;
  const pairAz = (imgIdx === dp.i) ? erpAzI : erpAzJ;
  if(pairAz == null) return;
  const st = pairStatus(dp);
  let mainColor = COL.amber;
  if(st === 'confirmed') mainColor = COL.green;
  else if(st === 'rejected' || st === 'eliminated') mainColor = COL.red;

  let line = null;
  if(SHOW_ERP_MAIN_LINE){
    line = document.createElement('div'); line.className = 'erp-line erp-main';
    line.style.left = xOf(pairAz) + 'px'; line.style.background = mainColor; ov.appendChild(line);
  }
  const otherImg = (imgIdx === dp.i) ? dp.j : dp.i;
  const stText = st === 'confirmed' ? '已確認' : (st === 'rejected' || st === 'eliminated') ? '已否決' : '待確認';
  const bandTitle = (az) => '往 img ' + otherImg + ' · 水平拖曳校正 @' + az.toFixed(0) + '° · ' + stText;
  const bandH = Math.max(BAND_MIN_PX, Math.round(H * BAND_FRAC));
  const bandW = Math.max(24, (BAND_HALF_DEG * 2 / 360) * W);
  const settled = (st === 'confirmed' || st === 'rejected' || st === 'eliminated');
  const x0 = xOf(pairAz);

  // 同圖垂直虛線（平時隱藏）
  const vlink = document.createElement('div');
  vlink.className = 'erp-passage-vlink';
  vlink.style.left = x0 + 'px';
  ov.appendChild(vlink);

  function makeBand(edge){
    const mk = document.createElement('div');
    mk.className = 'erp-main-dot ' + (edge === 'top' ? 'erp-band-top' : 'erp-band-bot');
    mk.style.left = x0 + 'px';
    mk.style.width = bandW + 'px';
    mk.style.color = mainColor;
    mk.title = bandTitle(pairAz);
    const bandCls = 'erp-passage-band ' + (edge === 'bot' ? 'bot ' : '') + (settled ? 'settled' : 'pending');
    mk.innerHTML =
      '<div class="' + bandCls + '" style="color:' + mainColor + ';min-height:' + bandH + 'px">' +
        '<div class="erp-passage-bar"></div>' +
        '<div class="erp-passage-ticks"><span class="t-end"></span><span class="t-mid"></span><span class="t-end"></span></div>' +
        '<div class="erp-passage-label">往 img ' + otherImg + '</div>' +
        (edge === 'top'
          ? '<div class="erp-passage-status" style="color:' + mainColor + '">' + stText + '</div>'
          : '') +
      '</div>';
    ov.appendChild(mk);

    mk.onpointerenter = function(){
      _passageHoverWhich = which;
      refreshPassageLinks();
    };
    mk.onpointerleave = function(){
      if(_passageDragWhich !== which){
        if(_passageHoverWhich === which) _passageHoverWhich = null;
        refreshPassageLinks();
      }
    };

    mk.onpointerdown = function(e){
      if(e.button !== 0) return;
      e.preventDefault(); e.stopPropagation();
      const mode = getMode();
      if(mode === 'allConn' || mode === 'reject' || mode === 'noRoi'){
        const other = (imgIdx === dp.i) ? dp.j : dp.i;
        handleCircleClick({ role:'candidate', az:pairAz, i:dp.i, j:dp.j,
          targetImg:other, srcImg:imgIdx, which:which, source:'erp' });
        return;
      }
      let dragMovedLocal = false;
      _passageDragWhich = which;
      refreshPassageLinks();
      function move(ev){
        dragMovedLocal = true;
        const r = ov.getBoundingClientRect();
        const x = Math.max(0, Math.min(r.width, ev.clientX - r.left));
        const az = (x / r.width * 360 - 180 + 360) % 360;
        if(imgIdx === dp.i) erpAzI = az; else erpAzJ = az;
        if(line) line.style.left = x + 'px';
        ov.querySelectorAll('.erp-main-dot').forEach(el => {
          el.style.left = x + 'px';
          el.title = bandTitle(az);
        });
        vlink.style.left = x + 'px';
        if(pairStatus(dp) === 'confirmed'){
          const cc = STATE.confirmed.find(c => pairKey(c.i, c.j) === pairKey(dp.i, dp.j));
          if(cc){
            if(imgIdx === cc.i) cc.az_i = round1(az);
            else if(imgIdx === cc.j) cc.az_j = round1(az);
          }
        }
        refreshPassageLinks();
      }
      function up(){
        document.removeEventListener('pointermove', move);
        document.removeEventListener('pointerup', up);
        _passageDragWhich = null;
        const still = ov.querySelector('.erp-main-dot:hover');
        if(!still) _passageHoverWhich = null;
        refreshPassageLinks();
        if(dragMovedLocal){
          scheduleSave();
          emit('state-changed');
          if(getHouseData()) buildMarkersForImage(nav.curImage, nav.curPair);
        }
      }
      document.addEventListener('pointermove', move);
      document.addEventListener('pointerup', up);
    };

    mk.oncontextmenu = function(e){
      e.preventDefault(); e.stopPropagation();
      if(st === 'confirmed'){
        const ok = confirm('您確定要重新否決此已確認的連通道 ' + dp.i + '–' + dp.j + ' 嗎？');
        if(ok){ rejectConfirmed(dp.i, dp.j); }
      }
    };
    return mk;
  }

  makeBand('top');
  makeBand('bot');
  refreshPassageLinks();
  // drawErpMarkers 會清 ov；若遮罩開著，重疊一次（cache hit 不重抓）
  if(_erpRegOn && !_renderingErp) updateErpRegionOverlay();
}

// =========================================================================
//  ARROW NAV（操作 erpTopImg/erpBotImg，任何時候都可用）
// =========================================================================
/**
 * ERP 面板換圖（上/下方向）。純操作 erpTopImg/erpBotImg 後 re-derive；不再依賴 nav.curPair。
 * @param {'top'|'bot'} which
 * @param {number} dir -1 或 +1
 */
export function changeErpPaneImage(which, dir){
  const hd = getHouseData();
  if(!hd) return;
  const N = hd.images.length;
  ensureErpPanes(nav, hd);                 // 保證起點合法

  let curI = nav.erpTopImg, curJ = nav.erpBotImg;
  let target = (which === 'top') ? (curI + dir + N) % N : (curJ + dir + N) % N;
  const other = (which === 'top') ? curJ : curI;
  // 撞到同一張圖時，往同方向再跳一格（略過那張圖）。
  if(target === other){
    target = (target + dir + N) % N;
    if(target === other){ toast('圖片數量太少，無法切換到不同的另一張圖'); return; }
  }
  setErpPaneImage(which, target);
}

/**
 * 把某面板絕對切到 imgIdx（路線圖點擊/換圖共用）。含相同圖防撞。
 * 純操作 erpTopImg/erpBotImg，載入該面板圖並重繪（renderERP 內 re-derive 配對）。
 * @param {'top'|'bot'} which
 * @param {number} imgIdx
 */
export function setErpPaneImage(which, imgIdx){
  const hd = getHouseData();
  if(!hd) return;
  const N = hd.images.length;
  ensureErpPanes(nav, hd);

  let curI = nav.erpTopImg, curJ = nav.erpBotImg;
  if(which === 'top'){ curI = ((imgIdx % N) + N) % N; }
  else { curJ = ((imgIdx % N) + N) % N; }
  // 相同圖防撞：往 +1 找下一張不同者
  if(curI === curJ){
    let guard = 0;
    while(curI === curJ && guard < N){
      if(which === 'top') curI = (curI + 1) % N; else curJ = (curJ + 1) % N;
      guard++;
    }
    if(curI === curJ){ toast('圖片數量太少，無法切換到不同的另一張圖'); return; }
  }

  nav.erpTopImg = curI; nav.erpBotImg = curJ;
  nav.curImage = (which === 'top') ? curI : curJ;
  // 載入該面板圖到 3D 材質快取背景（維持 3D/ERP 一致），curPair 由 renderERP 衍生
  loadImageView(nav.curImage, null);
  setTimeout(renderERP, 50);
}

/**
 * 為某圖挑一個 curPair（3D 換圖與路線圖點擊共用）。
 * 優先「含該圖的 pending pair」；否則「任一含該圖的 pair」；都沒有則回 null（孤立照片）。
 * @param {number} imgIdx
 * @returns {object|null}
 */
export function pickPairForImage(imgIdx){
  const hd = getHouseData();
  if(!hd) return null;
  let p = hd.pairs.find(x => (x.i === imgIdx || x.j === imgIdx) && pairStatus(x) === 'pending');
  if(!p){ p = hd.pairs.find(x => (x.i === imgIdx || x.j === imgIdx)); }
  return p || null;
}

/**
 * 路線圖點節點（3D 模式）/ 3D 檢視器換圖共用的「絕對跳到某圖」邏輯。
 * 選一個含該圖的 pair 為 curPair（無 pair 的孤立照片 -> curPair=null 直接 loadImageView）。
 * @param {number} imgIdx
 */
export function goto3DImage(imgIdx){
  const hd = getHouseData();
  if(!hd) return;
  const N = hd.images.length;
  const idx = ((imgIdx % N) + N) % N;
  nav.curPair = pickPairForImage(idx);   // 可能為 null（孤立照片）
  nav.viewFrom = null; nav.reciprocalKey = null;
  loadImageView(idx, nav.curPair);       // curPair=null 時 loadImageView 只貼圖、不畫候選
  emit('view-changed');
}

/**
 * 3D 檢視器換圖（左右箭頭）。挑一個含新圖的 pending（否則任意含之者）為 curPair。
 * @param {number} dir -1 或 +1
 */
export function change3DViewerImage(dir){
  const hd = getHouseData();
  if(!hd) return;
  const N = hd.images.length;
  const nextImg = (nav.curImage + dir + N) % N;
  goto3DImage(nextImg);
}

// =========================================================================
//  unconn 放圈推薦面板（列同方向 pending 候選 → 瀏覽/連通/否決/手動後備）
// =========================================================================
/**
 * 開啟推薦面板：以 srcImg 上 az 方位，recommendForCircle 取同方向 pending 候選並顯示第 0 個。
 * @param {number} srcImg 放圈的圖 @param {'top'|'bot'} srcWhich 該圖面板 @param {number} az 圈方位
 */
export function openRecoPanel(srcImg, srcWhich, az){
  const hd = getHouseData(); if(!hd) return;
  const panel = document.getElementById('reco-panel'); if(!panel) return;
  const list = recommendForCircle(srcImg, az, hd, STATE, pairStatus);
  _reco = { active:true, srcImg, srcWhich, az, list, idx:0, previewOther:null, showCandidates:false };
  panel.style.display = 'flex';
  const title = document.getElementById('reco-panel-title');
  if(title) title.textContent = '推薦連通 · img ' + srcImg + ' @' + az.toFixed(0) + '°';
  showRecoCandidate();
}

/** 關閉推薦面板並清掉對側預覽圈。供 modes 離開 unconn 時呼叫。*/
export function closeRecoPanel(){
  const wasActive = _reco.active;
  _reco = { active:false, srcImg:-1, srcWhich:'top', az:0, list:[], idx:0, previewOther:null, showCandidates:false };
  const panel = document.getElementById('reco-panel');
  if(panel) panel.style.display = 'none';
  if(wasActive && erpMode){ renderERP(); }     // 清掉對側預覽圈
}

/** @returns {boolean} 推薦面板是否啟用。*/
export function isRecoActive(){ return _reco.active; }
/** @returns {{which:string,az:number}|null} 對側面板目前的預覽圈（drawErpMarkers 讀取畫琥珀預覽）。*/
export function getRecoPreview(){
  if(!_reco.active || _reco.previewOther == null) return null;
  const otherWhich = (_reco.srcWhich === 'top') ? 'bot' : 'top';
  return { which:otherWhich, az:_reco.previewOther };
}

/** 顯示目前索引的候選：切對側面板到候選對面圖、畫預覽圈、更新面板內容。*/
function showRecoCandidate(){
  const body = document.getElementById('reco-panel-body');
  const foot = document.getElementById('reco-panel-foot');
  if(!body) return;
  const hd = getHouseData();
  const n = _reco.list.length;

  if(!_reco.showCandidates){
    _reco.previewOther = null;
    body.innerHTML =
      '<div class="rp-empty" style="margin-bottom:12px; line-height:1.5;">' +
        '已在此方位放置未連接通道標記。<br>' +
        '若希望尋找模型預測的對面候選圖，請點擊下方按鈕：' +
      '</div>' +
      '<button class="sp-btn primary" id="btn-show-reco" style="width:100%; height:36px; font-weight:bold;">' +
        '🔍 尋找推薦候選 (' + n + ')' +
      '</button>';
    if(foot) foot.style.display = 'none';
    const btn = document.getElementById('btn-show-reco');
    if(btn){
      btn.onclick = () => {
        _reco.showCandidates = true;
        showRecoCandidate();
      };
    }
    if(erpMode) renderERP();
    return;
  }

  // 0 筆或已瀏覽完 -> 手動後備提示
  if(n === 0 || _reco.idx >= n){
    _reco.previewOther = null;
    body.innerHTML = '<div class="rp-empty">無' + (n === 0 ? '' : '（更多）') +
      '推薦候選。<br>可用「箭頭 / 路線圖」換對面圖找連通，於對面面板放第二個橘圈，再按動作列的「連接」鈕（雙圈連接流程不變）。</div>';
    if(foot) foot.style.display = 'none';
    if(erpMode) renderERP();
    return;
  }
  if(foot) foot.style.display = 'flex';
  const c = _reco.list[_reco.idx];

  // 切對側面板到候選對面圖（不動放圈的此側）
  const otherWhich = (_reco.srcWhich === 'top') ? 'bot' : 'top';
  if(otherWhich === 'top'){ nav.erpTopImg = c.otherImg; } else { nav.erpBotImg = c.otherImg; }
  // 對側預覽方位：候選在對面圖看回來的方位（畫琥珀預覽圈供直接看圖判斷）
  _reco.previewOther = (c.oppAz != null) ? c.oppAz : null;

  const fullOname = (hd && hd.images[c.otherImg]) ? hd.images[c.otherImg] : '';
  const oname = fullOname.length > 14 ? '…' + fullOname.slice(-14) : fullOname;
  body.innerHTML =
    '<div class="rp-counter">候選 ' + (_reco.idx + 1) + ' / ' + n + '</div>' +
    '<div class="rp-card">' +
      '<div class="rp-target">img ' + c.otherImg + '</div>' +
      '<div class="rp-prob">' + (c.prob * 100).toFixed(0) + '%</div>' +
      '<div class="rp-sub">' + oname + '<br>Δaz ' + c.dAz.toFixed(0) + '° · 對面方位 ' +
        (c.oppAz != null ? c.oppAz.toFixed(0) + '°' : '—') + '</div>' +
      '<div class="rp-nav">' +
        '<button class="sp-btn ghost" id="reco-prev">‹ 上一個</button>' +
        '<button class="sp-btn ghost" id="reco-next">下一個 ›</button>' +
      '</div>' +
    '</div>';
  const prev = document.getElementById('reco-prev'), next = document.getElementById('reco-next');
  if(prev){ prev.disabled = _reco.idx <= 0; prev.onclick = () => stepReco(-1); }
  if(next){ next.onclick = () => stepReco(1); }

  // 對側面板換圖 + 重繪（renderERP 會經 drawErpMarkers 讀 getRecoPreview 畫預覽圈）
  loadImageView(c.otherImg, null);
  if(erpMode) setTimeout(renderERP, 30); else renderERP();
}

/** 瀏覽上一個/下一個候選。*/
function stepReco(dir){
  const n = _reco.list.length;
  const ni = _reco.idx + dir;
  if(ni < 0){ return; }
  _reco.idx = ni;                 // 允許 == n（顯示手動後備提示）
  if(_reco.idx > n) _reco.idx = n;
  showRecoCandidate();
}

/** 重新計算目前圈的推薦（否決/連通後剩餘候選刷新）。保持面板開啟。*/
function refreshReco(){
  const hd = getHouseData(); if(!hd || !_reco.active) return;
  _reco.list = recommendForCircle(_reco.srcImg, _reco.az, hd, STATE, pairStatus);
  if(_reco.idx >= _reco.list.length) _reco.idx = _reco.list.length;   // 落到手動後備提示
  showRecoCandidate();
}

/** 「就是這個(連通)」：確認目前候選；成功後移除用掉的橘圈、刷新推薦（剩餘候選）。*/
function recoConnect(){
  if(!_reco.active || _reco.idx >= _reco.list.length){ return; }
  const c = _reco.list[_reco.idx];
  const ok = confirmRecommendedCandidate(_reco.srcImg, c.otherImg, c.az, c.oppAz);
  if(!ok) return;
  // 此側橘圈已被 confirmRecommendedCandidate 移除 -> 該圈的推薦作廢，關閉面板（留在 unconn 續放下一圈）。
  closeRecoPanel();
}

/** 「否決此候選」：rejectPairs 單筆，自動跳下一個候選。*/
function recoReject(){
  if(!_reco.active || _reco.idx >= _reco.list.length){ return; }
  const c = _reco.list[_reco.idx];
  rejectPairs([[c.i, c.j]]);
  refreshReco();                  // 剩餘候選刷新（idx 不變 -> 自動落到下一個）
}

/** 綁定推薦面板底部/關閉鈕（boot 一次）。*/
export function initRecoPanel(){
  const bind = (id, fn) => { const el = document.getElementById(id); if(el) el.onclick = fn; };
  bind('reco-connect', recoConnect);
  bind('reco-reject', recoReject);
  bind('reco-panel-close', closeRecoPanel);
}

/** 重置 ERP 拖曳方位（reset 房屋時）。*/
export function resetErpAz(){ erpAzI = null; erpAzJ = null; }

/** 訂閱事件：resize / 視角改變時若在 ERP 則重繪；reset 房屋時清方位。*/
export function initErp(){
  on('resize', () => { if(erpMode) renderERP(); });
  on('view-changed', () => { if(erpMode) renderERP(); });
  on('erp-reset-az', resetErpAz);

  const bindCtrl = (id, which) => {
    const el = document.getElementById(id);
    if(el){
      el.addEventListener('click', (e) => {
        if(e.target.closest('button')) return;
        setPaneFocus(which);
      });
    }
  };
  bindCtrl('erp-ctrl-top', 'top');
  bindCtrl('erp-ctrl-bot', 'bot');

  const regBtn = document.getElementById('btn-erp-regions');
  if(regBtn){
    regBtn.onclick = () => toggleErpRegions();
    regBtn.classList.toggle('on', _erpRegOn);
    regBtn.textContent = '區域遮罩';
  }
}
