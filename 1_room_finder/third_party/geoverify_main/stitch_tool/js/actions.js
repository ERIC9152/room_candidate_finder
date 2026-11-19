/* =========================================================================
   actions.js：標註動作：Y/N/A、否決、無 ROI 淘汰、全連接、重新否決、reset 房屋。
   行為逐字沿用原 app.js（含 ERP 情境下的對側自動切換與門聚類淘汰收緊）。
   依賴：state / nav / scene3d / modes / clustering(via state helpers) / geometry / config / util / events。
   狀態變更後一律 emit('state-changed')（queue/minimap/compass 訂閱重繪），
   視角相關（重畫 markers、面向新方位）則直接呼叫 scene3d，維持與原本一致的即時反應。
   ========================================================================= */

import { CLUSTER_DELTA } from './config.js';
import { angDiff } from './geometry.js';
import {
  STATE, getHouseData, getHouseName, pairKey, inList, isConfirmed, pairStatus,
  scheduleSave, doSave, cancelPendingSave, resetState
} from './state.js';
import { nav, firstPending } from './nav.js';
import {
  loadImageView, buildMarkersForImage, getCandidateMarker, promoteCandidateToConfirmed,
  faceAz, clearMarkers
} from './scene3d.js';
import { toggleMode, clearMode } from './modes.js';
import { isErpOn as erpOn, renderERP, getErpAzI, getErpAzJ } from './erp.js';
import { toast, round1 } from './util.js';
import { emit } from './events.js';

/** 統一：狀態改變後通知所有 UI 重繪（queue/minimap/compass/info）。*/
function stateChanged(){ emit('state-changed'); }
/** 若在 ERP，重繪 ERP。*/
function renderErpIfOn(){ if(erpOn()) renderERP(); }

// =========================================================================
//  共用確認 helper（actYes 與「連接」鈕共用；需求 6）
// =========================================================================
/**
 * 建立/更新一條確認邊並收尾（寫入 confirmed、移除 rejected/eliminated、autoEliminate 兩端、存檔）。
 * 對「hd.pairs 裡不存在的孤立照片人工連接」也照樣寫入 STATE.confirmed。
 * 不負責 UI 導覽（caller 自行決定重畫/切換與 emit）。
 * @param {number} i 較小 index 那張圖
 * @param {number} j 另一張圖
 * @param {number} azI i 圖上的門方位
 * @param {number} azJ j 圖上的門方位
 * @returns {{i:number,j:number,az_i:number,az_j:number}} 寫入的確認邊
 */
export function confirmEdge(i, j, azI, azJ){
  // 順序無關：以 min/max 正規化，方向對應要正確
  let a = i, b = j, aAz = azI, bAz = azJ;
  if(a > b){ a = j; b = i; aAz = azJ; bAz = azI; }
  // 先移除否決/淘汰狀態（若存在）
  STATE.rejected = STATE.rejected.filter(x => pairKey(x[0], x[1]) !== pairKey(a, b));
  STATE.eliminated = STATE.eliminated.filter(x => pairKey(x[0], x[1]) !== pairKey(a, b));
  // 去重：同一對只留一筆 confirmed
  STATE.confirmed = STATE.confirmed.filter(x => pairKey(x.i, x.j) !== pairKey(a, b));
  const edge = { i:a, j:b, az_i:round1(aAz), az_j:round1(bAz) };
  STATE.confirmed.push(edge);
  // autoEliminate 兩張圖（safeguard-aware）：以此邊為 pair 物件，各圖各跑一次
  const pairObj = { i:a, j:b, az_ij:aAz, az_ji:bAz };
  autoEliminateForImage(pairObj, a, aAz);
  autoEliminateForImage(pairObj, b, bAz);
  scheduleSave();
  return edge;
}

// =========================================================================
//  Y：確認連通
// =========================================================================
/** 確認目前 pair 連通。沿用原 actYes 全部邏輯（含自動淘汰與對側切換）。*/
export function actYes(){
  const p = nav.curPair;
  const hd = getHouseData();
  if(!p || pairStatus(p) === 'confirmed'){ toast('此配對已確認連通'); return; }
  let azI, azJ;
  const erpMode = erpOn();
  const mk = getCandidateMarker();
  if(erpMode){
    azI = (getErpAzI() != null) ? getErpAzI() : p.az_ij;
    azJ = (getErpAzJ() != null) ? getErpAzJ() : p.az_ji;
  } else {
    if(mk){
      if(nav.curImage === p.i){ azI = mk.userData.az; azJ = p.az_ji; }
      else if(nav.curImage === p.j){ azJ = mk.userData.az; azI = p.az_ij; }
      else { azI = p.az_ij; azJ = p.az_ji; }
    } else { azI = p.az_ij; azJ = p.az_ji; }
  }

  // 寫入確認邊（移除否決/淘汰、去重、autoEliminate 兩端、存檔）。與「連接」鈕共用 helper
  confirmEdge(p.i, p.j, azI, azJ);
  // 目前琥珀標記轉綠（可拖曳、可穿門，持久）
  if(mk){ promoteCandidateToConfirmed(nav.curImage === p.i ? p.j : p.i); }

  toast('已確認連通 ' + p.i + '–' + p.j + '！');

  stateChanged();
  renderErpIfOn();
}

// =========================================================================
//  N：否決（或對已確認的目前對重新否決）
// =========================================================================
/** 否決按鈕/鍵：對已確認的目前對彈確認框重新否決；否則切換否決模式。*/
export function actNo(){
  const p = nav.curPair;
  if(p && isConfirmed(p.i, p.j)){
    const ok = confirm('您確定要重新否決目前連通的配對 ' + p.i + '–' + p.j + ' 嗎？');
    if(ok){ rejectConfirmed(p.i, p.j); }
    return;
  }
  toggleRejectMode();
}
/** 切換否決模式（互斥；經 modes 狀態機，含自動進/出 ERP）。*/
export function toggleRejectMode(){ toggleMode('reject'); }
/** 切換無 ROI 模式。*/
export function toggleNoRoiMode(){ toggleMode('noRoi'); }
/** 切換全連接模式。*/
export function toggleAllConnMode(){ toggleMode('allConn'); }
/** 切換未連接通道模式。*/
export function toggleUnconn(){ toggleMode('unconn'); }

// =========================================================================
//  未連接通道：連接
// =========================================================================
/** 讀某圖目前的未連接通道橘圈方位陣列（兼容數字/字串 key）。*/
function extraAzList(imgIdx){
  return (STATE.extra_channels[imgIdx] || STATE.extra_channels[String(imgIdx)] || []);
}
/** 移除某圖某方位的橘圈（連接成功後清掉，因為已變綠色確認邊）。*/
function removeExtraAz(imgIdx, az){
  const key = String(imgIdx);
  if(!STATE.extra_channels[key]) return;
  STATE.extra_channels[key] = STATE.extra_channels[key].filter(a => Math.abs(a - az) > 0.05);
  if(STATE.extra_channels[key].length === 0) delete STATE.extra_channels[key];
}

/**
 * 純邏輯：驗證「上下面板各恰好 1 個橘圈」並算出要建立的確認邊（方向對應正確）。
 * 不觸碰 DOM/nav/STATE，供單元測試直接呼叫。
 * @param {number} top 上面板圖 index
 * @param {number} bot 下面板圖 index
 * @param {number[]} exTop 上面板圖的橘圈方位陣列
 * @param {number[]} exBot 下面板圖的橘圈方位陣列
 * @returns {{ok:false, reason:string, topN:number, botN:number} |
 *           {ok:true, i:number, j:number, azI:number, azJ:number}}
 */
export function computeUnconnEdge(top, bot, exTop, exBot){
  if(top == null || bot == null || top < 0 || bot < 0 || top === bot){
    return { ok:false, reason:'same-or-invalid', topN:(exTop||[]).length, botN:(exBot||[]).length };
  }
  const nTop = (exTop || []).length, nBot = (exBot || []).length;
  if(nTop !== 1 || nBot !== 1){
    return { ok:false, reason:'need-one-each', topN:nTop, botN:nBot };
  }
  const azTop = exTop[0], azBot = exBot[0];
  // 方向對應：i=min(top,bot) 的橘圈 -> az_i；j=max 的橘圈 -> az_j。
  const i = Math.min(top, bot), j = Math.max(top, bot);
  const azI = (i === top) ? azTop : azBot;
  const azJ = (j === top) ? azTop : azBot;
  return { ok:true, i, j, azI, azJ };
}

/**
 * 「連接」鈕（僅 unconn 模式顯示）：驗證上/下面板圖各恰好 1 個橘圈，成立則建立確認邊。
 * - 驗證與方向對應交給純函式 computeUnconnEdge。
 * - 對 hd.pairs 裡不存在的孤立照片人工連接也照樣寫入（confirmEdge 負責）。
 * - 成功後把兩個橘圈自 extra_channels 移除、重繪 ERP，留在模式中續連。
 */
export function connectUnconnPair(){
  const hd = getHouseData();
  if(!hd){ return; }
  const top = nav.erpTopImg, bot = nav.erpBotImg;
  const exTop = extraAzList(top), exBot = extraAzList(bot);
  const res = computeUnconnEdge(top, bot, exTop, exBot);
  if(!res.ok){
    if(res.reason === 'same-or-invalid'){ toast('請先於兩個面板選定不同的兩張圖片'); }
    else { toast('兩張圖片要各放一個未連接通道圈圈（目前 上:' + res.topN + ' 下:' + res.botN + '）'); }
    return;
  }
  const { i, j, azI, azJ } = res;
  const azTop = exTop[0], azBot = exBot[0];

  confirmEdge(i, j, azI, azJ);          // 寫入 confirmed（含移除 rejected/eliminated、autoEliminate、存檔）
  // 兩個橘圈已變成綠色確認邊 -> 從 extra_channels 移除
  removeExtraAz(top, azTop);
  removeExtraAz(bot, azBot);

  // 讓 ERP 主標記顯示這條確認邊（curPair 指向這一對）
  const existing = hd.pairs.find(pr => pairKey(pr.i, pr.j) === pairKey(i, j));
  if(existing){ nav.curPair = existing; }
  else { nav.curPair = { i:i, j:j, prob:0, az_ij:azI, az_ji:azJ }; }

  scheduleSave();
  toast('已連接 ' + i + '–' + j + '（綠色確認邊）！可繼續連下一對');
  stateChanged();
  renderErpIfOn();
}

/**
 * 退出 unconn 模式時：把所有剩餘（未被連接）的橘圈全部刪除（含舊存檔載入者）。
 * 由 modes.setMode 在離開 unconn 時呼叫。
 */
export function clearAllUnconn(){
  const had = Object.keys(STATE.extra_channels).length > 0;
  STATE.extra_channels = {};
  if(had){
    scheduleSave();
    stateChanged();
    renderErpIfOn();
    if(getHouseData() && nav.curImage >= 0) buildMarkersForImage(nav.curImage, nav.curPair);
  }
}

/**
 * unconn 推薦面板「就是這個(連通)」。把某推薦候選確認連通。
 * 此側（放圈圖 srcImg）方位優先用該圖的橘圈（若有），否則用候選在此圖看向對方的方位 candAz；
 * 對側方位優先用對側橘圈（若使用者也在對面放了橘圈），否則用候選的對面方位 oppAz。
 * 成功後移除此側用掉的橘圈（若有）與對側用掉的橘圈（若有），寫入 confirmed（含 autoEliminate、去重、存檔）。
 * 留在 unconn 模式，caller 負責刷新推薦面板 / 重繪。
 * @param {number} srcImg 放圈的圖 index（此側）
 * @param {number} otherImg 候選對面圖 index（對側）
 * @param {number} candAz 候選在此圖看向對方的方位（recommendForCircle.az；作 srcImg 無橘圈時的後備）
 * @param {number} oppAz 候選的對面方位（recommendForCircle.oppAz；作對側無橘圈時的後備）
 * @returns {boolean} 是否成功建立確認邊
 */
export function confirmRecommendedCandidate(srcImg, otherImg, candAz, oppAz){
  const hd = getHouseData();
  if(!hd || srcImg == null || otherImg == null || srcImg === otherImg){ toast('無法連通：候選無效'); return false; }
  const exSrc = extraAzList(srcImg), exOther = extraAzList(otherImg);
  // 此側方位：橘圈優先，否則候選此圖方位
  const azSrc = (exSrc.length > 0) ? exSrc[0] : candAz;
  // 對側方位：對面橘圈優先，否則候選對面方位
  const azOther = (exOther.length > 0) ? exOther[0] : oppAz;
  if(azSrc == null || azOther == null){ toast('無法連通：缺少方位資訊'); return false; }
  // 方向對應：i=min 圖 -> az_i
  const i = Math.min(srcImg, otherImg), j = Math.max(srcImg, otherImg);
  const azI = (i === srcImg) ? azSrc : azOther;
  const azJ = (j === srcImg) ? azSrc : azOther;

  confirmEdge(i, j, azI, azJ);          // 寫入 confirmed（移除 rejected/eliminated、autoEliminate、存檔）
  // 移除用掉的橘圈（此側一定用；對側僅在有橘圈時用）
  if(exSrc.length > 0) removeExtraAz(srcImg, azSrc);
  if(exOther.length > 0) removeExtraAz(otherImg, azOther);

  // 讓 ERP 主標記顯示這條確認邊
  const existing = hd.pairs.find(pr => pairKey(pr.i, pr.j) === pairKey(i, j));
  nav.curPair = existing || { i, j, prob:0, az_ij:azI, az_ji:azJ };

  scheduleSave();
  toast('已連接 ' + i + '–' + j + '（綠色確認邊）！');
  stateChanged();
  renderErpIfOn();
  return true;
}

// =========================================================================
//  無 ROI：淘汰某方位聚類內的候選
// =========================================================================
/**
 * 無 ROI：淘汰 clickedImgIdx 上 markerAz 附近 CLUSTER_DELTA 度內所有 pending 候選，
 * 並自動把對側切換至下一個最接近的候選。沿用原 eliminateRoiPair 全部邏輯。
 */
export function eliminateRoiPair(clickedImgIdx, targetImgIdx, markerAz, whichPane){
  const hd = getHouseData();
  let p = hd.pairs.find(x => pairKey(x.i, x.j) === pairKey(clickedImgIdx, targetImgIdx));
  if(!p){ p = { i:Math.min(clickedImgIdx, targetImgIdx), j:Math.max(clickedImgIdx, targetImgIdx) }; }

  // 1) 收集該方位附近 CLUSTER_DELTA 內所有 pending
  const eliminatedPairs = [];
  hd.pairs.forEach(q => {
    if(pairStatus(q) !== 'pending') return;
    let qAz = null;
    if(q.i === clickedImgIdx) qAz = q.az_ij;
    else if(q.j === clickedImgIdx) qAz = q.az_ji;
    else return;
    if(angDiff(qAz, markerAz) < CLUSTER_DELTA){ eliminatedPairs.push(q); }
  });
  if(p && !eliminatedPairs.some(x => pairKey(x.i, x.j) === pairKey(p.i, p.j)) && pairStatus(p) === 'pending'){
    eliminatedPairs.push(p);
  }

  // 2) 淘汰
  let count = 0;
  eliminatedPairs.forEach(q => {
    if(!inList(STATE.eliminated, q.i, q.j)){ STATE.eliminated.push([q.i, q.j]); count++; }
    STATE.confirmed = STATE.confirmed.filter(x => pairKey(x.i, x.j) !== pairKey(q.i, q.j));
    STATE.rejected = STATE.rejected.filter(x => pairKey(x[0], x[1]) !== pairKey(q.i, q.j));
  });
  toast('無 ROI 標記：已淘汰該方位附近 ' + count + ' 個候選配對');

  // 3) 下一個最接近的 pending
  let minDiff = 1e9, bestNextPair = null;
  hd.pairs.forEach(q => {
    if(pairStatus(q) !== 'pending') return;
    let qAz = null;
    if(q.i === clickedImgIdx) qAz = q.az_ij;
    else if(q.j === clickedImgIdx) qAz = q.az_ji;
    else return;
    const diff = angDiff(qAz, markerAz);
    if(diff < minDiff){ minDiff = diff; bestNextPair = q; }
  });

  // 4) 跳轉
  if(bestNextPair){
    const otherImg = (bestNextPair.i === clickedImgIdx) ? bestNextPair.j : bestNextPair.i;
    if(whichPane === 'top'){ nav.erpTopImg = clickedImgIdx; nav.erpBotImg = otherImg; }
    else { nav.erpBotImg = clickedImgIdx; nav.erpTopImg = otherImg; }
    nav.curPair = bestNextPair;
    nav.curImage = clickedImgIdx;
    loadImageView(nav.curImage, null);
    toast('已將對側切換至最接近的下一個預測 candidate');
  } else {
    toast('img ' + clickedImgIdx + ' 無剩餘候選，已切換至下一個全域待處理配對');
    const next = firstPending();
    nav.curPair = next; nav.viewFrom = null; nav.reciprocalKey = null;
    if(next){ loadImageView(next.i, next); nav.erpTopImg = next.i; nav.erpBotImg = next.j; }
  }

  clearMode();
  scheduleSave();
  stateChanged();
  renderErpIfOn();
}

// =========================================================================
//  否決某 pair（模式點擊）
// =========================================================================
/**
 * 否決 clickedImgIdx–targetImgIdx 這一對，並自動切換對側至下一個最接近候選。
 * 沿用原 rejectMarkerPair 全部邏輯。
 */
export function rejectMarkerPair(clickedImgIdx, targetImgIdx, markerAz, whichPane){
  const hd = getHouseData();
  let p = hd.pairs.find(x => pairKey(x.i, x.j) === pairKey(clickedImgIdx, targetImgIdx));
  if(!p){ p = { i:Math.min(clickedImgIdx, targetImgIdx), j:Math.max(clickedImgIdx, targetImgIdx) }; }
  const originalStatus = pairStatus(p);

  // 1) 否決
  if(isConfirmed(p.i, p.j)){
    rejectConfirmed(p.i, p.j);
  } else {
    if(!inList(STATE.rejected, p.i, p.j)){ STATE.rejected.push([p.i, p.j]); }
    STATE.eliminated = STATE.eliminated.filter(x => pairKey(x[0], x[1]) !== pairKey(p.i, p.j));
  }
  toast('已否決 ' + p.i + '–' + p.j);

  if(originalStatus !== 'pending'){
    nav.curPair = p;
    clearMode();
    scheduleSave();
    stateChanged();
    renderErpIfOn();
    return;
  }

  // 2) 下一個最接近的 pending
  let minDiff = 1e9, bestNextPair = null;
  hd.pairs.forEach(q => {
    if(pairStatus(q) !== 'pending') return;
    if(pairKey(q.i, q.j) === pairKey(p.i, p.j)) return;
    let qAz = null;
    if(q.i === clickedImgIdx) qAz = q.az_ij;
    else if(q.j === clickedImgIdx) qAz = q.az_ji;
    else return;
    const diff = angDiff(qAz, markerAz);
    if(diff < minDiff){ minDiff = diff; bestNextPair = q; }
  });

  // 3) 跳轉
  if(bestNextPair){
    const otherImg = (bestNextPair.i === clickedImgIdx) ? bestNextPair.j : bestNextPair.i;
    if(whichPane === 'top' || clickedImgIdx === nav.erpTopImg){ nav.erpTopImg = clickedImgIdx; nav.erpBotImg = otherImg; }
    else if(whichPane === 'bot' || clickedImgIdx === nav.erpBotImg){ nav.erpBotImg = clickedImgIdx; nav.erpTopImg = otherImg; }
    else { nav.erpTopImg = bestNextPair.i; nav.erpBotImg = bestNextPair.j; }
    nav.curPair = bestNextPair;
    nav.curImage = clickedImgIdx;
    loadImageView(nav.curImage, null);
    toast('已將對側切換至最接近的下一個預測 candidate');
  } else {
    toast('img ' + clickedImgIdx + ' 無剩餘候選，已切換至下一個全域待處理配對');
    const next = firstPending();
    nav.curPair = next; nav.viewFrom = null; nav.reciprocalKey = null;
    if(next){ loadImageView(next.i, next); nav.erpTopImg = next.i; nav.erpBotImg = next.j; }
  }

  clearMode();
  scheduleSave();
  stateChanged();
  renderErpIfOn();
}

// =========================================================================
//  批量否決：一次否決一串 pair，單次 save + 單次 state-changed + toast 總數
// =========================================================================
/**
 * 批量否決 list 內的每一對。逐一寫入 STATE.rejected（去重、若曾確認則走 rejectConfirmed 救回同門），
 * 全部處理完只 scheduleSave 一次、emit('state-changed') 一次、toast 總數一次。
 * 不做導覽切換（面板留在原處，reject 模式維持，供繼續點下一個圈）。
 * @param {Array<{i:number,j:number}|[number,number]>} list 要否決的對（物件或 [i,j] 皆可）
 * @returns {number} 實際新否決的筆數
 */
export function rejectPairs(list){
  if(!Array.isArray(list) || list.length === 0){ return 0; }
  let n = 0;
  list.forEach(item => {
    const i = Array.isArray(item) ? item[0] : item.i;
    const j = Array.isArray(item) ? item[1] : item.j;
    if(i == null || j == null) return;
    if(isConfirmed(i, j)){
      // 曾確認者：撤銷確認並救回同門淘汰（rejectConfirmed 內含 scheduleSave/emit，
      // 但批量情境下我們最後統一再發一次；此處重複發射無害，訂閱者為冪等重繪）。
      rejectConfirmed(i, j);
      n++;
      return;
    }
    if(!inList(STATE.rejected, i, j)){ STATE.rejected.push([i, j]); n++; }
    // 若原本被淘汰，改為明確否決（移出 eliminated）
    STATE.eliminated = STATE.eliminated.filter(x => pairKey(x[0], x[1]) !== pairKey(i, j));
  });
  if(n > 0){
    scheduleSave();
    stateChanged();
    renderErpIfOn();
    if(getHouseData() && nav.curImage >= 0) buildMarkersForImage(nav.curImage, nav.curPair);
    toast('已批量否決 ' + n + ' 個候選配對');
  } else {
    toast('沒有可否決的候選（可能都已否決/淘汰）');
  }
  return n;
}

// =========================================================================
//  自動淘汰（確認後同門聚類收緊）
// =========================================================================
/**
 * 確認 p 的門 az* 後，淘汰同一門聚類（距 az* < CLUSTER_DELTA）內其它 pending；
 * 但該方位若有使用者橘圈（視為獨立門）則保留。以 nav.curImage 為基準圖。沿用原 autoEliminate。
 */
export function autoEliminate(p, azStar){
  const base = nav.curImage;
  const star = (typeof azStar === 'number') ? azStar : (base === p.i ? p.az_ij : p.az_ji);
  autoEliminateForImage(p, base, star);
}

/**
 * autoEliminate 的基準圖參數化版本（連接 helper 對兩端各跑一次）。
 * @param {object} p 確認邊/pair（需 i/j/az_ij/az_ji 或作為 pairKey 排除自身用）
 * @param {number} base 基準圖 index
 * @param {number} star 該圖上的確認門方位
 */
export function autoEliminateForImage(p, base, star){
  const hd = getHouseData();
  if(!hd || typeof star !== 'number' || base < 0) return;
  const ex = (STATE.extra_channels[base] || STATE.extra_channels[String(base)] || []);
  hd.pairs.forEach(q => {
    if(pairKey(q.i, q.j) === pairKey(p.i, p.j)) return;
    if(pairStatus(q) !== 'pending') return;
    let qaz = null;
    if(q.i === base) qaz = q.az_ij; else if(q.j === base) qaz = q.az_ji; else return;
    if(angDiff(qaz, star) < CLUSTER_DELTA){
      const kept = ex.some(a => angDiff(a, qaz) < CLUSTER_DELTA);
      if(!kept) STATE.eliminated.push([q.i, q.j]);
    }
  });
}

// =========================================================================
//  A：此圖全連接
// =========================================================================
/** A 鍵/按鈕：ERP 下切換全連接模式（點面板執行）；3D 下直接對目前圖執行。*/
export function actAllConn(){
  if(erpOn()){
    toggleAllConnMode();
  } else {
    if(nav.curImage >= 0){ performAllConnOnImage(nav.curImage); }
  }
}

/**
 * 把 targetImg 標記為全連接：淘汰所有含它的 pending。ERP 下自動把對側切到下一個。
 * 沿用原 performAllConnOnImage 全部邏輯。
 */
export function performAllConnOnImage(targetImg){
  if(targetImg < 0){ return; }
  const hd = getHouseData();
  if(STATE.complete_images.indexOf(targetImg) < 0) STATE.complete_images.push(targetImg);
  let n = 0;
  hd.pairs.forEach(q => {
    if(pairStatus(q) !== 'pending') return;
    if(q.i === targetImg || q.j === targetImg){ STATE.eliminated.push([q.i, q.j]); n++; }
  });
  toast('img ' + targetImg + ' 標記全連接，淘汰 ' + n + ' 個候選');
  scheduleSave();

  clearMode();

  let foundNext = false;
  const erpMode = erpOn();
  if(erpMode){
    const p = nav.curPair;
    const otherImg = (targetImg === nav.erpTopImg) ? nav.erpBotImg : nav.erpTopImg;
    const targetImgAz = (targetImg === p.i) ? p.az_ij : p.az_ji;
    let minDiff = 1e9, bestNextPair = null;
    hd.pairs.forEach(q => {
      if(pairStatus(q) !== 'pending') return;
      if(q.i === targetImg || q.j === targetImg) return;
      let qAz = null;
      if(q.i === otherImg) qAz = q.az_ij;
      else if(q.j === otherImg) qAz = q.az_ji;
      else return;
      const diff = angDiff(qAz, targetImgAz);
      if(diff < minDiff){ minDiff = diff; bestNextPair = q; }
    });
    if(bestNextPair){
      const otherImgOpposite = (bestNextPair.i === otherImg) ? bestNextPair.j : bestNextPair.i;
      if(otherImg === nav.erpTopImg){ nav.erpTopImg = otherImg; nav.erpBotImg = otherImgOpposite; }
      else { nav.erpBotImg = otherImg; nav.erpTopImg = otherImgOpposite; }
      nav.curPair = bestNextPair;
      nav.curImage = otherImg;
      loadImageView(nav.curImage, null);
      foundNext = true;
      toast('已將對側切換至最接近的下一個預測 candidate');
    }
  }

  if(!foundNext){
    if(nav.curPair && pairStatus(nav.curPair) !== 'pending'){
      advanceAfterResolve();
    } else {
      stateChanged();
      renderErpIfOn();
    }
  } else {
    stateChanged();
    renderErpIfOn();
  }
}

/** 解析完目前對後前進到下一個 pending（並 emit 重繪）。沿用原 advanceAfterResolve。*/
export function advanceAfterResolve(){
  scheduleSave();
  const next = firstPending();
  nav.curPair = next; nav.viewFrom = null; nav.reciprocalKey = null;
  if(next){ loadImageView(next.i, next); }
  stateChanged();
  renderErpIfOn();
}

// =========================================================================
//  重新否決一個已確認的連接（右鍵綠圈 / 對已確認對按 N）
// =========================================================================
function pairAzFrom(a, b, fromImg){
  const hd = getHouseData();
  const p = hd.pairs.find(x => pairKey(x.i, x.j) === pairKey(a, b));
  if(!p) return null;
  return (p.i === fromImg) ? p.az_ij : (p.j === fromImg ? p.az_ji : null);
}
/** 把當初因確認 c 而被同門淘汰的候選救回為 pending。沿用原 undoClusterEliminations。*/
export function undoClusterEliminations(c){
  STATE.eliminated = STATE.eliminated.filter(p => {
    const aI = pairAzFrom(p[0], p[1], c.i), aJ = pairAzFrom(p[0], p[1], c.j);
    const inI = (aI != null && angDiff(aI, c.az_i) < CLUSTER_DELTA);
    const inJ = (aJ != null && angDiff(aJ, c.az_j) < CLUSTER_DELTA);
    return !(inI || inJ);
  });
}
/** 把已確認對 i–j 改為否決，救回同門被淘汰者，重畫。沿用原 rejectConfirmed。*/
export function rejectConfirmed(i, j){
  const hd = getHouseData();
  const c = STATE.confirmed.find(x => pairKey(x.i, x.j) === pairKey(i, j));
  STATE.confirmed = STATE.confirmed.filter(x => pairKey(x.i, x.j) !== pairKey(i, j));
  if(c) undoClusterEliminations(c);
  STATE.eliminated = STATE.eliminated.filter(p => pairKey(p[0], p[1]) !== pairKey(i, j));
  if(!inList(STATE.rejected, i, j)) STATE.rejected.push([i, j]);
  if(nav.reciprocalKey === pairKey(i, j)) nav.reciprocalKey = null;
  scheduleSave();
  if(hd) buildMarkersForImage(nav.curImage, nav.curPair);
  stateChanged();
  renderErpIfOn();
  toast('已把 ' + i + '–' + j + ' 改為否決');
}

// =========================================================================
//  重置此房屋所有標註
// =========================================================================
/** 兩段確認後清空 STATE 並立即存檔、重畫、跳第一個 pending。沿用原 resetCurrentHouse。*/
export function resetCurrentHouse(){
  const hd = getHouseData();
  if(!getHouseName()){ toast('請先選擇房子'); return; }
  const c1 = confirm('⚠️ 警告：您確定要重置目前房屋「' + getHouseName() + '」的所有標註進度嗎？\n此動作將清除所有已確認、已否決、已淘汰的連通配對以及手動標記圈圈，且無法復原！');
  if(!c1) return;
  const c2 = confirm('⚠️ 請再次確認：您真的要刪除所有標註資料並重新開始嗎？');
  if(!c2) return;

  resetState();
  clearMode();
  emit('reset-filter');
  nav.erpTopImg = -1; nav.erpBotImg = -1;
  emit('erp-reset-az');
  cancelPendingSave();

  const thEl = document.getElementById('threshold');
  if(thEl) thEl.value = STATE.threshold;
  const thVal = document.getElementById('th-val');
  if(thVal) thVal.textContent = (+STATE.threshold).toFixed(2);
  emit('apply-ui-toggles');

  doSave();
  stateChanged();

  nav.curPair = firstPending(); nav.viewFrom = null; nav.reciprocalKey = null;
  if(nav.curPair){ loadImageView(nav.curPair.i, nav.curPair); }
  else if(hd && hd.pairs.length){ loadImageView(0, null); }
  else { clearMarkers(); }

  renderErpIfOn();
  toast('已成功重置目前房子的所有標註狀態！');
}
