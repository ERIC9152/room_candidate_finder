/* =========================================================================
   nav.js：導覽狀態與導覽入口
   共享可變狀態 nav：curImage / curPair / viewFrom / reciprocalKey / erpTopImg / erpBotImg。
   導覽入口：gotoImage / jumpToPair / gotoCurrentPair / peekToPair / firstPending / walkThroughDoor。
   實際貼圖/相機/標記交給 scene3d.loadImageView；本模組只決定「看哪張圖、哪一對」。
   依賴：state / scene3d / events。與 scene3d 互相 import，但只在函式體內使用，無評估期循環。
   ERP 於 initErp 訂閱 'view-changed' 自行重繪，故本模組只需 emit('view-changed')。
   ========================================================================= */

import { getHouseData, pairStatus, pairKey } from './state.js';
import { loadImageView, faceAz } from './scene3d.js';
import { emit } from './events.js';

/** 共享導覽狀態（就地變更）。*/
export const nav = {
  curImage:-1,     // 目前球面顯示的圖 index
  curPair:null,    // 正在複核的候選 {i,j,...}
  viewFrom:null,   // 穿門時來源圖
  reciprocalKey:null, // 穿門走過的已確認邊 key（其標記顯示藍色）
  erpTopImg:-1,    // ERP 上/左面板圖 index
  erpBotImg:-1     // ERP 下/右面板圖 index
};

/**
 * 佇列中第一個 pending 候選 = 目前 pair。
 * @returns {object|null}
 */
export function firstPending(){
  const hd = getHouseData();
  if(!hd) return null;
  for(let k = 0; k < hd.pairs.length; k++){
    if(pairStatus(hd.pairs[k]) === 'pending') return hd.pairs[k];
  }
  return null;
}

/**
 * 進入目前應複核的 pair（第一個 pending）；無 pending 則顯示第 0 圖。
 */
export function gotoCurrentPair(){
  const hd = getHouseData();
  const p = firstPending();
  if(!p){
    nav.curPair = null;
    if(hd && hd.pairs.length){ loadImageView(0, null); }
    emit('view-changed');
    return;
  }
  jumpToPair(p);
}

/**
 * 跳到某一對並進入其 i 圖視角（面向 az_ij）。
 * @param {object} p pair
 */
export function jumpToPair(p){
  if(!p) return;
  nav.curPair = p; nav.viewFrom = null; nav.reciprocalKey = null;
  nav.erpTopImg = p.i; nav.erpBotImg = p.j;
  loadImageView(p.i, p);
  emit('view-changed');
}

/**
 * 直接載入某圖（快速換圖等），可帶 pair。
 * @param {number} imgIdx
 * @param {object|null} pair
 */
export function gotoImage(imgIdx, pair){
  loadImageView(imgIdx, pair);
  emit('view-changed');
}

/**
 * 預覽跳到某一對的「對面圖」（確認前先過去看一眼）。
 * @param {object} pair
 * @param {number} targetImg 對面圖 index
 */
export function peekToPair(pair, targetImg, clickedWhich){
  if(!pair) return;
  nav.curPair = pair; nav.viewFrom = nav.curImage;
  nav.reciprocalKey = pairKey(pair.i, pair.j);
  if(clickedWhich === 'top'){
    const srcImg = (pair.i === targetImg) ? pair.j : pair.i;
    nav.erpTopImg = srcImg; nav.erpBotImg = targetImg;
  } else if(clickedWhich === 'bot'){
    const srcImg = (pair.i === targetImg) ? pair.j : pair.i;
    nav.erpTopImg = targetImg; nav.erpBotImg = srcImg;
  } else {
    nav.erpTopImg = pair.i; nav.erpBotImg = pair.j;
  }
  loadImageView(targetImg, pair);   // 自動面向該圖指回來源的候選門
  emit('view-changed');
}

/**
 * 穿過一扇已確認的門走到對面圖，並面向反向圈（顯示藍色）。
 * @param {object} c STATE.confirmed 內的確認邊
 * @param {number} tgt 目標圖 index
 * @param {string} [clickedWhich] 'top'|'bot'
 */
export function walkThroughDoor(c, tgt, clickedWhich){
  if(!c) return;
  nav.viewFrom = nav.curImage; nav.curPair = null;
  nav.reciprocalKey = pairKey(c.i, c.j);
  if(clickedWhich === 'top'){
    const srcImg = (c.i === tgt) ? c.j : c.i;
    nav.erpTopImg = srcImg; nav.erpBotImg = tgt;
  } else if(clickedWhich === 'bot'){
    const srcImg = (c.i === tgt) ? c.j : c.i;
    nav.erpTopImg = tgt; nav.erpBotImg = srcImg;
  } else {
    nav.erpTopImg = c.i; nav.erpBotImg = c.j;
  }
  loadImageView(tgt, null);          // 該邊在對面圖被塗藍
  const az = (c.i === tgt) ? c.az_i : c.az_j;
  faceAz(az);
  emit('view-changed');
}
