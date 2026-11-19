/* =========================================================================
   recommend.js：未連接通道「放圈推薦候選」純函式（可 node 測試）
   低層模組：只 import geometry(angDiff) 與 state(pairStatus)。不觸碰 DOM。
   使用者在某圖放橘圈（方位 az）後，推薦「此圖看向對方的方位」與圈方位差
     < 角度窗（RECO_WINDOW，預設 25°）的所有 pending pair，依 prob 由高到低排序，
     供推薦面板瀏覽 / 連通 / 否決。
   ========================================================================= */

import { angDiff } from './geometry.js';

/** 推薦匹配的角度窗（度）。圈方位與「此圖看向對方的方位」差小於此值才視為候選。*/
export const RECO_WINDOW = 25;

/**
 * 為某圖的一個橘圈方位推薦候選 pending pair。
 * @param {number} imgIdx 放圈的圖 index
 * @param {number} az 橘圈方位角（度）
 * @param {object} hd house data（需 hd.pairs）
 * @param {object} STATE 需 STATE.confirmed/rejected/eliminated（供 statusFn 判定）
 * @param {(p:object)=>string} statusFn pairStatus 函式（注入以維持純函式；回 'pending' 等）
 * @param {number} [windowDeg=RECO_WINDOW] 角度窗（度）
 * @returns {Array<{i:number,j:number,otherImg:number,prob:number,az:number,oppAz:number,dAz:number}>}
 *   otherImg=對面圖；az=此圖看向對方的方位；oppAz=對面圖看回來的方位（供對面面板畫預覽圈）；
 *   dAz=與橘圈方位的夾角；依 prob 由高到低排序（prob 相同時 dAz 小者優先）。
 */
export function recommendForCircle(imgIdx, az, hd, STATE, statusFn, windowDeg){
  const win = (typeof windowDeg === 'number') ? windowDeg : RECO_WINDOW;
  if(!hd || !hd.pairs || typeof imgIdx !== 'number' || typeof az !== 'number') return [];
  const out = [];
  hd.pairs.forEach(p => {
    // 只收 pending：排除 confirmed/rejected/eliminated
    if(statusFn(p) !== 'pending') return;
    // 此圖看向對方的方位；不含 imgIdx 者跳過
    let myAz = null, otherImg = null, oppAz = null;
    if(p.i === imgIdx){ myAz = p.az_ij; otherImg = p.j; oppAz = p.az_ji; }
    else if(p.j === imgIdx){ myAz = p.az_ji; otherImg = p.i; oppAz = p.az_ij; }
    else return;
    if(myAz == null) return;
    const d = angDiff(myAz, az);
    if(d >= win) return;                          // 角度窗過濾
    out.push({
      i:p.i, j:p.j, otherImg, prob:(p.prob != null ? p.prob : 0),
      az:myAz, oppAz:(oppAz != null ? oppAz : null), dAz:d
    });
  });
  // prob 由高到低；prob 相同時，夾角小者優先
  out.sort((a, b) => (b.prob - a.prob) || (a.dAz - b.dAz));
  return out;
}
