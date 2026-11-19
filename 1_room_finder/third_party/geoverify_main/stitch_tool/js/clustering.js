/* =========================================================================
   clustering.js：門聚類（純函式，3D 與 ERP 共用）
   低層模組：只 import config / geometry。不觸碰 DOM，不 import 高層模組。
   依賴的 houseData / STATE / pairStatus 一律以參數注入，維持純函式與可測試性。
   演算法逐字沿用原 app.js（doorCandidates / circMean / clusterDoors）。
   ========================================================================= */

import { CLUSTER_DELTA } from './config.js';
import { angDiff } from './geometry.js';

/**
 * 目前圖視角下，所有含 imgIdx 且未否決/未淘汰的候選 = 一個「門候選」。
 * @param {number} imgIdx 目前圖 index
 * @param {object} houseData house 資料（含 pairs）
 * @param {(p:object)=>string} pairStatus 傳回 'confirmed'|'rejected'|'eliminated'|'pending'
 * @returns {Array<{az:number,target:number,prob:number,status:string,i:number,j:number}>}
 */
export function doorCandidates(imgIdx, houseData, pairStatus){
  const out = [];
  houseData.pairs.forEach(function(p){
    const st = pairStatus(p);
    if(st === 'rejected' || st === 'eliminated') return;   // 否決/淘汰的門不再顯示圈圈
    let az = null, tgt = null;
    if(p.i === imgIdx){ az = p.az_ij; tgt = p.j; }
    else if(p.j === imgIdx){ az = p.az_ji; tgt = p.i; }
    else return;
    out.push({ az:az, target:tgt, prob:p.prob, status:st, i:p.i, j:p.j });
  });
  return out;
}

/**
 * 一組成員方位的環狀平均（度 [0,360)）。
 * @param {Array<{az:number}>} members
 * @returns {number}
 */
export function circMean(members){
  let sx = 0, sy = 0;
  members.forEach(function(m){ const r = m.az * Math.PI / 180; sx += Math.cos(r); sy += Math.sin(r); });
  return (Math.atan2(sy, sx) * 180 / Math.PI + 360) % 360;
}

/**
 * 環狀單鏈聚類（CLUSTER_DELTA）：把同一實體門的多個目標併成一個聚類。
 * cluster.bestProb=成員最大 prob；status=有已確認成員→confirmed，否則 pending。
 * @param {number} imgIdx
 * @param {object} houseData
 * @param {object} STATE 需 STATE.confirmed
 * @param {(p:object)=>string} pairStatus
 * @param {(i:number,j:number)=>string} pairKey
 * @returns {Array<{members:Array,repAz:number,bestProb:number,status:string,repTarget:number}>}
 */
export function clusterDoors(imgIdx, houseData, STATE, pairStatus, pairKey){
  const cand = doorCandidates(imgIdx, houseData, pairStatus).slice()
    .sort(function(a, b){ return a.az - b.az; });
  const clusters = [];
  cand.forEach(function(c){
    let best = null, bd = 1e9;
    clusters.forEach(function(cl){
      const d = angDiff(c.az, cl.repAz);
      if(d < CLUSTER_DELTA && d < bd){ bd = d; best = cl; }
    });
    if(best){ best.members.push(c); best.repAz = circMean(best.members); }
    else clusters.push({ members:[c], repAz:c.az });
  });
  clusters.forEach(function(cl){
    let conf = null, bestProb = -1, tgt = null;
    cl.members.forEach(function(m){
      if(m.prob > bestProb){ bestProb = m.prob; tgt = m.target; }
      if(m.status === 'confirmed') conf = m;
    });
    cl.bestProb = bestProb; cl.status = conf ? 'confirmed' : 'pending'; cl.repTarget = tgt;
    if(conf){
      cl.repTarget = conf.target;
      const c = STATE.confirmed.find(function(x){ return pairKey(x.i, x.j) === pairKey(conf.i, conf.j); });
      if(c){ cl.repAz = (imgIdx === c.i) ? c.az_i : c.az_j; }
    }
  });
  return clusters;
}
