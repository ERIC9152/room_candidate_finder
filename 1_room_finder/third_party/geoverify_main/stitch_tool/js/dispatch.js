/* =========================================================================
   dispatch.js：唯一的圈圈點擊分派入口 handleCircleClick(targetInfo)
   3D（scene3d.onPointerUp）與 ERP（erp dot onpointerdown）都呼叫它，收斂兩套點擊路徑。

   targetInfo 欄位：
     role     : 'confirmed'|'reciprocal'|'candidate'|'refdoor'|'extra'|'gt'
     az       : 圈圈方位角
     i, j     : （confirmed/reciprocal）該確認邊兩端；（candidate）該對兩端
     targetImg: （candidate/refdoor/confirmed）此圈指向的對面圖
     srcImg   : 點擊發生的圖 index（3D=curImage；ERP=該面板圖）
     source   : '3d' | 'erp'
     which    : （ERP）'top'|'bot'，供對側自動切換用

   模式分派（依 modes.getMode）：
     reject  -> rejectMarkerPair
     noRoi   -> eliminateRoiPair
     allConn -> ERP 下對面板圖 performAllConnOnImage；3D 下不由點擊觸發（A 直接執行）
     none    -> 依 role 做穿門 / peek / 轉向
   ERP 沿用原限制：reject/noRoi 只對 confirmed/refdoor 生效，否則提示。
   3D ：reject 模式點任何圈 -> 否決該 pair；noRoi 模式點圈 -> 淘汰該方位。
   ========================================================================= */

import { STATE, getHouseData, pairKey } from './state.js';
import { angDiff } from './geometry.js';
import { nav, peekToPair, walkThroughDoor } from './nav.js';
import { getMode } from './modes.js';
import {
  eliminateRoiPair, performAllConnOnImage
} from './actions.js';
import { openBatchReject } from './batch-panel.js';
import { loadImageView, faceAz } from './scene3d.js';
import { renderERP, isErpOn } from './erp.js';
import { toast } from './util.js';

/**
 * 由 role 推得「要否決/淘汰的 (clickedImg, targetImg, az)」。
 * @returns {{clickedImg:number,targetImg:number,az:number}|null}
 */
function resolvePairTarget(t){
  const src = (typeof t.srcImg === 'number') ? t.srcImg : nav.curImage;
  if(t.role === 'candidate'){
    if(!nav.curPair) return null;
    const other = (src === nav.curPair.i) ? nav.curPair.j : nav.curPair.i;
    return { clickedImg:src, targetImg:other, az:t.az };
  }
  if(t.role === 'confirmed' || t.role === 'reciprocal'){
    const other = (t.i === src) ? t.j : (t.j === src ? t.i : t.targetImg);
    return { clickedImg:src, targetImg:other, az:t.az };
  }
  if(t.role === 'refdoor'){
    return { clickedImg:src, targetImg:t.targetImg, az:t.az };
  }
  return null;
}

/**
 * 唯一分派入口。
 * @param {object} t targetInfo（見檔頭）
 */
export function handleCircleClick(t){
  const mode = getMode();
  const which = t.which;

  // ---- 模式分派 ----
  if(mode === 'allConn'){
    // ERP：點面板/圈 -> 該面板圖全連接；3D：A 直接執行，不經此
    if(t.source === 'erp' && typeof t.srcImg === 'number'){ performAllConnOnImage(t.srcImg); }
    return;
  }
  if(mode === 'reject'){
    // reject 模式點任何圈 -> 以該圈方位所在聚類，彈出「同方向候選」批量否決面板（預設全勾）。
    //          面板固定於螢幕右側，不跟隨圈圈；確認後批量 reject、reject 模式維持。
    //          只有一筆時也走同一面板（一致性）。原單發 rejectMarkerPair 保留為 helper 供內部用。
    const src = (typeof t.srcImg === 'number') ? t.srcImg : nav.curImage;
    if(typeof src !== 'number' || src < 0 || typeof t.az !== 'number'){
      toast('此圈無對應方位，無法批量否決');
      return;
    }
    // gt 等非候選視覺圈本就不可點（3D 已擋 gt/rejected-view）；此處統一以「圈方位」聚類。
    openBatchReject(src, t.az, which);
    return;
  }
  if(mode === 'noRoi'){
    if(t.source === 'erp'){
      if(t.role === 'confirmed' || t.role === 'refdoor'){
        eliminateRoiPair(t.srcImg, t.targetImg, t.az, which);
      } else if(t.role === 'candidate'){
        const pi = (t.i != null) ? t.i : (nav.curPair && nav.curPair.i);
        const pj = (t.j != null) ? t.j : (nav.curPair && nav.curPair.j);
        if(pi == null || pj == null){ toast('目前兩張圖不是候選配對，無法執行無 ROI'); return; }
        const other = (t.srcImg === pi) ? pj : pi;
        eliminateRoiPair(t.srcImg, other, t.az, which);
      } else {
        toast('您點擊的是非連通/非預測標記，無法執行無 ROI。請點擊黃圈或淡黃圈！');
      }
    } else {
      // 3D：點圈 -> 淘汰該方位附近候選
      const tg = resolvePairTarget(t);
      if(tg){ eliminateRoiPair(tg.clickedImg, tg.targetImg, tg.az, which); }
      else { toast('此圈無對應方位，無法執行無 ROI'); }
    }
    return;
  }

  // ---- mode === 'none'：依 role 做導覽 ----
  dispatchNavByRole(t);
}

/**
 * mode='none' 時依圈圈 role 導覽（穿門 / peek / 轉向）。
 * 沿用原 handleMarkerClick 的分支行為，並保留 ERP 情境下的對側面板切換。
 * @param {object} t targetInfo
 */
function dispatchNavByRole(t){
  const hd = getHouseData();
  const role = t.role;

  if(role === 'confirmed' || role === 'reciprocal'){
    const c = STATE.confirmed.find(x => pairKey(x.i, x.j) === pairKey(t.i, t.j));
    if(!c) return;
    walkThroughDoor(c, t.targetImg, t.which);
    if(isErpOn()) renderERP();
    toast('穿過門 → img ' + t.targetImg);

  } else if(role === 'candidate'){
    // 點候選圈 -> 跳到對面那張圖確認；對面有相同候選圈可再點跳回
    if(nav.curPair){
      const oth = (nav.curImage === nav.curPair.i) ? nav.curPair.j : nav.curPair.i;
      peekToPair(nav.curPair, oth, t.which);
      if(isErpOn()) renderERP();
      toast('跳到對面 img ' + oth + '（Y 連通 / N 否決；再點圈可跳回）');
    }

  } else if(role === 'refdoor'){
    // 淡琥珀預測門 -> 跳到它預測連到的那張圖確認
    const src = Number(t.i != null ? t.i : t.srcImg);
    const target = Number(t.targetImg);
    if(!isNaN(src) && !isNaN(target)){
      const tp = hd.pairs.find(p => pairKey(p.i, p.j) === pairKey(src, target));
      if(tp){
        peekToPair(tp, target, t.which);
        if(isErpOn()) renderERP();
        toast('跳到預測目標 img ' + target + '（prob ' + (tp.prob * 100).toFixed(0) + '%；Y 連通 / N 否決）');
      }
    }

  } else if(role === 'extra'){
    // 橘色未連接圈 -> 找此方向機率最高的預測目標跳過去確認
    const srcImg = (typeof t.i === 'number') ? t.i : nav.curImage;
    let best = null, bd = 1e9;
    hd.pairs.forEach(p => {
      const az = (p.i === srcImg) ? p.az_ij : (p.j === srcImg ? p.az_ji : null);
      if(az == null) return;
      const dd = angDiff(az, t.az);
      if(dd < bd){ bd = dd; best = p; }
    });
    if(best && bd < 25){
      const oth2 = (best.i === srcImg) ? best.j : best.i;
      if(isErpOn()){
        nav.curPair = best;
        if(srcImg === nav.erpTopImg){ nav.erpBotImg = oth2; }
        else if(srcImg === nav.erpBotImg){ nav.erpTopImg = oth2; }
        else { nav.erpTopImg = srcImg; nav.erpBotImg = oth2; }
        nav.curImage = srcImg;
        loadImageView(srcImg, null);
        toast('此方向最可能連到 img ' + oth2 + '（prob ' + (best.prob * 100).toFixed(0) + '%；已載入至對側面板）');
        setTimeout(renderERP, 50);
      } else {
        peekToPair(best, oth2);
        toast('此方向最可能連到 img ' + oth2 + '（prob ' + (best.prob * 100).toFixed(0) + '%；Y/N；再點跳回）');
      }
    } else {
      toast('此方向沒有模型預測目標，請點擊左右箭頭自行切換對側圖片');
    }

  } else if(role === 'rejected-view'){
    // 已否決/淘汰配對的紅圈：點擊只提示，不做導覽（可用 Y 重新連通）。
    toast('此配對已否決/淘汰（按 Y 可重新確認連通）');
  }
}
