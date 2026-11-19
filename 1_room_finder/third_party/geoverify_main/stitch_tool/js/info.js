/* =========================================================================
   info.js：中央資訊面板 / 狀態列（目前對 / 穿門檢視 / 通道方位角度 / 無配對提示）
   ERP/3D 無衍生配對時，「目前沒有配對」提示改顯示於此狀態列（#i-pair），
     不再取代面板內容；面板兩張圖照常顯示、可換圖、可放圈。
   訂閱 'view-changed' / 'state-changed' / 'erp-derived' 自動更新。
   依賴：state / nav / erp(isErpOn,getErpDerived) / scene3d(getCandidateMarker) / util / events。
   ========================================================================= */

import { pairStatus, getHouseData } from './state.js';
import { nav } from './nav.js';
import { isErpOn, getErpDerived, getSemanticInfo } from './erp.js';
import { getCandidateMarker } from './scene3d.js';
import { statusLabel } from './util.js';
import { passageSourceHint } from './srchint.js';
import { on } from './events.js';

/** 依目前導覽狀態更新資訊面板文字。*/
export function refreshInfoPair(){
  const el = document.getElementById('i-pair'); if(!el) return;
  const hd = getHouseData();

  // ERP 模式：狀態列反映「面板衍生配對」語意（唯一真相在 erp._derived）。
  if(isErpOn() && hd){
    const d = getErpDerived();
    if(d.kind === 'none' || !d.pair){
      el.innerHTML = '<span style="color:var(--muted)">目前兩張圖不是候選配對（連通/否決已停用；可用箭頭或路線圖換圖、或放未連接通道圈）</span>';
      return;
    }
    const p = d.pair;
    const st = (d.kind === 'confirmed') ? 'confirmed' : pairStatus(p);
    const probTxt = (p.prob != null && p.prob > 0) ? (' · prob ' + (p.prob * 100).toFixed(1) + '%') : '';
    // 通道方位來源/降級提示（見 srchint.js）：門語意觸發 > az_source。
    const si = getSemanticInfo();
    const semFlags = si ? { i: !!(si.i && si.i.triggered), j: !!(si.j && si.j.triggered) } : null;
    const sh = passageSourceHint(p, semFlags);
    let azSrcTxt = sh.text
      ? (' · ' + (sh.fallback ? '<span style="color:#e05a3a">⚠ ' + sh.text + '</span>' : sh.text))
      : '';
    if(si && si.noSemanticForHouse){
      azSrcTxt += ' · <span style="color:var(--muted)">此房無語意圖</span>';
    }
    const manual = (d.kind === 'confirmed' && p._manual) ? '（人工邊）' : '';
    el.innerHTML = '目前對 <b>' + p.i + '–' + p.j + '</b>' + probTxt + azSrcTxt + ' · ' + statusLabel(st) + manual;
    return;
  }

  // 3D 模式
  const curInImg = nav.curPair && (nav.curPair.i === nav.curImage || nav.curPair.j === nav.curImage);
  if(nav.curPair && curInImg){
    const st = pairStatus(nav.curPair);
    const mk = getCandidateMarker();
    const azTxt = mk ? mk.userData.az.toFixed(1) + '°' : '';
    const sh = passageSourceHint(nav.curPair, null);
    const azSrcTxt = sh.text
      ? (' · ' + (sh.fallback ? '<span style="color:#e05a3a">⚠ ' + sh.text + '</span>' : sh.text))
      : '';
    const probTxt = (nav.curPair.prob != null) ? (nav.curPair.prob * 100).toFixed(1) + '%' : '—';
    el.innerHTML = '目前對 <b>' + nav.curPair.i + '–' + nav.curPair.j + '</b> · prob ' +
      probTxt + ' · ' + statusLabel(st) + (azTxt ? (' · 通道方位 ' + azTxt + azSrcTxt) : '');
  } else if(nav.reciprocalKey != null){
    el.innerHTML = '穿門檢視 img ' + nav.curImage + '（點藍圈返回；下一對 ' +
      (nav.curPair ? '<b>' + nav.curPair.i + '–' + nav.curPair.j + '</b>' : '—') + ' 按 Esc 前往）';
  } else if(nav.curPair){
    el.innerHTML = '已確認，下一對 <b>' + nav.curPair.i + '–' + nav.curPair.j + '</b>（點綠圈穿門，或按 Esc 前往）';
  } else if(hd){
    // 3D 中無配對（孤立照片/穿門後）：狀態列提示，Y/N 停用（見 ui.refreshActionAvailability）。
    el.innerHTML = '<span style="color:var(--muted)">img ' + nav.curImage +
      '：目前無候選配對（連通/否決已停用；用箭頭或路線圖換圖）</span>';
  } else {
    el.textContent = '';
  }
}

/** 訂閱事件：視角/狀態/ERP 衍生改變時刷新。*/
export function initInfo(){
  on('view-changed', refreshInfoPair);
  on('state-changed', refreshInfoPair);
  on('erp-derived', refreshInfoPair);
}
