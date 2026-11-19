/* =========================================================================
   srcbar.js：全局來源/降級提示列。
   固定於畫面下方、body 層、高 z-index，**不被任何檢視模式覆蓋**（先前塞在 #info
   會被 crop-mode / erp-ctrl-top 蓋住，非 3D 模式看不到）。
   只在「非主路徑」時出現：correspondence 不足改用上游模型方位（紅）／門語意校正（中性）；
   主路徑（幾何驗證）不顯示，避免常駐雜訊。文字來源集中於 srchint.passageSourceHint。
   ========================================================================= */
import { deriveViewPair } from './viewmodes/pairsel.js';
import { passageSourceHint } from './srchint.js';
import { on } from './events.js';

function render(){
  const el = document.getElementById('src-hint-bar'); if(!el) return;
  let d = null;
  try { d = deriveViewPair(); } catch(_) { d = null; }
  if(!d || !d.pair){ el.style.display = 'none'; return; }
  const sem = !!(d.semantic && (d.semantic.i || d.semantic.j));
  const sh = passageSourceHint(d.pair, d.semantic);
  if(!sh.fallback && !sem){ el.style.display = 'none'; return; }   // 主路徑（幾何驗證）不顯示
  el.textContent = (sh.fallback ? '⚠ ' : '') + sh.text;
  el.style.background = sh.fallback ? '#e05a3a' : 'rgba(40,44,52,0.94)';
  el.style.color = '#fff';
  el.style.display = 'block';
}

export function initSrcBar(){
  on('view-changed', render);
  on('erp-derived', render);
  on('state-changed', render);
  render();
}
