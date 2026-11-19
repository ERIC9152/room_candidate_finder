/* =========================================================================
   pairing.js：ERP 面板與「衍生配對」的單一真相邏輯（結構解耦）
   低層純邏輯模組：只 import state（讀 pairs/confirmed/pairKey/pairStatus）。
   不 import 高層（erp/scene3d/dispatch/ui），也不觸碰 DOM，維持可單元測試。

   兩個核心：
     ensureErpPanes(nav, hd)：保證 nav.erpTopImg/erpBotImg 永遠有效且互異。
                                  進 ERP / 換圖前呼叫；無 curPair 也能初始化。
     derivePairForPanes(top,bot,hd,STATE)：由「面板正在顯示的兩張圖」衍生出目前配對語意：
       { kind:'pair',      pair }      兩圖為 hd.pairs 中一對   -> 主標記可拖曳、Y/N 可用
       { kind:'confirmed', pair }      兩圖為 STATE.confirmed 一邊（含 pairs 沒有的人工邊）
                                        -> 綠色主標記、Y 停用（已確認）、N=撤銷確認
       { kind:'none',      pair:null } 都不是 -> 不畫主標記、Y/N 停用

   說明：kind==='confirmed' 的 pair 為衍生 shim（帶 az_ij/az_ji 取自確認角），
        方便 renderERP/drawErpMarkers 沿用既有「以 curPair 取主標記方位」路徑。
   ========================================================================= */

import { pairKey } from './state.js';

/**
 * 找出「含 imgIdx 的相鄰圖」以初始化 ERP 另一面板：
 * 優先任一含 imgIdx 的 pair 的對側圖；否則回退 (imgIdx+1) 或 0，確保與 imgIdx 不同。
 * @param {number} imgIdx
 * @param {object|null} hd house data（含 images/pairs）
 * @returns {number} 一個與 imgIdx 不同的合法圖 index（N<=1 時回 imgIdx 本身）
 */
export function neighborImage(imgIdx, hd){
  const N = hd && hd.images ? hd.images.length : 0;
  if(N <= 1) return imgIdx;
  if(hd && hd.pairs){
    const p = hd.pairs.find(x => x.i === imgIdx || x.j === imgIdx);
    if(p){ return (p.i === imgIdx) ? p.j : p.i; }
  }
  const cand = (imgIdx + 1) % N;
  return (cand === imgIdx) ? ((imgIdx + 2) % N) : cand;
}

/**
 * 保證 nav.erpTopImg / erpBotImg 為有效且互異的圖 index（ERP 顯示唯一真相）。
 * 依序嘗試：既有合法且互異 -> 保留；否則由 curPair 初始化；再否則由 curImage + 相鄰圖；
 * 最後退回 0/1。就地變更 nav；回傳 {top,bot}。
 * @param {object} nav 導覽狀態（就地變更 erpTopImg/erpBotImg）
 * @param {object|null} hd house data
 * @returns {{top:number, bot:number}}
 */
export function ensureErpPanes(nav, hd){
  const N = hd && hd.images ? hd.images.length : 0;
  const valid = (x) => typeof x === 'number' && x >= 0 && x < N;

  let top = nav.erpTopImg, bot = nav.erpBotImg;

  // 1) 既有兩者皆合法且互異 -> 直接沿用
  if(valid(top) && valid(bot) && top !== bot){
    nav.erpTopImg = top; nav.erpBotImg = bot;
    return { top, bot };
  }

  // 2) 由 curPair 初始化
  const p = nav.curPair;
  if(p && valid(p.i) && valid(p.j) && p.i !== p.j){
    nav.erpTopImg = p.i; nav.erpBotImg = p.j;
    return { top:p.i, bot:p.j };
  }

  // 3) 由 curImage（或既有合法的一側）+ 相鄰圖
  let base = valid(top) ? top : (valid(bot) ? bot : (valid(nav.curImage) ? nav.curImage : 0));
  if(N <= 1){ nav.erpTopImg = 0; nav.erpBotImg = 0; return { top:0, bot:0 }; }
  const other = neighborImage(base, hd);
  top = base; bot = other;
  if(top === bot){ bot = (top + 1) % N; }         // 極端保險

  nav.erpTopImg = top; nav.erpBotImg = bot;
  return { top, bot };
}

/**
 * 由目前兩面板圖衍生「配對語意」。ERP 與 curPair 徹底解耦後的唯一配對來源。
 * @param {number} top 上/左面板圖 index
 * @param {number} bot 下/右面板圖 index
 * @param {object|null} hd house data（含 pairs）
 * @param {object} STATE 需 STATE.confirmed
 * @returns {{kind:'pair'|'confirmed'|'none', pair:(object|null)}}
 */
export function derivePairForPanes(top, bot, hd, STATE){
  if(top == null || bot == null || top < 0 || bot < 0 || top === bot){
    return { kind:'none', pair:null };
  }
  // 1) hd.pairs 中的候選對（含已被否決/淘汰者，仍算「pair」語意，狀態由 pairStatus 決定）
  if(hd && hd.pairs){
    const existing = hd.pairs.find(pr => pairKey(pr.i, pr.j) === pairKey(top, bot));
    if(existing){ return { kind:'pair', pair:existing }; }
  }
  // 2) STATE.confirmed 中的人工/孤立邊（hd.pairs 沒有）：衍生 shim，帶確認角當 az_ij/az_ji
  const cc = STATE && STATE.confirmed
    ? STATE.confirmed.find(x => pairKey(x.i, x.j) === pairKey(top, bot)) : null;
  if(cc){
    // shim 以 min/max 正規化 i<j，az_ij/az_ji 對應到 top/bot 的確認角
    const i = Math.min(top, bot), j = Math.max(top, bot);
    const az_ij = (i === cc.i) ? cc.az_i : cc.az_j;
    const az_ji = (j === cc.i) ? cc.az_i : cc.az_j;
    return { kind:'confirmed', pair:{ i, j, prob:0, az_ij, az_ji, _manual:true } };
  }
  // 3) 都不是 -> 無配對（UI 不得因此癱瘓；Y/N 停用即可）
  return { kind:'none', pair:null };
}
