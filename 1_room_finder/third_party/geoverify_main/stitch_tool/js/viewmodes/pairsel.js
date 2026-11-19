/* =========================================================================
   viewmodes/pairsel.js：檢視模式共用的「目前配對」推導(非插件,純 helper)
   mode-crop / mode-morph 共用;插件之間不得互相 import(否則停用一個插件的
   import 行會被另一個插件的依賴拖回來,違反可插拔性)。
   邏輯與 erp.renderERP 同款:優先 nav.curPair,否則用 ERP 面板推導;
   已確認配對用校正後方位;未確認且語意輔助開啟時套用門語意校正。

   方位優先序(三個檢視模式與 ERP 一致):
     人工 ROI 校正(STATE.confirmed) > 語意門校正 > az_ij/az_ji(幾何 az_ab)
   ========================================================================= */

import { STATE, getHouseData, getHouseName, pairKey } from '../state.js';
import { nav } from '../nav.js';
import { ensureErpPanes, derivePairForPanes } from '../pairing.js';
import { getPairOverride } from '../semantic.js';
import { emit } from '../events.js';

/**
 * @returns {{pair:object, azI:number, azJ:number, confirmed:boolean,
 *            semantic:{i:boolean,j:boolean}}|null}
 *   azI = 在 i 圖看向 j 的方位;azJ = 在 j 圖看向 i(已確認取校正角)。
 *   semantic.i/j = 該側方位是否來自語意門校正(供標籤標示)。
 */
export function deriveViewPair(){
  const hd = getHouseData();
  if(!hd) return null;
  let pair = nav.curPair;
  if(!pair){
    const { top, bot } = ensureErpPanes(nav, hd);
    pair = derivePairForPanes(top, bot, hd, STATE).pair;
  }
  if(!pair) return null;
  const cc = STATE.confirmed.find(x => pairKey(x.i, x.j) === pairKey(pair.i, pair.j));
  if(cc){
    return { pair, azI: cc.az_i, azJ: cc.az_j, confirmed: true, semantic:{ i:false, j:false } };
  }
  let azI = pair.az_ij, azJ = pair.az_ji;
  const semantic = { i:false, j:false };
  if(STATE.ui && STATE.ui.semanticAssist){
    // 資料(房子門清單/該對共視區域)尚未抓到時 getPairOverride 回 null,本次先用原值;
    // 抓完的 onReady 發 'view-changed',registry 會呼叫目前檢視模式的 refresh() 重畫一次。
    const ov = getPairOverride(getHouseName(), hd, pair, () => emit('view-changed'));
    if(ov && !ov.noSemanticForHouse){
      if(ov.i && ov.i.triggered){ azI = ov.i.az; semantic.i = true; }
      if(ov.j && ov.j.triggered){ azJ = ov.j.az; semantic.j = true; }
    }
  }
  return { pair, azI, azJ, confirmed:false, semantic };
}
