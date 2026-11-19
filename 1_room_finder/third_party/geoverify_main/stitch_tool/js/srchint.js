/* =========================================================================
   srchint.js：來源/降級透明化。
   通則：任何時候不是走主路徑（足夠 correspondence → 幾何驗證），就要在畫面
   明白說出改用了哪套替代方案、為什麼。此處集中定義提示文字，各檢視模式與
   狀態列一致調用，新增 fallback 時只改這裡。
   低層 helper：不 import 任何其它模組。
   ========================================================================= */

/**
 * 通道方位的來源提示。
 * @param {object} pair 具 az_source（'geo'|'coarse'）的配對
 * @param {{i:boolean,j:boolean}} [semantic] 門語意校正是否觸發（任一側）
 * @returns {{text:string, fallback:boolean}} 提示文字；fallback=true 代表非主路徑、需標示
 */
export function passageSourceHint(pair, semantic){
  if(semantic && (semantic.i || semantic.j))
    return { text: '通道方位：門語意校正（門中心）', fallback: false };
  const src = pair && pair.az_source;
  if(src === 'geo')
    return { text: '通道方位：幾何驗證', fallback: false };
  if(src === 'coarse')
    return { text: 'correspondence 不足，通道方位改用上游模型預測（非幾何驗證）', fallback: true };
  return { text: '', fallback: false };
}
