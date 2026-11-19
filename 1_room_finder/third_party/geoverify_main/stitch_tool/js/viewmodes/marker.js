/* =========================================================================
   viewmodes/marker.js：連通道「頂帶」標示(方位角-only)
   低層模組:只 import gnomonic(投影約定單一來源)、不碰 DOM(只吃 ctx)。

   設計約束:
     ① 不可像匹配點(correspondence) → 不畫點、不畫連線
     ② 不可有方向性箭頭(會被讀成「往那邊走」) → 不畫箭頭
     ③ 不可與共識區域遮罩相似 → 不做大面積半透明填色、不用 REG_PALETTE 色盤;
        改用中性白 + 深色底襯(亮圖上也看得見)
     ④ 未來通道方位可能來自上游模型,標示必須「只靠一個方位角」就能畫。所以本模組的介面只吃 az(+ 投影參數),沒有垂直範圍、沒有 polygon、
        沒有 ROI 框。舊的 ROI「框」需要垂直範圍,方位角給不出來,故改頂帶。

   視覺語言:畫布最上緣一條窄橫帶,中心對齊通道方位、角寬 ±HALF_DEG,
   往下 3 條短刻度(中央長、兩端短)+ 小標籤「通道」。
   ========================================================================= */

import { azToDir, dirToCropRaw } from './gnomonic.js';

const HALF_DEG = 15;                 // 帶子角半寬(度);呼叫端可用 opts.halfDeg 覆寫
const BAND_FRAC = 0.055;             // 帶高 = 畫布高 × 此值(規格 4–6%)
const BAND_MIN_PX = 20;
const INK = '#ffffff';               // 中性白。與區域色盤刻意無交集
const BACK = 'rgba(8,12,18,.72)';    // 深色底襯
const LABEL = '通道';

/** az → 該畫布的水平位置;背面回 null。viewLat = 鏡頭俯仰(帶子仍畫在畫布頂緣)。*/
function projX(azDeg, viewAz, fovDeg, size, viewLat = 0){
  const p = dirToCropRaw(azToDir(azDeg), viewAz, fovDeg, size, viewLat);   // 標的本身在水平線上
  return p ? p[0] : null;
}

/** ERP 的 u 座標約定(同 SPEC.md / gnomonic.js):u = ((az+180) mod 360)/360。*/
function erpU(azDeg){ return (((azDeg % 360) + 540) % 360) / 360; }

/**
 * 實際畫一條帶(座標已算好)。xC 落在畫布外時不畫標籤。
 * 過縫拆成兩半時,只有含中心的那半該掛字。
 */
function paintBand(ctx, xL, xC, xR, W, H){
  const bh = Math.max(BAND_MIN_PX, Math.round(H * BAND_FRAC));
  const l = Math.max(-8, xL), r = Math.min(W + 8, xR);
  if(r <= l) return;
  ctx.save();
  ctx.fillStyle = BACK;
  ctx.fillRect(l - 6, 0, (r - l) + 12, bh);
  ctx.fillStyle = INK;
  ctx.fillRect(l, 3, Math.max(2, r - l), 4);
  ctx.strokeStyle = INK;
  ctx.lineWidth = 2;
  [[l, 6], [xC, 10], [r, 6]].forEach(([x, len]) => {
    if(x < l - 1 || x > r + 1) return;
    const px = Math.round(x) + 0.5;
    ctx.beginPath();
    ctx.moveTo(px, 7);
    ctx.lineTo(px, 7 + len);
    ctx.stroke();
  });
  if(xC >= 0 && xC <= W){
    ctx.fillStyle = INK;
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(LABEL, Math.max(20, Math.min(W - 20, xC)), bh - 4);
  }
  ctx.restore();
}

/**
 * ERP 畫布版(regions 模式):帶子畫在該圖頂緣 u(az)±Δ。
 * 過縫平移 −W/0/+W 各畫一次(與 mode-regions drawSide 同約定)。
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} azDeg 通道方位角：**唯一的資料輸入**
 * @param {number} W,H   畫布尺寸(投影參數)
 * @param {{halfDeg?:number}} [opts]
 * @returns {boolean} 是否畫出
 */
export function drawPassageBandErp(ctx, azDeg, W, H, opts){
  if(azDeg == null || !isFinite(azDeg) || !W || !H) return false;
  const half = (opts && opts.halfDeg) || HALF_DEG;
  const xc = erpU(azDeg) * W;
  const wpx = half / 360 * W * 2;
  let drew = false;
  for(const dx of [-W, 0, W]){
    const c = xc + dx, l = c - wpx / 2, r = c + wpx / 2;
    if(r < 0 || l > W) continue;
    paintBand(ctx, l, c, r, W, H);
    drew = true;
  }
  return drew;
}

/**
 * 透視畫布版(crop / walk):帶子畫在畫布頂緣,x = 方位投影到螢幕的水平位置。
 * 帶寬同樣以 az±Δ 投影(gnomonic 非線性,故逐點取樣取極值再夾進畫布)。
 * 方位落在視野外 → 不畫(回 false)。
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} azDeg   通道方位角：**唯一的資料輸入**
 * @param {number} viewAz  鏡頭朝向(投影參數)
 * @param {number} fovDeg  視角(投影參數)
 * @param {number} size    畫布邊長(投影參數)
 * @param {{halfDeg?:number}} [opts]
 * @returns {boolean} 是否畫出
 */
export function drawPassageBandPersp(ctx, azDeg, viewAz, fovDeg, size, opts, viewLat = 0){
  if(azDeg == null || !isFinite(azDeg) || !size) return false;
  const half = (opts && opts.halfDeg) || HALF_DEG;
  const xc = projX(azDeg, viewAz, fovDeg, size, viewLat);
  if(xc == null || xc < 0 || xc > size) return false;      // 視野外
  let lo = xc, hi = xc;
  const N = 12;
  for(let k = 0; k <= N; k++){
    const x = projX(azDeg - half + 2 * half * k / N, viewAz, fovDeg, size, viewLat);
    if(x == null) continue;                                 // 該取樣點在相機背面
    if(x < lo) lo = x;
    if(x > hi) hi = x;
  }
  paintBand(ctx, Math.max(0, lo), xc, Math.min(size, hi), size, size);
  return true;
}
