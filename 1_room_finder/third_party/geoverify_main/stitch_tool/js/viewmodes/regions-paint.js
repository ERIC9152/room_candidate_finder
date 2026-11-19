/* =========================================================================
   viewmodes/regions-paint.js：共視區域遮罩:透視畫布繪製 + 兩側取用
   低層模組(由 mode-walk / mode-crop 共用):只 import gnomonic、不碰 DOM。
   2026-07-29 從 mode-walk.js 原樣抽出(commit 84af7ce 的「形狀不隨鏡頭變形」
   近平面裁切管線),行為不得變。

   過縫約定(見 pose_regions/README.md):polygon 的 x 已 unwrap,可超出 [0,W)。
   swap 旗標:區域檔只存 A_B 一個方向;命中反序檔時 server 回 swap=true,
   此時 pair.i 要用 json 的 'b' 側,對側用 'a'。
   ========================================================================= */

import { erpPxToDir, dirToCropRaw, clipDirsNear } from './gnomonic.js';

const DENSIFY_DEG = 2;         // 遮罩邊加密步長(度)
const DENSIFY_MAX = 64;        // 單邊加密上限

/** 區域配色(walk / crop / regions 三模式同色序 → 跨模式看同一個 R 編號同色)。*/
export const REG_PALETTE = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45'];

/** ERP 直邊在透視圖上是曲線 → 依弧長加密,每段約 DENSIFY_DEG 度。*/
export function densifyToDirs(poly, W, H){
  const step = W * DENSIFY_DEG / 360;          // 每度對應的 ERP 像素(H/180 同值)
  const out = [];
  for(let i = 0; i < poly.length; i++){
    const [x0, y0] = poly[i];
    const [x1, y1] = poly[(i + 1) % poly.length];
    const n = Math.max(1, Math.min(DENSIFY_MAX, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / step)));
    for(let s = 0; s < n; s++){
      const t = s / n;
      out.push(erpPxToDir(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, W, H));
    }
  }
  return out;
}

/** 點是否在 polygon 內(ray casting);x 可超出 [0,W) 的過縫 poly 也適用。*/
export function pointInPoly(x, y, poly){
  let inside = false;
  for(let i = 0, j = poly.length - 1; i < poly.length; j = i++){
    const xi = poly[i][0], yi = poly[i][1];
    const xj = poly[j][0], yj = poly[j][1];
    if(((yi > y) !== (yj > y)) &&
       (x < (xj - xi) * (y - yi) / ((yj - yi) || 1e-12) + xi)){
      inside = !inside;
    }
  }
  return inside;
}

/**
 * 共視區遮罩:多邊形在 ERP 上是固定的,畫出來的形狀也必須只隨鏡頭「換位置」不「換形狀」。
 * 作法:加密 → 轉方向空間 → 近平面裁切 → 整條投影(不逐頂點剔除,超界交給 canvas 裁)。
 * 過縫不必分 −W/0/+W 三份:ERP 的 x 平移 W 在方向空間是同一組向量,一次就對。
 * 鏡頭深入區域內部時,裁切後的環本來就會蓋滿全幀：不需要(也不該)整格填色的保底。
 * @param {Array<{poly:Array,color?:string}>} sidePolys 該側的 region 清單
 * @param {number} W,H   ERP 原圖尺寸
 * @param {number} az,fov 鏡頭朝向 / 視角
 * @param {number} size  畫布邊長
 */
export function paintRegionsOn(ctx, sidePolys, W, H, az, fov, size, lat = 0){
  if(!sidePolys || !sidePolys.length || !W || !H || !size) return;
  sidePolys.forEach((item, k) => {
    const poly = item.poly;
    if(!poly || poly.length < 3) return;
    const col = item.color || REG_PALETTE[k % REG_PALETTE.length];
    const ring = clipDirsNear(densifyToDirs(poly, W, H), az, lat);
    if(ring.length < 3) return;
    const pts = [];
    for(const d of ring){
      const p = dirToCropRaw(d, az, fov, size, lat);
      if(p) pts.push(p);
    }
    if(pts.length < 3) return;
    ctx.beginPath();
    pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
    ctx.closePath();
    ctx.fillStyle = col + '4d';
    ctx.fill();
    ctx.strokeStyle = col;
    ctx.lineWidth = 2;
    ctx.stroke();
    // 標籤:重心出框但畫面中心落在區域內(鏡頭在區域裡)→ 改標在中心,不讓編號憑空消失
    let cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
    let cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
    const inFrame = cx >= 0 && cx < size && cy >= 0 && cy < size;
    if(!inFrame){
      if(!pointInPoly(size / 2, size / 2, pts)) return;
      cx = size / 2; cy = size / 2;
    }
    ctx.fillStyle = col;
    ctx.font = 'bold 16px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('R' + (item.k != null ? item.k : k), cx, cy);
  });
}

/** 單側區域面積(占全圖比例)。schema wp1-step3-v2 已存 area_frac(凸包面積/WH),
    直接用;缺欄位(舊版檔)才用 polygon shoelace 現算。x 已 unwrap,直接算即可。*/
export function areaFracOf(side, W, H){
  const a = side && side.area_frac;
  if(typeof a === 'number' && isFinite(a) && a >= 0) return a;
  const poly = (side && side.polygon) || [];
  if(poly.length < 3 || !W || !H) return 0;
  let s = 0;
  for(let i = 0; i < poly.length; i++){
    const [x0, y0] = poly[i];
    const [x1, y1] = poly[(i + 1) % poly.length];
    s += x0 * y1 - x1 * y0;
  }
  return Math.abs(s) / 2 / (W * H);
}

/**
 * regions JSON → pair.i / pair.j 兩側的 polygon 清單(已套 swap 約定)。
 * 同一 region 在兩側拿到同一個 color 與 k → 「同色 = 同一塊實體空間」。
 * @returns {{i:Array,j:Array,Wi:number,Hi:number,Wj:number,Hj:number}}
 */
export function sidesFromRegions(rj){
  const regs = (rj && !rj.error && rj.regions) || [];
  if(!regs.length) return { i:[], j:[], Wi:0, Hi:0, Wj:0, Hj:0 };
  const keyI = rj.swap ? 'b' : 'a', keyJ = rj.swap ? 'a' : 'b';
  const [Wi, Hi] = keyI === 'a' ? (rj.img_size_a || [0, 0]) : (rj.img_size_b || [0, 0]);
  const [Wj, Hj] = keyJ === 'a' ? (rj.img_size_a || [0, 0]) : (rj.img_size_b || [0, 0]);
  const i = [], j = [];
  regs.forEach((reg, k) => {
    const col = REG_PALETTE[k % REG_PALETTE.length];
    const pi = reg[keyI] && reg[keyI].polygon;
    const pj = reg[keyJ] && reg[keyJ].polygon;
    if(pi && pi.length >= 3) i.push({ poly:pi, color:col, k, area:areaFracOf(reg[keyI], Wi, Hi) });
    if(pj && pj.length >= 3) j.push({ poly:pj, color:col, k, area:areaFracOf(reg[keyJ], Wj, Hj) });
  });
  return { i, j, Wi, Hi, Wj, Hj };
}
