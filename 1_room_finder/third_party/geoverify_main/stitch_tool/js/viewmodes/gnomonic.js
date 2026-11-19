/* =========================================================================
   viewmodes/gnomonic.js：ERP ⇄ 透視(gnomonic)投影共用工具(mode-crop / mode-morph 用)
   低層模組:不 import 任何其它模組、不碰 DOM(除了建立離屏 canvas)。

   座標約定完全依 SPEC.md:
     u = ((az + 180) mod 360)/360(0=左緣), v ∈ [0,1](0=上), 方向向量
     d = (cos2πu·sinπv, cosπv, sin2πu·sinπv);相機 forward = az 方向(v=0.5)、
     up = +Y、right = az+90 方向。
   ========================================================================= */

const TAU = Math.PI * 2;

/** az(度) → 水平單位方向向量 [x,y,z](v=0.5)。*/
export function azToDir(azDeg){
  const u = ((azDeg % 360) + 540) % 360 / 360;   // ((az+180) mod 360)/360,容忍負角
  return [Math.cos(TAU * u), 0, Math.sin(TAU * u)];
}

/**
 * 相機基底:forward / right / up。latDeg>0 = 抬頭往上看(俯仰,無 roll)。
 * right 恆保持水平(不 roll),up = right × forward。
 * **latDeg=0 走特例回傳字面 [0,1,0]**：用叉積算會得到 0.9999999999999999 之類的值,
 * 進而讓 nearest 取樣偶爾取到相鄰像素;特例可保證與加入俯仰前的輸出逐位相同。
 */
export function cameraBasis(azDeg, latDeg = 0){
  const fh = azToDir(azDeg);
  const r = azToDir(azDeg + 90);
  if(!latDeg) return { f: fh, r, u: [0, 1, 0] };
  const t = latDeg * Math.PI / 180;
  const cl = Math.cos(t), sl = Math.sin(t);
  const f = [fh[0] * cl, sl, fh[2] * cl];
  const u = [r[1] * f[2] - r[2] * f[1],
             r[2] * f[0] - r[0] * f[2],
             r[0] * f[1] - r[1] * f[0]];
  return { f, r, u };
}

/**
 * 載入 ERP 圖並縮樣成 ImageData(等比,寬 maxW)。同 URL 會重複下載：呼叫端自行快取。
 * @returns {Promise<{data:Uint8ClampedArray,width:number,height:number}>}
 */
export function loadErpData(url, maxW = 2048){
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const w = Math.min(maxW, img.naturalWidth);
      const h = Math.round(w * img.naturalHeight / img.naturalWidth);
      const c = document.createElement('canvas');
      c.width = w; c.height = h;
      const ctx = c.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(img, 0, 0, w, h);
      try { resolve(ctx.getImageData(0, 0, w, h)); }
      catch(e){ reject(e); }
    };
    img.onerror = () => reject(new Error('載入 ERP 失敗: ' + url));
    img.src = url;
  });
}

/**
 * 把 ERP(ImageData)以 gnomonic 投影畫成 size×size 透視圖(nearest 取樣)。
 * @param {ImageData} erp    ERP 縮圖
 * @param {number} azDeg     相機朝向(度)
 * @param {number} fovDeg    水平視角(度,~90)
 * @param {number} size      輸出邊長(px)
 * @param {CanvasRenderingContext2D} outCtx 目標 2D context(畫在 (0,0)-(size,size))
 * @param {number} [latDeg=0] 俯仰(度),>0 抬頭。0 時與加入俯仰前的輸出逐位相同。
 */
export function renderGnomonic(erp, azDeg, fovDeg, size, outCtx, latDeg = 0){
  // 迴圈內另有 `let u`(ERP 水平比例),故基底的 up 一律命名 up。
  const { f, r, u: up } = cameraBasis(azDeg, latDeg);
  const tanH = Math.tan(fovDeg * Math.PI / 360);
  const W = erp.width, H = erp.height, src = erp.data;
  const out = outCtx.createImageData(size, size);
  const dst = out.data;
  let o = 0;
  for(let py = 0; py < size; py++){
    const ny = -(2 * (py + 0.5) / size - 1) * tanH;      // 螢幕 y 向下 → up 分量取負
    for(let px = 0; px < size; px++){
      const nx = (2 * (px + 0.5) / size - 1) * tanH;
      let dx = f[0] + nx * r[0] + ny * up[0];
      let dy = f[1] + nx * r[1] + ny * up[1];             // lat=0 時 = ny(f[1]=r[1]=0,up=[0,1,0])
      let dz = f[2] + nx * r[2] + ny * up[2];
      const inv = 1 / Math.sqrt(dx * dx + dy * dy + dz * dz);
      dx *= inv; dy *= inv; dz *= inv;
      const v = Math.acos(Math.max(-1, Math.min(1, dy))) / Math.PI;
      let u = Math.atan2(dz, dx) / TAU;
      u -= Math.floor(u);
      const sx = Math.min(W - 1, (u * W) | 0);
      const sy = Math.min(H - 1, (v * H) | 0);
      const s = (sy * W + sx) * 4;
      dst[o++] = src[s]; dst[o++] = src[s + 1]; dst[o++] = src[s + 2]; dst[o++] = 255;
    }
  }
  outCtx.putImageData(out, 0, 0);
}

/**
 * ERP 像素 → 裁切圖像素(mode-morph 把 match 點投到透視圖上)。
 * @param {number} x,y   ERP 像素(原圖座標)
 * @param {number} W,H   ERP 原圖尺寸
 * @param {number} azDeg 相機朝向
 * @param {number} fovDeg 視角
 * @param {number} size  裁切圖邊長
 * @returns {[number,number]|null} 裁切圖座標;背面或出界回 null
 */
export function erpPxToCrop(x, y, W, H, azDeg, fovDeg, size, latDeg = 0){
  const { f, r, u: up } = cameraBasis(azDeg, latDeg);
  const u = x / W, v = y / H;
  const sv = Math.sin(Math.PI * v);
  const d = [Math.cos(TAU * u) * sv, Math.cos(Math.PI * v), Math.sin(TAU * u) * sv];
  const zf = d[0] * f[0] + d[1] * f[1] + d[2] * f[2];    // d·forward
  if(zf <= 1e-6) return null;                             // 背面
  const tanH = Math.tan(fovDeg * Math.PI / 360);
  const xr = (d[0] * r[0] + d[1] * r[1] + d[2] * r[2]) / zf;      // d·right / zf
  const yu = (d[0] * up[0] + d[1] * up[1] + d[2] * up[2]) / zf;   // d·up / zf
  const px = (xr / tanH + 1) / 2 * size;
  const py = (-yu / tanH + 1) / 2 * size;
  if(px < 0 || px >= size || py < 0 || py >= size) return null;
  return [px, py];
}

/**
 * ERP 像素 → 3D 單位方向向量(座標約定同 renderGnomonic)。
 * 多邊形要先轉方向空間才能做近平面裁切：逐頂點 erpPxToCrop 會把出界頂點整顆丟掉,
 * 剩下的頂點重新閉合會讓形狀隨鏡頭變形(看起來像凸包被重算)。
 * @param {number} x,y ERP 像素(原圖座標;x 可超出 [0,W),ERP 對 x 週期性等價)
 * @param {number} W,H ERP 原圖尺寸
 * @returns {[number,number,number]} 單位方向
 */
export function erpPxToDir(x, y, W, H){
  const u = x / W, v = y / H;
  const sv = Math.sin(Math.PI * v);
  return [Math.cos(TAU * u) * sv, Math.cos(Math.PI * v), Math.sin(TAU * u) * sv];
}

/**
 * 3D 方向 → 裁切圖像素,**不做畫面邊界剔除**(交給 canvas 自行裁切)。
 * 只擋相機背面;呼叫端應先用 clipDirsNear 之類的近平面裁切保證全在前方。
 * 對 d 的長度不敏感(分子分母同乘),所以裁切時線性內插出來的非單位向量可直接餵進來。
 * @param {number[]} d   方向向量(不需正規化)
 * @param {number} azDeg 相機朝向
 * @param {number} fovDeg 視角
 * @param {number} size  裁切圖邊長
 * @returns {[number,number]|null} 裁切圖座標(可能落在 [0,size) 之外);背面回 null
 */
export function dirToCropRaw(d, azDeg, fovDeg, size, latDeg = 0){
  const { f, r, u: up } = cameraBasis(azDeg, latDeg);
  const zf = d[0] * f[0] + d[1] * f[1] + d[2] * f[2];    // d·forward
  if(zf <= 1e-6) return null;
  const tanH = Math.tan(fovDeg * Math.PI / 360);
  const xr = (d[0] * r[0] + d[1] * r[1] + d[2] * r[2]) / zf;
  const yu = (d[0] * up[0] + d[1] * up[1] + d[2] * up[2]) / zf;
  return [(xr / tanH + 1) / 2 * size, (-yu / tanH + 1) / 2 * size];
}

/**
 * 方向環對近平面(d·forward = eps)做 Sutherland–Hodgman 裁切。
 * gnomonic 把大圓映成直線,所以兩端點之間線性內插出的交點就是精確解。
 * @param {Array<number[]>} dirs 封閉環(頭尾自動相接)
 * @param {number} azDeg 相機朝向
 * @returns {Array<number[]>} 全部位於相機前方的環;完全在背面回空陣列
 */
export function clipDirsNear(dirs, azDeg, latDeg = 0){
  const { f } = cameraBasis(azDeg, latDeg);
  const eps = 1e-3;                                      // 太小會讓投影座標爆到千倍 size
  const sideOf = d => d[0] * f[0] + d[1] * f[1] + d[2] * f[2] - eps;
  const out = [];
  for(let i = 0; i < dirs.length; i++){
    const a = dirs[i], b = dirs[(i + 1) % dirs.length];
    const sa = sideOf(a), sb = sideOf(b);
    if(sa >= 0) out.push(a);
    if((sa >= 0) !== (sb >= 0)){
      const t = sa / (sa - sb);
      out.push([
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
      ]);
    }
  }
  return out;
}

/**
 * 裁切圖像素 → ERP 原圖像素(renderGnomonic 的反變換)。
 * @returns {[number,number]|null} (x,y) 原圖座標;不該失敗(畫面內皆前方)
 */
export function cropToErpPx(px, py, W, H, azDeg, fovDeg, size, latDeg = 0){
  // 注意:函式尾端已有 `let u`(ERP 水平比例),故基底的 up 改名 up,避免重複宣告。
  const { f, r, u: up } = cameraBasis(azDeg, latDeg);
  const tanH = Math.tan(fovDeg * Math.PI / 360);
  const nx = (2 * (px + 0.5) / size - 1) * tanH;
  const ny = -(2 * (py + 0.5) / size - 1) * tanH;
  let dx = f[0] + nx * r[0] + ny * up[0];
  let dy = f[1] + nx * r[1] + ny * up[1];
  let dz = f[2] + nx * r[2] + ny * up[2];
  const inv = 1 / Math.sqrt(dx * dx + dy * dy + dz * dz);
  dx *= inv; dy *= inv; dz *= inv;
  const v = Math.acos(Math.max(-1, Math.min(1, dy))) / Math.PI;
  let u = Math.atan2(dz, dx) / TAU;
  u -= Math.floor(u);
  return [u * W, v * H];
}

/** ERP ImageData 共用快取(最多 8 張,先進先出;mode-crop / mode-morph 共用)。*/
const _erpCache = new Map();
export function getErpCached(url, maxW = 2048){
  if(_erpCache.has(url)) return _erpCache.get(url);
  const p = loadErpData(url, maxW);
  _erpCache.set(url, p);
  if(_erpCache.size > 8){ _erpCache.delete(_erpCache.keys().next().value); }
  return p;
}

/** 一組方位角(度)的圓形平均;空陣列回 null。*/
export function circMeanDeg(azList){
  if(!azList || !azList.length) return null;
  let sx = 0, sy = 0;
  azList.forEach(a => { const r = a * Math.PI / 180; sx += Math.cos(r); sy += Math.sin(r); });
  return (Math.atan2(sy, sx) * 180 / Math.PI + 360) % 360;
}
