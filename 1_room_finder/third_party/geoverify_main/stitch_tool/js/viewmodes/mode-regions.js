/* =========================================================================
   viewmodes/mode-regions.js：「共視區域遮罩」檢視模式插件
   讀 區域 JSON(/api/regions,schema wp1-step3-v1):兩張 ERP 上下
   並排,同一 region 兩側塗同色半透明遮罩+編號。「A 的這塊 和 B 的那塊 是同一塊
   實體空間」。方向資訊不畫(遮罩直觀、方位線不直觀;方向請用
   「裁切對」模式看,那才是為方向設計的呈現)。

   過縫約定(見 pose_regions/README.md):polygon x 可能超出 [0,W) 以保連續;
   繪製時同一多邊形平移 −W / 0 / +W 各畫一次,超界自然裁掉。
   swap 旗標:區域檔只存 A_B 一個方向;命中反序檔時 server 回 swap=true,
   此時本面板(pair.i)要用 json 的 'b' 側,對側用 'a'。
   ========================================================================= */

import { registerViewMode } from './registry.js';
import { drawPassageBandErp } from './marker.js';
import { deriveViewPair } from './pairsel.js';
import { getHouseData, getHouseName, pairStatus } from '../state.js';
import { fetchRegions, nadirWarning } from '../api.js';

const CW = 1024, CH = 512;    // canvas 內部解析度(ERP 2:1)
const PALETTE = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45'];

let _root = null, _visible = false, _token = 0, _lastKey = '';
// 通道頂帶:本模式預設關：「檢視模式單一用途」的既有回饋(區域模式只講區域)
let _bandOn = false;
const _imgCache = new Map();   // url -> Promise<HTMLImageElement>(遮罩底圖直接用 <img>,不需像素存取)

function getImg(url){
  if(_imgCache.has(url)) return _imgCache.get(url);
  const p = new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error('載入失敗 ' + url));
    im.src = url;
  });
  _imgCache.set(url, p);
  if(_imgCache.size > 8){ _imgCache.delete(_imgCache.keys().next().value); }
  return p;
}

function buildDom(){
  if(_root) return;
  _root = document.createElement('div');
  _root.id = 'regions-mode';
  _root.className = 'viewmode-overlay';
  _root.innerHTML =
    '<div class="vm-head">共視區域遮罩(幾何驗證產物;同色 = 同一塊實體空間)</div>' +
    '<div class="vm-body vcol">' +
      '<div class="regions-pane"><div class="crop-label" id="regions-label-a"></div>' +
        '<canvas id="regions-canvas-a" width="' + CW + '" height="' + CH + '"></canvas></div>' +
      '<div class="regions-pane"><div class="crop-label" id="regions-label-b"></div>' +
        '<canvas id="regions-canvas-b" width="' + CW + '" height="' + CH + '"></canvas></div>' +
    '</div>' +
    '<div class="vm-hint">' +
      '<button id="regions-band" type="button" title="各圖頂緣的連通道方位標示(只吃方位角)">通道帶 OFF</button>' +
      '<span id="regions-hint"></span>' +
    '</div>';
  const viewer = document.getElementById('viewer');
  if(viewer) viewer.appendChild(_root);

  const bandBtn = document.getElementById('regions-band');
  if(bandBtn) bandBtn.onclick = () => {
    _bandOn = !_bandOn;
    syncBandBtn();
    _lastKey = '';                      // 帶子畫在 canvas 上 → 重畫整張
    refresh();
  };
}

function syncBandBtn(){
  const b = document.getElementById('regions-band');
  if(!b) return;
  b.textContent = _bandOn ? '通道帶 ON' : '通道帶 OFF';
  b.classList.toggle('on', _bandOn);
}

function setHint(html){ const el = document.getElementById('regions-hint'); if(el) el.innerHTML = html || ''; }

/** 在 ctx 上畫一側的所有 region(座標 = 原圖像素,依 W,H 縮放;過縫平移三次)。*/
function drawSide(ctx, regions, side, W, H){
  regions.forEach((reg, k) => {
    const poly = (reg[side] && reg[side].polygon) || [];
    if(poly.length < 3) return;
    const col = PALETTE[k % PALETTE.length];
    for(const dx of [-W, 0, W]){
      const pts = poly.map(([x, y]) => [(x + dx) / W * CW, y / H * CH]);
      if(Math.max(...pts.map(p => p[0])) < 0 || Math.min(...pts.map(p => p[0])) > CW) continue;
      ctx.beginPath();
      pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
      ctx.closePath();
      ctx.fillStyle = col + '4d';           // ~30% alpha
      ctx.fill();
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.stroke();
      const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
      const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
      if(cx >= 0 && cx < CW){
        ctx.fillStyle = col; ctx.font = 'bold 20px sans-serif'; ctx.textAlign = 'center';
        ctx.fillText('R' + k, cx, cy);
      }
    }
  });
}

function refresh(){
  if(!_visible) return;
  const hd = getHouseData(), house = getHouseName();
  if(!hd || !house) return;
  const d = deriveViewPair();
  if(!d){ setHint('目前沒有可用的配對(請先在佇列選一對)'); _lastKey = ''; return; }
  const { pair, azI, azJ } = d;
  const key = [house, pair.i, pair.j, azI.toFixed(1), azJ.toFixed(1), _bandOn ? 'b1' : 'b0'].join('|');
  const la = document.getElementById('regions-label-a');
  const lb = document.getElementById('regions-label-b');
  if(la) la.innerHTML = 'img <b>' + pair.i + '</b>';
  if(lb) lb.innerHTML = 'img <b>' + pair.j + '</b>';
  if(key === _lastKey) return;
  setHint('載入區域資料…');
  const tk = ++_token;
  const fa = hd.images[pair.i], fb = hd.images[pair.j];
  const urlOf = f => '/images/' + encodeURIComponent(house) + '/' + encodeURIComponent(f);
  Promise.all([fetchRegions(house, fa, fb), getImg(urlOf(fa)), getImg(urlOf(fb))]).then(([rj, ia, ib]) => {
    if(tk !== _token || !_visible) return;
    const ca = document.getElementById('regions-canvas-a');
    const cb = document.getElementById('regions-canvas-b');
    if(!ca || !cb) return;
    const xa = ca.getContext('2d'), xb = cb.getContext('2d');
    xa.clearRect(0, 0, CW, CH); xb.clearRect(0, 0, CW, CH);
    xa.drawImage(ia, 0, 0, CW, CH); xb.drawImage(ib, 0, 0, CW, CH);
    // 頂帶只吃方位角 → 沒有區域資料也照畫(畫在遮罩之前,不搶區域的視覺焦點)
    if(_bandOn){
      drawPassageBandErp(xa, azI, CW, CH);
      drawPassageBandErp(xb, azJ, CW, CH);
    }
    if(rj && rj.error){
      _lastKey = key;
      setHint('⚠ ' + rj.error);
      return;
    }
    // swap:檔案是 B_A 方向 → 本頁上面板(pair.i)用 json 'b' 側
    const sideTop = rj.swap ? 'b' : 'a', sideBot = rj.swap ? 'a' : 'b';
    const [Wa, Ha] = rj.swap ? rj.img_size_b : rj.img_size_a;
    const [Wb, Hb] = rj.swap ? rj.img_size_a : rj.img_size_b;
    const regs = rj.regions || [];
    drawSide(xa, regs, sideTop, Wa, Ha);
    drawSide(xb, regs, sideBot, Wb, Hb);
    const f = rj.flags || {};
    const st = pairStatus(pair);
    const bits = [
      st + (pair.prob != null ? ' ' + (pair.prob * 100).toFixed(0) + '%' : ''),
      regs.length + ' 個共視區域',
      'conf_used=' + (rj.conf_used != null ? rj.conf_used : '—'),
      'inlier=' + (rj.inlier_rate != null ? rj.inlier_rate : '—'),
    ];
    if(f.no_covisibility) bits.push('<b style="color:#e05a3a">無共視證據</b>');
    if(f.degenerate_rot_only) bits.push('<b style="color:#e0a53a">近零基線(兩點位幾乎同位)</b>');
    if(f.doorway_wedge) bits.push('<b style="color:#12b886">門洞楔形</b>');
    const nw = nadirWarning(rj);
    if(nw) bits.push('<b style="color:#e05a3a">⚠ ' + nw + '</b>');
    _lastKey = key;
    setHint(bits.join(' · ') + (regs.length ? '' : ' · 此對無區域(內點不足)'));
  }).catch(e => { if(tk === _token) setHint('載入失敗:' + e); });
}

registerViewMode({
  id: 'regions',
  label: '區域遮罩',
  title: '共視區域遮罩:幾何驗證的多邊形,兩側同色 = 同一塊實體空間',
  ownButton: true,
  isActive(){ return _visible; },
  enter(){ buildDom(); _visible = true; _root.style.display = 'flex'; _lastKey = ''; syncBandBtn(); refresh(); },
  exit(){ _visible = false; if(_root) _root.style.display = 'none'; _token++; },
  refresh,
});
