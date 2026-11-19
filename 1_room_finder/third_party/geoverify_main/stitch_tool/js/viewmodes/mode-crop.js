/* =========================================================================
   viewmodes/mode-crop.js：模式 A「透視裁切對」
   兩張 gnomonic 並排;可切對望/同向。
   互動:在 canvas 上拖曳改方位。水平=yaw、垂直=俯仰(2026-07-30 加,上限 ±LAT_MAX);
   滾輪改 FOV。
   疊圖:共視區域遮罩(兩側同色 = 同一塊實體空間)
   + 連通道頂帶(只吃方位角);兩者各自有開關,互動重繪時一起重畫。
   ========================================================================= */

import { registerViewMode } from './registry.js';
import { getErpCached, renderGnomonic } from './gnomonic.js';
import { paintRegionsOn, sidesFromRegions } from './regions-paint.js';
import { drawPassageBandPersp } from './marker.js';
import { deriveViewPair } from './pairsel.js';
import { passageSourceHint } from '../srchint.js';
import { getHouseData, getHouseName, pairStatus } from '../state.js';
import { fetchRegions } from '../api.js';
import { on } from '../events.js';

const SIZE = 512;
const FOV_DEFAULT = 90;
const FOV_MIN = 30;
const FOV_MAX = 140;
const DRAG_DEG_PER_PX = 0.35;   // 拖曳靈敏度(度/px),水平與垂直共用
const LAT_MAX = 70;             // 俯仰上限(度);再大 gnomonic 邊緣會拉伸到不可讀

let _root = null;
let _visible = false;
let _token = 0;
let _sameDir = false;

// 目前配對基準角 + 使用者偏移
let _baseAzA = 0, _baseAzB = 0;
let _offA = 0, _offB = 0;       // 相對基準的 yaw 偏移
let _latA = 0, _latB = 0;       // 俯仰(度),>0 抬頭;基準恆為水平線
let _fovA = FOV_DEFAULT, _fovB = FOV_DEFAULT;
let _pairKey = '';              // house|i|j：換對時重置偏移
let _erpA = null, _erpB = null; // 快取 ImageData,拖曳時即時重繪
let _pair = null;
let _raf = 0;
let _dirty = false;

// 疊圖狀態
let _regOn = true, _bandOn = true;
let _sides = null;              // sidesFromRegions() 結果;無資料時 null
let _regNote = '';              // 無區域資料時的說明(顯示在 hint)
let _srcHint = { text: '', fallback: false };   // 通道方位來源/降級提示(見 srchint.js)
let _azPassA = 0, _azPassB = 0; // 兩側的通道方位(az_ij / az_ji;與鏡頭偏移無關)

// 拖曳狀態
let _drag = null;  // { side:'a'|'b', x0, y0, off0, lat0 }

function buildDom(){
  if(_root) return;
  _root = document.createElement('div');
  _root.id = 'crop-mode';
  _root.className = 'viewmode-overlay';
  _root.innerHTML =
    '<div class="vm-head" id="crop-head">透視裁切對 · 拖曳旋轉 · 滾輪調 FOV</div>' +
    '<div class="vm-body">' +
      '<div class="crop-pane">' +
        '<div class="crop-label" id="crop-label-a"></div>' +
        '<canvas id="crop-canvas-a" class="crop-interactive" width="' + SIZE + '" height="' + SIZE + '"></canvas>' +
      '</div>' +
      '<div class="crop-band"><div class="crop-band-arrow" id="crop-band-arrow">⇄</div>' +
        '<div class="crop-band-text" id="crop-band-text">同一通道</div></div>' +
      '<div class="crop-pane">' +
        '<div class="crop-label" id="crop-label-b"></div>' +
        '<canvas id="crop-canvas-b" class="crop-interactive" width="' + SIZE + '" height="' + SIZE + '"></canvas>' +
      '</div>' +
    '</div>' +
    '<div class="vm-hint" id="crop-hint">' +
      '<button id="crop-dir-toggle" type="button" title="對望=互相看;同向=B 朝行進方向">對望 ⇄ 同向</button>' +
      '<button id="crop-reset-view" type="button" title="重置方位偏移與 FOV">重置視角</button>' +
      '<button id="crop-reg" type="button" title="疊加 共視區域遮罩(兩側同色 = 同一塊實體空間)">遮罩 ON</button>' +
      '<button id="crop-band" type="button" title="畫布頂緣的連通道方位標示(只吃方位角)">通道帶 ON</button>' +
      '<span id="crop-hint-text"></span>' +
    '</div>';
  const viewer = document.getElementById('viewer');
  if(viewer) viewer.appendChild(_root);

  const btn = document.getElementById('crop-dir-toggle');
  if(btn) btn.onclick = () => {
    _sameDir = !_sameDir;
    btn.textContent = _sameDir ? '同向 → 對望' : '對望 ⇄ 同向';
    btn.classList.toggle('on', _sameDir);
    // 同向切換改基準 B 角,偏移保留相對意義較怪 → 重置 B 偏移
    _offB = 0; _latB = 0;
    applyBaseFromPair();
    schedulePaint();
    updateLabels();
  };

  const reset = document.getElementById('crop-reset-view');
  if(reset) reset.onclick = () => {
    _offA = 0; _offB = 0;
    _latA = 0; _latB = 0;
    _fovA = FOV_DEFAULT; _fovB = FOV_DEFAULT;
    schedulePaint();
    updateLabels();
    setHint('已重置方位與 FOV');
  };

  const regBtn = document.getElementById('crop-reg');
  if(regBtn) regBtn.onclick = () => {
    _regOn = !_regOn;
    syncToggleBtns();
    schedulePaint();
  };
  const bandBtn = document.getElementById('crop-band');
  if(bandBtn) bandBtn.onclick = () => {
    _bandOn = !_bandOn;
    syncToggleBtns();
    schedulePaint();
  };

  bindCanvas('crop-canvas-a', 'a');
  bindCanvas('crop-canvas-b', 'b');
}

/** 兩顆疊圖鈕的文字/高亮/停用狀態(無區域資料時遮罩鈕停用)。*/
function syncToggleBtns(){
  const hasReg = !!(_sides && (_sides.i.length || _sides.j.length));
  const regBtn = document.getElementById('crop-reg');
  if(regBtn){
    regBtn.textContent = _regOn ? '遮罩 ON' : '遮罩 OFF';
    regBtn.classList.toggle('on', _regOn && hasReg);
    regBtn.disabled = !hasReg;
    regBtn.title = hasReg
      ? '疊加 共視區域遮罩(兩側同色 = 同一塊實體空間)'
      : (_regNote || '此對沒有共視區域資料');
  }
  const bandBtn = document.getElementById('crop-band');
  if(bandBtn){
    bandBtn.textContent = _bandOn ? '通道帶 ON' : '通道帶 OFF';
    bandBtn.classList.toggle('on', _bandOn);
  }
}

function bindCanvas(id, side){
  const cv = document.getElementById(id);
  if(!cv || cv._cropBound) return;
  cv._cropBound = true;
  cv.style.cursor = 'grab';
  cv.style.touchAction = 'none';

  cv.addEventListener('pointerdown', e => {
    if(!_visible || !_erpA) return;
    cv.setPointerCapture(e.pointerId);
    cv.style.cursor = 'grabbing';
    _drag = {
      side,
      x0: e.clientX,
      y0: e.clientY,
      off0: side === 'a' ? _offA : _offB,
      lat0: side === 'a' ? _latA : _latB,
    };
    e.preventDefault();
  });

  cv.addEventListener('pointermove', e => {
    if(!_drag || _drag.side !== side) return;
    const dx = e.clientX - _drag.x0;
    const dy = e.clientY - _drag.y0;
    // 拖右 = 往右看 = 方位增加(與常見 panorama 一致:拖圖向右內容向左);
    // 拖下 = 往上看 = 俯仰增加(同一套「拖圖 = 內容跟著手走」的直覺)。
    const dAz = dx * DRAG_DEG_PER_PX;
    const dLat = clamp(_drag.lat0 + dy * DRAG_DEG_PER_PX, -LAT_MAX, LAT_MAX);
    if(side === 'a'){ _offA = _drag.off0 + dAz; _latA = dLat; }
    else { _offB = _drag.off0 + dAz; _latB = dLat; }
    schedulePaint();
    updateLabels();
  });

  const end = e => {
    if(!_drag || _drag.side !== side) return;
    _drag = null;
    cv.style.cursor = 'grab';
    try { cv.releasePointerCapture(e.pointerId); } catch(_){}
  };
  cv.addEventListener('pointerup', end);
  cv.addEventListener('pointercancel', end);

  cv.addEventListener('wheel', e => {
    if(!_visible || !_erpA) return;
    e.preventDefault();
    const step = e.deltaY > 0 ? 5 : -5;   // 滾下 = FOV 變大(拉遠)
    if(side === 'a'){
      _fovA = clamp(_fovA + step, FOV_MIN, FOV_MAX);
    } else {
      _fovB = clamp(_fovB + step, FOV_MIN, FOV_MAX);
    }
    schedulePaint();
    updateLabels();
  }, { passive:false });
}

function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }
function normAz(a){ return ((a % 360) + 360) % 360; }
function setHint(t){ const el = document.getElementById('crop-hint-text'); if(el) el.textContent = t || ''; }

function azA(){ return _baseAzA + _offA; }
function azB(){ return _baseAzB + _offB; }

function applyBaseFromPair(){
  const d = deriveViewPair();
  if(!d){ _pair = null; return false; }
  const { pair, azI, azJ, semantic } = d;
  _pair = pair;
  _srcHint = passageSourceHint(pair, semantic);
  _baseAzA = azI;
  _baseAzB = _sameDir ? azJ + 180 : azJ;
  // 通道方位本身與「對望/同向」和拖曳偏移無關：同向時 B 背對通道,帶子自然落在視野外
  _azPassA = azI;
  _azPassB = azJ;
  return true;
}

function updateLabels(){
  if(!_pair) return;
  const la = document.getElementById('crop-label-a');
  const lb = document.getElementById('crop-label-b');
  const st = pairStatus(_pair);
  const a = normAz(azA()), b = normAz(azB());
  if(la) la.innerHTML =
    '從 <b>img ' + _pair.i + '</b> · ' +
    '<span class="crop-az">@ ' + a.toFixed(1) + '°</span>' +
    ' · FOV ' + _fovA.toFixed(0) + '°' +
    (_offA ? ' <span class="crop-az">(Δ' + (_offA > 0 ? '+' : '') + _offA.toFixed(0) + '°)</span>' : '') +
    (_latA ? ' <span class="crop-az">(↕' + (_latA > 0 ? '+' : '') + _latA.toFixed(0) + '°)</span>' : '');
  if(lb){
    const title = _sameDir
      ? '<b>img ' + _pair.j + '</b> 朝行進方向'
      : '從 <b>img ' + _pair.j + '</b> 看向 img ' + _pair.i;
    lb.innerHTML = title +
      ' · <span class="crop-az">@ ' + b.toFixed(1) + '°</span>' +
      ' · FOV ' + _fovB.toFixed(0) + '°' +
      (_offB ? ' <span class="crop-az">(Δ' + (_offB > 0 ? '+' : '') + _offB.toFixed(0) + '°)</span>' : '') +
    (_latB ? ' <span class="crop-az">(↕' + (_latB > 0 ? '+' : '') + _latB.toFixed(0) + '°)</span>' : '');
  }
  const bandText = document.getElementById('crop-band-text');
  const bandArrow = document.getElementById('crop-band-arrow');
  if(bandArrow) bandArrow.textContent = _sameDir ? '⇉' : '⇄';
  if(bandText) bandText.textContent =
    (_sameDir ? '同向 · ' : '對望 · ') +
    (st === 'confirmed' ? '已確認' : st === 'pending' ? '候選' : st) +
    (_pair.prob != null ? ' ' + (_pair.prob * 100).toFixed(0) + '%' : '');
}

function schedulePaint(){
  _dirty = true;
  if(_raf) return;
  _raf = requestAnimationFrame(() => {
    _raf = 0;
    if(_dirty) paintNow();
  });
}

function paintNow(){
  _dirty = false;
  if(!_visible || !_erpA || !_erpB) return;
  const ca = document.getElementById('crop-canvas-a');
  const cb = document.getElementById('crop-canvas-b');
  if(!ca || !cb) return;
  const xa = ca.getContext('2d'), xb = cb.getContext('2d');
  const va = azA(), vb = azB();
  renderGnomonic(_erpA, va, _fovA, SIZE, xa, _latA);
  renderGnomonic(_erpB, vb, _fovB, SIZE, xb, _latB);
  // 疊圖跟著鏡頭一起重畫(拖曳/滾輪/reset/對望同向切換都走 schedulePaint → 這裡)
  if(_regOn && _sides){
    paintRegionsOn(xa, _sides.i, _sides.Wi, _sides.Hi, va, _fovA, SIZE, _latA);
    paintRegionsOn(xb, _sides.j, _sides.Wj, _sides.Hj, vb, _fovB, SIZE, _latB);
  }
  let bandSeen = '';
  if(_bandOn){
    const la = drawPassageBandPersp(xa, _azPassA, va, _fovA, SIZE, null, _latA);
    const lb = drawPassageBandPersp(xb, _azPassB, vb, _fovB, SIZE, null, _latB);
    if(!la || !lb) bandSeen = ' · 通道帶:' + (la ? '' : '左') + (la || lb ? '' : '/') + (lb ? '' : '右') + '在視野外';
  }
  const nReg = _sides ? Math.max(_sides.i.length, _sides.j.length) : 0;
  setHint(
    (_sameDir ? '同向' : '對望') +
    ' · 拖曳旋轉視角(可上下) · 滾輪調 FOV(' + FOV_MIN + '–' + FOV_MAX + '°)' +
    ' · 左 FOV' + _fovA.toFixed(0) + '° / 右 FOV' + _fovB.toFixed(0) + '°' +
    (_regOn && nReg ? ' · 遮罩 ' + nReg + ' 區(同色=同一塊)' : (_regNote ? ' · ' + _regNote : '')) +
    bandSeen +
    (_srcHint.text ? ' · ' + (_srcHint.fallback ? '⚠ ' : '') + _srcHint.text : '')
  );
}

function refresh(){
  if(!_visible) return;
  const hd = getHouseData(), house = getHouseName();
  if(!hd || !house) return;
  if(!applyBaseFromPair()){
    setHint('目前沒有可用的配對(請先在佇列選一對,或在 ERP 面板選兩張互為候選的圖)');
    _erpA = _erpB = null;
    _sides = null; _regNote = '';
    _pairKey = '';
    syncToggleBtns();
    return;
  }
  const pk = [house, _pair.i, _pair.j].join('|');
  if(pk !== _pairKey){
    _pairKey = pk;
    _offA = 0; _offB = 0;
    _fovA = FOV_DEFAULT; _fovB = FOV_DEFAULT;
    _sides = null; _regNote = '';          // 換對:舊區域先清掉,別畫到新的一對上
    syncToggleBtns();
  }
  updateLabels();

  const tk = ++_token;
  const fa = hd.images[_pair.i], fb = hd.images[_pair.j];
  const urlOf = f => '/images/' + encodeURIComponent(house) + '/' + encodeURIComponent(f);
  setHint('載入 ERP…');
  // 區域 JSON 缺(404 / 內點不足)是常態,不擋畫面：只關掉遮罩並在 hint 說明
  const regsP = fetchRegions(house, fa, fb).then(rj => {
    if(rj && rj.error){ _sides = null; _regNote = '無共視區域:' + rj.error; return; }
    const s = sidesFromRegions(rj);
    if(!s.i.length && !s.j.length){ _sides = null; _regNote = '此對無共視區域(內點不足)'; return; }
    _sides = s; _regNote = '';
  }).catch(() => { _sides = null; _regNote = '共視區域載入失敗'; });

  Promise.all([getErpCached(urlOf(fa)), getErpCached(urlOf(fb)), regsP]).then(([ea, eb]) => {
    if(tk !== _token || !_visible) return;
    _erpA = ea; _erpB = eb;
    syncToggleBtns();
    paintNow();
    updateLabels();
  }).catch(e => { if(tk === _token) setHint('載入失敗:' + e.message); });
}

registerViewMode({
  id: 'crop',
  label: '裁切對',
  title: '透視裁切對:拖曳旋轉 / 滾輪 FOV;可切對望/同向',
  ownButton: true,
  isActive(){ return _visible; },
  enter(){
    buildDom();
    _visible = true;
    _root.style.display = 'flex';
    _pairKey = '';
    const btn = document.getElementById('crop-dir-toggle');
    if(btn){
      btn.textContent = _sameDir ? '同向 → 對望' : '對望 ⇄ 同向';
      btn.classList.toggle('on', _sameDir);
    }
    syncToggleBtns();
    refresh();
  },
  exit(){
    _visible = false;
    if(_root) _root.style.display = 'none';
    _token++;
    _drag = null;
    if(_raf){ cancelAnimationFrame(_raf); _raf = 0; }
  },
  refresh,
});

on('state-changed', () => { if(_visible) refresh(); });
