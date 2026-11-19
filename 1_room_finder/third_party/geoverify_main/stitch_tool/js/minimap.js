/* =========================================================================
   minimap.js：路線圖（2D canvas，力導向佈局，綠邊=已確認）+ 放大彈窗
   訂閱 'house-loaded' / 'state-changed' / 'view-changed' 自動重繪。

   力導向佈局 + 放大彈窗：
   - 佈局改為力導向（layout.forceLayout）：已確認邊當彈簧、節點間庫倫斥力、孤立節點推外圈。
     確定性（無 random，初始圓形排列），結果 cache；house-loaded / state-changed 才重算，
     view-changed 只重繪（避免每次換圖跳動）。小圖與彈窗共用同一份佈局（各自依 canvas 尺寸縮放）。
   - 節點著色：有確認邊=綠、孤立=灰、目前圖高亮（3D=琥珀；ERP 上圖=藍、下圖=橘）。
   - 「⤢ 放大」/ 雙擊小圖 → 全螢幕 70% modal：放大力導向圖、節點標 img 編號、hover 顯檔名、
     點節點跳圖（3D 模式跳 3D；ERP 模式左鍵=設上圖、右鍵=設下圖）、孤立圖列成快速入口列、Esc/背景關閉。
   - 小圖擠不下時縮節點但命中半徑放寬（見 nodeRadiusFor / HIT_PAD）。

   需求 2（沿用）：ERP 顯示兩個路線圖；點節點換該面板圖。toggle 收合/展開存 STATE.ui.minimap。
   依賴：config / state / nav / erp / layout / util / events。
   ========================================================================= */

import { MAP_COL } from './config.js';
import { STATE, getHouseData, getHouseName, scheduleSave } from './state.js';
import { nav } from './nav.js';
import { isErpOn, isErpHoriz, goto3DImage, setErpPaneImage } from './erp.js';
import { forceLayout, isIsolated } from './layout.js';
import { toast } from './util.js';
import { on, emit } from './events.js';

/** 命中判定的額外容差（節點半徑之外再加幾 px），小圖節點縮小時仍好點。*/
const HIT_PAD = 8;

/**
 * 佈局快取：以 (n_images, confirmed 邊指紋, W, H, nodeR) 為 key 的正規化 positions（[0,1]x[0,1]）。
 */
let _layoutsCache = new Map();

/** 目前 confirmed 邊清單（[i,j]），只取節點都在範圍內者。*/
function confirmedEdges(N){
  const out = [];
  STATE.confirmed.forEach(c => { if(c.i < N && c.j < N && c.i !== c.j) out.push([c.i, c.j]); });
  return out;
}

/** 佈局指紋：節點數 + 排序後的確認邊 key 串（confirmed 變了才重算）。*/
function layoutKey(N, edges){
  const keys = edges.map(e => (e[0] < e[1] ? e[0] + '_' + e[1] : e[1] + '_' + e[0])).sort();
  return N + '|' + keys.join(',');
}

/**
 * 取得（必要時重算）正規化佈局。
 */
function getNormLayout(N, nodeR, W, H){
  if(N <= 0) return null;
  const edges = confirmedEdges(N);
  const key = layoutKey(N, edges) + '|' + W + '|' + H + '|' + nodeR;
  if(_layoutsCache.has(key)) return _layoutsCache.get(key);

  let init = null;
  for(const [k, v] of _layoutsCache.entries()){
    if(k.startsWith(N + '|') && k.endsWith('|' + W + '|' + H + '|' + nodeR)){
      init = project(W, H, v);
      break;
    }
  }

  const iter = init ? 60 : 180;
  const pos = forceLayout(N, edges, W, H, iter, init, { nodeR });

  const pad = Math.max(10, Math.min(W, H) * 0.10);
  const iw = W - pad * 2, ih = H - pad * 2;
  const norm = pos.map(p => ({
    x: iw > 0 ? (p.x - pad) / iw : 0.5,
    y: ih > 0 ? (p.y - pad) / ih : 0.5
  }));
  _layoutsCache.set(key, norm);
  return norm;
}

/** 讓 house 切換 / reset 時強制丟棄佈局快取（下一次重算）。*/
export function invalidateMapLayout(){ _layoutsCache.clear(); }

/**
 * 把正規化佈局投影到某 canvas 的像素座標（保留 pad 邊界）。
 * @param {number} W @param {number} H @param {Array<{x,y}>} norm
 * @returns {Array<{x:number,y:number}>}
 */
function project(W, H, norm){
  const pad = Math.max(10, Math.min(W, H) * 0.10);
  const iw = W - pad * 2, ih = H - pad * 2;
  return norm.map(p => ({ x:pad + p.x * iw, y:pad + p.y * ih }));
}

/** 依 canvas 尺寸與節點數決定節點半徑（小圖 / 節點多時縮小，但有下限保可視）。*/
function nodeRadiusFor(W, H, N){
  const base = Math.min(W, H) / 2;
  let r = base * 0.05;
  if(N > 16) r *= 0.8;
  if(N > 28) r *= 0.75;
  return Math.max(3.2, Math.min(r, 9));
}

/**
 * 在某 canvas 畫力導向路線圖。
 * @param {string} canvasId
 * @param {number} highlightImg 高亮節點（3D=curImage；ERP=該面板圖）
 * @param {string} hiColor 高亮色
 * @param {object} [opts] {labelAll:bool 大圖全標號, big:bool 放大版樣式}
 */
function drawInto(canvasId, highlightImg, hiColor, opts){
  opts = opts || {};
  const cv = document.getElementById(canvasId); if(!cv) return;
  const g = cv.getContext('2d');
  const W = cv.width, H = cv.height; g.clearRect(0, 0, W, H);
  const hd = getHouseData(); if(!hd) return;
  const N = hd.n_images;
  const nodeR = opts.big ? Math.max(6, nodeRadiusFor(W, H, N)) : nodeRadiusFor(W, H, N);
  const norm = getNormLayout(N, nodeR, W, H); if(!norm) return;
  const pos = project(W, H, norm);
  const edges = confirmedEdges(N);

  // 邊（綠）
  g.lineWidth = opts.big ? 2.5 : 2; g.strokeStyle = MAP_COL.edge; g.globalAlpha = .8;
  edges.forEach(e => { g.beginPath(); g.moveTo(pos[e[0]].x, pos[e[0]].y); g.lineTo(pos[e[1]].x, pos[e[1]].y); g.stroke(); });
  g.globalAlpha = 1;

  // 節點
  for(let k = 0; k < N; k++){
    const iso = isIsolated(k, edges);
    const isHi = (k === highlightImg);
    const r = isHi ? nodeR * 1.35 : nodeR;
    g.beginPath(); g.arc(pos[k].x, pos[k].y, r, 0, 7);
    g.fillStyle = isHi ? hiColor : (iso ? MAP_COL.isolated : MAP_COL.connected);
    g.fill();
    if(isHi){ g.lineWidth = 2; g.strokeStyle = '#fff'; g.stroke(); }
    else if(iso && opts.big){ g.lineWidth = 1.5; g.strokeStyle = '#8894a8'; g.stroke(); }
  }

  // 節點編號
  g.fillStyle = MAP_COL.label; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.font = (opts.big ? 11 : 9) + 'px sans-serif';
  for(let t = 0; t < N; t++){
    if(opts.labelAll || N <= 24 || t % 2 === 0){
      g.fillText(t, pos[t].x, pos[t].y - (nodeR + (opts.big ? 9 : 7)));
    }
  }
}

/** 重繪所有小路線圖（3D 單張 + ERP 雙張）。ERP 上/下圖用不同高亮色。*/
export function buildMinimap(){
  const erp = isErpOn();
  if(!erp){
    drawInto('minimap', nav.curImage, MAP_COL.hi3d);
  } else {
    drawInto('minimap-erp-top', nav.erpTopImg, MAP_COL.hiTop);
    drawInto('minimap-erp-bot', nav.erpBotImg, MAP_COL.hiBot);
  }
  if(_modalOpen) drawModal();
}

/**
 * canvas 點擊 → 最近節點（距離 < 節點半徑 + HIT_PAD）→ imgIdx（無命中回 -1）。
 * @param {HTMLCanvasElement} cv @param {MouseEvent} e @param {boolean} [big]
 */
function hitNode(cv, e, big){
  const hd = getHouseData(); if(!hd) return -1;
  const N = hd.n_images;
  const W = cv.width, H = cv.height;
  const nodeR = big ? Math.max(7, nodeRadiusFor(W, H, N) * 1.2) : nodeRadiusFor(W, H, N);
  const norm = getNormLayout(N, nodeR, W, H); if(!norm) return -1;
  const pos = project(W, H, norm);
  const r = cv.getBoundingClientRect();
  const sx = cv.width / r.width, sy = cv.height / r.height;
  const x = (e.clientX - r.left) * sx, y = (e.clientY - r.top) * sy;
  let best = -1, bestD = 1e9;
  const hitR = nodeR * 1.35 + HIT_PAD + (big ? 4 : 0);
  for(let k = 0; k < N; k++){
    const d = Math.hypot(pos[k].x - x, pos[k].y - y);
    if(d < bestD){ bestD = d; best = k; }
  }
  return (bestD <= hitR) ? best : -1;
}

// =========================================================================
//  放大彈窗（modal）
// =========================================================================
let _modalOpen = false;

/** 開啟放大彈窗。*/
export function openMapModal(){
  const hd = getHouseData(); if(!hd) return;
  const m = document.getElementById('map-modal'); if(!m) return;
  _modalOpen = true;
  m.style.display = 'flex';
  const modeEl = document.getElementById('map-modal-mode');
  if(modeEl){
    modeEl.textContent = isErpOn()
      ? (isErpHoriz() ? '左鍵=設為左圖 · 右鍵=設為右圖' : '左鍵=設為上圖 · 右鍵=設為下圖')
      : '點節點跳到該圖（3D）';
  }
  drawModal();
  buildIsoList();
}

/** 關閉放大彈窗。*/
export function closeMapModal(){
  _modalOpen = false;
  const m = document.getElementById('map-modal');
  if(m) m.style.display = 'none';
  const tip = document.getElementById('map-modal-tip');
  if(tip) tip.style.opacity = '0';
}

/** 在彈窗大 canvas 畫力導向圖（全標號、放大樣式）；ERP 模式同時高亮兩張圖。*/
function drawModal(){
  const cv = document.getElementById('map-modal-canvas'); if(!cv) return;
  const hd = getHouseData(); if(!hd) return;
  const N = hd.n_images;
  const g = cv.getContext('2d');
  const W = cv.width, H = cv.height; g.clearRect(0, 0, W, H);
  const erp = isErpOn();
  const nodeR = Math.max(7, nodeRadiusFor(W, H, N) * 1.2);
  const norm = getNormLayout(N, nodeR, W, H); if(!norm) return;
  const pos = project(W, H, norm);
  const edges = confirmedEdges(N);

  g.lineWidth = 2.5; g.strokeStyle = MAP_COL.edge; g.globalAlpha = .8;
  edges.forEach(e => { g.beginPath(); g.moveTo(pos[e[0]].x, pos[e[0]].y); g.lineTo(pos[e[1]].x, pos[e[1]].y); g.stroke(); });
  g.globalAlpha = 1;

  for(let k = 0; k < N; k++){
    const iso = isIsolated(k, edges);
    let hiColor = null;
    if(erp){ if(k === nav.erpTopImg) hiColor = MAP_COL.hiTop; else if(k === nav.erpBotImg) hiColor = MAP_COL.hiBot; }
    else if(k === nav.curImage) hiColor = MAP_COL.hi3d;
    const isHi = !!hiColor;
    const r = isHi ? nodeR * 1.3 : nodeR;
    g.beginPath(); g.arc(pos[k].x, pos[k].y, r, 0, 7);
    g.fillStyle = isHi ? hiColor : (iso ? MAP_COL.isolated : MAP_COL.connected);
    g.fill();
    if(isHi){ g.lineWidth = 2.5; g.strokeStyle = '#fff'; g.stroke(); }
    else if(iso){ g.lineWidth = 1.5; g.strokeStyle = '#8894a8'; g.stroke(); }
  }
  g.fillStyle = MAP_COL.label; g.textAlign = 'center'; g.textBaseline = 'middle'; g.font = '12px sans-serif';
  for(let t = 0; t < N; t++){ g.fillText(t, pos[t].x, pos[t].y - (nodeR + 10)); }
}

/** 彈窗底部「未連接圖片」快速入口列（孤立節點）。*/
function buildIsoList(){
  const wrap = document.getElementById('map-modal-iso-list'); if(!wrap) return;
  wrap.innerHTML = '';
  const hd = getHouseData(); if(!hd) return;
  const N = hd.n_images;
  const edges = confirmedEdges(N);
  const isoTitle = document.getElementById('map-modal-iso-title');
  let count = 0;
  for(let k = 0; k < N; k++){
    if(!isIsolated(k, edges)) continue;
    count++;
    const chip = document.createElement('div');
    chip.className = 'iso-chip';
    chip.innerHTML = '<i></i>img ' + k + ' · ' + hd.images[k].slice(0, 10);
    chip.title = hd.images[k];
    chip.onclick = () => { gotoFromModal(k, 'left'); };
    chip.oncontextmenu = (e) => { e.preventDefault(); gotoFromModal(k, 'right'); };
    wrap.appendChild(chip);
  }
  if(isoTitle){
    isoTitle.textContent = count > 0
      ? ('未連接圖片（' + count + '）— ' + (isErpOn() ? '左鍵設上圖/右鍵設下圖' : '點擊前往'))
      : '所有圖片皆已連通 🎉';
  }
}

/**
 * 彈窗點節點/孤立 chip 的跳圖：3D 模式跳該圖；ERP 模式 left=上/左圖、right=下/右圖。
 * @param {number} idx @param {'left'|'right'} btn
 */
function gotoFromModal(idx, btn){
  if(isErpOn()){
    if(btn === 'right') setErpPaneImage('bot', idx);
    else setErpPaneImage('top', idx);
    const hd = getHouseData();
    toast('已將' + (btn === 'right' ? (isErpHoriz() ? '右' : '下') : (isErpHoriz() ? '左' : '上')) + '圖設為 img ' + idx
      + (hd ? '（' + hd.images[idx].slice(0, 12) + '）' : ''));
    drawModal(); buildIsoList();
  } else {
    goto3DImage(idx);
    closeMapModal();
  }
}

// =========================================================================
//  可見性 / toggle（沿用需求 2）
// =========================================================================
/** 依 ERP 開關 / 收合狀態切換單張(3D) vs 雙張(ERP) 路線圖顯示與標籤。*/
export function applyMinimapVisibility(){
  const shown = STATE.ui.minimap !== false;
  const erp = isErpOn();
  const horiz = isErpHoriz();
  const set = (id, disp) => { const el = document.getElementById(id); if(el) el.style.display = disp; };
  set('minimap-wrap', (shown && !erp) ? 'block' : 'none');
  set('minimap-erp-wrap', (shown && erp) ? 'flex' : 'none');
  const tb = document.getElementById('minimap-toggle');
  if(tb) tb.textContent = shown ? '▣ 路線圖' : '▢ 路線圖';
  const lt = document.getElementById('minimap-erp-label-top');
  const lb = document.getElementById('minimap-erp-label-bot');
  // 保留放大小鈕（用 innerHTML 重建標籤時把 ⤢ 帶回）
  if(lt) lt.innerHTML = (horiz ? '左圖' : '上圖') + ' <span class="minimap-expand-sm" id="minimap-expand-top" title="放大">⤢</span>';
  if(lb) lb.innerHTML = (horiz ? '右圖' : '下圖') + ' <span class="minimap-expand-sm" id="minimap-expand-bot" title="放大">⤢</span>';
  rewireExpandSm();
  if(shown) buildMinimap();
}

/** 切換路線圖顯示/隱藏。*/
export function toggleMinimap(){
  STATE.ui.minimap = (STATE.ui.minimap === false);
  applyMinimapVisibility();
  scheduleSave();
  emit('state-changed');
}

/** ERP 小標籤內的 ⤢ 是 innerHTML 重建的，需重新綁定。*/
function rewireExpandSm(){
  const t = document.getElementById('minimap-expand-top');
  if(t) t.onclick = (e) => { e.stopPropagation(); openMapModal(); };
  const b = document.getElementById('minimap-expand-bot');
  if(b) b.onclick = (e) => { e.stopPropagation(); openMapModal(); };
}

/** 綁定 canvas 點擊/雙擊、toggle、放大鈕、modal 關閉。boot 時一次。*/
function wireMinimapClicks(){
  const bind = (id, handler) => { const cv = document.getElementById(id); if(cv) cv.addEventListener('click', handler); };
  bind('minimap', function(e){ const idx = hitNode(e.currentTarget, e); if(idx >= 0) goto3DImage(idx); });
  bind('minimap-erp-top', function(e){ const idx = hitNode(e.currentTarget, e); if(idx >= 0) setErpPaneImage('top', idx); });
  bind('minimap-erp-bot', function(e){ const idx = hitNode(e.currentTarget, e); if(idx >= 0) setErpPaneImage('bot', idx); });

  // 雙擊小圖 -> 放大
  ['minimap', 'minimap-erp-top', 'minimap-erp-bot'].forEach(id => {
    const cv = document.getElementById(id);
    if(cv) cv.addEventListener('dblclick', openMapModal);
  });

  const tb = document.getElementById('minimap-toggle');
  if(tb) tb.onclick = toggleMinimap;
  const ex = document.getElementById('minimap-expand');
  if(ex) ex.onclick = (e) => { e.stopPropagation(); openMapModal(); };
  rewireExpandSm();

  // modal 大 canvas：左鍵/右鍵跳圖 + hover 顯檔名
  const mc = document.getElementById('map-modal-canvas');
  if(mc){
    mc.addEventListener('click', function(e){ const idx = hitNode(e.currentTarget, e, true); if(idx >= 0) gotoFromModal(idx, 'left'); });
    mc.addEventListener('contextmenu', function(e){ e.preventDefault(); const idx = hitNode(e.currentTarget, e, true); if(idx >= 0) gotoFromModal(idx, 'right'); });
    mc.addEventListener('mousemove', function(e){
      const tip = document.getElementById('map-modal-tip'); if(!tip) return;
      const idx = hitNode(e.currentTarget, e, true);
      const hd = getHouseData();
      if(idx >= 0 && hd){ tip.textContent = 'img ' + idx + ' · ' + hd.images[idx]; tip.style.opacity = '1'; }
      else { tip.style.opacity = '0'; }
    });
    mc.addEventListener('mouseleave', function(){ const tip = document.getElementById('map-modal-tip'); if(tip) tip.style.opacity = '0'; });
  }
  const closeBtn = document.getElementById('map-modal-close');
  if(closeBtn) closeBtn.onclick = closeMapModal;
  const modal = document.getElementById('map-modal');
  if(modal) modal.addEventListener('click', function(e){ if(e.target === modal) closeMapModal(); });
  // Esc 關閉彈窗（比其它 Esc 處理優先；捕獲階段攔截，避免退出模式）
  document.addEventListener('keydown', function(e){
    if(_modalOpen && e.key === 'Escape'){ e.stopPropagation(); closeMapModal(); }
  }, true);
}

/** @returns {boolean} 放大彈窗是否開啟（供其它模組查詢）。*/
export function isMapModalOpen(){ return _modalOpen; }

/** 訂閱事件：資料/狀態/視角變更時重繪；ERP 開關/排列時切換單/雙。*/
export function initMinimap(){
  on('house-loaded', () => { invalidateMapLayout(); applyMinimapVisibility(); });
  on('state-changed', buildMinimap);   // confirmed 改變 -> getNormLayout 依指紋自動重算
  on('view-changed', buildMinimap);    // 只重繪（佈局指紋未變則沿用 cache，不跳動）
  on('apply-minimap-vis', applyMinimapVisibility);
  wireMinimapClicks();
  applyMinimapVisibility();
}
