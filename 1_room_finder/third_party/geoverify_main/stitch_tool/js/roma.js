/* =========================================================================
   roma.js：RoMa 即時比對（點按鈕 -> 對目前 ERP 上下/左右兩面板圖跑 RoMa -> 畫匹配線）
   只依賴 erp.js 的 isErpOn()（單向依賴，仿 ui.js 既有模式，不讓 erp.js 反過來 import 這裡）。
   在 #erp-mode 之上疊一層獨立的 SVG（#erp-match-svg）畫跨面板匹配線。不能放進單一面板的
   .erp-ov，因為那只涵蓋自己那一格，畫不出連到對面面板的線。
   依賴：api / state / nav / erp(isErpOn only) / util / events。
   ========================================================================= */

import { fetchRomaMatch } from './api.js';
import { getHouseData, getHouseName } from './state.js';
import { nav } from './nav.js';
import { isErpOn } from './erp.js';
import { toast } from './util.js';
import { on } from './events.js';

/**
 * 視覺化參數（參考 case_analyse/generate_roma_match_vis.py 的離線繪圖手法搬到即時 SVG 版）：
 *   - 按空間格點分色：同一叢集的匹配點/連線用同一色，一眼看出對應區域。
 *   - 分格挑選連線：每格最多挑 PER_CELL 條，確保線分散在整張圖，不會全擠在信心度最高的單一角落。
 *   - 信心度正規化：用「這批匹配點自己的」min/max 正規化，同時驅動透明度與線寬，
 *     信心分布很窄時（RoMa 常見）也能看出相對強弱。
 */
const GRID_COLS = 8, GRID_ROWS = 4;   // 分格格點數（列x欄），越細分色越多但單格線數上限仍是 PER_CELL
const PER_CELL = 3;                    // 每格最多貢獻幾條線
const MAX_LINES = 100;                 // 全部連線數上限（分格挑選後再依信心裁到這個數字）

/** 分類色盤（8 色固定順序，深色背景可讀性已驗證），依格點編號循環指派。*/
const CELL_COLORS = [
  '#3987e5', '#199e70', '#c98500', '#008300',
  '#9085e9', '#e66767', '#d55181', '#d95926',
];

let _active = false;      // overlay 是否顯示中
let _loading = false;
let _data = null;         // { matches, w, h, topIdx, botIdx } | null

/** @returns {boolean} overlay 目前是否顯示。*/
export function isRomaActive(){ return _active; }

/** 按鈕點擊：目前顯示中且仍對應同一對圖 -> 關閉；否則對目前面板配對重新抓取並畫線。*/
export function toggleRomaMatch(){
  if(_loading) return;
  const stillFresh = _active && _data && _data.topIdx === nav.erpTopImg && _data.botIdx === nav.erpBotImg;
  if(stillFresh){ clearRomaOverlay(); return; }
  runRomaMatch();
}

/** 對目前 ERP 上/下（或左/右）面板的兩張圖跑 RoMa 比對並畫線。*/
function runRomaMatch(){
  const hd = getHouseData();
  const house = getHouseName();
  if(!hd || !house || !isErpOn()) return;
  const topIdx = nav.erpTopImg, botIdx = nav.erpBotImg;
  if(topIdx < 0 || botIdx < 0 || topIdx === botIdx){
    toast('尚未選定兩張不同的圖片');
    return;
  }
  const imgA = hd.images[topIdx], imgB = hd.images[botIdx];
  const confEl = document.getElementById('roma-conf-input');
  let conf = confEl ? parseFloat(confEl.value) : 0.5;
  if(!(conf >= 0 && conf <= 1)) conf = 0.5;   // 防呆：空白/非數字/超出範圍時退回預設
  _loading = true;
  setButtonLoading(true);
  toast('RoMa 比對中（conf>=' + conf + '）…（第一次呼叫可能需要載入模型，請稍候）');
  fetchRomaMatch(house, imgA, imgB, conf).then(res => {
    _loading = false;
    setButtonLoading(false);
    if(!res || res.error){
      toast('RoMa 比對失敗：' + (res && res.error ? res.error : '未知錯誤'));
      return;
    }
    if(!Array.isArray(res.matches)){
      toast('RoMa 比對回傳格式異常');
      return;
    }
    // 抓取期間使用者可能已換圖：只在仍是同一對時才顯示，否則資料已過期，直接丟棄。
    if(topIdx !== nav.erpTopImg || botIdx !== nav.erpBotImg){
      toast('面板圖片已變更，RoMa 結果已捨棄，請重新點擊');
      return;
    }
    _data = { matches:res.matches, w:res.w, h:res.h, topIdx, botIdx };
    _active = true;
    setButtonOn(true);
    drawRomaOverlay();
    toast('RoMa 比對完成：' + res.matches.length + ' 個匹配點');
  }).catch(e => {
    _loading = false;
    setButtonLoading(false);
    toast('RoMa 比對請求失敗：' + e);
  });
}

/** 清除 overlay 與快取的匹配資料。*/
export function clearRomaOverlay(){
  _active = false;
  _data = null;
  setButtonOn(false);
  const svg = document.getElementById('erp-match-svg');
  if(svg) svg.innerHTML = '';
  const badge = document.getElementById('roma-match-badge');
  if(badge) badge.style.display = 'none';
}

/**
 * 依目前面板實際螢幕位置重繪匹配線（resize / 換圖 / renderERP 完成後呼叫）。
 * 若目前面板圖已不是抓資料時的那兩張（使用者換圖了），直接清除，絕不畫錯圖的線。
 */
function drawRomaOverlay(){
  const svg = document.getElementById('erp-match-svg');
  if(!svg || !_active || !_data) return;
  if(_data.topIdx !== nav.erpTopImg || _data.botIdx !== nav.erpBotImg){
    clearRomaOverlay();
    return;
  }
  const innerTop = document.getElementById('erp-inner-top');
  const innerBot = document.getElementById('erp-inner-bot');
  if(!innerTop || !innerBot) return;
  const svgRect = svg.getBoundingClientRect();
  const rTop = innerTop.getBoundingClientRect();
  const rBot = innerBot.getBoundingClientRect();
  if(!svgRect.width || !svgRect.height || !rTop.width || !rBot.width){
    svg.innerHTML = '';
    return;
  }

  const W = _data.w || 1, H = _data.h || 1;
  // 原圖像素座標 -> 該面板 .erp-inner 的螢幕座標 -> 相對 svg 自身（inset:0 於 #erp-mode）的座標。
  // 與 erp.js 的 xOf(az) 同一套「比例換算」邏輯，這裡用像素比例。
  const toTopXY = (x, y) => [
    rTop.left - svgRect.left + (x / W) * rTop.width,
    rTop.top - svgRect.top + (y / H) * rTop.height,
  ];
  const toBotXY = (x, y) => [
    rBot.left - svgRect.left + (x / W) * rBot.width,
    rBot.top - svgRect.top + (y / H) * rBot.height,
  ];

  const matches = _data.matches;
  if(!matches.length){
    svg.innerHTML = '';
    const badge = document.getElementById('roma-match-badge');
    if(badge) badge.style.display = 'none';
    return;
  }

  // 信心度正規化（用這批匹配點自己的 min/max）。RoMa 的信心值常常擠在一個
  // 窄範圍內，固定門檻會讓大多數線看起來一樣淡）。
  const confs = matches.map(m => m[4]);
  const minC = Math.min.apply(null, confs), maxC = Math.max.apply(null, confs);
  const range = Math.max(maxC - minC, 1e-6);
  const normConf = (c) => (c - minC) / range;

  // 依 A 圖上的座標算格點編號（0 .. GRID_COLS*GRID_ROWS-1），色盤依編號循環指派。
  const cellIndex = (x, y) => {
    const cx = Math.min(GRID_COLS - 1, Math.max(0, Math.floor((x / W) * GRID_COLS)));
    const cy = Math.min(GRID_ROWS - 1, Math.max(0, Math.floor((y / H) * GRID_ROWS)));
    return cy * GRID_COLS + cx;
  };
  const colorOf = (x, y) => CELL_COLORS[cellIndex(x, y) % CELL_COLORS.length];

  // 分格挑選：每格最多 PER_CELL 條「該格內信心度最高」的線，確保整張圖都有代表，而不是
  // 全部集中在信心度最高的單一叢集（例如門洞附近常出現數百個高信心匹配點）。
  const buckets = new Map();
  matches.forEach((m, i) => {
    const cell = cellIndex(m[0], m[1]);
    if(!buckets.has(cell)) buckets.set(cell, []);
    buckets.get(cell).push(i);
  });
  let picked = [];
  buckets.forEach(idxs => {
    idxs.sort((a, b) => matches[b][4] - matches[a][4]);
    picked.push(...idxs.slice(0, PER_CELL));
  });
  picked.sort((a, b) => matches[b][4] - matches[a][4]);
  picked = picked.slice(0, MAX_LINES);

  const ns = 'http://www.w3.org/2000/svg';
  const frag = document.createDocumentFragment();
  picked.forEach(i => {
    const [xa, ya, xb, yb, conf] = matches[i];
    const [x1, y1] = toTopXY(xa, ya);
    const [x2, y2] = toBotXY(xb, yb);
    const t = normConf(conf);
    const op = 0.22 + 0.65 * t;
    const width = 0.6 + 2.0 * t;
    const col = colorOf(xa, ya);

    const line = document.createElementNS(ns, 'line');
    line.setAttribute('x1', x1); line.setAttribute('y1', y1);
    line.setAttribute('x2', x2); line.setAttribute('y2', y2);
    line.setAttribute('stroke', col);
    line.setAttribute('stroke-width', String(width));
    line.setAttribute('opacity', String(op));
    frag.appendChild(line);

    [[x1, y1], [x2, y2]].forEach(([cx, cy]) => {
      const c = document.createElementNS(ns, 'circle');
      c.setAttribute('cx', cx); c.setAttribute('cy', cy);
      c.setAttribute('r', String(1.8 + 1.4 * t));
      c.setAttribute('fill', col);
      c.setAttribute('opacity', String(Math.min(0.95, op + 0.15)));
      frag.appendChild(c);
    });
  });
  svg.innerHTML = '';
  svg.appendChild(frag);

  const badge = document.getElementById('roma-match-badge');
  if(badge){
    badge.style.display = 'block';
    badge.textContent = 'RoMa: ' + matches.length + ' 匹配點（分格取樣顯示 ' + picked.length + ' 條）';
  }
}

function setButtonOn(v){
  const btn = document.getElementById('btn-roma-match');
  if(btn) btn.classList.toggle('on', !!v);
}

function setButtonLoading(v){
  const btn = document.getElementById('btn-roma-match');
  if(!btn) return;
  btn.disabled = !!v;
  btn.textContent = v ? 'RoMa 比對中…' : 'RoMa 比對';
}

/**
 * 依 ERP 開關狀態顯示/隱藏比對鈕；關閉 ERP 時一併清 overlay；開著且 overlay 啟用中則重繪
 * （面板換圖/視角改變後座標會變）。掛在 view-changed 上，與 erp.js 現有的重繪時機一致，
 * 不需要 erp.js 額外 import 這個模組（單向依賴，仿 ui.js 既有架構）。
 */
function onNavChange(){
  const on_ = isErpOn();
  const btn = document.getElementById('btn-roma-match');
  const confWrap = document.getElementById('roma-conf-wrap');
  if(btn) btn.style.display = on_ ? 'inline-block' : 'none';
  if(confWrap) confWrap.style.display = on_ ? 'inline-flex' : 'none';
  if(!on_){ if(_active) clearRomaOverlay(); return; }
  if(_active) drawRomaOverlay();
}

/** 綁定按鈕點擊與重繪訂閱（boot 一次）。*/
export function initRoma(){
  const btn = document.getElementById('btn-roma-match');
  if(btn) btn.onclick = toggleRomaMatch;
  on('resize', () => { if(_active) drawRomaOverlay(); });
  on('erp-derived', onNavChange);
  on('view-changed', onNavChange);
}
