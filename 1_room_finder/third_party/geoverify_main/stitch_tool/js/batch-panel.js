/* =========================================================================
   batch-panel.js：批量否決浮動清單面板
   reject 模式中點任何圈（3D 或 ERP）→ 不再只否決單一 pair，而是：
     以該圈方位所在「聚類」（clustering.clusterDoors 的同 cluster members）取出全部 pending 候選，
     彈出固定於螢幕右側的浮動面板：每列「i–j · prob% · az°」+ checkbox（預設全勾）、
     頂部「全選/全不選」、底部「否決所選(N筆)」「取消」。確認後批量 reject（actions.rejectPairs）。
     面板關閉後 reject 模式維持（可繼續點下一個圈）。只有一筆時也走同一面板（一致性）。

   依賴：config / geometry(angDiff) / state / clustering / actions(rejectPairs) / util。
   不觸碰 nav（面板固定於螢幕側邊，不跟隨圈圈）。DOM 元件於 index.html 預留（#batch-panel）。
   ========================================================================= */

import { CLUSTER_DELTA } from './config.js';
import { angDiff } from './geometry.js';
import { STATE, getHouseData, pairStatus, pairKey } from './state.js';
import { clusterDoors } from './clustering.js';
import { rejectPairs } from './actions.js';
import { toast } from './util.js';
import { nav } from './nav.js';
import { setPanePreview, clearPanePreview, setErpPaneImage, isErpOn } from './erp.js';

/** 目前面板列出的候選（{i,j,prob,az}）。*/
let _items = [];
let _srcWhich = 'top';

/**
 * 找出「imgIdx 上、方位 az 所在聚類」的全部 pending 候選 pair。
 * 先用 clusterDoors 取同方向聚類（repAz 與 az 差 < CLUSTER_DELTA 者），取其 members；
 * 若聚類 members 未涵蓋（例如該圈是本對主標記、剛好落在聚類邊界），退回「所有含 imgIdx、
 * 在此圖看向對方方位與 az 差 < CLUSTER_DELTA 的 pending」。去重、依 prob 降序。
 * @param {number} imgIdx @param {number} az
 * @returns {Array<{i:number,j:number,prob:number,az:number}>}
 */
export function collectClusterPending(imgIdx, az){
  const hd = getHouseData();
  if(!hd || typeof imgIdx !== 'number' || typeof az !== 'number') return [];
  const seen = new Set();
  const out = [];
  const push = (i, j, prob, a) => {
    const k = pairKey(i, j);
    if(seen.has(k)) return;
    seen.add(k);
    const p = hd.pairs.find(x => pairKey(x.i, x.j) === k);
    let other = null;
    let oppAz = null;
    if(p){
      other = (p.i === imgIdx) ? p.j : p.i;
      oppAz = (p.i === other) ? p.az_ij : p.az_ji;
    }
    out.push({ i, j, prob:(prob != null ? prob : 0), az:a, other, oppAz });
  };

  // 1) 同方向聚類的 members（pending 者）
  const clusters = clusterDoors(imgIdx, hd, STATE, pairStatus, pairKey);
  let matched = null, bd = 1e9;
  clusters.forEach(cl => { const d = angDiff(cl.repAz, az); if(d < CLUSTER_DELTA && d < bd){ bd = d; matched = cl; } });
  if(matched){
    matched.members.forEach(m => {
      // member 帶 i/j/prob/status/az（doorCandidates 產出）；只收 pending
      if(m.status === 'pending'){ push(m.i, m.j, m.prob, m.az); }
    });
  }

  // 2) 後備：直接掃 hd.pairs（涵蓋聚類抓不到的邊界情況）
  hd.pairs.forEach(p => {
    if(pairStatus(p) !== 'pending') return;
    let a = null;
    if(p.i === imgIdx) a = p.az_ij; else if(p.j === imgIdx) a = p.az_ji; else return;
    if(a == null) return;
    if(angDiff(a, az) < CLUSTER_DELTA){ push(p.i, p.j, p.prob, a); }
  });

  out.sort((x, y) => (y.prob - x.prob) || (x.i - y.i) || (x.j - y.j));
  return out;
}

/**
 * 開啟批量否決面板：列出 imgIdx 上 az 方位聚類內全部 pending 候選（預設全勾）。
 * @param {number} imgIdx 被點圈的圖
 * @param {number} az 圈方位
 */
export function openBatchReject(imgIdx, az, srcWhich){
  const panel = document.getElementById('batch-panel'); if(!panel) return;
  _srcWhich = srcWhich || (nav && nav.erpTopImg === imgIdx ? 'top' : 'bot');
  _items = collectClusterPending(imgIdx, az);
  if(_items.length === 0){
    toast('此方位附近沒有可否決的待處理候選');
    closeBatchReject();
    return;
  }
  const title = document.getElementById('batch-panel-title');
  if(title) title.textContent = '批量否決 · img ' + imgIdx + ' @' + az.toFixed(0) + '°';
  renderBody();
  updateDoCount();
  panel.style.display = 'flex';
}

/** 繪製面板清單列（含全選/全不選）。*/
function renderBody(){
  const body = document.getElementById('batch-panel-body'); if(!body) return;
  body.innerHTML = '';
  const head = document.createElement('div');
  head.className = 'bp-selall';
  head.innerHTML = '同方向候選 ' + _items.length + ' 筆　<a id="bp-all">全選</a>　<a id="bp-none">全不選</a>';
  body.appendChild(head);
  _items.forEach((it, idx) => {
    const row = document.createElement('label');
    row.className = 'bp-row';
    row.innerHTML =
      '<input type="checkbox" data-idx="' + idx + '">' +
      '<span class="bp-pair">' + it.i + '–' + it.j + '</span>' +
      '<span class="bp-meta">' + (it.prob * 100).toFixed(0) + '% · ' + it.az.toFixed(0) + '°</span>';
    row.addEventListener('click', (e) => {
      if(e.target.tagName === 'INPUT') return;
      if(!isErpOn()) return;
      body.querySelectorAll('.bp-row').forEach(r => r.classList.remove('active'));
      row.classList.add('active');
      const otherWhich = (_srcWhich === 'top') ? 'bot' : 'top';
      setErpPaneImage(otherWhich, it.other);
      setPanePreview(otherWhich, it.oppAz);
    });
    body.appendChild(row);
  });
  body.querySelectorAll('input[type=checkbox]').forEach(cb => cb.addEventListener('change', updateDoCount));
  const all = document.getElementById('bp-all'), none = document.getElementById('bp-none');
  if(all) all.onclick = () => { setAll(true); };
  if(none) none.onclick = () => { setAll(false); };
}

/** 全選/全不選。*/
function setAll(v){
  const body = document.getElementById('batch-panel-body'); if(!body) return;
  body.querySelectorAll('input[type=checkbox]').forEach(cb => { cb.checked = v; });
  updateDoCount();
}

/** 取得目前勾選的候選。*/
function checkedItems(){
  const body = document.getElementById('batch-panel-body'); if(!body) return [];
  const out = [];
  body.querySelectorAll('input[type=checkbox]').forEach(cb => {
    if(cb.checked){ const i = +cb.getAttribute('data-idx'); if(_items[i]) out.push(_items[i]); }
  });
  return out;
}

/** 更新「否決所選(N筆)」鈕文字與停用狀態。*/
function updateDoCount(){
  const n = checkedItems().length;
  const btn = document.getElementById('batch-panel-do'); if(!btn) return;
  btn.textContent = '否決所選 (' + n + '筆)';
  btn.disabled = n === 0;
  btn.classList.toggle('disabled', n === 0);
}

/** 執行批量否決（rejectPairs），關閉面板，reject 模式維持。*/
function doReject(){
  const items = checkedItems();
  if(items.length === 0){ return; }
  rejectPairs(items.map(it => [it.i, it.j]));   // 單次 save + 單次 state-changed + toast 總數
  closeBatchReject();
  // 不清 mode：reject 模式維持，可繼續點下一個圈
}

/** 關閉面板。*/
export function closeBatchReject(){
  const panel = document.getElementById('batch-panel');
  if(panel) panel.style.display = 'none';
  _items = [];
  clearPanePreview();
}

/** 綁定面板按鈕（boot 一次）。*/
export function initBatchPanel(){
  const bind = (id, fn) => { const el = document.getElementById(id); if(el) el.onclick = fn; };
  bind('batch-panel-do', doReject);
  bind('batch-panel-cancel', closeBatchReject);
  bind('batch-panel-close', closeBatchReject);
}
