/* =========================================================================
   queue.js：候選佇列 + 統計列 + 統計分類過濾
   rebuildQueue / updateStats / toggleStatFilter / resetStatFilter。
   訂閱 'house-loaded' / 'state-changed' / 'view-changed' 自動重繪。
   依賴：config / state / nav / events。過濾與顯示邏輯逐字沿用原 app.js。
   ========================================================================= */

import { COL } from './config.js';
import { STATE, getHouseData, pairStatus } from './state.js';
import { nav, jumpToPair } from './nav.js';
import { on } from './events.js';

/** 統計分類過濾旗標：'all'|'confirmed'|'rejected'|'eliminated'|'pending'。*/
let filterCategory = 'all';

/**
 * 依排序模式回傳 hd.pairs 的排序副本。'prob'（預設）沿用伺服器已排序的順序（不重排，省事）；
 * 'roma'（UI 文案已改「幾何驗證」，排序鍵沿用 'roma' 字串避免動到舊 state）依 geo_n_inlier
 * （幾何驗證內點數）由高到低，null（不在 _ranking_eval.csv＝未過粗篩/沒跑 RoMa）沉到最後、
  * 組內再依 prob 排（次要排序鍵）。排序鍵取 geo_n_inlier，
 * 判別力比較見 pose_regions/covis_ranking_report.md
 * （roma_conf_* 欄位仍保留於資料層，供向後相容/除錯對照）。
 * 不動 hd.pairs 本身（回傳新陣列）。
 * @param {object} hd house data
 * @returns {Array<object>}
 */
function getSortedPairs(hd){
  const sortEl = document.getElementById('sort-mode');
  const mode = sortEl ? sortEl.value : 'prob';
  if(mode !== 'roma') return hd.pairs;
  return hd.pairs.slice().sort((a, b) => {
    const ra = (a.geo_n_inlier != null) ? a.geo_n_inlier : -1;
    const rb = (b.geo_n_inlier != null) ? b.geo_n_inlier : -1;
    if(rb !== ra) return rb - ra;
    return (b.prob || 0) - (a.prob || 0);
  });
}

/**
 * 過濾門檻要比對的欄位值，依排序模式而定：機率模式比 p.prob；幾何驗證模式比
 * p.geo_inlier_rate（沒有幾何驗證證據的視為 0，門檻 > 0 時會被濾掉；門檻 = 0 時仍會顯示，
 * 與機率模式門檻=0 顯示全部的行為一致）。inlier_rate 天然是 0~1，跟現有機率門檻拉條的刻度
 * 天然吻合，不用另外設計不同量級的拉條；排序仍用 n_inlier（見 getSortedPairs，看重總證據量）。
 * @param {object} p pair
 * @param {boolean} romaMode
 * @returns {number}
 */
function getFilterValue(p, romaMode){
  if(!romaMode) return p.prob;
  return (p.geo_inlier_rate != null) ? p.geo_inlier_rate : 0;
}

/** 重建佇列列表並更新統計。*/
export function rebuildQueue(){
  const hd = getHouseData();
  if(!hd) return;
  const sortModeEl = document.getElementById('sort-mode');
  const romaMode = sortModeEl && sortModeEl.value === 'roma';
  const thEl = document.getElementById('threshold');
  const th = thEl ? +thEl.value : 0.5;
  const hideDoneEl = document.getElementById('hidedone-toggle');
  const hideDone = hideDoneEl && hideDoneEl.checked;
  const gtEl = document.getElementById('gt-toggle');
  const gtOn = gtEl && gtEl.checked;
  const ul = document.getElementById('queue'); if(!ul) return;
  ul.innerHTML = '';
  let shown = 0, remaining = 0;
  getSortedPairs(hd).forEach(p => {
    const st = pairStatus(p);
    if(st === 'pending') remaining++;
    const fv = getFilterValue(p, romaMode);

    if(filterCategory !== 'all'){
      if(st !== filterCategory) return;
      if(st === 'pending' && fv < th) return;
    } else {
      // 顯示過濾：門檻藏低機率(或低 RoMa 信心) pending；已解析者除非 hideDone
      const passesTh = fv >= th || st !== 'pending';
      if(!passesTh) return;
      if(hideDone && st !== 'pending' && st !== 'confirmed') return;   // 藏 rejected/eliminated
    }

    shown++;
    const li = document.createElement('li');
    let isCurrent = false;
    if(nav.curPair && p.i === nav.curPair.i && p.j === nav.curPair.j){ li.className = 'current'; isCurrent = true; }
    if(st !== 'pending') li.className += ' done';
    let badge = '';
    if(gtOn)
      badge = ' <span title="GT" style="color:' + (p.gt_connected ? COL.green : '#777') + '">'
        + (p.gt_connected ? '✓GT' : '·') + '</span>';
    // 每列只顯示「跟目前排序模式一致」的那個數字（機率 vs 幾何驗證內點數），避免兩個不同單位
    // 的數字擠在一起看起來像同一種東西；另一個數字放 title tooltip，滑鼠移上去才看得到。
    const probPct = (p.prob * 100).toFixed(0) + '%';
    let probText, probTitle;
    if(romaMode && p.geo_n_inlier != null){
      probText = '幾何 ' + p.geo_n_inlier;
      probTitle = '機率 ' + probPct;
    } else {
      probText = probPct;
      probTitle = (p.geo_n_inlier != null) ? ('幾何驗證內點 ' + p.geo_n_inlier) : '';
    }
    li.innerHTML = '<span class="dot ' + st + '"></span>' +
      '<span class="pair">' + p.i + '–' + p.j + badge + '</span>' +
      '<span class="prob" title="' + probTitle + '">' + probText + '</span>';
    li.onclick = () => { jumpToPair(p); };
    ul.appendChild(li);
    if(isCurrent) _currentLi = li;                         // 記住目前列，稍後捲到可見
  });
  const qc = document.getElementById('q-count');
  if(qc) qc.textContent = '(' + shown + ')';
  updateStats(remaining);
  scrollCurrentIntoView();
}

/** 目前 pair 對應的佇列列（rebuildQueue 內設定）。*/
let _currentLi = null;

/** 把目前 pair 的佇列列捲到可見（若不在視窗內）。*/
function scrollCurrentIntoView(){
  const li = _currentLi; _currentLi = null;
  if(!li) return;
  const wrap = document.getElementById('queue-wrap');
  if(!wrap) return;
  const liTop = li.offsetTop, liBot = liTop + li.offsetHeight;
  const viewTop = wrap.scrollTop, viewBot = viewTop + wrap.clientHeight;
  if(liTop < viewTop || liBot > viewBot){
    // 置中顯示（若可）
    wrap.scrollTop = Math.max(0, liTop - wrap.clientHeight / 2 + li.offsetHeight / 2);
  }
}

/**
 * 更新統計數字（已確認/已否決/已淘汰/剩餘）。
 * @param {number} [remaining]
 */
export function updateStats(remaining){
  const hd = getHouseData();
  if(remaining === undefined){
    remaining = hd ? hd.pairs.filter(p => pairStatus(p) === 'pending').length : 0;
  }
  const set = (id, v) => { const el = document.getElementById(id); if(el) el.textContent = v; };
  set('st-conf', STATE.confirmed.length);
  set('st-rej', STATE.rejected.length);
  set('st-elim', STATE.eliminated.length);
  set('st-rem', remaining);
}

/**
 * 切換統計分類過濾（相同則回 all），並更新按鈕高亮與佇列。
 * @param {string} cat
 */
export function toggleStatFilter(cat){
  filterCategory = (filterCategory === cat) ? 'all' : cat;
  document.getElementById('btn-stat-conf').classList.toggle('active', filterCategory === 'confirmed');
  document.getElementById('btn-stat-rej').classList.toggle('active', filterCategory === 'rejected');
  document.getElementById('btn-stat-elim').classList.toggle('active', filterCategory === 'eliminated');
  document.getElementById('btn-stat-rem').classList.toggle('active', filterCategory === 'pending');
  rebuildQueue();
}

/** 重置過濾為 all 並清按鈕高亮（換房子/reset 時）。*/
export function resetStatFilter(){
  filterCategory = 'all';
  ['conf', 'rej', 'elim', 'rem'].forEach(key => {
    const el = document.getElementById('btn-stat-' + key);
    if(el) el.classList.remove('active');
  });
}

/** 訂閱事件：房子載入 / 狀態改變 / 視角改變時重建佇列。*/
export function initQueue(){
  on('house-loaded', rebuildQueue);
  on('state-changed', rebuildQueue);
  on('view-changed', rebuildQueue);
  on('reset-filter', resetStatFilter);
}
