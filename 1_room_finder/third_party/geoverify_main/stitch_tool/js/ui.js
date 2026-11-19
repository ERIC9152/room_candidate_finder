/* =========================================================================
   ui.js：按鈕綁定 + 快捷鍵 + 門顯示層開關 + 統計過濾 + UI toggle + 動作可用性刷新

     - 換圖箭頭移出圖面：3D 用動作列導覽組的 btn-prev-img/btn-next-img；ERP 用面板控制列
       erp-prev/next-top/bot（控制列在圖框外，不再需要 pointerdown 防冒泡）。
     - 鍵盤：3D ←/→ 換圖；ERP ←/→ 換上(左)面板、Shift+←/→ 換下(右)面板。Y/N/U/R/A/E/Esc 不變。
     - refreshActionAvailability()：集中管理 Y/N/連接/回3D雙鈕 的停用/顯示（訂閱 view-changed/
       state-changed/erp-derived），不散落各處。停用＝btn.disabled + .disabled + title 原因。
   依賴：state / nav / pairing / scene3d / modes / actions / erp / queue / events。
   ========================================================================= */

import { STATE, getHouseData, getHouseName, pairStatus, pairKey, scheduleSave } from './state.js';
import { nav, jumpToPair, gotoCurrentPair } from './nav.js';
import { derivePairForPanes } from './pairing.js';
import { resetView, getLon, setCameraLonTo } from './scene3d.js';
import { clearMode, getMode } from './modes.js';
import {
  actYes, actNo, actAllConn, toggleUnconn, toggleNoRoiMode, resetCurrentHouse, connectUnconnPair
} from './actions.js';
import {
  toggleErp, toggleErpLayout, changeErpPaneImage, change3DViewerImage, renderERP, isErpOn,
  exitErpToImage, getErpDerived, setPaneFocus, getPaneFocus
} from './erp.js';
import { rebuildQueue, toggleStatFilter } from './queue.js';
import { on, emit } from './events.js';
import { getCachedHouseDoors, ensureHouseDoors } from './semantic.js';

/** 依 STATE.ui 勾選門顯示層三個開關 + 語意輔助。*/
export function applyUiToggles(){
  const set = (id, v) => { const el = document.getElementById(id); if(el) el.checked = v; };
  set('tg-pairwise', STATE.ui.pairwise);
  set('tg-compass', STATE.ui.compass);
  set('tg-rings', STATE.ui.rings);
  set('semantic-toggle', STATE.ui.semanticAssist);
  refreshSemanticHint();
}

/**
 * 語意輔助側欄提示。house 級的「有沒有語意圖」訊息，獨立於目前 pair（不需要選到
 * 對、不需要在 ERP 才看得到），讓使用者一勾選就知道這房子能不能用。資料未抓過時先顯示
 * 「載入中」，抓完後 ensureHouseDoors 的 onReady 會再呼叫一次本函式补上結果。
 */
export function refreshSemanticHint(){
  const hintEl = document.getElementById('semantic-hint');
  if(!hintEl) return;
  if(!STATE.ui.semanticAssist){ hintEl.textContent = ''; return; }
  const house = getHouseName();
  if(!house){ hintEl.textContent = ''; return; }
  const doors = getCachedHouseDoors(house);
  if(doors === undefined){
    hintEl.textContent = '載入語意門資料中…';
    ensureHouseDoors(house, () => refreshSemanticHint());
    return;
  }
  if(doors === null){ hintEl.textContent = '此房無語意圖'; return; }
  const n = Object.keys(doors).length;
  hintEl.textContent = n + ' 張圖有門語意資料 · ≥80% 包住才校正，其餘維持 az_ab';
}

/**
 * 依目前排序模式（機率/幾何驗證），把門檻拉條同步成該模式該顯示的東西：拉條數值（讀對應的
 * STATE 欄位）、標籤文字、幾何驗證提示區塊可見度。呼叫時機：房子載入後（main.js）、排序模式
 * 切換後（本檔 wireUI 的 sort-mode onchange）。
 */
export function syncThresholdUI(){
  const sel = document.getElementById('sort-mode');
  const romaMode = sel && sel.value === 'roma';
  const val = romaMode ? STATE.romaThreshold : STATE.threshold;
  const th = document.getElementById('threshold');
  if(th) th.value = val;
  const thVal = document.getElementById('th-val');
  if(thVal) thVal.textContent = (+val).toFixed(2);
  const label = document.getElementById('th-label');
  if(label) label.textContent = romaMode ? '幾何驗證門檻' : '機率門檻';
  const romaHint = document.getElementById('th-roma-hint');
  if(romaHint) romaHint.style.display = romaMode ? 'block' : 'none';
}

/** 綁定所有 UI 事件與快捷鍵。*/
export function wireUI(){
  const bind = (id, ev, fn) => { const el = document.getElementById(id); if(el) el[ev] = fn; };

  bind('btn-yes', 'onclick', actYes);
  bind('btn-no', 'onclick', actNo);
  bind('btn-unconn', 'onclick', toggleUnconn);
  bind('btn-allconn', 'onclick', actAllConn);
  bind('btn-back', 'onclick', gotoBack);
  bind('btn-reset', 'onclick', () => { resetView(); });
  bind('btn-erp', 'onclick', toggleErp);
  bind('btn-erp-layout', 'onclick', toggleErpLayout);
  bind('btn-reset-house', 'onclick', resetCurrentHouse);
  bind('btn-noroi', 'onclick', toggleNoRoiMode);

  // unconn 模式的「連接」鈕（上下面板各 1 橘圈 -> 建立綠色確認邊）
  bind('btn-connect', 'onclick', connectUnconnPair);
  // 任務 3: 面板焦點與單顆回 3D 鈕
  bind('focus-top', 'onclick', () => setPaneFocus('top'));
  bind('focus-bot', 'onclick', () => setPaneFocus('bot'));
  bind('btn-back-3d', 'onclick', () => exitErpToImage(getPaneFocus()));

  // 模式橫幅「退出(Esc)」鈕
  bind('mode-banner-exit', 'onclick', () => { clearMode(); gotoBack(); });

  bind('close-erp-guide', 'onclick', function(e){
    e.stopPropagation();
    const gd = document.getElementById('erp-guide');
    if(gd) gd.style.display = 'none';
  });

  // 3D 檢視器換圖：移到動作列導覽組（不再浮在圖上）
  bind('btn-prev-img', 'onclick', () => change3DViewerImage(-1));
  bind('btn-next-img', 'onclick', () => change3DViewerImage(1));

  // ERP 面板控制列左右箭頭（控制列在圖框外，天然不冒泡到面板；不需 pointerdown 防護）
  bind('erp-prev-top', 'onclick', () => changeErpPaneImage('top', -1));
  bind('erp-next-top', 'onclick', () => changeErpPaneImage('top', 1));
  bind('erp-prev-bot', 'onclick', () => changeErpPaneImage('bot', -1));
  bind('erp-next-bot', 'onclick', () => changeErpPaneImage('bot', 1));

  // 門檻滑桿：機率模式比 STATE.threshold，幾何驗證模式比 STATE.romaThreshold（同一顆拉條依
  // 排序模式雙重用途，見 syncThresholdUI）。拖曳時只更新顯示數字，STATE 寫入交給 scheduleSave
  // （它自己會判斷目前模式該寫哪個欄位）。
  const th = document.getElementById('threshold');
  if(th){
    th.oninput = () => {
      const v = document.getElementById('th-val');
      if(v) v.textContent = (+th.value).toFixed(2);
      rebuildQueue();
      if(isErpOn()) renderERP();
      scheduleSave();
    };
  }
  // GT 開關
  bind('gt-toggle', 'onchange', () => {
    rebuildQueue();
    emit('rebuild-markers');
    if(isErpOn()) renderERP();
  });
  // 隱藏已處理列
  bind('hidedone-toggle', 'onchange', () => {
    rebuildQueue();
    if(isErpOn()) renderERP();
  });

  // 語意輔助開關（用子計畫一門語意圖校正通道方位；預設關，只影響未確認的 pair）。
  // 方位優先序：人工 ROI 校正 > 語意門校正 > az_ab。ERP 直接重畫；其它檢視模式
  // （裁切對/走一遍/區域遮罩）走 'view-changed'，由 registry 呼叫目前模式的 refresh()
  // 重新讀 deriveViewPair，四種呈現才會是同一個角度。
  bind('semantic-toggle', 'onchange', () => {
    const el = document.getElementById('semantic-toggle');
    STATE.ui.semanticAssist = el.checked;
    refreshSemanticHint();
    if(isErpOn()) renderERP();
    emit('view-changed');
    scheduleSave();
  });

  // 佇列排序模式（機率 / 幾何驗證）：切換時門檻拉條跟著換身份（標籤 + 顯示各自記憶的值）。
  bind('sort-mode', 'onchange', () => {
    const hint = document.getElementById('sort-roma-hint');
    const sel = document.getElementById('sort-mode');
    if(hint) hint.style.display = (sel && sel.value === 'roma') ? 'block' : 'none';
    syncThresholdUI();
    rebuildQueue();
  });

  // 門顯示層開關
  ['pairwise', 'compass', 'rings'].forEach(key => {
    const el = document.getElementById('tg-' + key);
    if(el) el.onchange = () => {
      STATE.ui[key] = el.checked;
      emit('rebuild-markers');
      if(isErpOn()) renderERP();
      scheduleSave();
    };
  });

  // 點羅盤條 -> 轉到該方位
  const compassEl = document.getElementById('compass');
  if(compassEl){
    compassEl.addEventListener('click', function(e){
      const r = compassEl.getBoundingClientRect();
      const rel = (((e.clientX - r.left) / r.width) - 0.5) * 360;
      setCameraLonTo((getLon() + rel + 720) % 360);
    });
  }

  // 快捷鍵
  document.addEventListener('keydown', e => {
    if(e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
    // 方向鍵換圖。3D：←/→ 換目前圖；ERP：←/→ 換上(左)面板、Shift+←/→ 換下(右)面板。
    if(e.key === 'ArrowLeft' || e.key === 'ArrowRight'){
      const dir = (e.key === 'ArrowRight') ? 1 : -1;
      if(isErpOn()){
        const focus = getPaneFocus();
        const pane = e.shiftKey ? (focus === 'top' ? 'bot' : 'top') : focus;
        changeErpPaneImage(pane, dir);
      }
      else { change3DViewerImage(dir); }
      e.preventDefault();
      return;
    }
    const k = e.key.toLowerCase();
    if(k === 'y'){ actYes(); }
    else if(k === 'n'){ actNo(); }
    else if(k === 'u'){ toggleUnconn(); }
    else if(k === 'r'){ toggleNoRoiMode(); }
    else if(k === 'a'){ actAllConn(); }
    else if(k === 'e'){ toggleErpShortcut(); }
    else if(k === 'escape'){ clearMode(); gotoBack(); }
  });

  // 統計過濾按鈕
  bind('btn-stat-conf', 'onclick', () => toggleStatFilter('confirmed'));
  bind('btn-stat-rej', 'onclick', () => toggleStatFilter('rejected'));
  bind('btn-stat-elim', 'onclick', () => toggleStatFilter('eliminated'));
  bind('btn-stat-rem', 'onclick', () => toggleStatFilter('pending'));

  // 事件：套用 UI toggle（房子載入 / reset 後）
  on('apply-ui-toggles', applyUiToggles);

  // 集中管理動作可用性。view-changed（3D 導覽/loadImageView）、state-changed（判定後）、
  // erp-derived（ERP 面板衍生完成）都刷新一次。
  on('view-changed', refreshActionAvailability);
  on('state-changed', refreshActionAvailability);
  on('erp-derived', refreshActionAvailability);
  on('pane-focus-changed', refreshActionAvailability);
  refreshActionAvailability();
}

// =========================================================================
//  動作可用性刷新（停用/顯示集中一處）
// =========================================================================
/**
 * 依目前模式 / ERP 開關 / 「衍生配對」是否存在，統一設定：
 *   - Y（連通）：無衍生配對 → 停用（原因：兩張圖不是候選/確認對）；
 *                已確認對 → 停用（原因：已是連通）。
 *   - N（否決）：無衍生配對 → 停用；否則可用（作為否決/撤銷確認入口）。
 *   - 連接鈕：僅 unconn 模式顯示（由 erp.updateErpBackButtons 控制 display，這裡不重複）。
 *   - 回3D雙鈕：unconn 中隱藏（同上，display 由 erp 控制）。此處僅補回3D鈕在無有效面板時的停用。
 * 停用機制：btn.disabled + .disabled class + title 說明原因。
 */
export function refreshActionAvailability(){
  const setDisabled = (id, disabled, reason) => {
    const el = document.getElementById(id);
    if(!el) return;
    el.disabled = !!disabled;
    el.classList.toggle('disabled', !!disabled);
    if(disabled && reason){ el.setAttribute('title', reason); }
    else { el.removeAttribute('title'); }
  };

  // 目前「衍生配對」語意：ERP 用面板衍生結果；3D 用 nav.curPair + hd.pairs/confirmed 判定。
  const info = currentPairAvailability();

  // Y：只有「pending 的候選對」或「rejected/eliminated 的候選對（可重新連通）」時可用。
  //    無配對 or 已確認 → 停用。
  let yDisabled = false, yReason = '';
  if(!info.hasPair){ yDisabled = true; yReason = '目前兩張圖不是候選配對，無法判定連通'; }
  else if(info.status === 'confirmed'){ yDisabled = true; yReason = '此配對已確認連通'; }
  setDisabled('btn-yes', yDisabled, yReason);

  // N：無配對 → 停用；有配對（含已確認：N=撤銷確認）→ 可用。
  let nDisabled = false, nReason = '';
  if(!info.hasPair){ nDisabled = true; nReason = '目前兩張圖不是候選配對，無法否決'; }
  setDisabled('btn-no', nDisabled, nReason);
}

/**
 * 取得「目前配對」的可用性資訊（跨 3D/ERP 收斂）。
 * hasPair：是否有衍生/當前配對（pair 或人工確認邊）；status：pairStatus 或 'confirmed'(人工邊)。
 * @returns {{hasPair:boolean, status:string|null}}
 */
function currentPairAvailability(){
  const hd = getHouseData();
  if(!hd) return { hasPair:false, status:null };

  if(isErpOn()){
    // ERP：直接讀 erp 模組的面板衍生結果（唯一真相）。若尚未 render，退回即時衍生一次。
    let d = getErpDerived();
    if(d.top < 0 || d.bot < 0){ d = derivePairForPanes(nav.erpTopImg, nav.erpBotImg, hd, STATE); }
    if(d.kind === 'none' || !d.pair) return { hasPair:false, status:null };
    // kind==='pair' 用 pairStatus；kind==='confirmed'（人工邊）視為 confirmed
    const status = (d.kind === 'confirmed') ? 'confirmed' : pairStatus(d.pair);
    return { hasPair:true, status };
  }

  // 3D：以 nav.curPair 為準；穿門檢視（curPair=null 但在看某圖）→ 無配對。
  const p = nav.curPair;
  if(!p) return { hasPair:false, status:null };
  // curPair 可能是 hd.pairs 的一員或人工 shim（含 _manual）。人工 shim 且在 confirmed → confirmed。
  if(p._manual){
    const isC = STATE.confirmed.some(c => pairKey(c.i, c.j) === pairKey(p.i, p.j));
    return { hasPair:true, status: isC ? 'confirmed' : 'pending' };
  }
  return { hasPair:true, status: pairStatus(p) };
}

/**
 * E 快捷鍵切換 ERP。
 * - 非 ERP：進 ERP（沿用 toggleErp）。
 * - ERP 中：若有作用中模式先清模式（unconn 等自動帶入者 modes 會一併退出 ERP 並清橘圈）；
 *   若清模式後仍在 ERP（手動進入且無模式），則退出到「上/左圖」的 3D（exitErpToImage('top')）。
 */
function toggleErpShortcut(){
  if(!isErpOn()){ toggleErp(); return; }
  if(getMode() !== 'none'){
    clearMode();                     // modes 會對自動帶入的模式一併退出 ERP
    if(!isErpOn()) return;           // 已退出 ERP：完成
  }
  exitErpToImage(getPaneFocus());    // 退出 ERP：回焦點面板那張圖的 3D
}

/**
 * 回目前對：有 curPair 則跳它，否則進第一個 pending。
 * 若 ERP 面板本來就已經顯示這一對（不論上下/左右順序），保留使用者原本的排列，不呼叫
 * jumpToPair（它一律把 i 排上面、j 排下面）。修正否決/未連接通道等模式退出（Esc／退出鈕／
 * 回目前對鈕）後，圖片上下被強制重排、看起來像互換了的問題。
 */
function gotoBack(){
  if(nav.curPair){
    const p = nav.curPair;
    const paneAlreadyShowsPair = isErpOn() &&
      ((nav.erpTopImg === p.i && nav.erpBotImg === p.j) ||
       (nav.erpTopImg === p.j && nav.erpBotImg === p.i));
    if(paneAlreadyShowsPair){ emit('view-changed'); return; }
    jumpToPair(p);
  }
  else gotoCurrentPair();
}
