/* =========================================================================
   modes.js：互斥模式狀態機 + 模式旗標 UI + erpAutoEntered 邏輯
   把原本四個互斥布林（unconn/reject/noRoi/allConn）收斂成單一字串
     mode: 'none' | 'unconn' | 'reject' | 'noRoi' | 'allConn'
   並建立唯一的 setMode() 入口，統一處理：
     - reject/noRoi/allConn 若在 3D 時自動帶入 ERP（記 erpAutoEntered）
     - 離開該模式時，若 ERP 是被自動帶入的，才自動切回 3D
   依賴：erp（isErpOn/toggleErp）、util。與 erp 互相 import，只在函式體內使用，無評估期循環。
   ========================================================================= */

import { isErpOn, toggleErp, updateErpBackButtons } from './erp.js';
import { clearAllUnconn } from './actions.js';
import { closeBatchReject } from './batch-panel.js';
import { closeRecoPanel } from './erp.js';

/** 目前模式（就地變更）。*/
let mode = 'none';
/** ERP 是否為「因模式切換而自動帶入」（決定關模式時是否自動切回 3D）。*/
let erpAutoEntered = false;

/** @returns {string} 目前模式 */
export function getMode(){ return mode; }
/** @returns {boolean} ERP 是否為模式自動帶入 */
export function isErpAutoEntered(){ return erpAutoEntered; }
/** ERP 被關閉時清除自動帶入旗標（由 erp.toggleErp 於關閉時呼叫）。*/
export function clearErpAutoEntered(){ erpAutoEntered = false; }

/** 需要在 3D 時自動帶入 ERP 對比視圖的模式（unconn 也自動進 ERP）。*/
function needsErp(m){ return m === 'unconn' || m === 'reject' || m === 'noRoi' || m === 'allConn'; }

/**
 * 設定模式（唯一入口）。處理自動進/出 ERP 與旗標 UI。
 * @param {string} next 'none'|'unconn'|'reject'|'noRoi'|'allConn'
 */
export function setMode(next){
  const prev = mode;
  if(next === prev){ updateModeFlags(); return; }

  // 離開一個「曾自動帶入 ERP」的模式 -> 若 ERP 仍開著且是自動帶入的，切回 3D
  const leavingAutoErpMode = needsErp(prev) && erpAutoEntered;

  // 離開 reject 模式時，關閉批量否決浮動面板
  if(prev === 'reject'){ closeBatchReject(); }
  // 離開 unconn 模式時，清掉所有剩餘（未連接）的橘圈；並關閉推薦面板
  if(prev === 'unconn'){ clearAllUnconn(); closeRecoPanel(); }

  mode = next;

  // 進入需要 ERP 的模式：若目前不在 ERP，自動帶入並記旗標
  if(needsErp(next)){
    if(!isErpOn()){
      erpAutoEntered = true;
      toggleErp();            // 進入 ERP（toggleErp 內部會再呼叫 updateModeFlags）
    }
  }

  // 若正離開自動帶入 ERP 的模式，且新模式不再需要 ERP，切回 3D
  if(leavingAutoErpMode && !needsErp(next)){
    erpAutoEntered = false;
    if(isErpOn()) toggleErp();  // 切回 3D（toggleErp 內部會再呼叫 updateModeFlags）
  }

  updateModeFlags();
  // ERP「回 3D」兩顆鈕與「連接」鈕的顯示，隨模式改變
  updateErpBackButtons();
}

/** 切換某模式（相同則關閉為 none）。用於按鈕/快捷鍵。*/
export function toggleMode(m){ setMode(mode === m ? 'none' : m); }

/** 直接清為 none（Esc / 動作完成後）。*/
export function clearMode(){ setMode('none'); }

/** 各模式的橫幅設定：名稱、背景、文字色，及 3D/ERP 兩種操作提示。*/
const MODE_BANNER = {
  unconn: { name:'未連接通道模式', bg:'var(--orange)', fg:'#111',
            hint3d:'點球面放橘圈標出「有門但還沒連」（右鍵點橘圈可刪除）',
            hintErp:'上下面板各放一個橘圈對準通道 → 按「連接」建立綠色確認邊（右鍵刪橘圈；離開模式會清除未連接橘圈）' },
  reject: { name:'否決模式', bg:'#e05a3a', fg:'#fff',
            hint3d:'點圈圈 → 側邊列出「同方向候選」清單，勾選要否決的一次批量否決（可繼續點下一個圈）',
            hintErp:'點圈圈 → 側邊列出「同方向候選」清單，勾選要否決的一次批量否決（可繼續點下一個圈）' },
  noRoi:  { name:'無 ROI 模式', bg:'var(--blue)', fg:'#fff',
            hint3d:'點該方位上的圈圈，淘汰此方位附近 7° 內所有候選配對',
            hintErp:'點該方位上的圈圈（黃圈或淡黃圈），淘汰此方位附近 7° 內所有候選配對' },
  allConn:{ name:'此圖全連接模式', bg:'var(--green)', fg:'#111',
            hint3d:'目前圖將被標記全連接，淘汰其餘含此圖的待處理候選',
            hintErp:'點欲標記為全連接的圖片面板，淘汰該圖其餘待處理候選' }
};

/**
 * 依目前 mode 更新動作列按鈕高亮與「模式橫幅」（名稱 + 操作提示 + 退出鈕）。
 * mode flag 升級為固定顯示的橫幅；none 模式隱藏。
 */
export function updateModeFlags(){
  const erp = isErpOn();
  const set = (id, on) => { const el = document.getElementById(id); if(el) el.classList.toggle('on', on); };
  set('btn-unconn', mode === 'unconn');
  set('btn-no',     mode === 'reject');
  set('btn-noroi',  mode === 'noRoi');
  set('btn-allconn', mode === 'allConn');

  const banner = document.getElementById('mode-banner');
  if(!banner) return;
  const cfg = MODE_BANNER[mode];
  if(!cfg){ banner.style.display = 'none'; return; }
  banner.style.display = 'flex';
  banner.style.background = cfg.bg;
  banner.style.color = cfg.fg;
  const nm = document.getElementById('mode-banner-name');
  const ht = document.getElementById('mode-banner-hint');
  if(nm) nm.textContent = cfg.name;
  if(ht) ht.textContent = erp ? cfg.hintErp : cfg.hint3d;
}
