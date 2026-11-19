/* =========================================================================
   state.js：標註狀態 STATE + house 資料 + 狀態判定 + 存檔（debounce POST）
   低層模組：import config / api / events。不 import 高層（nav/actions/erp/ui…）。
   STATE schema 與 server API 完全不變（舊 state 檔可直接載入）。

   共享可變狀態（module singleton）：
     STATE：標註內容（confirmed/rejected/eliminated/extra_channels/complete_images/threshold/ui）
     houseName：目前房子名
     houseData：目前房子資料（/api/house 回傳）
   其它模組透過 import 這些具名匯出讀取；STATE 以「就地變更」維持同一物件參考。
   ========================================================================= */

import { UI_DEFAULT, COL } from './config.js';
import { saveState } from './api.js';
import { emit } from './events.js';

/** 建立一份乾淨的 STATE。
 * threshold=機率門檻（機率排序模式用）；romaThreshold=幾何驗證 inlier_rate 門檻（「幾何驗證」
 * 排序模式用；欄位名沿用舊名 romaThreshold 以維持舊 state 檔相容，語意已從 RoMa 平均信心
 * 改為幾何驗證內點率，見 queue.js getFilterValue）。
 * 兩者各自獨立記憶，切換排序模式不會互相覆蓋（0.5 機率跟 0.5 內點率是完全不同的東西）。*/
export function freshState(){
  return {
    confirmed:[], rejected:[], eliminated:[], extra_channels:{},
    complete_images:[], threshold:0.5, romaThreshold:0, ui:Object.assign({}, UI_DEFAULT)
  };
}

/** 共享狀態（就地變更以維持物件參考）。*/
export const STATE = freshState();

/** house 相關共享狀態。以物件包裹，讓 setter 能更新同一參考。*/
export const houseRef = { name:null, data:null };

/** @returns {string|null} 目前房子名 */
export function getHouseName(){ return houseRef.name; }
/** @param {string|null} n */
export function setHouseName(n){ houseRef.name = n; }
/** @returns {object|null} 目前房子資料 */
export function getHouseData(){ return houseRef.data; }
/** @param {object|null} d */
export function setHouseData(d){ houseRef.data = d; }

/**
 * 以就地變更方式，把 st 正規化後套進共享 STATE（維持參考不變）。
 * 舊 state 檔（缺欄位/舊格式）都能安全載入。
 * @param {object} st 伺服器回傳的原始 state
 * @returns {object} STATE（同一參考）
 */
export function normalizeState(st){
  const fresh = freshState();
  const s = (st && typeof st === 'object') ? st : {};
  STATE.confirmed      = Array.isArray(s.confirmed) ? s.confirmed : [];
  STATE.rejected       = Array.isArray(s.rejected)  ? s.rejected.map(p => [p[0], p[1]]) : [];
  STATE.eliminated     = Array.isArray(s.eliminated)? s.eliminated.map(p => [p[0], p[1]]) : [];
  STATE.extra_channels = (s.extra_channels && typeof s.extra_channels === 'object') ? s.extra_channels : {};
  STATE.complete_images= Array.isArray(s.complete_images) ? s.complete_images : [];
  STATE.threshold      = (typeof s.threshold === 'number') ? s.threshold : 0.5;
  STATE.romaThreshold  = (typeof s.romaThreshold === 'number') ? s.romaThreshold : 0;
  STATE.ui             = Object.assign({}, UI_DEFAULT, (s.ui && typeof s.ui === 'object') ? s.ui : {});
  return STATE;
}

/**
 * 以就地變更方式，把 STATE 重置為乾淨狀態（reset 房屋用）。
 * @returns {object} STATE（同一參考）
 */
export function resetState(){
  const f = freshState();
  STATE.confirmed = f.confirmed;
  STATE.rejected = f.rejected;
  STATE.eliminated = f.eliminated;
  STATE.extra_channels = f.extra_channels;
  STATE.complete_images = f.complete_images;
  STATE.threshold = f.threshold;
  STATE.ui = f.ui;
  return STATE;
}

// ---------- 狀態判定（pair 狀態機） ----------

/** 規約化的 pair key（順序無關）。*/
export function pairKey(i, j){ return i < j ? i + '_' + j : j + '_' + i; }
/** list（[[i,j],...]）中是否含此對。*/
export function inList(list, i, j){ return list.some(p => pairKey(p[0], p[1]) === pairKey(i, j)); }
/** 是否已確認。*/
export function isConfirmed(i, j){ return STATE.confirmed.some(c => pairKey(c.i, c.j) === pairKey(i, j)); }
/** 是否已否決。*/
export function isRejected(i, j){ return inList(STATE.rejected, i, j); }
/** 是否已淘汰。*/
export function isEliminated(i, j){ return inList(STATE.eliminated, i, j); }
/**
 * pair 狀態：'confirmed' | 'rejected' | 'eliminated' | 'pending'
 * @param {{i:number,j:number}} p
 * @returns {string}
 */
export function pairStatus(p){
  if(isConfirmed(p.i, p.j)) return 'confirmed';
  if(isRejected(p.i, p.j)) return 'rejected';
  if(isEliminated(p.i, p.j)) return 'eliminated';
  return 'pending';
}

// ---------- 存檔（debounce POST） ----------

let saveTimer = null;
let savePending = false;

/** @returns {boolean} 是否尚有待存變更 */
export function isSavePending(){ return savePending; }

/**
 * 標記未儲存並在 700ms 後 debounce 存檔。會把目前門檻拉條的值寫回 STATE。依排序模式決定寫
 * threshold（機率門檻）還是 romaThreshold（幾何驗證 inlier_rate 門檻），因為同一顆拉條在兩種
 * 模式下代表不同東西，不能不分青紅皂白都寫進同一個欄位（否則會把另一個模式的門檻值覆蓋掉）。
 * 不直接呼叫各 UI 重繪；呼叫端自行 emit('state-changed') 觸發重繪。
 */
export function scheduleSave(){
  const th = document.getElementById('threshold');
  if(th){
    const sortEl = document.getElementById('sort-mode');
    const romaMode = sortEl && sortEl.value === 'roma';
    if(romaMode) STATE.romaThreshold = +th.value;
    else STATE.threshold = +th.value;
  }
  savePending = true;
  const ind = document.getElementById('save-ind');
  if(ind){ ind.textContent = '● 未儲存'; ind.style.color = COL.amber; }
  if(saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(doSave, 700);
}

/** 立即 POST 目前 STATE 到伺服器。*/
export function doSave(){
  if(!houseRef.name) return;
  saveState(houseRef.name, STATE).then(function(){
    savePending = false;
    const s = document.getElementById('save-ind');
    if(s){ s.textContent = '✓ 已儲存'; s.style.color = '#5c6';
           setTimeout(function(){ if(!savePending) s.textContent = ''; }, 1500); }
  }).catch(function(e){
    const s = document.getElementById('save-ind');
    if(s) s.textContent = '⚠ 儲存失敗';
    console.error(e);
  });
}

/** 取消任何待存的 timer（reset 房屋時用）。*/
export function cancelPendingSave(){
  if(saveTimer){ clearTimeout(saveTimer); saveTimer = null; }
  savePending = false;
}
