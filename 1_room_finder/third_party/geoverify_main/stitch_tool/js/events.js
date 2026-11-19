/* =========================================================================
   events.js：極簡 pub/sub 事件匯流排
   低層模組：不 import 任何其它模組。
   目的：動作函式只 emit，UI 元件（minimap / queue / compass …）訂閱重繪，
   根治「改了狀態卻忘了呼叫重繪」這類 bug（原 buildMinimap 散落 13 處）。

   慣用事件：
     'house-loaded'：選完房子、house data 與 state 就緒後
     'state-changed'：任何 STATE（confirmed/rejected/… 或 ui）寫入後
     'view-changed'：curImage / curPair / 導覽狀態改變後
   ========================================================================= */

const _subs = Object.create(null);   // event name -> [handler,...]

/**
 * 訂閱事件。
 * @param {string} name  事件名稱
 * @param {Function} fn  處理函式（可接收 emit 傳入的 payload）
 * @returns {Function} 取消訂閱的函式
 */
export function on(name, fn){
  (_subs[name] || (_subs[name] = [])).push(fn);
  return function off(){
    const arr = _subs[name];
    if(!arr) return;
    const i = arr.indexOf(fn);
    if(i >= 0) arr.splice(i, 1);
  };
}

/**
 * 發送事件，同步呼叫所有訂閱者。
 * @param {string} name     事件名稱
 * @param {*}      [payload] 傳給訂閱者的資料
 */
export function emit(name, payload){
  const arr = _subs[name];
  if(!arr) return;
  // 複製一份以防訂閱者在回呼中變更訂閱清單
  arr.slice().forEach(fn => {
    try { fn(payload); }
    catch(e){ console.error('event handler error for "'+name+'":', e); }
  });
}
