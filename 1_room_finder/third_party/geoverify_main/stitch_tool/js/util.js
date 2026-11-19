/* =========================================================================
   util.js：小工具（toast / loading / 數值 / 狀態標籤）
   低層模組：不 import 任何其它模組。DOM 操作僅限既有 id。
   ========================================================================= */

/** 四捨五入到小數 1 位。*/
export function round1(x){ return Math.round(x * 10) / 10; }

/** 顯示/隱藏全景載入遮罩。*/
export function showLoading(on){
  const el = document.getElementById('loading');
  if(el) el.style.display = on ? 'flex' : 'none';
}

let _toastT = null;
/** 短暫顯示提示訊息。*/
export function toast(msg){
  const t = document.getElementById('toast');
  if(!t) return;
  t.textContent = msg; t.style.opacity = '1';
  if(_toastT) clearTimeout(_toastT);
  _toastT = setTimeout(() => { t.style.opacity = '0'; }, 2200);
}

/** pair 狀態的中文標籤。*/
export function statusLabel(s){
  return { pending:'待處理', confirmed:'已確認', rejected:'已否決', eliminated:'已淘汰' }[s] || s;
}
