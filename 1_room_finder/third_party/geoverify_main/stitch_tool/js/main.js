/* =========================================================================
   main.js：組裝與啟動（boot）
   1) 初始化 three.js 場景與各 UI 訂閱者（minimap/compass/queue/info/erp/scene3d）。
   2) 綁定 UI 與快捷鍵。
   3) 載入房子清單、選房子（讀 house data + state），emit 'house-loaded' 觸發所有重繪。
   4) 連接跨模組事件（reject-confirmed）。
   模組間避免評估期循環：低層不 import 高層；跨層互動走 events 或此處注入。
   ========================================================================= */

import { on, emit } from './events.js';
import { fetchHouses, fetchHouseData, fetchState } from './api.js';
import { setHouseName, setHouseData, normalizeState } from './state.js';
import { nav, gotoCurrentPair } from './nav.js';
import { initThree, initScene3d } from './scene3d.js';
import { initMinimap } from './minimap.js';
import { initCompass } from './compass.js';
import { initQueue, resetStatFilter } from './queue.js';
import { initInfo } from './info.js';
import { initSrcBar } from './srcbar.js';
import { initErp, initRecoPanel } from './erp.js';
// import { initRoma } from './roma.js';  // RoMa 即時比對已停用:改以共視區域取代 raw RoMa
// 顯示。UI 的按鈕與 roma-conf-input 已移除;js/roma.js 與後端 /api/roma_match、
// 離線 cache 仍保留,需要時可再啟用。
import { initViewModes } from './viewmodes/index.js';   // 檢視模式插件系統
import { clearMode } from './modes.js';
import { rejectConfirmed } from './actions.js';
import { initBatchPanel } from './batch-panel.js';
import { wireUI, applyUiToggles, syncThresholdUI } from './ui.js';
import { toast } from './util.js';

/**
 * 載入房子清單填入下拉，並選定測試房子（或第一個）。
 */
function loadHousesList(preferHouse){
  return fetchHouses().then(list => {
    const sel = document.getElementById('house-select'); if(!sel) return;
    sel.innerHTML = '';
    list.forEach(h => {
      const o = document.createElement('option'); o.value = h.house;
      o.textContent = h.house + '  (' + h.n_images + 'img / ' + h.n_pairs + 'pr)';
      o.dataset.meta = h.n_gt_connected_pairs;
      sel.appendChild(o);
    });
    const want = preferHouse || 'DollhouseTask_58472_NoOutdoor';
    if(list.some(h => h.house === want)) sel.value = want;
    else if(list[0]) sel.value = list[0].house;
    sel.onchange = () => selectHouse(sel.value);
    if(sel.value) selectHouse(sel.value);
  }).catch(e => toast('讀取房子清單失敗: ' + e));
}

/** 「＋加房」（面向未知資料）：資料夾選擇器 → 逐檔上傳到 bundle 根 → 刷新清單並選中新房。
    瀏覽器拿不到磁碟路徑，故上傳檔案內容；資料夾內若有 HOTSPOT 檔（不限層級，例如混在
    images/ 裡也找得到）且房子資料本身還沒有 GT，伺服器讀取時會即時算出 GT 對照；
    房號對不上（常見誤帶了別棟房子的標註）會印在伺服器 log，不會安靜地不顯示。*/
function wireAddHouse(){
  const btn = document.getElementById('btn-add-house');
  const inp = document.getElementById('add-house-input');
  if(!btn || !inp) return;
  btn.onclick = () => inp.click();
  inp.onchange = async () => {
    const files = [...inp.files];
    inp.value = '';                          // 允許重選同一資料夾
    if(!files.length) return;
    const house = files[0].webkitRelativePath.split('/')[0];
    const need = house + '/' + house + '.json';
    if(!files.some(f => f.webkitRelativePath === need)){
      toast('這不是房子產出包（缺 ' + house + '.json）');
      return;
    }
    const meta = document.getElementById('house-meta');
    let done = 0;
    for(const f of files){
      try{
        const r = await fetch('/api/bundle_put?rel=' + encodeURIComponent(f.webkitRelativePath),
                              { method:'POST', body:f });
        if(!r.ok) throw new Error(await r.text());
      }catch(e){ toast('上傳失敗 ' + f.webkitRelativePath + '：' + e.message); return; }
      done++;
      if(meta) meta.textContent = '上傳 ' + house + ' … ' + done + '/' + files.length;
    }
    toast('已加入 ' + house + '（' + files.length + ' 檔）');
    loadHousesList(house);
  };
}

/**
 * 選一間房子：重置導覽/模式/過濾，讀 house data 與 state，emit 'house-loaded'。
 * [bug 1 修復點之一] 'house-loaded' 一到，minimap/queue/compass 立即重繪，不再空白。
 * @param {string} name
 */
function selectHouse(name){
  setHouseName(name);
  nav.curPair = null; nav.curImage = -1; nav.viewFrom = null; nav.reciprocalKey = null;
  nav.erpTopImg = -1; nav.erpBotImg = -1;
  clearMode();
  resetStatFilter();
  emit('view-changed');   // 清空目前 markers/資訊由 loadImageView 之後接手

  const iName = document.getElementById('i-name');
  if(iName) iName.textContent = '載入資料…';

  Promise.all([
    fetchHouseData(name),
    fetchState(name)
  ]).then(([hd, st]) => {
    setHouseData(hd);
    normalizeState(st);
    applyUiToggles();
    syncThresholdUI();
    const meta = document.getElementById('house-meta');
    if(meta) meta.textContent = hd.n_images + ' 圖 · ' + hd.pairs.length + ' 對' +
      (hd.has_gt === false ? '（無 GT 真值）' : ' · GT連通 ' + hd.n_gt_connected_pairs);

    emit('house-loaded');   // minimap/queue 訂閱重繪（根治選房後空白）
    gotoCurrentPair();      // 進入第一個 pending（會 loadImageView + renderCompass）
  }).catch(e => toast('讀取房子資料失敗: ' + e));
}

/** 啟動。*/
(function boot(){
  if(typeof THREE === 'undefined'){
    const el = document.getElementById('i-name');
    if(el) el.textContent = 'three.js 載入失敗';
    return;
  }
  // 場景與訂閱者
  initThree();
  initScene3d();
  initMinimap();
  initCompass();
  initQueue();
  initInfo();
  initSrcBar();
  initErp();
  // 綁定批量否決面板 與 unconn 推薦面板的按鈕（否則面板可開、按鈕無反應）
  initBatchPanel();
  initRecoPanel();
  // initRoma();  // 2026-08-05 停用，見上方 import 註記
  initViewModes();   // 生成新模式按鈕 + 模式互斥(既有 3D/ERP 行為不變)

  // UI 綁定
  wireUI();

  // 跨模組事件：右鍵綠圈 -> 重新否決
  on('reject-confirmed', ({ i, j }) => rejectConfirmed(i, j));

  // 載入房子
  wireAddHouse();
  loadHousesList();
})();
