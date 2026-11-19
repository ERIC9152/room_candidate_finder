/* =========================================================================
   viewmodes/registry.js：檢視模式插件註冊表(插件化核心)

   「檢視模式」= 主畫面怎麼呈現目前的圖/配對:3D 球面、2D ERP 對比、透視裁切對、
   動態 morph…每個模式一個插件檔,放在 js/viewmodes/,由 index.js 逐行 import 註冊
   (新增/停用一個模式 = 在 index.js 增/刪一行 import,其餘模式不受影響)。

   插件介面(物件):
     id        : string  唯一識別('3d'|'erp'|'crop'|'morph'|…)
     label     : string  按鈕文字(ownButton 時用)
     title     : string  按鈕 tooltip
     ownButton : bool    true = 由 registry 在 #viewmode-btns 自動生成按鈕;
                         false = 沿用既有 UI(如 3d/erp 的舊按鈕,保持行為不變)
     isActive(): bool    模式目前是否顯示中
     enter()   : void    掛載(顯示自己的 overlay / 開啟既有模式)
     exit()    : void    卸載(隱藏 overlay / 關閉既有模式)
     refresh() : void?   活躍中收到 'view-changed'(換圖/換對)時重繪(可省略)

   互斥規則:setViewMode(id) 先 exit 其它 active 插件再 enter 目標。
   相容既有流程:標註模式(modes.js)會繞過 registry 直接 toggleErp() 自動帶入 ERP。registry 訂閱 'view-changed',偵測到 ERP 被外部開啟時自動 exit 其它 overlay 模式,
   確保既有標註流程完全不變(ERP 永遠讓路)。
   ========================================================================= */

import { on, emit } from '../events.js';

const _plugins = [];

/** 註冊一個檢視模式插件(由各 mode-*.js 於 import 時自行呼叫)。*/
export function registerViewMode(p){
  if(!p || !p.id || _plugins.some(x => x.id === p.id)) return;
  _plugins.push(p);
}

/** @returns {Array} 已註冊插件(唯讀用途)。*/
export function getViewModes(){ return _plugins.slice(); }

/** @returns {string} 目前活躍模式 id;無 overlay/ERP 時 = '3d'。*/
export function getViewMode(){
  for(const p of _plugins){
    if(p.id !== '3d' && p.isActive && p.isActive()) return p.id;
  }
  return '3d';
}

/**
 * 切換檢視模式(唯一入口):先退出其它 active 插件,再進入目標。
 * @param {string} id 目標模式;'3d' = 只退出所有 overlay 模式
 */
export function setViewMode(id){
  const cur = getViewMode();
  if(cur === id){ refreshButtons(); return; }
  _plugins.forEach(p => {
    if(p.id !== id && p.id !== '3d' && p.isActive && p.isActive()){
      try { p.exit(); } catch(e){ console.error('viewmode exit error:', p.id, e); }
    }
  });
  const tgt = _plugins.find(p => p.id === id);
  if(tgt && id !== '3d'){
    try { tgt.enter(); } catch(e){ console.error('viewmode enter error:', id, e); }
  }
  refreshButtons();
  emit('view-changed');
}

/** 切換某模式(同一顆鈕再按一次 = 回 3D)。*/
export function toggleViewMode(id){
  setViewMode(getViewMode() === id ? '3d' : id);
}

/** 依 ownButton 插件生成按鈕(#viewmode-btns),高亮隨 getViewMode()。*/
function buildButtons(){
  const host = document.getElementById('viewmode-btns');
  if(!host) return;
  host.innerHTML = '';
  _plugins.filter(p => p.ownButton).forEach(p => {
    const b = document.createElement('button');
    b.id = 'btn-vm-' + p.id;
    b.textContent = p.label || p.id;
    if(p.title) b.title = p.title;
    b.onclick = () => toggleViewMode(p.id);
    host.appendChild(b);
  });
  refreshButtons();
}

/** 刷新按鈕高亮(含 registry 生成鈕;既有 btn-erp 的高亮由 erp.js 自管,不動)。*/
function refreshButtons(){
  const cur = getViewMode();
  _plugins.filter(p => p.ownButton).forEach(p => {
    const b = document.getElementById('btn-vm-' + p.id);
    if(b) b.classList.toggle('on', cur === p.id);
  });
}

/**
 * boot 一次:生成按鈕 + 訂閱 view-changed。
 * view-changed 時:(1) 若 ERP 被外部(標註模式自動帶入/舊按鈕)開啟且有其它 overlay
 * 活躍 → 退出該 overlay(ERP 讓路);(2) 刷新按鈕;(3) 呼叫活躍插件 refresh()。
 */
export function initViewModes(){
  buildButtons();
  on('view-changed', () => {
    const erp = _plugins.find(p => p.id === 'erp');
    if(erp && erp.isActive && erp.isActive()){
      _plugins.forEach(p => {
        if(p.id !== 'erp' && p.id !== '3d' && p.isActive && p.isActive()){
          try { p.exit(); } catch(e){ console.error('viewmode exit error:', p.id, e); }
        }
      });
    }
    refreshButtons();
    const cur = _plugins.find(p => p.id === getViewMode());
    if(cur && cur.refresh){
      try { cur.refresh(); } catch(e){ console.error('viewmode refresh error:', cur.id, e); }
    }
  });
  on('house-loaded', () => {
    // 換房子:退出所有 overlay 模式,回 3D(既有行為:selectHouse 已 clearMode)
    _plugins.forEach(p => {
      if(p.id !== '3d' && p.id !== 'erp' && p.isActive && p.isActive()){
        try { p.exit(); } catch(e){ console.error('viewmode exit error:', p.id, e); }
      }
    });
    refreshButtons();
  });
}
