/* =========================================================================
   viewmodes/mode-erp.js：既有 2D ERP 對比模式包成插件(行為不變)。
   實作完全委派 erp.js 的 toggleErp()/isErpOn():
     - 既有入口全部保留:btn-erp 按鈕、E 快捷鍵、標註模式自動帶入(modes.js)。
     - registry 只在需要互斥時透過本插件開/關 ERP;ERP 被外部開啟時,registry 的
       view-changed 監聽會讓其它 overlay 模式讓路(見 registry.initViewModes)。
   停用本插件(從 index.js 移除 import)= ERP 退出檢視模式互斥系統;
   舊按鈕仍可用(標註模式依賴 ERP,完全移除需連同標註模式一起停用)。
   ========================================================================= */

import { registerViewMode } from './registry.js';
import { isErpOn, toggleErp } from '../erp.js';

registerViewMode({
  id: 'erp',
  label: 'ERP對比',
  title: '2D ERP 上下(左右)對比',
  ownButton: false,        // 沿用既有 btn-erp(E),不另生按鈕 → 行為不變
  isActive(){ return isErpOn(); },
  enter(){ if(!isErpOn()) toggleErp(); },
  exit(){ if(isErpOn()) toggleErp(); },
});
