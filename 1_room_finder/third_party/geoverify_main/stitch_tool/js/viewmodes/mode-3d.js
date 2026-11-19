/* =========================================================================
   viewmodes/mode-3d.js：3D 球面模式(基底層)包成插件。
   three.js 場景永遠掛在 #viewer 底層(scene3d.js),其它模式是蓋在上面的 overlay;
   所以「進入 3D」= 沒有任何 overlay 模式活躍,enter/exit 皆為 no-op
   (setViewMode('3d') 會先退出其它插件,base 層自然露出)。行為與重構前完全一致。
   ========================================================================= */

import { registerViewMode } from './registry.js';

registerViewMode({
  id: '3d',
  label: '3D',
  title: '3D 球面檢視(基底層)',
  ownButton: false,        // 沿用既有導覽(btn-back-3d / E 鍵),不另生按鈕
  isActive(){ return true; },   // 永遠存在的基底層;getViewMode 只在無其它 active 時回報 '3d'
  enter(){},
  exit(){},
});
