/* =========================================================================
   compass.js：羅盤條（底部 360° 尺）：renderCompass / resizeCompass / drawCompass
   drawCompass 每幀重畫刻度（訂閱 'frame'）；resizeCompass 訂閱 'resize'；
   renderCompass(imgIdx) 由 scene3d 於重畫標記時呼叫、並訂閱 'view-changed'。
   依賴：config / geometry / clustering / state / nav / scene3d(getLon,getCamera)。
   刻度顏色/幾何逐字沿用原 app.js。
   ========================================================================= */

import { COL } from './config.js';
import { azToLon } from './geometry.js';
import { clusterDoors } from './clustering.js';
import { STATE, getHouseData, pairStatus, pairKey } from './state.js';
import { nav } from './nav.js';
import { getLon, getCamera } from './scene3d.js';
import { on } from './events.js';

/** 目前羅盤上顯示的門刻度。*/
let compassDoors = [];

/**
 * 依目前圖重建羅盤刻度資料與標題（偵測到 N 個門 / M 已連）。
 * @param {number} imgIdx
 */
export function renderCompass(imgIdx){
  const wrap = document.getElementById('compass-wrap'), dc = document.getElementById('door-count');
  compassDoors = [];
  const hd = getHouseData();
  if(!hd || !STATE.ui.compass){
    if(wrap) wrap.style.display = 'none';
    if(dc) dc.style.display = 'none';
    return;
  }
  const thEl = document.getElementById('threshold');
  const th = thEl ? +thEl.value : 0.5;
  let nDoors = 0, nConf = 0;
  if(STATE.ui.pairwise){
    clusterDoors(imgIdx, hd, STATE, pairStatus, pairKey).forEach(cl => {
      if(cl.status !== 'confirmed' && cl.bestProb < th) return;
      compassDoors.push({ az:cl.repAz, status:cl.status });
      nDoors++; if(cl.status === 'confirmed') nConf++;
    });
  }
  const ex = (STATE.extra_channels[imgIdx] || STATE.extra_channels[String(imgIdx)] || []);
  ex.forEach(a => { compassDoors.push({ az:a, status:'extra' }); });
  if(wrap) wrap.style.display = 'block';
  if(dc){
    dc.style.display = 'block';
    dc.textContent = '此圖偵測到 ' + nDoors + ' 個門候選（' + nConf + ' 已連'
      + (ex.length ? ('，手動 ' + ex.length) : '') + '）';
  }
  resizeCompass();
}

/** 依容器寬度調整 canvas 尺寸。*/
export function resizeCompass(){
  const cv = document.getElementById('compass'); if(!cv) return;
  const w = cv.clientWidth || (cv.parentNode && cv.parentNode.clientWidth) || 0;
  if(w && cv.width !== w){ cv.width = w; cv.height = 46; }
}

/** 每幀重畫羅盤（FOV 視窗、正中線、門刻度）。*/
export function drawCompass(){
  const wrap = document.getElementById('compass-wrap');
  if(!wrap || wrap.style.display === 'none') return;
  const cv = document.getElementById('compass'); if(!cv) return;
  const g = cv.getContext('2d');
  const W = cv.width, H = cv.height; if(!W) return;
  const camera = getCamera(); if(!camera) return;
  const lon = getLon();
  g.clearRect(0, 0, W, H);
  const fovFrac = camera.fov / 360;
  g.fillStyle = 'rgba(52,152,219,.18)'; g.fillRect(W / 2 - W * fovFrac / 2, 0, W * fovFrac, H);
  g.strokeStyle = 'rgba(52,152,219,.6)'; g.lineWidth = 1; g.strokeRect(W / 2 - W * fovFrac / 2, 0.5, W * fovFrac, H - 1);
  g.strokeStyle = 'rgba(255,255,255,.35)'; g.beginPath(); g.moveTo(W / 2, 0); g.lineTo(W / 2, H); g.stroke();
  compassDoors.forEach(d => {
    const rel = ((azToLon(d.az) - lon + 540) % 360) - 180;
    const x = W / 2 + (rel / 360) * W;
    const col = d.status === 'confirmed' ? COL.green : d.status === 'extra' ? COL.orange : COL.amber;
    g.globalAlpha = 0.95;
    if(d.status === 'confirmed' || d.status === 'extra'){
      g.fillStyle = col; g.beginPath(); g.arc(x, H / 2, 6, 0, 7); g.fill();
    } else {
      g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); g.arc(x, H / 2, 6, 0, 7); g.stroke();
    }
    g.globalAlpha = 1;
  });
}

/** 訂閱事件：每幀重畫、resize 調整、換視角/狀態時重建刻度。*/
export function initCompass(){
  on('frame', drawCompass);
  on('resize', resizeCompass);
  // renderCompass 需要目前 imgIdx；由 scene3d 於 buildMarkersForImage 內主動呼叫。
  // 這裡再補一個保險：狀態/視角改變時，若有目前圖就重建刻度。
  const rebuild = () => { if(nav.curImage >= 0 && getHouseData()) renderCompass(nav.curImage); };
  on('state-changed', rebuild);
}
