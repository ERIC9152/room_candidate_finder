/* =========================================================================
   scene3d.js：three.js 全景場景：初始化 / 動畫 / 相機 / raycast / pointer 事件 /
                材質快取 / 標記建立 / buildMarkersForImage / buildDoorLayer / applyFaint /
                loadImageView（貼圖+相機+標記）。
   依賴全域 THREE。import config/geometry/clustering/state/nav/modes/dispatch/compass/events/util。
   與 nav / dispatch 互相 import，但只在函式體內使用，無評估期循環。
   相機互動、幾何、標記外觀逐字沿用原 app.js。
   ========================================================================= */

/* global THREE */
import { SPHERE_R, MARKER_SIZE, COL, CLUSTER_DELTA } from './config.js';
import { azToPos, posToAz, azToLon, angDiff } from './geometry.js';
import { clusterDoors } from './clustering.js';
import {
  STATE, getHouseData, getHouseName, pairStatus, pairKey, scheduleSave
} from './state.js';
import { nav } from './nav.js';
import { getMode } from './modes.js';
import { handleCircleClick } from './dispatch.js';
import { renderCompass } from './compass.js';
import { emit, on } from './events.js';
import { toast, showLoading, round1 } from './util.js';

// ---------- three.js singletons ----------
let scene, camera, renderer, sphere, raycaster, mouse;
const loader = new THREE.TextureLoader();
/** filename -> THREE.Texture */
const texCache = {};

// ---------- camera / drag state ----------
let isDragCam = false, downX = 0, downY = 0, downLon = 0, downLat = 0;
let lon = 0, lat = 0;
let draggingMarker = null, dragMoved = false;

/** 目前場景中的 THREE.Sprite 標記。*/
let markers = [];

/** 需求 3：目前 hover 的可點圈圈（cursor:pointer + 微放大）。*/
let hoveredMarker = null;
/** hover 放大倍率。*/
const HOVER_SCALE = 1.18;

/** @returns {THREE.Camera} */
export function getCamera(){ return camera; }
/** @returns {number} 目前相機經度 lon */
export function getLon(){ return lon; }

/**
 * 相機面向某方位 az（lon=azToLon(az), lat=0）。
 * @param {number} az
 */
export function faceAz(az){ lon = azToLon(az); lat = 0; }

/**
 * 直接設定相機經度 lon（羅盤點擊用），並歸零 lat。
 * @param {number} newLon
 */
export function setCameraLonTo(newLon){ lon = newLon; lat = 0; }

// =========================================================================
//  SETUP
// =========================================================================
/** 初始化 three.js 場景、相機、球體、事件。*/
export function initThree(){
  const c = document.getElementById('viewer');
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(75, c.clientWidth / c.clientHeight, 0.1, 1100);

  renderer = new THREE.WebGLRenderer({ antialias:true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(c.clientWidth, c.clientHeight);
  c.appendChild(renderer.domElement);

  const geo = new THREE.SphereGeometry(SPHERE_R, 60, 40);
  geo.scale(-1, 1, 1);                                  // inside-out
  sphere = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color:0x808080 }));
  scene.add(sphere);

  raycaster = new THREE.Raycaster();
  mouse = new THREE.Vector2();

  const hud = document.createElement('div');
  hud.id = 'scene3d-hud';
  hud.style.position = 'absolute';
  hud.style.inset = '0';
  hud.style.pointerEvents = 'none';
  hud.style.overflow = 'hidden';
  hud.style.zIndex = '3';
  c.appendChild(hud);

  const el = renderer.domElement;
  el.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('pointerup', onPointerUp);
  el.addEventListener('wheel', onWheel, { passive:true });
  el.addEventListener('contextmenu', onContextMenu);
  window.addEventListener('resize', onResize);
  animate();
}

function onResize(){
  const c = document.getElementById('viewer');
  camera.aspect = c.clientWidth / c.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(c.clientWidth, c.clientHeight);
  emit('resize');
}

function animate(){
  requestAnimationFrame(animate);
  lat = Math.max(-85, Math.min(85, lat));
  const phi = THREE.MathUtils.degToRad(90 - lat);
  const theta = THREE.MathUtils.degToRad(lon);
  const tx = 500 * Math.sin(phi) * Math.cos(theta);
  const ty = 500 * Math.cos(phi);
  const tz = 500 * Math.sin(phi) * Math.sin(theta);
  camera.lookAt(tx, ty, tz);
  renderer.render(scene, camera);
  
  // 更新 HUD 標記位置
  const hud = document.getElementById('scene3d-hud');
  if(hud){
    const W = hud.clientWidth;
    markers.forEach(m => {
      if(m.userData.shape === 'roi' && m.userData.hudEl){
        const pos = m.position.clone().project(camera);
        if(pos.z > 1.0){ // 在相機後方
          m.userData.hudEl.style.display = 'none';
        } else {
          m.userData.hudEl.style.display = '';
          const px = (pos.x * 0.5 + 0.5) * W;
          m.userData.hudEl.style.transform = 'translateX(' + px + 'px)';
        }
      }
    });
  }
  
  emit('frame');
}

// =========================================================================
//  POINTER / DRAG / CLICK
// =========================================================================
function onPointerDown(e){
  if(e.button !== 0) return;            // 只處理左鍵；右鍵保留給刪除橘圈
  updateMouse(e);
  const hit = pickMarker();
  if(hit){                              // 拖曳標記（若可拖）或放開時點擊
    draggingMarker = hit;
    dragMoved = false;
    return;
  }
  isDragCam = true; dragMoved = false;
  downX = e.clientX; downY = e.clientY; downLon = lon; downLat = lat;
}

function onPointerMove(e){
  if(draggingMarker){
    updateMouse(e);
    const p = pickSphere();
    if(p){
      dragMoved = true;
      const m = draggingMarker;
      if(m.userData.draggable){
        const az = posToAz(p);
        m.position.copy(azToPos(az));
        m.userData.az = az;
        emit('view-changed');
        // live-update 已持久化標記的方位
        const role = m.userData.role;
        if(role === 'confirmed' || role === 'reciprocal') updateConfirmedAz(m);
        else if(role === 'extra') persistExtraFromMarkers();
      }
    }
    return;
  }
  if(isDragCam){
    lon = (downX - e.clientX) * 0.1 + downLon;
    lat = (e.clientY - downY) * 0.1 + downLat;
    if(Math.abs(e.clientX - downX) + Math.abs(e.clientY - downY) > 3) dragMoved = true;
    return;
  }
  // 需求 3：非拖曳時，hover 可點圈 -> cursor:pointer + 微放大。
  updateMouse(e);
  const hov = pickMarker();
  setHover((hov && isClickableMarker(hov)) ? hov : null);
}

/** 可被點擊/互動的圈（排除 GT 淡標記與已否決紅圈的 no-op 視覺標記）。*/
function isClickableMarker(m){
  const r = m.userData.role;
  return r !== 'gt' && r !== 'rejected-view';
}

/** 設定 hover 標記：舊的復原 baseScale，新的放大並把 canvas cursor 設 pointer。*/
function setHover(m){
  if(hoveredMarker === m) return;
  if(hoveredMarker){
    const bx = hoveredMarker.userData.baseScaleX || hoveredMarker.userData.baseScale || MARKER_SIZE;
    const by = hoveredMarker.userData.baseScaleY || hoveredMarker.userData.baseScale || MARKER_SIZE;
    hoveredMarker.scale.set(bx, by, 1);
    if(hoveredMarker.userData.hudEl){
      hoveredMarker.userData.hudEl.style.filter = '';
    }
  }
  hoveredMarker = m;
  if(m){
    const bx = m.userData.baseScaleX || m.userData.baseScale || MARKER_SIZE;
    const by = m.userData.baseScaleY || m.userData.baseScale || MARKER_SIZE;
    m.scale.set(bx * HOVER_SCALE, by * HOVER_SCALE, 1);
    if(m.userData.hudEl){
      m.userData.hudEl.style.filter = 'brightness(1.5) drop-shadow(0 0 6px ' + m.userData.color + ')';
    }
  }
  if(renderer && renderer.domElement){
    renderer.domElement.style.cursor = m ? 'pointer' : '';
  }
}

function onPointerUp(){
  if(draggingMarker){
    const m = draggingMarker; draggingMarker = null;
    if(!dragMoved){ onMarkerClick(m); }
    else if(m.userData.draggable){ scheduleSave(); emit('state-changed'); }
    return;
  }
  if(isDragCam){
    isDragCam = false;
    if(!dragMoved && getMode() === 'unconn'){ placeUnconnAtPointer(); }
  }
}

/**
 * 3D 標記點擊 → 統一分派入口。把 sprite 轉成 dispatch 認得的 targetInfo。
 * 收斂了原本 3D 與 ERP 各自處理模式的分歧（見 dispatch.js）。
 * @param {THREE.Sprite} m
 */
function onMarkerClick(m){
  const d = m.userData;
  handleCircleClick({
    role: d.role,
    az: d.az,
    i: d.i,
    j: d.j,
    targetImg: d.targetImg,
    srcImg: (typeof d.i === 'number') ? d.i : nav.curImage,
    source: '3d'
  });
}

function onWheel(e){
  const fov = camera.fov + e.deltaY * 0.05;
  camera.fov = THREE.MathUtils.clamp(fov, 10, 100);
  camera.updateProjectionMatrix();
}

function onContextMenu(e){
  e.preventDefault();                   // 3D 檢視器內不彈瀏覽器選單
  updateMouse(e);
  const m = pickMarker();
  if(!m) return;
  if(m.userData.role === 'extra'){ deleteExtraMarker(m); }
  else if(m.userData.role === 'confirmed' || m.userData.role === 'reciprocal'){
    emit('reject-confirmed', { i:m.userData.i, j:m.userData.j });  // 右鍵綠圈 -> 重新否決
  }
}

function updateMouse(e){
  const r = renderer.domElement.getBoundingClientRect();
  mouse.x = ((e.clientX - r.left) / r.width) * 2 - 1;
  mouse.y = -((e.clientY - r.top) / r.height) * 2 + 1;
}
function pickMarker(){
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(markers, false);
  if(hits.length > 0 && hits[0].object.visible) return hits[0].object;
  
  // HUD 標記垂直區域檢測 (X 軸相近即選中)
  let bestHud = null;
  let minDist = 0.08; // 畫面 X 軸寬度比例的容忍度 (8%)
  for(let i = 0; i < markers.length; i++){
    const m = markers[i];
    if(m.userData.shape === 'roi' && m.userData.hudEl && m.userData.hudEl.style.display !== 'none'){
      const pos = m.position.clone().project(camera);
      if(pos.z < 1.0){ // 在前方
        const dx = Math.abs(pos.x - mouse.x);
        if(dx < minDist){
          minDist = dx;
          bestHud = m;
        }
      }
    }
  }
  return bestHud;
}
function pickSphere(){
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObject(sphere, false);
  return hits.length ? hits[0].point : null;
}

// =========================================================================
//  MARKERS (canvas ring sprite that always faces camera)
// =========================================================================
function ringTexture(color, filled){
  const s = 128, cv = document.createElement('canvas'); cv.width = cv.height = s;
  const g = cv.getContext('2d'); g.clearRect(0, 0, s, s);
  const cx = s / 2, cy = s / 2, R = s * 0.40;
  if(filled){
    g.beginPath(); g.arc(cx, cy, R, 0, 7); g.fillStyle = color; g.globalAlpha = 0.28; g.fill(); g.globalAlpha = 1;
  }
  g.beginPath(); g.arc(cx, cy, R, 0, 7); g.lineWidth = s * 0.09; g.strokeStyle = color; g.stroke();
  g.beginPath(); g.arc(cx, cy, s * 0.055, 0, 7); g.fillStyle = color; g.fill();   // inner dot
  const t = new THREE.Texture(cv); t.needsUpdate = true; return t;
}
const _ringCache = {};
function getRing(color, filled){
  const k = color + (filled ? 'F' : '');
  if(!_ringCache[k]) _ringCache[k] = ringTexture(color, filled);
  return _ringCache[k];
}
/** 門示意貼圖 v2：改成與 ERP 類似的垂直通道頂帶樣式 */
function roiTexture(color, filled){
  const w = 64, h = 128, cv = document.createElement('canvas'); cv.width = w; cv.height = h;
  const g = cv.getContext('2d'); g.clearRect(0, 0, w, h);
  
  // 畫底色 (暗色半透明)
  g.fillStyle = 'rgba(8, 12, 18, 0.72)';
  g.fillRect(8, 8, 48, 112);
  
  // 邊框
  g.strokeStyle = color;
  g.lineWidth = 2;
  g.strokeRect(8, 8, 48, 112);
  
  if(filled){
    g.fillStyle = color;
    g.globalAlpha = 0.3;
    g.fillRect(8, 8, 48, 112);
    g.globalAlpha = 1.0;
  }
  
  // 通道帶頂部白線 (模擬 .erp-passage-bar)
  g.fillStyle = '#ffffff';
  g.fillRect(12, 14, 40, 4);
  
  // 下方飾線 (模擬 .erp-passage-ticks)
  g.fillRect(14, 22, 2, 6);
  g.fillRect(31, 22, 2, 10);
  g.fillRect(48, 22, 2, 6);
  
  const t = new THREE.Texture(cv); t.needsUpdate = true; return t;
}
const _roiCache = {};
function getRoi(color, filled){
  const k = 'door3|' + color + (filled ? 'F' : '');
  if(!_roiCache[k]) _roiCache[k] = roiTexture(color, filled);
  return _roiCache[k];
}

/**
 * 建立一個朝向相機的環形 sprite 標記並加入場景。
 * @param {number} az   方位角
 * @param {string} color 顏色
 * @param {object} [opts] {filled,size,role,draggable,i,j,targetImg,faint}
 * @returns {THREE.Sprite}
 */
export function makeMarker(az, color, opts){
  opts = opts || {};
  const shape = opts.shape || 'ring';
  const tex = (shape === 'roi') ? getRoi(color, opts.filled) : getRing(color, opts.filled);
  const mat = new THREE.SpriteMaterial({ map:tex, depthTest:false, transparent:true });
  const spr = new THREE.Sprite(mat);
  const sz = opts.size || MARKER_SIZE;
  const scaleX = (shape === 'roi') ? sz * 0.8 : sz;
  const scaleY = (shape === 'roi') ? sz * 1.6 : sz;
  spr.scale.set(scaleX, scaleY, 1);
  spr.position.copy(azToPos(az));
  spr.renderOrder = 10;
  spr.userData = { az:az, role:opts.role || '', color:color, shape:shape,
                   draggable:!!opts.draggable, i:opts.i, j:opts.j, targetImg:opts.targetImg,
                   faint:!!opts.faint, baseScaleX:scaleX, baseScaleY:scaleY, baseScale:sz };   // baseScale 供 hover 微放大復原（需求 3）
  
  if(shape === 'roi'){
    spr.visible = false; // 隱藏 3D Sprite，改用 HUD 顯示
    const hudEl = document.createElement('div');
    hudEl.style.position = 'absolute';
    hudEl.style.top = '0';
    hudEl.style.bottom = '0';
    hudEl.style.width = '0';
    hudEl.style.willChange = 'transform';
    
    // vlink (垂直虛線)
    const vlink = document.createElement('div');
    vlink.className = 'erp-passage-vlink show';
    hudEl.appendChild(vlink);
    
    const bandW = 48;
    const settled = (opts.role === 'confirmed' || opts.role === 'reciprocal');
    const stText = opts.role === 'reciprocal' ? '返回通道' : (opts.role === 'confirmed' ? '已確認' : '待確認');
    
    // Top band
    const topBand = document.createElement('div');
    topBand.className = 'erp-main-dot erp-band-top';
    topBand.style.width = bandW + 'px';
    topBand.style.color = color;
    topBand.innerHTML = 
      '<div class="erp-passage-band ' + (settled ? 'settled' : 'pending') + '" style="color:' + color + '">' +
        '<div class="erp-passage-bar"></div>' +
        '<div class="erp-passage-ticks"><span class="t-end"></span><span class="t-mid"></span><span class="t-end"></span></div>' +
        '<div class="erp-passage-label" style="margin-top:2px">' + stText + '</div>' +
      '</div>';
    hudEl.appendChild(topBand);
    
    // Bot band
    const botBand = document.createElement('div');
    botBand.className = 'erp-main-dot erp-band-bot';
    botBand.style.width = bandW + 'px';
    botBand.style.color = color;
    botBand.innerHTML = 
      '<div class="erp-passage-band bot ' + (settled ? 'settled' : 'pending') + '" style="color:' + color + '">' +
        '<div class="erp-passage-bar"></div>' +
        '<div class="erp-passage-ticks"><span class="t-end"></span><span class="t-mid"></span><span class="t-end"></span></div>' +
      '</div>';
    hudEl.appendChild(botBand);
    
    const hudContainer = document.getElementById('scene3d-hud');
    if(hudContainer) hudContainer.appendChild(hudEl);
    spr.userData.hudEl = hudEl;
  }
  
  scene.add(spr); markers.push(spr);
  return spr;
}
/** 移除所有標記。*/
export function clearMarkers(){
  for(let i = 0; i < markers.length; i++) {
    scene.remove(markers[i]);
    if(markers[i].userData.hudEl && markers[i].userData.hudEl.parentNode){
      markers[i].userData.hudEl.parentNode.removeChild(markers[i].userData.hudEl);
    }
  }
  markers = [];
  hoveredMarker = null;                                   // 清 hover 參考，避免懸空
  if(renderer && renderer.domElement){ renderer.domElement.style.cursor = ''; }
}
function setMarkerColor(m, color, filled){
  m.material.map = (m.userData.shape === 'roi') ? getRoi(color, filled) : getRing(color, filled);
  m.material.needsUpdate = true; m.userData.color = color;
}
/** @returns {THREE.Sprite|undefined} 目前的候選（琥珀）標記 */
export function getCandidateMarker(){ return markers.find(m => m.userData.role === 'candidate'); }
/** 把目前候選標記轉為已確認（綠、實心、可穿門）。*/
export function promoteCandidateToConfirmed(targetImg){
  const mk = getCandidateMarker();
  if(mk){ setMarkerColor(mk, COL.green, true); mk.userData.role = 'confirmed'; mk.userData.targetImg = targetImg; }
  return mk;
}

// =========================================================================
//  IMAGE VIEW (texture + camera facing + markers)
// =========================================================================
/**
 * 載入圖 imgIdx 全景；若給 pair 則畫其候選標記並面向 az_ij。
 * @param {number} imgIdx
 * @param {object|null} pair
 */
export function loadImageView(imgIdx, pair){
  nav.curImage = imgIdx;
  const hd = getHouseData();
  const fname = hd.images[imgIdx];
  const nm = document.getElementById('i-name');
  if(nm) nm.textContent = 'img ' + imgIdx + '  ·  ' + fname;
  showLoading(true);
  const apply = (tex) => {
    sphere.material.map = tex; sphere.material.color.set(0xffffff); sphere.material.needsUpdate = true;
    showLoading(false);
    buildMarkersForImage(imgIdx, pair);
    let faceAzVal = null;
    if(pair){ faceAzVal = (pair.i === imgIdx) ? pair.az_ij : pair.az_ji; }
    if(faceAzVal != null){ faceAz(faceAzVal); }
    emit('view-changed');
  };
  if(texCache[fname]){ apply(texCache[fname]); return; }
  loader.load('/images/' + encodeURIComponent(getHouseName()) + '/' + encodeURIComponent(fname),
    tex => { texCache[fname] = tex; apply(tex); },
    undefined,
    err => { showLoading(false); toast('影像載入失敗'); console.error(err); });
}

/**
 * 為目前顯示的圖畫出所有相關標記（確認/橘/候選/GT/門顯示層），並更新羅盤。
 * @param {number} imgIdx
 * @param {object|null} pair
 */
export function buildMarkersForImage(imgIdx, pair){
  clearMarkers();
  const gtEl = document.getElementById('gt-toggle');
  const gtOn = gtEl && gtEl.checked;

  // 1) 觸及此圖的已確認連接 -> 綠（穿門經過的那條為藍），可拖曳、點擊穿門
  STATE.confirmed.forEach(c => {
    let here = null, tgt = null, az = null;
    if(c.i === imgIdx){ az = c.az_i; tgt = c.j; here = true; }
    else if(c.j === imgIdx){ az = c.az_j; tgt = c.i; here = true; }
    if(!here) return;
    const isRecip = (nav.reciprocalKey && pairKey(c.i, c.j) === nav.reciprocalKey);
    // 當前配對的已確認邊 → ROI 方框（與 ERP 主標記語彙一致）；其餘維持圓圈
    const isCurPair = !!(pair && pairKey(c.i, c.j) === pairKey(pair.i, pair.j));
    makeMarker(az, isRecip ? COL.blue : COL.green, { role: isRecip ? 'reciprocal' : 'confirmed',
      filled:true, draggable:true, i:c.i, j:c.j, targetImg:tgt,
      size:isCurPair ? MARKER_SIZE + 4 : MARKER_SIZE, shape:isCurPair ? 'roi' : 'ring' });
  });

  // 2) 使用者未連接通道（橘），可拖曳
  const ex = STATE.extra_channels[imgIdx] || STATE.extra_channels[String(imgIdx)] || [];
  ex.forEach(az => {
    makeMarker(az, COL.orange, { role:'extra', draggable:true, size:MARKER_SIZE });
  });

  // 3) 目前候選（琥珀/藍色反向）：pair 為 pending 且在正確圖時
  if(pair && pairStatus(pair) === 'pending'){
    let az = null, tgt = null;
    if(pair.i === imgIdx){ az = pair.az_ij; tgt = pair.j; }
    else if(pair.j === imgIdx){ az = pair.az_ji; tgt = pair.i; }
    if(az != null){
      const isRecip = (nav.reciprocalKey && pairKey(pair.i, pair.j) === nav.reciprocalKey);
      makeMarker(az, isRecip ? COL.blue : COL.amber, { role:'candidate', draggable:true, filled:false,
        i:pair.i, j:pair.j, targetImg:tgt, size:MARKER_SIZE + 4, shape:'roi' });
    }
  }

  // 4) 已否決/淘汰的目前 pair（需求 1）：畫紅圈（不可拖曳，點擊 no-op/toast）。
  //    讓從佇列點到已否決項時，3D 也看得到紅圈（與 ERP 主標記紅色一致）。
  if(pair){
    const stp = pairStatus(pair);
    if(stp === 'rejected' || stp === 'eliminated'){
      let az = null, tgt = null;
      if(pair.i === imgIdx){ az = pair.az_ij; tgt = pair.j; }
      else if(pair.j === imgIdx){ az = pair.az_ji; tgt = pair.i; }
      if(az != null){
        makeMarker(az, COL.red, { role:'rejected-view', draggable:false, filled:true,
          i:pair.i, j:pair.j, targetImg:tgt, size:MARKER_SIZE + 4, shape:'roi' });
      }
    }
  }

  // 5) GT overlay：gt_az_* 的淡標記
  if(gtOn && pair){
    const gaz = (pair.i === imgIdx) ? pair.gt_az_ij : (pair.j === imgIdx ? pair.gt_az_ji : null);
    if(gaz != null) makeMarker(gaz, COL.gt, { role:'gt', size:MARKER_SIZE, faint:true });
  }
  // 6) 門顯示層（v2）：其它預測未確認門（淡琥珀）
  buildDoorLayer(imgIdx, pair);
  renderCompass(imgIdx);
  applyFaint();
}

function applyFaint(){
  markers.forEach(m => { if(m.userData.faint){ m.material.opacity = 0.45; } });
}

/**
 * 在 3D 場景畫「其它預測未確認門」（淡琥珀空心）。
 * confirmed/active 已在 buildMarkersForImage 他處畫。
 * @param {number} imgIdx
 * @param {object|null} pair
 */
export function buildDoorLayer(imgIdx, pair){
  const hd = getHouseData();
  if(!hd) return;
  const thEl = document.getElementById('threshold');
  const th = thEl ? +thEl.value : 0.5;
  let activeAz = null;
  if(pair && pairStatus(pair) === 'pending')
    activeAz = (pair.i === imgIdx) ? pair.az_ij : (pair.j === imgIdx ? pair.az_ji : null);
  if(STATE.ui.pairwise && STATE.ui.rings){
    clusterDoors(imgIdx, hd, STATE, pairStatus, pairKey).forEach(cl => {
      if(cl.status === 'confirmed') return;
      if(cl.bestProb < th) return;
      if(activeAz != null && angDiff(cl.repAz, activeAz) < CLUSTER_DELTA) return;
      const m = makeMarker(cl.repAz, COL.refdoor, { role:'refdoor', filled:false, size:MARKER_SIZE,
        i:imgIdx, targetImg:cl.repTarget });
      m.material.opacity = 0.5;
    });
  }
}

// =========================================================================
//  MARKER DRAG PERSISTENCE
// =========================================================================
function updateConfirmedAz(m){
  const d = m.userData;
  const c = STATE.confirmed.find(x => pairKey(x.i, x.j) === pairKey(d.i, d.j));
  if(!c) return;
  if(nav.curImage === c.i) c.az_i = round1(d.az);
  else if(nav.curImage === c.j) c.az_j = round1(d.az);
}
/** 由目前橘色標記重建此圖的 extra list。*/
function persistExtraFromMarkers(){
  const key = String(nav.curImage), arr = [];
  markers.forEach(m => { if(m.userData.role === 'extra') arr.push(round1(m.userData.az)); });
  if(arr.length) STATE.extra_channels[key] = arr; else delete STATE.extra_channels[key];
}
/** 刪除一個未連接通道橘圈（右鍵點它）。*/
function deleteExtraMarker(m){
  const idx = markers.indexOf(m); if(idx >= 0) markers.splice(idx, 1);
  scene.remove(m);
  persistExtraFromMarkers();
  scheduleSave();
  renderCompass(nav.curImage);
  emit('state-changed');
  toast('已刪除未連接通道');
}
/** U 模式：在指標處球面放橘圈。*/
function placeUnconnAtPointer(){
  const p = pickSphere(); if(!p) return;
  const az = posToAz(p);
  const key = String(nav.curImage);
  if(!STATE.extra_channels[key]) STATE.extra_channels[key] = [];
  STATE.extra_channels[key].push(round1(az));
  makeMarker(az, COL.orange, { role:'extra', draggable:true, size:MARKER_SIZE });
  toast('已放未連接通道 @' + az.toFixed(0) + '°');
  scheduleSave();
  emit('state-changed');
}

/** 重置相機 fov 到 75。*/
export function resetView(){ camera.fov = 75; camera.updateProjectionMatrix(); }

/** 訂閱事件：GT / 門顯示層開關改變時，重畫目前圖的 3D 標記。*/
export function initScene3d(){
  on('rebuild-markers', () => {
    if(getHouseData() && nav.curImage >= 0) buildMarkersForImage(nav.curImage, nav.curPair);
  });
}
