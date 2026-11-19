/* =========================================================================
   viewmodes/mode-walk.js：模式 D「走一遍」

   任務7(2026-07-29)時間軸:
   - holdDoor: A 起手朝 B 相機方位(通道 az)明確停留
   - turnA:    轉到 A 側主區重心並停住(共視區入鏡放慢)
   - fade:     極短交叉淡化(或硬切)到 B
   - holdIn:   B 直接落地、面朝同一塊主區
   固定 FOV(預設 FOV_NEAR 48);approach 拉近與 lookBack 預設關閉。
   END_WITH_LOOKBACK 常數開關可恢復舊結尾(不做成 UI)。
   ========================================================================= */

import { registerViewMode } from './registry.js';
import { getErpCached, renderGnomonic, cropToErpPx } from './gnomonic.js';
import { paintRegionsOn, pointInPoly, sidesFromRegions } from './regions-paint.js';
import { drawPassageBandPersp } from './marker.js';
import { deriveViewPair } from './pairsel.js';
import { getHouseData, getHouseName, pairStatus } from '../state.js';
import { fetchRegions, nadirWarning } from '../api.js';
import { on } from '../events.js';

const SIZE = 512;
const FOV_WIDE = 90;
const FOV_NEAR = 48;
const HOLD_MS = 700;
const HOLD_DOOR_MS = 900;      // 起手朝 B 相機方位的明確停留
const FADE_MS = 300;           // 任務7:極短交叉淡化(「直接跳轉」)
const KEYFRAMES = 12;          // 舊 approach 用;END_WITH_LOOKBACK 時 lookBack 仍用 ROT_FRAMES
const ROT_FRAMES = 12;
const TURN_MIN_DEG = 1.5;
const TURN_DEG_PER_FRAME = 15;
const FRAME_FAST_MS = 100;
const FRAME_SLOW_MS = 350;
const SCORE_N = 5;
const PIN_W = 140;
/** 任務7:預設 false。true 時在 holdIn 後接 lookBack(舊行為)。勿做 UI 按鈕。*/
const END_WITH_LOOKBACK = false;
/** 全程固定 FOV;先做 48,若太窄可改 FOV_WIDE。*/
const FOV_FIXED = FOV_NEAR;

let _root = null, _visible = false;
let _token = 0, _lastKey = '';
let _playing = false, _raf = 0;
let _t0 = 0, _progress = 0;
let _dir = 'ab';
let _prep = null;
let _blocked = null;
let _pinOn = true;
let _regOn = true;
let _bandOn = true;            // 通道頂帶(影片要共識區塊＋通道位置同顯 → 預設開)

/** 任務7 段長:holdDoor → turnA → holdStart → fade → holdIn [→ lookBack?]。
    degenerate:跳過轉向,起手即主區(或通道方位)停留後淡化。*/
function segs(prep){
  const sumOf = list => (list || []).reduce((s, x) => s + x, 0);
  const list = [];
  const degen = prep && prep.degenerate;
  if(!degen){
    list.push({ kind:'holdDoor', ms:HOLD_DOOR_MS });
    if(prep && prep.turnA && prep.turnA.length){
      list.push({ kind:'turnA', ms:Math.max(sumOf(prep.turnADur), 300) });
    }
  }
  list.push({ kind:'holdStart', ms:HOLD_MS * (degen ? 2 : 1) });
  list.push({ kind:'fade',      ms:FADE_MS });
  list.push({ kind:'holdIn',    ms:HOLD_MS * 2 });
  if(END_WITH_LOOKBACK){
    const lbMs = (prep && prep.lookBackDur) ? sumOf(prep.lookBackDur) : ROT_FRAMES * FRAME_FAST_MS;
    list.push({ kind:'lookBack', ms:Math.max(lbMs, 400) });
  }
  return list;
}

function totalMs(prep){ return segs(prep).reduce((s, x) => s + x.ms, 0); }

function atProgress(p, prep){
  const list = segs(prep);
  const T = totalMs(prep);
  let t = Math.max(0, Math.min(1, p)) * T;
  for(let i = 0; i < list.length; i++){
    if(t <= list[i].ms || i === list.length - 1){
      const u = list[i].ms > 0 ? Math.min(1, t / list[i].ms) : 1;
      return { kind:list[i].kind, u, tInSeg:t, segMs:list[i].ms, idx:i };
    }
    t -= list[i].ms;
  }
  return { kind:list[list.length - 1].kind, u:1, tInSeg:0, segMs:1, idx:list.length - 1 };
}

/** 段落序號(段數依退化/有無轉向而異,所以由段索引現算,不寫死 ①①′②②′③)。*/
const SEG_NUM = ['①', '②', '③', '④', '⑤', '⑥', '⑦'];

/** 依各幀時長表找目前幀(approach / lookBack 共用)。*/
function timedFrame(u, frames, durs, scores){
  if(!frames || !frames.length) return { fr:null, score:0 };
  const total = (durs || []).reduce((s, x) => s + x, 0) || 1;
  let t = u * total;
  for(let i = 0; i < frames.length; i++){
    const d = (durs && durs[i]) || FRAME_FAST_MS;
    if(t <= d || i === frames.length - 1){
      return { fr:frames[i], score:scores ? scores[i] : 0, i };
    }
    t -= d;
  }
  return { fr:frames[frames.length - 1], score:0, i:frames.length - 1 };
}

function buildDom(){
  if(_root) return;
  _root = document.createElement('div');
  _root.id = 'walk-mode';
  _root.className = 'viewmode-overlay';
  _root.innerHTML =
    '<div class="vm-head" id="walk-head">走一遍(起手朝通道 → 轉主區 → B 直跳 · FOV' + FOV_FIXED + ')</div>' +
    '<div class="vm-body">' +
      '<div class="crop-pane walk-pane">' +
        '<div class="crop-label" id="walk-label"></div>' +
        '<div class="walk-stage">' +
          '<canvas id="walk-canvas" width="' + SIZE + '" height="' + SIZE + '"></canvas>' +
          '<span id="walk-badge" class="morph-badge walk-badge" style="display:none"></span>' +
          '<span id="walk-phase" class="walk-phase"></span>' +
        '</div>' +
      '</div>' +
    '</div>' +
    '<div class="vm-hint walk-controls">' +
      '<button id="walk-play" type="button">▶ 播放</button>' +
      '<button id="walk-dir" type="button" title="單程方向">A→B</button>' +
      '<button id="walk-pin" type="button" title="穿門後角落釘住 A 最後一幀">釘窗 ON</button>' +
      '<button id="walk-reg" type="button" title="疊加 共視區域遮罩">遮罩 ON</button>' +
      '<button id="walk-band" type="button" title="畫布頂緣的連通道方位標示(只吃方位角)">通道帶 ON</button>' +
      '<input id="walk-scrub" type="range" min="0" max="1000" value="0" step="1" />' +
      '<span id="walk-info"></span>' +
    '</div>';
  const viewer = document.getElementById('viewer');
  if(viewer) viewer.appendChild(_root);

  const play = document.getElementById('walk-play');
  if(play) play.onclick = () => {
    if(_blocked) return;
    if(_playing){ stopPlay(); return; }
    if(_progress >= 0.999) _progress = 0;
    startPlay();
  };
  const dirBtn = document.getElementById('walk-dir');
  if(dirBtn) dirBtn.onclick = () => {
    _dir = _dir === 'ab' ? 'ba' : 'ab';
    dirBtn.textContent = _dir === 'ab' ? 'A→B' : 'B→A';
    stopPlay();
    _lastKey = '';
    refresh();
  };
  const pinBtn = document.getElementById('walk-pin');
  if(pinBtn) pinBtn.onclick = () => {
    _pinOn = !_pinOn;
    pinBtn.textContent = _pinOn ? '釘窗 ON' : '釘窗 OFF';
    pinBtn.classList.toggle('on', _pinOn);
    paint();
  };
  const regBtn = document.getElementById('walk-reg');
  if(regBtn) regBtn.onclick = () => {
    _regOn = !_regOn;
    regBtn.textContent = _regOn ? '遮罩 ON' : '遮罩 OFF';
    regBtn.classList.toggle('on', _regOn);
    _lastKey = '';
    refresh();
  };
  const bandBtn = document.getElementById('walk-band');
  if(bandBtn) bandBtn.onclick = () => {
    _bandOn = !_bandOn;
    bandBtn.textContent = _bandOn ? '通道帶 ON' : '通道帶 OFF';
    bandBtn.classList.toggle('on', _bandOn);
    _lastKey = '';                      // 帶子畫在預渲染幀裡 → 走 rebake
    refresh();
  };
  const scrub = document.getElementById('walk-scrub');
  if(scrub){
    scrub.oninput = () => {
      stopPlay();
      _progress = (+scrub.value) / 1000;
      paint();
    };
  }
}

function setInfo(t){ const el = document.getElementById('walk-info'); if(el) el.textContent = t || ''; }
function setLabel(html){ const el = document.getElementById('walk-label'); if(el) el.innerHTML = html || ''; }
function setPhase(t){ const el = document.getElementById('walk-phase'); if(el) el.textContent = t || ''; }

function setBlocked(reason){
  _blocked = reason || null;
  const play = document.getElementById('walk-play');
  const scrub = document.getElementById('walk-scrub');
  if(play){
    play.disabled = !!_blocked;
    play.title = _blocked || '播放 / 暫停';
    play.textContent = '▶ 播放';
  }
  if(scrub) scrub.disabled = !!_blocked;
}

function setBadge(text){
  const el = document.getElementById('walk-badge');
  if(!el) return;
  if(!text){ el.style.display = 'none'; return; }
  el.style.display = '';
  el.textContent = text;
  el.classList.toggle('bad', true);
}

function mapPose(rj, walkAb){
  if(!rj || rj.error){
    return { ok:false, reason:(rj && rj.error) || '此對沒有區域 JSON' };
  }
  const flags = rj.flags || {};
  if(!rj.pose || flags.no_covisibility){
    return {
      ok:false,
      reason:'此 pair 因 correspondence 過少,無可信任姿態估計',
      conf_used: rj.conf_used,
    };
  }
  const pose = rj.pose;
  const needT = walkAb ? !!rj.swap : !rj.swap;
  const R = needT ? transpose3(pose.R) : pose.R;
  return {
    ok:true,
    R,
    degenerate: !!flags.degenerate_rot_only,
    conf_used: rj.conf_used,
  };
}

function transpose3(M){
  return [
    [M[0][0], M[1][0], M[2][0]],
    [M[0][1], M[1][1], M[2][1]],
    [M[0][2], M[1][2], M[2][2]],
  ];
}

function matVec3(M, v){
  return [
    M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
    M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
    M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2],
  ];
}

function mapAzThroughR(azDeg, R){
  const r = azDeg * Math.PI / 180;
  const d = [Math.sin(r), 0, Math.cos(r)];
  const e = matVec3(R, d);
  const horiz = Math.hypot(e[0], e[2]);
  if(horiz < 1e-4) return null;
  return Math.atan2(e[0], e[2]) * 180 / Math.PI;
}

function normAz(a){ return ((a % 360) + 360) % 360; }

function deltaAz(from, to){
  let d = normAz(to) - normAz(from);
  if(d > 180) d -= 360;
  if(d <= -180) d += 360;
  return d;
}

function easeCos(u){
  return 0.5 - 0.5 * Math.cos(Math.PI * Math.max(0, Math.min(1, u)));
}

/**
 * 一幀上的兩層疊圖。ov = { reg, bandAz }:
 *   reg    共視遮罩(遮罩 OFF 時呼叫端已給 null)：繪製本體在 regions-paint.js
 *   bandAz 該側的通道方位(頂帶標示;方位角-only)：繪製本體在 marker.js
 * walk 的幀是預渲染的,兩層都直接畫進 bake 出來的幀(與遮罩既有管線一致);
 * 兩個開關都走 `_lastKey=''; refresh()` rebake。
 */
function paintOverlays(ctx, ov, az, fov){
  if(!ov) return;
  if(ov.reg) paintRegionsOn(ctx, ov.reg.polys, ov.reg.W, ov.reg.H, az, fov, SIZE);
  if(_bandOn && ov.bandAz != null) drawPassageBandPersp(ctx, ov.bandAz, az, fov, SIZE);
}

/**
 * 共視區「畫面覆蓋分數」0~1。
 * 舊算法用「poly 邊有多少投進裁切圖」。共視區占滿畫面時邊反而在 FOV 外,
 * 分數掉到 0 → 反而加速。改:裁切圖網格取樣 → 反投 ERP → 是否在 poly 內。
 */
function regionInViewScore(sidePolys, W, H, az, fov){
  if(!sidePolys || !sidePolys.length || !W || !H) return 0;
  const N = SCORE_N;
  let hit = 0, total = 0;
  // 只取前 4 個 region 估分(主區足夠)
  const items = sidePolys.length > 4 ? sidePolys.slice(0, 4) : sidePolys;
  for(let iy = 0; iy < N; iy++){
    for(let ix = 0; ix < N; ix++){
      const px = (ix + 0.5) / N * SIZE;
      const py = (iy + 0.5) / N * SIZE;
      const erp = cropToErpPx(px, py, W, H, az, fov, SIZE);
      if(!erp) continue;
      total++;
      const [ex, ey] = erp;
      let ok = false;
      for(const item of items){
        const poly = item.poly;
        if(!poly || poly.length < 3) continue;
        for(const dx of [-W, 0, W]){
          if(pointInPoly(ex + dx, ey, poly)){ ok = true; break; }
        }
        if(ok) break;
      }
      if(ok) hit++;
    }
  }
  if(!total) return 0;
  return Math.min(1, hit / total / 0.25);
}

/** 讓出主線程一幀,避免預渲染卡死 UI。*/
function yieldFrame(){
  return new Promise(r => setTimeout(r, 0));
}

/** 共用的 i/j 兩側 polygon 依「行走方向」改標 start/end(swap 已在共用模組處理)。*/
function sidePolysFrom(rj, walkAb){
  const s = sidesFromRegions(rj);
  return walkAb
    ? { start:s.i, end:s.j, Ws:s.Wi, Hs:s.Hi, We:s.Wj, He:s.Hj }
    : { start:s.j, end:s.i, Ws:s.Wj, Hs:s.Hj, We:s.Wi, He:s.Hi };
}

/** polygon 重心 → 方位角。已把過縫的 x unwrap 成連續值(可超出 [0,W)),
    所以算術平均 x 再換算即可,不需要圓形平均。*/
function polyCentroidAz(poly, W){
  if(!poly || poly.length < 3 || !W) return null;
  let sx = 0;
  for(const p of poly) sx += p[0];
  const az = (sx / poly.length) / W * 360 - 180;
  return isFinite(az) ? normAz(az) : null;
}

/**
 * 主區(過渡兩端的鏡頭錨點)。**面積最大**,但淡化前後必須是「同一塊空間」,
 * 所以先在兩側都有 polygon 的 region 裡取兩側面積和最大者,再各取該 region
 * 在該側的重心;沒有兩側齊備的 region 時才退回逐側各取面積最大。
 * (規格原文寫「每側各取面積最大」。但實測 7069 份 region JSON 裡,25.5% 的
 *  多區 pair 兩側 argmax 落在不同 region,那樣淡化前後看的是兩塊不同空間,
 *  「同色=同一塊」的說服力就沒了。單一主區時兩種規則結果相同。)
 * 區域依 n 降冪排序 → 面積同分時取排前者,結果決定性。
 *
 * ⚠ 語意紅線:重心只是「鏡頭視覺錨點」,**不是**通道方位估計。
 *   2026-07-22 實測區域中心估通道方位誤差中位 39°~62°,az_ab 才是好的通道估計。
 *   不要把這個值回寫或顯示成任何「通道位置」。
 */
function mainRegionAz(sides){
  const biggest = items => items.reduce((best, it) => (!best || it.area > best.area) ? it : best, null);
  const endByK = new Map();
  sides.end.forEach(it => endByK.set(it.k, it));
  const both = [];
  sides.start.forEach(it => {
    const e = endByK.get(it.k);
    if(e) both.push({ k:it.k, s:it, e, area:it.area + e.area });
  });
  let s = null, e = null, k = null;
  if(both.length){
    const best = both.reduce((b, it) => (!b || it.area > b.area) ? it : b, null);
    s = best.s; e = best.e; k = best.k;
  } else {
    s = biggest(sides.start);
    e = biggest(sides.end);
  }
  return {
    k,
    azS: s ? polyCentroidAz(s.poly, sides.Ws) : null,
    azE: e ? polyCentroidAz(e.poly, sides.We) : null,
  };
}

function frameDurFromScore(score){
  return FRAME_FAST_MS + (FRAME_SLOW_MS - FRAME_FAST_MS) * score;
}

async function bakeFovsTimed(erp, az, fovFrom, fovTo, ov, onProg){
  const frames = [], scores = [], durs = [];
  for(let k = 0; k < KEYFRAMES; k++){
    const t = KEYFRAMES === 1 ? 0 : k / (KEYFRAMES - 1);
    const fov = fovFrom + (fovTo - fovFrom) * t;
    const c = document.createElement('canvas');
    c.width = SIZE; c.height = SIZE;
    const ctx = c.getContext('2d');
    renderGnomonic(erp, az, fov, SIZE, ctx);
    paintOverlays(ctx, ov, az, fov);
    const score = (ov && ov.reg)
      ? regionInViewScore(ov.reg.polys, ov.reg.W, ov.reg.H, az, fov)
      : 0;
    frames.push(c);
    scores.push(score);
    durs.push(frameDurFromScore(score));
    if(onProg) onProg(k + 1);
    await yieldFrame();
  }
  return { frames, scores, durs };
}

/** 原地轉向(A 端轉向主區 / B 端轉身回看門):每幀依共視區是否入鏡調時長。*/
async function bakeAzTimed(erp, azFrom, azTo, fov, ov, onProg, baseDone, nFrames){
  const d = deltaAz(azFrom, azTo);
  const N = Math.max(1, nFrames || ROT_FRAMES);
  const frames = [], scores = [], durs = [];
  for(let k = 0; k < N; k++){
    const t = N === 1 ? 0 : k / (N - 1);
    const az = azFrom + d * t;
    const c = document.createElement('canvas');
    c.width = SIZE; c.height = SIZE;
    const ctx = c.getContext('2d');
    renderGnomonic(erp, az, fov, SIZE, ctx);
    paintOverlays(ctx, ov, az, fov);
    const score = (ov && ov.reg)
      ? regionInViewScore(ov.reg.polys, ov.reg.W, ov.reg.H, az, fov)
      : 0;
    frames.push(c);
    scores.push(score);
    durs.push(frameDurFromScore(score));
    if(onProg) onProg((baseDone || 0) + k + 1);
    await yieldFrame();
  }
  return { frames, scores, durs };
}

function bakeOne(erp, az, fov, ov){
  const c = document.createElement('canvas');
  c.width = SIZE; c.height = SIZE;
  const ctx = c.getContext('2d');
  renderGnomonic(erp, az, fov, SIZE, ctx);
  paintOverlays(ctx, ov, az, fov);
  return c;
}

function blit(ctx, fr){ if(fr) ctx.drawImage(fr, 0, 0); }
function cross(ctx, a, b, alpha){
  if(a){ ctx.globalAlpha = 1 - alpha; ctx.drawImage(a, 0, 0); }
  if(b){ ctx.globalAlpha = alpha; ctx.drawImage(b, 0, 0); }
  ctx.globalAlpha = 1;
}

/** 釘窗:穿門後角落續掛 A 端最後一幀。該幀現在是「A 側主區」,標籤照實寫。*/
function drawPin(ctx, pinFrame){
  if(!_pinOn || !pinFrame) return;
  const m = 10;
  const w = PIN_W, h = PIN_W;
  const x = SIZE - w - m, y = m;
  ctx.save();
  ctx.fillStyle = 'rgba(15,22,30,.85)';
  ctx.fillRect(x - 3, y - 3, w + 6, h + 22);
  ctx.strokeStyle = '#f4b400';
  ctx.lineWidth = 2;
  ctx.strokeRect(x - 1, y - 1, w + 2, h + 2);
  ctx.drawImage(pinFrame, x, y, w, h);
  ctx.fillStyle = '#f4b400';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('A 端(對照)', x + w / 2, y + h + 14);
  ctx.restore();
}

function paint(){
  const cv = document.getElementById('walk-canvas');
  if(!cv || !_prep) return;
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, SIZE, SIZE);
  const p = Math.max(0, Math.min(1, _progress));
  const scrub = document.getElementById('walk-scrub');
  if(scrub && document.activeElement !== scrub) scrub.value = String(Math.round(p * 1000));

  const { kind, u, idx } = atProgress(p, _prep);
  const e = easeCos(u);
  const S = _prep.startName, E = _prep.endName;
  const P = _prep;
  const N = SEG_NUM[idx] || '';
  const azS = P.azStart.toFixed(0) + '°', azE = P.azEnter.toFixed(0) + '°';
  let showPin = false;

  const azDoor = (P.doorS != null) ? P.doorS.toFixed(0) + '°' : '—';
  switch(kind){
    case 'holdDoor':
      blit(ctx, P.doorFrame
        || (P.turnA && P.turnA[0])
        || P.startFrame);
      setPhase(N + ' ' + S + ' 起手朝通道(B 相機) ' + azDoor);
      break;
    case 'turnA': {
      const { fr, score } = timedFrame(u, P.turnA, P.turnADur, P.turnAScore);
      blit(ctx, fr);
      setPhase(N + ' ' + S + ' 轉向共識主區 ' + azS + (score > 0.15 ? '(入鏡放慢)' : ''));
      break;
    }
    case 'holdStart':
      blit(ctx, P.startFrame
        || (P.turnA && P.turnA[P.turnA.length - 1])
        || P.doorFrame);
      setPhase(N + ' ' + S + ' ' + (P.noteS === '主區重心' ? '停在共識主區 ' + azS : '通道方位 ' + azS + '(無主區)')
        + (P.degenerate ? ' · 近零基線·同點拍攝' : ''));
      break;
    case 'fade':
      cross(ctx, P.startFrame, P.enterFrame, e);
      setPhase(N + ' B 直跳 · ' + S + '@' + azS + ' → ' + E + '@' + azE + ' · 淡化' + FADE_MS + 'ms');
      showPin = true;
      break;
    case 'holdIn':
      blit(ctx, P.enterFrame);
      setPhase(N + ' 已在 ' + E + ' · ' + (P.noteE === '主區重心'
        ? '正對同一塊共識主區 ' + azE
        : '無主區·維持 R 對齊視線 ' + azE));
      showPin = true;
      break;
    case 'lookBack': {
      const { fr, score } = timedFrame(u, P.lookBack, P.lookBackDur, P.lookBackScore);
      blit(ctx, fr);
      if(score > 0.15){
        setPhase(N + ' ' + E + ' 轉身 · 共視區入鏡(放慢)');
      } else {
        setPhase(N + ' ' + E + ' 轉身回看通道');
      }
      showPin = true;
      break;
    }
    default:
      break;
  }
  if(showPin) drawPin(ctx, P.startFrame);
}

function tick(now){
  _raf = 0;
  if(!_visible || !_playing || !_prep) return;
  const elapsed = now - _t0;
  _progress = Math.min(1, elapsed / totalMs(_prep));
  paint();
  if(_progress >= 1){
    stopPlay();
    setInfo((_prep.infoBase || '') + ' · 結束');
    return;
  }
  _raf = requestAnimationFrame(tick);
}

function startPlay(){
  if(_blocked || !_prep) return;
  _playing = true;
  _t0 = performance.now() - _progress * totalMs(_prep);
  const play = document.getElementById('walk-play');
  if(play) play.textContent = '⏸ 暫停';
  if(!_raf) _raf = requestAnimationFrame(tick);
}

function stopPlay(){
  _playing = false;
  if(_raf){ cancelAnimationFrame(_raf); _raf = 0; }
  const play = document.getElementById('walk-play');
  if(play && !_blocked) play.textContent = '▶ 播放';
}

function showBlockedCanvas(msg){
  const cv = document.getElementById('walk-canvas');
  if(!cv) return;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#0d1218';
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.fillStyle = '#e05a3a';
  ctx.font = '16px sans-serif';
  ctx.textAlign = 'center';
  const lines = [];
  let line = '';
  for(const ch of msg){
    line += ch;
    if(line.length >= 18){ lines.push(line); line = ''; }
  }
  if(line) lines.push(line);
  const y0 = SIZE / 2 - (lines.length - 1) * 12;
  lines.forEach((ln, i) => ctx.fillText(ln, SIZE / 2, y0 + i * 24));
}

function refresh(){
  if(!_visible) return;
  const hd = getHouseData(), house = getHouseName();
  if(!hd || !house) return;
  const d = deriveViewPair();
  if(!d){
    setLabel('');
    setInfo('目前沒有可用的配對(請先在佇列選一對)');
    setBlocked('目前沒有可用的配對');
    setBadge('');
    setPhase('');
    _prep = null; _lastKey = '';
    return;
  }
  const { pair, azI, azJ, confirmed } = d;
  const key = [house, pair.i, pair.j, azI.toFixed(1), azJ.toFixed(1), _dir,
               _regOn ? 'r1' : 'r0', _bandOn ? 'b1' : 'b0'].join('|');
  const st = pairStatus(pair);
  const iLab = _dir === 'ab' ? pair.i : pair.j;
  const jLab = _dir === 'ab' ? pair.j : pair.i;
  setLabel(
    'img <b>' + iLab + '</b> → img <b>' + jLab + '</b> · ' + st +
    (pair.prob != null ? ' ' + (pair.prob * 100).toFixed(0) + '%' : '')
  );
  if(key === _lastKey) return;

  stopPlay();
  _prep = null;
  _progress = 0;
  setInfo('載入…');
  setPhase('');
  setBlocked(null);
  setBadge('');
  const tk = ++_token;
  const fa = hd.images[pair.i], fb = hd.images[pair.j];
  const urlOf = f => '/images/' + encodeURIComponent(house) + '/' + encodeURIComponent(f);

  Promise.all([
    fetchRegions(house, fa, fb),
    getErpCached(urlOf(fa)),
    getErpCached(urlOf(fb)),
  ]).then(async ([rj, ea, eb]) => {
    if(tk !== _token || !_visible) return;
    try {
      const walkAb = _dir === 'ab';
      const gates = mapPose(rj, walkAb);
      if(!gates.ok){
        setBlocked(gates.reason);
        setInfo(gates.reason);
        setPhase(gates.reason);
        setBadge('');
        _lastKey = key;
        _prep = null;
        showBlockedCanvas(gates.reason);
        return;
      }

      let erpS, erpE, doorS, doorE, startIdx, endIdx;
      if(walkAb){
        erpS = ea; erpE = eb;
        doorS = azI; doorE = azJ;
        startIdx = pair.i; endIdx = pair.j;
      } else {
        erpS = eb; erpE = ea;
        doorS = azJ; doorE = azI;
        startIdx = pair.j; endIdx = pair.i;
      }

      // 區域一律先算(視角鎖定與遮罩開關無關：遮罩 OFF 也照樣朝主區)。
      const sides = sidePolysFrom(rj, walkAb);
      const main = mainRegionAz(sides);

      // A 端:轉到 A 側主區重心;該側無主區 → 維持通道方位。
      const azStart = main.azS != null ? normAz(main.azS) : normAz(doorS);
      const noteS = main.azS != null ? '主區重心' : '通道方位';

      // B 端:落地直接看同一塊主區;無主區 → pose R 對齊。
      let azEnter, noteE;
      if(main.azE != null){
        azEnter = main.azE;
        noteE = '主區重心';
      } else {
        const mapped = mapAzThroughR(azStart, gates.R);
        azEnter = mapped != null ? mapped : doorE + 180;
        noteE = mapped != null ? 'R對齊' : 'R失敗·門+180';
      }
      azEnter = normAz(azEnter);

      // 任務7:固定 FOV;無 approach 拉近。degenerate 跳過轉向。
      const degen = !!gates.degenerate;
      const fovUse = FOV_FIXED;
      const turnDelta = degen ? 0 : deltaAz(doorS, azStart);
      const nTurn = Math.abs(turnDelta) < TURN_MIN_DEG ? 0
        : Math.max(3, Math.min(ROT_FRAMES, Math.round(Math.abs(turnDelta) / TURN_DEG_PER_FRAME)));
      const nLb = END_WITH_LOOKBACK ? ROT_FRAMES : 0;
      const totalBake = nTurn + nLb + 3;   // doorFrame + startFrame + enterFrame + turn + lb
      const onProg = (done) => {
        if(tk !== _token) return;
        setInfo('預渲染 ' + done + '/' + totalBake + '…');
        setPhase('預渲染中 ' + Math.round(100 * done / totalBake) + '%');
      };
      setInfo('預渲染 0/' + totalBake + '…');

      const regS = (_regOn && sides.start.length)
        ? { polys:sides.start, W:sides.Ws, H:sides.Hs } : null;
      const regE = (_regOn && sides.end.length)
        ? { polys:sides.end, W:sides.We, H:sides.He } : null;
      const ovS = { reg:regS, bandAz:doorS };
      const ovE = { reg:regE, bandAz:doorE };

      // 起手幀:朝通道方位(B 相機)
      const doorFrame = bakeOne(erpS, doorS, fovUse, ovS);
      onProg(1);
      await yieldFrame();
      if(tk !== _token || !_visible) return;

      const turn = nTurn
        ? await bakeAzTimed(erpS, doorS, azStart, fovUse, ovS, onProg, 1, nTurn)
        : { frames:[], scores:[], durs:[] };
      if(tk !== _token || !_visible) return;

      const startFrame = bakeOne(erpS, azStart, fovUse, ovS);
      onProg(1 + nTurn + 1);
      await yieldFrame();
      if(tk !== _token || !_visible) return;

      const enterFrame = bakeOne(erpE, azEnter, fovUse, ovE);
      onProg(1 + nTurn + 2);
      await yieldFrame();
      if(tk !== _token || !_visible) return;

      const lb = nLb
        ? await bakeAzTimed(erpE, azEnter, doorE, fovUse, ovE, onProg, 1 + nTurn + 2, nLb)
        : { frames:[], scores:[], durs:[] };
      if(tk !== _token || !_visible) return;

      const sumMs = list => list.reduce((s, x) => s + x, 0);
      const turnMs = sumMs(turn.durs), lbMs = sumMs(lb.durs);
      const nReg = Math.max(sides.start.length, sides.end.length);
      const infoBase =
        (degen ? '近零基線·同點拍攝' : '起手朝通道→轉主區→B直跳') +
        ' · FOV' + fovUse +
        ' · 淡化' + FADE_MS + 'ms' +
        (_regOn && nReg ? ' · 遮罩' + nReg : '') +
        ' · 通道@' + doorS.toFixed(0) + '°' +
        ' · A@' + azStart.toFixed(0) + '°(' + noteS + ')' +
        ' → B@' + azEnter.toFixed(0) + '°(' + noteE + ')' +
        (main.k != null ? ' · 主區 R' + main.k : '') +
        (nTurn ? ' · 轉向' + (turnMs / 1000).toFixed(1) + 's' : '') +
        (END_WITH_LOOKBACK ? ' · 回看' + (lbMs / 1000).toFixed(1) + 's' : '') +
        ' · conf=' + (gates.conf_used != null ? gates.conf_used : '—');

      _prep = {
        turnA: turn.frames,
        turnADur: turn.durs,
        turnAScore: turn.scores,
        lookBack: lb.frames,
        lookBackDur: lb.durs,
        lookBackScore: lb.scores,
        doorFrame, startFrame, enterFrame,
        doorS: normAz(doorS),
        azStart, azEnter,
        noteS, noteE,
        mainK: main.k,
        degenerate: degen,
        conf_used: gates.conf_used,
        startName: 'img' + startIdx,
        endName: 'img' + endIdx,
        infoBase,
      };
      _lastKey = key;
      _progress = 0;
      setBlocked(null);
      const badgeBits = [];
      if(gates.conf_used === 0.3) badgeBits.push('低信心 conf=0.3');
      const nw = nadirWarning(rj);
      if(nw) badgeBits.push('⚠ ' + nw);
      setBadge(badgeBits.join(' · '));
      setInfo(infoBase + ' · 全程~' + (totalMs(_prep) / 1000).toFixed(1) + 's');
      setLabel(
        'img <b>' + startIdx + '</b> → img <b>' + endIdx + '</b> · ' + st +
        (pair.prob != null ? ' ' + (pair.prob * 100).toFixed(0) + '%' : '') +
        (confirmed ? ' · ROI校正' : '') +
        ' · 過渡 A@' + azStart.toFixed(0) + '°(' + noteS + ') → B@' + azEnter.toFixed(0) + '°(' + noteE + ')'
      );
      paint();
    } catch(e){
      if(tk === _token){
        console.error('walk bake error', e);
        setBlocked('預渲染失敗');
        setInfo('預渲染失敗: ' + (e && e.message ? e.message : e));
        setPhase('預渲染失敗');
        showBlockedCanvas('預渲染失敗');
      }
    }
  }).catch(e => {
    if(tk === _token){
      console.error('walk load error', e);
      setBlocked('載入失敗');
      setInfo('載入失敗: ' + (e && e.message ? e.message : e));
      showBlockedCanvas('載入失敗');
    }
  });
}

registerViewMode({
  id: 'walk',
  label: '走一遍',
  title: 'A 起手朝通道→轉共識主區→B 直跳面朝主區;固定FOV;釘窗+遮罩+通道帶',
  ownButton: true,
  isActive(){ return _visible; },
  enter(){
    buildDom();
    _visible = true;
    _root.style.display = 'flex';
    _lastKey = '';
    _progress = 0;
    const dirBtn = document.getElementById('walk-dir');
    if(dirBtn) dirBtn.textContent = _dir === 'ab' ? 'A→B' : 'B→A';
    const pinBtn = document.getElementById('walk-pin');
    if(pinBtn){
      pinBtn.textContent = _pinOn ? '釘窗 ON' : '釘窗 OFF';
      pinBtn.classList.toggle('on', _pinOn);
    }
    const regBtn = document.getElementById('walk-reg');
    if(regBtn){
      regBtn.textContent = _regOn ? '遮罩 ON' : '遮罩 OFF';
      regBtn.classList.toggle('on', _regOn);
    }
    const bandBtn = document.getElementById('walk-band');
    if(bandBtn){
      bandBtn.textContent = _bandOn ? '通道帶 ON' : '通道帶 OFF';
      bandBtn.classList.toggle('on', _bandOn);
    }
    refresh();
  },
  exit(){
    _visible = false;
    stopPlay();
    if(_root) _root.style.display = 'none';
    _token++;
  },
  refresh,
});

on('state-changed', () => { if(_visible){ _lastKey = ''; refresh(); } });
