/* =========================================================================
   viewmodes/mode-morph.js：「動態 morph」檢視模式插件(已停用,見 viewmodes/index.js)
   用 RoMa cache 的稀疏 matches(/api/roma_match,同離線 cache,命中即秒回)建
   三角網格 warp,把「A 的裁切視角」向「B 的對應視角」漸變(來回播放)。
   真連通看起來像「走過去」;假連通(外觀相似誤匹配)會扭曲穿幫。

   流程(每對一次準備,之後 rAF 動畫):
     1. deriveViewPair() 取得配對與 azI(A 看向 B 的方位)。
     2. fetchRomaMatch(conf=0.3) 拿 matches(原圖像素)。
     3. A 裁切相機 = azI;B 裁切相機 = 「A 側落在 FOV 內的那些 match」B 側方位的圓形
        平均(保證匹配內容出現在 B 裁切圖裡;<8 點退回 azJ+180=行進方向)。
     4. match 點投到兩張裁切圖 → 對應點對;IDW 算 11×11 網格頂點位移場(雙向)。
     5. 動畫:t 來回(smoothstep),warped-A(α=1−t)+ warped-B(α=t)交叉淡化,
        每格兩個三角形 drawImage 仿射貼圖;顯示即時 FPS(驗收 ≥24fps)。
   ========================================================================= */

import { registerViewMode } from './registry.js';
import { getErpCached, renderGnomonic, erpPxToCrop, circMeanDeg } from './gnomonic.js';
import { deriveViewPair } from './pairsel.js';
import { getHouseData, getHouseName, pairStatus } from '../state.js';
import { fetchRomaMatch } from '../api.js';
import { toast } from '../util.js';

const FOV = 90, SIZE = 512;
const GRID = 10;              // 網格格數(頂點 11×11;三角形 200/層)
const CONF = 0.3;             // match 取樣門檻(窄門洞 pair 在 0.65 常湊不出足夠點數)
const MAX_PTS = 250;          // IDW 用的對應點上限(conf 降冪)
const CYCLE_MS = 1600;        // 一個來回週期

let _root = null, _visible = false, _playing = true;
let _token = 0, _lastKey = '';
let _prep = null;             // { cropA, cropB, meshA, meshB, n } | null
let _raf = 0, _frames = 0, _fpsT = 0;

// ---- DOM ----
function buildDom(){
  if(_root) return;
  _root = document.createElement('div');
  _root.id = 'morph-mode';
  _root.className = 'viewmode-overlay';
  _root.innerHTML =
    '<div class="vm-head" id="morph-head">動態 morph</div>' +
    '<div class="vm-body">' +
      '<div class="crop-pane"><div class="crop-label" id="morph-label"></div>' +
        '<canvas id="morph-canvas" width="' + SIZE + '" height="' + SIZE + '"></canvas></div>' +
    '</div>' +
    '<div class="vm-hint">' +
      '<button id="morph-play">⏸ 暫停</button>' +
      '<span id="morph-fps" class="morph-badge"></span>' +
      '<span id="morph-info"></span>' +
    '</div>';
  const viewer = document.getElementById('viewer');
  if(viewer) viewer.appendChild(_root);
  const pb = document.getElementById('morph-play');
  if(pb) pb.onclick = () => {
    _playing = !_playing;
    pb.textContent = _playing ? '⏸ 暫停' : '▶ 播放';
    if(_playing && _visible && !_raf) _raf = requestAnimationFrame(tick);
  };
}

function setInfo(t){ const el = document.getElementById('morph-info'); if(el) el.textContent = t || ''; }

// ---- 網格位移場(IDW) ----
/** 對 (GRID+1)² 個頂點,用 1/(d²+soft) 反距離加權出位移向量場。*/
function buildMesh(src, dst){
  const n = GRID + 1, step = SIZE / GRID, soft = 900;   // soft≈30px²:近點不爆量
  const disp = new Float32Array(n * n * 2);
  for(let gy = 0; gy < n; gy++){
    for(let gx = 0; gx < n; gx++){
      const vx = gx * step, vy = gy * step;
      let sw = 0, sx = 0, sy = 0;
      for(let k = 0; k < src.length; k++){
        const dx = vx - src[k][0], dy = vy - src[k][1];
        const w = 1 / (dx * dx + dy * dy + soft);
        sw += w;
        sx += w * (dst[k][0] - src[k][0]);
        sy += w * (dst[k][1] - src[k][1]);
      }
      const o = (gy * n + gx) * 2;
      disp[o] = sx / sw; disp[o + 1] = sy / sw;
    }
  }
  return disp;
}

// ---- 三角形仿射貼圖 ----
function drawTri(ctx, img, s0, s1, s2, d0, d1, d2){
  const e1x = s1[0] - s0[0], e1y = s1[1] - s0[1];
  const e2x = s2[0] - s0[0], e2y = s2[1] - s0[1];
  const det = e1x * e2y - e2x * e1y;
  if(Math.abs(det) < 1e-6) return;
  const f1x = d1[0] - d0[0], f1y = d1[1] - d0[1];
  const f2x = d2[0] - d0[0], f2y = d2[1] - d0[1];
  const a = (f1x * e2y - f2x * e1y) / det;
  const b = (f1y * e2y - f2y * e1y) / det;
  const c = (f2x * e1x - f1x * e2x) / det;
  const d = (f2y * e1x - f1y * e2x) / det;
  const e = d0[0] - a * s0[0] - c * s0[1];
  const f = d0[1] - b * s0[0] - d * s0[1];
  // dest 三角形往形心外擴 0.6px 蓋縫
  const cx = (d0[0] + d1[0] + d2[0]) / 3, cy = (d0[1] + d1[1] + d2[1]) / 3;
  const ex = p => [p[0] + Math.sign(p[0] - cx) * 0.6, p[1] + Math.sign(p[1] - cy) * 0.6];
  const g0 = ex(d0), g1 = ex(d1), g2 = ex(d2);
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(g0[0], g0[1]); ctx.lineTo(g1[0], g1[1]); ctx.lineTo(g2[0], g2[1]);
  ctx.closePath(); ctx.clip();
  ctx.transform(a, b, c, d, e, f);
  ctx.drawImage(img, 0, 0);
  ctx.restore();
}

/** 畫一層 warp:頂點 = 規則網格 + t×位移場。*/
function drawWarpLayer(ctx, img, disp, t, alpha){
  const n = GRID + 1, step = SIZE / GRID;
  ctx.globalAlpha = alpha;
  const P = (gx, gy) => {
    const o = (gy * n + gx) * 2;
    return [gx * step + t * disp[o], gy * step + t * disp[o + 1]];
  };
  for(let gy = 0; gy < GRID; gy++){
    for(let gx = 0; gx < GRID; gx++){
      const s00 = [gx * step, gy * step], s10 = [(gx + 1) * step, gy * step];
      const s01 = [gx * step, (gy + 1) * step], s11 = [(gx + 1) * step, (gy + 1) * step];
      drawTri(ctx, img, s00, s10, s11, P(gx, gy), P(gx + 1, gy), P(gx + 1, gy + 1));
      drawTri(ctx, img, s00, s11, s01, P(gx, gy), P(gx + 1, gy + 1), P(gx, gy + 1));
    }
  }
  ctx.globalAlpha = 1;
}

// ---- 動畫 ----
function tick(now){
  _raf = 0;
  if(!_visible || !_playing) return;
  const cv = document.getElementById('morph-canvas');
  if(cv && _prep){
    const ph = (now % CYCLE_MS) / CYCLE_MS;
    const tt = ph < 0.5 ? ph * 2 : 2 - ph * 2;          // 來回
    const t = tt * tt * (3 - 2 * tt);                    // smoothstep
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, SIZE, SIZE);
    drawWarpLayer(ctx, _prep.cropA, _prep.meshA, t, 1 - t);
    drawWarpLayer(ctx, _prep.cropB, _prep.meshB, 1 - t, t);
    _frames++;
    if(now - _fpsT >= 500){
      const fps = _frames * 1000 / (now - _fpsT);
      _frames = 0; _fpsT = now;
      const el = document.getElementById('morph-fps');
      if(el){ el.textContent = fps.toFixed(0) + ' fps'; el.classList.toggle('bad', fps < 24); }
    }
  }
  _raf = requestAnimationFrame(tick);
}

// ---- 資料準備 ----
function refresh(){
  if(!_visible) return;
  const hd = getHouseData(), house = getHouseName();
  if(!hd || !house) return;
  const d = deriveViewPair();
  const label = document.getElementById('morph-label');
  if(!d){
    setInfo('目前沒有可用的配對(請先在佇列選一對)');
    if(label) label.textContent = '';
    _prep = null; _lastKey = '';
    return;
  }
  const { pair, azI, azJ } = d;
  const key = [house, pair.i, pair.j, azI.toFixed(1)].join('|');
  const st = pairStatus(pair);
  const gt = (pair.gt_connected != null) ? (pair.gt_connected ? ' · GT連通' : ' · GT不連通') : '';
  if(label) label.innerHTML = 'img <b>' + pair.i + '</b> → img <b>' + pair.j + '</b>' +
    ' <span class="crop-az">@ ' + azI.toFixed(1) + '°</span> · ' + st +
    (pair.prob != null ? ' ' + (pair.prob * 100).toFixed(0) + '%' : '') + gt;
  if(key === _lastKey) return;
  _prep = null;
  setInfo('準備 morph(RoMa matches + 投影)…');
  const tk = ++_token;
  const imgA = hd.images[pair.i], imgB = hd.images[pair.j];
  const urlOf = f => '/images/' + encodeURIComponent(house) + '/' + encodeURIComponent(f);
  Promise.all([
    fetchRomaMatch(house, imgA, imgB, CONF),
    getErpCached(urlOf(imgA)), getErpCached(urlOf(imgB)),
  ]).then(([res, ea, eb]) => {
    if(tk !== _token || !_visible) return;
    if(!res || res.error || !Array.isArray(res.matches)){
      setInfo('RoMa 匹配取得失敗:' + (res && res.error ? res.error : '未知'));
      return;
    }
    const W = res.w || 8192, H = res.h || 4096;
    // B 相機方位 = A 側在 FOV 內那些 match 的 B 側方位圓平均(<8 點退回 azJ+180)
    const azBs = [];
    for(const m of res.matches){
      if(erpPxToCrop(m[0], m[1], W, H, azI, FOV, SIZE)) azBs.push(m[2] / W * 360 - 180);
    }
    const azB = (azBs.length >= 8) ? circMeanDeg(azBs) : (azJ + 180) % 360;
    // 對應點對(兩側都要落在裁切圖內),conf 降冪取前 MAX_PTS
    const ms = res.matches.slice().sort((x, y) => y[4] - x[4]);
    const src = [], dst = [];
    for(const m of ms){
      const pa = erpPxToCrop(m[0], m[1], W, H, azI, FOV, SIZE);
      if(!pa) continue;
      const pb = erpPxToCrop(m[2], m[3], W, H, azB, FOV, SIZE);
      if(!pb) continue;
      src.push(pa); dst.push(pb);
      if(src.length >= MAX_PTS) break;
    }
    if(src.length < 8){
      setInfo('可用匹配不足(' + src.length + ' 點)。這一對缺乏共視證據,無法 morph');
      _lastKey = key;                      // 記住結果,避免重試迴圈
      return;
    }
    // 裁切圖(離屏)
    const mk = () => { const c = document.createElement('canvas'); c.width = SIZE; c.height = SIZE; return c; };
    const cropA = mk(), cropB = mk();
    renderGnomonic(ea, azI, FOV, SIZE, cropA.getContext('2d'));
    renderGnomonic(eb, azB, FOV, SIZE, cropB.getContext('2d'));
    _prep = { cropA, cropB, meshA: buildMesh(src, dst), meshB: buildMesh(dst, src), n: src.length };
    _lastKey = key;
    setInfo(src.length + ' 個對應點 · conf≥' + CONF + ' · A@' + azI.toFixed(0) + '° ⇄ B@' + azB.toFixed(0) + '°' +
            ' · 真連通應像「走過去」,假連通會扭曲穿幫');
    _frames = 0; _fpsT = performance.now();
    if(_playing && !_raf) _raf = requestAnimationFrame(tick);
  }).catch(e => { if(tk === _token) setInfo('morph 準備失敗:' + e); });
}

registerViewMode({
  id: 'morph',
  label: '動態morph',
  title: '用 RoMa 匹配把 A 的視角向 B 漸變:真連通像走過去,假連通會扭曲穿幫',
  ownButton: true,
  isActive(){ return _visible; },
  enter(){
    buildDom(); _visible = true; _root.style.display = 'flex';
    _lastKey = ''; _frames = 0; _fpsT = performance.now();
    if(getHouseData() && !nav_hasPair()) toast('提示:先在左側佇列選一對再進 morph 效果最好');
    refresh();
    if(_playing && !_raf) _raf = requestAnimationFrame(tick);
  },
  exit(){
    _visible = false;
    if(_root) _root.style.display = 'none';
    _token++;
    if(_raf){ cancelAnimationFrame(_raf); _raf = 0; }
  },
  refresh,
});

/** 有沒有可推導的配對(僅供進入提示,不影響流程)。*/
function nav_hasPair(){ return !!deriveViewPair(); }
