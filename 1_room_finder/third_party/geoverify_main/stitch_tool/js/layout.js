/* =========================================================================
   layout.js：路線圖力導向佈局（純函式，可 node 測試）
   低層模組：不 import 任何其它模組，不觸碰 DOM。
   把原本的環狀佈局換成力導向。
     - 已確認邊 = 彈簧（理想邊長 restLen），拉近相連節點；
     - 所有節點兩兩之間 = 庫倫斥力，避免重疊；
     - 孤立節點（沒有任何 confirmed 邊者）= 額外往外圈推力，聚到邊緣，
       方便「處理孤立照片」這個核心工作流一眼看出誰還沒連。
   確定性：初始位置用圓形排列（角度由 index 決定，非隨機），固定迭代步數，
     無 Math.random，故同一組 (nodes,edges,W,H,iter) 一定得到同一份 positions。
   ========================================================================= */

/**
 * 判斷某節點是否為孤立（不在任何 confirmed 邊上）。
 * @param {number} idx 節點 index
 * @param {Array<[number,number]>} edges 邊清單（[i,j] 陣列）
 * @returns {boolean}
 */
export function isIsolated(idx, edges){
  for(let k = 0; k < edges.length; k++){
    if(edges[k][0] === idx || edges[k][1] === idx) return false;
  }
  return true;
}

/**
 * 力導向佈局（確定性）。回傳每個節點的 {x,y}（座落在 [0,W]x[0,H] 內，留邊界 pad）。
 *
 * 演算法：
 *   1) 初始位置：圓形排列（角度 = i/N * 2π，半徑 = 初始環半徑），孤立節點放外圈、相連節點放內圈。
 *   2) 迭代 iter 步：每步累加
 *        - 斥力：所有節點對，庫倫式 k_rep / dist^2，沿連線方向互推（近距離加下限避免爆量）。
 *        - 彈簧：每條 confirmed 邊，(dist - restLen) * k_spring 沿連線互拉。
 *        - 向心/外推：相連節點受微弱向心力（收往中心）、孤立節點受微弱外推力（推離中心），
 *          使兩群分層（相連者成圖、孤立者散在外圈）。
 *      位移每步乘冷卻係數（cooling）遞減，收斂穩定。
 *   3) 夾到邊界內（保留 pad）。
 *
 * @param {number} N 節點數
 * @param {Array<[number,number]>} edges confirmed 邊（[i,j]，i/j 為節點 index，可含越界者，內部略過）
 * @param {number} W 畫布寬
 * @param {number} H 畫布高
 * @param {number} [iter=160] 迭代步數（越多越穩定；增量更新時可給少量步數）
 * @param {Array<{x:number,y:number}>|null} [init=null] 初始位置（增量更新沿用上次結果避免跳動）；null 則圓形排列
 * @returns {Array<{x:number,y:number}>} 每個節點的座標
 */
export function forceLayout(N, edges, W, H, iter, init, opts){
  iter = (typeof iter === 'number' && iter > 0) ? iter : 160;
  edges = edges || [];
  if(N <= 0) return [];
  const cx = W / 2, cy = H / 2;
  const pad = Math.max(12, Math.min(W, H) * 0.08);
  const R = Math.min(W, H) / 2 - pad;         // 可用半徑
  const innerR = R * 0.55;                     // 相連節點初始環
  const outerR = R * 0.95;                      // 孤立節點初始環

  const nodeR = (opts && typeof opts.nodeR === 'number') ? opts.nodeR : Math.max(4, Math.min(W, H) * 0.028);

  // 標記孤立節點
  const iso = new Array(N);
  for(let i = 0; i < N; i++) iso[i] = isIsolated(i, edges);

  // 1) 初始位置（確定性）：優先沿用 init；否則圓形排列
  const pos = new Array(N);
  for(let i = 0; i < N; i++){
    if(init && init[i] && typeof init[i].x === 'number' && typeof init[i].y === 'number'){
      pos[i] = { x:init[i].x, y:init[i].y };
    } else {
      const a = (i / Math.max(1, N)) * 2 * Math.PI - Math.PI / 2;
      const rr = iso[i] ? outerR : innerR;
      // 加一個依 index 決定的微小相位擾動（非隨機），避免完全共線節點斥力方向退化。
      const jitter = ((i * 2654435761) % 1000) / 1000 - 0.5;    // 確定性偽擾動 [-0.5,0.5)
      pos[i] = { x:cx + rr * Math.cos(a) + jitter, y:cy + rr * Math.sin(a) - jitter };
    }
  }
  if(N === 1){ pos[0] = { x:cx, y:cy }; return pos; }

  // --- 連通分量分中心 (解 R2) ---
  const parent = new Array(N);
  for(let i = 0; i < N; i++) parent[i] = i;

  function find(i){
    let root = i;
    while(parent[root] !== root){
      root = parent[root];
    }
    let curr = i;
    while(curr !== root){
      let nxt = parent[curr];
      parent[curr] = root;
      curr = nxt;
    }
    return root;
  }

  function union(i, j){
    const rootI = find(i);
    const rootJ = find(j);
    if(rootI !== rootJ){
      parent[rootI] = rootJ;
    }
  }

  for(let k = 0; k < edges.length; k++){
    const u = edges[k][0];
    const v = edges[k][1];
    if(u >= 0 && u < N && v >= 0 && v < N && u !== v){
      union(u, v);
    }
  }

  const compMap = new Map();
  for(let i = 0; i < N; i++){
    if(!iso[i]){
      const r = find(i);
      if(!compMap.has(r)){
        compMap.set(r, []);
      }
      compMap.get(r).push(i);
    }
  }

  const components = Array.from(compMap.values());
  components.sort((a, b) => {
    if(b.length !== a.length){
      return b.length - a.length;
    }
    const minA = Math.min(...a);
    const minB = Math.min(...b);
    return minA - minB;
  });

  const C = components.length;
  const nodeCenter = new Array(N);
  if(C >= 2){
    for(let k = 0; k < C; k++){
      const angle = -Math.PI / 2 + (k / C) * 2 * Math.PI;
      const tCx = cx + R * 0.35 * Math.cos(angle);
      const tCy = cy + R * 0.35 * Math.sin(angle);
      for(const idx of components[k]){
        nodeCenter[idx] = { x:tCx, y:tCy };
      }
    }
  } else {
    for(let i = 0; i < N; i++){
      nodeCenter[i] = { x:cx, y:cy };
    }
  }

  // 力參數（相對畫布尺寸縮放，讓大小 canvas 表現一致）
  const scale = Math.min(W, H);
  const restLen = scale * 0.22;                 // 彈簧理想邊長
  const kSpring = 0.02;                          // 彈簧強度
  const kRep = scale * scale * 0.9;             // 斥力常數
  const kCenter = 0.25;                          // 相連節點向心 (調大以防飛到邊界)
  const minDist = scale * 0.02;                  // 斥力距離下限
  const maxStep = scale * 0.10;                  // 單步位移上限

  const fx = new Array(N), fy = new Array(N);
  for(let step = 0; step < iter; step++){
    for(let i = 0; i < N; i++){ fx[i] = 0; fy[i] = 0; }

    // 斥力（所有非孤立節點對）
    for(let i = 0; i < N; i++){
      if(iso[i]) continue;
      for(let j = i + 1; j < N; j++){
        if(iso[j]) continue;
        let dx = pos[i].x - pos[j].x, dy = pos[i].y - pos[j].y;
        let d2 = dx * dx + dy * dy;
        let d = Math.sqrt(d2) || 0.0001;
        if(d < minDist){ d = minDist; d2 = minDist * minDist; }
        const f = kRep / d2;
        const ux = dx / d, uy = dy / d;
        fx[i] += ux * f; fy[i] += uy * f;
        fx[j] -= ux * f; fy[j] -= uy * f;
      }
    }

    // 彈簧（confirmed 邊）
    for(let k = 0; k < edges.length; k++){
      const a = edges[k][0], b = edges[k][1];
      if(a < 0 || b < 0 || a >= N || b >= N || a === b) continue;
      if(iso[a] || iso[b]) continue;
      let dx = pos[b].x - pos[a].x, dy = pos[b].y - pos[a].y;
      const d = Math.sqrt(dx * dx + dy * dy) || 0.0001;
      const f = (d - restLen) * kSpring;
      const ux = dx / d, uy = dy / d;
      fx[a] += ux * f; fy[a] += uy * f;
      fx[b] -= ux * f; fy[b] -= uy * f;
    }

    // 向心（相連）
    for(let i = 0; i < N; i++){
      if(iso[i]) continue;
      const target = nodeCenter[i];
      const dx = pos[i].x - target.x, dy = pos[i].y - target.y;
      fx[i] -= dx * kCenter;
      fy[i] -= dy * kCenter;
    }

    // 套用位移（冷卻 + 單步上限）
    const cooling = 1 - step / (iter * 1.2);
    for(let i = 0; i < N; i++){
      if(iso[i]) continue;
      let mx = fx[i] * cooling, my = fy[i] * cooling;
      const m = Math.sqrt(mx * mx + my * my);
      if(m > maxStep){ mx = mx / m * maxStep; my = my / m * maxStep; }
      pos[i].x += mx; pos[i].y += my;
    }
  }

  // --- 2) 孤立節點改為後處理環佈局 ---
  const isolatedIndices = [];
  for(let i = 0; i < N; i++){
    if(iso[i]){
      isolatedIndices.push(i);
    }
  }
  let remCount = isolatedIndices.length;
  let currentRing = 0;
  let nextIndexToPlace = 0;
  const r0 = Math.min(W, H) / 2 - pad - nodeR;
  while(remCount > 0){
    const r = r0 - currentRing * 2.8 * nodeR;
    let cap = 1;
    if(r > 0){
      cap = Math.floor((2 * Math.PI * r) / (2.4 * nodeR));
      if(cap < 1) cap = 1;
    }
    const m = Math.min(remCount, cap);
    for(let p = 0; p < m; p++){
      const nodeIdx = isolatedIndices[nextIndexToPlace + p];
      const angle = -Math.PI / 2 + (p / m) * 2 * Math.PI;
      pos[nodeIdx].x = cx + r * Math.cos(angle);
      pos[nodeIdx].y = cy + r * Math.sin(angle);
    }
    nextIndexToPlace += m;
    remCount -= m;
    currentRing++;
  }

  // --- 3) 碰撞消解後處理 ---
  const minSep = 2 * nodeR + 2;
  for(let round = 0; round < 40; round++){
    let adjusted = false;
    for(let i = 0; i < N; i++){
      for(let j = i + 1; j < N; j++){
        if(iso[i] && iso[j]) continue;
        let dx = pos[i].x - pos[j].x;
        let dy = pos[i].y - pos[j].y;
        let dist = Math.sqrt(dx * dx + dy * dy);
        if(dist < minSep){
          adjusted = true;
          let ux, uy;
          if(dist < 1e-4){
            const angle = ((i * 17 + j * 31) % 360) * Math.PI / 180;
            ux = Math.cos(angle);
            uy = Math.sin(angle);
            dist = 0;
          } else {
            ux = dx / dist;
            uy = dy / dist;
          }
          const pushAmount = (minSep - dist) / 2;
          pos[i].x += ux * pushAmount;
          pos[i].y += uy * pushAmount;
          pos[j].x -= ux * pushAmount;
          pos[j].y -= uy * pushAmount;
        }
      }
    }
    // 每次消解完都進行 boundary clamp，容許節點沿邊界滑動以消解重疊
    for(let k = 0; k < N; k++){
      pos[k].x = Math.max(pad, Math.min(W - pad, pos[k].x));
      pos[k].y = Math.max(pad, Math.min(H - pad, pos[k].y));
    }
    if(!adjusted) break;
  }

  // 3) 夾到邊界內
  for(let i = 0; i < N; i++){
    pos[i].x = Math.max(pad, Math.min(W - pad, pos[i].x));
    pos[i].y = Math.max(pad, Math.min(H - pad, pos[i].y));
  }
  return pos;
}
