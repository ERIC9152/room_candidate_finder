/* =========================================================================
   api.js：伺服器 API 封裝（server.py 端點，見 SPEC）
   低層模組：不 import 任何其它模組。URL 與 method 完全不變。
   ========================================================================= */

/**
 * GET /api/houses -> [{house,n_images,n_pairs,n_gt_connected_pairs}]
 * @returns {Promise<Array>}
 */
export function fetchHouses(){
  return fetch('/api/houses').then(r => r.json());
}

/**
 * GET /api/house/<house> -> {house,n_images,width,height,images[],pairs[],az_convention}
 * @param {string} house
 * @returns {Promise<object>}
 */
export function fetchHouseData(house){
  return fetch('/api/house/' + encodeURIComponent(house)).then(r => r.json());
}

/**
 * GET /api/state/<house> -> 已存標註（或 {}）。失敗時回傳 {}。
 * @param {string} house
 * @returns {Promise<object>}
 */
export function fetchState(house){
  return fetch('/api/state/' + encodeURIComponent(house))
    .then(r => r.json())
    .catch(() => ({}));
}

/**
 * POST /api/state/<house>（JSON body）-> 存標註。
 * @param {string} house
 * @param {object} state
 * @returns {Promise<object>}
 */
export function saveState(house, state){
  return fetch('/api/state/' + encodeURIComponent(house), {
    method:'POST',
    headers:{ 'Content-Type':'application/json' },
    body: JSON.stringify(state)
  }).then(r => r.json());
}

/**
 * POST /api/roma_match/<house>（JSON body {img_a,img_b,conf}）-> {matches,w,h,count} 或 {error}。
 * matches: [[xa,ya,xb,yb,conf], ...]（原圖像素座標，A=img_a、B=img_b）。
 * 伺服器端會重用/寫入與 run_roma_above_threshold.py 相同的 roma_match/cache/*.npz，
 * 所以同一對圖重複點擊會直接命中快取、秒回。
 * @param {string} house
 * @param {string} imgA 檔名（不含路徑）
 * @param {string} imgB 檔名（不含路徑）
 * @param {number} [conf=0.5] RoMa 信心門檻
 * @returns {Promise<object>}
 */
/**
 * GET /api/regions/<house>?a=&b= -> 共視區域 JSON（wp1-step3-v1；不含 inliers）＋ swap 旗標。
 * swap=true 表示檔案是 B_A 方向，regions 的 a/b 兩側要對調使用。404 = 此對沒跑過離線批次。
 * @param {string} house
 * @param {string} imgA 檔名（不含路徑）
 * @param {string} imgB 檔名（不含路徑）
 * @returns {Promise<object>}
 */
export function fetchRegions(house, imgA, imgB){
  return fetch('/api/regions/' + encodeURIComponent(house) +
               '?a=' + encodeURIComponent(imgA) + '&b=' + encodeURIComponent(imgB))
    .then(r => r.json());
}

/**
 * 天底帶警示:主區(regions[0],批次端按內點數降冪排序)任一側 polygon 重心落在
 * lat < −60° 的腳架/地板帶時,回傳警示字串(掛警示,不跳過該對)。該帶在各張全景圖上
 * 外觀相同,是系統性假匹配熱點。批次端另有 --lat_filter 在 RANSAC 前先濾一次。
 * @param {object} rj /api/regions 回傳 JSON
 * @returns {string|null} 警示文字;主區不在天底帶回 null
 */
export function nadirWarning(rj){
  const reg = rj && rj.regions && rj.regions[0];
  if(!reg) return null;
  const latOf = side => {
    const poly = reg[side] && reg[side].polygon;
    const size = side === 'a' ? rj.img_size_a : rj.img_size_b;
    const H = size && size[1];
    if(!poly || poly.length < 3 || !H) return null;
    const cy = poly.reduce((s, p) => s + p[1], 0) / poly.length;
    return 90 - (cy / H) * 180;
  };
  const la = latOf('a'), lb = latOf('b');
  if((la != null && la < -60) || (lb != null && lb < -60)){
    return '主區在天底帶(lat<−60°)·疑似腳架假匹配';
  }
  return null;
}

/**
 * GET /api/semantic_doors/<house> -> {house, doors:{<image檔名不含副檔名>:[{az,az0,az1,width_deg,px}]}}。
 * 404（該房沒有語意圖，見 prepare_semantic_doors.py）或任何請求失敗一律回傳 null。呼叫端視為
 * 「此房無語意圖」，不拋例外（語意輔助勾選對沒有語意圖的房子必須完全不炸）。
 * @param {string} house
 * @returns {Promise<object|null>}
 */
export function fetchSemanticDoors(house){
  return fetch('/api/semantic_doors/' + encodeURIComponent(house))
    .then(r => r.ok ? r.json() : null)
    .catch(() => null);
}

export function fetchRomaMatch(house, imgA, imgB, conf){
  return fetch('/api/roma_match/' + encodeURIComponent(house), {
    method:'POST',
    headers:{ 'Content-Type':'application/json' },
    body: JSON.stringify({ img_a:imgA, img_b:imgB, conf:(conf != null ? conf : 0.5) })
  }).then(r => r.json());
}
