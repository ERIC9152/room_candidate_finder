/* =========================================================================
   semantic.js：語意輔助：用子計畫一的門語意圖校正通道方位
   低層模組：只 import api.js（不 import state/erp，維持既有分層。低層不依賴高層）。

   演算法與門檻的驗證依據見 semantic_doors/semantic_door_eval.md，此處逐字沿用：
     對某一側（i 或 j），取該圖的門清單與該對的共視區域（/api/regions）多邊形，算「每個
     region 落在某道門方位區間內的比例」，取所有 region×門組合中的最大值；≥0.8 才把該側
     通道方位改成那道門的中心方位 az，否則維持原值不動。
   region 的方位區間與 az_overlap 計算方式逐字對應 semantic_door_eval.py 的
   region_az_span()/az_overlap()（同一套 px_to_az 慣例：0=圖中央、((x/W)*360-180) mod 360），
   刻意不用 regions JSON 已存的 az_range 欄位（那是未 mod 360 的簽名版本，且是否所有 schema
   版本都有該欄位未知）。改為和驗證腳本一樣直接從 polygon 現算，確保數值與驗證結果逐一對應。
   ========================================================================= */

import { fetchSemanticDoors, fetchRegions } from './api.js';

/** 共視區落在門內的比例 ≥ 此值才採用門中心方位。
 *  此值由門先驗差分測試選定（見 semantic_doors/semantic_door_eval.md）；
 *  更動須重跑該評估，否則校正結果與驗證結論不再對應。*/
export const CONTAIN_THRESHOLD = 0.8;

/** pipeline 方位角約定：0=圖中央，((x/W)*360-180) mod 360，與 az_ij/az_ji 同一套。*/
function pxToAz(x, W){ return (((x / W) * 360 - 180) % 360 + 360) % 360; }

/** 由多邊形（原圖像素座標）算方位區間 [r0,r1] 與寬度（度）。逐字對應 region_az_span()。*/
function regionAzSpan(poly, W){
  let x0 = Infinity, x1 = -Infinity;
  for(const p of poly){ if(p[0] < x0) x0 = p[0]; if(p[0] > x1) x1 = p[0]; }
  return { r0: pxToAz(x0, W), r1: pxToAz(x1, W), rw: (x1 - x0) / W * 360 };
}

/** 兩個（可能過縫）角區間的重疊角度。與交付路徑 roma_geo.semantic.az_overlap 相同。*/
function azOverlap(a0, a1, b0, b1){
  const segs = (x0, x1) => {
    const raw = x1 - x0;
    const start = ((x0 % 360) + 360) % 360;
    const width = ((raw % 360) + 360) % 360;
    if(width === 0){
      if(Math.abs(raw) < 1e-12) return [];
      return [[0, 360]];
    }
    const end = start + width;
    if(end <= 360 + 1e-12) return [[start, Math.min(end, 360)]];
    return [[start, 360], [0, end - 360]];
  };
  let total = 0;
  for(const [s1, e1] of segs(a0, a1)){
    for(const [s2, e2] of segs(b0, b1)){
      const lo = Math.max(s1, s2), hi = Math.min(e1, e2);
      if(hi > lo) total += hi - lo;
    }
  }
  return total;
}

function baseName(fname){
  const i = fname.lastIndexOf('.');
  return i > 0 ? fname.slice(0, i) : fname;
}

// ---------- 快取（house 門資料 / pair 共視區域）----------
const _doorsCache = new Map();      // house -> doorsMap|null（null=該房無語意圖）
const _doorsInflight = new Map();
const _regionsCache = new Map();    // "house|a|b" -> regionsJson|null（null=此對無區域資料）
const _regionsInflight = new Map();

/** @returns {object|null|undefined} undefined=尚未抓過；null=該房無語意圖；否則 doors map。*/
export function getCachedHouseDoors(house){
  return _doorsCache.get(house);
}

/**
 * 確保 house 的語意門資料已抓取（有快取直接回傳；抓取中則搭上同一個 in-flight promise）。
 * onReady 保證在資料就緒時被呼叫恰一次，即使多個呼叫端（ui.js 側欄提示 / erp.js 覆寫套用）
 * 同時搭上同一個進行中的請求，也都各自拿得到通知。不能只讓「第一個發起請求的人」拿到。
 */
export function ensureHouseDoors(house, onReady){
  if(_doorsCache.has(house)){
    const v = _doorsCache.get(house);
    if(onReady) onReady(v);
    return Promise.resolve(v);
  }
  let p = _doorsInflight.get(house);
  if(!p){
    p = fetchSemanticDoors(house).then(d => {
      const map = (d && d.doors && typeof d.doors === 'object') ? d.doors : null;
      _doorsCache.set(house, map);
      _doorsInflight.delete(house);
      return map;
    });
    _doorsInflight.set(house, p);
  }
  if(onReady) p.then(onReady);
  return p;
}

function regionsKey(house, a, b){ return house + '|' + a + '|' + b; }

/** @returns {object|null|undefined} undefined=尚未抓過；null=此對無區域資料；否則原始 regions JSON。*/
function getCachedRegions(house, imgA, imgB){
  return _regionsCache.get(regionsKey(house, imgA, imgB));
}

/** 同 ensureHouseDoors：onReady 保證對每個呼叫端各觸發一次，即使搭上同一個 in-flight 請求。*/
function ensureRegions(house, imgA, imgB, onReady){
  const key = regionsKey(house, imgA, imgB);
  if(_regionsCache.has(key)){
    const v = _regionsCache.get(key);
    if(onReady) onReady(v);
    return Promise.resolve(v);
  }
  let p = _regionsInflight.get(key);
  if(!p){
    p = fetchRegions(house, imgA, imgB).then(rj => {
      const val = (rj && !rj.error) ? rj : null;
      _regionsCache.set(key, val);
      _regionsInflight.delete(key);
      return val;
    }).catch(() => {
      _regionsCache.set(key, null);
      _regionsInflight.delete(key);
      return null;
    });
    _regionsInflight.set(key, p);
  }
  if(onReady) p.then(onReady);
  return p;
}

/**
 * 單側校正計算（純函式，資料皆已備妥時同步回傳）。取所有 region×門組合中包住比例最大者；
 * ≥CONTAIN_THRESHOLD 才觸發（改用該門中心 az），否則回傳原值、triggered:false。
 * @param {Array} doorsForImg 該圖門清單 [{az,az0,az1,width_deg,px}]（無門圖傳 []）
 * @param {Array<Array>} regionPolys 該側所有 region 的 polygon（原圖像素座標）
 * @param {number} W 該圖寬度（像素）
 * @param {number} azOriginal 原本的通道方位（az_ij 或 az_ji）
 * @returns {{triggered:boolean, az:number, doorAz:number|null, doorSpan:[number,number]|null, coverFrac:number}}
 */
export function computeSideOverride(doorsForImg, regionPolys, W, azOriginal){
  if(!doorsForImg || !doorsForImg.length || !regionPolys || !regionPolys.length || !W){
    return { triggered:false, az:azOriginal, doorAz:null, doorSpan:null, coverFrac:0 };
  }
  let best = null;   // {frac, door}
  regionPolys.forEach(poly => {
    if(!poly || poly.length < 3) return;
    const { r0, r1, rw } = regionAzSpan(poly, W);
    if(rw <= 0) return;
    doorsForImg.forEach(d => {
      const frac = azOverlap(r0, r1, d.az0, d.az1) / rw;
      if(!best || frac > best.frac) best = { frac, door:d };
    });
  });
  if(!best) return { triggered:false, az:azOriginal, doorAz:null, doorSpan:null, coverFrac:0 };
  if(best.frac >= CONTAIN_THRESHOLD){
    return { triggered:true, az:best.door.az, doorAz:best.door.az,
             doorSpan:[best.door.az0, best.door.az1], coverFrac:best.frac };
  }
  return { triggered:false, az:azOriginal, doorAz:best.door.az, doorSpan:[best.door.az0, best.door.az1],
           coverFrac:best.frac };
}

/**
 * 對目前 pair 的兩側獨立計算語意校正。資料（house 門清單 / 該對區域）尚未就緒時回傳 null，
 * 並在抓取完成後呼叫 onReady（呼叫端應在 onReady 觸發時、確認仍是同一 pair 後重新呼叫本函式）。
 * @param {string} house
 * @param {object} hd house data（需要 .images）
 * @param {{i:number,j:number,az_ij:number,az_ji:number}} pair
 * @param {function} [onReady] 非同步資料就緒時的回呼（不保證仍是「目前」pair，呼叫端要自行判斷）
 * @returns {null|{noSemanticForHouse:boolean, i:object|null, j:object|null}}
 */
export function getPairOverride(house, hd, pair, onReady){
  const doors = getCachedHouseDoors(house);
  if(doors === undefined){
    ensureHouseDoors(house, () => onReady && onReady());
    return null;
  }
  if(doors === null){
    return { noSemanticForHouse:true, i:null, j:null };
  }
  const imgI = hd.images[pair.i], imgJ = hd.images[pair.j];
  const regionsRaw = getCachedRegions(house, imgI, imgJ);
  if(regionsRaw === undefined){
    ensureRegions(house, imgI, imgJ, () => onReady && onReady());
    return null;
  }
  const doorsI = doors[baseName(imgI)] || [];
  const doorsJ = doors[baseName(imgJ)] || [];
  if(!regionsRaw){
    // 此對沒有共視區域資料（未跑過幾何驗證批次）：兩側都無法判斷，維持原值。
    return {
      noSemanticForHouse:false,
      i: { triggered:false, az:pair.az_ij, doorAz:null, doorSpan:null, coverFrac:0 },
      j: { triggered:false, az:pair.az_ji, doorAz:null, doorSpan:null, coverFrac:0 },
    };
  }
  // swap:區域檔只存 A_B 一個方向，命中反序檔時 rj.swap=true，i 側要用 json 的 'b'（與
  // erp.js updateErpRegionOverlay / mode-regions.js 的 sideTop/sideBot 同一套規則）。
  const swap = !!regionsRaw.swap;
  const sideI = swap ? 'b' : 'a', sideJ = swap ? 'a' : 'b';
  const sizeI = (swap ? regionsRaw.img_size_b : regionsRaw.img_size_a) || [0, 0];
  const sizeJ = (swap ? regionsRaw.img_size_a : regionsRaw.img_size_b) || [0, 0];
  const regs = regionsRaw.regions || [];
  const polysI = regs.map(r => r[sideI] && r[sideI].polygon).filter(p => p && p.length >= 3);
  const polysJ = regs.map(r => r[sideJ] && r[sideJ].polygon).filter(p => p && p.length >= 3);

  return {
    noSemanticForHouse:false,
    i: computeSideOverride(doorsI, polysI, sizeI[0], pair.az_ij),
    j: computeSideOverride(doorsJ, polysJ, sizeJ[0], pair.az_ji),
  };
}
