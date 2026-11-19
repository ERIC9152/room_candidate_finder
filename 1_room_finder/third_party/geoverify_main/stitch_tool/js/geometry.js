/* =========================================================================
   geometry.js：方位 <-> 球面幾何（純函式）
   低層模組：僅依賴全域 THREE（由 index.html 的 classic script 提供）。
   公式逐字照抄 SPEC.md「方位↔球面幾何」，已數值驗證正確，切勿改動。
   ========================================================================= */

import { MARKER_R } from './config.js';

/* global THREE */

/**
 * 方位角 az（度）-> 球面 3D 座標（放在半徑 MARKER_R、水平線 v=0.5）。
 * @param {number} az 方位角，度 [0,360)
 * @returns {THREE.Vector3}
 */
export function azToPos(az){
  const u = ((az + 180) % 360) / 360, v = 0.5;
  return new THREE.Vector3(
    MARKER_R * Math.cos(2 * Math.PI * u) * Math.sin(Math.PI * v),
    MARKER_R * Math.cos(Math.PI * v),
    MARKER_R * Math.sin(2 * Math.PI * u) * Math.sin(Math.PI * v)
  );
}

/**
 * 球面命中點 P -> 方位角 az（度）。azToPos 的逆運算。
 * @param {THREE.Vector3} P 命中點
 * @returns {number} 方位角，度 [0,360)
 */
export function posToAz(P){
  const u = (Math.atan2(P.z, P.x) / (2 * Math.PI) + 1) % 1;
  return (u * 360 - 180 + 360) % 360;
}

/**
 * 方位角 -> panorama_viewer 相機經度 lon。
 * @param {number} az 方位角，度
 * @returns {number} lon，度 [0,360)
 */
export function azToLon(az){ return (az + 180) % 360; }

/**
 * 兩方位角的最小夾角（度，0..180）。
 * @param {number} a 度
 * @param {number} b 度
 * @returns {number} 夾角，度
 */
export function angDiff(a, b){
  return Math.abs(((a - b) % 360 + 540) % 360 - 180);
}
