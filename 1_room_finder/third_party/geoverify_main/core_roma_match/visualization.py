"""切片匹配視覺化工具 (Confidence Coloring + Diverse Line Selection)。

提供：
- select_diverse_indices: 取信心前 N% 且空間分散的配對子集索引（畫線用）。
- conf_to_bgr / draw_confidence_colorbar: 依信心 plasma 上色與 colorbar。
- visualize_slice_matches: 將單一切片對 (p_a | p_b) 的匹配標註成一張影像。

備註：本模組只依賴 cv2 / numpy，不載入 torch / romatch，方便其他端 (例如
上層模組) 直接 import 重用，與大張映射圖的 plasma 配色保持一致。
"""
import cv2
import numpy as np


def select_diverse_indices(matches, top_percent=0.05, grid=(8, 8),
                           src_w=None, src_h=None, ax_index=(0, 1), conf_index=4):
    """挑選『信心最高且空間分散』的匹配子集索引，用於畫連線。

    - top_percent: 取總數的百分之幾 (0.05 = 前 5%)。
    - grid: (cols, rows)，依來源影像 A 的座標切格，確保不同位置都有代表，
            避免所有線都集中在同一個區域。
    - 作法: 每一格內依信心由高到低排序，再以 round-robin 逐輪從各格挑出
            目前最高信心者（優先全域信心高的格子），直到達到目標數量。
    回傳: matches 中被選中的索引 list。
    """
    n = len(matches)
    if n == 0:
        return []
    target = max(1, int(round(n * top_percent)))

    xs = [m[ax_index[0]] for m in matches]
    ys = [m[ax_index[1]] for m in matches]
    w = src_w if src_w else (max(xs) + 1e-6)
    h = src_h if src_h else (max(ys) + 1e-6)
    cols, rows = grid

    from collections import defaultdict
    cells = defaultdict(list)  # (cx, cy) -> [index, ...]
    for i, m in enumerate(matches):
        cx = min(int(m[ax_index[0]] / w * cols), cols - 1)
        cy = min(int(m[ax_index[1]] / h * rows), rows - 1)
        cells[(cx, cy)].append(i)

    # 每格內依信心由高到低排序
    for k in cells:
        cells[k].sort(key=lambda idx: matches[idx][conf_index], reverse=True)

    pointers = {k: 0 for k in cells}
    selected = []
    while len(selected) < target:
        avail = [(k, matches[cells[k][pointers[k]]][conf_index])
                 for k in cells if pointers[k] < len(cells[k])]
        if not avail:
            break
        avail.sort(key=lambda x: x[1], reverse=True)
        for k, _ in avail:
            if len(selected) >= target:
                break
            selected.append(cells[k][pointers[k]])
            pointers[k] += 1
    return selected


def conf_to_bgr(confs, vmin, vmax):
    """將信心值陣列依固定 vmin/vmax 映射成 plasma 的 BGR 顏色 (0-255)。

    使用 OpenCV 內建 colormap，與 generate_roma_match_vis.py 的 plasma 一致；
    若該版本 OpenCV 沒有 PLASMA 則退回 VIRIDIS。
    """
    confs = np.asarray(confs, dtype=np.float32)
    norm = (confs - vmin) / (vmax - vmin + 1e-6)
    idx = np.clip(norm * 255.0, 0, 255).astype(np.uint8).reshape(-1, 1)
    cmap = getattr(cv2, 'COLORMAP_PLASMA', cv2.COLORMAP_VIRIDIS)
    return cv2.applyColorMap(idx, cmap).reshape(-1, 3)  # (N, 3) BGR


def draw_confidence_colorbar(img, vmin, vmax, width=18, pad=12, label="Conf"):
    """在影像右側加上一條 plasma 垂直 colorbar 與 min/max 標籤。"""
    h, w = img.shape[:2]
    bar_h = int(h * 0.7)
    y0 = (h - bar_h) // 2
    grad = np.linspace(255, 0, bar_h).astype(np.uint8).reshape(-1, 1)
    cmap = getattr(cv2, 'COLORMAP_PLASMA', cv2.COLORMAP_VIRIDIS)
    bar = cv2.applyColorMap(grad, cmap)            # (bar_h, 1, 3)
    bar = np.repeat(bar, width, axis=1)            # (bar_h, width, 3)

    extra = width + pad + 70
    canvas = cv2.copyMakeBorder(img, 0, 0, 0, extra,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
    x0 = w + pad
    canvas[y0:y0 + bar_h, x0:x0 + width] = bar
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"{vmax:.2f}", (x0 - 2, y0 - 6), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{vmin:.2f}", (x0 - 2, y0 + bar_h + 16), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, label, (x0 - 2, y0 + bar_h // 2), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def visualize_slice_matches(p_a, p_b, matches, draw_lines=True,
                            line_percent=0.05, line_alpha=0.5):
    """將單一切片對 (p_a | p_b) 的匹配視覺化成一張 BGR 影像。

    - 所有匹配點依信心以 plasma 上色（取代原本單一綠色）。
    - 連線只畫『信心前 line_percent 且空間分散』的子集，並以半透明
      疊圖呈現，使背景物體仍清晰可見。
    - 右側附上 colorbar。
    p_a / p_b 為等高的 BGR numpy 影像；matches 為 (xa, ya, xb, yb, conf)。
    """
    vis = np.hstack([p_a, p_b])
    offset = p_a.shape[1]  # B 圖在合併圖中的水平位移

    if not matches:
        return vis

    m_arr = np.array(matches, dtype=np.float32)
    confs = m_arr[:, 4]
    vmin, vmax = float(confs.min()), float(confs.max())
    colors = conf_to_bgr(confs, vmin, vmax)  # (N, 3) BGR，點與線共用

    # --- 連線 (信心前 N% + 空間分散，半透明) ---
    if draw_lines:
        sel = select_diverse_indices(
            matches, top_percent=line_percent, grid=(8, 8),
            src_w=p_a.shape[1], src_h=p_a.shape[0])
        if sel:
            overlay = vis.copy()
            for i in sel:
                xa, ya, xb, yb = m_arr[i, 0], m_arr[i, 1], m_arr[i, 2], m_arr[i, 3]
                c = tuple(int(v) for v in colors[i])
                cv2.line(overlay,
                         (int(xa), int(ya)),
                         (int(xb) + offset, int(yb)),
                         c, 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, line_alpha, vis, 1 - line_alpha, 0, vis)

    # --- 所有點 (依信心上色) ---
    for i in range(len(matches)):
        c = tuple(int(v) for v in colors[i])
        cv2.circle(vis, (int(m_arr[i, 0]), int(m_arr[i, 1])), 2, c, -1, cv2.LINE_AA)
        cv2.circle(vis, (int(m_arr[i, 2]) + offset, int(m_arr[i, 3])), 2, c, -1, cv2.LINE_AA)

    return draw_confidence_colorbar(vis, vmin, vmax, label="Conf")
