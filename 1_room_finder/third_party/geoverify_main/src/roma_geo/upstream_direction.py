"""
[模組用途]
把上游粗篩模型的方向輸出換算成本專案的方位角規約。

方向有兩個來源：幾何驗證（`az_source="geo"`，RoMa 有足夠對應且解得出位姿時採用）、
上游粗篩（`az_source="coarse"`，其餘所有情況）。本模組負責後者的解讀。

[方位角規約]
    az(x) = ((x / W) * 360 - 180) mod 360        0° = 影像中央，±180° = 接縫

⚠ 子計畫一的 `segmentation/aligner.py` 採用 `az = (x / W) * 360`（0° = 影像左緣），
 **與本專案差 180°**。接新來源前必須先確認對方採用哪一種規約，否則方向會整批反向。

[已提供的解碼]
    from_sincos   矩陣 ch1=sin(az) / ch2=cos(az)
    from_x        ERP 影像上的 x 座標
    from_box      影像上的框，取中心 x（含跨接縫處理）

[怎麼接新的上游]
上游粗篩之後將由子計畫的模型整體取代，其輸出型態未定（方位角或影像上的框），
故本模組尚未含讀檔邏輯。格式確定後在此新增

    load_<來源名>(house) -> {frozenset({img_i, img_j}): (az_ij, az_ji)}

並在 `prepare_house_data.build_house` 決定方向處接上；路徑走 `config.py`，不要寫死。
`stitch_tool/prepare_house_data.py` 的呼叫端不必更動。

⚠ RoMa 對應不足時沒有位姿也沒有共視區域，方向只能來自上游。若上游給的是框，
框中心可作為門口位置，但**必須標記來源**：那是模型預測，與幾何驗證過的共視區域不同。
"""
import math

#: 方位角規約字串，與 `stitch_tool` house JSON 的 `az_convention` 欄位一致。
CONVENTION = "az(x)=((x/W)*360-180)%360 ; 0deg = image center"


def from_sincos(sin_v, cos_v) -> float:
    """矩陣 ch1/ch2 → 方位角（度，0~360）。"""
    return math.degrees(math.atan2(float(sin_v), float(cos_v))) % 360.0


def from_x(x, width) -> float:
    """ERP 影像上的 x 座標 → 方位角（度，0~360）。"""
    return ((float(x) / float(width)) * 360.0 - 180.0) % 360.0


def from_box(box, width) -> float:
    """ERP 影像上的框 → 方位角（度，0~360）。`box` = `(x0, y0, x1, y1)`，取中心 x。

    `width` 須為該圖的原始寬（**per-house 不同**，實測 5376 ~ 8192）。
    框橫跨接縫時（`x0 > x1`，例如 8000 → 200）以接縫為界計算中心，
    直接取平均會得到影像正中央這個錯誤答案。
    """
    x0, x1 = float(box[0]), float(box[2])
    w = float(width)
    cx = 0.5 * (x0 + x1) if x1 >= x0 else (0.5 * (x0 + x1 + w)) % w
    return from_x(cx, w)
