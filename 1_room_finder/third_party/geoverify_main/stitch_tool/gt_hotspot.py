"""
[模組用途]
HOTSPOT 標註真值（Ground Truth）的載入、對映、比對回報，供兩個呼叫端共用同一份實作：

  - `prepare_house_data.py`：建 `data/<house>.json` 時把 GT 烘進每一對（`build_house`）。
  - `server.py`：讀取既有房子資料時，若該資料還沒有 GT 欄位，就在 `/api/house/<house>`
    當場算出來補上（`augment_house_with_gt`）。讓「bundle 內帶 HOTSPOT 檔」這件事
    真的有效果，而不只是把檔案原樣存進包裡卻沒有任何程式碼讀它。

規則只寫一份的理由：這個 repo 已經因為「同一條規則兩處各自實作、改一邊忘了改另一邊」
修過兩次 bug（方位規約字串、共視區域欄位裁切），GT 規則不重蹈覆轍。

[HOTSPOT 檔格式]
    {"HOTSPOTOFROOM": [
       {"IDName": "<擁有者>.jpg",
        "Roomidx": 0,
        "ToIDName": {"IDName": ["<鄰接>.jpg", ...]},
        "HSLoc": {"Coordinate": [{"x": 285.0, "y": 1832.0}, ...]},
        "HSLoc_3D": ..., "hscount": ...}]}
`HSLoc_3D` 與 `hscount` 未使用。`Roomidx` 相同表示同房。`ToIDName.IDName[k]` 與
`HSLoc.Coordinate[k]` 逐項對應。

[方位規約]
    az = ((x / W) * 360 - 180) mod 360        W 為該圖原始寬度，0 度＝影像中央、右為正。
與 `upstream_direction.py` 的規約一致（見該檔），但此處是由標註座標直接換算，不經模型。
"""
import glob
import json
import os


def find_hotspot_files(dir_: str, recursive: bool = False):
    """在 `dir_` 底下找所有 `*HOTSPOT*.json`，已排序。

    `recursive=True` 時連子資料夾一併找：面向未知資料，使用者手動組裝或透過「＋加房」
    上傳的 bundle 不保證 HOTSPOT 檔乖乖放在房子資料夾最上層。這不是假設性的邊界情況，
    是實際發生過的真實檔案佈局（誤放進 `images/` 底下）。伺服器即時補 GT 的路徑
    （`augment_house_with_gt`）要靠這個才找得到，不然會安靜漏掉，使用者只會覺得
    「功能沒接上」。

    `prepare_house_data.py` 讀資料集資料夾維持 `recursive=False`：那裡的檔案佈局是
    流程自己產生的（見 `pack_bundle.py` 的 `pack_house`／資料集本身），不需要這層保險，
    範圍收斂也比較不容易誤撿到不相干的同名檔。
    """
    if recursive:
        return sorted(glob.glob(os.path.join(dir_, "**", "*HOTSPOT*.json"), recursive=True))
    return sorted(glob.glob(os.path.join(dir_, "*HOTSPOT*.json")))


def load_gt(hotspot_paths, width) -> dict:
    """讀取一批 `*_HOTSPOT.json`，回傳 `{src_img: {tgt_img: az_deg}}` 有向連接＋門方位。

    `hotspot_paths` 內部一律先排序才處理，呼叫端不必自行排序、傳入順序也不影響結果：
    多樓層的房子每層一個 HOTSPOT 檔，樓梯配對在相鄰兩層皆有標註、座標略有差異
    （實測 5~120 px）。下方為後寫覆蓋前寫，未排序時採用哪一筆取決於呼叫端蒐集檔案的
    順序（例如檔案系統列舉順序，Linux 與 Windows 不一致）。排序收斂在這裡處理一次，
    上層呼叫端（含測試）都不必記得自己排。這是從 `prepare_house_data.py` 抽出來時
    特別保留的既有防呆，原始版本已有相同註解與行為。

    `width`：該房影像的原始寬度（同一棟房影像尺寸相同，呼叫端只需算一次或直接讀房層
    `width` 欄位，不必逐張開圖）。`None` 或 0 時仍回傳連接關係，但方位一律 `None`。
    """
    gt = {}
    for hs_path in sorted(hotspot_paths):
        try:
            J = json.load(open(hs_path, encoding="utf-8"))
        except Exception:
            continue
        for room in J.get("HOTSPOTOFROOM", []):
            src = room.get("IDName")
            if not src:
                continue
            tgts = (room.get("ToIDName", {}) or {}).get("IDName", []) or []
            coords = (room.get("HSLoc", {}) or {}).get("Coordinate", []) or []
            d = gt.setdefault(src, {})
            for k, tgt in enumerate(tgts):
                az = None
                if k < len(coords) and width:
                    x = float(coords[k].get("x", 0.0))
                    az = ((x / width) * 360.0 - 180.0) % 360.0
                d[tgt] = az
    return gt


def load_gt_locations(hotspot_paths) -> dict:
    """讀取 HOTSPOT 像素座標。回傳 `{src_img: {tgt_img: (x, y)}}`。

    排序與後寫覆蓋規則與 `load_gt` 相同。不換算方位；案例圖要把標註打在點上。
    """
    loc = {}
    for hs_path in sorted(hotspot_paths):
        try:
            J = json.load(open(hs_path, encoding="utf-8"))
        except Exception:
            continue
        for room in J.get("HOTSPOTOFROOM", []):
            src = room.get("IDName")
            if not src:
                continue
            tgts = (room.get("ToIDName", {}) or {}).get("IDName", []) or []
            coords = (room.get("HSLoc", {}) or {}).get("Coordinate", []) or []
            d = loc.setdefault(src, {})
            for k, tgt in enumerate(tgts):
                if k >= len(coords):
                    continue
                x = float(coords[k].get("x", 0.0))
                y = float(coords[k].get("y", 0.0))
                d[tgt] = (x, y)
    return loc


def gt_pair_fields(gt: dict, img_i: str, img_j: str) -> dict:
    """把 GT 對映到單一對 `(img_i, img_j)`，回傳房子資料每對要寫入的三個欄位。

    `img_i`／`img_j` 須含副檔名，與 HOTSPOT 的 `IDName`／`ToIDName` 同格式（即 `images`
    清單的原始項）。抽自 `prepare_house_data.build_house` 組 pair dict 那段，
    `server.py` 即時補 GT 時共用同一份對映規則。

    沒有任何標註時 `gt_connected` 為 None：缺席與「已確認不連通」分開表示。
    """
    if not gt:
        return {"gt_connected": None, "gt_az_ij": None, "gt_az_ji": None}
    gt_ij = gt.get(img_i, {})
    gt_ji = gt.get(img_j, {})
    return {
        "gt_connected": bool((img_j in gt_ij) or (img_i in gt_ji)),
        "gt_az_ij": (round(gt_ij[img_j], 2) % 360.0
                     if img_j in gt_ij and gt_ij[img_j] is not None else None),
        "gt_az_ji": (round(gt_ji[img_i], 2) % 360.0
                     if img_i in gt_ji and gt_ji[img_i] is not None else None),
    }


def has_gt_fields(house_data: dict) -> bool:
    """判斷房子資料是否已經算過 GT（無論算出來的結論是有沒有）。

    `has_gt` 為真，或任一對已帶 `gt_connected` 鍵，即使值是 `False`：`has_gt=False`
    且每對都有 `gt_connected` 鍵，代表建檔當時真的找過 HOTSPOT 但沒找到，是「已有答案」
    代表「已有答案」。`augment_house_with_gt` 不應該覆寫它去猜 bundle 資料夾裡湊巧出現的
    檔案。只有 schema 上完全沒有這些欄位（例如 `export_public.py` 的對外交付檔，
    刻意不含 `gt_*`，見該檔 `FORBIDDEN_PREFIXES`）才算「還沒算過」。
    """
    if house_data.get("has_gt"):
        return True
    pairs = house_data.get("pairs") or []
    return bool(pairs) and "gt_connected" in pairs[0]


def _referenced_images(gt: dict) -> set:
    """回傳 HOTSPOT 資料裡出現過的所有影像檔名，不論是當 owner（src）還是 target。"""
    imgs = set(gt.keys())
    for tgts in gt.values():
        imgs.update(tgts.keys())
    return imgs


def match_report(gt: dict, images) -> str:
    """比對 HOTSPOT 提到的影像與本房影像清單，回傳一行人類可讀的摘要。

    使用者曾把別棟房子的 HOTSPOT 放錯資料夾（房號只差一個數字），兩邊影像 UUID
    前綴完全不同、交集為零；安靜地不顯示 GT 只會讓人以為這個功能沒接上（使用者
    原本就是這樣誤解的）。所以無論完全對不上、部分對上、或全數相符，都要把實際
    比對到的張數報出來，不能只在「正常」時才輸出。

    不齊全有**兩個互相獨立的方向**，兩者可能同時發生，故分開判斷後合併敘述：

      - HOTSPOT 提到的影像有些不屬於本房（`n_match < n_hs`）：混入了其他房子的標註。
      - 本房的影像有些沒被標註涵蓋（`n_match < n_house`）：少放了某一層樓的 HOTSPOT。

    只看前者是不夠的：兩層樓的房子只放 floor1 時，被提到的影像確實全都屬於本房，
    前者不成立，若因此回報「全數相符」，使用者會以為真值完整，而未涵蓋的那些影像
    相關的配對會一律顯示 GT 不連通，看起來像模型誤報。這與本函式要防的「安靜誤導」
    是同一類問題，只是方向相反。
    """
    hs_images = _referenced_images(gt)
    house_images = set(images)
    n_hs, n_house = len(hs_images), len(house_images)

    if n_hs == 0:
        return "HOTSPOT 檔案存在，但未解析出任何影像連接（可能是空檔或格式不符）。"

    n_match = len(hs_images & house_images)
    if n_match == 0:
        return ("HOTSPOT 提到的 %d 張影像與本房 %d 張無一相符，可能放錯房"
                "（房號、資料夾接錯，或誤帶了其他房子的標註）。" % (n_hs, n_house))

    notes = []
    if n_match < n_hs:
        notes.append("HOTSPOT 提到的 %d 張影像中有 %d 張不屬於本房（可能混入其他房子的標註）"
                     % (n_hs, n_hs - n_match))
    if n_match < n_house:
        # 措辭保留另一種可能：孤立的全景圖本來就不會出現在任何連接裡，不一定是缺檔。
        notes.append("本房 %d 張影像中有 %d 張未被標註涵蓋（可能少放了某一層樓的 HOTSPOT，"
                     "也可能該圖本來就沒有連接），這些影像的配對會一律顯示 GT 不連通"
                     % (n_house, n_house - n_match))
    if notes:
        return "；".join(notes) + "。"
    return "HOTSPOT %d 張影像與本房全數相符。" % n_hs


def augment_house_with_gt(house_data: dict, house_dir: str):
    """`/api/house/<house>` 讀取路徑上的即時補 GT。

    `house_data` 須是已經要回給前端的房子資料（若原檔是對外交付檔，須先經
    `server.public_house_to_tool` 轉換過，理由見該函式：交付檔的 `has_gt` 目前寫死
    `False`，本函式讀到的正是那個「還沒算過」狀態，才需要接手嘗試補上）。
    `house_dir` 是該房在 bundle 內的資料夾（`<bundle_root>/<house>/`）。

    回傳 `(house_data_out, message)`：
      - `house_data` 已有 GT 欄位（見 `has_gt_fields`）→ 原樣回傳、不動任何檔案系統，
        `message=None`。既有的內部資料包用這條路徑短路，不受影響。
      - bundle 內（含子資料夾）找不到 `*HOTSPOT*.json` → 原樣回傳，`message=None`
        （行為與本功能實作之前相同）。
      - 找到 HOTSPOT → 回傳補好 `gt_connected`／`gt_az_ij`／`gt_az_ji`（每對）與
        `has_gt`／`n_gt_connected_pairs`（房層）的新 dict；`message` 一律非 `None`，
        是 HOTSPOT 影像與本房影像的比對摘要。呼叫端應該把它印出來。安靜地不顯示
        GT 正是本函式存在的理由（使用者曾因此誤以為功能沒接上，見 `match_report`）。
        注意 `has_gt` 不是「HOTSPOT 檔案存在」就設真：若其中提到的影像與本房完全
        對不上（例如房號放錯），代表這份 HOTSPOT 根本不適用於這棟房子，`has_gt`
        仍是 `False`。不然前端會顯示「GT連通 0」，看起來像已驗證的結論，實際只是資料
        不適用，等同產生一份假 GT。
    """
    if has_gt_fields(house_data):
        return house_data, None
    paths = find_hotspot_files(house_dir, recursive=True)
    if not paths:
        return house_data, None

    width = house_data.get("width")
    gt = load_gt(paths, width)
    images = house_data.get("images") or []
    msg = match_report(gt, images)
    n_match = len(_referenced_images(gt) & set(images))

    out = dict(house_data)
    pairs = []
    n_conn = 0
    for p in (house_data.get("pairs") or []):
        q = dict(p)
        i, j = q.get("i"), q.get("j")
        if isinstance(i, int) and isinstance(j, int) and 0 <= i < len(images) and 0 <= j < len(images):
            q.update(gt_pair_fields(gt, images[i], images[j]))
        else:
            # 索引異常（理論上不會發生，防呆用）：退化成「不連通」，不讓單一壞對
            # 拖垮整個房子的回應。
            q.update({"gt_connected": False, "gt_az_ij": None, "gt_az_ji": None})
        if q["gt_connected"]:
            n_conn += 1
        pairs.append(q)
    out["pairs"] = pairs
    # has_gt 只在 HOTSPOT 至少有一張影像真的屬於這棟房子（n_match>0）時才算「有」。
    # 完全對不上時 gt 字典非空（裡面全是別棟房子的資料），但 n_conn 必然是 0。若讓
    # has_gt 只看 bool(gt)，前端會顯示「GT連通 0」，看起來像「已驗證這棟房子零連接」
    # 的結論，實際上是「這份 HOTSPOT 根本不適用於這棟房子」，兩者意義天差地遠，
    # 不能混為一談（也是本函式不能產生假 GT 的關鍵）。
    out["has_gt"] = n_match > 0
    out["n_gt_connected_pairs"] = n_conn
    return out, msg
