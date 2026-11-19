"""
[腳本用途]
拼接工具的本機伺服器 (沿用 panorama_viewer 的作法並擴充)。
提供房子清單、連通性資料、ERP 影像、以及人工標註進度的存讀。

[用法]
  python stitch_tool/server.py            # 預設 http://localhost:8010
  python stitch_tool/server.py --port 8010

需先跑過 prepare_house_data.py 以產生 data/*.json。
"""
import os
import re
import sys
import json
import glob
import socket
import argparse
import mimetypes
from urllib.parse import unquote
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)            # 專案根目錄
STATE_DIR = os.path.join(SCRIPT_DIR, "state")     # 人工標註進度（工具本地，非包內）
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)                  # 供 /api/roma_match 延遲載入 roma_match

import gt_hotspot                                 # noqa: E402（同層模組，即時補 GT，見該檔）

# 面向未知資料：工具載入「房子產出包（bundle）」，不再回頭讀完整資料集樹。
# 一個 bundle 根目錄底下每棟房一個子資料夾（見 pack_bundle.py）：
#   <bundle_root>/<house>/{<house>.json, images/, regions/, semantic.json?, *_HOTSPOT.json?}
# 使用時手動指定要看的包（可只放選定的一或數棟）。根目錄＝--bundle > 環境變數
# STITCH_BUNDLE_ROOT > 預設 <ROOT_DIR>/bundles。
# *_HOTSPOT.json？只在 <house>.json 還沒有 GT 欄位時才有作用：/api/house/<house> 讀取時
# 會即時算出 GT 補上（不覆寫既有欄位），對外交付檔（本來就不含 GT）正是最需要這條路徑的
# 情境。邏輯與 prepare_house_data.py 建檔時共用同一份實作，見 gt_hotspot.py。房號對不上
# 或只對上部分影像都會印在伺服器 log，不會安靜略過。這是實際發生過的使用者誤解來源。
BUNDLE_ROOT = os.environ.get("STITCH_BUNDLE_ROOT") or os.path.join(ROOT_DIR, "bundles")

os.makedirs(STATE_DIR, exist_ok=True)

_SAFE = re.compile(r"^[A-Za-z0-9_\-.]+$")         # 房子名 / 檔名白名單，擋路徑穿越


def _safe_id(s):
    """白名單識別名。單獨的 . 與 .. 即使字元集合允許也不通過。"""
    return bool(s) and s not in (".", "..") and _SAFE.match(s) is not None


def _house_dir(house):
    """bundle 內某房的資料夾（不存在回 None）。"""
    d = os.path.join(BUNDLE_ROOT, house)
    return d if os.path.isdir(d) else None


def _images_dir(house):
    """bundle 內某房的影像資料夾。"""
    return os.path.join(BUNDLE_ROOT, house, "images")


# ---- 對外交付檔的相容層 ----
# 包內的 <house>.json 有兩種來源：流程內部的房子資料，或 export_public.py 產出的
# **對外交付檔**（下游實際收到的那一份）。後者刻意不含評估欄位，且幾何證據欄位改用
# 對外名稱。此處在讀取時做名稱對映，讓工具能直接載入未經修改的交付檔。工具跑得起來
# 本身就是「交付內容足夠」的檢驗，跑不起來就當場知道少了什麼。
#
# 對映方向為對外名 → 內部名：前端既有程式碼沿用內部名，換方向要動七處 JS，
# 而本工具未必是最終形態，改動面越小越好。
PUBLIC_TO_TOOL = {"has_geometry": "geo_posed",
                  "n_inlier": "geo_n_inlier",
                  "inlier_rate": "geo_inlier_rate",
                  "conf_used": "geo_conf_used",
                  "az_wedge_i_deg": "geo_az_wedge_i_deg",
                  "az_wedge_j_deg": "geo_az_wedge_j_deg"}


def is_public_house(d):
    """判斷房子資料是否為對外交付檔。

    以 `has_geometry` 為標記：該欄位只存在於交付檔（內部資料用 `geo_posed`）。
    """
    for p in (d.get("pairs") or []):
        return "has_geometry" in p
    return False


def public_house_to_tool(d):
    """對外交付檔 → 工具可讀的房子資料。不修改輸入。

    做三件事：幾何證據欄位改回內部名；標明無 GT 真值（交付檔本身不含真值，故先寫死
    `has_gt=False`；呼叫端接著會用 `gt_hotspot.augment_house_with_gt` 檢查 bundle
    內有沒有 HOTSPOT 檔，有的話這裡的 `False` 會被即時算出的結果覆蓋，兩處因此保持
    一致，不會出現「明明補了 GT，`has_gt` 卻還是 False」的矛盾）；把每對內嵌的
    共視區域抽走，改由 `/api/regions` 供應，維持前端既有的取用方式，也避免房子資料
    一次回傳全部多邊形。
    """
    out = dict(d)
    pairs = []
    for p in (d.get("pairs") or []):
        q = {PUBLIC_TO_TOOL.get(k, k): v for k, v in p.items() if k != "regions"}
        if "geo_posed" in q:
            q["geo_posed"] = int(bool(q["geo_posed"]))
        pairs.append(q)
    out["pairs"] = pairs
    out["has_gt"] = False
    out.setdefault("n_gt_connected_pairs", 0)
    return out


def embedded_regions(d, a, b):
    """從對外交付檔取某一對的共視區域，組成 `/api/regions` 的回應。

    交付檔的區域側別已正規化為「a 側 = 該對的 i 圖」，故 swap 只取決於查詢時
    哪張圖當上半部。回傳 None 表示這一對沒有區域。
    """
    stem = lambda s: os.path.splitext(s)[0]
    names = [stem(x) for x in (d.get("images") or [])]
    W, H = d.get("width"), d.get("height")
    for p in (d.get("pairs") or []):
        if not p.get("regions"):
            continue
        try:
            ni, nj = names[p["i"]], names[p["j"]]
        except (IndexError, KeyError):
            continue
        if {ni, nj} != {a, b}:
            continue
        # 同一棟房的影像尺寸相同，交付檔以房層 width/height 記錄一次。
        return {"regions": p["regions"], "swap": (a != ni),
                "img_size_a": [W, H], "img_size_b": [W, H]}
    return None

# ---- RoMa 即時比對（/api/roma_match）：延遲載入單例 ----
# RoMaMatcher 建構會載入 torch + CUDA + 模型權重，成本不小；一般標註工作完全用不到它，
# 所以只在第一次真的點了「RoMa 比對」鈕時才 import/建構，且只建構一次（快取的圖對之後秒回）。
_roma_matcher = None
_roma_matcher_error = None


def _get_roma_matcher():
    global _roma_matcher, _roma_matcher_error
    if _roma_matcher is not None:
        return _roma_matcher
    if _roma_matcher_error is not None:
        raise RuntimeError(_roma_matcher_error)
    try:
        # 將專案根目錄加入 sys.path。
        if ROOT_DIR not in sys.path:
            sys.path.append(ROOT_DIR)
        from roma_match.matcher import RoMaMatcher
        _roma_matcher = RoMaMatcher(conf_threshold=0.5, use_cache=True, model_type="roma_v2")
        return _roma_matcher
    except Exception as e:
        _roma_matcher_error = f"RoMa matcher unavailable ({e}). Check RoMa weights / CUDA setup."
        raise RuntimeError(_roma_matcher_error)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=SCRIPT_DIR, **k)

    def end_headers(self):
        # 本地開發工具：JS/HTML/CSS 一律 no-store。瀏覽器對 ES module 的快取很兇，
        # 改完程式常常連 Ctrl+F5 都帶不動子模組（歷史痛點，通用規則有記載）。從源頭關掉。
        p = self.path.split("?", 1)[0].lower()
        if p == "/" or p.endswith((".js", ".html", ".css")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # ---- helpers ----
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg):
        self._json({"error": msg}, code)

    # ---- routing ----
    def do_GET(self):
        path = self.path.split("?", 1)[0]

        if path == "/" or path == "":
            self.path = "/index.html"
            return SimpleHTTPRequestHandler.do_GET(self)

        if path == "/api/houses":
            # 掃 bundle 根目錄底下每個 <house>/<house>.json。
            houses = []
            for hd in sorted(glob.glob(os.path.join(BUNDLE_ROOT, "*"))):
                house = os.path.basename(hd)
                hj = os.path.join(hd, f"{house}.json")
                if not os.path.isfile(hj):
                    continue
                try:
                    d = json.load(open(hj, encoding="utf-8"))
                    houses.append({"house": d["house"], "n_images": d["n_images"],
                                   "n_pairs": len(d["pairs"]),
                                   "n_gt_connected_pairs": d.get("n_gt_connected_pairs", 0)})
                except Exception:
                    pass
            return self._json(houses)

        m = re.match(r"^/api/house/([^/]+)$", path)
        if m:
            house = unquote(m.group(1))
            if not _safe_id(house):
                return self._err(400, "bad house name")
            hdir = os.path.join(BUNDLE_ROOT, house)
            fp = os.path.join(hdir, f"{house}.json")
            if not os.path.exists(fp):
                return self._err(404, "house not in bundle; run pack_bundle.py")
            d = json.load(open(fp, encoding="utf-8"))
            d = public_house_to_tool(d) if is_public_house(d) else d
            # 即時補 GT：先做完對外交付檔的相容轉換（上面那行）才檢查，因為轉換後的
            # has_gt 才是這次要回給前端的真實狀態（見 public_house_to_tool 的說明）。
            # 找不到 HOTSPOT 或房子資料本來就有 GT 都不印東西，只有「動手補了」或
            # 「補的時候發現對不上」才值得佔一行 log，安靜地不顯示 GT 正是使用者
            # 曾經誤以為功能沒接上的原因。
            d, gt_msg = gt_hotspot.augment_house_with_gt(d, hdir)
            if gt_msg:
                print(f"[gt] {house}: {gt_msg}", file=sys.stderr)
            return self._json(d)

        m = re.match(r"^/api/state/([^/]+)$", path)
        if m:
            house = unquote(m.group(1))
            if not _safe_id(house):
                return self._err(400, "bad house name")
            fp = os.path.join(STATE_DIR, f"{house}.json")
            if os.path.exists(fp):
                return self._json(json.load(open(fp, encoding="utf-8")))
            return self._json({})

        m = re.match(r"^/images/([^/]+)/([^/]+)$", path)
        if m:
            house, fname = unquote(m.group(1)), unquote(m.group(2))
            if not (_safe_id(house) and _safe_id(fname)):
                return self._err(400, "bad path")
            return self._serve_image(house, fname)

        # GET /api/regions/<house>?a=<imgA檔名>&b=<imgB檔名>
        # 讀包內共視區域 JSON（<bundle>/<house>/regions/{house}_{a}_{b}.json，
        # schema wp1-step3-v1/v2，見 pose_regions/README.md）。檔名只存 A_B 一個方向；
        # 若命中反序檔則回 swap:true，前端把 regions 的 a/b 兩側對調使用。
        m = re.match(r"^/api/regions/([^/]+)$", path)
        if m:
            house = unquote(m.group(1))
            if not _safe_id(house):
                return self._err(400, "bad house name")
            from urllib.parse import parse_qs
            q = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
            a = os.path.splitext((q.get("a") or [""])[0])[0]
            b = os.path.splitext((q.get("b") or [""])[0])[0]
            if not (a and b and _safe_id(a) and _safe_id(b)):
                return self._err(400, "bad image name")
            rdir = os.path.join(BUNDLE_ROOT, house, "regions")
            p_ab = os.path.join(rdir, f"{house}_{a}_{b}.json")
            p_ba = os.path.join(rdir, f"{house}_{b}_{a}.json")
            if os.path.exists(p_ab):
                d = json.load(open(p_ab, encoding="utf-8")); d["swap"] = False
            elif os.path.exists(p_ba):
                d = json.load(open(p_ba, encoding="utf-8")); d["swap"] = True
            else:
                # 對外交付檔的包沒有 regions/，區域內嵌在每一對裡。
                hp = os.path.join(BUNDLE_ROOT, house, f"{house}.json")
                hd = json.load(open(hp, encoding="utf-8")) if os.path.exists(hp) else {}
                d = embedded_regions(hd, a, b) if is_public_house(hd) else None
                if d is None:
                    return self._err(404, "此對沒有區域 JSON（先跑 pose_regions/batch_pose_from_cache.py）")
            d.pop("inliers", None)   # 遮罩用不到內點明細，省流量（每檔可達數百 KB）
            return self._json(d)

        # GET /api/semantic_doors/<house>：子計畫一門語意圖抽取結果（見
        # prepare_semantic_doors.py / stitch_tool/semantic/<house>.json）。
        # 該房沒有語意圖時缺檔，回 404，前端顯示「此房無語意圖」而不觸發校正。
        m = re.match(r"^/api/semantic_doors/([^/]+)$", path)
        if m:
            house = unquote(m.group(1))
            if not _safe_id(house):
                return self._err(400, "bad house name")
            fp = os.path.join(BUNDLE_ROOT, house, "semantic.json")
            if not os.path.exists(fp):
                return self._err(404, "此包無門語意（bundle 內無 semantic.json）")
            return self._json(json.load(open(fp, encoding="utf-8")))

        # 其餘交給靜態檔 (index.html 等)
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        m = re.match(r"^/api/state/([^/]+)$", path)
        if m:
            house = unquote(m.group(1))
            if not _safe_id(house):
                return self._err(400, "bad house name")
            try:
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) if n else b"{}"
                obj = json.loads(raw.decode("utf-8"))
            except Exception as e:
                return self._err(400, f"bad json: {e}")
            fp = os.path.join(STATE_DIR, f"{house}.json")
            json.dump(obj, open(fp, "w", encoding="utf-8"), ensure_ascii=False)
            return self._json({"ok": True, "saved": os.path.basename(fp)})

        m = re.match(r"^/api/roma_match/([^/]+)$", path)
        if m:
            house = unquote(m.group(1))
            if not _safe_id(house):
                return self._err(400, "bad house name")
            try:
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) if n else b"{}"
                body = json.loads(raw.decode("utf-8"))
            except Exception as e:
                return self._err(400, f"bad json: {e}")
            return self._do_roma_match(house, body)

        if path == "/api/bundle_put":
            return self._do_bundle_put()

        return self._err(404, "no such endpoint")

    def _do_roma_match(self, house, body):
        """POST /api/roma_match/<house> {img_a, img_b, conf?} -> {matches, w, h, count}.
        img_a/img_b are image basenames (as in house JSON's `images` list). Runs RoMa direct
        matching (reuses/produces the SAME roma_match/cache/*.npz files as the offline
        run_roma_above_threshold.py batch script, so repeat clicks on an already-matched pair
        are instant). Never raises into the request handler - errors come back as JSON so a
        missing GPU/model doesn't take down the rest of the annotation tool."""
        img_a, img_b = body.get("img_a"), body.get("img_b")
        if not img_a or not img_b:
            return self._err(400, "img_a and img_b required")
        if not (_safe_id(img_a) and _safe_id(img_b)):
            return self._err(400, "bad image name")
        img_dir = _images_dir(house)
        if not os.path.isdir(img_dir):
            return self._err(404, "house image dir not found in bundle")
        path_a, path_b = os.path.join(img_dir, img_a), os.path.join(img_dir, img_b)
        if not (os.path.isfile(path_a) and os.path.isfile(path_b)):
            return self._err(404, "image(s) not found")

        try:
            conf = float(body.get("conf", 0.5))
        except (TypeError, ValueError):
            conf = 0.5

        try:
            matcher = _get_roma_matcher()
        except Exception as e:
            return self._err(503, str(e))

        matcher.conf_threshold = conf
        try:
            matches, info = matcher.match_direct(path_a, path_b)
        except Exception as e:
            return self._err(500, f"RoMa matching failed: {e}")

        return self._json({
            "matches": matches, "w": info.get("w"), "h": info.get("h"), "count": len(matches),
        })

    def _do_bundle_put(self):
        """POST /api/bundle_put?rel=<house>/<subpath>  body=檔案位元組。
        面向未知資料的「＋加房」：前端資料夾選擇器把選到的每個檔逐一上傳，server 依相對
        路徑寫進 bundle 根目錄。rel 的每一段都做白名單驗證，擋路徑穿越（.. 或非法字元）。"""
        from urllib.parse import parse_qs
        q = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
        rel = unquote((q.get("rel") or [""])[0]).replace("\\", "/")
        parts = [p for p in rel.split("/") if p not in ("", ".")]
        if not parts or any(p == ".." or not _safe_id(p) for p in parts):
            return self._err(400, "bad rel path")
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(n) if n else b""
        except Exception as e:
            return self._err(400, f"read fail: {e}")
        dst = os.path.join(BUNDLE_ROOT, *parts)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(data)
        return self._json({"ok": True, "wrote": "/".join(parts), "bytes": len(data)})

    def _serve_image(self, house, fname):
        fp = os.path.join(_images_dir(house), fname)
        if not os.path.isfile(fp):
            return self._err(404, "image not found in bundle")
        ctype = mimetypes.guess_type(fp)[0] or "application/octet-stream"
        try:
            size = os.path.getsize(fp)
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            with open(fp, "rb") as f:
                while True:
                    chunk = f.read(1 << 16)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt, *args):
        # 安靜些，只印非影像請求
        if "/images/" not in (self.path or ""):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main():
    global BUNDLE_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--bundle", default=None,
                    help="房子產出包根目錄（預設環境變數 STITCH_BUNDLE_ROOT 或 <root>/bundles）")
    args = ap.parse_args()
    if args.bundle:
        BUNDLE_ROOT = args.bundle

    n_house = sum(1 for hd in glob.glob(os.path.join(BUNDLE_ROOT, "*"))
                  if os.path.isfile(os.path.join(hd, os.path.basename(hd) + ".json")))
    with ThreadingHTTPServer(("", args.port), Handler) as httpd:
        ip = get_ip()
        print("=" * 60)
        print("  拼接標註工具 (Stitch Labeler) 已啟動")
        print(f"  產出包根目錄: {BUNDLE_ROOT}")
        print(f"  房子數: {n_house}")
        print(f"  本機:   http://localhost:{args.port}")
        print(f"  區網:   http://{ip}:{args.port}")
        print("  Ctrl+C 結束")
        print("=" * 60)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止。")


if __name__ == "__main__":
    main()
