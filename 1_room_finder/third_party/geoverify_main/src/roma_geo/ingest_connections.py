"""
[模組用途]
把上游連通性偵測模型輸出的 candidate-connections JSON，轉成本專案既有的
「連通性矩陣」格式（matrices/<版本>/），讓下游 pipeline 零改動接上。

    <上游輸出目錄>/<house>.json   →   <out_dir>/<house>.npy + <house>_images.json

低耦合：只讀上游的檔、寫本專案的矩陣，不 import 上游的 code。之後換一批（test set／
新版本）只要改 --input_dir / --out_dir，不動任何下游。

[上游 JSON 格式（2026-08-12 版、已驗證）]
    { house_id, threshold(0.5), notes, view_convention, pairs:[
        { pano_a:"<uuid>.jpg", pano_b:"<uuid>.jpg", probability:0~1,
          corresponding_views:{ view_a, yaw_a_degrees, view_b, yaw_b_degrees } } ] }
    yaw 規約：影像中心=0°、右為正、0~360、6 view（0/60/120/180/240/300）。
    ⚠ 這與本專案內部規約一致（az(x)=((x/W)*360-180)%360，0°=中央、右為正），
      實測對齊、無 180° 翻面；故 yaw 直接當 az，不做轉換。

[矩陣契約（見 matrices/README.md）]
    <house>.npy (N,N,3): ch0=連通 logit · ch1=sin(az) · ch2=cos(az)
    <house>_images.json: 長度 N 影像檔名（含副檔名），順序與 row/col 對齊
    - ch0：prepare_house_data 以 prob=sigmoid(ch0) 顯示 ⇒ 這裡存 logit=log(p/(1-p))，
      還原後 prob 逐位等於上游的機率；prob>=0.5 對應 logit 門檻 0.0。
    - ch1/ch2：M[i,j] 存「在 i 圖看向 j」的方位（a=i 時取 yaw_a，否則 yaw_b）。
"""
import argparse
import json
import math
import os

import numpy as np

from .config import PROJECT_ROOT
from .errors import RomaGeoError
from .names import glob_in_dir, require_path_component

HERE = PROJECT_ROOT
DEFAULT_INPUT_FALLBACK = os.path.normpath(
    os.path.join(HERE, "..", "candidate_connections_validation"))
DEFAULT_OUT = os.path.join(HERE, "matrices", "candidate_val")


def default_input_dir():
    """上游連通 JSON 目錄。讀取當下的環境變數，不在匯入時固定。"""
    return os.environ.get("ROMA_GEO_CONNECTIONS") or DEFAULT_INPUT_FALLBACK

PROB_EPS = 1e-6  # 夾住 0/1 避免 logit 爆 inf
TOO_FEW_IMAGES = "至少需要兩張影像才能組成連通矩陣"


def prob_to_logit(p: float) -> float:
    p = min(max(float(p), PROB_EPS), 1.0 - PROB_EPS)
    return math.log(p / (1.0 - p))


def build_matrix(cl: dict, input_name="上游連通 JSON"):
    """上游單一 house 的 JSON dict → (M(N,N,3), images[N])。回傳也附一份 pair 明細供自檢。"""
    if not isinstance(cl, dict):
        raise RomaGeoError(
            "頂層型別為 %s" % type(cl).__name__,
            input_name=input_name,
            condition="頂層須為物件",
        )
    pairs = cl.get("pairs")
    if not isinstance(pairs, list):
        raise RomaGeoError(
            "pairs 型別為 %s" % type(pairs).__name__,
            input_name=input_name,
            condition="pairs 須為陣列",
        )
    for i, p in enumerate(pairs):
        if not isinstance(p, dict):
            raise RomaGeoError(
                "pairs[%d] 型別為 %s" % (i, type(p).__name__),
                input_name=input_name,
                condition="每一項配對須為物件",
            )
        for key in ("pano_a", "pano_b"):
            val = p.get(key)
            if not isinstance(val, str) or not val:
                raise RomaGeoError(
                    "%s=%r" % (key, val),
                    input_name=input_name,
                    condition="影像名須為非空字串",
                )
            require_path_component(val, input_name=input_name)
        if "probability" not in p:
            raise RomaGeoError(
                "pairs[%d] 缺少 probability" % i,
                input_name=input_name,
                condition="必要欄位缺席",
            )
        try:
            float(p["probability"])
        except (TypeError, ValueError) as e:
            raise RomaGeoError(
                "probability=%r" % (p["probability"],),
                input_name=input_name,
                condition="pairs[%d] 的 probability 須為數值" % i,
            ) from e
    names = sorted({p["pano_a"] for p in pairs} | {p["pano_b"] for p in pairs})
    idx = {n: k for k, n in enumerate(names)}
    N = len(names)
    if N < 2:
        raise RomaGeoError(
            "影像數 %d" % N,
            input_name=input_name,
            condition=TOO_FEW_IMAGES,
        )
    M = np.zeros((N, N, 3), dtype=np.float32)

    for p in pairs:
        i, j = idx[p["pano_a"]], idx[p["pano_b"]]
        lg = prob_to_logit(p["probability"])
        M[i, j, 0] = lg
        M[j, i, 0] = lg  # 連通分數無方向，兩向同值
        cv = p.get("corresponding_views") or {}
        ya = cv.get("yaw_a_degrees")
        yb = cv.get("yaw_b_degrees")
        # ch1/ch2 有方向：M[i,j]=在 pano_a(i) 看向 pano_b(j)=yaw_a；M[j,i]=yaw_b。
        if ya is not None:
            M[i, j, 1], M[i, j, 2] = math.sin(math.radians(ya)), math.cos(math.radians(ya))
        if yb is not None:
            M[j, i, 1], M[j, i, 2] = math.sin(math.radians(yb)), math.cos(math.radians(yb))
    return M, names, N


def selfcheck(cl: dict, M: np.ndarray, names) -> dict:
    """把矩陣解碼回去，確認 prob / yaw 逐對還原上游原值。回傳最大誤差。"""
    idx = {n: k for k, n in enumerate(names)}
    max_dp, max_daz = 0.0, 0.0
    for p in cl["pairs"]:
        i, j = idx[p["pano_a"]], idx[p["pano_b"]]
        prob_back = 1.0 / (1.0 + math.exp(-float(M[i, j, 0])))
        max_dp = max(max_dp, abs(prob_back - p["probability"]))
        cv = p.get("corresponding_views") or {}
        for (a, b), y in ((( i, j), cv.get("yaw_a_degrees")), ((j, i), cv.get("yaw_b_degrees"))):
            if y is None:
                continue
            az_back = math.degrees(math.atan2(float(M[a, b, 1]), float(M[a, b, 2]))) % 360.0
            d = abs((az_back - (y % 360.0) + 180.0) % 360.0 - 180.0)
            max_daz = max(max_daz, d)
    return {"max_prob_err": max_dp, "max_az_err_deg": max_daz}


def convert_connections(input_dir, out_dir):
    """上游連通 JSON 目錄 → 連通矩陣目錄。回傳寫出的房子名清單。"""
    files = sorted(glob_in_dir(input_dir, "*.json"))
    if not files:
        raise RomaGeoError(
            "底下沒有 *.json",
            input_name="上游連通輸出",
            condition="目錄 %s 沒有 JSON" % input_dir,
        )

    prepared = []
    skipped = 0
    for f in files:
        base = os.path.basename(f)
        try:
            with open(f, encoding="utf-8") as fh:
                cl = json.load(fh)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            raise RomaGeoError(
                "%s：%s" % (type(e).__name__, e),
                input_name="上游連通 JSON %s" % base,
                condition="無法解析為 JSON",
            ) from e
        house = cl.get("house_id") or os.path.splitext(base)[0]
        require_path_component(house, input_name="上游連通 JSON %s" % base)
        try:
            M, names, N = build_matrix(cl, input_name="上游連通 JSON %s" % base)
        except RomaGeoError as e:
            if e.condition == TOO_FEW_IMAGES:
                print("[skip] %s: 影像數不足，不寫矩陣" % house)
                skipped += 1
                continue
            raise
        except KeyError as e:
            raise RomaGeoError(
                "缺少必要欄位 %s" % e,
                input_name="上游連通 JSON %s" % base,
                condition="必要欄位缺席",
            ) from e
        prepared.append((base, house, cl, M, names, N))

    if not prepared:
        raise RomaGeoError(
            "全部 %d 份影像數不足，無法組成連通矩陣" % skipped if skipped else "底下沒有可轉的 JSON",
            input_name="上游連通輸出",
            condition=TOO_FEW_IMAGES if skipped else "目錄 %s 沒有可匯出的 JSON" % input_dir,
        )

    seen = {}
    for base, house, _cl, _M, _names, _N in prepared:
        if house in seen:
            raise RomaGeoError(
                "識別名稱 %s 同時出現在 %s 與 %s" % (house, seen[house], base),
                input_name="上游連通 JSON %s" % base,
                condition="同一批來源的識別名稱重複",
            )
        seen[house] = base

    os.makedirs(out_dir, exist_ok=True)
    worst = {"max_prob_err": 0.0, "max_az_err_deg": 0.0}
    houses = []
    for _base, house, cl, M, names, N in prepared:
        n_pairs = len(cl["pairs"])
        complete = (N * (N - 1) // 2 == n_pairs)
        chk = selfcheck(cl, M, names)
        worst = {k: max(worst[k], chk[k]) for k in worst}

        np.save(os.path.join(out_dir, f"{house}.npy"), M)
        from .export_public import _atomic_write_json
        _atomic_write_json(os.path.join(out_dir, "%s_images.json" % house), names)
        flag = "" if complete else f"  非完整圖(N*(N-1)/2={N*(N-1)//2}≠{n_pairs})"
        print(f"[ok] {house}: N={N} pairs={n_pairs} "
              f"maxΔprob={chk['max_prob_err']:.2e} maxΔaz={chk['max_az_err_deg']:.2e}°{flag}")
        houses.append(house)

    print(f"\n寫出 {len(houses)} 房 → {out_dir}")
    print(f"自檢最大誤差：Δprob={worst['max_prob_err']:.2e}  Δaz={worst['max_az_err_deg']:.2e}°")
    return houses


def main():
    ap = argparse.ArgumentParser(description="candidate-connections JSON → matrices/<版本>/")
    ap.add_argument("--input_dir", default=default_input_dir(),
                    help="上游輸出的 JSON 目錄（預設 env ROMA_GEO_CONNECTIONS 或 ../candidate_connections_validation）")
    ap.add_argument("--out_dir", default=DEFAULT_OUT,
                    help="矩陣輸出目錄（一個版本一個目錄；預設 matrices/candidate_val）")
    args = ap.parse_args()
    try:
        convert_connections(args.input_dir, args.out_dir)
    except RomaGeoError as e:
        raise SystemExit("[error] %s" % e)
    print("下游用法：prepare_house_data.py --oof_dir <out_dir>  或  設定 ROMA_GEO_MATRICES=<out_dir>")
    print("連通門檻：ch0 是 logit，prob>=0.5 ⇔ logit>=0.0（select_pairs 的門檻設 0.0）")


if __name__ == "__main__":
    main()
