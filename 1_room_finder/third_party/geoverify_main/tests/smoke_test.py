"""
[腳本用途]
Roma_Geo 功能煙霧測試：透過對外入口 `roma_geo.verify_pair()` 跑完整核心路徑
（RoMa 匹配 → 幾何驗證 → 共視區域），並與 Output/roma_regions/ 既有的
參考 JSON（由 torch 2.6.0+cu124 / numpy 2.4.3 環境產出）逐欄對照。

固定測試配對：DollhouseTask_58553 …59efa7a0 ↔ …0b14c8da
（GT 連通、跨房門洞，參考值 876 用點 / 593 內點 / 67.7% / 3 區域 / conf 0.65）。

verify_pair 的預設參數與正式批次 batch_pose_from_cache.py 相同（含 lat_max=60），
所以這裡的比對是**同參數**的，差異只來自環境與 RoMa 的隨機抽樣。

[用法]
  venv\\Scripts\\python.exe tests\\smoke_test.py

`tests/run_all.py` 會在偵測到 GPU 與資料集時一併執行本腳本；兩者缺一即自動略過。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
sys.path.append(ROOT)

from roma_geo.config import datasets_root                    # noqa: E402
from roma_geo import verify_pair, to_public         # noqa: E402

HOUSE = "DollhouseTask_58553"
IMG_A = "08ddb590-a6df-409d-850f-e45f59efa7a0"
IMG_B = "08ddb590-ad9f-4647-8dab-ccfd0b14c8da"
IMG_DIR = os.path.join(datasets_root(), HOUSE, HOUSE)

# ⚠ 參考結果**不可**放在 Output/roma_regions/：那是 batch_pose_from_cache.py 的輸出
# 目錄，不帶 --resume 跑一次全量就會把它覆蓋成當下環境的結果，此後本測試變成與自身
# 比對而永遠通過。放在 tests/ 底下，批次寫不到。
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke_ref_58553.json")


def circ(a, b):
    return abs((a - b + 180) % 360 - 180)


def main():
    pa = os.path.join(IMG_DIR, IMG_A + ".jpg")
    pb = os.path.join(IMG_DIR, IMG_B + ".jpg")
    for p in (pa, pb, REF):
        if not os.path.exists(p):
            print(f"[FAIL] 找不到 {p}")
            return 1
    ref = json.load(open(REF, encoding="utf-8"))

    print("=" * 70)
    print("  Roma_Geo 煙霧測試。經由 verify_pair() 單一入口")
    print("=" * 70)

    # 關快取，強制真的跑 GPU forward（不然只量到讀 npz 的速度）
    res = verify_pair(pa, pb, use_cache=False)
    t = res["timing"]
    print(f"\n[1] verify_pair 完成：匹配 {t['match_s']:.2f}s + 驗證 {t['verify_s']:.3f}s"
          f" = {t['total_s']:.2f}s")

    print(f"\n[2] 參數一致性（必須與正式批次相同，否則產出與 _summary.csv 對不上）")
    same_params = res["params"] == ref["params"]
    print(f"    新：{json.dumps(res['params'], ensure_ascii=False)}")
    print(f"    舊：{json.dumps(ref['params'], ensure_ascii=False)}")
    print(f"    {'[OK] 逐項相同' if same_params else '[FAIL] 參數不同。比較無意義'}")

    print(f"\n[3] 逐欄對照")
    print(f"    {'欄位':<18}{'新環境':>14}{'舊環境(參考)':>16}")

    def show(label, new, old, fmt="{}"):
        f = lambda v: fmt.format(v) if v is not None else "None"      # noqa: E731
        print(f"    {label:<18}{f(new):>14}{f(old):>16}")

    show("schema_version", res.get("schema_version"), ref.get("schema_version"))
    show("conf_used", res.get("conf_used"), ref.get("conf_used"))
    for k in ("raw", "conf", "used", "inlier", "lat_drop"):
        show(f"n.{k}", res["n"].get(k), ref["n"].get(k))
    show("inlier_rate", res.get("inlier_rate"), ref.get("inlier_rate"), "{:.4f}")
    show("n_regions", len(res.get("regions") or []), len(ref.get("regions") or []))

    pn, po = res.get("pose"), ref.get("pose")
    ok_az = (pn is None) == (po is None)
    if pn and po:
        show("az_ab_deg", pn.get("az_ab_deg"), po.get("az_ab_deg"), "{:.2f}")
        show("az_ba_deg", pn.get("az_ba_deg"), po.get("az_ba_deg"), "{:.2f}")
        show("rot_angle_deg", pn.get("rot_angle_deg"), po.get("rot_angle_deg"), "{:.2f}")
        d_ab, d_ba = circ(pn["az_ab_deg"], po["az_ab_deg"]), circ(pn["az_ba_deg"], po["az_ba_deg"])
        print(f"\n    方位角差：az_ab {d_ab:.2f}° · az_ba {d_ba:.2f}°")
        ok_az = d_ab < 5.0 and d_ba < 5.0

    # --- 對外契約 ---
    pub = to_public(res)
    print(f"\n[4] to_public() 對外契約")
    print(f"    欄位：{', '.join(pub.keys())}")
    leaked = [k for k in ("pose", "inliers", "quality", "params", "n", "flags") if k in pub]
    full_kb = len(json.dumps(res, ensure_ascii=False)) / 1024
    pub_kb = len(json.dumps(pub, ensure_ascii=False)) / 1024
    print(f"    體積：完整 {full_kb:.0f} KB → 對外 {pub_kb:.0f} KB")
    print(f"    {'[OK] 內部欄位未外洩' if not leaked else '[FAIL] 外洩：' + str(leaked)}")

    print("\n" + "=" * 70)
    # RoMa 的 sample() 本來就隨機抽點，加上 torch 版本不同，逐點相同不可能；
    # 判準是「同一個幾何結論」：都有位姿、區域數相同、方位角差 < 5°。
    ok = (same_params and not leaked and ok_az
          and (pn is not None) == (po is not None)
          and len(res.get("regions") or []) == len(ref.get("regions") or []))
    print(f"  結論：{'PASS：新舊環境幾何結論一致，對外契約正確' if ok else 'CHECK：見上方'}")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
