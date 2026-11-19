"""python -m roma_geo a.jpg b.jpg [--public] [--out res.json]"""
import json

from . import DEFAULTS, MATCH_MODE_DEFAULT, to_public, verify_pair


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Roma_Geo 單對幾何驗證")
    ap.add_argument("img_a")
    ap.add_argument("img_b")
    ap.add_argument("--mode", default="direct", choices=["direct", "core"])
    ap.add_argument("--conf_min", type=float, default=DEFAULTS["conf_min"])
    ap.add_argument("--no_cache", action="store_true", help="不讀寫 RoMa 匹配快取")
    ap.add_argument("--public", action="store_true", help="只輸出對外契約欄位")
    ap.add_argument("--out", help="寫入 JSON 檔（預設印到 stdout）")
    a = ap.parse_args(argv)

    res = verify_pair(a.img_a, a.img_b, mode=a.mode, conf_min=a.conf_min,
                      use_cache=not a.no_cache)
    payload = to_public(res) if a.public else res
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
        print("已寫入 %s（%.1f KB）" % (a.out, len(text) / 1024))
        print("  %s · %d 區域 · az_ab %s · %.2fs"
              % ("有位姿" if res.get("pose") else "無位姿",
                 len(res.get("regions") or []),
                 (res.get("pose") or {}).get("az_ab_deg"), res["timing"]["total_s"]))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
