#!/usr/bin/env python3
"""
[腳本用途]
測試模組的單一進入點：一道指令跑完本專案可在當前機器上執行的全部檢查。

分三層，各層的前提條件不同。缺少前提的層級回報「略過」並說明原因，不視為失敗。
本模組會交付給環境各異的對象，多數環境不會具備 GPU 或完整資料集。

    1  單元測試      無前提。純函式、契約、schema、決定性；數秒完成。
    2  交付物檢查    需要磁碟上已有產出。檢查資料本身是否符合契約，缺哪類就略過哪類。
    3  端到端        需要 GPU 與資料集。實際跑一對真實影像，與入庫的參考結果比對。

[用法]
    venv\\Scripts\\python.exe tests\\run_all.py           # 預設：跑得動的全跑
    venv\\Scripts\\python.exe tests\\run_all.py --fast    # 只跑第 1 層
    venv\\Scripts\\python.exe tests\\run_all.py --full    # 併入環境驗收與跨環境一致性；
                                                          # 且缺前提時視為失敗
    venv\\Scripts\\python.exe tests\\run_all.py -k azimuth   # 只跑檔名含該字串的單元測試

離開碼 0 表示沒有任何一項失敗。

[與 environments/ 的關係]
`environments/check_env.py`（這台機器跑不跑得動）與 `cross_env_check.py`（整批結論穩
不穩）驗的是環境，且分別由安裝腳本與 Kaggle 流程直接呼叫，故留在原處；
本腳本以 `--full` 一併執行，使單一進入點仍涵蓋得到。
"""
import argparse
import os
import subprocess
import sys
import time
import unittest

from _support import ROOT, TESTS_DIR  # noqa: F401（匯入時補 sys.path）

OK, FAIL, SKIP = "OK", "FAIL", "SKIP"

PY = sys.executable
SMOKE_HOUSE = "DollhouseTask_58553"
SMOKE_IMGS = ("08ddb590-a6df-409d-850f-e45f59efa7a0",
              "08ddb590-ad9f-4647-8dab-ccfd0b14c8da")


class Section:
    def __init__(self, name, status, detail="", seconds=0.0):
        self.name, self.status, self.detail, self.seconds = name, status, detail, seconds


def rule(title):
    # 子行程直接寫終端而父行程的輸出是緩衝的，不主動 flush 會使兩者順序錯亂。
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    sys.stdout.flush()


# ---------- 1 單元測試 ----------
def run_unit_tests(pattern=None, verbose=False):
    rule("1 單元測試（無前提）")
    loader = unittest.TestLoader()
    suite = loader.discover(TESTS_DIR, pattern="test_*.py", top_level_dir=TESTS_DIR)
    if pattern:
        suite = filter_suite(suite, pattern)
    t0 = time.perf_counter()
    # warnings=None：不重設警告過濾器，保留 _support 對 ResourceWarning 的具名關閉。
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1, warnings=None,
                                     stream=sys.stdout)
    res = runner.run(suite)
    dt = time.perf_counter() - t0
    n_bad = len(res.failures) + len(res.errors)
    detail = "%d 項" % res.testsRun
    if n_bad:
        detail += "，失敗 %d" % n_bad
    return Section("單元測試", FAIL if n_bad else OK, detail, dt)


def filter_suite(suite, pattern):
    out = unittest.TestSuite()
    for t in suite:
        if isinstance(t, unittest.TestSuite):
            sub = filter_suite(t, pattern)
            if sub.countTestCases():
                out.addTest(sub)
        elif pattern.lower() in t.id().lower():
            out.addTest(t)
    return out


# ---------- 2 交付物檢查 ----------
def run_deliverables(scan_all=False, datasets_dir=None):
    rule("2 交付物檢查（需要磁碟上已有產出）")
    import check_deliverables as cd
    t0 = time.perf_counter()
    results = cd.run_checks(limit=0 if scan_all else 200, datasets_dir=datasets_dir)
    n_fail = cd.report(results)
    dt = time.perf_counter() - t0
    n_skip = sum(1 for r in results if r.status == cd.SKIP)
    detail = "%d 類通過" % (len(results) - n_fail - n_skip)
    if n_fail:
        detail += "，%d 類不符" % n_fail
    if n_skip:
        detail += "，%d 類無產出" % n_skip
    return Section("交付物檢查", FAIL if n_fail else OK, detail, dt)


# ---------- 3 端到端 ----------
def smoke_prerequisites():
    """回傳 (可執行, 說明)。先檢查資料集再檢查 GPU，避免無謂地載入 torch。"""
    from config import datasets_root
    img_dir = os.path.join(datasets_root(), SMOKE_HOUSE, SMOKE_HOUSE)
    for u in SMOKE_IMGS:
        if not os.path.exists(os.path.join(img_dir, u + ".jpg")):
            return False, "找不到測試影像（%s）" % SMOKE_HOUSE
    try:
        import torch
    except Exception:
        return False, "未安裝 torch"
    if not torch.cuda.is_available():
        return False, "無可用的 CUDA 裝置"
    return True, ""


def run_smoke(strict=False):
    rule("3 端到端煙霧測試（需要 GPU 與資料集）")
    ok, why = smoke_prerequisites()
    if not ok:
        print("略過：%s" % why)
        return Section("端到端", FAIL if strict else SKIP, why)
    t0 = time.perf_counter()
    sys.stdout.flush()
    rc = subprocess.call([PY, os.path.join(TESTS_DIR, "smoke_test.py")], cwd=ROOT)
    return Section("端到端", OK if rc == 0 else FAIL, "", time.perf_counter() - t0)


# ---------- 環境層（--full） ----------
def run_env_script(name, rel_path, args=(), strict=False):
    rule("%s（%s）" % (name, rel_path))
    path = os.path.join(ROOT, rel_path)
    if not os.path.isfile(path):
        print("略過：找不到 %s" % rel_path)
        return Section(name, FAIL if strict else SKIP, "腳本不存在")
    t0 = time.perf_counter()
    sys.stdout.flush()
    rc = subprocess.call([PY, path] + list(args), cwd=ROOT)
    return Section(name, OK if rc == 0 else FAIL, "", time.perf_counter() - t0)


# ---------- 匯總 ----------
def summarise(sections, total_s):
    rule("總結")
    for s in sections:
        t = ("%.1fs" % s.seconds) if s.seconds else "-"
        print("  [%-4s] %-12s %-6s %s" % (s.status, s.name, t, s.detail))
    n_fail = sum(1 for s in sections if s.status == FAIL)
    n_skip = sum(1 for s in sections if s.status == SKIP)
    print("\n合計 %.1f 秒 · %d 層通過 · %d 層失敗 · %d 層略過"
          % (total_s, len(sections) - n_fail - n_skip, n_fail, n_skip))
    if n_fail:
        print("\n有項目未通過，詳見上方各層輸出。")
    elif n_skip:
        print("\n可執行的項目全數通過；略過的層級缺少前提條件（GPU、資料集或既有產出）。")
    else:
        print("\n全數通過。")
    return n_fail


def main(argv=None):
    ap = argparse.ArgumentParser(description="Roma_Geo 測試總入口")
    ap.add_argument("--fast", action="store_true", help="只跑單元測試")
    ap.add_argument("--full", action="store_true",
                    help="併入環境驗收與跨環境一致性；缺前提時視為失敗")
    ap.add_argument("--deliverables-all", action="store_true",
                    help="交付物檢查改為全量（預設每類抽驗 200 份）")
    ap.add_argument("-k", dest="pattern", default=None, help="只跑名稱含此字串的單元測試")
    ap.add_argument("-v", "--verbose", action="store_true", help="逐項列出單元測試")
    ap.add_argument("--datasets", default=None, help="資料集根目錄（交付物檢查判定受限來源）")
    a = ap.parse_args(argv)

    t_start = time.perf_counter()
    sections = [run_unit_tests(a.pattern, a.verbose)]
    if not a.fast:
        sections.append(run_deliverables(a.deliverables_all, datasets_dir=a.datasets))
        sections.append(run_smoke(strict=a.full))
        if a.full:
            sections.append(run_env_script("環境驗收", os.path.join("environments", "check_env.py"),
                                           strict=True))
            sections.append(run_env_script("跨環境一致性",
                                           os.path.join("environments", "cross_env_check.py"),
                                           args=["--n", "30", "--tag", "local"], strict=True))
    return 1 if summarise(sections, time.perf_counter() - t_start) else 0


if __name__ == "__main__":
    raise SystemExit(main())
