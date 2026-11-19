"""
[腳本用途]
Kaggle Notebook 儲存格腳本：把本專案從零建起來並執行驗收。

**整份內容直接貼進一個 Notebook 儲存格執行即可**，不需要先 clone：取得程式碼是
本腳本自己的第一項工作。也可以在已有 repo 的情況下用 `%run` 或當一般腳本執行。

依序呼叫本專案自己的腳本：

  1. environments/setup_env.sh        建立環境（Linux 版建置腳本）
  2. environments/check_env.py        版本與 GPU 驗收
  3. tests/smoke_test.py              端到端功能測試
  4. environments/cross_env_check.py  跨環境一致性檢查（呼叫 roma_geo.verify_pairs）
  完整鏈另含：select_pairs（連通矩陣挑對）→ batch_match → 幾何 → 排序 → 工具資料

資料集會自動偵測：掃 /kaggle/input 底下的 DollhouseTask_* 找出根目錄，
再透過環境變數 ROMA_GEO_DATASETS 交給 config.py 解析，不建立符號連結。

[Kaggle 準備事項]
  - Notebook 開啟 GPU 與網際網路
  - 掛載含 DollhouseTask_* 的資料集
  - 私有 repo 需在 Kaggle Secrets 設定 GITHUB_TOKEN（見下方 CONFIG）

⚠ GPU 限制：需選 T4（sm_75）或 L4（sm_89）。Tesla P100（sm_60）與 V100（sm_70）
   不適用，本腳本在安裝前檢查並中止。
"""

# ==================== CONFIG（貼進 cell 後改這裡）====================
REPO_URL    = "https://github.com/<owner>/<repo>.git"  # 專案 repo
REPO_BRANCH = "main"
SECRET_NAME = "GITHUB_TOKEN"   # 私有 repo：Kaggle Secrets 的名稱；公開 repo 設為 ""
N_PAIRS     = 30               # 一致性檢查的取樣對數（快速正確性關卡）
TAG         = "kaggle"         # 產出檔名的標籤
SKIP_SETUP  = False            # 環境已建好時設 True，省下重裝時間
TAIL_LINES  = 15               # 每個步驟在儲存格顯示的末尾行數；完整輸出一律進 log 檔

# ---- 完整鏈條：矩陣挑對 → 批次匹配 → 幾何驗證 → 排序評估 → 工具資料 ----
FULL_CHAIN   = True            # False 則只跑到一致性檢查為止
BATCH_LIMIT  = 0               # 批次匹配的配對數上限；0 = 全部
WORKERS      = "auto"          # 平行 worker 數；auto = GPU 張數。Kaggle 雙 T4 會用上兩張
TIME_BUDGET_H = 8.0            # 批次匹配的時間預算（小時）。Kaggle 單一工作階段有時限，
                               # 超過即停止並保留已完成的快取，下次重跑會接續。
# ====================================================================

# 環境驗收、挑對、批次匹配失敗即中止。挑對失敗時若繼續，batch_match 會退回
# 舊 _summary.csv 快照（約 7,069 對），與現行連通矩陣無關。
BLOCKING_STEPS = {
    "2 環境驗收",
    "5 從連通矩陣挑配對",
    "6 批次匹配（填滿快取）",
}


def full_chain_steps(py, workers, batch_limit, time_budget_h, regions_dir):
    """完整鏈命令列。先從連通矩陣寫 _pairs.csv，再匹配，避免吃舊 summary 快照。"""
    return [
        ("5 從連通矩陣挑配對",
         "%s environments/select_pairs.py --force" % py),
        ("6 批次匹配（填滿快取）",
         "%s environments/batch_match.py --workers %s --aggregate_scores%s%s"
         % (py, workers,
            " --limit %d" % batch_limit if batch_limit else "",
            " --time_budget_h %g" % time_budget_h if time_budget_h else "")),
        ("7 幾何驗證批次（區域 JSON + _summary.csv）",
         "%s pose_regions/batch_pose_from_cache.py --out %s --resume"
         % (py, regions_dir)),
        ("8 排序評估（_ranking_eval.csv）",
         "%s pose_regions/covis_ranking_eval.py" % py),
        ("9 工具房屋資料（stitch_tool/data）",
         "%s stitch_tool/prepare_house_data.py" % py),
        ("10 工具語意門資料（stitch_tool/semantic）",
         "%s stitch_tool/prepare_semantic_doors.py" % py),
    ]

import glob
import os
import re
import shutil
import subprocess
import sys
import zipfile

WORKING = "/kaggle/working"
REPO_DIRNAME = "Roma_Geo"

# CUDA 13 不支援的架構。
UNSUPPORTED_SM = {(6, 0): "Tesla P100", (7, 0): "Tesla V100"}


def is_kaggle():
    return os.path.isdir("/kaggle/input")


def mask(text):
    """遮蔽 URL 內嵌的權杖，避免寫進 Notebook 輸出。"""
    return re.sub(r"//[^/@:]+(:[^/@]*)?@", "//***@", str(text))


LOG_PATH = os.path.join(WORKING if os.path.isdir(WORKING) else ".", "roma_geo_run.log")
_LOG = None


def log_handle():
    """完整輸出寫進 log 檔，儲存格只留末尾數行與結論。"""
    global _LOG
    if _LOG is None:
        _LOG = open(LOG_PATH, "a", buffering=1, encoding="utf-8", errors="replace")
    return _LOG


def run(cmd, cwd=None, env=None, check=True, tail=None):
    """執行指令：完整輸出進 log 檔，儲存格只回顯末尾 tail 行。"""
    f = log_handle()
    f.write("\n" + "=" * 70 + "\n$ %s\n" % mask(cmd) + "=" * 70 + "\n")
    f.flush()
    print("$ %s" % mask(cmd), flush=True)
    start = f.tell()
    rc = subprocess.run(cmd, shell=True, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                        env={**os.environ, **(env or {})}).returncode
    f.flush()
    n = TAIL_LINES if tail is None else tail
    if n:
        with open(LOG_PATH, encoding="utf-8", errors="replace") as rf:
            rf.seek(start)
            lines = [ln for ln in rf.read().splitlines() if ln.strip()]
        for ln in lines[-n:]:
            # 子行程輸出可能含權杖（git clone 失敗會回顯完整 URL）。
            print("  " + mask(ln), flush=True)
    if rc != 0 and check:
        raise SystemExit("FAILED (rc=%d): %s\n完整輸出見 %s" % (rc, mask(cmd), LOG_PATH))
    return rc


def check_gpu():
    """安裝前確認 GPU 架構是否適用 cu130。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:                                   # noqa: BLE001
        print("[warn] 無法執行 nvidia-smi（%s），跳過 GPU 預檢。" % type(e).__name__)
        return
    if not out:
        print("[warn] nvidia-smi 沒有回報 GPU：Notebook 可能未開啟加速器。")
        return
    print("[gpu] %s" % out.replace("\n", " | "))
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            maj, mnr = (int(x) for x in parts[1].split("."))
        except ValueError:
            continue
        if (maj, mnr) in UNSUPPORTED_SM:
            raise SystemExit(
                "[abort] 偵測到 %s（sm_%d%d），CUDA 13.0 不支援該架構。\n"
                "        請在 Notebook 設定改用 T4（sm_75）或 L4（sm_89）後重跑。"
                % (UNSUPPORTED_SM[(maj, mnr)], maj, mnr))


def find_datasets_root():
    """自動偵測資料集：找 /kaggle/input 底下的 DollhouseTask_*，回傳其父層。"""
    hits = [h for h in glob.glob("/kaggle/input/**/DollhouseTask_*", recursive=True)
            if os.path.isdir(h)]
    if not hits:
        print("[warn] /kaggle/input 底下找不到 DollhouseTask_*；需要影像的步驟會失敗。")
        return None
    root = os.path.dirname(sorted(hits, key=len)[0])
    n_houses = len(glob.glob(os.path.join(root, "DollhouseTask_*")))
    print("[data] 資料集根目錄：%s（%d 間房）" % (root, n_houses))
    if n_houses < 5:
        raise SystemExit(
            "[abort] 只找到 %d 間房，資料集可能掛錯或不完整（預期 57 間）。" % n_houses)
    return root


def resolve_repo_url():
    """組出 clone 用的 URL；私有 repo 從 Kaggle Secrets 取權杖。"""
    if not SECRET_NAME:
        return REPO_URL
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret(SECRET_NAME)
    except Exception as e:                                   # noqa: BLE001
        print("[warn] 讀不到 Kaggle Secret %r（%s）；以未帶權杖的 URL 嘗試 clone。"
              % (SECRET_NAME, type(e).__name__))
        return REPO_URL
    return re.sub(r"^https://", "https://%s@" % token, REPO_URL)


def obtain_repo():
    """取得程式碼。已在 repo 內執行時直接沿用，否則 clone 到 /kaggle/working。"""
    # 當成一般腳本執行時 __file__ 存在，且本檔就在 repo 的 environments/ 底下。
    here = globals().get("__file__")
    if here:
        root = os.path.dirname(os.path.dirname(os.path.abspath(here)))
        if os.path.exists(os.path.join(root, "roma_geo.py")):
            print("[repo] 沿用既有目錄：%s" % root)
            return root

    target = os.path.join(WORKING, REPO_DIRNAME)
    if os.path.isdir(os.path.join(target, ".git")):
        print("[repo] 已存在，更新中：%s" % target)
        run("git fetch --depth 1 origin %s && git reset --hard origin/%s"
            % (REPO_BRANCH, REPO_BRANCH), cwd=target, check=False)
    else:
        if os.path.exists(target):
            shutil.rmtree(target)
        run("git clone --depth 1 --branch %s %s %s"
            % (REPO_BRANCH, resolve_repo_url(), target))
    return target


def find_python(repo):
    for rel in (("venv", "bin", "python"), ("venv", "Scripts", "python.exe")):
        p = os.path.join(repo, *rel)
        if os.path.exists(p):
            return p
    print("[warn] 找不到專案 venv，改用目前的直譯器：%s" % sys.executable)
    return sys.executable


def zip_outputs(repo, dest):
    """把產出打包，Kaggle 才能從 Output 分頁下載。"""
    items = []
    for rel in ("environments", "Output/roma_regions", "pose_regions"):
        d = os.path.join(repo, rel)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith((".csv", ".md")) and not f.startswith("_door"):
                items.append(os.path.join(d, f))
    # roma_scores.npz 在 roma_match/ 底下，不落在上述目錄或 zip_cache 的範圍內。
    scores = os.path.join(repo, "roma_match", "roma_scores.npz")
    if os.path.exists(scores):
        items.append(scores)
    if not items:
        return None
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in items:
            z.write(p, os.path.relpath(p, repo))
    print("[out] %d 個檔案 -> %s" % (len(items), dest))
    return dest


def zip_dir(src_dir, dest, arc_base, pattern="*"):
    """把某個目錄下符合樣式的檔案打包（遞迴），路徑相對於 arc_base。"""
    files = sorted(glob.glob(os.path.join(src_dir, "**", pattern), recursive=True))
    files = [p for p in files if os.path.isfile(p)]
    if not files:
        print("[out] %s 底下沒有 %s 可打包。" % (src_dir, pattern))
        return None
    total = sum(os.path.getsize(p) for p in files)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, os.path.relpath(p, arc_base))
    print("[out] %d 個檔案（%.1f MB）-> %s（壓縮後 %.1f MB）"
          % (len(files), total / 1e6, dest, os.path.getsize(dest) / 1e6))
    return dest


def zip_cache(repo, dest):
    """打包 RoMa 匹配快取，帶回本機即可跳過重算。"""
    cache_dir = os.path.join(repo, "roma_match", "cache")
    npz = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
    if not npz:
        print("[cache] 沒有匹配快取可打包（WRITE_CACHE=False 時為預期）。")
        return None
    total = sum(os.path.getsize(p) for p in npz)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in npz:
            z.write(p, os.path.join("roma_match/cache", os.path.basename(p)))
    print("[cache] %d 個 npz（%.1f MB）-> %s（壓縮後 %.1f MB）"
          % (len(npz), total / 1e6, dest, os.path.getsize(dest) / 1e6))
    return dest


def main():
    on_kaggle = is_kaggle()
    if on_kaggle:
        os.makedirs(WORKING, exist_ok=True)
        # 放寬 hub 權重的下載時限。
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
        check_gpu()

    repo = obtain_repo()

    ds = find_datasets_root() if on_kaggle else None
    if ds:
        os.environ["ROMA_GEO_DATASETS"] = ds

    if on_kaggle and not SKIP_SETUP:
        run("bash %s" % os.path.join(repo, "environments", "setup_env.sh"), cwd=repo)

    py = find_python(repo)
    env = {"PYTHONPATH": repo}
    if os.environ.get("ROMA_GEO_DATASETS"):
        env["ROMA_GEO_DATASETS"] = os.environ["ROMA_GEO_DATASETS"]

    out_dir = WORKING if on_kaggle else os.path.join(repo, "environments")
    regions_dir = os.path.join(repo, "Output", "roma_regions")

    # 前四步為快速關卡。一致性檢查關快取，強制每對重算。
    steps = [
        ("1 路徑設定", "%s config.py" % py),
        ("2 環境驗收", "%s environments/check_env.py" % py),
        ("3 端到端煙霧測試", "%s tests/smoke_test.py" % py),
        ("4 跨環境一致性檢查",
         "%s environments/cross_env_check.py --n %d --tag %s --out_dir %s"
         % (py, N_PAIRS, TAG, out_dir)),
    ]

    # 完整鏈條：矩陣挑對 → 匹配 → 工具可讀資料。批次匹配開快取寫檔，可續跑。
    if FULL_CHAIN:
        steps += full_chain_steps(
            py, WORKERS, BATCH_LIMIT, TIME_BUDGET_H, regions_dir)

    # 其餘步驟容許失敗，以便一次看完全貌。
    results = []
    for title, cmd in steps:
        print("\n" + "#" * 72 + "\n#  " + title + "\n" + "#" * 72)
        rc = run(cmd, cwd=repo, env=env, check=False)
        results.append((title, rc))
        if rc != 0 and title in BLOCKING_STEPS:
            print("\n[abort] %s 失敗（rc=%d），中止後續步驟。\n"
                  "        完整輸出見 %s" % (title, rc, LOG_PATH))
            break

    if on_kaggle:
        zip_outputs(repo, os.path.join(WORKING, "roma_geo_results.zip"))
        if FULL_CHAIN:
            zip_cache(repo, os.path.join(WORKING, "roma_cache.zip"))
            zip_dir(regions_dir, os.path.join(WORKING, "roma_regions.zip"),
                    repo, "*.json")
            zip_dir(os.path.join(repo, "stitch_tool"),
                    os.path.join(WORKING, "stitch_tool_data.zip"), repo, "*.json")

    print("\n" + "=" * 72)
    for title, rc in results:
        print("  %-22s %s" % (title, "PASS" if rc == 0 else "CHECK (rc=%d)" % rc))
    print("  完整輸出              %s" % LOG_PATH)
    print("=" * 72)
    return 0 if all(rc == 0 for _, rc in results) else 1


# 不使用 argparse：儲存格的 sys.argv 不是本腳本的參數。要調整行為請改 CONFIG 區塊。
# 儲存格沒有 __file__，此時不丟 SystemExit，避免留下多餘的例外輸出。
if __name__ == "__main__":
    _rc = main()
    if globals().get("__file__"):
        raise SystemExit(_rc)
