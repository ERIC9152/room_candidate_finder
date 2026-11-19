"""
[腳本用途]
環境驗收：確認這台機器跑不跑得動。做三個實跑驗證（CUDA kernel、romav2 匯入、
幾何核心 selftest），並列出本機版本與參考配置的對照供除錯參考。

[用法]
  ..\venv\Scripts\python.exe environments\check_env.py

**判準是三個實跑驗證，不是版本相等。** 本模組會交付給環境各異的對象使用，
版本相等既不充分（同版本仍可能因其他原因跑不動），也不必要（不同版本多半正常）。
版本對照表在實跑失敗時是第一線線索，故照印，但不影響結論。

arch 一項顯示各自 GPU 的 compute capability，各機器本來就不同。
"""
import platform
import subprocess
import sys
from pathlib import Path

# 參考配置：開發與驗證時所用的版本。差異只作提示，不影響判準。
TARGET = {
    "python": "3.11.0",
    "torch": "2.10.0+cu130",
    "cuda": "13.0",
    "numpy": "1.26.4",
    "cv2": "4.11.0",
    "sklearn": "1.8.0",
    "PIL": "12.0.0",
}

OK, DIFF = "OK  ", "DIFF"       # DIFF = 與參考配置不同，僅提示，不影響判準


def _ver(mod_name, attr="__version__"):
    try:
        mod = __import__(mod_name)
        return str(getattr(mod, attr))
    except Exception as e:  # noqa: BLE001 - 缺套件也要照常印出來
        return f"<{type(e).__name__}>"


def row(label, actual, target, minor_only=False):
    """列出本機版本與參考配置的對照。回傳是否相同，僅供彙總顯示。

    minor_only：只比 major.minor（Python patch 版差異不影響數值結果）。
    """
    if minor_only:
        hit = actual.split(".")[:2] == target.split(".")[:2]
    else:
        hit = actual == target
    print(f"  [{(OK if hit else DIFF):8}] {label:<12} {actual:<20} (參考 {target})")
    return hit


def main():
    print("=" * 68)
    print("  Roma_Geo 環境驗收")
    print("=" * 68)

    print(f"\nOS          : {platform.system()} {platform.release()} {platform.machine()}")

    print("\n--- 版本對照（僅供參考，不影響判準）---")
    same_ver = True
    same_ver &= row("python", platform.python_version(), TARGET["python"], minor_only=True)

    try:
        import torch
    except ImportError:
        print("  [FATAL   ] torch 未安裝。先跑 setup_env.ps1")
        return 1

    same_ver &= row("torch", torch.__version__, TARGET["torch"])
    same_ver &= row("cuda build", str(torch.version.cuda), TARGET["cuda"])
    same_ver &= row("numpy", _ver("numpy"), TARGET["numpy"])
    same_ver &= row("opencv", _ver("cv2"), TARGET["cv2"])
    same_ver &= row("sklearn", _ver("sklearn"), TARGET["sklearn"])
    same_ver &= row("pillow", _ver("PIL"), TARGET["PIL"])

    print("\n--- GPU ---")
    if not torch.cuda.is_available():
        print("  [FATAL   ] torch.cuda.is_available() = False")
        return 1
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    sm = f"sm_{cap[0]}{cap[1]}"
    arch_list = torch.cuda.get_arch_list()
    print(f"  GPU         : {name}  capability {cap} -> {sm}")
    print(f"  arch_list   : {arch_list}")

    # wheel 未涵蓋本機 GPU 的 arch 時，is_available() 仍回 True，但一發 kernel
    # 就出 no kernel image。
    #
    # 不做字面比對：CUDA 的二進位相容規則是「為 sm_X.y 編的 cubin 可在 sm_X.z
    # （z >= y）的裝置上執行」，同 major 且 minor 不高於裝置即可。
    covering = []
    for a in arch_list:
        if not a.startswith("sm_"):
            continue
        d = a[3:]
        if not d.isdigit():
            continue
        maj, mnr = int(d[:-1]), int(d[-1])   # sm_89 -> (8,9) · sm_120 -> (12,0)
        if maj == cap[0] and mnr <= cap[1]:
            covering.append(a)

    if sm in arch_list:
        print(f"  [{OK:8}] wheel 直接含 {sm}")
    elif covering:
        print(f"  [{OK:8}] wheel 未直接含 {sm}，但 {covering[-1]} 的 cubin 向上相容（同 major，minor 較低）")
    else:
        # 只作預告：真正的判準是下面那次 CUDA kernel 實跑，涵蓋不到的話它會失敗。
        print(f"  [WARN    ] wheel 不涵蓋 {sm}：預期下方 CUDA kernel 實跑會失敗")

    try:
        drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        print(f"  driver      : {drv}")
    except Exception as e:  # noqa: BLE001
        print(f"  driver      : <查詢失敗 {type(e).__name__}>")

    print("\n--- 實跑驗證 ---")
    passed = 0

    # 1) 真的發一次 CUDA kernel（is_available() 為 True 但 kernel 炸掉是本次對齊的核心風險）
    try:
        a = torch.randn(512, 512, device="cuda")
        val = float((a @ a).sum().item())
        torch.cuda.synchronize()
        print(f"  [PASS    ] CUDA kernel 實跑（512x512 matmul, sum={val:.1f}）")
        passed += 1
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL    ] CUDA kernel: {type(e).__name__}: {e}")

    # 2) romav2 匯入
    try:
        import romav2  # noqa: F401
        print("  [PASS    ] romav2 匯入")
        passed += 1
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL    ] romav2 匯入: {type(e).__name__}: {e}")

    # 3) 幾何核心 selftest（純 numpy，合成資料，不吃 GPU）
    root = Path(__file__).resolve().parent.parent
    selftest = root / "pose_regions" / "estimate_relative_pose.py"
    if selftest.exists():
        r = subprocess.run(
            [sys.executable, str(selftest), "--selftest"],
            capture_output=True, text=True, cwd=str(root), timeout=300,
        )
        tail = (r.stdout or r.stderr).strip().splitlines()
        if r.returncode == 0:
            print("  [PASS    ] pose_regions --selftest")
            for line in tail[-3:]:
                print(f"             {line}")
            passed += 1
        else:
            print(f"  [FAIL    ] pose_regions --selftest (exit {r.returncode})")
            for line in tail[-8:]:
                print(f"             {line}")
    else:
        print(f"  [SKIP    ] pose_regions 尚未搬入 ({selftest})")

    ok = passed >= 3
    print("\n" + "=" * 68)
    print(f"  結果：實跑 {passed}/3 · {'通過' if ok else '有項目未通過（見上）'}")
    if not same_ver:
        print("  版本與參考配置有落差（見上表）。三項實跑都過就代表這個環境可用;"
              "落差僅供除錯時參考。")
    print("=" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
