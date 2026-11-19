#!/usr/bin/env bash
# =============================================================================
# Roma_Geo 環境建置（Linux / Kaggle）
#
# 目標配置：Python 3.11 · torch 2.10.0+cu130 · numpy 1.26.4 · opencv 4.11 ·
#           scikit-learn 1.8.0 · pillow 12.0.0
#
# 與 setup_env.ps1 為同一份配置的兩個平台版本，修改其一時另一個要同步。
#
# 支援 Ada (sm_89) 與 Blackwell (sm_120) 世代 GPU。
# CUDA 13 不支援 sm_60/sm_70（Tesla P100、V100），這些卡不適用本配置。
#
# 用法：
#   bash environments/setup_env.sh                # 建立 ../venv 並驗收
#   VENV=/tmp/venv bash environments/setup_env.sh # 指定 venv 位置
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="${VENV:-$PROJECT_ROOT/venv}"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"

step() { printf '\n>>> %s\n' "$1"; }

# --- 1. uv ---
step "1/6 檢查 uv"
# 安裝器可能放在 ~/.local/bin 或 ~/.cargo/bin，兩者都補進 PATH。
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "未偵測到 uv，自動安裝..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { echo "uv 安裝後仍不在 PATH，中止。"; exit 1; }
uv --version

# --- 2. Python 3.11 ---
step "2/6 準備 Python 3.11"
uv python install 3.11

# --- 3. venv ---
# --clear：可重複執行，重跑即乾淨重建。
step "3/6 建立 venv ($VENV)"
uv venv "$VENV" --python 3.11 --clear

PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "venv 建立失敗：找不到 $PY"; exit 1; }

# --- 4. torch (cu130) ---
# torch 先裝，後續套件遷就它的版本。
step "4/6 安裝 torch 2.10.0 + cu130"
uv pip install "torch==2.10.0" torchvision --index-url "$TORCH_INDEX" --python "$PY"

# --- 5. 釘版核心套件 ---
step "5/6 安裝核心套件（釘版）"
uv pip install -r "$SCRIPT_DIR/requirements-core.txt" --python "$PY"

# --- 6. romav2 ---
# 不安裝 fused-local-corr。
step "6/6 安裝 romav2（GitHub 源碼）"
uv pip install "git+https://github.com/Parskatt/romav2.git" --python "$PY"

step "重新釘版（romav2 可能升級了 numpy 等相依）"
uv pip install -r "$SCRIPT_DIR/requirements-core.txt" --python "$PY"

step "執行驗收"
"$PY" "$SCRIPT_DIR/check_env.py"
