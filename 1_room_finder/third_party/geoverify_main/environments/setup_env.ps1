<#
.SYNOPSIS
    Roma_Geo 環境建置
.DESCRIPTION
    目標配置：Python 3.11 · torch 2.10.0+cu130 · numpy 1.26.4 · opencv 4.11 ·
              scikit-learn 1.8.0 · pillow 12.0.0

    支援 Ada (sm_89) 與 Blackwell (sm_120) 世代 GPU。

    安裝順序：torch -> 釘版核心套件 -> romav2 -> 重新釘版。
    romav2 為 git 源碼安裝，可能升級 numpy 等相依，故其後再釘一次並驗證。

.EXAMPLE
    cd Roma_Geo\environments
    .\setup_env.ps1
#>

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$VENV = "..\venv"
$TORCH_INDEX = "https://download.pytorch.org/whl/cu130"

function Write-Step {
    param([string]$msg)
    Write-Host "`n>>> $msg" -ForegroundColor Yellow
}

# $ErrorActionPreference = "Stop" 對原生 exe 不生效，uv 失敗只設 $LASTEXITCODE，
# 因此每個 uv 呼叫後都要顯式檢查。
function Assert-LastExit {
    param([string]$what)
    if ($LASTEXITCODE -ne 0) {
        throw "$what 失敗 (exit $LASTEXITCODE) —— 中止，不要在殘缺環境上繼續安裝。"
    }
}

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "  Roma_Geo 環境建置（py3.11 / torch 2.10.0+cu130）" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

# --- 1. uv ---
Write-Step "1/6 檢查 uv"
# 安裝器把 uv 放進 ~\.local\bin，該路徑要重開 shell 才會進 PATH，故先補上。
$env:Path = "$HOME\.local\bin;" + $env:Path
if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Host "未偵測到 uv，自動安裝..." -ForegroundColor DarkGray
    irm https://astral.sh/uv/install.ps1 | iex
    $env:Path = "$HOME\.local\bin;" + $env:Path
} else {
    Write-Host "uv 已安裝。" -ForegroundColor Green
}

# --- 2. Python 3.11 ---
# uv 0.12.1 (Windows) 的 managed Python minor 版連結會誤報
# "Missing expected target directory"，故繞過該連結，直接以解壓後的
# python.exe 絕對路徑餵給 uv venv。
Write-Step "2/6 尋找 Python 3.11 直譯器"
# 此步容許失敗（僅用於在全新機器上觸發下載），故不做 Assert-LastExit。
# 不可加 2>&1：PowerShell 5.1 會把原生 exe 的 stderr 包成 NativeCommandError，
# 在 Stop 偏好下成為終止性錯誤。
uv python install 3.11

$py311 = $null
$candidates = Get-ChildItem "$env:APPDATA\uv\python" -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^cpython-3\.11\.\d+-' } |
    Sort-Object Name -Descending
foreach ($c in $candidates) {
    $exe = Join-Path $c.FullName "python.exe"
    if (Test-Path $exe) { $py311 = $exe; break }
}
if (-not $py311) {
    # 退路：系統上已有的 3.11
    $sys = Get-Command "python3.11" -ErrorAction SilentlyContinue
    if ($sys) { $py311 = $sys.Source }
}
if (-not $py311) {
    throw "找不到 Python 3.11 直譯器。請手動安裝 Python 3.11 後重跑。"
}
Write-Host "使用直譯器：$py311" -ForegroundColor Green

# --- 3. venv ---
# --clear：可重複執行，重跑即乾淨重建。
Write-Step "3/6 建立 venv ($VENV, Python 3.11)"
uv venv $VENV --python $py311 --clear
Assert-LastExit "uv venv"

# --- 4. torch (cu130) ---
# torch 先裝，後續套件遷就它的版本。
Write-Step "4/6 安裝 torch 2.10.0 + cu130"
uv pip install "torch==2.10.0" torchvision --index-url $TORCH_INDEX --python $VENV
Assert-LastExit "torch 安裝"

# --- 5. 釘版核心套件 ---
Write-Step "5/6 安裝核心套件（釘版）"
uv pip install -r requirements-core.txt --python $VENV
Assert-LastExit "核心套件安裝"

# --- 6. romav2 ---
# 不安裝 fused-local-corr。
Write-Step "6/6 安裝 romav2（GitHub 源碼）"
uv pip install "git+https://github.com/Parskatt/romav2.git" --python $VENV
Assert-LastExit "romav2 安裝"

Write-Step "重新釘版（romav2 可能升級了 numpy 等相依）"
uv pip install -r requirements-core.txt --python $VENV
Assert-LastExit "重新釘版"

Write-Host "`n=========================================================" -ForegroundColor Cyan
Write-Host "  安裝完成 —— 執行驗收" -ForegroundColor Green
Write-Host "=========================================================" -ForegroundColor Cyan
& "$VENV\Scripts\python.exe" check_env.py
