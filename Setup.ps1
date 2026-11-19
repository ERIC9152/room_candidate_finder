# One-time setup (needs internet and an NVIDIA GPU). All environments are created inside this package:
#   main env       : 1_room_finder\third_party\geoverify_main\venv   (GeoVerify scan; also CoVisPose inference + CPU steps of 2_pose_floorplan)
#   review app     : 1_room_finder\.venv                               (Tkinter + Pillow)
#   SAM3 env       : 1_room_finder\third_party\sam3-pano-seg\.runtime (bundled; only re-pointed to this folder)
#   HorizonNet env : 2_pose_floorplan\third_party\HorizonNet\venv     (HorizonNet inference)
# An existing environment is reused instead when RCF_MAIN_PYTHON / RCF_HN_PYTHON / RCF_REVIEW_PYTHON point at its python.exe.
# The package can be copied with its environments: an environment that does not start on this computer (a Windows venv
# stores absolute paths) is detected and rebuilt here. Run this once after copying or moving the folder.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File Setup.ps1
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\tools\_env.ps1"
Set-Location $PkgRoot
if (-not ($PkgRoot -match '^[\x00-\x7F]*$')) { Write-Host "WARNING: '$PkgRoot' contains non-ASCII characters; HorizonNet may fail. Move the folder (e.g. C:\room_candidate_finder)." -ForegroundColor Yellow }
function Remove-BrokenEnv([string]$Dir, [string]$Label) {
    # Called only after the environment's python.exe failed to start or to import its packages (see Test-Py in tools\_env.ps1).
    if (Test-Path $Dir) {
        Write-Host "== The $Label environment in this folder does not run on this computer (copied from another computer, or incomplete); rebuilding it" -ForegroundColor Yellow
        Remove-Item -Recurse -Force $Dir
    }
}

# 1) main environment: GeoVerify's own installer (uv; Python 3.11, torch 2.10 cu130, RoMa v2) creates
#    1_room_finder\third_party\geoverify_main\venv; then shapely/matplotlib/scipy for the CPU steps
$Py = Get-MainPy
if (-not $Py) {
    if (Test-Path "$FinderRoot\third_party\geoverify_main\venv") {
        # no need to delete it: the GeoVerify installer recreates the venv from scratch (uv venv --clear)
        Write-Host '== The main environment in this folder does not run on this computer (copied from another computer, or incomplete); rebuilding it' -ForegroundColor Yellow
    }
    Write-Host '== Creating the main environment with the GeoVerify installer (several minutes)' -ForegroundColor Cyan
    # GeoVerify's installer is UTF-8 without a BOM and contains Chinese text; Windows PowerShell 5.1 would read it in the
    # system code page (e.g. Big5) and fail to parse it. Run a temporary copy saved with a BOM instead (the original is not
    # changed); its "Set-Location $PSScriptRoot" is pointed at the original folder so the relative paths stay the same.
    $envDir = "$FinderRoot\third_party\geoverify_main\environments"
    $src = [IO.File]::ReadAllText("$envDir\setup_env.ps1", [Text.Encoding]::UTF8)
    $pin = 'Set-Location $PSScriptRoot'
    if (-not $src.Contains($pin)) { throw "Unexpected GeoVerify installer: '$pin' not found in $envDir\setup_env.ps1" }
    $src = $src.Replace($pin, "Set-Location '" + $envDir.Replace("'", "''") + "'")
    $tmp = Join-Path $env:TEMP "rcf_geoverify_setup_env_$PID.ps1"
    [IO.File]::WriteAllText($tmp, $src, (New-Object Text.UTF8Encoding $true))
    try { & powershell -NoProfile -ExecutionPolicy Bypass -File $tmp } finally { Remove-Item $tmp -ErrorAction SilentlyContinue }
    $Py = Get-MainPy
    if (-not $Py) { throw 'Main environment was not created; see the installer output above.' }
}
Write-Host "main env: $Py"
# GeoVerify's installer creates the venv with uv, which does not include pip: add it with ensurepip
& $Py -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pip') else 1)"
if ($LASTEXITCODE -ne 0) {
    & $Py -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) {
        $env:Path = "$HOME\.local\bin;" + $env:Path          # where the GeoVerify installer puts uv
        & uv pip install pip --python $Py
        if ($LASTEXITCODE -ne 0) { throw 'Could not add pip to the main environment (ensurepip and uv both failed).' }
    }
}
& $Py -m pip install -r "$PoseRoot\requirements_main_extra.txt" -c "$PoseRoot\constraints_main.txt"
if ($LASTEXITCODE -ne 0) { throw 'pip install failed (main env)' }

# 1b) RoMa v2 weights and the DINOv3 code it loads through torch.hub: keep them inside the package
#     (1_room_finder\models\torch_home, see finder\config.py) so a copy of the package can scan without downloading again
$RomaHome = "$FinderRoot\models\torch_home"
$RomaCkpt = "$RomaHome\hub\checkpoints\romav2.0.1.pt"
if (-not (Test-Path $RomaCkpt)) {
    $DefHub = Join-Path $HOME '.cache\torch\hub'
    if (Test-Path "$DefHub\checkpoints\romav2.0.1.pt") {
        Write-Host '== Copying the RoMa v2 files already on this computer into the package' -ForegroundColor Cyan
        New-Item -ItemType Directory -Force -Path "$RomaHome\hub\checkpoints" | Out-Null
        Copy-Item "$DefHub\checkpoints\romav2.0.1.pt" $RomaCkpt
        Get-ChildItem $DefHub -Directory -Filter 'facebookresearch_dinov3*' | ForEach-Object { Copy-Item $_.FullName "$RomaHome\hub" -Recurse -Force }
        if (Test-Path "$DefHub\trusted_list") { Copy-Item "$DefHub\trusted_list" "$RomaHome\hub" -Force }
    }
}
Write-Host '== Checking the RoMa v2 weights (downloaded now if they are not in the package yet)' -ForegroundColor Cyan
$env:TORCH_HOME = $RomaHome
& $Py -c "from romav2 import RoMaV2; RoMaV2(); print('  RoMa v2 weights ready')"
if ($LASTEXITCODE -ne 0) { Write-Host 'WARNING: RoMa v2 could not be loaded now; the first scan (Import_Dataset.ps1) will try again and needs internet for the download.' -ForegroundColor Yellow }

# 2) HorizonNet environment (Python 3.9, torch 1.12 cu113)
$HnPy = Get-HnPy
if (-not $HnPy) {
    $HnVenv = "$PoseRoot\third_party\HorizonNet\venv"
    Remove-BrokenEnv $HnVenv 'HorizonNet'
    Write-Host '== Creating the HorizonNet environment (Python 3.9)' -ForegroundColor Cyan
    $made = $false
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.9 -m venv $HnVenv
        $made = ($LASTEXITCODE -eq 0)
    }
    if (-not $made) {
        # No Python 3.9 on this computer: let uv (installed by the GeoVerify installer in step 1) download one.
        $env:Path = "$HOME\.local\bin;" + $env:Path
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            Write-Host '   Python 3.9 not found (py -3.9); downloading it with uv'
            if (Test-Path $HnVenv) { Remove-Item -Recurse -Force $HnVenv }
            & uv venv $HnVenv --python 3.9 --seed
            $made = ($LASTEXITCODE -eq 0)
        }
    }
    if (-not $made) { throw 'Python 3.9 not found and uv could not provide it. Install Python 3.9.x from python.org, then run Setup.ps1 again.' }
    $HnPy = "$HnVenv\Scripts\python.exe"
    & $HnPy -m pip install --upgrade pip
    & $HnPy -m pip install -r "$PoseRoot\requirements_horizonnet.txt"
    if ($LASTEXITCODE -ne 0) { throw 'pip install failed (HorizonNet env)' }
}
Write-Host "HorizonNet env: $HnPy"
# HorizonNet builds an ImageNet ResNet-50 before loading its checkpoint; keep that file inside the package.
$env:TORCH_HOME = "$PoseRoot\models\torch_home"
& $HnPy -c "import torchvision; torchvision.models.resnet50(pretrained=True); print('  ResNet-50 ImageNet weights ready')"

# 3) review app: any Python 3 with Tkinter (python.org installer, "tcl/tk and IDLE" kept) + Pillow, in 1_room_finder\.venv
if (-not (Get-ReviewPy)) {
    Remove-BrokenEnv "$FinderRoot\.venv" 'review'
    # use the first Python that really runs and has Tkinter (py -3 = the launcher's default version)
    $base = $null
    $ErrorActionPreference = 'Continue'
    foreach ($c in @('py -3', 'python', 'py -3.12', 'py -3.11', 'py -3.10', 'py -3.9')) {
        if (-not (Get-Command $c.Split(' ')[0] -ErrorAction SilentlyContinue)) { continue }
        $global:LASTEXITCODE = 1
        try { Invoke-Expression "$c -c `"import tkinter`" 2>`$null" | Out-Null } catch { continue }
        if ($LASTEXITCODE -eq 0) { $base = $c; break }
    }
    $ErrorActionPreference = 'Stop'
    if (-not $base) { throw 'No Python 3 with Tkinter found. Install it from python.org (keep "tcl/tk and IDLE" checked), then run Setup.ps1 again.' }
    Write-Host "== Creating the review environment with $base" -ForegroundColor Cyan
    Invoke-Expression "$base -m venv `"$FinderRoot\.venv`""
    $Rv = "$FinderRoot\.venv\Scripts\python.exe"
    if (-not (Test-Path $Rv)) { throw "Could not create the review environment with $base." }
    & $Rv -m pip install --upgrade pip
    & $Rv -m pip install "pillow>=9"
    & $Rv -c "import tkinter, PIL; print('  review environment OK')"
    if ($LASTEXITCODE -ne 0) { throw 'Review environment check failed (Tkinter or Pillow missing).' }
}
# 4) SAM3 (bundled, ready-made environment in 1_room_finder\third_party\sam3-pano-seg\.runtime): re-point its venv at this folder
$Sam3 = Get-Sam3Root
if ($Sam3) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$FinderRoot\tools\fix_sam3_env.ps1" -Sam3Root $Sam3
    & (Join-Path $Sam3 '.runtime\sam3-env\Scripts\python.exe') -c "import torch, sam3, sam3_pano; print('  SAM3 env: torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
    $ok3 = $LASTEXITCODE -eq 0
} else {
    Write-Host '  sam3-pano-seg not found: the door-aware wedge is skipped (plain wedge is used); see README.md section 3.3' -ForegroundColor Yellow
    $ok3 = $true
}

# 5) checks
Write-Host "`n== Checking environments" -ForegroundColor Cyan
Push-Location $PoseRoot
& $Py -B -m pipeline.envcheck
$ok1 = $LASTEXITCODE -eq 0
Pop-Location
& $HnPy -c "import torch, cv2, pylsd, scipy, sklearn, shapely, PIL; print('  HorizonNet env: torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
$ok2 = $LASTEXITCODE -eq 0
if ($ok1 -and $ok2 -and $ok3) { Write-Host "`nSetup OK. Next: powershell -ExecutionPolicy Bypass -File 2_pose_floorplan\Verify.ps1" -ForegroundColor Green }
else { Write-Host "`nSetup incomplete (see above)." -ForegroundColor Red; exit 1 }
