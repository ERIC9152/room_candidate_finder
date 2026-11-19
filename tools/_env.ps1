# Shared helpers, dot-sourced by Setup.ps1 and the scripts in 1_room_finder\ and 2_pose_floorplan\.
# Every environment lives inside this folder and is created by Setup.ps1. To reuse an environment that
# already exists somewhere else, point one of these variables at its python.exe before running a script:
#   RCF_MAIN_PYTHON    main env: GeoVerify scan + CoVisPose inference + CPU steps (Python 3.11, torch 2.10 cu130)
#   RCF_HN_PYTHON      HorizonNet env (Python 3.9, torch 1.12 cu113)
#   RCF_REVIEW_PYTHON  review app (any Python 3.9+ with Tkinter and Pillow)
#   RCF_SAM3_ROOT      sam3-pano-seg project (door-aware wedge); default 1_room_finder\third_party\sam3-pano-seg
$PkgRoot    = Split-Path -Parent $PSScriptRoot          # package root
$FinderRoot = Join-Path $PkgRoot '1_room_finder'       # same-room filtering (scan, door openings, review)
$PoseRoot   = Join-Path $PkgRoot '2_pose_floorplan'    # relative poses, floor plans, viewer
function Find-First($cands) { foreach ($c in $cands) { if ($c -and (Test-Path $c)) { return (Resolve-Path $c).Path } }; return $null }
function Test-Py([string]$Exe, [string]$Code = 'import sys') {
    # True only when this python.exe really starts on this computer and can import what it is needed for.
    # A Windows venv stores the absolute path of its base Python: after the package is copied to another computer
    # (or the base Python is removed) python.exe is still there but cannot start. Such an environment counts as
    # missing, and Setup.ps1 rebuilds it.
    if (-not $Exe) { return $false }
    if (-not (Test-Path $Exe)) { return $false }
    $ErrorActionPreference = 'Continue'      # local to this function: text on stderr must not abort
    $ok = $false
    try {
        & $Exe -c $Code 2>&1 | Out-Null
        $ok = ($LASTEXITCODE -eq 0)
    } catch { $ok = $false }
    return $ok
}
function Find-Py($cands, [string]$Code) { foreach ($c in $cands) { if (Test-Py $c $Code) { return (Resolve-Path $c).Path } }; return $null }
function Get-MainPy   { Find-Py @($env:RCF_MAIN_PYTHON,   "$FinderRoot\third_party\geoverify_main\venv\Scripts\python.exe") 'import torch, cv2' }
function Get-HnPy     { Find-Py @($env:RCF_HN_PYTHON,     "$PoseRoot\third_party\HorizonNet\venv\Scripts\python.exe") 'import torch, cv2, pylsd' }
function Get-ReviewPy { Find-Py @($env:RCF_REVIEW_PYTHON, "$FinderRoot\.venv\Scripts\python.exe") 'import tkinter, PIL' }
function Get-Sam3Root([string]$Given = '') {
    # A sam3-pano-seg folder counts only if its runtime and checkpoint are there.
    foreach ($c in @($Given, $env:RCF_SAM3_ROOT, "$FinderRoot\third_party\sam3-pano-seg")) {
        if ($c -and (Test-Path (Join-Path $c '.runtime\sam3-env\Scripts\python.exe')) -and (Test-Path (Join-Path $c 'checkpoints\sam3.pt'))) {
            return (Resolve-Path $c).Path
        }
    }
    return $null
}
function Assert-GpuFree {
    # Only one program should use the GPU at a time.
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        Write-Host 'nvidia-smi not found: cannot check whether the GPU is in use (is the NVIDIA driver installed?).' -ForegroundColor Yellow
        return
    }
    $busy = & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>$null
    # On Windows (WDDM driver) this list also holds every desktop program that draws on the screen (explorer.exe, the
    # browser, the Start menu, ...), with memory [N/A]. Those do not compete for compute; only other Python programs do
    # (another pipeline step, a training run, a notebook), so only those are reported.
    $busy = @($busy | Where-Object { $_ -match 'pythonw?\.exe' })
    if ($busy.Count -gt 0) {
        Write-Host "Other programs are using the GPU:`n$busy" -ForegroundColor Yellow
        $ans = Read-Host "Only one program should use the GPU at a time. Continue anyway? (y/N)"
        if ($ans -ne 'y') { exit 1 }
    }
}
