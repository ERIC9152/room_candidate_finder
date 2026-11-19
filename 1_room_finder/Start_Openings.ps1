# SAM3 "open doorway" masks for a scan, then the door-aware wedge scores (open_scores.jsonl).
# Called by Import_Dataset.ps1; run it on its own only to add door openings to an existing scan.
# Needs the bundled sam3-pano-seg (GPU) and the main environment. Resumable: run again to continue.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File 1_room_finder\Start_Openings.ps1 -Scan scans\mydata [-Sam3Root <sam3-pano-seg folder>]
#        (-Scan is relative to 1_room_finder) sam3-pano-seg: -Sam3Root, else env RCF_SAM3_ROOT, else 1_room_finder\third_party\sam3-pano-seg
param([Parameter(Mandatory=$true)][string]$Scan, [string]$Sam3Root = "")
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\..\tools\_env.ps1"
Set-Location $FinderRoot
$given = $Sam3Root
$Sam3Root = Get-Sam3Root $given
$py = Get-MainPy
if (-not $py) { throw "Main environment not found. Run Setup.ps1 first." }
$scanAbs = (Resolve-Path $Scan).Path
if (-not (Test-Path (Join-Path $scanAbs "manifest.json"))) { throw "$Scan has no manifest.json; scan it first (Import_Dataset.ps1)." }

# 1. which panoramas still need a mask
$todoFile = Join-Path $scanAbs "openings_todo.txt"
& $py -m finder.openings --scan $scanAbs --list-todo $todoFile
if ($LASTEXITCODE -ne 0) { throw "Listing the panoramas to process failed." }
$todo = @(Get-Content -Encoding UTF8 $todoFile | Where-Object { $_ })

if ($todo.Count -gt 0) {
    if (-not $Sam3Root) { throw "sam3-pano-seg not found (it needs .runtime\sam3-env and checkpoints\sam3.pt). Use -Sam3Root, env RCF_SAM3_ROOT, or 1_room_finder\third_party\sam3-pano-seg. $given" }
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$FinderRoot\tools\fix_sam3_env.ps1" -Sam3Root $Sam3Root   # venv paths follow the folder
    if ($LASTEXITCODE -ne 0) { throw "Re-pointing the SAM3 environment failed." }
    $sam3Py = Join-Path $Sam3Root ".runtime\sam3-env\Scripts\python.exe"
    $ckpt = Join-Path $Sam3Root "checkpoints\sam3.pt"
    foreach ($req in @($sam3Py, $ckpt)) {
        if (-not (Test-Path $req)) { throw "SAM3 file not found: $req (point -Sam3Root or env RCF_SAM3_ROOT at sam3-pano-seg)." }
    }
    # 2. stage every panorama under a unique flat name <house>__<id><ext> (hard link; copy if not possible)
    $stage = Join-Path $scanAbs "openings_stage"
    $raw = Join-Path $scanAbs "openings_raw"
    New-Item -ItemType Directory -Force -Path $stage, $raw | Out-Null
    foreach ($line in $todo) {
        $house, $id, $path = $line -split "`t", 3
        $link = Join-Path $stage ("{0}__{1}{2}" -f $house, $id, [IO.Path]::GetExtension($path))
        if (-not (Test-Path -LiteralPath $link)) {
            try { New-Item -ItemType HardLink -Path $link -Value $path | Out-Null } catch { Copy-Item -LiteralPath $path -Destination $link }
        }
    }
    # 3. SAM3, one run over everything (about 3-4 s per panorama on an RTX GPU)
    Write-Host ("SAM3 open doorway: {0} panoramas; progress in {1}" -f $todo.Count, (Join-Path $scanAbs "openings.log"))
    $env:PYTHONUTF8 = "1"
    $env:SAM3_SOURCE_ROOT = Join-Path $Sam3Root "third_party\sam3"
    $env:SAM3_CKPT = $ckpt
    Push-Location $Sam3Root
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & $sam3Py -m sam3_pano.seg.unified --input $stage --output $raw --sam3-ckpt $ckpt --categories door `
        --door-prompt-override "open doorway" --sam3-score-threshold 0.0 --pano-vote --mask-only --skip-existing `
        *> (Join-Path $scanAbs "openings.log")
    $code = $LASTEXITCODE; $ErrorActionPreference = $prev
    Pop-Location
    if ($code -ne 0) { throw "SAM3 failed (exit code $code); see openings.log." }
    # 4. shrink into openings\<house>\<id>_mask.png, then drop the staging and full-size copies
    & $py tools\import_openings.py --scan $scanAbs --src $raw
    if ($LASTEXITCODE -ne 0) { throw "Importing the door-opening masks failed." }
    Remove-Item -Recurse -Force $stage, $raw
}

# 5. door-aware wedge scores
& $py -m finder.openings --scan $scanAbs
if ($LASTEXITCODE -ne 0) { throw "Computing the door-aware wedge failed." }
Write-Host "Done. The review app uses the door-aware wedge automatically (default 40 deg)."
