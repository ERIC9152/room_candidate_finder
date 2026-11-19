# Spanning forest vs pose-graph optimisation, scored against ZInD test ground truth.
#   219 complete rooms with >= 3 panoramas, 1087 panoramas, 2536 pairs (files in this folder, made by
#   scripts\make_zind_room_groups.py from C:\zind_data and C:\zind\zind_partition.json).
# Steps: prepare (CPU) -> CoVisPose on every pair (GPU, about 20-30 min on an RTX 3060; resumable) -> evaluation (CPU).
# Usage (from anywhere):
#   powershell -ExecutionPolicy Bypass -File 2_pose_floorplan\evaluation\zind_test_posegraph\Run_ZInD_PoseGraph_Eval.ps1
# Options: -Name <run name> (default zind_test_posegraph; output in 2_pose_floorplan\runs\<Name>), -Cpu (CoVisPose on the CPU),
#          -ZindData <ZInD data folder> (default C:\zind_data), -Partition <zind_partition.json> (default C:\zind\zind_partition.json)
# gt.json (ZInD ground truth) is not in the Git repository; it is regenerated here from your ZInD copy when missing.
# Interrupted? Run the same command again: finished pairs are skipped.
param([string]$Name = 'zind_test_posegraph', [switch]$Cpu, [string]$ZindData = 'C:\zind_data', [string]$Partition = 'C:\zind\zind_partition.json')
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\..\..\..\tools\_env.ps1"
Set-Location $PoseRoot
$Py = Get-MainPy
if (-not $Py) { throw 'Main environment not found. Run Setup.ps1 first.' }
$In = 'evaluation\zind_test_posegraph'
$Run = "runs\$Name"
$env:PYTHONUNBUFFERED = '1'
function Check($what) {
    if ($LASTEXITCODE -ne 0) { Write-Host "$what failed (exit $LASTEXITCODE). Fix the message above and run the same command again." -ForegroundColor Red; exit 1 }
}
if (-not (Test-Path "$In\gt.json")) {
    Write-Host "`n== 0/3 ZInD input (gt.json missing): scripts\make_zind_room_groups.py" -ForegroundColor Cyan
    & $Py -B scripts\make_zind_room_groups.py --data-root $ZindData --partition $Partition --root-abs $ZindData --out $In
    Check 'make_zind_room_groups'
}
if (-not (Test-Path "$Run\manifest.json")) {
    Write-Host "`n== 1/3 prepare (CPU)" -ForegroundColor Cyan
    & $Py -B -m pipeline.prepare --groups "$In\room_groups.json" --index "$In\index.json" --output $Run
    Check 'prepare'
} else { Write-Host "Run folder exists, resuming: $Run" -ForegroundColor Cyan }
if (-not $Cpu) { Assert-GpuFree }
if (Test-Path "$Run\inference.lock") {
    $lockPid = Get-Content "$Run\inference.lock"
    if (Get-Process -Id $lockPid -ErrorAction SilentlyContinue) { Write-Host "CoVisPose already running (PID $lockPid)." -ForegroundColor Red; exit 1 }
    Remove-Item "$Run\inference.lock"
}
Write-Host "`n== 2/3 CoVisPose (GPU; CPU when none)" -ForegroundColor Cyan
$ErrorActionPreference = 'Continue'      # python warnings on stderr must not abort
$c = @('-B', 'scripts\manual_room_groups_v2.py', 'run', '--run', $Run)
if ($Cpu) { $c += @('--device', 'cpu') }
& $Py @c
Check 'CoVisPose'
Write-Host "`n== 3/3 evaluation: spanning forest vs pose graph (CPU)" -ForegroundColor Cyan
& $Py -B scripts\eval_pose_graph_zind.py --run $Run --gt "$In\gt.json"
Check 'evaluation'
Write-Host "`nDONE. Result: $PoseRoot\$Run\pose_graph_zind_eval_*.json" -ForegroundColor Green
