# Room groups (room_groups.json exported by the review app) -> CoVisPose poses -> HorizonNet outlines -> floor plans -> viewer.
# Resumable: run the same command again after an interruption; finished work is skipped.
# Usage:
#   powershell -ExecutionPolicy Bypass -File 2_pose_floorplan\Run_Pipeline.ps1 -Groups 1_room_finder\scans\<dataset>\room_groups.json -Name <name>
#   (paths are relative to the folder you run it from; the output goes to 2_pose_floorplan\runs\<name>)
# Options:
#   -ScanDir <folder with the scan's manifest.json>   when room_groups.json was saved outside the scan folder
#   -Index <data_index.json>                           room names (Dollhouse only)
#   -Title "<text>"                                    viewer page title
#   -HnCpu                                             run HorizonNet on the CPU (only if the GPU is too new for torch 1.12)
#   -Cpu                                               run CoVisPose and HorizonNet on the CPU (slower; also chosen automatically when no CUDA GPU is usable)
#   -NoPoseGraph                                       step 4 = spanning tree only (pipeline.poses, the method before 2026-11-19);
#                                                      default = pose-graph optimisation (pipeline.pose_graph)
param(
    [Parameter(Mandatory = $true)][string]$Groups,
    [Parameter(Mandatory = $true)][string]$Name,
    [string]$ScanDir = '', [string]$Index = '', [string]$Title = '', [switch]$HnCpu, [switch]$Cpu, [switch]$NoPoseGraph
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\..\tools\_env.ps1"
# resolve the inputs against the caller's folder before moving into 2_pose_floorplan
$Groups = (Resolve-Path $Groups).Path
if ($ScanDir) { $ScanDir = (Resolve-Path $ScanDir).Path }
if ($Index)   { $Index = (Resolve-Path $Index).Path }
Set-Location $PoseRoot
$Py = Get-MainPy; $HnPy = Get-HnPy
if (-not $Py)   { throw 'Main environment not found. Run Setup.ps1 first.' }
if (-not $HnPy) { throw 'HorizonNet environment not found. Run Setup.ps1 first.' }
$Run = "runs\$Name"
$env:PYTHONUNBUFFERED = '1'
function Step($title, [scriptblock]$body) {
    Write-Host "`n== $title" -ForegroundColor Cyan
    $global:LASTEXITCODE = 0
    & $body
    if ($LASTEXITCODE -ne 0) { Write-Host "$title failed (exit $LASTEXITCODE). Fix the message above and run the same command again." -ForegroundColor Red; exit 1 }
}
if (-not (Test-Path "$Run\manifest.json")) {
    $a = @('-B', '-m', 'pipeline.prepare', '--groups', $Groups, '--output', $Run)
    if ($ScanDir) { $a += @('--scan-dir', $ScanDir) }
    if ($Index)   { $a += @('--index', $Index) }
    Step '1/6 prepare (CPU)' { & $Py @a }
} else { Write-Host "Run folder exists, resuming: $Run" -ForegroundColor Cyan }

if (-not $Cpu) { Assert-GpuFree }
if (Test-Path "$Run\inference.lock") {
    $lockPid = Get-Content "$Run\inference.lock"
    if (Get-Process -Id $lockPid -ErrorAction SilentlyContinue) { Write-Host "CoVisPose already running (PID $lockPid)." -ForegroundColor Red; exit 1 }
    Remove-Item "$Run\inference.lock"
}
$log = "$Run\covispose_gpu_$(Get-Date -Format yyyyMMdd_HHmmss).log"
$ErrorActionPreference = 'Continue'      # python warnings on stderr must not abort
$c = @('-B', 'scripts\manual_room_groups_v2.py', 'run', '--run', $Run)
if ($Cpu) { $c += @('--device', 'cpu') }
Step '2/6 CoVisPose (GPU; CPU when none)' { & $Py @c 2>&1 | Tee-Object -FilePath $log }
$h = @('-B', '-m', 'pipeline.horizonnet', '--run', $Run, '--hn-python', $HnPy)
if ($HnCpu -or $Cpu) { $h += '--no-cuda' }
Step '3/6 HorizonNet (GPU; CPU when none)' { & $Py @h }
if ($NoPoseGraph) { Step '4/6 poses: spanning tree (CPU)' { & $Py -B -m pipeline.poses --run $Run } } else { Step '4/6 poses: pose graph (CPU)' { & $Py -B -m pipeline.pose_graph --run $Run } }
Step '5/6 floor plans (CPU)' { & $Py -B -m pipeline.plans --run $Run }
$v = @('-B', '-m', 'pipeline.viewer', '--run', $Run)
if ($Title) { $v += @('--title', $Title) }
Step '6/6 viewer (CPU)' { & $Py @v }
Write-Host "`nDONE. Open $PoseRoot\$Run\viewer\viewer.html" -ForegroundColor Green
