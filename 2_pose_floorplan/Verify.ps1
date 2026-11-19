# Run the whole pipeline on the bundled sample (9 panoramas, 3 room groups) and compare with the reference results.
# Usage: powershell -ExecutionPolicy Bypass -File 2_pose_floorplan\Verify.ps1          (about 2-4 minutes; uses the GPU)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\..\tools\_env.ps1"
Set-Location $PoseRoot
$Name = "_verify_$(Get-Date -Format yyyyMMdd_HHmmss)"
& powershell -ExecutionPolicy Bypass -File "$PoseRoot\Run_Pipeline.ps1" -Groups sample\room_groups.json -ScanDir sample\scan -Index sample\data_index_sample.json -Name $Name -Title 'Verify sample'
if ($LASTEXITCODE -ne 0) { Write-Host 'Pipeline failed on the sample.' -ForegroundColor Red; exit 1 }
& (Get-MainPy) -B -m pipeline.verify --run "runs\$Name"
if ($LASTEXITCODE -eq 0) { Write-Host "`nThe package works on this computer." -ForegroundColor Green } else { Write-Host "`nResults differ from the reference (see above)." -ForegroundColor Red }
exit $LASTEXITCODE
