# Step A: copy a folder of panoramas into 1_room_finder\data\<Name>, scan every pair with GeoVerify (GPU), then detect
# door openings with the bundled SAM3 (door-aware wedge). Resumable: run the same command again to continue.
# Usage (from the package root):
#   powershell -ExecutionPolicy Bypass -File 1_room_finder\Import_Dataset.ps1 -Source D:\new_panos -Name new_panos
#   powershell -ExecutionPolicy Bypass -File 1_room_finder\Import_Dataset.ps1 -Name new_panos -NoCopy
#       (-NoCopy: the panoramas are already in 1_room_finder\data\new_panos; only scan)
# Options: -Limit N    stop after N new pairs (smoke test)
#          -NoOpenings skip the SAM3 door-opening step (the review app then uses the plain wedge)
# Source layout: one sub-folder per house, each holding that house's panoramas (2:1 jpg/png).
# Output: 1_room_finder\scans\<Name>\ (open it with Start_Review.bat)
param([string]$Source = '', [Parameter(Mandatory=$true)][string]$Name, [int]$Limit = 0, [switch]$NoCopy, [switch]$NoOpenings)
. "$PSScriptRoot\..\tools\_env.ps1"
if (-not $NoCopy) {
    if (-not $Source) { throw 'Give -Source <folder>, or -NoCopy when 1_room_finder\data\<Name> is already in place.' }
    $src = (Resolve-Path $Source).Path        # resolved against the caller's folder
}
Set-Location $FinderRoot
$data = "data\$Name"; $scan = "scans\$Name"
if (-not $NoCopy) {
    robocopy $src (Join-Path $FinderRoot $data) /E /NFL /NDL /NJH /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Copy failed (robocopy exit code $LASTEXITCODE)." }
    Write-Host "Copied to $FinderRoot\$data; scanning..."
} elseif (-not (Test-Path $data)) {
    throw "$FinderRoot\$data not found."
}
$py = Get-MainPy
if (-not $py) { throw 'Main environment not found. Run Setup.ps1 first.' }
$a = @('-u', '-m', 'finder.scan', '--root', $data, '--out', $scan)
if ($Limit -gt 0) { $a += @('--limit', $Limit) }
& $py @a
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($NoOpenings) { exit 0 }
$sam3 = Get-Sam3Root
if ($sam3) {
    & "$FinderRoot\Start_Openings.ps1" -Scan $scan -Sam3Root $sam3
} else {
    Write-Host 'sam3-pano-seg not found (1_room_finder\third_party\sam3-pano-seg or RCF_SAM3_ROOT): door-opening detection skipped; the review app uses the plain wedge. Run Start_Openings.ps1 later to add it.'
}
