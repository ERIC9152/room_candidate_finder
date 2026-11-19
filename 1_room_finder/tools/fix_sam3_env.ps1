# Re-point the bundled SAM3 environment at the folder it is in now.
# A Windows venv stores absolute paths: pyvenv.cfg (home = base Python) and the two editable-install finders
# (sam3 -> third_party\sam3\sam3, sam3_pano -> sam3_pano). After the package is copied or moved, those still name
# the old location, so python.exe would fail to start. This rewrites them; it does nothing when they already match.
# Called by Setup.ps1 and Start_Openings.ps1. Usage: powershell -ExecutionPolicy Bypass -File tools\fix_sam3_env.ps1 [-Sam3Root <folder>]
param([string]$Sam3Root = '')
$ErrorActionPreference = 'Stop'
if (-not $Sam3Root) { $Sam3Root = Join-Path (Split-Path -Parent $PSScriptRoot) 'third_party\sam3-pano-seg' }
if (-not (Test-Path $Sam3Root)) { Write-Host "sam3-pano-seg not found: $Sam3Root"; exit 0 }
$Sam3Root = (Resolve-Path $Sam3Root).Path.TrimEnd('\')
$cfg = Join-Path $Sam3Root '.runtime\sam3-env\pyvenv.cfg'
if (-not (Test-Path $cfg)) { Write-Host "no SAM3 environment in $Sam3Root (.runtime\sam3-env)"; exit 0 }

$utf8 = New-Object System.Text.UTF8Encoding $false
$text = [IO.File]::ReadAllText($cfg)
if ($text -notmatch '(?m)^home\s*=\s*(.+?)\\\.runtime\\python312\s*$') { throw "unexpected pyvenv.cfg format: $cfg" }
$old = $Matches[1].Trim()
if ($old -eq $Sam3Root) { Write-Host "SAM3 environment paths OK ($Sam3Root)"; exit 0 }
if (-not ($Sam3Root -match '^[\x00-\x7F]*$')) { Write-Host "WARNING: '$Sam3Root' contains non-ASCII characters; the SAM3 environment may fail." -ForegroundColor Yellow }

$files = @($cfg) + @(Get-ChildItem (Join-Path $Sam3Root '.runtime\sam3-env\Lib\site-packages') -Filter '__editable___*_finder.py' | ForEach-Object { $_.FullName })
foreach ($f in $files) {
    $t = [IO.File]::ReadAllText($f)
    $n = $t.Replace($old, $Sam3Root)                                    # pyvenv.cfg: plain backslashes
    $n = $n.Replace($old.Replace('\', '\\'), $Sam3Root.Replace('\', '\\'))  # finders: escaped backslashes in Python literals
    if ($n -ne $t) { [IO.File]::WriteAllText($f, $n, $utf8) }
}
Write-Host "SAM3 environment re-pointed: $old -> $Sam3Root"
