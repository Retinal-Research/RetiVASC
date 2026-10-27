# RetiVASC: one-time setup on a Windows machine.
#   powershell -ExecutionPolicy Bypass -File install.ps1 [-Gpu nvidia|cpu]
# Creates .venv here (nothing is installed system-wide), installs the server and AutoMorph v2
# with the matching PyTorch build, records it in retivasc.local.json, then runs the self-test.
param([ValidateSet('nvidia', 'cpu')][string]$Gpu = 'nvidia')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) { py -3.12 -m venv .venv }
$py = (Resolve-Path .venv\Scripts\python.exe).Path
# model weights are not in git: download them from the GitHub release (about 2.5 GB, once)
& $py tools\release.py fetch
if ($LASTEXITCODE -ne 0) { throw 'downloading the model weights failed' }
& $py -m pip install --upgrade pip
if ($Gpu -eq 'nvidia') {
  # CUDA 12.8 wheels cover current NVIDIA cards, RTX 50-series included
  & $py -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
} else {
  & $py -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
}
& $py -m pip install -r requirements-server.txt

$cfgPath = 'retivasc.local.json'
$cfg = if (Test-Path $cfgPath) { Get-Content -Raw $cfgPath | ConvertFrom-Json } else { Get-Content -Raw retivasc.local.example.json | ConvertFrom-Json }
$cfg | Add-Member -Force python $py
$cfg | Add-Member -Force gpu_python $py
if (-not $cfg.server) { $cfg | Add-Member -Force server ([pscustomobject]@{ host = '127.0.0.1'; port = 8080; retention_days = 30; device = $(if ($Gpu -eq 'nvidia') { 'cuda' } else { 'cpu' }); allowed_folders = @() }) }
$cfg | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $cfgPath

if (-not (Test-Path webui\dist\index.html)) {
  Push-Location webui; npm ci; npx vite build; Pop-Location
}
& $py -X utf8 -m retivasc_server selftest
Write-Host "`nNext: add an administrator   $py -m retivasc_server adduser NAME --admin"
Write-Host "Then start the server        start.bat"
