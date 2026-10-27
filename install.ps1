# RetiVASC: one-time setup on a Windows machine.
#   powershell -ExecutionPolicy Bypass -File install.ps1 [-Gpu nvidia|cpu]
# Needs only Node.js 20+. Python is not needed beforehand: uv (https://docs.astral.sh/uv/) fetches
# its own Python 3.12 into .venv here, so no system Python, conda or PATH is touched. Then it
# installs the server and AutoMorph v2 with the matching PyTorch build, records it in
# retivasc.local.json, and runs the self-test.
param([ValidateSet('nvidia', 'cpu')][string]$Gpu = 'nvidia')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" } }

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
  throw 'Node.js 20 or newer is required to build the web UI: https://nodejs.org (then open a new terminal and rerun)'
}
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
  Write-Host 'Installing uv (user-level, into ~\.local\bin) ...'
  powershell -NoProfile -ExecutionPolicy Bypass -Command 'irm https://astral.sh/uv/install.ps1 | iex'; Check 'installing uv'
  $uv = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
}

if (-not (Test-Path .venv\Scripts\python.exe)) { & $uv venv --python 3.12 .venv; Check 'creating .venv' }
$py = (Resolve-Path .venv\Scripts\python.exe).Path
# model weights are not in git: download them from the GitHub release (about 2.5 GB, once)
& $py tools\release.py fetch; Check 'downloading the model weights'
if ($Gpu -eq 'nvidia') {
  # CUDA 12.8 wheels cover current NVIDIA cards, RTX 50-series included
  & $uv pip install --python $py torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
} else {
  & $uv pip install --python $py torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
}
Check 'installing PyTorch'
& $uv pip install --python $py -r requirements-server.txt; Check 'installing requirements-server.txt'

$cfgPath = 'retivasc.local.json'
$cfg = if (Test-Path $cfgPath) { Get-Content -Raw $cfgPath | ConvertFrom-Json } else { Get-Content -Raw retivasc.local.example.json | ConvertFrom-Json }
$cfg | Add-Member -Force python $py
$cfg | Add-Member -Force gpu_python $py
if (-not $cfg.server) { $cfg | Add-Member -Force server ([pscustomobject]@{ host = '127.0.0.1'; port = 8080; retention_days = 30; allowed_folders = @() }) }
$cfg.server | Add-Member -Force device $(if ($Gpu -eq 'nvidia') { 'auto' } else { 'cpu' })  # auto: the GPU with most free memory
$cfg | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $cfgPath

if (-not (Test-Path webui\dist\index.html)) {
  Push-Location webui
  npm ci; Check 'npm ci'
  npx vite build; Check 'building the web UI'
  Pop-Location
}
& $py -X utf8 -m retivasc_server selftest; Check 'the self-test'
Write-Host "`nNext: add an administrator   $py -m retivasc_server adduser NAME --admin"
Write-Host "Then start the server        start.bat"
