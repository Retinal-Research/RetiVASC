@echo off
rem RetiVASC server: web UI and full AutoMorph v2 on one port (see "server" in retivasc.local.json).
rem First-time setup: install.ps1. Add users with:  python -m retivasc_server adduser NAME [--admin]
cd /d "%~dp0"
set PY=python
for /f "usebackq delims=" %%p in (`powershell -NoProfile -Command "try { (Get-Content -Raw retivasc.local.json | ConvertFrom-Json).python } catch { '' }"`) do if not "%%p"=="" set PY=%%p
"%PY%" tools\release.py fetch || exit /b 1
if not exist webui\dist\index.html (
  pushd webui
  if not exist node_modules call npm ci
  call npx vite build
  popd
)
for /f "usebackq delims=" %%u in (`powershell -NoProfile -Command "try { $s=(Get-Content -Raw retivasc.local.json | ConvertFrom-Json).server; 'http://localhost:' + $(if ($s.port) { $s.port } else { 8080 }) + '/' } catch { 'http://localhost:8080/' }"`) do set URL=%%u
start "" %URL%
"%PY%" -X utf8 -m retivasc_server serve
