#!/usr/bin/env bash
# RetiVASC server on Linux or macOS: web UI and full AutoMorph v2 on one port (see "server" in retivasc.local.json).
# First-time setup: ./install.sh. Add users with:  .venv/bin/python -m retivasc_server adduser NAME [--admin]
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -x "$PY" ] || { echo "Run ./install.sh first."; exit 1; }
"$PY" tools/release.py fetch
if [ ! -f webui/dist/index.html ]; then (cd webui && { [ -d node_modules ] || npm ci; } && npx vite build); fi
exec "$PY" -m retivasc_server serve "$@"
