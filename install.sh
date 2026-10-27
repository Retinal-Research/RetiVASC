#!/usr/bin/env bash
# RetiVASC: one-time setup on Linux or macOS.   ./install.sh [nvidia|cpu|mac]   (default: detected)
# Needs only Node.js 20+. Python is not needed beforehand: uv (https://docs.astral.sh/uv/) fetches
# its own Python 3.12 into .venv here, so no system Python, conda or PATH is touched. Then it
# installs the server and AutoMorph v2 with the matching PyTorch build, records it in
# retivasc.local.json, and runs the self-test.
set -euo pipefail
cd "$(dirname "$0")"
if [ "$(uname)" = Darwin ]; then DETECTED=mac
elif command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then DETECTED=nvidia
else DETECTED=cpu; fi
GPU="${1:-$DETECTED}"
echo "Installing for: $GPU"
if [ "$GPU" = mac ] && [ "$(uname -m)" != arm64 ]; then
  echo "PyTorch 2.7 has no builds for Intel Macs. Use an Apple-silicon Mac, Linux or Windows for the server;"
  echo "the browser version (README, 'Browser only') works on any Mac."
  exit 1
fi
command -v node >/dev/null || { echo "Node.js 20 or newer is required to build the web UI: https://nodejs.org"; exit 1; }
UV="$(command -v uv || true)"
if [ -z "$UV" ]; then
  echo "Installing uv (user-level, into ~/.local/bin) ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV="$HOME/.local/bin/uv"
fi
[ -x .venv/bin/python ] || "$UV" venv --python 3.12 .venv
PY="$PWD/.venv/bin/python"
# model weights are not in git: download them from the GitHub release (about 2.5 GB, once)
"$PY" tools/release.py fetch
case "$GPU" in
  nvidia) "$UV" pip install --python "$PY" torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128; DEVICE=auto ;;
  cpu)    "$UV" pip install --python "$PY" torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu; DEVICE=cpu ;;
  mac)    "$UV" pip install --python "$PY" torch==2.7.1 torchvision==0.22.1; DEVICE=auto ;;  # mps if available, else cpu
  *) echo "usage: ./install.sh [nvidia|cpu|mac]"; exit 2 ;;
esac
"$UV" pip install --python "$PY" -r requirements-server.txt
"$PY" - "$PY" "$DEVICE" <<'PYEOF'
import json, sys
from pathlib import Path
p = Path("retivasc.local.json")
cfg = json.loads(p.read_text()) if p.exists() else json.loads(Path("retivasc.local.example.json").read_text())
cfg["python"] = cfg["gpu_python"] = sys.argv[1]
cfg.setdefault("server", {"host": "127.0.0.1", "port": 8080, "retention_days": 30, "allowed_folders": []})["device"] = sys.argv[2]
p.write_text(json.dumps(cfg, indent=2))
PYEOF
if [ ! -f webui/dist/index.html ]; then (cd webui && npm ci && npx vite build); fi
"$PY" -m retivasc_server selftest
echo
echo "Next: add an administrator   $PY -m retivasc_server adduser NAME --admin"
echo "Then start the server        ./start.sh"
