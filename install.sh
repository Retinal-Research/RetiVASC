#!/usr/bin/env bash
# RetiVASC: one-time setup on Linux or macOS.   ./install.sh [nvidia|cpu|mac]
# Creates .venv here, installs the server and AutoMorph v2 with the matching PyTorch build,
# records it in retivasc.local.json, then runs the self-test.
set -euo pipefail
cd "$(dirname "$0")"
GPU="${1:-nvidia}"
[ -d .venv ] || python3.12 -m venv .venv
PY="$PWD/.venv/bin/python"
# model weights are not in git: download them from the GitHub release (about 2.5 GB, once)
"$PY" tools/release.py fetch
"$PY" -m pip install --upgrade pip
case "$GPU" in
  nvidia) "$PY" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128; DEVICE=cuda ;;
  cpu)    "$PY" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu; DEVICE=cpu ;;
  mac)    "$PY" -m pip install torch==2.7.1 torchvision==0.22.1; DEVICE=mps ;;
  *) echo "usage: ./install.sh [nvidia|cpu|mac]"; exit 2 ;;
esac
"$PY" -m pip install -r requirements-server.txt
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
echo "Then start the server        $PY -m retivasc_server serve"
