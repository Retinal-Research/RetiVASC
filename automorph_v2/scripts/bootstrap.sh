#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap.sh [--backend cpu|cu118|mps] [--python PATH]
                            [--with-report] [--with-test]

Defaults to the conservative CPU backend. CUDA 11.8 is supported on Linux;
MPS is supported on macOS. The script creates and uses .venv without requiring
shell activation.
EOF
}

backend="cpu"
python_executable=""
with_report=0
with_test=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      [[ $# -ge 2 ]] || { echo "--backend requires a value" >&2; exit 2; }
      backend="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || { echo "--python requires a value" >&2; exit 2; }
      python_executable="$2"
      shift 2
      ;;
    --with-report)
      with_report=1
      shift
      ;;
    --with-test)
      with_test=1
      with_report=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$backend" in
  cpu|cu118|mps) ;;
  *) echo "Unsupported backend: $backend" >&2; exit 2 ;;
esac

os_name="$(uname -s)"
case "$os_name:$backend" in
  Darwin:cpu|Darwin:mps) backend_file="torch-macos.txt" ;;
  Darwin:cu118)
    echo "CUDA 11.8 is not available on macOS; use --backend cpu or mps." >&2
    exit 2
    ;;
  Linux:cpu) backend_file="torch-cpu.txt" ;;
  Linux:cu118) backend_file="torch-cu118.txt" ;;
  Linux:mps)
    echo "MPS is only available on macOS; use --backend cpu or cu118." >&2
    exit 2
    ;;
  *)
    echo "Unsupported operating system: $os_name" >&2
    exit 2
    ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
venv_dir="$project_root/.venv"
venv_python="$venv_dir/bin/python"

if [[ -z "$python_executable" ]]; then
  if command -v python3.10 >/dev/null 2>&1; then
    python_executable="python3.10"
  elif command -v python3.11 >/dev/null 2>&1; then
    python_executable="python3.11"
  elif command -v python3.12 >/dev/null 2>&1; then
    python_executable="python3.12"
  elif command -v python3 >/dev/null 2>&1; then
    python_executable="python3"
  else
    echo "Python was not found. Install Python 3.10 (recommended), 3.11, or 3.12; or pass --python PATH." >&2
    exit 1
  fi
fi

version_check='import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(f"AutoMorph v2 requires Python >=3.10,<3.13; found {sys.version.split()[0]}")
print(f"Python: {sys.version.split()[0]}")'

"$python_executable" -c "$version_check"
if [[ ! -x "$venv_python" ]]; then
  echo "Creating isolated environment at $venv_dir"
  "$python_executable" -m venv "$venv_dir"
else
  echo "Reusing existing environment at $venv_dir"
fi

"$venv_python" -c "$version_check"
"$venv_python" -m pip install --upgrade \
  "pip==24.2" "setuptools==80.9.0" "wheel==0.44.0"
"$venv_python" -m pip install -r "$project_root/requirements/$backend_file"
"$venv_python" -m pip install -r "$project_root/requirements/runtime.txt"
"$venv_python" -m pip install --no-deps --no-build-isolation -e "$project_root"

if [[ "$with_report" -eq 1 ]]; then
  "$venv_python" -m pip install -r "$project_root/requirements/report.txt"
fi
if [[ "$with_test" -eq 1 ]]; then
  "$venv_python" -m pip install -r "$project_root/requirements/test.txt"
fi

"$venv_python" "$project_root/scripts/doctor.py" \
  --root "$project_root" --backend "$backend" --write-root "$project_root"

printf '\nAutoMorph v2 environment is ready.\n'
printf 'Python: %s\n' "$venv_python"
printf 'Run:    %q -m automorph_v2.cli --help\n' "$venv_python"
