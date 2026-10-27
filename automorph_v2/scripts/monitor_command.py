"""Run a command and record its external NVIDIA memory peak without patching it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Sequence


def _gpu_process_memory() -> dict[int, float]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {}
    completed = subprocess.run(
        [
            executable,
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    rows: dict[int, float] = {}
    if completed.returncode != 0:
        return rows
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            rows[int(fields[0])] = float(fields[1])
        except ValueError:
            continue
    return rows


def _gpu_device_memory() -> dict[int, float]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {}
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    rows: dict[int, float] = {}
    if completed.returncode != 0:
        return rows
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            rows[int(fields[0])] = float(fields[1])
        except ValueError:
            continue
    return rows


def monitor(command: Sequence[str], interval_s: float = 0.05) -> dict[str, object]:
    if not command:
        raise ValueError("command cannot be empty")
    started = time.perf_counter()
    process = subprocess.Popen(list(command))
    target_peak_mib = 0.0
    all_processes_peak_mib = 0.0
    gpu_device_peak_mib: dict[int, float] = {}
    samples = 0
    target_rows_seen = 0
    while True:
        rows = _gpu_process_memory()
        device_rows = _gpu_device_memory()
        samples += 1
        target_value = rows.get(process.pid, 0.0)
        if target_value:
            target_rows_seen += 1
        target_peak_mib = max(target_peak_mib, target_value)
        all_processes_peak_mib = max(all_processes_peak_mib, sum(rows.values()))
        for index, used_mib in device_rows.items():
            gpu_device_peak_mib[index] = max(
                gpu_device_peak_mib.get(index, 0.0), used_mib
            )
        return_code = process.poll()
        if return_code is not None:
            break
        time.sleep(interval_s)
    return {
        "command": list(command),
        "pid": process.pid,
        "return_code": return_code,
        "wall_s": time.perf_counter() - started,
        "poll_interval_s": interval_s,
        "samples": samples,
        "target_gpu_rows_seen": target_rows_seen,
        "nvidia_smi_target_peak_mib": target_peak_mib,
        "nvidia_smi_all_processes_peak_mib": all_processes_peak_mib,
        "nvidia_smi_gpu_device_peak_mib": gpu_device_peak_mib,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    result = monitor(command, args.interval)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return int(result["return_code"])


if __name__ == "__main__":
    raise SystemExit(main())
