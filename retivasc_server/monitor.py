"""Live load of this machine for the status bar: CPU, memory and each GPU, sampled once a second.

One thread samples for every viewer, so many open pages do not add load or skew the CPU reading.
It starts with the first request and stops after a minute without one. GPUs are read from
nvidia-smi (NVIDIA) or ioreg (Apple silicon, no sudo needed); a GPU that cannot be read is left out.
The Apple Neural Engine has no load counter outside `sudo powermetrics`, so it is not shown.
"""

from __future__ import annotations

import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import deque

import psutil

HISTORY = 60   # samples kept, one a second
IDLE = 60      # seconds without a viewer before sampling stops

HOST = socket.gethostname().split(".")[0]  # the status bar says which machine the numbers are from

_history: deque[dict] = deque(maxlen=HISTORY)
_lock = threading.Lock()
_last_view = 0.0
_thread: threading.Thread | None = None


def _nvidia() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
    gpus = []
    for line in out.stdout.strip().splitlines():
        i, name, util, used, total = (s.strip() for s in line.split(","))
        gpus.append({"id": f"cuda:{i}", "name": name, "util": float(util),
                     "mem_used": int(used) << 20, "mem_total": int(total) << 20})
    return gpus


def _apple() -> list[dict]:
    out = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                         capture_output=True, text=True, timeout=5).stdout
    util = re.search(r'"Device Utilization %"=(\d+)', out)
    if not util:
        return []
    used = re.search(r'"In use system memory"=(\d+)', out)
    # unified memory: the GPU shares the system RAM, so its ceiling is the RAM itself
    return [{"id": "mps", "name": "Apple GPU", "util": float(util.group(1)),
             "mem_used": int(used.group(1)) if used else None, "mem_total": psutil.virtual_memory().total}]


def _gpus() -> list[dict]:
    try:
        return _apple() if sys.platform == "darwin" else _nvidia()
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return []


def _sample() -> dict:
    mem = psutil.virtual_memory()
    return {"t": time.time(), "cpu": psutil.cpu_percent(interval=None),
            "mem_used": mem.total - mem.available, "mem_total": mem.total, "gpus": _gpus()}


def _run() -> None:
    global _thread
    psutil.cpu_percent(interval=None)  # the first reading has no interval to measure over
    while True:
        time.sleep(1)
        s = _sample()
        with _lock:
            _history.append(s)
            if time.time() - _last_view > IDLE:
                _history.clear()
                _thread = None
                return


def history() -> list[dict]:
    """The last minute of samples, oldest first; starts the sampler if it is not running."""
    global _last_view, _thread
    with _lock:
        _last_view = time.time()
        if _thread is None:
            _thread = threading.Thread(target=_run, daemon=True, name="monitor")
            _thread.start()
        return list(_history)
