"""Inference devices on this machine, as seen by the Python that runs AutoMorph v2.

GPUs come first: NVIDIA (and AMD through ROCm, which PyTorch also reports as cuda), then
Apple silicon (mps). The CPU is the last resort. "auto" picks, when a job starts, the GPU
with the most free memory, so a card busy with other work is passed over.
"""

from __future__ import annotations

import json
import subprocess
import threading

from . import automorph

# a device counts only if it really runs a small convolution: a GPU can be listed yet unusable
# (seen: virtual Macs report an Apple GPU that cannot allocate any memory)
PROBE = r"""
import json, torch
def works(dev):
    try:
        x = torch.nn.functional.conv2d(torch.ones(1, 3, 32, 32, device=dev), torch.ones(4, 3, 3, 3, device=dev))
        return float(x.sum().item()) == 4 * 30 * 30 * 27
    except Exception:
        return False
out = []
for i in range(torch.cuda.device_count()):
    if works(f"cuda:{i}"):
        free, total = torch.cuda.mem_get_info(i)
        out.append({"id": f"cuda:{i}", "name": torch.cuda.get_device_name(i),
                    "memory_gb": round(total / 2**30, 1), "free_gb": round(free / 2**30, 1)})
if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and works("mps"):
    out.append({"id": "mps", "name": "Apple GPU (Metal)"})
print(json.dumps(out))
"""
CPU = {"id": "cpu", "name": "CPU (slow, last resort)"}

_cache: list[dict] | None = None
_lock = threading.Lock()


def probe() -> list[dict]:
    """Every usable device, GPUs first and the CPU last. Never raises: without torch it is just the CPU."""
    global _cache
    try:
        out = subprocess.run([str(automorph.GPU_PY), "-c", PROBE], env=automorph.env(), capture_output=True,
                             text=True, timeout=180)
        gpus = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else []
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        gpus = []
    with _lock:
        _cache = gpus + [CPU]
        return list(_cache)


def cached() -> list[dict]:
    """The last probe; empty until the first one has finished."""
    with _lock:
        return list(_cache or [])


def known() -> list[dict]:
    """The last probe, probing once if there was none."""
    return cached() or probe()


def best(devices: list[dict]) -> str:
    cuda = [d for d in devices if d["id"].startswith("cuda")]
    if cuda:
        return max(cuda, key=lambda d: (d["free_gb"], d["memory_gb"]))["id"]
    return "mps" if any(d["id"] == "mps" for d in devices) else "cpu"


def resolve(requested: str) -> str:
    """The device a job runs on: an explicit choice as given, "auto" (or nothing) as the best GPU right now."""
    requested = (requested or "auto").lower()
    return best(probe()) if requested == "auto" else requested


def valid(requested: str) -> bool:
    return requested in ("", "auto") or any(d["id"] == requested for d in known()) or requested == "cuda"
