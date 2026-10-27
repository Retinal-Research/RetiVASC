"""Small, cross-platform progress reporting shared by the v2 stages.

The machine-readable mode deliberately uses line-delimited JSON so the parent
pipeline can render one unified display without parsing human text.  Stage
cores create and update tasks on their main thread; worker threads never write
progress output.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, TextIO


PROGRESS_MODES = ("auto", "bar", "plain", "protocol", "off")
PROTOCOL_PREFIX = "AUTOMORPH_PROGRESS "


def parse_progress_line(line: str) -> dict[str, Any] | None:
    """Return a protocol payload, or ``None`` for an ordinary output line."""
    candidate = line.rstrip("\r\n")
    if not candidate.startswith(PROTOCOL_PREFIX):
        return None
    try:
        payload = json.loads(candidate[len(PROTOCOL_PREFIX) :])
    except (json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_tqdm():
    """Import tqdm lazily; releases without the optional package still work."""
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm


def _is_interactive(stream: TextIO) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, OSError):
        return False


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    seconds = max(0, int(seconds + 0.5))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class ProgressReporter:
    """Factory for sequential progress tasks owned by the creating thread."""

    def __init__(
        self,
        stage: str,
        mode: str = "auto",
        *,
        stream: TextIO | None = None,
        min_interval: float = 1.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        normalized = mode.lower()
        if normalized not in PROGRESS_MODES:
            choices = ", ".join(PROGRESS_MODES)
            raise ValueError(f"progress mode must be one of: {choices}")
        if min_interval < 0:
            raise ValueError("min_interval cannot be negative")
        self.stage = stage
        self.requested_mode = normalized
        self.stream = stream if stream is not None else sys.stdout
        self.min_interval = min_interval
        self.clock = clock
        self.owner_thread = threading.get_ident()

    def task(self, phase: str, total: int, unit: str) -> "ProgressTask":
        return ProgressTask(self, phase=phase, total=total, unit=unit)


@dataclass
class ProgressTask:
    """One phase within a stage, updated exclusively by its main thread."""

    reporter: ProgressReporter
    phase: str
    total: int
    unit: str

    def __post_init__(self) -> None:
        if self.total < 0:
            raise ValueError("progress total cannot be negative")
        self.current = 0
        self.started = self.reporter.clock()
        self.last_plain_emit = float("-inf")
        self.closed = False
        self.bar = None

        requested = self.reporter.requested_mode
        use_bar = requested == "bar" or (
            requested == "auto" and _is_interactive(self.reporter.stream)
        )
        tqdm_type = _load_tqdm() if use_bar else None
        if use_bar and tqdm_type is not None:
            self.mode = "bar"
            self.bar = tqdm_type(
                total=self.total,
                desc=f"{self.reporter.stage} {self.phase}",
                unit=self.unit,
                file=self.reporter.stream,
                dynamic_ncols=True,
                leave=True,
            )
        elif requested == "protocol":
            self.mode = "protocol"
        elif requested == "off":
            self.mode = "off"
        else:
            # Non-interactive auto and a missing tqdm installation both use a
            # durable, newline-based display suitable for run.log.
            self.mode = "plain"

        self._emit("start", force=True)

    def __enter__(self) -> "ProgressTask":
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> bool:
        self.close("error" if exc_type is not None else "complete")
        return False

    def _assert_owner(self) -> None:
        if threading.get_ident() != self.reporter.owner_thread:
            raise RuntimeError("progress updates must be emitted by the owning main thread")

    def _payload(self, event: str) -> dict[str, Any]:
        elapsed = max(0.0, self.reporter.clock() - self.started)
        rate = self.current / elapsed if elapsed > 0 and self.current > 0 else 0.0
        remaining = max(0, self.total - self.current)
        eta = remaining / rate if rate > 0 else None
        return {
            "event": event,
            "stage": self.reporter.stage,
            "phase": self.phase,
            "current": self.current,
            "total": self.total,
            "unit": self.unit,
            "elapsed_s": round(elapsed, 3),
            "rate": round(rate, 3),
            "eta_s": round(eta, 3) if eta is not None else None,
        }

    def _emit(self, event: str, *, force: bool = False) -> None:
        if self.mode in {"off", "bar"}:
            return
        payload = self._payload(event)
        if self.mode == "protocol":
            print(
                PROTOCOL_PREFIX + json.dumps(payload, separators=(",", ":")),
                file=self.reporter.stream,
                flush=True,
            )
            return

        now = self.reporter.clock()
        if not force and now - self.last_plain_emit < self.reporter.min_interval:
            return
        self.last_plain_emit = now
        total = payload["total"]
        percent = (100.0 * self.current / total) if total else 100.0
        eta = _duration(payload["eta_s"])
        elapsed = _duration(payload["elapsed_s"])
        suffix = "" if event in {"start", "progress"} else f" [{event}]"
        print(
            f"{self.reporter.stage} {self.phase}: {self.current}/{total} {self.unit} "
            f"({percent:.1f}%) | elapsed {elapsed} | "
            f"rate {payload['rate']:.2f} {self.unit}/s | ETA {eta}{suffix}",
            file=self.reporter.stream,
            flush=True,
        )

    def update(self, amount: int = 1) -> None:
        self._assert_owner()
        if self.closed:
            raise RuntimeError("cannot update a closed progress task")
        if amount < 0:
            raise ValueError("progress update cannot be negative")
        self.current += amount
        if self.bar is not None:
            self.bar.update(amount)
        else:
            self._emit("progress")

    def close(self, status: str = "complete") -> None:
        self._assert_owner()
        if self.closed:
            return
        self.closed = True
        if self.bar is not None:
            if status != "complete":
                self.bar.set_postfix_str(status)
            self.bar.close()
        else:
            self._emit(status, force=True)

