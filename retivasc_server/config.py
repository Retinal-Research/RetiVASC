"""Machine-specific settings, kept out of the code and out of git.

They live in <retivasc>/retivasc.local.json (start from retivasc.local.example.json).
An environment variable, where one is named, overrides the file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "retivasc.local.json"


def load() -> dict:
    try:
        return json.loads(FILE.read_text(encoding="utf-8-sig"))  # PowerShell 5.1 writes a BOM
    except (OSError, ValueError):
        return {}


def get(key: str, env: str | None = None, default: str = "") -> str:
    return (os.environ.get(env) if env else None) or load().get(key) or default
