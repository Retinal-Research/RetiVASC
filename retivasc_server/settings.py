"""Server settings from retivasc.local.json (see retivasc.local.example.json)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class Settings:
    auth: str = "accounts"  # "accounts": sign in; "open": anyone, one private guest session per browser
    host: str = "127.0.0.1"
    port: int = 8080
    data_dir: Path = config.ROOT / "work"
    retention_days: int = 30
    device: str = "auto"  # AutoMorph picks cuda, then Apple mps, then cpu
    allowed_folders: list[Path] = field(default_factory=list)  # server folders users may analyse in place
    ssl_certfile: str = ""
    ssl_keyfile: str = ""

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"


def load() -> Settings:
    raw = config.load().get("server", {})
    s = Settings()
    s.auth = raw.get("auth", s.auth)
    if s.auth not in ("accounts", "open"):
        raise ValueError('server.auth must be "accounts" or "open"')
    s.host = raw.get("host", s.host)
    s.port = int(raw.get("port", s.port))
    s.data_dir = Path(raw.get("data_dir", s.data_dir))
    s.retention_days = int(raw.get("retention_days", s.retention_days))
    s.device = raw.get("device", s.device)
    s.allowed_folders = [Path(p) for p in raw.get("allowed_folders", [])]
    s.ssl_certfile = raw.get("ssl_certfile", "")
    s.ssl_keyfile = raw.get("ssl_keyfile", "")
    return s
