"""User accounts (bcrypt hashes in <data_dir>/users.json) and signed session tokens."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import threading
import time
from pathlib import Path

import bcrypt

SESSION_HOURS = 12
GUEST_PREFIX = "guest-"  # anonymous sessions in open mode; never a real user name
_lock = threading.Lock()


class Accounts:
    def __init__(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        self.file = data_dir / "users.json"
        key = data_dir / "secret.key"
        if not key.exists():
            key.write_bytes(os.urandom(32))
        self.secret = key.read_bytes()

    def _users(self) -> dict:
        return json.loads(self.file.read_text(encoding="utf-8")) if self.file.exists() else {}

    def add(self, name: str, password: str, admin: bool = False) -> None:
        if name.startswith(GUEST_PREFIX):
            raise ValueError(f"user names may not start with {GUEST_PREFIX}")
        if not name or len(password) < 8:
            raise ValueError("a user name and a password of at least 8 characters are required")
        with _lock:
            users = self._users()
            users[name] = {"hash": bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode(), "admin": admin}
            self.file.write_text(json.dumps(users, indent=1), encoding="utf-8")

    def remove(self, name: str) -> bool:
        with _lock:
            users = self._users()
            if users.pop(name, None) is None:
                return False
            self.file.write_text(json.dumps(users, indent=1), encoding="utf-8")
            return True

    def names(self) -> list[str]:
        return sorted(self._users())

    def check(self, name: str, password: str) -> dict | None:
        u = self._users().get(name)
        if u and bcrypt.checkpw(password.encode(), u["hash"].encode()):
            return {"name": name, "admin": bool(u.get("admin"))}
        return None

    def is_admin(self, name: str) -> bool:
        return bool(self._users().get(name, {}).get("admin"))

    # sessions: "<name>|<expiry>|<signature>", base64 encoded; no server-side state
    def token(self, name: str, hours: float = SESSION_HOURS) -> str:
        body = f"{name}|{int(time.time() + hours * 3600)}"
        sig = hmac.new(self.secret, body.encode(), hashlib.sha256).hexdigest()
        return base64.urlsafe_b64encode(f"{body}|{sig}".encode()).decode()

    def guest_token(self, days: int) -> tuple[str, str]:
        """A new anonymous session that lasts as long as its jobs are kept."""
        name = GUEST_PREFIX + os.urandom(8).hex()
        return name, self.token(name, hours=days * 24)

    def user_of(self, token: str | None, guests: bool = False) -> str | None:
        if not token:
            return None
        try:
            name, expiry, sig = base64.urlsafe_b64decode(token.encode()).decode().rsplit("|", 2)
        except (ValueError, UnicodeDecodeError):
            return None
        good = hmac.new(self.secret, f"{name}|{expiry}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, good) or int(expiry) < time.time():
            return None
        if name.startswith(GUEST_PREFIX):
            return name if guests else None
        if name not in self._users():
            return None
        return name
