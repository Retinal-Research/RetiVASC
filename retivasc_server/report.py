"""AI reports through the lab's language-model API, for every RetiVASC page (the public site too),
without the API key leaving this machine.

The browser sends only its fact sheet: the numbers RetiVASC computed, with their names
(webui/src/engine/facts.ts), never a photograph. The prompt is built here from
webui/src/engine/prompts.json, exactly as the browser builds it for its own model, so the endpoint
writes RetiVASC reports and nothing else. Requests are limited per IP (a short window and a day) and
per day overall. Nothing is stored or logged but the counts.

Settings: "report" in retivasc.local.json (see retivasc.local.example.json); the key comes from the
environment variable it names, or from a file, never from the repository.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from . import config

PROMPTS = json.loads((config.ROOT / "webui" / "src" / "engine" / "prompts.json").read_text(encoding="utf-8"))
LANGUAGES = ("en", "zh")
SCOPES = ("photo", "folder")


@dataclass
class ReportSettings:
    enabled: bool = False
    provider: str = "asu"  # "fake" writes a canned report, for tests
    url: str = ""
    model: str = ""
    key_env: str = "RETIVASC_REPORT_KEY"
    key_file: str = ""
    # pages that may call it from another origin (the public site, a dev server)
    origins: list[str] = field(default_factory=lambda: ["https://retinal-research.github.io"])
    window_seconds: int = 60  # per IP: at most per_window requests in any window
    per_window: int = 3
    per_ip_per_day: int = 30
    per_day: int = 500  # all IPs together
    max_tokens: int = 1500
    timeout: float = 120

    def key(self) -> str:
        if os.environ.get(self.key_env):
            return os.environ[self.key_env]
        if self.key_file:
            try:
                return open(os.path.expanduser(self.key_file), encoding="utf-8").read().strip()
            except OSError:
                return ""
        return ""


def load() -> ReportSettings:
    raw = config.load().get("report", {})
    s = ReportSettings()
    for k, v in raw.items():
        if hasattr(s, k):
            setattr(s, k, type(getattr(s, k))(v) if not isinstance(getattr(s, k), list) else list(v))
    return s


# ---------------------------------------------------------------- limits

class Limiter:
    """Requests per IP in a sliding window and per UTC day, and per day for everyone."""

    def __init__(self, s: ReportSettings):
        self.s = s
        self.lock = threading.Lock()
        self.recent: dict[str, deque] = defaultdict(deque)
        self.day = ""
        self.daily: dict[str, int] = defaultdict(int)
        self.total = 0

    def take(self, ip: str, now: float | None = None) -> str | None:
        """None if the request may go ahead (and counts it), else why not."""
        now = time.time() if now is None else now
        today = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d")
        with self.lock:
            if today != self.day:
                self.day, self.daily, self.total = today, defaultdict(int), 0
            q = self.recent[ip]
            while q and q[0] <= now - self.s.window_seconds:
                q.popleft()
            if self.total >= self.s.per_day:
                return "the report service has reached today's limit; please try again tomorrow"
            if self.daily[ip] >= self.s.per_ip_per_day:
                return f"you have used today's {self.s.per_ip_per_day} reports; please try again tomorrow"
            if len(q) >= self.s.per_window:
                wait = int(q[0] + self.s.window_seconds - now) + 1
                return f"too many reports at once; please wait {wait} s"
            q.append(now)
            self.daily[ip] += 1
            self.total += 1
            return None


# ---------------------------------------------------------------- the request: a fact sheet, checked

def _text(v, limit: int, what: str) -> str:
    if not isinstance(v, str) or len(v) > limit:
        raise HTTPException(400, f"{what}: text of at most {limit} characters")
    return v


def sheet_of(raw) -> dict:
    """The fact sheet as facts.ts makes it, within sizes no real sheet comes near."""
    if not isinstance(raw, dict) or raw.get("scope") not in SCOPES or raw.get("language") not in LANGUAGES:
        raise HTTPException(400, "facts: not a RetiVASC fact sheet")
    groups = raw.get("groups")
    if not isinstance(groups, list) or not 1 <= len(groups) <= 16:
        raise HTTPException(400, "facts: 1 to 16 groups")
    out = []
    for g in groups:
        rows = g.get("rows") if isinstance(g, dict) else None
        if not isinstance(rows, list) or not 1 <= len(rows) <= 40:
            raise HTTPException(400, "facts: 1 to 40 rows a group")
        out.append({
            "title": _text(g.get("title"), 120, "group title"),
            "note": _text(g.get("note") or "", 300, "group note"),
            "rows": [[_text(r[0], 120, "row name"), _text(r[1], 200, "row value")] for r in rows
                     if isinstance(r, list) and len(r) == 2] or [],
        })
    terms = raw.get("terms", [])
    if not isinstance(terms, list) or any(t not in PROMPTS["terms"] for t in terms):
        raise HTTPException(400, "facts: unknown definitions")
    return {"scope": raw["scope"], "language": raw["language"], "groups": out, "terms": terms}


# ---------------------------------------------------------------- the prompt, as src/engine/report.ts builds it

def facts_text(sheet: dict) -> str:
    zh = sheet["language"] == "zh"
    colon = "：" if zh else ":"
    blocks = []
    for g in sheet["groups"]:
        note = (f"（{g['note']}）" if zh else f" ({g['note']})") if g.get("note") else ""
        lines = [f"{g['title']}{note}{colon}"] + [f"- {k}{colon}{'' if zh else ' '}{v}" for k, v in g["rows"]]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def report_prompt(sheet: dict, focus: str) -> str:
    l, zh = sheet["language"], sheet["language"] == "zh"
    colon = "：" if zh else ":"
    compares = any(g["title"] in ("Comparison between steps", "步骤之间的比较") for g in sheet["groups"])
    sections = [x for i, x in enumerate(PROMPTS["sections"][sheet["scope"]][l]) if sheet["scope"] == "photo" or i != 3 or compares]
    listed = "、".join(f"“## {x}”" for x in sections) if zh else ", ".join(f'"## {x}"' for x in sections)
    lines = [
        f"{PROMPTS['results'][sheet['scope']][l]}{colon}",
        facts_text(sheet),
        "",
        f"{PROMPTS['definitions'][l]}{colon}",
        "\n".join(f"- {PROMPTS['terms'][t][l]}" for t in sheet["terms"]),
        "",
        PROMPTS["write"][l].replace("{sections}", listed),
        PROMPTS["guide"][sheet["scope"]][l],
        PROMPTS["focus"][l].replace("{focus}", focus) if focus else "",
    ]
    return "\n".join(lines).strip()


def messages_of(body: dict) -> tuple[str, list[dict], str]:
    """System text, the conversation, and the language: the report, or a follow-up question about it."""
    sheet = sheet_of(body.get("facts"))
    focus = _text(body.get("focus") or "", 200, "focus").strip()
    l = sheet["language"]
    messages = [{"role": "user", "content": report_prompt(sheet, focus)}]
    question = body.get("question")
    if question is not None:
        messages.append({"role": "assistant", "content": _text(body.get("report"), 8000, "report")})
        history = body.get("history") or []
        if not isinstance(history, list) or len(history) > 5:
            raise HTTPException(400, "at most 5 follow-up questions a report")
        for h in history:
            q = _text(h.get("q") if isinstance(h, dict) else None, 300, "question")
            messages += [{"role": "user", "content": PROMPTS["question"][l].replace("{question}", q)},
                         {"role": "assistant", "content": _text(h.get("a"), 4000, "answer")}]
        messages.append({"role": "user", "content": PROMPTS["question"][l].replace("{question}", _text(question, 300, "question"))})
    return PROMPTS["system"][l], messages, l


# ---------------------------------------------------------------- providers

def fake(s: ReportSettings, system: str, messages: list[dict]) -> Iterator[str]:
    """A canned report built from the prompt's first lines: for tests, never for users."""
    first = messages[-1]["content"].splitlines()
    text = "## Summary\n- (test report) " + " ".join(first[1:4]) + "\n\n## Notes\n- Written by the test provider."
    for i in range(0, len(text), 20):
        yield text[i:i + 20]


def asu(s: ReportSettings, system: str, messages: list[dict]) -> Iterator[str]:
    """ASU's language-model API. Filled in when its interface is known."""
    raise RuntimeError("the ASU API is not connected yet")


PROVIDERS = {"fake": fake, "asu": asu}


# ---------------------------------------------------------------- the endpoint

def install(app: FastAPI, s: ReportSettings) -> None:
    limiter = Limiter(s)
    app.state.report_limiter = limiter

    def cors(request: Request, response: Response) -> None:
        origin = request.headers.get("origin")
        if not origin:
            return
        same = origin.split("://", 1)[-1] == request.headers.get("host")
        if not same and origin not in s.origins:
            raise HTTPException(403, "this site may not use the report service")
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"

    def client_ip(request: Request) -> str:
        host = request.client.host if request.client else ""
        # through the Cloudflare tunnel every request comes from this machine; Cloudflare names the visitor
        if host in ("127.0.0.1", "::1") and request.headers.get("cf-connecting-ip"):
            return request.headers["cf-connecting-ip"]
        return host

    @app.options("/api/report")
    def preflight(request: Request):
        r = Response(status_code=204)
        cors(request, r)
        r.headers["Access-Control-Allow-Methods"] = "GET, POST"
        r.headers["Access-Control-Allow-Headers"] = "Content-Type"
        r.headers["Access-Control-Max-Age"] = "600"
        return r

    @app.get("/api/report")
    def status(request: Request):
        ready = s.enabled and s.provider in PROVIDERS and (s.provider == "fake" or bool(s.key()))
        r = JSONResponse({"enabled": ready, "model": s.model if ready else "",
                          "limits": {"window_seconds": s.window_seconds, "per_window": s.per_window, "per_ip_per_day": s.per_ip_per_day}})
        cors(request, r)
        return r

    @app.post("/api/report")
    async def report(request: Request):
        headers = Response()
        cors(request, headers)  # a site that may not use it gets no answer it can read
        allowed = {k: v for k, v in headers.headers.items() if k.lower() != "content-length"}
        try:
            if not s.enabled or s.provider not in PROVIDERS:
                raise HTTPException(503, "the report service is not set up on this server")
            try:
                body = await request.json()
            except ValueError:
                raise HTTPException(400, "send the fact sheet as JSON")
            system, messages, _ = messages_of(body)  # checked before it counts against the limits
            refused = limiter.take(client_ip(request))
            if refused:
                raise HTTPException(429, refused)
        except HTTPException as e:  # with the CORS headers, so the page can show why
            return JSONResponse({"detail": e.detail}, status_code=e.status_code, headers=allowed)
        write = PROVIDERS[s.provider]

        def stream() -> Iterator[str]:
            try:
                yield from write(s, system, messages)
            except Exception as e:  # the browser shows what went wrong; the key never appears in it
                yield f"\n\n[report service error: {e}]"

        return StreamingResponse(stream(), media_type="text/plain; charset=utf-8", headers=allowed)
