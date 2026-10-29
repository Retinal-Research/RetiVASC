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
MAX_BODY = 2 * 1024 * 1024  # bytes: a whole folder table with its follow-ups fits; the dollars a day are the real limit
SCOPES = ("photo", "folder")


@dataclass
class ReportSettings:
    enabled: bool = False
    provider: str = "gateway"  # the lab's LLM gateway (lab-llm); "fake" writes a canned report, for tests
    url: str = ""
    model: str = ""
    key_env: str = "RETIVASC_REPORT_KEY"  # RetiVASC's app key for the gateway, or key_file
    key_file: str = ""
    # pages that may call it from another origin (the public site, a dev server)
    origins: list[str] = field(default_factory=lambda: ["https://retinal-research.github.io"])
    window_seconds: float = 1  # per IP: at most per_window requests in any window
    per_window: int = 1
    max_running: int = 3  # per IP: answers being written at once
    ip_dollars_per_day: float = 1.0  # per IP, what the lab's API charged (UTC day)
    dollars_per_day: float = 10.0  # all IPs together
    # charged when the API reports no cost (an error part-way, say), to stay on the safe side
    fallback_cost: float = 0.002
    max_tokens: int = 8000  # reasoning models count their thinking here too
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
    """Per IP: requests in a sliding window, answers being written at once, and dollars a day; and
    dollars a day for everyone. Spending is what the lab's API reports for each answer."""

    def __init__(self, s: ReportSettings):
        self.s = s
        self.lock = threading.Lock()
        self.recent: dict[str, deque] = defaultdict(deque)
        self.running: dict[str, int] = defaultdict(int)
        self.day = ""
        self.spent: dict[str, float] = defaultdict(float)
        self.total = 0.0

    def _today(self, now: float) -> None:
        today = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d")
        if today != self.day:
            self.day, self.spent, self.total = today, defaultdict(float), 0.0

    def take(self, ip: str, now: float | None = None) -> str | None:
        """None if the request may go ahead (it then counts as running until done()), else why not."""
        now = time.time() if now is None else now
        with self.lock:
            self._today(now)
            q = self.recent[ip]
            while q and q[0] <= now - self.s.window_seconds:
                q.popleft()
            if self.total >= self.s.dollars_per_day:
                return "the report service has used today's budget; please try again tomorrow"
            if self.spent[ip] >= self.s.ip_dollars_per_day:
                return "you have used today's budget for reports; please try again tomorrow"
            if self.running[ip] >= self.s.max_running:
                return f"{self.s.max_running} reports are being written for you already; please wait for one to finish"
            if len(q) >= self.s.per_window:
                return "too many requests at once; please wait a moment"
            q.append(now)
            self.running[ip] += 1
            return None

    def today(self, ip: str, now: float | None = None) -> dict:
        """What this IP has spent today (UTC day) against its limit, and whether the day's budget for
        everyone still has room."""
        with self.lock:
            self._today(time.time() if now is None else now)
            return {"used": round(self.spent.get(ip, 0.0), 6), "limit": self.s.ip_dollars_per_day,
                    "open": self.total < self.s.dollars_per_day}

    def done(self, ip: str, cost: float, now: float | None = None) -> None:
        with self.lock:
            self._today(time.time() if now is None else now)
            self.running[ip] = max(0, self.running[ip] - 1)
            self.spent[ip] += cost
            self.total += cost


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
    if not isinstance(groups, list) or len(groups) > 500:
        raise HTTPException(400, "facts: at most 500 groups")
    out = []
    for g in groups:
        rows = g.get("rows") if isinstance(g, dict) else None
        if not isinstance(rows, list) or not 1 <= len(rows) <= 5000:
            raise HTTPException(400, "facts: 1 to 5000 rows a group")
        out.append({
            "title": _text(g.get("title"), 500, "group title"),
            "note": _text(g.get("note") or "", 2000, "group note"),
            "rows": [[_text(r[0], 500, "row name"), _text(r[1], 5000, "row value")] for r in rows
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


def report_prompt(sheet: dict, request: str, needs: list[str] = ()) -> str:
    """The results, their definitions, the person's request (the default one when it is empty), and what
    the steps after the report need from it (a table): fixed texts from prompts.json, never the page's."""
    l = sheet["language"]
    parts = [f"{PROMPTS['results'][sheet['scope']][l]}:", facts_text(sheet) if sheet["groups"] else PROMPTS["none"]]
    if sheet["terms"]:
        parts += ["", f"{PROMPTS['definitions'][l]}:", "\n".join(f"- {PROMPTS['terms'][t][l]}" for t in sheet["terms"])]
    parts += ["", f"{PROMPTS['heading']}:", request.strip() or PROMPTS["request"][sheet["scope"]]]
    for n in needs:
        parts += ["", PROMPTS["needs"][n][l]]
    return "\n".join(parts).strip()


EFFORTS = ("low", "medium", "high")


def messages_of(body: dict) -> tuple[str, list[dict], str]:
    """System text (always this server's), the conversation, and the reasoning effort: the report, or a
    follow-up question about it. The person may write the request, never the system prompt."""
    sheet = sheet_of(body.get("facts"))
    request = _text(body.get("prompt") or "", 20000, "prompt")
    effort = body.get("effort") or "medium"
    if effort not in EFFORTS:
        raise HTTPException(400, "effort: low, medium or high")
    needs = body.get("needs") or []
    if not isinstance(needs, list) or len(needs) > len(PROMPTS["needs"]) or any(n not in PROMPTS["needs"] for n in needs):
        raise HTTPException(400, f"needs: any of {', '.join(PROMPTS['needs'])}")
    l = sheet["language"]
    messages = [{"role": "user", "content": report_prompt(sheet, request, needs)}]
    question = body.get("question")
    if question is not None:
        messages.append({"role": "assistant", "content": _text(body.get("report"), 100000, "report")})
        history = body.get("history") or []
        if not isinstance(history, list) or len(history) > 100:
            raise HTTPException(400, "at most 100 follow-up questions a report")
        for h in history:
            q = _text(h.get("q") if isinstance(h, dict) else None, 20000, "question")
            messages += [{"role": "user", "content": PROMPTS["question"][l].replace("{question}", q)},
                         {"role": "assistant", "content": _text(h.get("a"), 100000, "answer")}]
        messages.append({"role": "user", "content": PROMPTS["question"][l].replace("{question}", _text(question, 20000, "question"))})
    return PROMPTS["system"][l], messages, effort


# ---------------------------------------------------------------- providers

def fake(s: ReportSettings, system: str, messages: list[dict], charge, effort: str = "medium") -> Iterator[str]:
    """A canned report built from the prompt's first lines: for tests, never for users."""
    first = messages[-1]["content"].splitlines()
    text = "## Summary\n- (test report) " + " ".join(first[1:4]) + "\n\n## Notes\n- Written by the test provider."
    for i in range(0, len(text), 20):
        yield text[i:i + 20]
    charge(0.001)


def gateway(s: ReportSettings, system: str, messages: list[dict], charge, effort: str = "medium") -> Iterator[str]:
    """The lab's LLM gateway on this machine (OpenAI-style chat, streamed), which holds the CreateAI token.
    RetiVASC has only its own app key for it."""
    import httpx

    body = {"messages": [{"role": "system", "content": system}, *messages], "max_tokens": s.max_tokens, "stream": True,
            "reasoning": {"effort": effort}}
    if s.model:
        body["model"] = s.model
    headers = {"Authorization": f"Bearer {s.key()}", "Content-Type": "application/json"}
    with httpx.stream("POST", f"{s.url.rstrip('/')}/chat/completions", json=body, headers=headers, timeout=s.timeout) as r:
        if r.status_code != 200:
            try:
                detail = json.loads(r.read()).get("error", {}).get("message", "")
            except ValueError:
                detail = ""
            raise RuntimeError(f"the language model service answered {r.status_code} {detail}".strip())
        for line in r.iter_lines():
            if not line.startswith("data: {"):
                continue
            chunk = json.loads(line[6:])
            if chunk.get("error"):
                raise RuntimeError(chunk["error"].get("message", "the language model service failed"))
            cost = ((chunk.get("usage") or {}).get("cost") or {}).get("total_token_cost")
            if cost is not None:
                charge(float(cost))
            for choice in chunk.get("choices") or []:
                text = (choice.get("delta") or {}).get("content")
                if text:
                    yield text


PROVIDERS = {"fake": fake, "gateway": gateway}


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
        """Whether reports can be written, and the caller's quota for today (what it has spent of its daily
        dollars), which the page shows next to the report."""
        ready = s.enabled and s.provider in PROVIDERS and (s.provider == "fake" or bool(s.key()))
        r = JSONResponse({"enabled": ready, "model": s.model if ready else "",
                          "limits": {"per_second": s.per_window / s.window_seconds, "at_once": s.max_running, "dollars_per_day": s.ip_dollars_per_day},
                          "today": limiter.today(client_ip(request))})
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
            raw = await request.body()
            if len(raw) > MAX_BODY:  # a real fact sheet with a long conversation stays far below this
                raise HTTPException(413, "the request is too large")
            try:
                body = json.loads(raw)
            except ValueError:
                raise HTTPException(400, "send the fact sheet as JSON")
            system, messages, effort = messages_of(body)  # checked before it counts against the limits
            ip = client_ip(request)
            refused = limiter.take(ip)
            if refused:
                raise HTTPException(429, refused)
        except HTTPException as e:  # with the CORS headers, so the page can show why
            return JSONResponse({"detail": e.detail}, status_code=e.status_code, headers=allowed)
        write = PROVIDERS[s.provider]

        cost: list[float] = []

        def stream() -> Iterator[str]:
            try:
                yield from write(s, system, messages, lambda c: cost.append(c), effort)
            except Exception as e:  # the browser shows what went wrong; the key never appears in it
                yield f"\n\n[report service error: {e}]"
            finally:  # also when the visitor stops it: the answer so far was paid for
                limiter.done(ip, sum(cost) if cost else s.fallback_cost)

        return StreamingResponse(stream(), media_type="text/plain; charset=utf-8", headers=allowed)
