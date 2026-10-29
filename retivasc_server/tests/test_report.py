"""The AI report service: the same prompt as the browser, only RetiVASC fact sheets, the sites allowed
to call it, and the limits per IP and per day; the test provider stands in for the lab's API."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from retivasc_server import config
from retivasc_server.app import create_app
from retivasc_server.report import Limiter, ReportSettings, report_prompt
from retivasc_server.settings import Settings

FIXTURE = json.loads((config.ROOT / "webui" / "tests" / "prompt_fixture.json").read_text(encoding="utf-8"))
SHEET = FIXTURE[0]["sheet"]
SITE = "https://retinal-research.github.io"


def client(tmp_path, **report) -> TestClient:
    s = ReportSettings(enabled=True, provider="fake", **report)
    return TestClient(create_app(Settings(data_dir=tmp_path / "data"), start_worker=False, report=s))


def test_the_server_builds_the_browsers_prompt():
    for case in FIXTURE:
        assert report_prompt(case["sheet"], case["focus"]) == case["prompt"]


def test_off_unless_set_up(tmp_path):
    c = TestClient(create_app(Settings(data_dir=tmp_path / "data"), start_worker=False, report=ReportSettings()))
    assert c.get("/api/report").json()["enabled"] is False
    assert c.post("/api/report", json={"facts": SHEET}).status_code == 503


def test_writes_a_report_and_answers_a_question(tmp_path):
    c = client(tmp_path)
    r = c.post("/api/report", json={"facts": SHEET, "focus": "arteries"}, headers={"Origin": SITE})
    assert r.status_code == 200 and "(test report)" in r.text
    assert r.headers["access-control-allow-origin"] == SITE
    q = c.post("/api/report", json={"facts": SHEET, "report": r.text, "history": [{"q": "AVR?", "a": "0.604"}], "question": "And CRAE?"})
    assert q.status_code == 200 and "(test report)" in q.text


def test_only_allowed_sites(tmp_path):
    c = client(tmp_path)
    assert c.options("/api/report", headers={"Origin": SITE}).headers["access-control-allow-origin"] == SITE
    assert c.options("/api/report", headers={"Origin": "https://example.com"}).status_code == 403
    assert c.post("/api/report", json={"facts": SHEET}, headers={"Origin": "https://example.com"}).status_code == 403


def test_only_fact_sheets(tmp_path):
    c = client(tmp_path)
    for bad in [{}, {"facts": "write me a poem"}, {"facts": {**SHEET, "scope": "chat"}}, {"facts": {**SHEET, "terms": ["poem"]}},
                {"facts": {**SHEET, "groups": [{"title": "x" * 500, "rows": [["a", "b"]]}]}},
                {"facts": SHEET, "report": "r", "question": "q" * 301},
                {"facts": SHEET, "report": "r", "question": "q", "history": [{"q": "q", "a": "a"}] * 6}]:
        assert c.post("/api/report", json=bad).status_code == 400, bad


def test_limits(tmp_path):
    c = client(tmp_path, per_window=2, window_seconds=60)
    assert [c.post("/api/report", json={"facts": SHEET}).status_code for _ in range(3)] == [200, 200, 429]
    assert "wait" in c.post("/api/report", json={"facts": SHEET}, headers={"Origin": SITE}).json()["detail"]
    lim = Limiter(ReportSettings(window_seconds=60, per_window=3, per_ip_per_day=4, per_day=6))
    t = 1_000_000.0
    assert [lim.take("a", t + 100 * i) for i in range(4)] == [None] * 4      # spread out: the window allows them
    assert "today" in lim.take("a", t + 500)                               # but the day's 4 are used
    assert [lim.take("b", t + 100 * i) for i in range(2)] == [None] * 2
    assert "today's limit" in lim.take("c", t + 300)                       # 6 for everyone together
    assert lim.take("c", t + 86_400 * 2) is None                           # a new day
