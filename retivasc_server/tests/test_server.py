"""The server end to end with a stand-in for AutoMorph: sign-in, ownership, uploads, the queue,
per-photo results, restarts and clean-up."""

from __future__ import annotations

import csv
import io
import json
import random
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from retivasc_server.app import create_app
from retivasc_server.automorph import FEATURES, load_manifest
from retivasc_server.settings import Settings


def fake_automorph(data_root: Path, device: str = "cuda") -> None:
    """Writes what AutoMorph v2 writes, for every staged photo."""
    rng = random.Random(1)
    res = data_root / "Results"
    for sub in ("M0/images", "M1", "M2/binary_vessel/raw_binary", "M3"):
        (res / sub).mkdir(parents=True, exist_ok=True)
    files = load_manifest(data_root)["files"]
    with open(res / "M1" / "M1_quality_final.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Name", "softmax_good", "softmax_usable", "softmax_bad", "good_sd", "usable_sd", "bad_sd", "Prediction", "Gradable"])
        for e in files:
            w.writerow([f"x/{e['stem']}.png", 0.9, 0.05, 0.05, 0, 0, 0, 0, True])
    for table in ("Macular_Features.csv", "Disc_Features.csv"):
        with open(res / "M3" / table, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Name", *FEATURES])
            for e in files:
                if ("ON" in e["stem"]) == (table == "Disc_Features.csv"):
                    w.writerow([f"x/{e['stem']}.png", *[repr(rng.random()) for _ in FEATURES]])
    for e in files:
        Image.new("RGB", (8, 8)).save(res / "M0" / "images" / f"{e['stem']}.png")
        Image.new("L", (8, 8)).save(res / "M2" / "binary_vessel" / "raw_binary" / f"{e['stem']}.png")
    (data_root / "retivasc_automorph.log").write_text("[M3 PROGRESS] M3 features: 1/1 images (100.0%)\n", encoding="utf-8")
    (data_root / "run.log").write_text("[DONE] run completed: completed=m0, m1, m2, m3; skipped=none; wall=1s\n", encoding="utf-8")


def photo_bytes(seed: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (20, 16), (seed * 40 % 255, 90, 30)).save(buf, format="TIFF")
    return buf.getvalue()


@pytest.fixture()
def server(tmp_path):
    s = Settings(data_dir=tmp_path / "data", allowed_folders=[tmp_path / "shared"])
    app = create_app(s, runner=fake_automorph, start_worker=False)
    app.state.accounts.add("alice", "password-a", admin=False)
    app.state.accounts.add("bob", "password-b", admin=False)
    app.state.accounts.add("root", "password-r", admin=True)
    return app, tmp_path


def client(app, name=None, password=None) -> TestClient:
    c = TestClient(app)
    if name:
        assert c.post("/api/login", json={"name": name, "password": password}).status_code == 200
    return c


def run_queue(app):
    store = app.state.store
    while store.queue:
        store.run_one(store.queue.popleft())


def test_sign_in_is_required(server):
    app, _ = server
    c = client(app)
    assert c.get("/api/jobs").status_code == 401
    assert c.post("/api/login", json={"name": "alice", "password": "nope-nope"}).status_code == 401
    c = client(app, "alice", "password-a")
    assert c.get("/api/me").json() == {"name": "alice", "admin": False, "mode": "accounts"}
    c.post("/api/logout")
    assert c.get("/api/me").status_code == 401


def test_analyze_photos_and_ownership(server):
    app, _ = server
    a = client(app, "alice", "password-a")
    files = [("files", (f"pt{i}_right_macula.tif", photo_bytes(i), "image/tiff")) for i in range(2)]
    files.append(("files", ("notes.txt", b"x", "text/plain")))  # ignored
    job = a.post("/api/jobs/analyze", files=files).json()
    assert job["status"] == "queued" and "queue" in job["progress"]["step"]
    run_queue(app)
    done = a.get(f"/api/jobs/{job['id']}").json()
    assert done["status"] == "done", done["error"]
    photos = done["result"]["photos"]
    assert len(photos) == 2 and photos[0]["quality"]["grade"] == "Good" and photos[0]["centred_on"] == "macula"
    assert len(photos[0]["features"]) == 72
    img = a.get(f"/api/jobs/{job['id']}/image/{photos[0]['images']['vessels']}")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    assert a.get(f"/api/jobs/{job['id']}/file/features.csv").status_code == 200
    # another user cannot see it; an admin can
    b = client(app, "bob", "password-b")
    assert b.get(f"/api/jobs/{job['id']}").status_code == 404
    assert b.get("/api/jobs").json() == []
    assert client(app, "root", "password-r").get(f"/api/jobs/{job['id']}").status_code == 200
    # paths cannot escape the job folder
    assert a.get(f"/api/jobs/{job['id']}/image/../../job.json").status_code == 404
    assert a.get(f"/api/jobs/{job['id']}/file/..%2Fjob.json").status_code == 404
    assert a.delete(f"/api/jobs/{job['id']}").json() == {"ok": True}
    assert a.get(f"/api/jobs/{job['id']}").status_code == 404



def test_rejects_unsafe_upload_names(server):
    app, _ = server
    a = client(app, "alice", "password-a")
    r = a.post("/api/jobs/analyze", files=[("files", ("../evil/1_right_macula.tif", photo_bytes(1)))])
    assert r.status_code == 400
    assert app.state.store.all() == []  # nothing half-made is left


def test_study_sheet_is_optional(server, monkeypatch):
    from retivasc_server import app as app_module

    app, _ = server
    a = client(app, "alice", "password-a")
    assert a.get("/api/server").json()["features"] == {"sheet": app_module.SHEET}
    monkeypatch.setattr(app_module, "SHEET", False)
    r = a.post("/api/jobs/sheet", data={"out_name": "x.xlsx"}, files=[("template", ("t.xlsx", b"x"))])
    assert r.status_code == 404


def test_restart_marks_running_job_failed_and_keeps_queue(server):
    app, tmp = server
    store = app.state.store
    a = client(app, "alice", "password-a")
    j1 = a.post("/api/jobs/analyze", files=[("files", ("1_right_macula.tif", photo_bytes(1)))]).json()
    j2 = a.post("/api/jobs/analyze", files=[("files", ("2_right_macula.tif", photo_bytes(2)))]).json()
    job = store.get(j1["id"])
    job.status = "running"
    store.save(job)
    store.queue.popleft()
    from retivasc_server.jobs import JobStore

    again = JobStore(store.dir)
    assert again.get(j1["id"]).status == "failed"
    assert list(again.queue) == [j2["id"]]


def test_cleanup_removes_old_finished_jobs(server):
    app, _ = server
    store = app.state.store
    a = client(app, "alice", "password-a")
    j = a.post("/api/jobs/analyze", files=[("files", ("1_right_macula.tif", photo_bytes(1)))]).json()
    run_queue(app)
    job = store.get(j["id"])
    job.finished = "2000-01-01T00:00:00+00:00"
    store.save(job)
    assert store.cleanup(retention_days=30) == 1 and store.all() == []


def test_access_log_has_no_file_names(server):
    app, tmp = server
    a = client(app, "alice", "password-a")
    a.post("/api/jobs/analyze", files=[("files", ("1234_right_macula.tif", photo_bytes(1)))])
    text = (tmp / "data" / "access.log").read_text(encoding="utf-8")
    assert "alice\tanalyze" in text and "1234" not in text


def test_open_mode_private_guest_sessions(tmp_path):
    s = Settings(data_dir=tmp_path / "data", auth="open")
    app = create_app(s, runner=fake_automorph, start_worker=False)
    one, two = TestClient(app), TestClient(app)  # two browsers, nobody signs in
    me1, me2 = one.get("/api/me").json(), two.get("/api/me").json()
    assert me1["mode"] == "open" and me1["name"].startswith("guest-") and me1["name"] != me2["name"]
    assert one.get("/api/me").json()["name"] == me1["name"]  # the session sticks
    job = one.post("/api/jobs/analyze", files=[("files", ("1_right_macula.tif", photo_bytes(1)))]).json()
    run_queue(app)
    assert one.get(f"/api/jobs/{job['id']}").json()["status"] == "done"
    assert two.get(f"/api/jobs/{job['id']}").status_code == 404 and two.get("/api/jobs").json() == []
    # a forged or foreign token does not open someone else's jobs
    two.cookies.set("retivasc_session", "Z3Vlc3QtMDAwMHwxfHg=")
    assert two.get(f"/api/jobs/{job['id']}").status_code == 404
    # one browser cannot fill the queue
    for i in range(3):
        assert one.post("/api/jobs/analyze", files=[("files", (f"{i}_right_macula.tif", photo_bytes(i)))]).status_code == 200
    assert one.post("/api/jobs/analyze", files=[("files", ("x_right_macula.tif", photo_bytes(9)))]).status_code == 429


def test_accounts_mode_rejects_guest_tokens(tmp_path):
    open_app = create_app(Settings(data_dir=tmp_path / "data", auth="open"), runner=fake_automorph, start_worker=False)
    guest = TestClient(open_app)
    guest.get("/api/me")
    token = guest.cookies.get("retivasc_session")
    closed = TestClient(create_app(Settings(data_dir=tmp_path / "data"), runner=fake_automorph, start_worker=False))
    closed.cookies.set("retivasc_session", token)
    assert closed.get("/api/me").status_code == 401  # switching back to accounts locks guests out


def test_mode_switch_command(tmp_path, monkeypatch):
    from retivasc_server import __main__ as cli
    from retivasc_server import config

    monkeypatch.setattr(config, "FILE", tmp_path / "retivasc.local.json")
    assert cli.main(["mode", "open"]) == 0
    assert json.loads((tmp_path / "retivasc.local.json").read_text())["server"]["auth"] == "open"
    assert cli.main(["mode", "accounts"]) == 0
    assert json.loads((tmp_path / "retivasc.local.json").read_text())["server"]["auth"] == "accounts"


def test_automorph_crash_is_retried_once(server):
    app, _ = server
    store = app.state.store
    calls = []

    def flaky(data_root, device="cuda"):
        calls.append(data_root)
        if len(calls) == 1:
            raise RuntimeError("AutoMorph v2 failed (exit 3221225477)")
        fake_automorph(data_root, device)

    store.runner = flaky
    a = client(app, "alice", "password-a")
    j = a.post("/api/jobs/analyze", files=[("files", ("1_right_macula.tif", photo_bytes(1)))]).json()
    run_queue(app)
    assert a.get(f"/api/jobs/{j['id']}").json()["status"] == "done" and len(calls) == 2
    # a second failure is reported, not retried for ever
    store.runner = lambda *x: (_ for _ in ()).throw(RuntimeError("AutoMorph v2 failed (exit 1)"))
    j = a.post("/api/jobs/analyze", files=[("files", ("2_right_macula.tif", photo_bytes(2)))]).json()
    run_queue(app)
    assert a.get(f"/api/jobs/{j['id']}").json()["status"] == "failed"
