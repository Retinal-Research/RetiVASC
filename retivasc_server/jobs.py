"""Job queue: each job is a folder under <data_dir>/jobs/<id>/ and runs on this machine's GPU,
one job at a time, in the order submitted.

  job.json      what was asked, by whom, current status and result summary
  input/        uploaded photos (and the sheet template for a sheet job)
  automorph/    the staged AutoMorph v2 run (images/, Results/, logs)
  out/          files for download (Excel sheet, change log, audit report, …)
  log.txt       the job's own log

Kinds:
  analyze  photos -> full AutoMorph v2 (M0–M3) -> per-photo quality, images and 72 features
  sheet    a photo folder (+ a sheet template) -> AutoMorph v2 -> filled sheet + audit;
           it may reuse the run of an earlier finished job instead of running AutoMorph again
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import automorph, devices
from .automorph import FEATURES, load_manifest, stage

# AutoMorph log lines look like "[M2 PROGRESS] M2 artery-vein-inference-write: 50/1238 images (4.0%) | …"
PROGRESS = re.compile(r"\[(M\d) PROGRESS\] (.*?): (\d+)/(\d+)")
STAGE_SPAN = {"M0": (0.0, 0.08), "M1": (0.08, 0.25), "M2": (0.25, 0.85), "M3": (0.85, 1.0)}
STAGE_NAME = {"M0": "Preparing photos", "M1": "Grading image quality", "M2": "Segmenting vessels, arteries/veins and disc",
              "M3": "Measuring 72 vascular features"}
IMAGES = {  # what the web UI shows per photo: Results/<path>/<stem>.png
    "photo": "M0/images",
    "vessels": "M2/binary_vessel/raw_binary",
    "artery_vein": "M2/artery_vein/raw",
    "disc": "M2/optic_disc_cup/raw",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    owner: str
    kind: str
    title: str
    created: str
    params: dict = field(default_factory=dict)
    status: str = "queued"  # queued | running | done | failed | cancelled
    started: str | None = None
    finished: str | None = None
    error: str | None = None
    result: dict = field(default_factory=dict)


class JobStore:
    def __init__(self, jobs_dir: Path, device: str = "auto", runner=None):
        self.dir = jobs_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.runner = runner or automorph.run  # replaced in tests, which then skip probing devices
        self.resolve = devices.resolve if runner is None else (lambda d: d)
        self.lock = threading.Lock()
        self.queue: deque[str] = deque()
        self.wake = threading.Event()
        self.current: str | None = None
        # a restart interrupts the running job; queued jobs simply wait again
        for job in self.all():
            if job.status == "running":
                job.status, job.error, job.finished = "failed", "the server restarted while this job was running", _now()
                self.save(job)
            elif job.status == "queued":
                self.queue.append(job.id)

    # -- storage
    def path(self, job_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{12}", job_id):
            raise KeyError(job_id)
        return self.dir / job_id

    def save(self, job: Job) -> None:
        p = self.path(job.id) / "job.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(job), indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)

    def get(self, job_id: str) -> Job:
        p = self.path(job_id) / "job.json"
        if not p.exists():
            raise KeyError(job_id)
        return Job(**json.loads(p.read_text(encoding="utf-8")))

    def all(self) -> list[Job]:
        jobs = []
        for p in self.dir.glob("*/job.json"):
            try:
                jobs.append(Job(**json.loads(p.read_text(encoding="utf-8"))))
            except (ValueError, TypeError):
                continue
        return sorted(jobs, key=lambda j: j.created, reverse=True)

    def create(self, owner: str, kind: str, title: str, params: dict) -> Job:
        job = Job(uuid.uuid4().hex[:12], owner, kind, title, _now(), params)
        (self.path(job.id) / "input").mkdir(parents=True)
        self.save(job)
        return job

    def submit(self, job: Job) -> None:
        with self.lock:
            self.queue.append(job.id)
        self.wake.set()

    def position(self, job_id: str) -> int | None:
        with self.lock:
            return list(self.queue).index(job_id) + 1 if job_id in self.queue else None

    def cancel(self, job_id: str) -> bool:
        with self.lock:
            if job_id not in self.queue:
                return False
            self.queue.remove(job_id)
        job = self.get(job_id)
        job.status, job.finished = "cancelled", _now()
        self.save(job)
        return True

    def delete(self, job_id: str) -> None:
        job = self.get(job_id)
        if job.status == "running":
            raise RuntimeError("a running job cannot be deleted")
        self.cancel(job_id)
        shutil.rmtree(self.path(job_id))

    def cleanup(self, retention_days: int) -> int:
        cutoff = time.time() - retention_days * 86400
        n = 0
        for job in self.all():
            if job.status in ("done", "failed", "cancelled"):
                when = datetime.fromisoformat(job.finished or job.created).timestamp()
                if when < cutoff:
                    shutil.rmtree(self.path(job.id), ignore_errors=True)
                    n += 1
        return n

    # -- progress
    def progress(self, job: Job) -> dict:
        if job.status == "queued":
            pos = self.position(job.id)
            return {"step": f"Waiting in the queue (position {pos})" if pos else "Starting", "fraction": 0.0}
        if job.status != "running":
            return {"step": job.status, "fraction": 1.0 if job.status == "done" else None}
        step, fraction = "Starting", 0.01
        log = self.path(job.id) / "automorph" / automorph.LOG
        if log.exists():
            with open(log, "rb") as f:
                f.seek(max(0, log.stat().st_size - 8000))
                tail = f.read().decode("utf-8", errors="replace")
            hits = PROGRESS.findall(tail)
            if hits:
                stage_id, _, done, total = hits[-1]
                lo, hi = STAGE_SPAN[stage_id]
                step = STAGE_NAME[stage_id]
                fraction = lo + (hi - lo) * (int(done) / max(1, int(total)))
        if (self.path(job.id) / "sheet.started").exists():
            step, fraction = "Filling and auditing the sheet", 0.97
        return {"step": step, "fraction": round(fraction, 3)}

    # -- the worker
    def worker(self) -> None:
        while True:
            self.wake.wait(timeout=5)
            self.wake.clear()
            while True:
                with self.lock:
                    if not self.queue:
                        break
                    job_id = self.queue.popleft()
                    self.current = job_id
                try:
                    self.run_one(job_id)
                finally:
                    with self.lock:
                        self.current = None

    def run_one(self, job_id: str) -> None:
        job = self.get(job_id)
        job.status, job.started = "running", _now()
        self.save(job)
        jdir = self.path(job_id)
        try:
            with open(jdir / "log.txt", "a", encoding="utf-8") as log:
                if job.kind == "analyze":
                    job.result = self._analyze(job, jdir, log)
                elif job.kind == "sheet":
                    job.result = self._sheet(job, jdir, log)
                else:
                    raise ValueError(f"unknown job kind {job.kind}")
            job.status = "done"
        except Exception as e:  # the job fails, the server carries on
            job.status, job.error = "failed", f"{type(e).__name__}: {e}"
            with open(jdir / "log.txt", "a", encoding="utf-8") as log:
                log.write(traceback.format_exc())
        job.finished = _now()
        self.save(job)

    def _run_automorph(self, job: Job, images_root: Path, data_root: Path, log, stager=stage) -> dict:
        manifest = stager(images_root, data_root)
        device = self.resolve(job.params.get("device") or self.device)
        job.params["ran_on"] = device
        self.save(job)
        log.write(f"{_now()} staged {len(manifest['files'])} photos; running AutoMorph v2 on {device}\n")
        log.flush()
        try:
            self.runner(data_root, device)
        except RuntimeError as e:
            # an AutoMorph stage occasionally dies without a Python error (seen once: M2 exited
            # with an access violation while the GPU was shared); run it again once, from scratch
            log.write(f"{_now()} AutoMorph v2 stopped unexpectedly ({e}); running it once more\n")
            log.flush()
            data_root.rename(data_root.with_name(f"{data_root.name}_failed_{int(time.time())}"))
            manifest = stager(images_root, data_root)
            self.runner(data_root, device)
        log.write(f"{_now()} AutoMorph v2 finished\n")
        return manifest

    def _analyze(self, job: Job, jdir: Path, log) -> dict:
        data_root = jdir / "automorph"
        manifest = self._run_automorph(job, jdir / "input", data_root, log)
        outputs = automorph.load(data_root)
        photos = []
        for e in manifest["files"]:
            stem = e["stem"]
            q = outputs.quality.get(stem)
            table, values = outputs.features.get(stem, (None, None))
            images = {k: f"{sub}/{stem}.png" for k, sub in IMAGES.items() if (data_root / "Results" / sub / f"{stem}.png").exists()}
            photos.append({
                "name": e["rel_path"], "stem": stem,
                "quality": None if q is None else {"good": q.p_good, "usable": q.p_usable, "reject": q.p_reject,
                                                   "grade": ["Good", "Usable", "Reject"][q.prediction], "gradable": q.gradable},
                "centred_on": None if table is None else ("macula" if table.startswith("Macular") else "optic disc"),
                "features": values, "images": images,
            })
        (jdir / "out").mkdir(exist_ok=True)
        _features_csv(jdir / "out" / "features.csv", photos, FEATURES)
        return {"photos": photos, "files": ["features.csv"]}

    def _sheet(self, job: Job, jdir: Path, log) -> dict:
        # the study-sheet module is optional and installed separately (see SHEET in app.py)
        from retivasc_sheet.__main__ import main as sheet_main
        from retivasc_sheet.inventory import scan
        from retivasc_sheet.serve import _summary
        from retivasc_sheet.staging import load_manifest as sheet_manifest
        from retivasc_sheet.staging import stage as sheet_stage

        p = job.params
        images_root = Path(p["images_root"])
        if p.get("reuse_job"):
            src = self.get(p["reuse_job"])
            if src.status != "done":
                raise RuntimeError("the run to reuse has not finished")
            data_root = self.path(src.id) / "automorph"
            log.write(f"{_now()} reusing the AutoMorph run of job {src.id}\n")
        else:
            data_root = jdir / "automorph"
            self._run_automorph(job, images_root, data_root, log, lambda src, dst: sheet_stage(scan(src), dst))
        (jdir / "sheet.started").touch()
        out = jdir / "out" / p["out_name"]
        argv = ["build", "--images", str(images_root), "--template", p["template"], "--data-root", str(data_root), "--out", str(out)]
        if p.get("previous"):
            argv += ["--previous", p["previous"]]
        if p.get("strict"):
            argv.append("--no-selection-rule")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = sheet_main(argv)
        text = buf.getvalue()
        log.write(text)
        files = sorted((f.name for f in (jdir / "out").iterdir() if f.is_file()), key=_download_order)
        return {"deliverable": code == 0, "summary": _summary(text, out), "files": files,
                "photos_in_run": len(sheet_manifest(data_root)["files"])}


def _download_order(name: str) -> tuple[int, str]:
    """The sheet first, then the reviewer's copy, then the records."""
    for rank, end in enumerate(("_for_doctor.xlsx", ".changes.csv", "_audit.md", "_audit.json", ".provenance.json")):
        if name.endswith(end):
            return rank + 1, name
    return (0 if name.endswith(".xlsx") else 9), name


def _features_csv(path: Path, photos: list[dict], features: list[str]) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["photo", "grade", "p_good", "p_usable", "p_reject", "gradable", "centred_on", *features])
        for ph in photos:
            q = ph["quality"] or {}
            vals = ph["features"] or {}
            w.writerow([ph["name"], q.get("grade", ""), q.get("good", ""), q.get("usable", ""), q.get("reject", ""),
                        q.get("gradable", ""), ph["centred_on"] or "", *[repr(vals[k]) if isinstance(vals.get(k), float) else (vals.get(k) if vals.get(k) is not None else "") for k in features]])
