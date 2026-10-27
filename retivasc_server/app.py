"""HTTP API and the built web UI, in one FastAPI app.

  POST   /api/login                 {name, password} -> session cookie
  POST   /api/logout
  GET    /api/me
  GET    /api/jobs                  your jobs (all jobs for an admin)
  POST   /api/jobs/analyze          multipart: files[] (photos), device (optional; "auto" = the best GPU)
  POST   /api/jobs/sheet            only with the optional study-sheet module (retivasc_sheet):
                                    files[] (a photo folder, paths kept) or server_folder;
                                    template (xlsx), previous (xlsx, optional), out_name, strict, reuse_job, device
  GET    /api/jobs/{id}             status, progress, result
  GET    /api/jobs/{id}/image/{p}   a result image (Results/<p>)
  GET    /api/jobs/{id}/file/{name} a download from out/
  DELETE /api/jobs/{id}             cancel if queued, then delete
  GET    /api/server                queue length, folders users may analyse in place, inference devices
"""

from __future__ import annotations

import importlib.util
import json
import mimetypes
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, devices
from .accounts import Accounts
from .jobs import JobStore
from .settings import Settings

# Windows takes these from the registry and may get them wrong (.mjs as text/plain), and
# browsers refuse a module script with the wrong type
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".onnx")

COOKIE = "retivasc_session"
CHUNK = 1 << 20
PHOTO_TYPES = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}
# a study's results sheet is filled by a separate, study-specific module that is not part of
# the public repository; without it the server offers single-photo analysis only
SHEET = importlib.util.find_spec("retivasc_sheet") is not None


def create_app(settings: Settings, runner=None, start_worker: bool = True, webui_dist: Path | None = None) -> FastAPI:
    app = FastAPI(title="RetiVASC", version=__version__, docs_url=None, redoc_url=None)
    accounts = Accounts(settings.data_dir)
    store = JobStore(settings.jobs_dir, settings.device, runner)
    access_log = settings.data_dir / "access.log"
    app.state.store, app.state.accounts = store, accounts

    def audit(user: str, action: str, job_id: str = "") -> None:
        # who did what and when; never file names, participant IDs or dates
        with open(access_log, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\t{user}\t{action}\t{job_id}\n")

    if start_worker:
        threading.Thread(target=store.worker, daemon=True, name="jobs").start()
        threading.Thread(target=devices.probe, daemon=True, name="devices").start()  # the list users choose from

        def janitor():
            while True:
                store.cleanup(settings.retention_days)
                time.sleep(3600)

        threading.Thread(target=janitor, daemon=True, name="cleanup").start()

    @app.middleware("http")
    async def headers(request: Request, call_next):
        response = await call_next(request)
        # cross-origin isolation lets the in-browser CPU fallback use several threads
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    open_mode = settings.auth == "open"

    def user(request: Request, response: Response) -> str:
        name = accounts.user_of(request.cookies.get(COOKIE), guests=open_mode)
        if name:
            return name
        if not open_mode:
            raise HTTPException(401, "sign in first")
        # open (public beta) mode: this browser gets its own private, anonymous session
        name, token = accounts.guest_token(settings.retention_days)
        response.set_cookie(COOKIE, token, httponly=True, samesite="strict", secure=bool(settings.ssl_certfile),
                            max_age=settings.retention_days * 86400)
        audit(name, "new guest session")
        return name

    def queue_limit(name: str) -> None:
        # in open mode nobody signs in, so keep one browser from filling the queue
        if open_mode and sum(1 for j in store.all() if j.owner == name and j.status in ("queued", "running")) >= 3:
            raise HTTPException(429, "you already have 3 jobs waiting; please wait for them to finish")

    def owned(job_id: str, name: str):
        try:
            job = store.get(job_id)
        except KeyError:
            raise HTTPException(404, "no such job")
        if job.owner != name and not accounts.is_admin(name):
            raise HTTPException(404, "no such job")
        return job

    # -- sessions
    @app.post("/api/login")
    async def login(request: Request, response: Response):
        body = await request.json()
        u = accounts.check(str(body.get("name", "")), str(body.get("password", "")))
        if not u:
            time.sleep(1)  # slow down guessing
            raise HTTPException(401, "wrong user name or password")
        response.set_cookie(COOKIE, accounts.token(u["name"]), httponly=True, samesite="strict",
                            secure=bool(settings.ssl_certfile), max_age=12 * 3600)
        audit(u["name"], "login")
        return u

    @app.post("/api/logout")
    def logout(response: Response, name: str = Depends(user)):
        response.delete_cookie(COOKIE)
        audit(name, "logout")
        return {"ok": True}

    @app.get("/api/me")
    def me(name: str = Depends(user)):
        return {"name": name, "admin": accounts.is_admin(name), "mode": settings.auth}

    @app.get("/api/server")
    def server(name: str = Depends(user)):
        return {"version": __version__, "mode": settings.auth, "queued": len(store.queue), "running": store.current is not None,
                "allowed_folders": [str(p) for p in settings.allowed_folders], "retention_days": settings.retention_days,
                "device": settings.device, "devices": devices.cached(), "features": {"sheet": SHEET}}

    # -- jobs
    def view(job) -> dict:
        d = {k: getattr(job, k) for k in ("id", "owner", "kind", "title", "created", "status", "started", "finished", "error")}
        d["progress"] = store.progress(job)
        return d

    @app.get("/api/jobs")
    def jobs(name: str = Depends(user)):
        admin = accounts.is_admin(name)
        return [view(j) for j in store.all() if admin or j.owner == name]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str, name: str = Depends(user)):
        j = owned(job_id, name)
        return {**view(j), "params": {k: v for k, v in j.params.items() if k not in ("template", "previous", "images_root")},
                "result": j.result}

    async def save_uploads(files: list[UploadFile], dest: Path, keep_paths: bool) -> int:
        n = 0
        for f in files:
            rel = PurePosixPath((f.filename or "").replace("\\", "/"))
            if rel.is_absolute() or ".." in rel.parts or not rel.name:
                raise HTTPException(400, "bad file name in the upload")
            if not keep_paths:
                rel = PurePosixPath(rel.name)
            target = dest.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as out:
                while chunk := await f.read(CHUNK):
                    out.write(chunk)
            n += 1
        return n

    def checked_device(device: str) -> dict:
        if not devices.valid(device):
            raise HTTPException(400, f"no device {device!r} on this server")
        return {"device": device} if device not in ("", "auto") else {}

    @app.post("/api/jobs/analyze")
    async def analyze(files: list[UploadFile] = File(...), device: str = Form(""), name: str = Depends(user)):
        photos = [f for f in files if Path(f.filename or "").suffix.lower() in PHOTO_TYPES]
        if not photos:
            raise HTTPException(400, "no photos (tif, jpg, png, bmp) in the upload")
        queue_limit(name)
        j = store.create(name, "analyze", f"{len(photos)} photo(s)", checked_device(device))
        try:
            await save_uploads(photos, store.path(j.id) / "input" / "photos", keep_paths=False)
        except Exception:
            shutil.rmtree(store.path(j.id), ignore_errors=True)  # no half-made job is left behind
            raise
        store.submit(j)
        audit(name, "analyze", j.id)
        return view(j)

    @app.post("/api/jobs/sheet")
    async def sheet(template: UploadFile = File(...), previous: UploadFile | None = File(None),
                    files: list[UploadFile] | None = File(None), server_folder: str = Form(""),
                    out_name: str = Form(...), strict: bool = Form(False), reuse_job: str = Form(""),
                    device: str = Form(""), name: str = Depends(user)):
        if not SHEET:
            raise HTTPException(404, "the study-sheet module is not installed on this server")
        if not out_name.endswith(".xlsx") or "/" in out_name or "\\" in out_name:
            raise HTTPException(400, "the output name must be a plain .xlsx file name")
        queue_limit(name)
        j = store.create(name, "sheet", out_name, {})
        try:
            return await _fill_sheet_job(j, template, previous, files, server_folder, out_name, strict, reuse_job,
                                         checked_device(device), name)
        except Exception:
            shutil.rmtree(store.path(j.id), ignore_errors=True)  # no half-made job is left behind
            raise

    async def _fill_sheet_job(j, template, previous, files, server_folder, out_name, strict, reuse_job, device, name):
        jdir = store.path(j.id)
        if reuse_job:
            src = owned(reuse_job, name)
            if src.kind != "sheet" or src.status != "done":
                raise HTTPException(400, "only a finished study-sheet job can be reused")
            images_root = Path(src.params["images_root"])
        elif server_folder:
            folder = Path(server_folder).resolve()
            if not any(folder == a.resolve() or folder.is_relative_to(a.resolve()) for a in settings.allowed_folders):
                raise HTTPException(403, "this folder is not on the server's list of allowed folders")
            if not folder.is_dir():
                raise HTTPException(400, "no such folder on the server")
            images_root = folder
        elif files:
            n = await save_uploads(files, jdir / "input", keep_paths=True)
            tops = {PurePosixPath((f.filename or "").replace("\\", "/")).parts[0] for f in files}
            if len(tops) != 1:
                raise HTTPException(400, "upload one folder (the photo folder itself)")
            images_root = jdir / "input" / tops.pop()
        else:
            raise HTTPException(400, "choose a photo folder: upload it or name a server folder")
        params = {"images_root": str(images_root), "out_name": out_name, "strict": strict, "reuse_job": reuse_job, **device}
        for key, f in (("template", template), ("previous", previous)):
            if f is not None and f.filename:
                (jdir / "sheets").mkdir(exist_ok=True)
                dest = jdir / "sheets" / f"{key}.xlsx"
                with open(dest, "wb") as out:
                    while chunk := await f.read(CHUNK):
                        out.write(chunk)
                params[key] = str(dest)
        j.params = params
        store.save(j)
        store.submit(j)
        audit(name, "sheet", j.id)
        return view(j)

    @app.get("/api/jobs/{job_id}/image/{path:path}")
    def image(job_id: str, path: str, w: int = 0, name: str = Depends(user)):
        owned(job_id, name)
        rel = PurePosixPath(path)
        if ".." in rel.parts or rel.suffix.lower() != ".png":
            raise HTTPException(404, "not found")
        p = store.path(job_id) / "automorph" / "Results" / Path(*rel.parts)
        if not p.is_file():
            raise HTTPException(404, "not found")
        if w <= 0:
            return FileResponse(p, media_type="image/png")
        w = min(max(w, 64), 2048)
        thumb = store.path(job_id) / "thumbs" / str(w) / Path(*rel.parts)
        if not thumb.exists():  # made once, kept with the job
            from PIL import Image

            thumb.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(p) as im:
                im.thumbnail((w, w), Image.NEAREST if im.mode in ("1", "L", "P") else Image.LANCZOS)
                im.save(thumb)
        return FileResponse(thumb, media_type="image/png")

    @app.get("/api/jobs/{job_id}/file/{fname}")
    def download(job_id: str, fname: str, name: str = Depends(user)):
        owned(job_id, name)
        p = store.path(job_id) / "out" / fname
        if "/" in fname or "\\" in fname or not p.is_file():
            raise HTTPException(404, "not found")
        audit(name, "download", job_id)
        return FileResponse(p, filename=fname)

    @app.delete("/api/jobs/{job_id}")
    def delete(job_id: str, name: str = Depends(user)):
        owned(job_id, name)
        try:
            store.delete(job_id)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        audit(name, "delete", job_id)
        return {"ok": True}

    @app.exception_handler(json.JSONDecodeError)
    async def bad_json(request, exc):
        return JSONResponse({"detail": "bad request"}, status_code=400)

    if webui_dist and (webui_dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=webui_dist, html=True), name="webui")
    return app
