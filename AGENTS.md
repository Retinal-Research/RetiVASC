# RetiVASC: deployment and maintenance manual for agents

This manual is for a coding agent asked to deploy, check or change RetiVASC. Commands are exact; run them from the repository root unless a step says otherwise.

The time budget is about three minutes of commands. Downloads add to it:
- the model weights are 0.85 GB for the browser and 2.5 GB for the server;
- the CUDA build of PyTorch is about 3 GB.

## 0. Decide what to deploy

| The user wants | Deploy | Needs |
|---|---|---|
| To try it, or a demo on one computer | **A. Browser only** | Python 3 (any), Node.js ≥ 20, Chrome/Edge with WebGPU |
| A shared service for a lab, results identical to AutoMorph | **B. Server** | Python 3.12, Node.js ≥ 20, an NVIDIA GPU with a current driver (CPU works, slowly) |
| A server on a Linux host without internet | **C. Docker** | Docker with the NVIDIA Container Toolkit on the host; a machine with internet to build the image |

If unsure, ask the user. B and C serve the web UI from A as well.

Preflight checks:

```bash
git --version
node --version            # v20 or newer
python --version          # B and C: 3.12.x (Windows: py -3.12 --version)
nvidia-smi                # B and C with a GPU: driver present, note free memory (needs about 6 GB)
```

On Windows, clone to a short path such as `C:\RetiVASC`: AutoMorph's model folders are deep, and Windows limits paths to 259 characters.

## A. Browser only (about 1 minute plus a 0.85 GB download)

```bash
python tools/release.py fetch --web     # downloads the browser models into webui/public/models, checks SHA-256
cd webui
npm ci
npm run dev                             # http://localhost:5178 (port is fixed: strictPort)
```

Verify the deployment:
- Open http://localhost:5178. The first workflow, *AutoMorph · full pipeline*, starts by itself on a sample photograph.
- The top bar names the engine. `WebGPU · <adapter>` is right; `CPU only` means WebGPU is unavailable, and the run is slow but correct.
- Every node turns green. *M3 · Features* lists 72 values.

To serve a static build instead: `npm run build` writes `webui/dist`; `npm run preview` serves it on port 5179. Any static web server works, but it must:
- send `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`, which multi-threaded CPU fallback needs;
- serve `.mjs` as `text/javascript` and `.wasm` as `application/wasm`.

## B. Server (about 3 minutes plus downloads)

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1        # -Gpu cpu for a machine without NVIDIA
.venv\Scripts\python -m retivasc_server adduser admin --admin --password "<choose one>"
start.bat
```

### Linux or macOS

```bash
./install.sh nvidia                                           # or: cpu | mac
.venv/bin/python -m retivasc_server adduser admin --admin --password '<choose one>'
.venv/bin/python -m retivasc_server serve
```

### What the installer does

The installer does the following, so you can repeat any step by hand:
1. Creates `.venv` with Python 3.12. Nothing is installed system-wide.
2. Downloads both weight packages: `python tools/release.py fetch`.
3. Installs PyTorch 2.7.1. CUDA 12.8 wheels cover current NVIDIA cards, including the RTX 50 series; there are also CPU and macOS builds. Then it installs `requirements-server.txt`.
4. Writes `retivasc.local.json` from `retivasc.local.example.json`, setting `python`, `gpu_python` and `server.device`.
5. Builds the web UI (`npm ci && npx vite build`) if `webui/dist` is missing.
6. Runs the acceptance self-test.

The self-test must print `SELFTEST PASSED` (exit 0). It runs the complete AutoMorph v2 on the six sample photographs (about 45 s on a GPU) and compares every quality probability and all 72 features with `retivasc_server/reference.json`:
- grades must match exactly;
- values must agree within 1e-3.

If it fails, stop and report the printed differences. Do not hand over the server.

### Verify the running server

```bash
curl -s http://127.0.0.1:8080/api/server -b cookies.txt -c cookies.txt   # 401 in accounts mode until signed in
curl -s -X POST http://127.0.0.1:8080/api/login -H 'Content-Type: application/json' \
     -d '{"name":"admin","password":"<the password>"}' -c cookies.txt
curl -s http://127.0.0.1:8080/api/server -b cookies.txt                 # {"version":..., "device":"cuda", ...}
```

Then open http://localhost:8080, sign in, and run *Full analysis on the server* on a sample photograph. It takes about 50 s and shows a grade, four images and 72 values.

### Access mode

- `accounts` (the default): everyone signs in. Add users with `adduser NAME [--admin] [--password …]`, remove them with `deluser NAME`, list them with `users`.
- `open` (public beta): no sign-in; every browser gets its own private guest session, limited to 3 waiting jobs.
- Switch with `python -m retivasc_server mode open` or `mode accounts`, then restart. Accounts are kept in both modes.

### Opening the server to other computers

By default the server listens on `127.0.0.1` only. Set `"host": "0.0.0.0"` in `retivasc.local.json` and allow the port in the firewall. **This exposes the service to the network: do it only when the owner asks.** For HTTPS, set `ssl_certfile` and `ssl_keyfile`, or put a reverse proxy in front.

## C. Docker (Linux GPU host without internet)

On a machine with internet:

```bash
python tools/release.py fetch                     # weights must be present before the build
docker build -t retivasc:0.1 .
docker save retivasc:0.1 | gzip > retivasc-0.1.tar.gz
```

On the host:

```bash
docker load < retivasc-0.1.tar.gz
docker compose -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml exec retivasc python -m retivasc_server selftest
docker compose -f deploy/docker-compose.yml exec retivasc python -m retivasc_server adduser admin --admin
```

The image uses `deploy/retivasc.local.json`:
- it listens on `0.0.0.0:8080`;
- data goes in the `/data` volume;
- `/photos` is mounted read-only as an allowed folder.

The Docker path has not yet been tested end to end; run the self-test.

## Configuration

Machine-specific settings live in `retivasc.local.json` at the repository root. It is ignored by git, and `retivasc.local.example.json` is the starting point. Environment variables override the file.

| Key | Env variable | Meaning | Default |
|---|---|---|---|
| `gpu_python` | `RETIVASC_GPU_PY` | Python with torch that runs AutoMorph | the Python running the server |
| `gpu_env` | `RETIVASC_GPU_ENV` | Extra environment for AutoMorph, `;`-separated, e.g. `CUDA_DEVICE_ORDER=PCI_BUS_ID;CUDA_VISIBLE_DEVICES=0` to pick a GPU | none |
| `automorph` | `RETIVASC_AUTOMORPH` | AutoMorph v2 folder | `automorph_v2/` |
| `server.auth` | | `accounts` or `open` | `accounts` |
| `server.host` | | `127.0.0.1`, or `0.0.0.0` for the network | `127.0.0.1` |
| `server.port` | | Port of the web UI and API (`serve --port` overrides it) | `8080` |
| `server.data_dir` | | Accounts, jobs, results, `access.log` | `work/` |
| `server.retention_days` | | Finished jobs are deleted after this many days | `30` |
| `server.device` | | `cuda`, `cpu` or `mps` | `cuda` |
| `server.allowed_folders` | | Server folders users may analyse in place | `[]` |
| `server.ssl_certfile`, `server.ssl_keyfile` | | Serve HTTPS | none |
| | `RETIVASC_SERVER` | Where the Vite dev server proxies `/api` | `http://127.0.0.1:8080` |

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `release.py fetch`: "the repository folder is too deep for Windows" | AutoMorph's checkpoints sit in deep folders, and Windows limits paths to 259 characters. Clone to a short path such as `C:\RetiVASC` |
| `release.py fetch`: HTTP 404 | The release `v<VERSION>` lacks the asset. Check `VERSION` and the release page. Pass `--repo OWNER/NAME` if `origin` is not the GitHub repository |
| `release.py check` reports changed files | A download was interrupted or a file was edited. Delete the reported files and run `fetch` again |
| The page loads but steps fail with "model not found" | `webui/dist` was built before the models were fetched. Run `npx vite build` again in `webui/` |
| The browser engine stays on "Loading" or M1 never starts | The server sends `.mjs` with the wrong MIME type. The bundled server fixes this; on another web server, map `.mjs` to `text/javascript` |
| `CPU only` in the top bar | WebGPU is unavailable (old browser, remote desktop, blocked GPU). It still works, slowly. Try Chrome or Edge on the machine itself |
| Self-test: grades differ, or features off by more than 1e-3 | The wrong torch build, or a GPU that cannot run it. Check `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` in `.venv` |
| `CUDA error: no kernel image is available` | The torch build is too old for the GPU, e.g. cu118 on an RTX 50-series card. Reinstall with the cu128 wheels as in `install.ps1` |
| The wrong GPU is used | Set `gpu_env` to `CUDA_DEVICE_ORDER=PCI_BUS_ID;CUDA_VISIBLE_DEVICES=<index>` |
| A job failed with exit code 3221225477 | AutoMorph crashed natively, usually because GPU memory is shared with another process. The server retries once; free the GPU and resubmit |
| Port already in use | Another server instance is running. Stop it, or use `serve --port 8081` |
| `install.ps1`: `py` not found | Install Python 3.12 from python.org with the launcher, or create `.venv` by hand and rerun |

## Architecture, for changes

- **`webui/`** (React, TypeScript, Vite, onnxruntime-web).
  - Node types are in `src/engine/specs.ts` and ready-made workflows in `src/engine/workflows.ts`.
  - All inference and pixel work happens in `src/runtime/engine.worker.ts`.
  - Ports of AutoMorph's pre- and post-processing are in `src/automorph/`; the 72 measurements are in `src/automorph/m3/`.
  - Models are listed with SHA-256 in `public/models/manifest.json`.
  - Server steps call `/api/*`, through `src/server/api.ts`.
- **`retivasc_server/`** (FastAPI).
  - `app.py`: API and static files.
  - `jobs.py`: queue and worker thread, one job at a time on the GPU.
  - `automorph.py`: stages photos, runs `python -m automorph_v2 run <data_root>` as a subprocess and reads its CSVs.
  - `accounts.py`: bcrypt accounts and HMAC session tokens.
  - `selftest.py`: the acceptance test.
- **`automorph_v2/`** is run unchanged, through its command line. Do not edit it to fix RetiVASC behaviour.
- **Optional study-sheet module.** The server can fill and audit a study's Excel results sheet when a separate, study-specific `retivasc_sheet` package is installed next to it. That package is not part of this repository. Without it, `/api/server` reports `"features": {"sheet": false}`, `/api/jobs/sheet` returns 404, and the web UI hides those steps.

API summary:

| Method | Path | What it does |
|---|---|---|
| POST | `/api/login` | Body `{name, password}`; returns a session cookie |
| POST | `/api/logout` | Ends the session |
| GET | `/api/me` | The signed-in user |
| GET | `/api/server` | Server status |
| GET | `/api/jobs` | The caller's jobs |
| POST | `/api/jobs/analyze` | Multipart upload of `files[]` (photos) |
| GET | `/api/jobs/{id}` | Status and result of one job |
| GET | `/api/jobs/{id}/image/{path}` | A result image |
| GET | `/api/jobs/{id}/file/{name}` | A download |
| DELETE | `/api/jobs/{id}` | Deletes a job |

### Checks after a change

```bash
python -m pytest -q retivasc_server/tests -p no:cacheprovider     # server: sign-in, ownership, uploads, queue, restart, clean-up
cd webui && npm run check                                          # TypeScript
```

With a running instance, these real-browser checks need `playwright-core` and a local Chrome:
- `node webui/tests/ui.mjs` (dev server on 5178);
- `RETIVASC_URL=http://localhost:8080/ node webui/tests/open.mjs` (open mode).

The Python self-test (`python -m retivasc_server selftest`) is the reference check for anything touching AutoMorph.

## Releasing new weights or a new version

1. Bump `VERSION`.
2. If the models changed, regenerate them: `python tools/export_ensembles.py` writes the ONNX files and `webui/public/models/manifest.json`.
3. Run `python tools/release.py pack`. It writes the two zip files to `release/`, which git ignores.
4. Create the GitHub release `v<VERSION>` and attach both zip files.
5. Commit the updated manifests. The weights themselves never go into git.

## Rules

- Never commit model weights (`*.onnx`, `*.pth`, `*.pt`), `retivasc.local.json`, `work/`, uploaded photographs, results or spreadsheets. `.gitignore` covers them; check `git status` before committing.
- Never write participant identifiers, dates of service or local data paths into code, tests or docs.
- Opening the server to a network, changing the access mode, and deleting jobs or users change what other people can see or reach. Ask the owner first.
