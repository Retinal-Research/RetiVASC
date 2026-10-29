# RetiVASC: deployment and maintenance manual for agents

This manual is for a coding agent asked to deploy, check or change RetiVASC. Commands are exact; run them from the repository root unless a step says otherwise.

The time budget is about three minutes of commands. Downloads add to it:
- the model weights are 0.85 GB for the browser and 2.5 GB for the server;
- the CUDA build of PyTorch is about 3 GB.

## 0. Decide what to deploy

| The user wants | Deploy | Needs |
|---|---|---|
| To try it, or a demo on one computer | **A. Browser only** | Python 3 (any), Node.js ≥ 20, Chrome/Edge with WebGPU |
| A shared service for a lab, results identical to AutoMorph | **B. Server** | Node.js ≥ 20, an NVIDIA GPU with a current driver (CPU works, slowly) |
| A server on a Linux host without internet | **C. Docker** | Docker with the NVIDIA Container Toolkit on the host; a machine with internet to build the image |

If unsure, ask the user. B and C serve the web UI from A as well.

Preflight checks:

```bash
git --version
node --version            # v20 or newer
python --version          # A only: any Python 3. B needs none: the installer gets Python 3.12 through uv
nvidia-smi                # B and C with a GPU: driver present, note free memory (needs about 6 GB)
```

On Windows, clone to a short path such as `C:\RetiVASC`: AutoMorph's model folders are deep, and Windows limits paths to 259 characters.

## A. Browser only (about 1 minute plus a 0.85 GB download)

```bash
python tools/release.py fetch --web     # downloads the browser models into webui/public/models, checks SHA-256
python tools/release.py fetch --vascx   # optional: the VascX models (0.56 GB) from Eyened/vascx on Hugging Face
python tools/release.py fetch --llm     # optional: the AI report's language model (1.4 GB) from onnx-community on Hugging Face
cd webui
npm ci
npm run dev                             # http://localhost:5178 (port is fixed: strictPort)
```

Verify the deployment:
- Open http://localhost:5178. The home page lists the pipelines; *Start* opens the canvas with the chosen one. Nothing runs until *Run* (the PI asked for this), so opening the page downloads no models.
- Choose *AutoMorph* and press *Run* on the sample photograph.
- The top bar names the engine. `WebGPU · <adapter>` is right; `CPU only` means WebGPU is unavailable, and the run is slow but correct.
- Every node turns green. *M3 · Features* lists 72 values.
- The *RetiVASC* pipeline needs the TPOT enhancement models and RRWNet, and *VascX* the VascX models; a pipeline whose models are missing is greyed out on the home page. Its AI report is written by the report service of a RetiVASC server (see B).
  - VascX: `python tools/release.py fetch --vascx`.
  - TPOT: download `best_SSIM.pth` and `fusion_head_best.pth` from Retinal-Research/TPOT-SR (`weights/`), then run `python tools/export_tpot.py <their folder>` with a Python that has torch, onnx and Pillow. It writes `webui/public/models/tpot/` and reference tensors; `cd webui && npx tsx tests/tpot.ts` then checks the ONNX files against PyTorch.
  - RRWNet: download `model.safetensors` from j-morano/rrwnet-hrf on Hugging Face at revision `f64c297d81f05bcce2c40796c1a9bee4c72a1908`, then run `python tools/export_rrwnet.py <model.safetensors>` with a Python that has torch, onnx, onnxruntime, safetensors and Pillow. It writes `webui/public/models/rrwnet/` and reference tensors; `cd webui && npx tsx tests/rrwnet.ts` checks the ONNX file against PyTorch. The weights are CC BY-NC 4.0: never commit them.
  - The in-browser language model (`python tools/release.py fetch --llm`) is frozen and no longer offered.

To serve a static build instead: `npm run build` writes `webui/dist`; `npm run preview` serves it on port 5179. Any static web server works, but it must:
- send `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`, which multi-threaded CPU fallback needs;
- serve `.mjs` as `text/javascript` and `.wasm` as `application/wasm`.

### Static site on GitHub Pages

The browser version needs no server at all: the page is static files, and every step runs on the visitor's computer.
- Pages limits a site to about 1 GB, so the models are hosted elsewhere, with one folder per model (`quality/`, `vessel/`, `artery_vein/`, `disc_cup/`). Build with `VITE_MODELS_URL` set to the folder that holds them, e.g. a Hugging Face model repository at a version tag: `https://huggingface.co/<owner>/<repo>/resolve/v0.1.0/`. The host must allow cross-origin requests, which Hugging Face does. The public site uses [hwang714/retivasc-automorph](https://huggingface.co/hwang714/retivasc-automorph); further pipelines get a model repository each, since their licences differ. Without the variable the page loads `models/` next to itself, as before.
- Pages cannot send the isolation headers. `webui/public/coi-serviceworker.js` adds them in the browser (the page reloads once on the first visit); where the server already sends them it does nothing.
- `.github/workflows/pages.yml` builds and deploys on every push to `main`, once the repository variable `RETIVASC_MODELS_URL` is set and Pages uses "GitHub Actions" as its source. Changes are tested on `beta` first.
- Offline sites (a hospital intranet, a mobile screening unit) build without `VITE_MODELS_URL`, so the models ship with the page and nothing is downloaded at run time; the server (B) does the same.
- VascX loads from its authors' repository (`VITE_VASCX_URL`, by default Eyened/vascx at a fixed revision), TPOT from `VITE_TPOT_URL` (repository variable `RETIVASC_TPOT_URL`) and RRWNet from `VITE_RRWNET_URL` (repository variable `RETIVASC_RRWNET_URL`; the converted copy is staged for hwang714/retivasc-rrwnet on Hugging Face). Without a TPOT or RRWNet host the build leaves that model out, and the home page greys out the RetiVASC pipeline.
- The language model loads from `VITE_LLM_URL` (by default onnx-community/Qwen3-1.7B-ONNX at a fixed revision; repository variable `RETIVASC_LLM_URL`), only when a report is written.
- Without a RetiVASC server the server steps are hidden; as in A, nothing runs until Run.

## B. Server (about 3 minutes plus downloads)

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1        # -Gpu cpu for a machine without NVIDIA
.venv\Scripts\python -m retivasc_server adduser admin --admin --password "<choose one>"
start.bat
```

### Linux or macOS

```bash
./install.sh                                                  # detects mac / nvidia / cpu; or name one
.venv/bin/python -m retivasc_server adduser admin --admin --password '<choose one>'
./start.sh
```

- macOS needs Apple silicon (M1 or later): PyTorch 2.7 has no Intel-Mac builds, and the installer stops there with that message. AutoMorph runs on the Apple GPU (`mps`); the few operators MPS lacks fall back to the CPU (`PYTORCH_ENABLE_MPS_FALLBACK=1` is set for it).
- Linux with an NVIDIA card needs a driver that supports CUDA 12.8 (`nvidia-smi` shows "CUDA Version: 12.8" or higher). AMD cards with ROCm also work, after reinstalling torch from the ROCm index; PyTorch reports them as `cuda`.
- The Neural Engine of Apple chips and other NPUs cannot run PyTorch models; they are not used.

### What the installer does

The installer does the following, so you can repeat any step by hand:
1. Creates `.venv` with Python 3.12 using [uv](https://docs.astral.sh/uv/), installing uv into the user's `~/.local/bin` first if it is missing. uv downloads its own Python, so no system Python, conda environment or PATH entry is used or changed. Packages are installed with `uv pip install --python .venv/...`; the venv has no pip of its own.
2. Downloads both weight packages: `python tools/release.py fetch`.
3. Installs PyTorch 2.7.1. CUDA 12.8 wheels cover current NVIDIA cards, including the RTX 50 series; there are also CPU and macOS builds. Then it installs `requirements-server.txt`.
4. Writes `retivasc.local.json` from `retivasc.local.example.json`, setting `python`, `gpu_python` and `server.device` (`auto`, or `cpu` when installed for CPU).
5. Builds the web UI (`npm ci && npx vite build`) if `webui/dist` is missing.
6. Runs the acceptance self-test.

The self-test must print `SELFTEST PASSED` (exit 0). It runs the complete AutoMorph v2 on the six sample photographs (about 45 s on a GPU) and compares every quality probability and all 72 features with `retivasc_server/reference.json`:
- grades must match exactly;
- values must agree within 1e-3 (relative) on an NVIDIA GPU, as the reference was made on one; within 1e-2 on the CPU or an Apple GPU, whose arithmetic rounds differently inside the networks and moves a few segmented pixels (the CPU differs by up to 1.4e-3 on the samples).

`selftest --device cpu` (or `mps`, `cuda:1`) checks one device. Results from different kinds of device are close but not identical, so run all photos of one study on the same kind of device.

If it fails, stop and report the printed differences. Do not hand over the server.

### Verify the running server

```bash
curl -s http://127.0.0.1:8080/api/server -b cookies.txt -c cookies.txt   # 401 in accounts mode until signed in
curl -s -X POST http://127.0.0.1:8080/api/login -H 'Content-Type: application/json' \
     -d '{"name":"admin","password":"<the password>"}' -c cookies.txt
curl -s http://127.0.0.1:8080/api/server -b cookies.txt                 # {"version":..., "device":"auto", "devices":[...], ...}
```

Then open http://localhost:8080, sign in, and run *Full analysis on the server* on a sample photograph. It takes about 50 s and shows a grade, four images and 72 values.

### Access mode

- `accounts` (the default): everyone signs in. Add users with `adduser NAME [--admin] [--password …]`, remove them with `deluser NAME`, list them with `users`.
- `open` (public beta): no sign-in; every browser gets its own private guest session, limited to 3 waiting jobs.
- Switch with `python -m retivasc_server mode open` or `mode accounts`, then restart. Accounts are kept in both modes.

### The AI report through the lab's API

The server can write AI reports through the lab's language-model API for every page, the public site included, without the key leaving the machine (`retivasc_server/report.py`, `/api/report`).
- Set it in `retivasc.local.json` under `report` (see `retivasc.local.example.json`): `enabled`, `provider`, `url`, `model`, the sites allowed to call it (`origins`), and the limits: `per_window` requests per IP in `window_seconds` (1 a second), `max_running` answers being written at once per IP (3), `ip_dollars_per_day` per IP ($1) and `dollars_per_day` for everyone ($10), counted from what the lab's API reports each answer cost. The key comes from the environment variable `key_env` names (default `RETIVASC_REPORT_KEY`) or from `key_file`; never put it in the repository or a chat.
- `GET /api/report` tells the page whether reports can be written and what the caller has spent today against its daily limit (`today`: `used`, `limit`, `open`); the page shows it as the visitor's quota. The day is the UTC day.
- The AI report's input is its *context*: any number of steps, and everything they give goes to the model (all 72 M3 values, every column of a Results table), as the person's prompt asks; nothing is filtered on the way. The browser sends its fact sheet, the person's request (the editable prompt) and the reasoning effort (low, medium, high). The size limits are generous (2 MB a request, 20,000 characters a prompt or question, 100 follow-ups); the dollars a day are the real limit. The system prompt is the server's own, from `webui/src/engine/prompts.json` on this machine; the page cannot change it. The server builds the message itself, the same as the browser's (`webui/tests/prompt.ts` and `retivasc_server/tests/test_report.py` hold both to `webui/tests/prompt_fixture.json`), and the system prompt keeps the model on the given results and declines unrelated requests. Nothing is stored.
- For the public site, expose only `/api/report` of this server through the Cloudflare tunnel (a public hostname in the Cloudflare dashboard with path `^/api/report`), and build the site with the repository variable `RETIVASC_REPORT_URL` set to `https://<that hostname>/api/report`. Behind the tunnel, limits count the visitor's address (`CF-Connecting-IP`).
- Reports are written through this service only. The in-browser model (`src/llm`, Qwen3-1.7B) is frozen: the page no longer offers it (a 1.4 GB download nobody wants), and only a workflow file with `engine: "local"` still runs it.
- `provider: "gateway"` writes through the lab's LLM gateway on the same machine (`~/Projekt/lab-llm`, `http://127.0.0.1:8095/v1`), which holds the ASU CreateAI token; RetiVASC has only its own app key (`key_file`, default `~/.config/lab-llm/apps/retivasc.key`). The default model is `openai/gpt6_luna`. `provider: "fake"` writes a canned report, for tests only.

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
| `server.device` | | The default device: `auto` (when each job starts, the NVIDIA/AMD GPU with the most free memory, else the Apple GPU, else the CPU), or a fixed `cuda:N`, `mps` or `cpu`. Users can override it per job in the web UI | `auto` |
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
| The wrong GPU is used | `python -m retivasc_server devices` lists what the server sees and what `auto` picks. Fix one with `server.device` (`cuda:N`), or hide cards with `gpu_env` `CUDA_DEVICE_ORDER=PCI_BUS_ID;CUDA_VISIBLE_DEVICES=<index>` (indices then count the visible cards only) |
| A job failed with exit code 3221225477 | AutoMorph crashed natively, usually because GPU memory is shared with another process. The server retries once; free the GPU and resubmit |
| Port already in use | Another server instance is running. Stop it, or use `serve --port 8081` |
| Installer: installing uv fails | No internet access to astral.sh, or a proxy blocks it. Install uv by another route (`pip install uv`, `winget install astral-sh.uv`), or create `.venv` with any Python 3.12 by hand and rerun |
| Installer: "Node.js 20 or newer is required" | Install Node.js LTS from nodejs.org, open a new terminal, rerun |

## Architecture, for changes

- **`webui/`** (React, TypeScript, Vite, onnxruntime-web).
  - Node types ("skills") are in `src/engine/specs.ts` and the pipeline library in `src/engine/workflows.ts`: RetiVASC, AutoMorph, VascX and Custom on the home page (`src/ui/Home.tsx`), then single steps and the server workflows. RetiVASC (ours) is TPOT's enhancement, then on the enhanced photo AutoMorph's quality grade, RRWNet's arteries/veins/vessels (instead of AutoMorph's segmentation), AutoMorph's optic disc and M3, RetiFlow's branching angles on RRWNet's probability maps, and the AI report. RetinalGPT and Retify are placeholder steps (`soon: true`): listed, never run.
  - Four model sets, each with its own manifest and folder (their licences differ): AutoMorph in `public/models/`, VascX in `public/models/vascx/` (CC BY-NC 4.0, fetched from Eyened/vascx, never committed), TPOT in `public/models/tpot/` (made by `tools/export_tpot.py`), RRWNet in `public/models/rrwnet/` (made by `tools/export_rrwnet.py`; weights CC BY-NC 4.0, never committed). `VITE_MODELS_URL`, `VITE_VASCX_URL`, `VITE_TPOT_URL` and `VITE_RRWNET_URL` point the page at hosted copies instead.
  - RRWNet's preprocessing is `src/rrwnet/prep.ts`, in float64 as RRWNet's Python does it; the network runs on a 584 x 584 square of the photo, and its probabilities are kept as the 8-bit AV3 map RetiFlow reads (and resized to AutoMorph's M2 sizes for M3).
  - RetiFlow is `src/retiflow`, a port of its Python (`index.ts` is `infer.process_prob`: mask completion, centrelines, the two-pass RBAD with endpoint repair). `mcp.ts` and `ndi.ts` reproduce the scikit-image, SciPy and OpenCV operations it uses, ties included; `NOTICE.md` keeps the licences. It runs in the engine worker (`retiflow` op), about 1.3 s a photo. Every photo is measured, gradable or not; the step takes M1's grade (optional) to say so, flags a vessel map covering under 2% or over 30% of the field of view, and if it fails returns no branching points with the reason instead of stopping a folder run. `tests/retiflow.ts` checks each step against the original, which `tools/retiflow_reference.py` runs; on the sample every step is identical. Two traps found doing it: scikit-image 0.26's `remove_small_objects(min_size=k)` removes pieces of k pixels or fewer (its documentation says fewer than k), and `ProbabilityConfig()`'s defaults are not the settings `infer.py` uses (config.py's).
  - The *Results table* step (`view.table`, `src/engine/table.ts`, `src/ui/TablePanel.tsx`) takes any steps with results: one row per photo (after a folder, every photo: it is a `cohort` step), in the Photo folder CSV's columns, with *Download CSV* named `<photo or folder>_<pipeline>_<date>.csv`. Each pipeline ends with one.
  - A step knows which steps its outputs feed (`RunContext.downstream`). The AI report uses it: when a Results table follows, it sends `needs: ["table"]`, the server adds the fixed instruction from `prompts.json` (`needs.table`), and the model ends with a JSON table that `takeTable` reads and the table step shows; its numbers are checked like the text's. For a language-model step the downstream types are part of its cache key. New needs go in `prompts.json` (the server accepts only those) and `tests/prompt.ts`.
  - The tutorial is `src/ui/Tour.tsx`: steps light up real elements by CSS selector (and canvas steps by node id: `photo`, `m1`, `report`). It opens by itself on a first visit, except in automated browsers (`navigator.webdriver`), and ends on the home page. If you rename those classes or ids, update the steps; `tests/tour.mjs` checks that every card lights its part.
  - VascX's preprocessing (field of view, 1024 crop) is `src/vascx/prep.ts`, following retinalysis-fundusprep 1.3.0. Its inference (windows, Gaussian blending, flips) and TPOT's (256 pass, fusion head at 2048 in tiles) are in `src/runtime/engine.worker.ts`.
  - All inference and pixel work happens in `src/runtime/engine.worker.ts`. The top bar's device menu restarts it on another device (`GpuChoice`: the faster or the power-saving graphics card, or the CPU; a web page cannot list the cards), and *Preload* asks it to load the models the canvas uses (`status` and `preload` ops).
  - The AI report (`ai.report`) uses a second worker, `src/llm/llm.worker.ts`: Qwen3-1.7B (4-bit) in onnxruntime-web, with its own BPE tokenizer (`src/llm/tokenizer.ts`, checked against Hugging Face tokenizers by `tests/tokenizer.ts`). The 1.4 GB model does not fit in WebAssembly memory once parsed (`std::bad_alloc`), so `src/llm/onnxsplit.ts` rewrites its protobuf to give the weights as external data, which go straight to the GPU. A conversation keeps its attention cache, so follow-up questions read only the new words.
  - What the report is told is computed in `src/engine/facts.ts` (one photo, or statistics over a folder, with comparisons between steps), in English or Chinese; the model only writes (`src/engine/report.ts`), and every number in its text is checked against those facts. Keep the numbers out of the model: a 1.7B model copies numbers well but compares and averages them badly.
  - A port with `multi: true` takes several connections (`RunContext.many`); a step with `cohort: true` downstream of a Photo folder runs once, after the last photo, on the run's records (`RunContext.cohort`).
  - The *Photo folder* input node (`src/engine/folder.ts`) takes up to 1000 photos. The executor then runs every step downstream of it once per photo, the whole downstream graph for one photo before the next, so memory holds one photo at a time. Each finished photo's grades and features are stored in IndexedDB; choosing the same folder again and pressing Run resumes. The engine keeps up to 2 GB of model weights loaded, which is every ensemble member, so models load once per folder.
  - Ports of AutoMorph's pre- and post-processing are in `src/automorph/`; the 72 measurements are in `src/automorph/m3/`.
  - Models are listed with SHA-256 in `public/models/manifest.json`.
  - Server steps call `/api/*`, through `src/server/api.ts`.
- **`retivasc_server/`** (FastAPI).
  - `app.py`: API and static files.
  - `jobs.py`: queue and worker thread, one job at a time on the GPU.
  - `automorph.py`: stages photos, runs `python -m automorph_v2 run <data_root>` as a subprocess and reads its CSVs.
  - `accounts.py`: bcrypt accounts and HMAC session tokens.
  - `selftest.py`: the acceptance test.
  - `monitor.py`: CPU, memory and GPU load for the status bar (psutil; `ioreg` on macOS, `nvidia-smi` on NVIDIA).
- **`automorph_v2/`** is run unchanged, through its command line. Do not edit it to fix RetiVASC behaviour.
- **Optional study-sheet module.** The server can fill and audit a study's Excel results sheet when a separate, study-specific `retivasc_sheet` package is installed next to it. That package is not part of this repository. Without it, `/api/server` reports `"features": {"sheet": false}`, `/api/jobs/sheet` returns 404, and the web UI hides those steps.

API summary:

| Method | Path | What it does |
|---|---|---|
| POST | `/api/login` | Body `{name, password}`; returns a session cookie |
| POST | `/api/logout` | Ends the session |
| GET | `/api/me` | The signed-in user |
| GET | `/api/server` | Server status |
| GET | `/api/stats` | The last minute of CPU, memory and GPU load, and the latest analysis time, for the status bar |
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

With a running instance, these real-browser checks need `playwright-core` and a local Chrome, or Playwright's own Chromium with `BROWSER_CHANNEL=chromium` (the lab Mac has no Chrome):
- `node webui/tests/ui.mjs` and `node webui/tests/layout.mjs` (dev server on 5178): the canvas, the folding sidebar, panel sizes, Preload, Stop and the device menu;
- `node webui/tests/llm.mjs` and `node webui/tests/report.mjs` (dev server on 5178, language model fetched): the language model, and the AI report on a photo, in Chinese, and after a folder;
- `RETIVASC_URL=http://localhost:8080/ node webui/tests/open.mjs` (open mode);
- `node webui/tests/retivasc.mjs`: the RetiVASC pipeline end to end, RRWNet and RetiFlow included, and its AI report;
- `node webui/tests/tour.mjs`: the tutorial.

Without a browser: `cd webui && npx tsx tests/tokenizer.ts` (the tokenizer against Hugging Face's), `npx tsx tests/tpot.ts`, `npx tsx tests/rrwnet.ts` and `npx tsx tests/retiflow.ts` (its first run writes the input that `tools/retiflow_reference.py` needs).

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
