# RetiVASC

RetiVASC measures the blood vessels of the retina in colour fundus photographs. Small changes in these vessels are being studied as signs of disease elsewhere in the body, including the brain. Measuring them by hand is slow and inconsistent, so RetiVASC does it automatically, in a web browser or on a lab server.

For each photograph, RetiVASC:

1. finds the circular field of view and crops the photograph to it;
2. grades image quality as good, usable or reject, so unreliable photographs can be set aside;
3. outlines the vessels, separates arteries from veins, and finds the optic disc and cup;
4. reports 72 measurements of the vessel tree. These are vessel density, calibre, tortuosity and fractal dimension, plus the summary calibres CRAE and CRVE and their ratio (AVR), for the whole image and for the standard rings around the optic disc.

The analysis is [AutoMorph](https://github.com/rmaphoh/AutoMorph), a published and validated pipeline, run with all of its models. RetiVASC adds two ways of using it:
- a visual workflow editor that runs in the browser;
- a server that a lab can run for its own users.

## Two ways to use it

**In the browser.** Open the page, pick a photograph, and press Run. Every step runs on your own computer's graphics card through WebGPU, so the photograph never leaves your computer.
- The models (0.85 GB) download once and are then cached by the browser.
- After that, a photograph takes about 10 seconds.
- Results agree with the original Python AutoMorph to within 0.4% on every measurement.

**On a lab server.** One computer with an NVIDIA GPU runs the complete original AutoMorph for everyone on the lab network.
- Users open the same web page, upload photographs, and download the grades, segmentation images and a CSV of the 72 measurements.
- Jobs wait in a queue and keep running after the page is closed. They are deleted after a set number of days.
- Each user sees only their own jobs.
- The server can require sign-in, or run in an open "public beta" mode where each browser gets its own private session.

Photographs are never sent anywhere else in either mode.

## Quick start

You need Git, Node.js 20 or newer, and Chrome or Edge. The browser version also needs any Python 3 to run the download script. The server installer needs no Python at all: it uses [uv](https://docs.astral.sh/uv/), installing it first if it is missing, to fetch a private Python 3.12 into the project folder, so nothing already on the machine is changed. On Windows, clone into a short folder such as `C:\RetiVASC`; some model files sit in deep folders, and Windows limits path length.

The model weights are too large for git. They are attached to the GitHub release, and `tools/release.py` downloads them and checks each file against the SHA-256 lists kept in this repository.

**Browser only:**

```bash
git clone https://github.com/Retinal-Research/RetiVASC.git
cd RetiVASC
python tools/release.py fetch --web        # 0.85 GB, once
cd webui && npm ci && npm run dev          # then open http://localhost:5178
```

**Lab server, Windows with an NVIDIA GPU:**

```powershell
git clone https://github.com/Retinal-Research/RetiVASC.git; cd RetiVASC
powershell -ExecutionPolicy Bypass -File install.ps1          # downloads 2.5 GB of weights, installs, self-tests
.venv\Scripts\python -m retivasc_server adduser alice --admin  # asks for a password
start.bat                                                      # http://localhost:8080
```

**Lab server, Linux or macOS:** run `./install.sh` (it detects an Apple-silicon Mac, an NVIDIA GPU or neither), add a user the same way, then start the server with `./start.sh`.

The server runs AutoMorph on a GPU: an NVIDIA card (AMD through ROCm also works) or the GPU of an Apple-silicon Mac. By default each job takes the GPU with the most free memory when it starts; the CPU is used only when there is no GPU. Each server step in the web page has a *Device* menu to choose another one, and `python -m retivasc_server devices` lists them.

The installer finishes with a self-test. The self-test runs the full pipeline on six sample photographs and compares every result with reference values. If it fails, the installation does not reproduce AutoMorph and should not be used.

Step-by-step deployment notes, every setting and troubleshooting are in [AGENTS.md](AGENTS.md).

## What is in this repository

| Folder | What it is |
|---|---|
| `webui/` | The web page: a canvas on which analysis steps are connected into workflows. It contains the browser version of AutoMorph, including a TypeScript port of the 72 measurements |
| `retivasc_server/` | The lab server: sign-in, job queue, uploads and downloads, clean-up, access log, self-test |
| `automorph_v2/` | AutoMorph v2, a faster repackaging of AutoMorph with the original models. The server runs it unchanged |
| `tools/` | Model export to ONNX, parity checks between browser and Python, and the release packaging script |
| `deploy/` | Docker files for a Linux GPU host without internet access |

## How closely the browser matches AutoMorph

The browser version is not a re-implementation by eye. Every step was compared with the original Python code on the same photographs:

- **Preprocessing.** Bit-identical.
- **Quality grading and segmentation.** These use all ensemble members, averaged in AutoMorph's order. Out of roughly 830,000 pixels per photograph, at most 25 vessel pixels, 7 artery/vein pixels and 4 optic-disc pixels differ.
- **The 72 measurements.** Fed the same segmentations, every value is bit-identical, except the fractal dimension, which agrees to 15 significant digits. This holds on 68 test photographs, including photographs centred on the macula or the disc and photographs where the disc was not found.
- **End to end.** All 72 values are within 0.4% of the Python results.

The browser models store their weights at half precision to halve the download; the arithmetic itself runs at full precision. The optic-disc model keeps full precision, because a one-pixel shift of the disc moves the measurement rings.

## Speed

| Task | Hardware | Time |
|---|---|---|
| One photograph in the browser, all models | RTX 5070 Ti, WebGPU | about 8–14 s once the models are cached |
| One photograph on the server, all models | RTX 5070 Ti | about 50 s, mostly loading models |
| About 1,240 photographs on the server | RTX 5070 Ti | about 37 min, unattended |
| Server self-test, 6 photographs | RTX 5070 Ti | about 45 s |

A *Quick* box on each browser step runs only the first model of each ensemble, for slower computers.

## Licence and acknowledgements

RetiVASC is released under the GNU General Public License v3.0 (see [LICENSE](LICENSE)). It has to be: the 72 measurements derive from [Retipy](https://github.com/alevalv/retipy), which is GPL-3.0. AutoMorph v2 in `automorph_v2/` keeps its own Apache-2.0 licence and notices (`automorph_v2/LICENSE`, `automorph_v2/THIRD_PARTY_NOTICES.md`). The model weights are redistributed under AutoMorph's original terms.

Parts of the code and documentation were written with the help of Claude, an AI assistant by Anthropic.

The sample photographs in `webui/public/samples/` come from Wikimedia Commons (public domain, CC0 or CC BY-SA 3.0); `samples.json` lists the author, licence and source of each.

If you use RetiVASC, please cite AutoMorph:

> Zhou Y, Wagner SK, Chia MA, et al. AutoMorph: Automated Retinal Vascular Morphology Quantification Via a Deep Learning Pipeline. *Translational Vision Science & Technology* 11(7):12, 2022. https://doi.org/10.1167/tvst.11.7.12
