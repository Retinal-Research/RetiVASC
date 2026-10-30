# RetiVASC

**Retinal Vascular Phenotype Platform for Brain Health** · [Geometry Systems Lab](https://gsl.lab.asu.edu/), Arizona State University

| | |
|---|---|
| Try it (no install) | https://retinal-research.github.io/RetiVASC/ |
| Code | https://github.com/Retinal-Research/RetiVASC |
| Models on Hugging Face | [AutoMorph](https://huggingface.co/hwang714/retivasc-automorph) · [TPOT](https://huggingface.co/hwang714/retivasc-tpot) · [RRWNet](https://huggingface.co/hwang714/retivasc-rrwnet) · [VascX](https://huggingface.co/Eyened/vascx) (its authors') |
| Lab | [Geometry Systems Lab](https://gsl.lab.asu.edu/), Arizona State University · [Retinal-Research on GitHub](https://github.com/Retinal-Research) |

RetiVASC measures the blood vessels of the retina in colour fundus photographs. Small changes in these vessels are being studied as signs of disease elsewhere in the body, including the brain. Measuring them by hand is slow and inconsistent, so RetiVASC does it automatically, in a web browser or on a lab server.

For each photograph, RetiVASC:

1. finds the circular field of view and crops the photograph to it;
2. grades image quality as good, usable or reject, so unreliable photographs can be set aside;
3. outlines the vessels, separates arteries from veins, and finds the optic disc and cup;
4. finds where arteries and veins branch, and measures the angle of each branching;
5. reports 72 measurements of the vessel tree. These are vessel density, calibre, tortuosity and fractal dimension, plus the summary calibres CRAE and CRVE and their ratio (AVR), for the whole image and for the standard rings around the optic disc;
6. writes a report: a language model turns the measurements into text, in English or Chinese, and answers questions about them.

The page opens on a home page with four pipelines, a QR code and share buttons (a phone is offered ways to send the link to a computer), and a one-minute tutorial that opens by itself on the first visit (the Tutorial button at the top right opens it again). Each step of a pipeline is a *skill* on a canvas: the pipelines are ready-made, and the skills can be connected into new ones.

| Pipeline | What it does |
|---|---|
| **RetiVASC** (ours) | The lab's [TPOT](https://arxiv.org/abs/2411.01403) enhancement first, at full resolution through TPOT-SR's fusion head. Every later step works on the enhanced photograph: AutoMorph's quality grade; [RRWNet](https://github.com/j-morano/rrwnet)'s arteries, veins and vessels, in place of AutoMorph's segmentation; AutoMorph's optic disc and 72 measurements; the lab's [RetiFlow](https://github.com/Retinal-Research/RetiFlow) branching angles on RRWNet's probability maps; and an AI report. RetinalGPT, the lab's conversational assistant for fundus photographs, and Retify are listed as coming soon. |
| **AutoMorph** | [AutoMorph](https://github.com/rmaphoh/AutoMorph), a published and validated pipeline, run with all of its models. |
| **VascX** | The [VascX models](https://huggingface.co/Eyened/vascx) of the Eyened Reading Center: their own field-of-view crop, then vessels, arteries/veins, the optic disc and the fovea, run as their Python packages run them. |
| **Custom** | A photograph and the skill library, to build your own. |

The input of every pipeline is one photograph or a folder of up to 1000 images. A folder runs one photograph at a time, can be stopped and resumed, and gives one CSV; the *Results table* step at the end of each pipeline shows the numbers of the steps connected to it, one row per photograph, and downloads them as a CSV named after the photo or folder, the pipeline and the date. Besides the 72 measurements, the CSV has the same simple measures for every pipeline, so AutoMorph and VascX can be compared: the share of the field of view covered by vessels, arteries, veins and the optic disc, and the position of the fovea; and, from RetiFlow, the number of branching points and their mean angle, for arteries and for veins.

**When a step fails** (a weak graphics card, say), the RetiVASC pipeline goes on. VascX's segmentation stands in for RRWNet, the cropped photo for the enhancement, and the AI report and the results table use whatever finished. The fallback steps run only when needed. The colours of vessels, arteries and veins can be changed in the appearance menu: by default they follow RRWNet's maps, with arteries in magenta and veins in cyan.

**The AI report** is RetiVASC's own skill: connect any steps to its *context* input, as many as you like, and it is given everything they produced (all 72 measurements, every column of a table) and writes what its prompt asks, so a doctor can also ask about specific values. You can edit its prompt and choose how hard it reasons; the rules it follows (a medical research assistant that uses only the given results, never diagnoses, and declines unrelated requests) are fixed on the server.
- The lab's language model writes it (GPT-6 Luna through ASU CreateAI). The RetiVASC server forwards only the numbers, never a photograph, keeps the API key, and limits each visitor (one request a second, three at once, $1 a day; $10 a day in all).
- RetiVASC computes every number; the model only writes. Each number in its text is looked up in the results, and words that judge a value (high, normal…) are flagged, because no reference ranges are given. The numbers it was given are shown next to the report.
- Connected to a *Results table*, it sees the table after it and also writes a table of what it found (as JSON, which the page reads), its numbers checked like the text's. Any step can tell what the steps after it are; this is how a step adapts its output to them.
- While the model thinks, the page shows the steps so far and the time each takes; the API sends only the answer, not the model's reasoning. Each visitor sees how much of their quota for the day they have used.
- After a folder, it writes one report on all the photographs: their quality, the spread of each measurement, and, when two steps measure the same thing (AutoMorph and VascX vessels, say), how they differ photo by photo.
- A language model can still be wrong. The report is for research, not a diagnosis.

RetiVASC can be used in two ways:
- in the browser, where every model runs on your own computer;
- on a server that a lab runs for its own users.

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
python tools/release.py fetch --vascx      # optional: the VascX models, 0.56 GB, from Hugging Face
python tools/release.py fetch --llm        # optional: the AI report's language model, 1.4 GB, from Hugging Face
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

**RRWNet** runs as an ONNX file made from its authors' weights by `tools/export_rrwnet.py`, which writes out the network and its preprocessing from their code. The browser's preprocessing agrees with the Python to within 4e-16; the network, with half-precision weights, to within 0.006 in any probability, and 31 of the 1,023,168 values of a sample's probability map fall on the other side of 0.5.

**RetiFlow** runs as a TypeScript port of its Python code (`webui/src/retiflow`), reproducing the SciPy, scikit-image and OpenCV operations it uses, down to the order in which ties are broken. `webui/tests/retiflow.ts` checks each of its steps against the original (`tools/retiflow_reference.py` runs it): on RRWNet's map of a sample photograph, the completed masks, both centrelines, the root, the endpoint repairs and every branching point and angle are identical to the Python's (scikit-image 0.26, SciPy 1.18, OpenCV 5.0).

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

The VascX models are not part of RetiVASC. `tools/release.py fetch --vascx` and the public site load them from the authors' repository, [Eyened/vascx](https://huggingface.co/Eyened/vascx), at a fixed revision. Their model cards license them under CC BY-NC 4.0, for non-commercial research only. The VascX preprocessing in `webui/src/vascx/prep.ts` follows retinalysis-fundusprep (AGPL-3.0).

The AI report's language model is not part of RetiVASC either: Qwen3-1.7B by the Qwen team (Alibaba), Apache-2.0, as [onnx-community/Qwen3-1.7B-ONNX](https://huggingface.co/onnx-community/Qwen3-1.7B-ONNX) publishes it; `tools/release.py fetch --llm` and the public site load it from there at a fixed revision.

RRWNet ([j-morano/rrwnet](https://github.com/j-morano/rrwnet), MIT; Morano, Aresta and Bogunović, *Expert Systems with Applications*, 2024) is not part of RetiVASC either. Its weights ([j-morano/rrwnet-hrf](https://huggingface.co/j-morano/rrwnet-hrf)) are CC BY-NC 4.0, for non-commercial use with attribution; `tools/export_rrwnet.py` converts them, and the public site loads the converted copy from Hugging Face. `webui/src/rrwnet/NOTICE.md` keeps the code's notice.

RetiFlow is the lab's ([Retinal-Research/RetiFlow](https://github.com/Retinal-Research/RetiFlow), MIT). Its port in `webui/src/retiflow` includes parts of scikit-image (BSD); `webui/src/retiflow/NOTICE.md` keeps both notices.

TPOT is the lab's model ([Retinal-Research/TPOT-SR](https://github.com/Retinal-Research/TPOT-SR)). `tools/export_tpot.py` turns its two checkpoints into the ONNX files the browser runs; it can also export the lab's adapter-tuned TPOT (`eyepacs_step3_aggressive_best.pt`) in place of the generator. The TPOT-SR repository does not state a licence yet.

Parts of the code and documentation were written with the help of Claude, an AI assistant by Anthropic.

The sample photographs in `webui/public/samples/` come from Wikimedia Commons (public domain, CC0 or CC BY-SA 3.0) and, in `samples/eyepacs/`, from EyePACS (the Kaggle Diabetic Retinopathy Detection data); `samples.json` lists the author, licence and source of each.

If you use RetiVASC, please cite AutoMorph:

> Zhou Y, Wagner SK, Chia MA, et al. AutoMorph: Automated Retinal Vascular Morphology Quantification Via a Deep Learning Pipeline. *Translational Vision Science & Technology* 11(7):12, 2022. https://doi.org/10.1167/tvst.11.7.12
