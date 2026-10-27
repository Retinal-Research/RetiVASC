# AutoMorph v2

AutoMorph v2 is an end-to-end retinal fundus image analysis pipeline. It standardizes fundus photographs, assesses image quality, segments vessels and artery/vein classes, locates the optic disc and cup, and produces retinal vascular morphology tables for downstream research.

This repository is a compatibility-first, performance-optimized downstream release of AutoMorph. It retains the accepted model ensembles, FP32 numerical order, thresholds, output encodings, and M0-M3 data contract while replacing avoidable process startup, model reloads, filesystem traffic, and serial CPU work. The original AutoMorph checkout remains separate and unchanged.

> [!IMPORTANT]
> AutoMorph v2 is research software, not a medical device. It is not intended for diagnosis, treatment decisions, or unsupervised clinical use.

## Run it — for humans

Clone the repository, fetch the model payloads with Git LFS, bootstrap the
Python 3.10 environment once, put fundus photographs in `DATA_ROOT/images/`,
and run one command. The launcher uses the repository-local `.venv`, selects 8
workers automatically on a high-core-count host, shows per-phase progress and
ETA, and keeps every generated artifact inside `DATA_ROOT`.

```powershell
# Windows + NVIDIA CUDA 11.8
git lfs install
git lfs pull
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -Backend cu118
.\run.ps1 run E:\path\to\DATA_ROOT --device auto
```

```bash
# Linux + NVIDIA; use --backend cpu without NVIDIA
git lfs install && git lfs pull
bash ./scripts/bootstrap.sh --backend cu118
./run.sh run /path/to/DATA_ROOT --device auto

# Apple Silicon
bash ./scripts/bootstrap.sh --backend mps
./run.sh run /path/to/DATA_ROOT --device auto
```

Minimum input layout:

```text
DATA_ROOT/
  images/
    image_001.png
    image_002.jpg
```

`resolution_information.csv` is now optional. If it is absent, v2 creates the
original AutoMorph `fundus,res` table automatically with the legacy default
`0.008 mm/original-pixel`. This preserves the original helper's behavior and
the accepted benchmark, but it is an assumed Topcon scale—not a measurement
derived from the raster image. For calibrated research, provide the table:

```csv
fundus,res
image_001.png,0.01
image_002.jpg,0.01
```

The `fundus` value must match the source filename. Use
`--require-calibrated-resolution` to reject a missing table, or
`--resolution-mm-per-pixel VALUE` to choose another uniform assumption. Raster
DPI is deliberately ignored because print/display DPI is not a trustworthy
retinal physical calibration.

v2 records the table hash and source in
`.automorph_v2/resolution_provenance.json`. A known auto-generated table
remains explicitly marked as an assumption across `--resume` and
`--overwrite`; changing its uniform value requires `--overwrite`. If
somebody edits an auto-generated CSV, v2 preserves it but marks it untrusted
instead of silently treating it as calibrated. Strict calibrated mode rejects
both auto-generated and modified-after-generation tables.

## Run it — for coding agents

Work from the repository root and treat `runtime/`, the 50 model checkpoints,
ensemble membership/order, M1 batch size 64, FP32 reduction order, thresholds,
filenames, and CSV serialization as frozen compatibility inputs; maintained
code belongs under `src/`. Never run experiments in original datasets or a
legacy/oracle tree, never use an oracle directory as candidate output, and put
all temporary fixtures and logs under ignored directories inside
`AutoMorph_v2`. Use `python -m automorph_v2 run DATA_ROOT --device auto` unless
the task explicitly targets one module, and preserve every legacy-facing
option when extending a wrapper. Before reporting success, verify the
environment, manifest, tests, `DATA_ROOT/run.log`, and the relevant strict
comparator; never claim cross-platform pixel identity from a Windows-only run.

```powershell
.\.venv\Scripts\python.exe .\scripts\doctor.py --backend cu118
.\.venv\Scripts\python.exe .\scripts\build_model_manifest.py --check
.\.venv\Scripts\python.exe -m pytest -q
```

Use the equivalent `./.venv/bin/python` commands on Linux and macOS. A valid change must leave private images, generated results, caches, `.venv`, and machine-absolute paths out of the publishable tree.

## Performance overview

AutoMorph v2 reproduces the original AutoMorph outputs exactly on its reference platform while running each stage several times faster (overall about 15x on an RTX 4090). A 16 GiB GPU is the practical recommendation for the default high-throughput profile; lower-memory hosts can use `--vessel-engine low-vram` or CPU. The acceptance reports and benchmark charts of the original release are not included in this distribution.

## Pipeline at a glance

| Stage | Purpose | Primary output |
|---|---|---|
| M0 | Detect the retinal field, normalize geometry, crop/pad, resize, and carry forward image resolution | `Results/M0/images/` and `crop_info.csv` |
| M1 | Run the eight-model EyePACS quality ensemble, save pure good/usable/bad classes, and record gradability | `Results/M1/M1_quality_final.csv`, `Good_quality/`, `Usable_quality/`, `Bad_quality/` |
| M2 | Segment binary vessels, arteries/veins, optic disc, and optic cup; create centred analysis regions | `Results/M2/` segmentation trees and M3 centring tables |
| M3 | Measure vessel density, fractal dimension, width, and multiple tortuosity metrics for whole and centred regions | `Results/M3/Disc_Features.csv` and `Macular_Features.csv` |

## What v2 prepares automatically

For a complete M0→M3 run, the only irreplaceable user input is a flat set of
fundus images. v2 now owns the layout and intermediate-data plumbing that the
legacy shell pipeline expected users to prepare manually.

| Legacy requirement | v2 behavior |
|---|---|
| `images/` | User supplies the source photographs; nested-directory import is not silently guessed |
| `resolution_information.csv` | Preserved and validated when supplied; otherwise generated atomically with `0.008 mm/pixel` or the value passed to `--resolution-mm-per-pixel` |
| `AUTOMORPH_DATA` | Derived from positional `DATA_ROOT` and exported automatically to retained legacy helpers |
| `Results/` directory tree | Created by the owning stage |
| M0 crop metadata, M1 quality CSV/routes, M2 masks, and M3 centring tables | Generated in order by the one-command pipeline |
| Retipy configuration, model paths, ensemble order, batch settings, and device choice | Resolved from the packaged runtime and strict v2 contract |
| Model checkpoints | Supplied as release assets through Git LFS and verified by `models/manifest.json`; they cannot be generated |
| Patient labels, spreadsheets, optic/macula labels | Not required for inference |

The generated resolution value has real scientific consequences. M0 computes
`Scale_resolution = res × Scale × 1000`; M2 disc/cup dimensions and M3 width,
CRAE, CRVE, and some derived AVR values use that physical scale. Segmentation
pixels, quality routing, CDR ratios, vessel density, fractal dimension, and
tortuosity are not changed by the uniform value. v2 records the source and
warning in `.automorph_v2/resolution_provenance.json` and never infers retinal
scale from JPEG/TIFF DPI.

## Progress, logs, and safe runs

`--progress auto` uses tqdm bars with elapsed time, rate, and ETA on an
interactive terminal. In redirected output or CI it switches to throttled,
newline-based progress. The full pipeline asks each child stage for JSONL
progress events and renders them once in the parent, so Windows `spawn`
workers never fight over a terminal and persistent logs contain no ANSI or
carriage-return noise. `--progress plain`, `--progress bar`, and
`--progress off` provide explicit control; display mode does not participate
in numerical fingerprints or alter model execution.

Every unified invocation recreates `DATA_ROOT/run.log`, including failed and
interrupted runs. It contains the run
ID, argv, resolved configuration, platform and Python version, all stage
stdout/stderr, normalized progress snapshots, stage boundaries, elapsed time,
and the final success, interruption, or error. It remains readable after a
failed run. Raw per-stage logs remain at
`DATA_ROOT/.automorph_v2/logs/{m0,m1,m2,m3}.log`, while atomic manifests under
`.automorph_v2/` drive strict resume. An OS-level lock prevents two AutoMorph
v2 processes from writing the same `DATA_ROOT` concurrently.

## How each stage was optimized

The strict ship path follows one rule: an optimization is accepted only if the reference-platform comparator still reports the same observable result. Faster-but-different arithmetic belongs in an explicit experimental path, never behind a silent default.

### M0 — image standardization

Legacy M0 performs independent decoding, retinal-field detection, crop/pad work, resizing, and PNG writes serially. AutoMorph v2 schedules image-level jobs through an ordered cross-platform thread pool and caps the automatically selected worker count to avoid storage and memory oversubscription.

OpenCV CPU processing remains the default because the workload is dominated by image decoding, geometry, and lossless encoding; moving each image through CUDA added transfer and synchronization overhead and was slower in the accepted benchmark. CUDA and Apple MPS array backends remain available as explicit options.



PNG compression defaults to level 1 rather than the slower legacy level 3. This changes compressed bytes and file size, but not decoded pixels. Crop centre, radius, scale, and resolution values retain the legacy calculation and are compared independently. Per-image decode failures are recorded and skipped by default, matching the legacy pipeline's effective continuation behavior; `--fail-on-m0-image-error` turns any such record into a nonzero stage exit for stricter ingestion workflows.

### M1 — retinal image quality

M1 still uses all eight EfficientNet-B4 checkpoints, 512x512 non-zero-pixel normalization, batch size 64, FP32 inference, the original checkpoint order, and NumPy float32 mean/standard-deviation aggregation. It saves the three prediction classes separately and records the original gradability rule in `M1_quality_final.csv`.


The optimized loader uses shared-process preprocessing threads instead of repeatedly starting `DataLoader` worker processes, which is especially expensive on Windows and macOS. It begins preparing images while models are loading and overlaps preparation of the next batch with current GPU work.


The architecture is constructed without downloading redundant pretrained weights because every accepted parameter is replaced by the packaged checkpoint. All eight ensemble outputs stay on the accelerator until the batch is complete, followed by one ordered device-to-host transfer instead of eight synchronization points. Aggregation returns to CPU in the same order as legacy so probability strings and route decisions remain exact.

### M2 — vessel, artery/vein, and disc/cup ensembles

M2 is the heaviest inference stage and contains three different model families. AutoMorph v2 runs them through one ordered orchestrator so paths, device selection, memory telemetry, and failure behavior are consistent.

The binary-vessel branch keeps its ten accepted checkpoints resident for the high-throughput engine, pipelines ordered preprocessing and output materialization, and avoids repeated setup between ensemble members. An explicit `--vessel-engine low-vram` option streams more state when capacity matters more than speed.


The artery/vein branch loads the eight accepted checkpoint groups safely on CPU before device transfer, overlaps CPU preprocessing with inference, and uses an inference-only forward path that skips model heads whose values legacy immediately discarded. The retained heads, ensemble arithmetic, thresholds, and PNG materialization remain unchanged.

The disc/cup branch preserves all eight checkpoints, legacy batch shape, masks, morphology, centring rules, and downstream CSV serialization. Its main algorithmic improvement vectorizes the exact disc/cup intersection calculation and pipelines independent preprocessing/postprocessing work. The strict path deliberately keeps legacy CPU aggregation where moving the reduction changed low CUDA bits.

M2 host-side work scales to the automatically selected 8-thread budget on a high-end machine, but the model ensemble still determines VRAM demand. In the accepted 100-source run, M2 reported 9,858 MiB peak PyTorch reserved memory; the run-window whole-device monitor observed 13,706 MiB without per-process attribution. A 16 GiB GPU is the practical resident-profile recommendation. Lower-memory hosts should use the low-VRAM engine or CPU and validate the chosen path.

### M3 — Retipy vascular measurements

Legacy M3 repeatedly starts old Retipy drivers, rescans the same directory trees, reparses configuration, rebuilds image state, and evaluates independent measurement groups serially. Most of its runtime is orchestration and branch-heavy CPU analysis rather than dense tensor computation.


AutoMorph v2 performs one preflight, discovers and validates every required pair once, constructs immutable measurement tasks, and distributes 189 independent evaluations through a portable `spawn` process pool. Nested OpenCV/BLAS thread pools are limited to one thread per worker to prevent oversubscription. Results are merged in the original schema and row order.


The Retipy-derived formulas for fractal dimension, vessel density, average width, distance tortuosity, squared-curvature tortuosity, and tortuosity density remain the authority. GPU/NPU offload was not selected because transfer and kernel-launch overhead do not suit these small, irregular branches; eight CPU workers reduced the accepted stage from 470.685 s to 2.993 s with every output cell unchanged.

### Pipeline orchestration

The root CLI joins M0-M3 without merging their scientific contracts. It resolves portable paths, selects CUDA/MPS/CPU explicitly, applies one shared worker budget with per-stage overrides, launches each stage in a clean child process, and writes atomic stage manifests.

`--threads auto` resolves to 8 when at least 16 logical CPUs are available, 4 when at least 8 are available, otherwise the available count with a minimum of 1. Explicit values and `--m0-workers`, `--m1-threads`, `--m2-threads`, and `--m3-workers` always take precedence.

`--resume` skips a stage only after its inputs, implementation, model files, configuration, and outputs match the recorded fingerprints. `--overwrite` removes only pipeline-owned outputs. A normal fresh run refuses ambiguous existing output instead of mixing old and new artifacts.

## ONNX, TensorRT, AMP, and CUDA 12 availability

The goal is not merely similar masks; it is the accepted AutoMorph result contract. ONNX Runtime, TensorRT, quantization, AMP, TF32, channels-last execution, changed batch shapes, changed ensemble reduction, and approximate arithmetic can alter low bits, thresholds, serialized probabilities, or deployment support. They therefore remain experiments until each platform passes the same comparator.

A controlled 21-image M1 run illustrates the boundary. PyTorch 2.7.1+cu118 matched all 147/147 non-path output strings. A cu126 build preserved 21/21 predictions and routes but changed 126/147 probability or standard-deviation strings, with a maximum numeric delta of 0.00042416. CUDA 11.8 remains the strict reference; CUDA 12 can be offered only as a separately labeled, revalidated option.

## Compatibility and exactness

| Platform/backend | Execution support | Strict 1:1 status |
|---|---|---|
| Windows x86-64 + NVIDIA, cu118 | M0/M3 CPU; M1/M2 CUDA | **PASS** on Python 3.10.21 / Torch 2.7.1 |
| Windows x86-64 CPU | Supported fallback | Functional; strict parity not certified |
| Linux x86-64 + NVIDIA, cu118 | Supported by bootstrap and portable paths | Hardware gate still required |
| Linux x86-64 CPU | Supported fallback | Functional; strict parity not certified |
| macOS Apple Silicon CPU/MPS | Supported by device resolution and `spawn` | Functional design; Apple hardware gate required |

Platform support means the project has an installation, path, multiprocessing, and device fallback strategy. It does not mean different hardware kernels are bit-identical. Only a named environment that passes the full acceptance gate receives a strict 1:1 claim.

The recommended interpreter is Python 3.10. The source package supports Python 3.10-3.12, with PyTorch 2.7.1 and torchvision 0.22.1 pinned per backend. The NVIDIA reference uses cu118 because PyTorch wheels provide the CUDA runtime and the host only needs a compatible NVIDIA driver.

## Useful controls

```text
automorph-v2 run DATA_ROOT [options]
```

| Option | Effect |
|---|---|
| `--device VALUE` | Choose `auto`, `cpu`, `cuda`, `cuda:N`, or `mps` for M1/M2 |
| `--threads VALUE` | Use `auto` or a positive integer; automatic default selects 8 on high-end hosts |
| `--from-stage m0..m3` / `--to-stage m0..m3` | Run one contiguous part of the pipeline |
| `--m0-device VALUE` | Override M0's accepted CPU default |
| `--m0-workers N`, `--m1-threads N`, `--m2-threads N`, `--m3-workers N` | Override one stage |
| `--fail-on-m0-image-error` | Stop if any source image cannot be decoded; default records and skips it |
| `--png-compression 0..9` | Lossless PNG compression; accepted fast default is 1 |
| `--vessel-engine MODE` | Choose `resident` for speed or `low-vram` for lower memory |
| `--resolution-mm-per-pixel MM` | Uniform scale used only if the resolution CSV is absent; default `0.008` |
| `--require-calibrated-resolution` | Require a user-supplied physical calibration table instead of using an assumption |
| `--progress MODE` | Choose `auto`, `bar`, `plain`, or `off`; default `auto` |
| `--resume` | Verify fingerprints and skip valid completed stages |
| `--overwrite` | Replace only outputs owned by the selected stages |

The module wrappers retain the legacy-facing parameter surface and add v2 controls:

```powershell
.\.venv\Scripts\python.exe -m v2.run_m0 --help
.\.venv\Scripts\python.exe -m v2.run_m1 --help
.\.venv\Scripts\python.exe -m v2.run_m2 --help
.\.venv\Scripts\python.exe -m v2.run_m3 --help
```

## Output layout

```text
DATA_ROOT/
  images/
  resolution_information.csv       # supplied or generated before M0
  run.log                           # complete latest invocation
  .automorph_v2.run.lock            # cross-platform writer lock metadata
  Results/
    M0/
    M1/
    M2/
    M3/
  .automorph_v2/
    pipeline.json
    resolution_provenance.json
    logs/
    stages/
```

Use a dedicated data root. The pipeline refuses filesystem roots, the user home directory, the release checkout, and unsafe ancestors as output targets.

## Repository layout

```text
AutoMorph_v2/
  src/           # maintained automorph_v2 and v2 Python packages
  runtime/       # frozen legacy runtime assets and checkpoint payloads
  models/        # deterministic model/support manifest
  scripts/       # bootstrap, doctor, acceptance, and reporting tools
  requirements/  # pinned CPU, CUDA 11.8, macOS, and test profiles
  tests/         # portable release and contract tests
  docs/          # benchmark and ship evidence
```

The project-local `.venv` is intentionally ignored by Git. Virtual environments contain machine-specific paths and must be recreated on each operating system rather than copied between machines.

## Verification and release evidence

The final packaged suite passes 62/62 tests. The frozen per-module acceptance suites previously passed 189/189 checks before packaging, and no scientific calculation path changed afterward. The recreated local environment passes 21/21 doctor checks, `pip check`, full SHA-256 validation of 50 checkpoints plus 5 support assets, and the 89/89 strict comparator.

```powershell
.\.venv\Scripts\python.exe .\scripts\doctor.py --backend cu118
.\.venv\Scripts\python.exe .\scripts\build_model_manifest.py --check
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

## Models and licensing

The release manifest inventories 50 checkpoints plus 5 required support assets and records their SHA-256 hashes in [`models/manifest.json`](models/manifest.json). Large checkpoint files should be transported with Git LFS.

Source-code licensing does not automatically grant permission to redistribute every trained checkpoint. Before publishing a checkpoint-bundled fork or release, confirm redistribution rights for every model or provide an authorized acquisition process whose files pass the manifest. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), [`LICENSE`](LICENSE), and the [Retipy GPL-3.0-or-later notice](LICENSES/RETIPY_GPL-3.0.md).

## Acknowledgements

AutoMorph v2 exists because of the original [AutoMorph project](https://github.com/rmaphoh/AutoMorph) and its authors. If this software contributes to your work, please cite Yukun Zhou et al., “AutoMorph: Automated Retinal Vascular Morphology Quantification Via a Deep Learning Pipeline,” *Translational Vision Science & Technology* 11(7), 2022, [doi:10.1167/tvst.11.7.12](https://doi.org/10.1167/tvst.11.7.12). This repository is a downstream engineering and compatibility effort, not a replacement for the original scientific work.

This work is conducted by [ASU GSL Lab](https://github.com/LLM-VLM-GSL) and [Aztech Labs](https://github.com/Aztech-Lab) with Codex (GPT5.6 Sol)
