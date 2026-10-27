# AutoMorph M2 v2

M2 v2 keeps the required AutoMorph model definitions and checkpoints intact,
but runs the three inference stages through one isolated, strict-FP32 entry
point. Run the command from the AutoMorph v2 source checkout:

```text
python -m v2.run_m2 DATA_ROOT
```

For example, a machine with enough CPU capacity can use eight host threads:

```text
python -m v2.run_m2 /path/to/DATA_ROOT --device auto --threads 8
```

The default is `--threads 4`.  Threads overlap CPU image decoding,
preprocessing, and postprocessing; they do not change the fixed ensemble,
checkpoint order, model batch sizes, or FP32 arithmetic contract.  Increasing
the value is useful only while CPU work is a bottleneck.

An exact but slower vessel memory mode is also available:

```text
python -m v2.run_m2 DATA_ROOT --vessel-engine low-vram
```

The default `resident` engine is recommended.  On the accepted 21-image run,
`low-vram` reduced the whole-device peak by only 368 MiB while adding about
6.4 seconds to the vessel stage and about 866 MiB of CPU cache.

## Input and output contract

`DATA_ROOT` is a writable run directory, not the AutoMorph source directory.
Before a complete run it must contain:

```text
DATA_ROOT/
  Results/
    M0/
      crop_info.csv
    M1/
      Good_quality/
        <accepted M1 images>
```

The command executes these stages in order:

1. `vessel`: the ten-model binary-vessel ensemble.
2. `artery-vein`: the eight-seed main/artery/vein ensemble.
3. `disc-cup`: the eight-model optic-disc/cup ensemble and its legacy M3
   centring/zone postprocessing.

The principal outputs are:

```text
DATA_ROOT/Results/M2/binary_vessel/
DATA_ROOT/Results/M2/artery_vein/
DATA_ROOT/Results/M2/optic_disc_cup/
DATA_ROOT/Results/M2/run_summary.json
DATA_ROOT/Results/M3/
```

`run_summary.json` records the resolved platform, PyTorch version, requested
device, thread count, per-stage timing, and PyTorch CUDA peak allocation and
reservation when CUDA is used.

The default weight locations remain the original, read-only AutoMorph trees:

```text
runtime/M2_Vessel_seg/Saved_model/
runtime/M2_Artery_vein/ALL-AV/
runtime/M2_lwnet_disc_cup/experiments/
```

M2 v2 reads model definitions and weights from those locations; it does not
modify them.  `--vessel-weights-dir`, `--artery-vein-weights-dir`, and
`--disc-cup-module-dir` are available when an installation stores those assets
elsewhere.  All generated data belongs below `DATA_ROOT`.

## Device and stage selection

`--device` accepts:

- `auto`: CUDA first, then Apple MPS, then CPU.
- `cpu`: portable path for Windows, Linux, and macOS.
- `cuda` or `cuda:N`: NVIDIA CUDA on Windows or Linux.
- `mps`: Apple Metal on supported macOS systems.

The full pipeline is the default.  An ordered subset can be selected, for
example:

```text
python -m v2.run_m2 DATA_ROOT --modules vessel artery-vein
```

Names must preserve the production order: `vessel`, `artery-vein`,
`disc-cup`.  Running `disc-cup` requires the binary-vessel and artery/vein
process outputs already present under the same `DATA_ROOT`.

## Existing-output protection

By default, a selected stage refuses to start when one of its owned output
trees already contains files.  This prevents a new run from silently mixing
with an earlier result set.  The safest invocation uses a fresh `DATA_ROOT`.

`--allow-existing-output` disables that guard and permits matching files to be
overwritten.  It does **not** clean the output tree, so stale files for images
that are no longer in the input can remain.  Use it only for an intentional
resume/re-run whose directory contents have already been checked.

## Pixel-equivalence boundary

The default path is the strict contract: FP32 inference, fixed production
batch sizes, original checkpoint/ensemble order, original reduction order, and
legacy output quantisation.  Acceptance against a legacy reference means the
same relative output files, shapes, modes, and `0` changed decoded pixels.

That contract prevents intentional algorithm drift; it cannot make different
hardware and numerical backends inherently bitwise identical.  CUDA, MPS, and
CPU kernels (and differing PyTorch, CUDA, Pillow, or OpenCV builds) can round
floating-point values differently.  Therefore each deployment platform must
pass the pixel comparison gate before its results are accepted.  If a target
platform fails that gate, use the already validated software/device stack or
treat that backend as unsupported for strict production rather than accepting
approximately matching pixels.

## Legacy CLI compatibility

The wrapper accepts every option spelling exposed by the three original M2
Python entry points.  Active v2 controls can be added without removing those
spellings.

The following legacy options are accepted but fixed to the production values
that define strict pixels:

| Option | Required strict value |
|---|---:|
| `--epochs` | `1` |
| `--batchsize` | `8` |
| `--learning_rate` | `0.0002` |
| `--discriminator` | `unet` |
| `--jn` | `20210630_uniform_thres40_ALL-SIX` |
| `--save_model` | `best` |
| `--train_test_mode` | `test` |
| `--pre_threshold` | `40` |
| `--seed_num` | `42` |
| `--dataset_test` | `ALL-SIX` |
| `--validation_ratio` | `10` |
| `--uniform` | `True` |
| `--alpha`, `--beta`, `--gamma` | `0.08`, `1.1`, `0.5` |
| `--batch-size` | `8` |
| `--job_name` | `20210724_ALL-AV_randomseed` |
| `--checkstart` | `1401` |
| `--im_size` | `512` |

A different value is rejected before inference.  In particular, model batch
sizes remain fixed even when `--threads 8` is used.

The remaining original spellings are also retained:

- `--worker_num` is an alias for `--threads`; `--workers` is the additional v2
  alias.
- `--device` is the active cross-platform selector described above.
- `--dataset` accepts `ALL-SIX` or `ALL-AV`; the individual stage contracts
  remain fixed internally.
- `--out_test` is accepted only when it resolves to
  `DATA_ROOT/Results/M2/binary_vessel`.
- `--load`, `--experiment_path`, `--config_file`, and `--results_path` are
  compatibility no-ops during strict inference.

Use `python -m v2.run_m2 --help` for the complete combined interface.

## Recorded 21-image acceptance

The frozen module benchmark used Windows 11 with an RTX 4090, Python 3.12.7,
and PyTorch 2.7.1+cu118. Its input was a private 21-image acceptance fixture
selected from the upstream M1 validation run.

| Path | Threads | Wall time | Speedup vs legacy | Whole-device peak |
|---|---:|---:|---:|---:|
| Legacy CUDA-compatible | 8-worker legacy loaders | 230.768 s | 1.00x | 10,702 MiB |
| M2 v2 strict default | 4 | 35.374 s | 6.52x | 10,264 MiB |
| M2 v2 strict high-CPU | 8 | 29.058 s | 7.94x | 10,264 MiB |

For the default four-thread path, all **882/882** M2 PNG files were both
decoded-pixel exact and SHA-256 byte exact against the isolated
legacy-compatible baseline; both M3 CSV files were byte exact.  The eight
thread path was independently SHA-256 exact for the same 882 PNG and two CSV
files.  PyTorch's internal peak for the accepted v2 run was 7,506.9 MiB
allocated and 9,858.0 MiB reserved; the table reports the independently polled
`nvidia-smi` whole-device peak after confirming the GPU was idle.

Benchmark images, per-image galleries and the acceptance reports are not
included in this distribution. Windows CUDA is the platform validated here.
Linux CUDA/CPU and macOS CPU/MPS are supported execution paths, but each
distinct hardware/software stack must pass its own pixel gate before being
called bitwise compatible.
