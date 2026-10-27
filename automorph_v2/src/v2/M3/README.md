# AutoMorph M3 v2

M3 v2 is a strict, CPU-parallel replacement for the legacy Retipy feature-analysis stage. It runs all six whole-image and Zone B/C groups, preserves the public CSV contract, and leaves the original M3/Retipy pipeline available as the comparison baseline.

## Run it

Run from the AutoMorph v2 source checkout with an isolated external data root:

```console
python -m v2.run_m3 /path/to/DATA_ROOT
```

The equivalent module launch is:

```console
python -m v2.run_m3 /path/to/DATA_ROOT
```

Four worker processes are used by default. On a machine with enough CPU and memory, select eight with either spelling:

```console
python -m v2.run_m3 /path/to/DATA_ROOT --workers 8
python -m v2.run_m3 /path/to/DATA_ROOT --threads 8
```

The legacy `-c/--configuration` option is retained:

```console
python -m v2.run_m3 /path/to/DATA_ROOT -c runtime/M3_feature_whole_pic/retipy/resources/retipy.config
```

The data root can also come from the legacy `AUTOMORPH_DATA` environment variable. For example, in PowerShell:

```powershell
$env:AUTOMORPH_DATA = "E:\path\to\DATA_ROOT"
python -m v2.run_m3
```

Or in a POSIX shell:

```sh
AUTOMORPH_DATA=/path/to/DATA_ROOT python -m v2.run_m3
```

Use `--allow-existing-output` only when an atomic replacement of existing M3 measurement and final CSV files is intended. Without it, the runner rejects a non-fresh output target. Run `python -m v2.run_m3 --help` for the complete interface.

As in the original six drivers, an exception limited to one image is recorded
and represented by an all-`-1` intermediate row (empty cells after the final
merge), so unusual sparse inputs keep the legacy result contract. Add
`--fail-on-metric-error` in validation jobs when any such row should abort the
run instead.

## Why this is CPU-only

M3 is dominated by branch-heavy connected-component traversal, local circle sampling, small polynomial fits, and CSV assembly. It is not a dense neural-network inference stage. Moving these operations to CUDA would introduce transfer, launch, ordering, and floating-point risks without a dependable speed benefit for the strict compatibility path.

`--device auto` and `--device cpu` therefore both select the CPU implementation. The runner does not allocate CUDA tensors or contexts, so attributable peak CUDA VRAM usage is **0 MiB**. Parallelism is across independent image/region/vessel tasks with worker processes, which also avoids the Python GIL; nested OpenCV and numerical-library thread pools are capped at one thread per worker to prevent oversubscription.

## Input contract

`<data_root>` must be a self-contained AutoMorph results tree. Preflight validates every input before creating outputs:

- `Results/M0/crop_info.csv`, containing unique `Name` and `Scale_resolution` columns.
- Paired PNG masks from `Results/M2/binary_vessel` and `Results/M2/artery_vein`. Every skeleton directory must have a process directory with exactly the same filenames.
- `Results/M3/Disc_centred/Disc_cup_results.csv` and `Results/M3/Macular_centred/Disc_cup_results.csv`. Whole-image mask names must match their table; Zone B/C masks may be a subset because legacy M2 keeps invalid disc/cup detections as metadata-only rows whose unavailable zone metrics remain blank after the legacy outer merge.

The six mask stems are:

| Centre | Region | M2 stem |
| --- | --- | --- |
| Disc | Zone B | `Zone_B_disc_centred` |
| Disc | Zone C | `Zone_C_disc_centred` |
| Macular | Zone B | `macular_Zone_B_centred` |
| Macular | Zone C | `macular_Zone_C_centred` |
| Macular | Whole | `macular_centred` |
| Disc | Whole | `disc_centred` |

Binary masks use `<stem>_binary_{skeleton,process}` below `binary_vessel`. Artery and vein masks use `<stem>_{artery,vein}_{skeleton,process}` below `artery_vein`.

## Output contract

The runner writes beneath `<data_root>/Results/M3` only:

- `Disc_centred/Disc_Zone_B_Measurement.csv`
- `Disc_centred/Disc_Zone_C_Measurement.csv`
- `Disc_centred/Disc_Measurement.csv`
- `Macular_centred/Macular_Zone_B_Measurement.csv`
- `Macular_centred/Macular_Zone_C_Measurement.csv`
- `Macular_centred/Macular_Measurement.csv`
- `Disc_Features.csv` and `Macular_Features.csv`, preserving the legacy 73-column final schema.
- `run_summary.json`, including platform, dependency versions, task counts, timings, and output paths.
- `run_errors.json`, empty after a successful run or populated before a metric failure is reported.

Unlike the legacy merge script, v2 does **not** delete the six intermediate measurement CSVs or the disc/cup inputs. CSV and JSON replacements are atomic.

## Compatibility and exactness

The implementation uses `pathlib` paths, an explicit portable `spawn` multiprocessing context, top-level picklable workers, and GUI-free image processing. With compatible NumPy, pandas, SciPy, scikit-image, Pillow, and `opencv-python-headless` installations, the functional target is Windows, Linux, and both Intel and Apple-silicon macOS.

Compatibility has two levels:

1. **Functional portability:** the same command, input contract, schema, ordering, and failure policy on all three operating-system families.
2. **Strict result equivalence:** compare every intermediate and final CSV against the legacy oracle with the same Python and dependency versions. Same-environment results are required to match exactly. Cross-platform bitwise identity is not claimed without a gate on that platform because BLAS/LAPACK polynomial fitting and CSV float formatting can vary by build even when the algorithm is unchanged.

Before accepting a platform/dependency combination, run the oracle gate and check schema, row/name order, missing-value positions, every finite numeric value, and final file hashes where byte identity is expected.

## What was optimized

- Vessel extraction preserves legacy row-major seeds, FIFO traversal, neighbor order, junction removal footprint, and emitted coordinate order while vectorizing the expensive neighborhood classification.
- Zone-width measurement caches exact integer circle offsets and clips them at image boundaries while retaining the legacy first-failure, zero-count, diameter, point-skip, and resolution behavior.
- All six groups and binary/artery/vein branches share one preflighted engine instead of six import-time scripts.
- Independent evaluations run in a portable process pool; result assembly remains in deterministic legacy group and filename order.

The legacy code is intentionally retained outside `v2` for regression and performance comparisons.

## Validated 21-image benchmark

The legacy compatibility run completed in **470.685056 s**. Another brief test overlapped that run, so its timing is qualified as a parity-oracle reference rather than a clean standalone performance baseline.

| Implementation | Workers | Evaluations | Wall time | Speedup vs. oracle | Metric failures | CUDA VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy parity oracle | 1 | 189 | 470.685056 s | 1.00x | 0 | 0 MiB |
| M3 v2 strict CPU | 1 | 189 | 21.298529 s | 22.10x | 0 | 0 MiB |
| M3 v2 strict CPU | 4 | 189 | 4.601569 s | 102.29x | 0 | 0 MiB |
| M3 v2 strict CPU | 8 | 189 | 2.625588 s | 179.27x | 0 | 0 MiB |

For all three v2 runs, all six intermediate measurement CSVs and both final feature CSVs were SHA-256 byte-for-byte identical to the legacy parity oracle. Every v2 measurement remained CPU-only and allocated no module CUDA VRAM. The one-worker row isolates the algorithmic gain; the four- and eight-worker rows add task-level parallelism.
