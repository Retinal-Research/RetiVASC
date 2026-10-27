# AutoMorph M0 v2

This directory is an isolated replacement for legacy `M0_Preprocess`. The
legacy files and pipeline entry points are not modified.

## One-command run

From the `automorph` directory:

```bash
python -m v2.run_m0 /path/to/AUTOMORPH_DATA
```

`AUTOMORPH_DATA` only needs `images/` and `resolution_information.csv`. The
command creates `Results/M0/images`, `crop_info.csv`, and diagnostic tables.
Its defaults are the accepted CPU configuration, so normal runs need no flags.

## What changed

- Image-level work runs concurrently with a cross-platform thread pool.
- PNG compression defaults to level 1 instead of OpenCV's slower level 3.
  Decoded pixels are unchanged; only file size/compression time changes.
- `--device cuda` and `--device mps` use Torch for the full-resolution
  mask/crop/pad array operation. Fundus geometry remains on OpenCV to preserve
  the legacy crop contract exactly.
- `--device auto` detects CUDA, then Apple MPS, and otherwise uses CPU.
- Per-image failures are recorded instead of being silently swallowed.
- Resume mode retains complete `crop_info.csv` rows for already completed
  images.
- Paths use `pathlib`; the same CLI works on Windows, Linux, and macOS.

## Compatible invocation

```bash
python -m v2.M0.cli --automorph-data /path/to/automorph_data
```

The output contract remains:

```text
<automorph_data>/Results/M0/images/*.png
<automorph_data>/Results/M0/crop_info.csv
```

Two extra diagnostics are written beside `crop_info.csv`:

```text
run_metrics.csv
run_summary.json
```

Useful controls:

```bash
# Default and measured-fastest configuration for M0
python -m v2.M0.cli --automorph-data DATA --device cpu --workers 8

# NVIDIA CUDA
python -m v2.M0.cli --automorph-data DATA --device cuda --workers 8

# Apple Silicon
python -m v2.M0.cli --automorph-data DATA --device mps --workers 4

# Explicit hardware auto-detection (CUDA, then MPS, then CPU)
python -m v2.M0.cli --automorph-data DATA --device auto
```

Level 1 PNG compression is lossless and is the recommended default. Use
`--png-compression 3` to match the legacy file-size/compression setting, or 0
for maximum write speed at substantially larger file size.

## Acceptance evidence

Benchmark images, per-image artifacts and the acceptance reports are not
included in this distribution.
