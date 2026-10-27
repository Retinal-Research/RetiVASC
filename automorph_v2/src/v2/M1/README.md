# AutoMorph M1 v2

M1 v2 preserves the legacy EyePACS quality contract: 512 x 512 preprocessing,
non-zero-pixel normalization, eight EfficientNet-B4 checkpoints in the same
order, NumPy mean/std aggregation, and the same Good/Bad routing rule.

The optimization keeps all eight model outputs on the accelerator and performs
one device-to-host transfer per batch instead of eight synchronized transfers.
Checkpoint construction uses `from_name`, because every model parameter is
immediately replaced by the fully matching checkpoint.

The accepted loader uses four shared-process threads. It begins preprocessing
the first batch while models and checkpoints load, then prepares the next batch
while the GPU is working. This avoids costly DataLoader process startup on
Windows and macOS without changing a single preprocessed tensor or model
operation.

## One-command run

From the `automorph` directory:

```bash
python -m v2.run_m1 /path/to/AUTOMORPH_DATA
```

To use eight preprocessing threads explicitly:

```bash
python -m v2.run_m1 /path/to/AUTOMORPH_DATA --threads 8
```

The input is `<AUTOMORPH_DATA>/Results/M0/images`. Outputs are written to
`<AUTOMORPH_DATA>/Results/M1` using the legacy names:

- `results_ensemble.csv`
- `M1_quality_final.csv` (per-image `Gradable` flag)
- `M1_quality_summary.csv` (three-class and gradability counts)
- `Good_quality/`
- `Usable_quality/`
- `Bad_quality/`

`Prediction` retains the original three classes (`0=good`, `1=usable`,
`2=bad`). `Gradable` retains AutoMorph's downstream rule: prediction 0, or
prediction 1 with `softmax_bad < 0.25`. The three directories contain the
three pure prediction classes; gradability is recorded in the CSV and is not
encoded by a directory name.

The command additionally writes `run_summary.json`. The public runner always
uses the accepted strict settings: FP32, batch 64, the original checkpoint
order, and CPU float32 aggregation in the original order. Only device choice
and preprocessing thread count are exposed. `auto` selects CUDA, then Apple
MPS, then CPU. The optional `--threads N` setting defaults to the validated
value 4; `--workers N` remains an alias for command compatibility.

The public command is also a compatibility superset of the legacy M1 Python
CLI. Its original spellings are accepted, including `--e`, `--b`,
`--train_on_dataset`, `--test_csv_dir`, `--n_class`, `--task_name`, and
`--local_rank`. Parameters that affect the validated numerical contract are
strictly locked to the production values: one epoch, batch 64,
`EyePACS_quality`, three classes, `Retinal_quality`, and `efficientnet`.
Unsupported values fail immediately instead of silently changing results.
The legacy round, dataset-label, and seed flags are accepted as no-ops.
`--test_csv_dir` overrides the input directory; the positional data root,
then `AUTOMORPH_DATA`, then the `automorph` directory determines the output
root. An indexed GPU can be selected with `--device cuda:N` or
`--local_rank N`.

Point the command only at a data root where `Results/M1` is intended to be
created or refreshed. Release tests use private, ignored fixture roots and
never write to a source dataset or legacy oracle.

Advanced experiments are available through `M1/cli.py`, including the old
process DataLoader, AMP, channels-last, alternate batch sizes, and the
legacy-sync engine. They are not part of the accepted exact path and must not
be treated as equivalent without a new alignment run.

## Validated input contract

M1 consumes the flat image directory produced by M0. Visible files must be
supported images and filename stems must be unique; ambiguous inputs now fail
early instead of silently producing a different image list from legacy.

## Acceptance artifacts

Benchmark images, per-image artifacts and the acceptance reports are not
included in this distribution.

On the recorded 99-image CUDA benchmark, the public command median is 7.662
seconds versus 24.388 seconds for legacy (3.183x). All seven non-path CSV
fields, Prediction, routing, directory membership, and copied image hashes are
exact.
CPU and Apple MPS are supported execution paths; bitwise equality across
different hardware backends is not claimed.
