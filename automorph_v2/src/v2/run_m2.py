#!/usr/bin/env python3
"""One-command entry point for the isolated AutoMorph M2 module."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

if __package__:
    from .M2.core import ALL_MODULES, AUTOMORPH_ROOT, RELEASE_ROOT, M2Config, run_m2
else:
    from M2.core import ALL_MODULES, AUTOMORPH_ROOT, RELEASE_ROOT, M2Config, run_m2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all three strict-compatible AutoMorph M2 branches"
    )
    parser.add_argument(
        "data",
        type=Path,
        nargs="?",
        help="AUTOMORPH_DATA containing Results/M1/Good_quality",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:N, or mps (default: auto)",
    )
    parser.add_argument(
        "--threads",
        "--workers",
        dest="threads",
        type=int,
        default=4,
        help="shared-process preprocessing/postprocessing threads (default: 4)",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=ALL_MODULES,
        default=list(ALL_MODULES),
        help="ordered subset to run (default: all three)",
    )
    parser.add_argument("--vessel-weights-dir", type=Path)
    parser.add_argument(
        "--input-quality-file",
        type=Path,
        help="optional txt/yaml listing M1 quality dirs to process (one per line); "
        "defaults to DATA_ROOT/m2_input_quality.txt or Good/Usable/Bad",
    )
    parser.add_argument(
        "--vessel-engine",
        choices=("resident", "low-vram"),
        default="resident",
        help="resident is fastest; low-vram trades host RAM/time for a smaller CUDA peak",
    )
    parser.add_argument("--artery-vein-weights-dir", type=Path)
    parser.add_argument("--disc-cup-module-dir", type=Path)
    parser.add_argument(
        "--intersection-backend",
        choices=("vectorized", "reference"),
        default="vectorized",
        help="exact disc/cup intersection implementation",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="overwrite matching files in non-empty stage directories (may retain stale files)",
    )
    parser.add_argument(
        "--progress",
        choices=("auto", "bar", "plain", "protocol", "off"),
        default="auto",
        help="progress display; auto uses tqdm on a terminal and plain lines in logs",
    )

    # Compatibility superset of all three original M2 Python CLIs. Options
    # that affect the accepted pixels are validated below; training-only and
    # ignored legacy values remain accepted no-ops.
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--load", default=False)
    parser.add_argument("--discriminator", default="unet")
    parser.add_argument("--jn", default="20210630_uniform_thres40_ALL-SIX")
    parser.add_argument("--worker_num", type=int)
    parser.add_argument("--save_model", default="best")
    parser.add_argument("--train_test_mode", default="test")
    parser.add_argument("--pre_threshold", type=float, default=40.0)
    parser.add_argument(
        "--dataset",
        choices=("ALL-SIX", "ALL-AV"),
        help="accepted legacy per-branch spelling; v2 fixes both datasets",
    )
    parser.add_argument("--seed_num", type=int, default=42)
    parser.add_argument("--dataset_test", default="ALL-SIX")
    parser.add_argument("--validation_ratio", type=float, default=10.0)
    parser.add_argument("--uniform", default="True")
    parser.add_argument("--out_test", type=Path)
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--beta", type=float, default=1.1)
    parser.add_argument("--gamma", type=float, default=0.5)

    parser.add_argument("--batch-size", dest="av_batch_size", type=int, default=8)
    parser.add_argument("--job_name", default="20210724_ALL-AV_randomseed")
    parser.add_argument("--checkstart", type=int, default=1401)

    parser.add_argument("--experiment_path")
    parser.add_argument("--config_file", type=Path)
    parser.add_argument("--im_size", default="512")
    parser.add_argument("--results_path", default="results")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.worker_num is not None:
        if args.threads != 4 and args.threads != args.worker_num:
            parser.error("--worker_num conflicts with --threads/--workers")
        args.threads = args.worker_num
    if args.threads < 1:
        parser.error("--threads must be at least one")

    fixed = (
        ("epochs", 1, "--epochs"),
        ("batchsize", 8, "--batchsize"),
        ("learning_rate", 2e-4, "--learning_rate"),
        ("discriminator", "unet", "--discriminator"),
        ("jn", "20210630_uniform_thres40_ALL-SIX", "--jn"),
        ("save_model", "best", "--save_model"),
        ("train_test_mode", "test", "--train_test_mode"),
        ("pre_threshold", 40.0, "--pre_threshold"),
        ("seed_num", 42, "--seed_num"),
        ("dataset_test", "ALL-SIX", "--dataset_test"),
        ("validation_ratio", 10.0, "--validation_ratio"),
        ("alpha", 0.08, "--alpha"),
        ("beta", 1.1, "--beta"),
        ("gamma", 0.5, "--gamma"),
        ("av_batch_size", 8, "--batch-size"),
        ("job_name", "20210724_ALL-AV_randomseed", "--job_name"),
        ("checkstart", 1401, "--checkstart"),
        ("im_size", "512", "--im_size"),
    )
    for attribute, expected, option in fixed:
        actual = getattr(args, attribute)
        if actual != expected:
            parser.error(
                f"{option} is fixed at {expected!r} by strict M2; got {actual!r}"
            )
    if str(args.uniform).lower() != "true":
        parser.error("--uniform is fixed at True by strict M2")
    return args


def resolve_data_root(args: argparse.Namespace) -> Path:
    value = args.data or os.getenv("AUTOMORPH_DATA") or RELEASE_ROOT
    return Path(value).expanduser().resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = resolve_data_root(args)
    expected_vessel_output = data_root / "Results" / "M2" / "binary_vessel"
    if args.out_test is not None:
        requested = args.out_test.expanduser().resolve()
        if requested != expected_vessel_output.resolve():
            raise SystemExit(
                "--out_test must resolve to the strict output directory: "
                f"{expected_vessel_output}"
            )

    summary = run_m2(
        M2Config(
            data_root=data_root,
            device=args.device,
            threads=args.threads,
            modules=tuple(args.modules),
            vessel_weights_dir=args.vessel_weights_dir,
            vessel_engine=args.vessel_engine,
            artery_vein_weights_dir=args.artery_vein_weights_dir,
            disc_cup_module_dir=args.disc_cup_module_dir,
            intersection_backend=args.intersection_backend,
            allow_existing_outputs=args.allow_existing_output,
            progress=args.progress,
            input_quality_file=args.input_quality_file,
        )
    )
    peak_mib = summary["peak_cuda_reserved_bytes"] / 2**20
    print(
        f"M2 v2 complete: {summary['total_wall_s']:.3f}s, "
        f"threads={summary['threads']}, peak CUDA reserved={peak_mib:.1f} MiB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
