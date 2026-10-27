#!/usr/bin/env python3
"""One-command, cross-platform entry point for AutoMorph M3 v2."""

from __future__ import annotations

import argparse
import multiprocessing
import os
from pathlib import Path
from typing import Sequence


RELEASE_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = RELEASE_ROOT / "runtime"
DEFAULT_CONFIGURATION = (
    RUNTIME_ROOT
    / "M3_feature_whole_pic"
    / "retipy"
    / "resources"
    / "retipy.config"
)

_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run every AutoMorph M3 whole-picture and Zone B/C feature "
            "measurement with the strict, cross-platform CPU engine."
        )
    )
    parser.add_argument(
        "data_root",
        nargs="?",
        type=Path,
        help=(
            "isolated AutoMorph data root containing Results/M0, Results/M2, "
            "and the M3 Disc_cup_results.csv inputs; may instead be supplied "
            "through the legacy AUTOMORPH_DATA environment variable"
        ),
    )
    parser.add_argument(
        "-c",
        "--configuration",
        type=Path,
        default=DEFAULT_CONFIGURATION,
        help="legacy retipy.config path (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        "--threads",
        dest="workers",
        type=int,
        default=4,
        help="parallel CPU worker processes; use 8 on a high-end machine (default: 4)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu"),
        default="auto",
        help=(
            "execution device; M3 is branch-heavy CPU analysis, so auto and "
            "cpu both select the exact CPU implementation (default: auto)"
        ),
    )
    parser.add_argument(
        "--start-method",
        choices=tuple(multiprocessing.get_all_start_methods()),
        default="spawn",
        help="multiprocessing start method (portable default: spawn)",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="atomically replace existing M3 measurement/final CSV files",
    )
    failure_group = parser.add_mutually_exclusive_group()
    failure_group.add_argument(
        "--fail-on-metric-error",
        dest="allow_metric_failures",
        action="store_false",
        help=(
            "abort instead of preserving legacy all--1 rows when an individual "
            "image metric raises"
        ),
    )
    failure_group.add_argument(
        "--allow-metric-failures",
        dest="allow_metric_failures",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(allow_metric_failures=True)
    parser.add_argument(
        "--progress",
        choices=("auto", "bar", "plain", "protocol", "off"),
        default="auto",
        help="progress display; auto uses tqdm on a terminal and plain lines in logs",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="legacy alias that forces --progress off",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.data_root is None:
        legacy_data_root = os.environ.get("AUTOMORPH_DATA")
        if not legacy_data_root:
            parser.error(
                "data_root is required unless the legacy AUTOMORPH_DATA "
                "environment variable is set"
            )
        args.data_root = Path(legacy_data_root)
    if args.workers < 1:
        parser.error("--workers/--threads must be at least 1")

    # Set limits before importing NumPy/OpenCV through the engine. Each M3
    # task is already one independent process; nested BLAS/OpenCV pools only
    # oversubscribe CPUs and make timings less stable.
    for name in _THREAD_ENVIRONMENT:
        os.environ[name] = "1"

    if __package__:
        from .M3.engine import M3Config, run_m3
    else:
        from M3.engine import M3Config, run_m3

    summary = run_m3(
        M3Config(
            data_root=args.data_root,
            workers=args.workers,
            configuration=args.configuration,
            allow_existing_output=args.allow_existing_output,
            allow_metric_failures=args.allow_metric_failures,
            start_method=args.start_method,
            quiet=args.quiet,
            progress="off" if args.quiet else args.progress,
        )
    )
    print(
        "M3 complete: "
        f"{summary['preflight']['retipy_evaluations']} evaluations, "
        f"{summary['workers']} CPU workers, "
        f"{summary['total_wall_s']:.3f}s, "
        f"failures={summary['metric_failures']}"
    )
    print(f"Disc features: {summary['final_outputs']['disc']}")
    print(f"Macular features: {summary['final_outputs']['macular']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
