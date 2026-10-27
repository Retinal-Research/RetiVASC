"""Public command-line interface for the complete AutoMorph v2 pipeline."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
import sys
from typing import Sequence

from . import __version__
from .pipeline import PIPELINE_STAGES, PipelineConfig, PipelineError, run_pipeline


_DEVICE = re.compile(r"^(?:auto|cpu|cuda|mps|cuda:[0-9]+)$", re.IGNORECASE)


def _device(value: str) -> str:
    normalized = value.lower()
    if not _DEVICE.fullmatch(normalized):
        raise argparse.ArgumentTypeError(
            "expected auto, cpu, cuda, cuda:N, or mps"
        )
    return normalized


def _positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive number") from error
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return number


def _thread_budget(value: str) -> int | str:
    if value.lower() == "auto":
        return "auto"
    return _positive_integer(value)


def recommended_thread_count(logical_cpus: int | None = None) -> int:
    """Return a conservative cross-platform worker budget for the host."""
    available = os.cpu_count() if logical_cpus is None else logical_cpus
    available = max(1, int(available or 1))
    if available >= 16:
        return 8
    if available >= 8:
        return 4
    return available


def _resolved_thread_count(value: int | str) -> int:
    return recommended_thread_count() if value == "auto" else int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="automorph-v2",
        description="Run the strict-compatible AutoMorph v2 pipeline.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run",
        help="run an ordered M0 -> M3 pipeline or a contiguous stage range",
    )
    run.add_argument(
        "data_root",
        type=Path,
        nargs="?",
        help="writable data root containing images/ (resolution CSV is auto-generated if absent)",
    )
    run.add_argument(
        "--device",
        type=_device,
        default="auto",
        help="M1/M2 device: auto, cpu, cuda, cuda:N, or mps (default: auto)",
    )
    run.add_argument(
        "--threads",
        type=_thread_budget,
        default="auto",
        metavar="AUTO|N",
        help=(
            "host workers for every stage: auto or a positive integer "
            "(default: auto; resolves to 8 on hosts with >=16 logical CPUs)"
        ),
    )
    run.add_argument(
        "--m0-device",
        type=str.lower,
        choices=("auto", "cpu", "cuda", "mps"),
        default="cpu",
        help="M0 array backend; CPU is the accepted fastest default",
    )
    run.add_argument("--m0-workers", type=_positive_integer)
    run.add_argument("--m1-threads", type=_positive_integer)
    run.add_argument("--m2-threads", type=_positive_integer)
    run.add_argument("--m3-workers", type=_positive_integer)
    run.add_argument(
        "--png-compression",
        type=int,
        choices=range(10),
        default=1,
        help="lossless M0 PNG compression level (default: 1)",
    )
    run.add_argument(
        "--vessel-engine",
        choices=("resident", "low-vram"),
        default="resident",
        help="strict M2 vessel implementation (default: resident)",
    )
    run.add_argument(
        "--resolution-mm-per-pixel",
        type=_positive_float,
        default=0.008,
        metavar="MM",
        help=(
            "uniform original-image scale used only when resolution_information.csv "
            "is absent (default: legacy Topcon approximation 0.008 mm/pixel)"
        ),
    )
    run.add_argument(
        "--require-calibrated-resolution",
        action="store_true",
        help="fail instead of auto-generating a missing resolution_information.csv",
    )
    run.add_argument(
        "--fail-on-m0-image-error",
        action="store_true",
        help=(
            "stop the pipeline when any M0 image fails; default records the "
            "failure and continues with valid images like legacy AutoMorph"
        ),
    )
    run.add_argument(
        "--progress",
        choices=("auto", "bar", "plain", "off"),
        default="auto",
        help="progress display: interactive bar, plain log lines, or off (default: auto)",
    )
    run.add_argument(
        "--from-stage",
        "--from",
        dest="from_stage",
        choices=PIPELINE_STAGES,
        default=PIPELINE_STAGES[0],
        help="first stage to run (default: m0)",
    )
    run.add_argument(
        "--to-stage",
        "--to",
        dest="to_stage",
        choices=PIPELINE_STAGES,
        default=PIPELINE_STAGES[-1],
        help="last stage to run, inclusive (default: m3)",
    )
    mode = run.add_mutually_exclusive_group()
    mode.add_argument(
        "--fresh",
        dest="mode",
        action="store_const",
        const="fresh",
        help="refuse any selected-stage output (the default)",
    )
    mode.add_argument(
        "--resume",
        dest="mode",
        action="store_const",
        const="resume",
        help="skip only completed stages whose fingerprints still match",
    )
    mode.add_argument(
        "--overwrite",
        dest="mode",
        action="store_const",
        const="overwrite",
        help="delete selected-stage outputs below DATA_ROOT, then run",
    )
    run.set_defaults(mode="fresh")
    return parser


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    data_value = args.data_root or os.environ.get("AUTOMORPH_DATA")
    if data_value is None:
        parser.error("DATA_ROOT is required unless AUTOMORPH_DATA is set")

    start = PIPELINE_STAGES.index(args.from_stage)
    stop = PIPELINE_STAGES.index(args.to_stage)
    if start > stop:
        parser.error("--from-stage must not come after --to-stage")

    resolved_threads = _resolved_thread_count(args.threads)
    config = PipelineConfig(
        data_root=Path(data_value),
        device=args.device,
        threads=resolved_threads,
        m0_device=args.m0_device,
        m0_workers=args.m0_workers,
        m1_threads=args.m1_threads,
        m2_threads=args.m2_threads,
        m3_workers=args.m3_workers,
        png_compression=args.png_compression,
        vessel_engine=args.vessel_engine,
        resolution_mm_per_pixel=args.resolution_mm_per_pixel,
        require_calibrated_resolution=args.require_calibrated_resolution,
        fail_on_m0_image_error=args.fail_on_m0_image_error,
        progress=args.progress,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        mode=args.mode,
    )
    thread_source = "auto" if args.threads == "auto" else "explicit"
    print(
        f"AutoMorph v2 configuration: threads={resolved_threads} "
        f"({thread_source}); device={args.device}",
        flush=True,
    )
    summary = run_pipeline(config)
    completed = ", ".join(summary["completed_stages"]) or "none"
    skipped = ", ".join(summary["skipped_stages"]) or "none"
    print(
        f"AutoMorph v2 complete: completed={completed}; skipped={skipped}; "
        f"wall={summary['wall_s']:.3f}s",
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run(args, parser)
    except PipelineError as error:
        print(f"automorph-v2: error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("automorph-v2: interrupted", file=sys.stderr)
        return 130
    parser.error(f"unknown command: {args.command}")
    return 2


__all__ = ["build_parser", "main", "recommended_thread_count"]


if __name__ == "__main__":
    raise SystemExit(main())
