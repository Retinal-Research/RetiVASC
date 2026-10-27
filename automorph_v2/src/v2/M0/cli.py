"""Command-line entry point for AutoMorph v2 M0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .core import RunConfig, run_m0
except ImportError:  # direct ``python cli.py`` invocation
    from core import RunConfig, run_m0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast legacy-compatible AutoMorph M0")
    parser.add_argument(
        "--automorph-data",
        type=Path,
        help="derive images/, resolution_information.csv, and Results/M0 from this root",
    )
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resolution-csv", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--workers", type=int, default=0, help="0 chooses up to 8 workers")
    parser.add_argument("--png-compression", type=int, default=1, choices=range(10))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.automorph_data:
        root = args.automorph_data.expanduser().resolve()
        input_dir = args.input_dir or root / "images"
        output_dir = args.output_dir or root / "Results" / "M0" / "images"
        resolution_csv = args.resolution_csv or root / "resolution_information.csv"
    else:
        if not (args.input_dir and args.output_dir and args.resolution_csv):
            raise SystemExit(
                "provide --automorph-data, or all of --input-dir/--output-dir/--resolution-csv"
            )
        input_dir, output_dir, resolution_csv = (
            args.input_dir,
            args.output_dir,
            args.resolution_csv,
        )

    summary = run_m0(
        RunConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            resolution_csv=resolution_csv,
            device=args.device,
            workers=args.workers,
            png_compression=args.png_compression,
            force=args.force,
            limit=args.limit,
        )
    )
    print(json.dumps(summary, indent=2))
    return 1 if summary["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
