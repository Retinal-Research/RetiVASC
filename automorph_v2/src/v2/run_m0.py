#!/usr/bin/env python3
'''One-command entry point for the isolated AutoMorph M0 v2 module.'''

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

if __package__:
    from .M0.core import RunConfig, run_m0
else:
    from M0.core import RunConfig, run_m0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run AutoMorph M0 v2 on one standard AUTOMORPH_DATA directory.'
    )
    parser.add_argument(
        'data', type=Path,
        help='directory containing images/ and resolution_information.csv',
    )
    parser.add_argument(
        '--device', choices=['cpu', 'cuda', 'mps', 'auto'], default='cpu',
        help='array backend; default: cpu (measured fastest for M0)',
    )
    parser.add_argument(
        '--workers', type=int, default=0,
        help='parallel image workers; default: min(8, available CPUs)',
    )
    parser.add_argument(
        '--png-compression', type=int, choices=range(10), default=1,
        help='lossless PNG level from 0 to 9; default: 1',
    )
    parser.add_argument('--force', action='store_true', help='reprocess existing outputs')
    parser.add_argument('--limit', type=int, default=0, help='process only the first N images')
    parser.add_argument(
        '--fail-on-image-error', action='store_true',
        help=(
            'return non-zero when individual images fail; by default failures are '
            'recorded and the legacy-compatible batch continues'
        ),
    )
    parser.add_argument(
        '--progress', choices=['auto', 'bar', 'plain', 'protocol', 'off'], default='auto',
        help='progress display; auto uses tqdm on a terminal and plain lines in logs',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    data = args.data.expanduser().resolve()
    summary = run_m0(RunConfig(
        input_dir=data / 'images',
        output_dir=data / 'Results' / 'M0' / 'images',
        resolution_csv=data / 'resolution_information.csv',
        device=args.device,
        workers=args.workers,
        png_compression=args.png_compression,
        force=args.force,
        limit=args.limit,
        progress=args.progress,
    ))
    print(
        'M0 v2 complete: {ok}/{total} successful, {failed} failed, '
        '{wall:.3f}s, {rate:.3f} images/s, backend={backend}'.format(
            ok=summary['image_count'] - summary['failure_count'],
            total=summary['image_count'],
            failed=summary['failure_count'],
            wall=summary['wall_s'],
            rate=summary['throughput_images_s'],
            backend=summary['backend'],
        )
    )
    return 1 if args.fail_on_image_error and summary['failure_count'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
