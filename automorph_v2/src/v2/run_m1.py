#!/usr/bin/env python3
'''One-command entry point for the isolated AutoMorph M1 v2 module.'''

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

if __package__:
    from .M1.core import M1Config, run_m1
else:
    from M1.core import M1Config, run_m1

HERE = Path(__file__).resolve().parent
RELEASE_ROOT = HERE.parents[1]
RUNTIME_ROOT = RELEASE_ROOT / "runtime"
DEFAULT_WEIGHTS = (
    RUNTIME_ROOT / 'M1_Retinal_Image_quality_EyePACS' /
    'Retinal_quality' / 'EyePACS_quality' / 'efficientnet'
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run AutoMorph M1 v2 quality assessment')
    parser.add_argument(
        'data', type=Path, nargs='?',
        help='AUTOMORPH_DATA containing Results/M0/images',
    )
    parser.add_argument(
        '--device', default='auto',
        help='auto, cpu, cuda, cuda:N, or mps (default: auto)',
    )
    parser.add_argument(
        '--threads', '--workers', dest='threads', type=int, default=4,
        help='shared-process preprocessing threads (default: 4)',
    )
    parser.add_argument('--weights-dir', type=Path, default=DEFAULT_WEIGHTS)

    # The legacy script exposed training-era flags even though production M1
    # always invoked it with the fixed values below. Accept those spellings so
    # existing launch commands remain usable, but reject values outside the
    # exact configuration validated by v2.
    parser.add_argument('-e', '--e', '--epochs', dest='epochs', type=int, default=1)
    parser.add_argument(
        '-b', '--b', '--batch-size', dest='batch_size', type=int, default=64,
    )
    parser.add_argument(
        '-f', '--train_on_dataset', dest='train_on_dataset',
        default='EyePACS_quality',
    )
    parser.add_argument('-dir', '--test_csv_dir', dest='test_csv_dir', type=Path)
    parser.add_argument(
        '-n_class', '--n_class', '--n_classes', dest='n_classes',
        type=int, default=3,
    )
    parser.add_argument(
        '-d', '--test_on_dataset', dest='test_on_dataset',
        default='customised_data',
    )
    parser.add_argument(
        '-t', '--task_name', dest='task_name', default='Retinal_quality',
    )
    parser.add_argument('-r', '--round', dest='round', type=int, default=0)
    parser.add_argument('-m', '--model', dest='model', default='efficientnet')
    parser.add_argument('--seed_num', type=int, default=42)
    parser.add_argument('--local_rank', '--local-rank', dest='local_rank', type=int)
    parser.add_argument(
        '--progress', choices=['auto', 'bar', 'plain', 'protocol', 'off'], default='auto',
        help='progress display; auto uses tqdm on a terminal and plain lines in logs',
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    fixed_values = (
        ('epochs', 1, '--epochs'),
        ('batch_size', 64, '--batch-size'),
        ('train_on_dataset', 'EyePACS_quality', '--train_on_dataset'),
        ('n_classes', 3, '--n_classes'),
        ('task_name', 'Retinal_quality', '--task_name'),
        ('model', 'efficientnet', '--model'),
    )
    for attribute, expected, option in fixed_values:
        actual = getattr(args, attribute)
        if actual != expected:
            parser.error(
                f'{option} is fixed at {expected!r} by the strict M1 v2 contract; '
                f'got {actual!r}'
            )

    if args.local_rank is not None:
        if args.local_rank < 0:
            parser.error('--local_rank must be a non-negative CUDA device index')
        requested = args.device.lower()
        if requested in {'cpu', 'mps'}:
            parser.error('--local_rank cannot be combined with an explicit CPU or MPS device')
        ranked_device = f'cuda:{args.local_rank}'
        if requested.startswith('cuda:') and requested != ranked_device:
            parser.error(
                f'--local_rank {args.local_rank} conflicts with --device {args.device!r}'
            )
        if requested not in {'auto', 'cuda', ranked_device}:
            parser.error(f'--local_rank cannot be combined with --device {args.device!r}')
        args.device = ranked_device
    return args


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    data_value = args.data or os.getenv('AUTOMORPH_DATA') or RELEASE_ROOT
    data = Path(data_value).expanduser().resolve()
    input_dir = (
        args.test_csv_dir.expanduser().resolve()
        if args.test_csv_dir is not None
        else data / 'Results' / 'M0' / 'images'
    )
    return data, input_dir, data / 'Results' / 'M1'


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _data, input_dir, output_dir = resolve_paths(args)
    summary = run_m1(M1Config(
        input_dir=input_dir,
        output_dir=output_dir,
        weights_dir=args.weights_dir,
        device=args.device,
        # Batch 64 is part of the accepted numerical contract. A different
        # batch shape changed low CUDA bits in validation, so the public runner
        # intentionally does not expose it.
        batch_size=args.batch_size,
        workers=0,
        precision='fp32',
        loader_backend='thread',
        preprocess_threads=args.threads,
        progress=args.progress,
    ))
    print(
        'M1 v2 complete: {count} images; classes: good={class_good}, '
        'usable={class_usable}, bad={class_bad}; gradability: '
        'gradable={gradable}, ungradable={ungradable}; '
        '{wall:.3f}s, device={device}, precision={precision}'.format(
            count=summary['image_count'],
            class_good=summary['prediction_good_count'],
            class_usable=summary['prediction_usable_count'],
            class_bad=summary['prediction_bad_count'],
            gradable=summary['gradable_count'],
            ungradable=summary['ungradable_count'],
            wall=summary['wall_s'],
            device=summary['device_resolved'], precision=summary['precision'],
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
