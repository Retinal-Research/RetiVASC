'''Explicit-path CLI for M1 v2 and its benchmark harness.'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .core import M1Config, run_m1
except ImportError:
    from core import M1Config, run_m1


def main() -> int:
    parser = argparse.ArgumentParser(description='Run optimized AutoMorph M1')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--weights-dir', type=Path, required=True)
    parser.add_argument(
        '--device', default='auto',
        help='auto, cpu, cuda, cuda:N, or mps (default: auto)',
    )
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--loader-backend', choices=['thread', 'process'], default='thread')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--copy-workers', type=int, default=8)
    parser.add_argument('--precision', choices=['fp32', 'amp'], default='fp32')
    parser.add_argument('--channels-last', action='store_true')
    parser.add_argument('--engine', choices=['optimized', 'legacy_sync'], default='optimized')
    args = parser.parse_args()
    summary = run_m1(M1Config(
        input_dir=args.input_dir, output_dir=args.output_dir, weights_dir=args.weights_dir,
        device=args.device, batch_size=args.batch_size, workers=args.workers,
        copy_workers=args.copy_workers, precision=args.precision, engine=args.engine,
        channels_last=args.channels_last, loader_backend=args.loader_backend,
        preprocess_threads=args.threads,
    ))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
