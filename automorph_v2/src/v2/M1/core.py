'''Efficient, cross-platform implementation of the AutoMorph M1 ensemble.'''

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

try:
    from ..progress import ProgressReporter, ProgressTask
except ImportError:  # imported as ``M1.core`` by the direct stage runner
    import sys

    _V2_ROOT = str(Path(__file__).resolve().parents[1])
    if _V2_ROOT not in sys.path:
        sys.path.insert(0, _V2_ROOT)
    from progress import ProgressReporter, ProgressTask

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
CSV_FIELDS = [
    'Name', 'softmax_good', 'softmax_usable', 'softmax_bad',
    'good_sd', 'usable_sd', 'bad_sd', 'Prediction',
]
QUALITY_FINAL_FIELDS = [*CSV_FIELDS, 'Gradable']
QUALITY_SUMMARY_FIELDS = [
    'total_images', 'good_count', 'usable_count', 'bad_count',
    'gradable_count', 'ungradable_count',
    'mean_softmax_good', 'mean_softmax_usable', 'mean_softmax_bad',
]
GRADABILITY_RULE = 'Prediction == 0 or (Prediction == 1 and softmax_bad < 0.25)'
CHECKPOINT_DIRS = [
    '7_seed_28', '6_seed_30', '5_seed_32', '4_seed_34',
    '3_seed_36', '2_seed_38', '1_seed_40', '0_seed_42',
]


@dataclass(frozen=True)
class M1Config:
    input_dir: Path
    output_dir: Path
    weights_dir: Path
    device: str = 'auto'
    batch_size: int = 64
    workers: int = 8
    precision: str = 'fp32'
    copy_workers: int = 8
    engine: str = 'optimized'
    channels_last: bool = False
    loader_backend: str = 'thread'
    preprocess_threads: int = 4
    progress: str = 'off'


class QualityDataset(Dataset):
    '''Legacy-equivalent 512 px resize and non-zero-pixel normalization.'''

    def __init__(self, input_dir: Path):
        # Legacy sorts raw filename strings. PureWindowsPath ordering is
        # case-insensitive and changes batch positions for names such as ON vs
        # macula, which can alter low CUDA bits even when predictions agree.
        visible_files = [
            path for path in input_dir.iterdir()
            if path.is_file() and not path.name.startswith('.')
        ]
        unsupported = sorted(
            path.name for path in visible_files if path.suffix.lower() not in IMAGE_EXTENSIONS
        )
        if unsupported:
            raise ValueError(
                'M1 expects only supported image files from M0; unsupported: '
                + ', '.join(unsupported[:5])
            )
        names = sorted(path.name for path in visible_files)
        stems = [Path(name).stem for name in names]
        if len(stems) != len(set(stems)):
            raise ValueError('M1 input has duplicate filename stems and is ambiguous to legacy M1')
        self.paths = [(input_dir / name).resolve() for name in names]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            # Pillow's default for RGB images is the legacy NEAREST behavior.
            array = np.array(image.resize((512, 512)))
        positive = array[array > 0]
        if not positive.size:
            raise ValueError(f'image has no positive pixels: {path}')
        mean = np.mean(positive)
        std = np.std(positive)
        if not np.isfinite(std) or std == 0:
            raise ValueError(f'image has invalid non-zero standard deviation: {path}')
        array = (array - mean) / std
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f'image must decode to 1 or 3 channels: {path}')
        array = np.ascontiguousarray(array.transpose((2, 0, 1)))
        return torch.from_numpy(array).float(), str(path)


class ThreadedBatchLoader:
    '''Ordered, cross-platform preprocessing with one batch of look-ahead.'''

    def __init__(self, dataset: QualityDataset, batch_size: int, threads: int):
        if not len(dataset):
            raise ValueError('M1 input directory is empty')
        self.dataset = dataset
        self.batch_size = batch_size
        self.pool = ThreadPoolExecutor(
            max_workers=threads,
            thread_name_prefix='m1-preprocess',
        )
        self.ranges = [
            range(start, min(start + batch_size, len(dataset)))
            for start in range(0, len(dataset), batch_size)
        ]
        self.pending = self._submit(self.ranges[0])
        self.wait_s = 0.0

    def _submit(
        self,
        indices: range,
    ) -> list[Future[tuple[torch.Tensor, str]]]:
        return [self.pool.submit(self.dataset.__getitem__, index) for index in indices]

    def __iter__(self):
        pending = self.pending
        for batch_index, indices in enumerate(self.ranges):
            wait_started = time.perf_counter()
            items = [future.result() for future in pending]
            self.wait_s += time.perf_counter() - wait_started
            if batch_index + 1 < len(self.ranges):
                pending = self._submit(self.ranges[batch_index + 1])
            yield (
                torch.stack([item[0] for item in items], dim=0),
                [item[1] for item in items],
            )

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    is_indexed_cuda = requested.startswith('cuda:')
    if requested not in {'auto', 'cpu', 'cuda', 'mps'} and not is_indexed_cuda:
        raise ValueError('device must be auto, cpu, cuda, cuda:N, or mps')
    if requested == 'auto':
        if torch.cuda.is_available():
            requested = 'cuda'
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            requested = 'mps'
        else:
            requested = 'cpu'
    if requested == 'cuda' or is_indexed_cuda:
        try:
            device = torch.device(requested)
        except (RuntimeError, ValueError) as error:
            raise ValueError(f'invalid CUDA device {requested!r}; expected cuda or cuda:N') from error
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f'CUDA device index {device.index} is out of range for '
                f'{torch.cuda.device_count()} visible device(s)'
            )
        return device
    if requested == 'mps' and not (
        getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError('MPS requested but unavailable')
    return torch.device(requested)


def build_model() -> nn.Module:
    from efficientnet_pytorch import EfficientNet

    model = EfficientNet.from_name('efficientnet-b4')
    model._fc = nn.Sequential(
        nn.Linear(1792, 256), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(256, 64), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(64, 3),
    )
    return model


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    # Modern zip-format checkpoints can be memory-mapped. load_state_dict then
    # copies the exact tensor values into the model, while avoiding an eager
    # extra read/copy of the whole 556 MiB ensemble checkpoint set.
    try:
        return torch.load(
            path,
            map_location='cpu',
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        # PyTorch versions before mmap and/or weights_only support.
        try:
            return torch.load(path, map_location='cpu', weights_only=True)
        except TypeError:
            return torch.load(path, map_location='cpu')
    except RuntimeError as error:
        # Legacy non-zip checkpoints cannot be memory-mapped.
        if 'mmap' not in str(error).lower():
            raise
        return torch.load(path, map_location='cpu', weights_only=True)


def load_models(
    weights_dir: Path,
    device: torch.device,
    channels_last: bool,
    progress: ProgressTask | None = None,
) -> list[nn.Module]:
    models = []
    for directory in CHECKPOINT_DIRS:
        checkpoint = weights_dir / directory / 'best_loss_checkpoint.pth'
        if not checkpoint.is_file():
            raise FileNotFoundError(f'missing M1 checkpoint: {checkpoint}')
        model = build_model()
        model.load_state_dict(_load_state(checkpoint))
        model.eval().requires_grad_(False)
        if channels_last and device.type == 'cuda':
            model = model.to(memory_format=torch.channels_last)
        models.append(model.to(device))
        if progress is not None:
            progress.update()
    return models


def _autocast(device: torch.device, precision: str):
    if precision == 'fp32':
        return torch.autocast(device_type=device.type, enabled=False)
    if precision != 'amp':
        raise ValueError('precision must be fp32 or amp')
    if device.type == 'cpu':
        raise ValueError('AMP is only supported by this runner on CUDA or MPS')
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def optimized_batch(
    models: list[nn.Module], images: torch.Tensor, device: torch.device, precision: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Synchronize once, then preserve legacy NumPy aggregation exactly.'''
    with torch.inference_mode(), _autocast(device, precision):
        probabilities = torch.stack(
            [torch.softmax(model(images), dim=1) for model in models], dim=0
        )
    probability_array = probabilities.float().cpu().numpy()
    means = np.mean(probability_array, axis=0)
    deviations = np.std(probability_array, axis=0)
    accumulator = torch.from_numpy(probability_array[0]).clone()
    for index in range(1, probability_array.shape[0]):
        accumulator += torch.from_numpy(probability_array[index])
    predictions = (accumulator / probability_array.shape[0]).argmax(dim=1).numpy()
    return means, deviations, predictions


def legacy_sync_batch(
    models: list[nn.Module], images: torch.Tensor, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Benchmark reference: reproduce legacy's eight GPU-to-CPU synchronizations.'''
    probabilities = []
    accumulator: torch.Tensor | int = 0
    with torch.no_grad():
        for model in models:
            prediction = torch.softmax(model(images), dim=1)
            cpu_prediction = prediction.type(torch.FloatTensor)
            accumulator = accumulator + cpu_prediction
            probabilities.append(cpu_prediction.cpu().detach().numpy())
    accumulator = accumulator / len(models)
    array = np.asarray(probabilities)
    return np.mean(array, axis=0), np.std(array, axis=0), accumulator.argmax(dim=1).numpy()


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', newline='', encoding='utf8') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(CSV_FIELDS)
        for row in rows:
            writer.writerow([
                row['Name'],
                *[
                    np.float32(row[field])
                    for field in (
                        'softmax_good', 'softmax_usable', 'softmax_bad',
                        'good_sd', 'usable_sd', 'bad_sd',
                    )
                ],
                np.int64(row['Prediction']),
            ])
    os.replace(temp, path)


def classify_good(row: dict[str, Any]) -> bool:
    prediction = int(row['Prediction'])
    return prediction == 0 or (prediction == 1 and float(row['softmax_bad']) < 0.25)


def write_gradability_artifacts(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, int]:
    '''Write explicit legacy-compatible class and gradability metadata.'''
    prediction_counts = {
        prediction: sum(int(row['Prediction']) == prediction for row in rows)
        for prediction in (0, 1, 2)
    }
    gradable = [classify_good(row) for row in rows]
    gradable_count = sum(gradable)
    counts = {
        'prediction_good_count': prediction_counts[0],
        'prediction_usable_count': prediction_counts[1],
        'prediction_bad_count': prediction_counts[2],
        'gradable_count': gradable_count,
        'ungradable_count': len(rows) - gradable_count,
    }

    final_path = output_dir / 'M1_quality_final.csv'
    final_temp = final_path.with_name(final_path.name + '.tmp')
    with final_temp.open('w', newline='', encoding='utf8') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(QUALITY_FINAL_FIELDS)
        for row, is_gradable in zip(rows, gradable):
            writer.writerow([
                row['Name'],
                *[
                    np.float32(row[field])
                    for field in (
                        'softmax_good', 'softmax_usable', 'softmax_bad',
                        'good_sd', 'usable_sd', 'bad_sd',
                    )
                ],
                np.int64(row['Prediction']),
                is_gradable,
            ])
    os.replace(final_temp, final_path)

    mean_values = {
        field: round(float(np.mean([row[field] for row in rows])), 4)
        for field in ('softmax_good', 'softmax_usable', 'softmax_bad')
    }
    summary_path = output_dir / 'M1_quality_summary.csv'
    summary_temp = summary_path.with_name(summary_path.name + '.tmp')
    with summary_temp.open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=QUALITY_SUMMARY_FIELDS,
            lineterminator='\n',
        )
        writer.writeheader()
        writer.writerow({
            'total_images': len(rows),
            'good_count': counts['prediction_good_count'],
            'usable_count': counts['prediction_usable_count'],
            'bad_count': counts['prediction_bad_count'],
            'gradable_count': counts['gradable_count'],
            'ungradable_count': counts['ungradable_count'],
            'mean_softmax_good': mean_values['softmax_good'],
            'mean_softmax_usable': mean_values['softmax_usable'],
            'mean_softmax_bad': mean_values['softmax_bad'],
        })
    os.replace(summary_temp, summary_path)
    return counts


def materialize_classification(
    rows: list[dict[str, Any]],
    output_dir: Path,
    workers: int,
    progress: ProgressTask | None = None,
) -> tuple[int, int, int]:
    class_dirs = {
        0: output_dir / 'Good_quality',
        1: output_dir / 'Usable_quality',
        2: output_dir / 'Bad_quality',
    }
    for directory in class_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    # M2 reads this file to know which quality directories to segment.
    # M2 runs full inference over every graded class; M3 later filters to gradable.
    (output_dir / 'm2_input_quality.txt').write_text(
        'Good_quality\nUsable_quality\nBad_quality\n', encoding='utf-8'
    )

    def copy_one(row: dict[str, Any]) -> int:
        source = Path(str(row['Name']))
        prediction = int(row['Prediction'])
        if prediction not in class_dirs:
            raise ValueError(f'unsupported M1 prediction class: {prediction}')
        for class_index, directory in class_dirs.items():
            if class_index != prediction:
                stale = directory / source.name
                if stale.exists():
                    stale.unlink()
        shutil.copy2(source, class_dirs[prediction] / source.name)
        return prediction

    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix='m1-copy') as pool:
        predictions = []
        for prediction in pool.map(copy_one, rows):
            predictions.append(prediction)
            if progress is not None:
                # pool.map yields on the caller; workers never touch progress.
                progress.update()
    expected_by_class = {
        class_index: {
            Path(str(row['Name'])).name
            for row, prediction in zip(rows, predictions)
            if prediction == class_index
        }
        for class_index in class_dirs
    }
    # These three directories are generated M1 outputs. Remove only stale image
    # files, preserving any unrelated notes or manifests a user placed there.
    for class_index, class_dir in class_dirs.items():
        expected = expected_by_class[class_index]
        for existing in class_dir.iterdir():
            if (
                existing.is_file()
                and existing.suffix.lower() in IMAGE_EXTENSIONS
                and existing.name not in expected
            ):
                existing.unlink()
    counts = tuple(predictions.count(class_index) for class_index in (0, 1, 2))
    return counts


def run_m1(config: M1Config) -> dict[str, Any]:
    started = time.perf_counter()
    input_dir = Path(config.input_dir).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve()
    weights_dir = Path(config.weights_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f'M1 input directory not found: {input_dir}')
    if (
        config.batch_size < 1
        or config.workers < 0
        or config.copy_workers < 0
        or config.preprocess_threads < 1
    ):
        raise ValueError('batch_size must be positive and worker counts cannot be negative')
    if config.engine not in {'optimized', 'legacy_sync'}:
        raise ValueError('engine must be optimized or legacy_sync')
    if config.loader_backend not in {'thread', 'process'}:
        raise ValueError('loader_backend must be thread or process')
    device = resolve_device(config.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    dataset = QualityDataset(input_dir)
    progress = ProgressReporter('M1', config.progress)
    threaded_loader: ThreadedBatchLoader | None = None
    if config.loader_backend == 'thread':
        # Construction immediately starts the first batch, overlapping its CPU
        # preprocessing with model construction, checkpoint I/O, and H2D load.
        threaded_loader = ThreadedBatchLoader(
            dataset,
            batch_size=config.batch_size,
            threads=config.preprocess_threads,
        )
        loader = threaded_loader
    else:
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.workers,
            pin_memory=device.type == 'cuda',
            persistent_workers=config.workers > 0,
            prefetch_factor=2 if config.workers > 0 else None,
        )

    rows: list[dict[str, Any]] = []
    try:
        with progress.task(
            'model-load', total=len(CHECKPOINT_DIRS), unit='models'
        ) as model_progress:
            load_started = time.perf_counter()
            models = load_models(
                weights_dir, device, config.channels_last, progress=model_progress
            )
            load_s = time.perf_counter() - load_started

        with progress.task(
            'inference', total=len(dataset), unit='images'
        ) as inference_progress:
            infer_started = time.perf_counter()
            for images, names in loader:
                images = images.to(device, dtype=torch.float32, non_blocking=device.type == 'cuda')
                if config.channels_last and device.type == 'cuda':
                    images = images.contiguous(memory_format=torch.channels_last)
                if config.engine == 'legacy_sync':
                    means, deviations, predictions = legacy_sync_batch(models, images, device)
                else:
                    means, deviations, predictions = optimized_batch(
                        models, images, device, config.precision
                    )
                for index, name in enumerate(names):
                    rows.append({
                        'Name': name,
                        'softmax_good': float(means[index, 0]),
                        'softmax_usable': float(means[index, 1]),
                        'softmax_bad': float(means[index, 2]),
                        'good_sd': float(deviations[index, 0]),
                        'usable_sd': float(deviations[index, 1]),
                        'bad_sd': float(deviations[index, 2]),
                        'Prediction': int(predictions[index]),
                    })
                inference_progress.update(len(names))
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            inference_s = time.perf_counter() - infer_started
    finally:
        if threaded_loader is not None:
            threaded_loader.close()
    write_csv_atomic(output_dir / 'results_ensemble.csv', rows)

    quality_counts = write_gradability_artifacts(output_dir, rows)

    copy_started = time.perf_counter()
    with progress.task('routing', total=len(rows), unit='images') as routing_progress:
        good_count, usable_count, bad_count = materialize_classification(
            rows, output_dir, config.copy_workers, progress=routing_progress
        )
    copy_s = time.perf_counter() - copy_started
    wall_s = time.perf_counter() - started
    summary = {
        'module': 'M1_v2',
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'weights_dir': str(weights_dir),
        'image_count': len(dataset),
        'good_count': good_count,
        'usable_count': usable_count,
        'bad_count': bad_count,
        **quality_counts,
        'good_quality_route_count': good_count,
        'usable_quality_route_count': usable_count,
        'bad_quality_route_count': bad_count,
        'classification_directory_semantics': 'prediction-class',
        'gradability_rule': GRADABILITY_RULE,
        'device_requested': config.device,
        'device_resolved': str(device),
        'batch_size': config.batch_size,
        'workers': config.workers,
        'loader_backend': config.loader_backend,
        'preprocess_threads': config.preprocess_threads,
        'preprocess_wait_s': (
            threaded_loader.wait_s if threaded_loader is not None else None
        ),
        'copy_workers': config.copy_workers,
        'precision': config.precision,
        'engine': config.engine,
        'channels_last': config.channels_last,
        'model_load_s': load_s,
        'inference_s': inference_s,
        'classification_copy_s': copy_s,
        'wall_s': wall_s,
        'inference_images_s': len(dataset) / inference_s if inference_s else 0.0,
        'end_to_end_images_s': len(dataset) / wall_s if wall_s else 0.0,
        'gpu_peak_allocated_mb': (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
        ),
        'torch': torch.__version__,
        'python': platform.python_version(),
        'platform': platform.platform(),
    }
    temp = output_dir / 'run_summary.json.tmp'
    temp.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    os.replace(temp, output_dir / 'run_summary.json')
    return summary
