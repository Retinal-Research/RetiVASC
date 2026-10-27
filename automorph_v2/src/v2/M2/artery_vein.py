'''Strict, cross-platform AutoMorph M2 artery/vein inference core.

The legacy source remains the authority for tensor arithmetic and PNG
materialisation.  This module keeps that observable contract while isolating
paths, preprocessing images in ordered threads, and offering an inference-only
forward path that skips model heads whose results legacy immediately discards.
'''

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from skimage import io
from skimage.morphology import remove_small_objects, skeletonize
from torchvision.utils import save_image

try:
    from ..progress import ProgressReporter, ProgressTask
except ImportError:  # imported as ``M2.artery_vein`` by direct stage runner
    import sys

    _V2_ROOT = str(Path(__file__).resolve().parents[1])
    if _V2_ROOT not in sys.path:
        sys.path.insert(0, _V2_ROOT)
    from progress import ProgressReporter, ProgressTask


RELEASE_ROOT = Path(__file__).resolve().parents[3]
LEGACY_MODULE_DIR = RELEASE_ROOT / 'runtime' / 'M2_Artery_vein'
LEGACY_MODEL_PATH = LEGACY_MODULE_DIR / 'scripts' / 'model.py'

ALL_AV_IMAGE_SIZE = (720, 720)
ALL_AV_JOB_NAME = '20210724_ALL-AV_randomseed'
ENSEMBLE_SEEDS = (28, 30, 32, 34, 36, 38, 40, 42)
CHECKPOINT_FILENAMES = (
    'CP_best_F1_all.pth',
    'CP_best_F1_A.pth',
    'CP_best_F1_V.pth',
)
OUTPUT_DIRECTORY_NAMES = (
    'resized',
    'raw',
    'resize_uncertainty',
    'raw_uncertainty',
    'artery_binary_process',
    'vein_binary_process',
    'artery_binary_skeleton',
    'vein_binary_skeleton',
)


@dataclass(frozen=True)
class ArteryVeinConfig:
    '''Configuration for the strict ALL-AV production contract.'''

    input_dir: Path
    output_dir: Path
    weights_dir: Path = LEGACY_MODULE_DIR / 'ALL-AV'
    device: str = 'auto'
    batch_size: int = 8
    preprocess_threads: int = 4
    engine: str = 'strict'
    job_name: str = ALL_AV_JOB_NAME
    image_size: tuple[int, int] = ALL_AV_IMAGE_SIZE
    write_summary: bool = True
    progress: str = 'off'


@dataclass(frozen=True)
class AVSample:
    image: torch.Tensor
    name: str
    width: int
    height: int


@dataclass(frozen=True)
class ModelTrio:
    seed: int
    main: torch.nn.Module
    artery: torch.nn.Module
    vein: torch.nn.Module


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    resized: Path
    raw: Path
    resize_uncertainty: Path
    raw_uncertainty: Path
    artery_binary_process: Path
    vein_binary_process: Path
    artery_binary_skeleton: Path
    vein_binary_skeleton: Path


class AVImageDataset:
    '''Legacy-equivalent ALL-AV preprocessing with deterministic discovery.'''

    def __init__(self, input_dir: Path, image_size: tuple[int, int] = ALL_AV_IMAGE_SIZE):
        self.input_dir = Path(input_dir).expanduser().resolve()
        self.image_size = tuple(image_size)
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f'M2 artery/vein input directory not found: {self.input_dir}')
        if self.image_size != ALL_AV_IMAGE_SIZE:
            raise ValueError(
                'strict ALL-AV mode requires the legacy --uniform=True image size '
                f'{ALL_AV_IMAGE_SIZE}, got {self.image_size}'
            )

        # Preserve the exact legacy os.listdir order because which images fall
        # into the smaller final batch can affect backend kernel selection.
        paths = [
            (self.input_dir / name).resolve()
            for name in os.listdir(self.input_dir)
            if not name.startswith('.') and (self.input_dir / name).is_file()
        ]
        stems = [path.stem for path in paths]
        if len(stems) != len(set(stems)):
            raise ValueError('M2 artery/vein input has duplicate filename stems')
        if not paths:
            raise ValueError('M2 artery/vein input directory is empty')
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> AVSample:
        path = self.paths[index]
        with Image.open(path) as source:
            width, height = source.size
            # Deliberately omit resample= to reproduce legacy Pillow behavior.
            resized = source.resize(self.image_size)
            image_array = np.array(resized)

        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise ValueError(f'M2 artery/vein expects an RGB image: {path}')

        # This unusual multiply-by-std expression is intentional.  The legacy
        # code spells it `(image - mean) / 1.0 * std`; changing it to division
        # by std would change every model input.
        selected = image_array[image_array[..., 0] > 00.0]
        mean = np.mean(selected, axis=0)
        std = np.std(selected, axis=0)
        image_array = (image_array - 1.0 * mean) / 1.0 * std
        image_array = image_array.transpose((2, 0, 1))
        image_tensor = torch.from_numpy(image_array).type(torch.FloatTensor)
        return AVSample(image_tensor, path.stem, int(width), int(height))


class ThreadedBatchLoader:
    '''Ordered preprocessing with one batch of look-ahead.'''

    def __init__(self, dataset: AVImageDataset, batch_size: int = 8, threads: int = 4):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        if threads < 1:
            raise ValueError('preprocess_threads must be positive')
        self.dataset = dataset
        self.batch_size = batch_size
        self.pool = ThreadPoolExecutor(
            max_workers=threads,
            thread_name_prefix='m2-av-preprocess',
        )
        self.ranges = [
            range(start, min(start + batch_size, len(dataset)))
            for start in range(0, len(dataset), batch_size)
        ]
        self.pending = self._submit(self.ranges[0])
        self.wait_s = 0.0

    def _submit(self, indices: range) -> list[Future[AVSample]]:
        return [self.pool.submit(self.dataset.__getitem__, index) for index in indices]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, list[str], list[int], list[int]]]:
        pending = self.pending
        for batch_index, _indices in enumerate(self.ranges):
            wait_started = time.perf_counter()
            samples = [future.result() for future in pending]
            self.wait_s += time.perf_counter() - wait_started
            if batch_index + 1 < len(self.ranges):
                pending = self._submit(self.ranges[batch_index + 1])
            yield (
                torch.stack([sample.image for sample in samples], dim=0),
                [sample.name for sample in samples],
                [sample.width for sample in samples],
                [sample.height for sample in samples],
            )

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    indexed_cuda = requested.startswith('cuda:')
    if requested not in {'auto', 'cpu', 'cuda', 'mps'} and not indexed_cuda:
        raise ValueError('device must be auto, cpu, cuda, cuda:N, or mps')
    if requested == 'auto':
        if torch.cuda.is_available():
            requested = 'cuda'
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            requested = 'mps'
        else:
            requested = 'cpu'
    if requested == 'cuda' or indexed_cuda:
        try:
            device = torch.device(requested)
        except (RuntimeError, ValueError) as error:
            raise ValueError(f'invalid CUDA device: {requested!r}') from error
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f'CUDA device index out of range: {device.index}')
        return device
    if requested == 'mps' and not (
        getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError('MPS requested but unavailable')
    return torch.device(requested)


@lru_cache(maxsize=1)
def _legacy_model_module() -> ModuleType:
    if not LEGACY_MODEL_PATH.is_file():
        raise FileNotFoundError(f'legacy M2 model definition not found: {LEGACY_MODEL_PATH}')
    spec = importlib.util.spec_from_file_location('_automorph_m2_av_legacy_model', LEGACY_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load legacy M2 model definition: {LEGACY_MODEL_PATH}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    '''Load exact checkpoint tensors with compatible mmap/weights-only fallbacks.'''

    try:
        return torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    except TypeError:
        try:
            return torch.load(path, map_location='cpu', weights_only=True)
        except TypeError:
            return torch.load(path, map_location='cpu')
    except RuntimeError as error:
        if 'mmap' not in str(error).lower():
            raise
        try:
            return torch.load(path, map_location='cpu', weights_only=True)
        except TypeError:
            return torch.load(path, map_location='cpu')


def checkpoint_paths(weights_dir: Path, job_name: str = ALL_AV_JOB_NAME) -> list[Path]:
    root = Path(weights_dir).expanduser().resolve()
    paths: list[Path] = []
    for seed in ENSEMBLE_SEEDS:
        directory = root / f'{job_name}_{seed}' / 'Discriminator_unet'
        paths.extend(directory / filename for filename in CHECKPOINT_FILENAMES)
    return paths


def load_model_trios(
    weights_dir: Path,
    device: torch.device,
    job_name: str = ALL_AV_JOB_NAME,
    *,
    progress: ProgressTask | None = None,
) -> list[ModelTrio]:
    '''Load seeds and main/A/V weights in the exact legacy ensemble order.'''

    paths = checkpoint_paths(weights_dir, job_name)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'missing M2 artery/vein checkpoint: {missing[0]}')

    legacy = _legacy_model_module()
    trios: list[ModelTrio] = []
    path_index = 0
    for seed in ENSEMBLE_SEEDS:
        main = legacy.Generator_main(input_channels=3, n_filters=32, n_classes=4, bilinear=False)
        artery = legacy.Generator_branch(input_channels=3, n_filters=32, n_classes=4, bilinear=False)
        vein = legacy.Generator_branch(input_channels=3, n_filters=32, n_classes=4, bilinear=False)
        for model in (main, artery, vein):
            checkpoint = paths[path_index]
            path_index += 1
            state = _load_state(checkpoint)
            model.load_state_dict(state)
            model.eval().requires_grad_(False)
            model.to(device=device)
            if progress is not None:
                progress.update()
        trios.append(ModelTrio(seed, main, artery, vein))
    return trios


def _strict_branch_features(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    '''Run Generator_branch without its unused four-class output head.'''

    x1 = model.inc(images)
    x2 = model.down1(x1)
    x3 = model.down2(x2)
    x4 = model.down3(x3)
    x5 = model.down4(x4)
    x = model.up1(x5, x4)
    x = model.up2(x, x3)
    x = model.up3(x, x2)
    return model.up4(x, x1)


def _strict_main_logits(
    model: torch.nn.Module,
    images: torch.Tensor,
    artery_features: torch.Tensor,
    vein_features: torch.Tensor,
) -> torch.Tensor:
    '''Run Generator_main without unused S1/S2/S3 side-output heads.'''

    x1 = model.inc(images)
    x2 = model.down1(x1)
    x3 = model.down2(x2)
    x4 = model.down3(x3)
    x5 = model.down4(x4)
    x = model.up1(x5, x4)
    x = model.up2(x, x3)
    x = model.up3(x, x2)
    x = model.up4(x, x1)
    # Keep stack -> mean exactly as written by legacy; algebraic rewrites save
    # memory but can change floating-point rounding and uncertainty PNG pixels.
    x_fusion = torch.mean(torch.stack([artery_features, x, vein_features], dim=0), dim=0)
    return model.outc(x_fusion)


def _member_probability(
    trio: ModelTrio,
    images: torch.Tensor,
    skip_unused_heads: bool,
) -> torch.Tensor:
    if skip_unused_heads:
        artery_features = _strict_branch_features(trio.artery, images)
        vein_features = _strict_branch_features(trio.vein, images)
        logits = _strict_main_logits(
            trio.main,
            images,
            artery_features.detach(),
            vein_features.detach(),
        )
    else:
        _artery_logits, artery_features = trio.artery(images)
        _vein_logits, vein_features = trio.vein(images)
        logits, _side1, _side2, _side3 = trio.main(
            images,
            artery_features.detach(),
            vein_features.detach(),
        )
    # Retain legacy clone/detach and softmax ordering for the strict gate.
    return F.softmax(logits.clone().detach(), dim=1)


def _ensemble_batch(
    model_trios: Sequence[ModelTrio],
    images: torch.Tensor,
    device: torch.device,
    *,
    skip_unused_heads: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(model_trios) != len(ENSEMBLE_SEEDS):
        raise ValueError(f'strict M2 ensemble requires {len(ENSEMBLE_SEEDS)} model trios')

    probabilities: list[torch.Tensor] = []
    accumulator: torch.Tensor | int = 0
    with torch.inference_mode():
        for trio in model_trios:
            probability = _member_probability(trio, images, skip_unused_heads)
            probabilities.append(probability)
            # Intentional legacy device choreography: each member contributes
            # to a CPU float32 accumulator, then the mean returns to the model
            # device.  An all-device reduction is a separate gated experiment.
            accumulator += probability.type(torch.FloatTensor)

        assert isinstance(accumulator, torch.Tensor)
        mean_probability = (accumulator / len(model_trios)).to(device=device)

        # Preserve the original left-to-right variance expression exactly.
        variance_sum = torch.square(mean_probability - probabilities[0])
        for probability in probabilities[1:]:
            variance_sum = variance_sum + torch.square(mean_probability - probability)
        uncertainty = torch.sqrt(variance_sum / len(model_trios))
        _values, prediction = torch.max(mean_probability, 1)
        prediction = prediction.type(torch.FloatTensor)
    return prediction, uncertainty


def legacy_ensemble_batch(
    model_trios: Sequence[ModelTrio],
    images: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    '''Reference engine: full legacy heads plus legacy ensemble arithmetic.'''

    return _ensemble_batch(model_trios, images, device, skip_unused_heads=False)


def strict_ensemble_batch(
    model_trios: Sequence[ModelTrio],
    images: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    '''Candidate strict engine: skip only mathematically dead inference heads.'''

    return _ensemble_batch(model_trios, images, device, skip_unused_heads=True)


def prepare_output_paths(output_dir: Path) -> OutputPaths:
    root = Path(output_dir).expanduser().resolve()
    directories = {name: root / name for name in OUTPUT_DIRECTORY_NAMES}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return OutputPaths(root=root, **directories)


def save_primary_batch(
    prediction: torch.Tensor,
    uncertainty: torch.Tensor,
    names: Sequence[str],
    widths: Sequence[int],
    heights: Sequence[int],
    paths: OutputPaths,
) -> None:
    '''Write the six per-image legacy prediction/uncertainty PNGs.'''

    if not (prediction.shape[0] == uncertainty.shape[0] == len(names) == len(widths) == len(heights)):
        raise ValueError('batch outputs and metadata have different lengths')

    for index, name in enumerate(names):
        resized_uncertainty = paths.resize_uncertainty / f'{name}.png'
        save_image(uncertainty[index, ...] * 255, str(resized_uncertainty))
        save_image(
            uncertainty[index, 1, ...] * 255,
            str(paths.resize_uncertainty / f'{name}_artery.png'),
        )
        save_image(
            uncertainty[index, 2, ...] * 255,
            str(paths.resize_uncertainty / f'{name}_vein.png'),
        )

        with Image.open(resized_uncertainty) as source:
            raw_uncertainty_image = source.resize((int(widths[index]), int(heights[index])))
        raw_uncertainty_tensor = torchvision.transforms.ToTensor()(raw_uncertainty_image)
        save_image(raw_uncertainty_tensor, str(paths.raw_uncertainty / f'{name}.png'))

        height, width = prediction[index, ...].shape
        artery = np.zeros((height, width))
        overlap = np.zeros((height, width))
        vein = np.zeros((height, width))
        artery[prediction[index, ...] == 1] = 255
        vein[prediction[index, ...] == 2] = 255
        overlap[prediction[index, ...] == 3] = 255
        vein = remove_small_objects(vein > 0, 30, connectivity=5)
        artery = remove_small_objects(artery > 0, 30, connectivity=5)

        # This is BGR for cv2.imwrite; readers see artery=red, overlap=green,
        # vein=blue.  Preserve the legacy float32*255 conversion and clipping.
        segmentation = np.concatenate(
            (vein[..., np.newaxis], overlap[..., np.newaxis], artery[..., np.newaxis]),
            axis=2,
        )
        resized_values = np.float32(segmentation) * 255
        if not cv2.imwrite(str(paths.resized / f'{name}.png'), resized_values):
            raise OSError(f'failed to write M2 output: {paths.resized / (name + ".png")}')
        raw_values = cv2.resize(
            resized_values,
            (int(widths[index]), int(heights[index])),
            interpolation=cv2.INTER_NEAREST,
        )
        if not cv2.imwrite(str(paths.raw / f'{name}.png'), raw_values):
            raise OSError(f'failed to write M2 output: {paths.raw / (name + ".png")}')


def _quantize_uncertainty_batch(uncertainty: torch.Tensor) -> np.ndarray:
    """Apply torchvision save_image arithmetic on-device, then transfer once."""

    packed: list[torch.Tensor] = []
    for index in range(uncertainty.shape[0]):
        encoded: list[torch.Tensor] = []
        for tensor in (
            uncertainty[index, ...],
            uncertainty[index, 1, ...],
            uncertainty[index, 2, ...],
        ):
            # Legacy first multiplies by 255 at the call site. make_grid then
            # expands a 2-D single channel to RGB before save_image performs
            # its second multiply, rounding, and clamp.
            grid = torchvision.utils.make_grid(tensor * 255)
            encoded.append(
                grid.mul(255)
                .add_(0.5)
                .clamp_(0, 255)
                .permute(1, 2, 0)
                .to(dtype=torch.uint8)
            )
        packed.append(torch.cat(encoded, dim=2))
    return torch.stack(packed, dim=0).to(device='cpu').numpy()


def _materialize_binary_image(paths: OutputPaths, image_name: str) -> None:
    image = io.imread(paths.resized / image_name).astype(np.int64)
    image = cv2.resize(image, (912, 912), interpolation=cv2.INTER_NEAREST)
    foreground = image > 0
    artery = foreground[..., 0] + foreground[..., 1]
    vein = foreground[..., 2] + foreground[..., 1]
    artery = remove_small_objects(artery, 30, connectivity=5)
    vein = remove_small_objects(vein, 30, connectivity=5)

    io.imsave(
        paths.artery_binary_process / image_name,
        255 * artery.astype('uint8'),
        check_contrast=False,
    )
    io.imsave(
        paths.vein_binary_process / image_name,
        255 * vein.astype('uint8'),
        check_contrast=False,
    )
    io.imsave(
        paths.artery_binary_skeleton / image_name,
        255 * skeletonize(artery).astype('uint8'),
        check_contrast=False,
    )
    io.imsave(
        paths.vein_binary_skeleton / image_name,
        255 * skeletonize(vein).astype('uint8'),
        check_contrast=False,
    )


def _write_complete_sample(
    paths: OutputPaths,
    *,
    name: str,
    width: int,
    height: int,
    label: np.ndarray,
    packed_uncertainty: np.ndarray,
) -> dict[str, float]:
    primary_started = time.perf_counter()
    filename = f'{name}.png'
    resized_uncertainty = paths.resize_uncertainty / filename
    Image.fromarray(packed_uncertainty[..., :4]).save(resized_uncertainty)
    Image.fromarray(packed_uncertainty[..., 4:7]).save(
        paths.resize_uncertainty / f'{name}_artery.png'
    )
    Image.fromarray(packed_uncertainty[..., 7:10]).save(
        paths.resize_uncertainty / f'{name}_vein.png'
    )

    with Image.open(resized_uncertainty) as source:
        raw_uncertainty_image = source.resize((int(width), int(height)))
    raw_uncertainty_tensor = torchvision.transforms.ToTensor()(raw_uncertainty_image)
    save_image(raw_uncertainty_tensor, str(paths.raw_uncertainty / filename))

    output_height, output_width = label.shape
    artery = np.zeros((output_height, output_width))
    overlap = np.zeros((output_height, output_width))
    vein = np.zeros((output_height, output_width))
    artery[label == 1] = 255
    vein[label == 2] = 255
    overlap[label == 3] = 255
    vein = remove_small_objects(vein > 0, 30, connectivity=5)
    artery = remove_small_objects(artery > 0, 30, connectivity=5)
    segmentation = np.concatenate(
        (vein[..., np.newaxis], overlap[..., np.newaxis], artery[..., np.newaxis]),
        axis=2,
    )
    resized_values = np.float32(segmentation) * 255
    if not cv2.imwrite(str(paths.resized / filename), resized_values):
        raise OSError(f'failed to write M2 output: {paths.resized / filename}')
    raw_values = cv2.resize(
        resized_values,
        (int(width), int(height)),
        interpolation=cv2.INTER_NEAREST,
    )
    if not cv2.imwrite(str(paths.raw / filename), raw_values):
        raise OSError(f'failed to write M2 output: {paths.raw / filename}')
    primary_seconds = time.perf_counter() - primary_started

    binary_started = time.perf_counter()
    _materialize_binary_image(paths, filename)
    return {
        'primary_output_s': primary_seconds,
        'binary_output_s': time.perf_counter() - binary_started,
    }


def materialize_binary_outputs(paths: OutputPaths) -> int:
    '''Create four 912 px process/skeleton outputs; discarded metrics are omitted.'''

    checkpoint_dir = paths.raw / '.ipynb_checkpoints'
    if checkpoint_dir.is_dir():
        shutil.rmtree(checkpoint_dir)

    image_names = sorted(path.name for path in paths.raw.iterdir() if path.is_file())
    for image_name in image_names:
        _materialize_binary_image(paths, image_name)
    return len(image_names)


def _synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps' and hasattr(torch, 'mps'):
        torch.mps.synchronize()


class _DeviceMemoryTracker:
    def __init__(self, device: torch.device):
        self.device = device
        self.cuda_index = device.index if device.index is not None else 0
        self.mps_peak_allocated = 0
        if device.type == 'cuda':
            # Required by PyTorch 2.7 on Windows in a fresh process.
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(self.cuda_index)

    def sample(self) -> None:
        if self.device.type == 'mps' and hasattr(torch, 'mps'):
            self.mps_peak_allocated = max(
                self.mps_peak_allocated,
                int(torch.mps.current_allocated_memory()),
            )

    def summary(self) -> tuple[int, int]:
        if self.device.type == 'cuda':
            return (
                int(torch.cuda.max_memory_allocated(self.cuda_index)),
                int(torch.cuda.max_memory_reserved(self.cuda_index)),
            )
        if self.device.type == 'mps':
            self.sample()
            return self.mps_peak_allocated, 0
        return 0, 0


def run_artery_vein(config: ArteryVeinConfig) -> dict[str, Any]:
    '''Run M2 artery/vein inference and return/write a structured timing summary.'''

    started = time.perf_counter()
    if config.batch_size != 8:
        raise ValueError('strict M2 artery/vein fixes batch_size=8')
    if config.preprocess_threads < 1:
        raise ValueError('preprocess_threads must be positive')
    if config.engine not in {'legacy', 'strict'}:
        raise ValueError('engine must be legacy or strict')
    if tuple(config.image_size) != ALL_AV_IMAGE_SIZE:
        raise ValueError('strict ALL-AV execution requires image_size=(720, 720)')

    device = resolve_device(config.device)
    dataset = AVImageDataset(config.input_dir, config.image_size)
    loader = ThreadedBatchLoader(dataset, config.batch_size, config.preprocess_threads)
    output_paths = prepare_output_paths(config.output_dir)
    weights_dir = Path(config.weights_dir).expanduser().resolve()
    memory = _DeviceMemoryTracker(device)
    progress = ProgressReporter('M2', config.progress)

    inference_s = 0.0
    primary_output_s = 0.0
    primary_output_task_s = 0.0
    binary_output_task_s = 0.0
    postprocess_tail_wait_s = 0.0
    completed = 0
    post_executor = ThreadPoolExecutor(
        max_workers=config.preprocess_threads,
        thread_name_prefix='m2-av-output',
    )
    pending: deque[Future[dict[str, float]]] = deque()
    max_pending = max(config.preprocess_threads * 2, config.batch_size)
    process_progress: ProgressTask | None = None

    def collect_one() -> None:
        nonlocal primary_output_task_s, binary_output_task_s
        result = pending.popleft().result()
        primary_output_task_s += result['primary_output_s']
        binary_output_task_s += result['binary_output_s']
        if process_progress is not None:
            process_progress.update()

    try:
        load_started = time.perf_counter()
        model_total = len(ENSEMBLE_SEEDS) * len(CHECKPOINT_FILENAMES)
        with progress.task(
            'artery-vein-model-load', total=model_total, unit='models'
        ) as model_progress:
            model_trios = load_model_trios(
                weights_dir,
                device,
                config.job_name,
                progress=model_progress,
            )
        _synchronize(device)
        model_load_s = time.perf_counter() - load_started
        memory.sample()
        process_progress = progress.task(
            'artery-vein-inference-write', total=len(dataset), unit='images'
        )

        for images, names, widths, heights in loader:
            images = images.to(device=device, dtype=torch.float32)
            _synchronize(device)
            inference_started = time.perf_counter()
            if config.engine == 'legacy':
                prediction, uncertainty = legacy_ensemble_batch(model_trios, images, device)
            else:
                prediction, uncertainty = strict_ensemble_batch(model_trios, images, device)
            _synchronize(device)
            inference_s += time.perf_counter() - inference_started
            memory.sample()

            output_started = time.perf_counter()
            packed_uncertainty = _quantize_uncertainty_batch(uncertainty)
            labels = prediction.numpy().copy()
            for index, name in enumerate(names):
                pending.append(
                    post_executor.submit(
                        _write_complete_sample,
                        output_paths,
                        name=name,
                        width=widths[index],
                        height=heights[index],
                        label=labels[index],
                        packed_uncertainty=packed_uncertainty[index],
                    )
                )
                if len(pending) >= max_pending:
                    collect_one()
            primary_output_s += time.perf_counter() - output_started
            completed += len(names)
    except BaseException:
        if process_progress is not None:
            process_progress.close('error')
        loader.close()
        try:
            post_executor.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            post_executor.shutdown(wait=True)
        raise
    else:
        loader.close()

    # The remaining binary materialisation is CPU-only. Release the 24-model
    # ensemble and last batch before that stage so a long PNG/skeleton pass does
    # not unnecessarily occupy accelerator memory.
    del model_trios
    del images, prediction, uncertainty
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif (
        device.type == 'mps'
        and hasattr(torch, 'mps')
        and hasattr(torch.mps, 'empty_cache')
    ):
        torch.mps.empty_cache()

    postprocess_tail_started = time.perf_counter()
    try:
        while pending:
            collect_one()
    except BaseException:
        if process_progress is not None:
            process_progress.close('error')
        raise
    else:
        if process_progress is not None:
            process_progress.close()
    finally:
        post_executor.shutdown(wait=True)
    postprocess_tail_wait_s = time.perf_counter() - postprocess_tail_started
    binary_count = completed
    binary_output_s = binary_output_task_s
    wall_s = time.perf_counter() - started
    peak_allocated, peak_reserved = memory.summary()
    summary: dict[str, Any] = {
        'module': 'M2_artery_vein_v2',
        'input_dir': str(dataset.input_dir),
        'output_dir': str(output_paths.root),
        'weights_dir': str(weights_dir),
        'image_count': len(dataset),
        'binary_output_count': binary_count,
        'device_requested': config.device,
        'device_resolved': str(device),
        'batch_size': config.batch_size,
        'preprocess_threads': config.preprocess_threads,
        'preprocess_wait_s': loader.wait_s,
        'engine': config.engine,
        'ensemble_seeds': list(ENSEMBLE_SEEDS),
        'model_count': len(ENSEMBLE_SEEDS) * 3,
        'image_size': list(config.image_size),
        'model_load_s': model_load_s,
        'inference_s': inference_s,
        'primary_output_s': primary_output_s,
        'primary_output_task_sum_s': primary_output_task_s,
        'binary_output_s': binary_output_s,
        'postprocess_tail_wait_s': postprocess_tail_wait_s,
        'wall_s': wall_s,
        'inference_images_s': len(dataset) / inference_s if inference_s else 0.0,
        'end_to_end_images_s': len(dataset) / wall_s if wall_s else 0.0,
        'device_peak_allocated_bytes': peak_allocated,
        'device_peak_reserved_bytes': peak_reserved,
        'gpu_peak_allocated_mb': peak_allocated / 2**20 if device.type == 'cuda' else 0.0,
        'gpu_peak_reserved_mb': peak_reserved / 2**20 if device.type == 'cuda' else 0.0,
        'torch': torch.__version__,
        'python': platform.python_version(),
        'platform': platform.platform(),
    }
    if config.write_summary:
        temporary = output_paths.root / 'run_summary.json.tmp'
        temporary.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        os.replace(temporary, output_paths.root / 'run_summary.json')
    return summary


__all__ = [
    'ALL_AV_IMAGE_SIZE',
    'ALL_AV_JOB_NAME',
    'ArteryVeinConfig',
    'AVImageDataset',
    'AVSample',
    'CHECKPOINT_FILENAMES',
    'ENSEMBLE_SEEDS',
    'ModelTrio',
    'OUTPUT_DIRECTORY_NAMES',
    'OutputPaths',
    'ThreadedBatchLoader',
    'checkpoint_paths',
    'legacy_ensemble_batch',
    'load_model_trios',
    'materialize_binary_outputs',
    'prepare_output_paths',
    'resolve_device',
    'run_artery_vein',
    'save_primary_batch',
    'strict_ensemble_batch',
]
