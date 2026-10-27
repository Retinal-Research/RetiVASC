"""Strict-compatible AutoMorph M2 binary-vessel inference core.

The legacy module is intentionally left untouched. This module imports the
legacy Segmenter class from an explicit path, loads the same ten checkpoints
in the same order, and preserves the legacy FP32 ensemble and PNG semantics.

All filesystem writes are rooted below VesselConfig.output_dir.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import pickle
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from skimage import io
from skimage.morphology import remove_small_objects, skeletonize
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

try:
    from ..progress import ProgressReporter, ProgressTask
except ImportError:  # imported as ``M2.vessel`` by the direct stage runner
    _V2_ROOT = str(Path(__file__).resolve().parents[1])
    if _V2_ROOT not in sys.path:
        sys.path.insert(0, _V2_ROOT)
    from progress import ProgressReporter, ProgressTask


STRICT_IMAGE_SIZE = (912, 912)
STRICT_BATCH_SIZE = 8
STRICT_THRESHOLD = 40.0
STRICT_DATASET = "ALL-SIX"
STRICT_JOB_NAME = "20210630_uniform_thres40_ALL-SIX"
STRICT_SEEDS = (24, 26, 28, 30, 32, 34, 36, 38, 40, 42)
CHECKPOINT_FILENAME = "G_best_F1_epoch.pth"

# Kept as documentation for the public compatibility wrapper. Several of
# these legacy training arguments are parsed but unused during inference.
LEGACY_ARGUMENT_DEFAULTS: Mapping[str, Any] = {
    "epochs": 1,
    "batchsize": 6,
    "learning_rate": 2e-4,
    "load": False,
    "discriminator": "unet",
    "jn": "unet",
    "worker_num": 2,
    "save_model": "regular",
    "train_test_mode": "trainandtest",
    "pre_threshold": 0.0,
    "dataset": None,
    "seed_num": 42,
    "dataset_test": None,
    "validation_ratio": 10.0,
    "uniform": "False",
    "out_test": "False",
    "alpha": 0.08,
    "beta": 1.1,
    "gamma": 0.5,
}


@dataclass(frozen=True)
class VesselConfig:
    """Configuration for one isolated strict M2 vessel run."""

    input_dir: Path
    output_dir: Path
    legacy_dir: Path
    device: str = "auto"
    preprocess_threads: int = 4
    postprocess_threads: int = 4
    checkpoint_root: Path | None = None
    batch_size: int = STRICT_BATCH_SIZE
    image_size: tuple[int, int] = STRICT_IMAGE_SIZE
    threshold: float = STRICT_THRESHOLD
    dataset: str = STRICT_DATASET
    job_name: str = STRICT_JOB_NAME
    seeds: tuple[int, ...] = STRICT_SEEDS
    write_summary: bool = True
    progress: str = "off"


@dataclass(frozen=True)
class VesselSample:
    name: str
    width: int
    height: int
    image: torch.Tensor


@dataclass(frozen=True)
class VesselBatch:
    names: tuple[str, ...]
    widths: tuple[int, ...]
    heights: tuple[int, ...]
    images: torch.Tensor


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    resize: Path
    resize_binary: Path
    raw: Path
    raw_binary: Path
    resize_uncertainty: Path
    raw_uncertainty: Path
    binary_process: Path
    binary_skeleton: Path

    @classmethod
    def create(cls, root: Path) -> "OutputPaths":
        root = Path(root)
        paths = cls(
            root=root,
            resize=root / "resize",
            resize_binary=root / "resize_binary",
            raw=root / "raw",
            raw_binary=root / "raw_binary",
            resize_uncertainty=root / "resize_uncertainty",
            raw_uncertainty=root / "raw_uncertainty",
            binary_process=root / "binary_process",
            binary_skeleton=root / "binary_skeleton",
        )
        for directory in paths.directories:
            directory.mkdir(parents=True, exist_ok=True)
        return paths

    @property
    def directories(self) -> tuple[Path, ...]:
        return (
            self.resize,
            self.resize_binary,
            self.raw,
            self.raw_binary,
            self.resize_uncertainty,
            self.raw_uncertainty,
            self.binary_process,
            self.binary_skeleton,
        )


def _validate_config(config: VesselConfig) -> None:
    if Path(config.input_dir).resolve() == Path(config.output_dir).resolve():
        raise ValueError("input_dir and output_dir must be different")
    if config.batch_size != STRICT_BATCH_SIZE:
        raise ValueError(
            f"Strict M2 fixes batch_size={STRICT_BATCH_SIZE}; got {config.batch_size}"
        )
    if tuple(config.image_size) != STRICT_IMAGE_SIZE:
        raise ValueError(
            f"Strict M2 fixes image_size={STRICT_IMAGE_SIZE}; got {config.image_size}"
        )
    if float(config.threshold) != STRICT_THRESHOLD:
        raise ValueError(
            f"Strict M2 fixes threshold={STRICT_THRESHOLD}; got {config.threshold}"
        )
    if config.dataset != STRICT_DATASET:
        raise ValueError(
            f"Strict M2 fixes dataset={STRICT_DATASET!r}; got {config.dataset!r}"
        )
    if config.job_name != STRICT_JOB_NAME:
        raise ValueError(
            f"Strict M2 fixes job_name={STRICT_JOB_NAME!r}; got {config.job_name!r}"
        )
    if tuple(config.seeds) != STRICT_SEEDS:
        raise ValueError(
            f"Strict M2 fixes checkpoint seeds={STRICT_SEEDS}; got {config.seeds}"
        )
    if config.preprocess_threads < 1:
        raise ValueError("preprocess_threads must be at least 1")
    if config.postprocess_threads < 1:
        raise ValueError("postprocess_threads must be at least 1")
    if not Path(config.input_dir).is_dir():
        raise FileNotFoundError(f"Input image directory not found: {config.input_dir}")
    model_path = Path(config.legacy_dir) / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"Legacy Segmenter source not found: {model_path}")


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve CPU, CUDA, or Apple MPS without changing global torch settings."""

    value = requested.strip().lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {requested}")
    if (
        device.type == "cuda"
        and device.index is not None
        and device.index >= torch.cuda.device_count()
    ):
        raise RuntimeError(f"CUDA device index out of range: {device.index}")
    if device.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported M2 device: {requested}")
    return device


def discover_images(input_dir: Path) -> list[Path]:
    """Match legacy os.listdir order and basename handling."""

    input_dir = Path(input_dir)
    paths = [
        input_dir / name
        for name in os.listdir(input_dir)
        if not name.startswith(".") and (input_dir / name).is_file()
    ]
    if not paths:
        raise ValueError(f"No input images found in {input_dir}")

    by_stem: dict[str, Path] = {}
    for path in paths:
        if path.stem in by_stem:
            raise ValueError(
                "Legacy M2 addresses images by basename, so duplicate stems are "
                f"ambiguous: {by_stem[path.stem].name!r}, {path.name!r}"
            )
        by_stem[path.stem] = path
    return paths


def preprocess_image(
    path: Path,
    *,
    image_size: tuple[int, int] = STRICT_IMAGE_SIZE,
    threshold: float = STRICT_THRESHOLD,
) -> VesselSample:
    """Reproduce SEDataset_out preprocessing operation-for-operation."""

    path = Path(path)
    with Image.open(path) as source:
        original_width, original_height = source.size
        # Deliberately omit resample= to retain the installed Pillow default,
        # exactly as the legacy implementation does.
        resized = source.resize(image_size)
        image_array = np.array(resized)

    if image_array.ndim != 3 or image_array.shape[2] < 3:
        raise ValueError(
            f"Legacy M2 requires an RGB-like image with at least 3 channels: {path}"
        )

    if np.sum(image_array[..., 2]) == 0:
        green = image_array[..., 1][..., np.newaxis]
        image_array = np.concatenate((green, green, green), axis=2)

    selected = image_array[image_array[..., 0] > threshold]
    mean_value = np.mean(selected, axis=0)
    std_value = np.std(selected, axis=0)
    image_array = (image_array - mean_value) / std_value

    if len(image_array.shape) == 2:
        image_array = np.expand_dims(image_array, axis=2)
    image_array = image_array.transpose((2, 0, 1))

    # type(torch.FloatTensor) in the legacy dataset performs the same
    # float64-to-float32 conversion and keeps preprocessing on CPU.
    tensor = torch.from_numpy(image_array).type(torch.FloatTensor)
    return VesselSample(
        name=path.stem,
        width=int(original_width),
        height=int(original_height),
        image=tensor,
    )


def _collate(samples: Sequence[VesselSample]) -> VesselBatch:
    return VesselBatch(
        names=tuple(sample.name for sample in samples),
        widths=tuple(sample.width for sample in samples),
        heights=tuple(sample.height for sample in samples),
        images=torch.stack([sample.image for sample in samples], dim=0),
    )


class OrderedThreadBatchLoader:
    """Ordered thread preprocessing with exactly one batch of look-ahead."""

    def __init__(
        self,
        paths: Sequence[Path],
        *,
        batch_size: int = STRICT_BATCH_SIZE,
        threads: int = 4,
        image_size: tuple[int, int] = STRICT_IMAGE_SIZE,
        threshold: float = STRICT_THRESHOLD,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if threads < 1:
            raise ValueError("threads must be at least 1")
        self.paths = tuple(Path(path) for path in paths)
        self.batch_size = int(batch_size)
        self.threads = int(threads)
        self.image_size = tuple(image_size)
        self.threshold = float(threshold)
        self._executor: ThreadPoolExecutor | None = None
        self._pending: list[Future[VesselSample]] | None = None
        self._next_index = 0
        self._closed = False

    def __len__(self) -> int:
        return math.ceil(len(self.paths) / self.batch_size)

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("OrderedThreadBatchLoader is closed")
        if self._executor is not None:
            return
        self._executor = ThreadPoolExecutor(
            max_workers=self.threads,
            thread_name_prefix="m2-vessel-preprocess",
        )
        self._pending = self._submit_next_batch()

    def _submit_next_batch(self) -> list[Future[VesselSample]] | None:
        if self._executor is None:
            raise RuntimeError("OrderedThreadBatchLoader has not been started")
        if self._next_index >= len(self.paths):
            return None
        stop = min(self._next_index + self.batch_size, len(self.paths))
        batch_paths = self.paths[self._next_index : stop]
        self._next_index = stop
        return [
            self._executor.submit(
                preprocess_image,
                path,
                image_size=self.image_size,
                threshold=self.threshold,
            )
            for path in batch_paths
        ]

    def __iter__(self) -> Iterator[VesselBatch]:
        self.start()
        while self._pending is not None:
            current = self._pending
            samples = [future.result() for future in current]
            # Submit before yielding so preprocessing overlaps GPU inference.
            self._pending = self._submit_next_batch()
            yield _collate(samples)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pending is not None:
            for future in self._pending:
                future.cancel()
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=True, cancel_futures=True)
            except TypeError:  # Python 3.8 compatibility
                self._executor.shutdown(wait=True)
        self._executor = None
        self._pending = None

    def __enter__(self) -> "OrderedThreadBatchLoader":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def checkpoint_paths(config: VesselConfig) -> tuple[Path, ...]:
    root = (
        Path(config.checkpoint_root)
        if config.checkpoint_root is not None
        else Path(config.legacy_dir) / "Saved_model"
    )
    train_root = root / f"train_on_{config.dataset}"
    return tuple(
        train_root
        / f"{config.job_name}_savebest_randomseed_{seed}"
        / CHECKPOINT_FILENAME
        for seed in config.seeds
    )


def _import_legacy_segmenter(legacy_dir: Path) -> type[torch.nn.Module]:
    model_path = (Path(legacy_dir) / "model.py").resolve()
    module_digest = hashlib.sha1(str(model_path).encode("utf-8")).hexdigest()[:12]
    module_name = f"_automorph_legacy_m2_vessel_{module_digest}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing.Segmenter

    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load legacy model module: {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    segmenter = getattr(module, "Segmenter", None)
    if segmenter is None:
        raise ImportError(f"Legacy model has no Segmenter class: {model_path}")
    return segmenter


def _load_checkpoint(path: Path) -> Any:
    """Use safe/mmap loading when available, with compatibility fallback."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"M2 vessel checkpoint not found: {path}")
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except (TypeError, RuntimeError, pickle.UnpicklingError):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except (TypeError, RuntimeError, pickle.UnpicklingError):
            # Trusted bundled legacy checkpoints, for old serialization only.
            return torch.load(path, map_location="cpu")


def _load_models(
    config: VesselConfig,
    device: torch.device,
    *,
    progress: ProgressTask | None = None,
) -> list[torch.nn.Module]:
    segmenter = _import_legacy_segmenter(config.legacy_dir)
    paths = checkpoint_paths(config)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing M2 vessel checkpoints:\n{formatted}")

    models: list[torch.nn.Module] = []
    for path in paths:
        model = segmenter(
            input_channels=3,
            n_filters=32,
            n_classes=1,
            bilinear=False,
        )
        state = _load_checkpoint(path)
        model.load_state_dict(state)
        del state
        model.eval()
        model.to(device=device)
        models.append(model)
        if progress is not None:
            progress.update()
    return models


def strict_ensemble_statistics(
    probabilities: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute legacy mean and uncertainty with exact left-to-right order."""

    if len(probabilities) != len(STRICT_SEEDS):
        raise ValueError(
            f"Strict vessel ensemble requires {len(STRICT_SEEDS)} predictions"
        )

    mean = probabilities[0] + probabilities[1]
    for probability in probabilities[2:]:
        mean = mean + probability
    mean = mean / len(probabilities)

    variance_sum = torch.square(mean - probabilities[0]) + torch.square(
        mean - probabilities[1]
    )
    for probability in probabilities[2:]:
        variance_sum = variance_sum + torch.square(mean - probability)
    uncertainty = torch.sqrt(variance_sum / len(probabilities))
    return mean, uncertainty


def _save_image_rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Perform torchvision save_image quantization on tensor's device."""

    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("Expected a single image")
        tensor = tensor[0]
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(tensor.shape)}")
    if tensor.shape[0] == 1:
        tensor = torch.cat((tensor, tensor, tensor), dim=0)
    elif tensor.shape[0] != 3:
        raise ValueError(f"Expected one or three channels, got {tensor.shape[0]}")

    # This is the operation sequence used by torchvision.utils.save_image.
    return (
        tensor.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to(device="cpu", dtype=torch.uint8)
        .numpy()
    )


def _quantize_sample_outputs(
    mean: torch.Tensor,
    uncertainty: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize soft, binary, and uncertainty maps before leaving device."""

    uncertainty_rgb = _save_image_rgb_uint8(uncertainty)
    mean_rgb = _save_image_rgb_uint8(mean)

    # zeros_like fixes the legacy CPU-target/CUDA-mask indexing bug while
    # retaining the intended >= 0.5 comparison and FP32 values.
    binary = torch.zeros_like(mean)
    binary[mean >= 0.5] = 1
    binary_rgb = _save_image_rgb_uint8(binary)
    return mean_rgb, binary_rgb, uncertainty_rgb


def _write_sample_outputs(
    paths: OutputPaths,
    *,
    name: str,
    width: int,
    height: int,
    mean_rgb: np.ndarray,
    binary_rgb: np.ndarray,
    uncertainty_rgb: np.ndarray,
) -> Mapping[str, float]:
    """Write all eight legacy products for one image."""

    started = time.perf_counter()
    filename = f"{name}.png"
    resize_uncertainty_path = paths.resize_uncertainty / filename
    raw_uncertainty_path = paths.raw_uncertainty / filename
    resize_path = paths.resize / filename
    resize_binary_path = paths.resize_binary / filename
    raw_path = paths.raw / filename
    raw_binary_path = paths.raw_binary / filename

    Image.fromarray(uncertainty_rgb).save(resize_uncertainty_path)
    with Image.open(resize_uncertainty_path) as image:
        # Deliberately omit resample= to match legacy Pillow behavior.
        raw_uncertainty_image = image.resize((int(width), int(height))).convert("L")
    raw_uncertainty_tensor = ToTensor()(raw_uncertainty_image)
    save_image(raw_uncertainty_tensor, raw_uncertainty_path)

    Image.fromarray(mean_rgb).save(resize_path)
    Image.fromarray(binary_rgb).save(resize_binary_path)

    with Image.open(resize_path) as image:
        mask_pred_image = image.resize((int(width), int(height))).convert("L")
    mask_pred_tensor = ToTensor()(mask_pred_image)
    raw_binary_tensor = torch.zeros_like(mask_pred_tensor)
    raw_binary_tensor[mask_pred_tensor >= 0.5] = 1
    save_image(mask_pred_tensor, raw_path)
    save_image(raw_binary_tensor, raw_binary_path)

    morphology_started = time.perf_counter()
    image = io.imread(str(resize_binary_path), as_gray=True).astype(np.int64)
    processed = image > 0
    processed = remove_small_objects(processed, 30, connectivity=5)
    io.imsave(
        str(paths.binary_process / filename),
        255 * processed.astype("uint8"),
        check_contrast=False,
    )
    skeleton = skeletonize(processed)
    io.imsave(
        str(paths.binary_skeleton / filename),
        255 * skeleton.astype("uint8"),
        check_contrast=False,
    )
    morphology_seconds = time.perf_counter() - morphology_started

    # Legacy computes fractal dimension, density, and average width here, but
    # the caller discards all three and writes no table. Keep observable images.
    return {
        "postprocess_s": time.perf_counter() - started,
        "morphology_s": morphology_seconds,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "synchronize"):
            mps.synchronize()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_vessel(config: VesselConfig) -> Mapping[str, Any]:
    """Run the strict ten-model binary-vessel ensemble."""

    total_started = time.perf_counter()
    _validate_config(config)

    discovery_started = time.perf_counter()
    image_paths = discover_images(config.input_dir)
    outputs = OutputPaths.create(config.output_dir)
    discovery_seconds = time.perf_counter() - discovery_started
    progress = ProgressReporter("M2", config.progress)

    device = resolve_device(config.device)
    cuda_index = device.index if device.index is not None else 0
    if device.type == "cuda":
        # PyTorch 2.7 on Windows requires the allocator to be initialized
        # before reset_peak_memory_stats is called in a fresh process.
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(cuda_index)

    loader = OrderedThreadBatchLoader(
        image_paths,
        batch_size=config.batch_size,
        threads=config.preprocess_threads,
        image_size=config.image_size,
        threshold=config.threshold,
    )
    loader.start()  # First CPU batch overlaps the ten checkpoint loads.

    model_load_started = time.perf_counter()
    try:
        with progress.task(
            "vessel-model-load", total=len(checkpoint_paths(config)), unit="models"
        ) as model_progress:
            models = _load_models(config, device, progress=model_progress)
        _synchronize(device)
        model_load_seconds = time.perf_counter() - model_load_started

        post_executor = ThreadPoolExecutor(
            max_workers=config.postprocess_threads,
            thread_name_prefix="m2-vessel-postprocess",
        )
        pending: deque[Future[Mapping[str, float]]] = deque()
        max_pending = max(config.postprocess_threads * 2, config.batch_size)
        postprocess_sum = 0.0
        morphology_sum = 0.0
        preprocess_wait_seconds = 0.0
        inference_transfer_seconds = 0.0
        batches = 0
        process_progress = progress.task(
            "vessel-inference-write", total=len(image_paths), unit="images"
        )

        def collect_one() -> None:
            nonlocal postprocess_sum, morphology_sum
            result = pending.popleft().result()
            postprocess_sum += float(result["postprocess_s"])
            morphology_sum += float(result["morphology_s"])
            process_progress.update()

        try:
            iterator = iter(loader)
            while True:
                wait_started = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    preprocess_wait_seconds += time.perf_counter() - wait_started
                    break
                preprocess_wait_seconds += time.perf_counter() - wait_started
                batches += 1

                inference_started = time.perf_counter()
                images = batch.images.to(device=device, dtype=torch.float32)
                probabilities: list[torch.Tensor] = []
                with torch.inference_mode():
                    for model in models:
                        logits = model(images)
                        probabilities.append(torch.sigmoid(logits))
                        del logits
                    mean, uncertainty = strict_ensemble_statistics(probabilities)

                    for index, name in enumerate(batch.names):
                        mean_rgb, binary_rgb, uncertainty_rgb = (
                            _quantize_sample_outputs(
                                mean[index, ...],
                                uncertainty[index, ...],
                            )
                        )
                        pending.append(
                            post_executor.submit(
                                _write_sample_outputs,
                                outputs,
                                name=name,
                                width=batch.widths[index],
                                height=batch.heights[index],
                                mean_rgb=mean_rgb,
                                binary_rgb=binary_rgb,
                                uncertainty_rgb=uncertainty_rgb,
                            )
                        )
                        if len(pending) >= max_pending:
                            collect_one()

                del images, probabilities, mean, uncertainty
                inference_transfer_seconds += time.perf_counter() - inference_started

            postprocess_tail_started = time.perf_counter()
            while pending:
                collect_one()
            postprocess_tail_seconds = time.perf_counter() - postprocess_tail_started
        except BaseException:
            process_progress.close("error")
            raise
        else:
            process_progress.close()
        finally:
            post_executor.shutdown(wait=True)
    finally:
        loader.close()

    _synchronize(device)
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(cuda_index))
        peak_reserved = int(torch.cuda.max_memory_reserved(cuda_index))
    else:
        peak_allocated = None
        peak_reserved = None

    summary: dict[str, Any] = {
        "module": "M2_Vessel_seg",
        "mode": "strict",
        "input_dir": str(Path(config.input_dir).resolve()),
        "output_dir": str(Path(config.output_dir).resolve()),
        "legacy_dir": str(Path(config.legacy_dir).resolve()),
        "device": str(device),
        "images": len(image_paths),
        "batches": batches,
        "batch_size": config.batch_size,
        "image_size": list(config.image_size),
        "threshold": config.threshold,
        "preprocess_threads": config.preprocess_threads,
        "postprocess_threads": config.postprocess_threads,
        "checkpoint_seeds": list(config.seeds),
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "timings_s": {
            "discovery_and_output_setup": discovery_seconds,
            "model_load": model_load_seconds,
            "preprocess_wait": preprocess_wait_seconds,
            "inference_quantize_transfer": inference_transfer_seconds,
            "postprocess_tail_wait": postprocess_tail_seconds,
            "postprocess_task_sum": postprocess_sum,
            "morphology_task_sum": morphology_sum,
            "total_wall": time.perf_counter() - total_started,
        },
    }
    if config.write_summary:
        _write_json_atomic(Path(config.output_dir) / "run_summary.json", summary)
    return summary


__all__ = [
    "CHECKPOINT_FILENAME",
    "LEGACY_ARGUMENT_DEFAULTS",
    "OrderedThreadBatchLoader",
    "OutputPaths",
    "STRICT_BATCH_SIZE",
    "STRICT_DATASET",
    "STRICT_IMAGE_SIZE",
    "STRICT_JOB_NAME",
    "STRICT_SEEDS",
    "STRICT_THRESHOLD",
    "VesselBatch",
    "VesselConfig",
    "VesselSample",
    "checkpoint_paths",
    "discover_images",
    "preprocess_image",
    "resolve_device",
    "run_vessel",
    "strict_ensemble_statistics",
]
