"""Strict-compatible AutoMorph M2 optic-disc/cup segmentation core.

The legacy implementation remains untouched. This module keeps its image
preprocessing, eight-checkpoint order, ensemble arithmetic, output encoding,
and downstream M3 routing while making every path explicit.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from glob import glob
import gc
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import shutil
import sys
import time
import types
from typing import Any, Callable, Iterator, Literal, Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure
from skimage.morphology import remove_small_objects
import torch
import torch.nn.functional as torch_f
from torchvision.transforms import functional as tv_f
from torchvision.utils import save_image

try:
    from ..progress import ProgressReporter, ProgressTask
except ImportError:  # imported as ``M2.disc_cup`` by direct stage runner
    _V2_ROOT = str(Path(__file__).resolve().parents[1])
    if _V2_ROOT not in sys.path:
        sys.path.insert(0, _V2_ROOT)
    from progress import ProgressReporter, ProgressTask


CHECKPOINT_IDS: tuple[int, ...] = (28, 30, 32, 34, 36, 38, 40, 42)
IMAGE_SIZE: tuple[int, int] = (512, 512)
LEGACY_BATCH_SIZE = 16


@dataclass(frozen=True, slots=True)
class DiscCupConfig:
    """Execution contract for the strict disc/cup branch."""

    data_root: Path
    input_dir: Path | None = None
    legacy_module_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
        / "runtime"
        / "M2_lwnet_disc_cup"
    )
    device: str = "auto"
    preprocess_threads: int = 4
    postprocess_threads: int = 4
    batch_size: int = LEGACY_BATCH_SIZE
    image_size: tuple[int, int] = IMAGE_SIZE
    checkpoint_ids: tuple[int, ...] = CHECKPOINT_IDS
    aggregation: Literal["legacy_cpu", "device"] = "legacy_cpu"
    intersection_backend: Literal["reference", "vectorized"] = "vectorized"
    run_optic_postprocess: bool = True
    loader_lookahead_batches: int = 2
    progress: str = "off"

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root).resolve())
        if self.input_dir is not None:
            object.__setattr__(self, "input_dir", Path(self.input_dir).resolve())
        object.__setattr__(
            self, "legacy_module_root", Path(self.legacy_module_root).resolve()
        )
        if self.batch_size != LEGACY_BATCH_SIZE:
            raise ValueError(
                "Strict M2 disc/cup fixes batch_size=16 because changing batch "
                "shape can change backend kernels and output pixels."
            )
        if tuple(self.image_size) != IMAGE_SIZE:
            raise ValueError("Strict M2 disc/cup fixes image_size=(512, 512).")
        if tuple(self.checkpoint_ids) != CHECKPOINT_IDS:
            raise ValueError(
                "Strict M2 disc/cup fixes checkpoints to 28,30,32,34,36,38,40,42."
            )
        if self.preprocess_threads < 1:
            raise ValueError("preprocess_threads must be at least one")
        if self.postprocess_threads < 1:
            raise ValueError("postprocess_threads must be at least one")
        if self.loader_lookahead_batches < 1:
            raise ValueError("loader_lookahead_batches must be at least one")


@dataclass(frozen=True, slots=True)
class ImageEntry:
    stem: str
    path: Path


@dataclass(frozen=True, slots=True)
class PreparedImage:
    stem: str
    tensor: torch.Tensor
    original_size: tuple[int, int]


@dataclass(frozen=True, slots=True)
class PreparedBatch:
    names: tuple[str, ...]
    images: torch.Tensor
    original_sizes: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class DiscCupOutputPaths:
    resized: Path
    raw: Path
    resize_uncertainty: Path
    raw_uncertainty: Path

    @classmethod
    def from_data_root(cls, data_root: Path) -> "DiscCupOutputPaths":
        base = data_root / "Results" / "M2" / "optic_disc_cup"
        return cls(
            resized=base / "resized",
            raw=base / "raw",
            resize_uncertainty=base / "resize_uncertainty",
            raw_uncertainty=base / "raw_uncertainty",
        )

    def create(self) -> None:
        for path in (
            self.resized,
            self.raw,
            self.resize_uncertainty,
            self.raw_uncertainty,
        ):
            path.mkdir(parents=True, exist_ok=True)


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve NVIDIA CUDA, Apple MPS, or CPU."""

    requested_text = str(requested)
    if requested_text == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested_text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")
    if (
        device.type == "cuda"
        and device.index is not None
        and device.index >= torch.cuda.device_count()
    ):
        raise RuntimeError(f"CUDA device index out of range: {device.index}")
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
    return device


def discover_legacy_images(image_dir: Path) -> list[ImageEntry]:
    """Mirror legacy listdir, splitext, then first glob match discovery."""

    image_dir = Path(image_dir).resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"M1 Good_quality directory not found: {image_dir}")
    ids = [
        os.path.splitext(name)[0]
        for name in os.listdir(image_dir)
        if not name.startswith(".")
    ]
    entries: list[ImageEntry] = []
    for stem in ids:
        matches = glob(str(image_dir / f"{stem}.*"))
        if not matches:
            raise FileNotFoundError(f"No image matches legacy stem {stem!r}")
        entries.append(ImageEntry(stem=stem, path=Path(matches[0])))
    return entries


def preprocess_legacy_image(
    entry: ImageEntry, image_size: tuple[int, int] = IMAGE_SIZE
) -> PreparedImage:
    """Reproduce PIL bilinear resize followed by torchvision ToTensor."""

    with Image.open(entry.path) as image:
        original_size = (int(image.size[0]), int(image.size[1]))
        resized = tv_f.resize(image, image_size, Image.BILINEAR)
        tensor = tv_f.to_tensor(resized)
    return PreparedImage(entry.stem, tensor, original_size)


class ThreadedBatchLoader:
    """Ordered shared-process preprocessing with bounded look-ahead."""

    def __init__(
        self,
        entries: Sequence[ImageEntry],
        *,
        batch_size: int = LEGACY_BATCH_SIZE,
        threads: int = 4,
        image_size: tuple[int, int] = IMAGE_SIZE,
        lookahead_batches: int = 2,
    ) -> None:
        self.entries = tuple(entries)
        self.batch_size = batch_size
        self.threads = threads
        self.image_size = image_size
        self.lookahead_batches = lookahead_batches
        self.wait_seconds = 0.0
        self._executor: ThreadPoolExecutor | None = None
        self._pending: deque[list[Future[PreparedImage]]] = deque()
        self._next_index = 0
        self._started = False
        self._closed = False

    def __len__(self) -> int:
        return math.ceil(len(self.entries) / self.batch_size)

    def _submit_one_batch(self) -> bool:
        if self._next_index >= len(self.entries):
            return False
        assert self._executor is not None
        stop = min(self._next_index + self.batch_size, len(self.entries))
        batch_entries = self.entries[self._next_index : stop]
        self._next_index = stop
        self._pending.append(
            [
                self._executor.submit(
                    preprocess_legacy_image, entry, self.image_size
                )
                for entry in batch_entries
            ]
        )
        return True

    def start(self) -> None:
        if self._started:
            return
        if self._closed:
            raise RuntimeError("loader is already closed")
        self._executor = ThreadPoolExecutor(
            max_workers=self.threads, thread_name_prefix="m2-disc-preprocess"
        )
        self._started = True
        while len(self._pending) < self.lookahead_batches:
            if not self._submit_one_batch():
                break

    def __iter__(self) -> Iterator[PreparedBatch]:
        self.start()
        while self._pending:
            started = time.perf_counter()
            futures = self._pending.popleft()
            prepared = [future.result() for future in futures]
            self.wait_seconds += time.perf_counter() - started
            while len(self._pending) < self.lookahead_batches:
                if not self._submit_one_batch():
                    break
            yield PreparedBatch(
                names=tuple(item.stem for item in prepared),
                images=torch.stack([item.tensor for item in prepared], dim=0),
                original_sizes=tuple(item.original_size for item in prepared),
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        self._pending.clear()

    def __enter__(self) -> "ThreadedBatchLoader":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


_GET_ARCH_CACHE: dict[Path, Callable[..., torch.nn.Module]] = {}


def load_legacy_get_arch(legacy_module_root: Path) -> Callable[..., torch.nn.Module]:
    """Import this branch privately so generic legacy models packages cannot clash."""

    legacy_module_root = Path(legacy_module_root).resolve()
    cached = _GET_ARCH_CACHE.get(legacy_module_root)
    if cached is not None:
        return cached
    models_dir = legacy_module_root / "models"
    source = models_dir / "get_model.py"
    if not source.is_file():
        raise FileNotFoundError(f"Legacy disc model factory not found: {source}")

    suffix = hashlib.sha1(str(models_dir).encode("utf-8")).hexdigest()[:12]
    package_name = f"_automorph_v2_disc_models_{suffix}"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(models_dir)]  # type: ignore[attr-defined]
        package.__package__ = package_name
        sys.modules[package_name] = package

    module_name = f"{package_name}.get_model"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create import spec for {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    get_arch = getattr(module, "get_arch")
    _GET_ARCH_CACHE[legacy_module_root] = get_arch
    return get_arch


def _load_checkpoint(path: Path) -> dict[str, Any]:
    """Load on CPU with weights-only and mmap compatibility fallbacks."""

    attempts = (
        {"weights_only": True, "mmap": True},
        {"weights_only": True},
        {},
    )
    last_error: BaseException | None = None
    for kwargs in attempts:
        try:
            value = torch.load(path, map_location="cpu", **kwargs)
            if not isinstance(value, dict):
                raise TypeError(f"Checkpoint is not a mapping: {path}")
            return value
        except (TypeError, RuntimeError) as error:
            last_error = error
    assert last_error is not None
    raise last_error


def load_ensemble_models(
    config: DiscCupConfig,
    device: torch.device,
    *,
    get_arch: Callable[..., torch.nn.Module] | None = None,
    progress: ProgressTask | None = None,
) -> list[torch.nn.Module]:
    """Build and load all eight WNet members in legacy order."""

    if get_arch is None:
        get_arch = load_legacy_get_arch(config.legacy_module_root)
    experiment_root = (
        config.legacy_module_root
        / "experiments"
        / "wnet_All_three_1024_disc_cup"
    )
    models: list[torch.nn.Module] = []
    for checkpoint_id in config.checkpoint_ids:
        checkpoint_path = experiment_root / str(checkpoint_id) / "model_checkpoint.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Disc/cup checkpoint not found: {checkpoint_path}")
        model = get_arch("wnet", n_classes=3)
        checkpoint = _load_checkpoint(checkpoint_path)
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, dict):
            raise KeyError(f"model_state_dict missing from {checkpoint_path}")
        model.load_state_dict(state, strict=True)
        # UNet1 cannot be pruned: x1 is concatenated with x before UNet2.
        # Returning x2 alone only avoids retaining the already-consumed x1.
        if hasattr(model, "mode"):
            model.mode = "test"
        model.eval()
        model.to(device)
        models.append(model)
        del checkpoint, state
        if progress is not None:
            progress.update()
    return models


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        synchronize = getattr(torch.mps, "synchronize", None)
        if synchronize is not None:
            synchronize()


def ensemble_predict_legacy(
    models: Sequence[torch.nn.Module],
    images: torch.Tensor,
    device: torch.device,
    *,
    aggregation: Literal["legacy_cpu", "device"] = "legacy_cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return legacy decoded CPU labels and device uncertainty."""

    if len(models) != len(CHECKPOINT_IDS):
        raise ValueError(f"Expected eight models, got {len(models)}")
    images = images.to(device=device, dtype=torch.float32)
    probabilities: list[torch.Tensor] = []
    probability_sum: int | torch.Tensor = 0
    with torch.inference_mode():
        for model in models:
            output = model(images)
            logits = output[-1] if isinstance(output, (tuple, list)) else output
            probability = torch_f.softmax(logits, dim=1)
            probabilities.append(probability)
            if aggregation == "legacy_cpu":
                probability_sum += probability.type(torch.FloatTensor)
            elif aggregation == "device":
                probability_sum += probability
            else:
                raise ValueError(f"Unknown aggregation mode: {aggregation}")

        assert isinstance(probability_sum, torch.Tensor)
        mean_probability = (probability_sum / 8).to(device=device)
        squared_deviation = torch.square(mean_probability - probabilities[0])
        for probability in probabilities[1:]:
            squared_deviation = squared_deviation + torch.square(
                mean_probability - probability
            )
        uncertainty = torch.sqrt(squared_deviation / 8)
        _, decoded = torch.max(mean_probability, 1)
        decoded = decoded.type(torch.FloatTensor)
    return decoded, uncertainty


def write_prediction_batch(
    decoded: torch.Tensor,
    uncertainty: torch.Tensor,
    batch: PreparedBatch,
    paths: DiscCupOutputPaths,
) -> None:
    """Encode the four legacy M2 output families."""

    for index, image_name in enumerate(batch.names):
        width, height = batch.original_sizes[index]
        uncertainty_small = paths.resize_uncertainty / f"{image_name}.png"
        save_image(uncertainty[index, ...] * 255, str(uncertainty_small))
        save_image(
            uncertainty[index, 1, ...] * 255,
            str(paths.resize_uncertainty / f"{image_name}_disc.png"),
        )
        save_image(
            uncertainty[index, 2, ...] * 255,
            str(paths.resize_uncertainty / f"{image_name}_cup.png"),
        )
        with Image.open(uncertainty_small) as uncertainty_image:
            resized_uncertainty = uncertainty_image.resize((width, height))
            uncertainty_tensor = tv_f.to_tensor(resized_uncertainty)
        save_image(
            uncertainty_tensor,
            str(paths.raw_uncertainty / f"{image_name}.png"),
        )

        label = decoded[index, ...]
        image_r = np.zeros((label.shape[0], label.shape[1]))
        image_g = np.zeros((label.shape[0], label.shape[1]))
        image_b = np.zeros((label.shape[0], label.shape[1]))
        image_r[label == 1] = 255
        image_b[label == 2] = 255
        image_b = remove_small_objects(image_b > 0, 50)
        image_r = remove_small_objects(image_r > 0, 100)
        encoded = np.concatenate(
            (
                image_b[..., np.newaxis],
                image_g[..., np.newaxis],
                image_r[..., np.newaxis],
            ),
            axis=2,
        )
        encoded_float = np.float32(encoded) * 255
        if not cv2.imwrite(str(paths.resized / f"{image_name}.png"), encoded_float):
            raise OSError(f"Failed to write resized disc/cup mask for {image_name}")
        raw = cv2.resize(
            encoded_float, (width, height), interpolation=cv2.INTER_NEAREST
        )
        if not cv2.imwrite(str(paths.raw / f"{image_name}.png"), raw):
            raise OSError(f"Failed to write raw disc/cup mask for {image_name}")


def quantize_uncertainty_batch_legacy(
    uncertainty: torch.Tensor,
    *,
    consume: bool = False,
) -> np.ndarray:
    """Batch the exact ``save_image`` arithmetic and perform one D2H copy.

    Legacy calls ``save_image(uncertainty * 255)``.  ``save_image`` then does
    a second multiply by 255, adds 0.5, clamps, permutes, and converts to CPU
    uint8.  Keeping the two multiplies separate matters for strict floating
    point equivalence.  Channels 3 and 4 in the returned NHWC array are the
    scalar disc and cup images; the worker expands each to RGB when writing.

    ``consume=True`` lets the production path reuse the uncertainty storage,
    which is dead immediately after submission.  Tests and external callers
    can retain the input with the default non-consuming behavior.
    """

    if uncertainty.ndim != 4 or uncertainty.shape[1] != 3:
        raise ValueError("uncertainty must have shape (batch, 3, height, width)")
    with torch.inference_mode():
        quantization_work = uncertainty if consume else uncertainty.clone()
        quantization_work.mul_(255)
        quantization_work.mul_(255).add_(0.5).clamp_(0, 255)
        quantized = quantization_work.to(dtype=torch.uint8)
        packed = torch.cat(
            (quantized, quantized[:, 1:2], quantized[:, 2:3]), dim=1
        )
        # This is the sole device-to-host transfer for all uncertainty output
        # variants in one inference batch.
        packed_cpu = packed.permute(0, 2, 3, 1).to(device="cpu")
    return packed_cpu.numpy()


def _encode_label_legacy(label: np.ndarray) -> np.ndarray:
    """Reproduce the legacy BGR mask construction and object filtering."""

    image_r = np.zeros((label.shape[0], label.shape[1]))
    image_g = np.zeros((label.shape[0], label.shape[1]))
    image_b = np.zeros((label.shape[0], label.shape[1]))
    image_r[label == 1] = 255
    image_b[label == 2] = 255
    image_b = remove_small_objects(image_b > 0, 50)
    image_r = remove_small_objects(image_r > 0, 100)
    encoded = np.concatenate(
        (
            image_b[..., np.newaxis],
            image_g[..., np.newaxis],
            image_r[..., np.newaxis],
        ),
        axis=2,
    )
    return np.float32(encoded) * 255


def _write_quantized_prediction(
    decoded: np.ndarray,
    packed_uncertainty: np.ndarray,
    image_name: str,
    original_size: tuple[int, int],
    paths: DiscCupOutputPaths,
) -> float:
    """Write one already-quantized image; safe to run in a worker thread."""

    started = time.perf_counter()
    width, height = original_size
    if packed_uncertainty.ndim != 3 or packed_uncertainty.shape[2] != 5:
        raise ValueError("packed uncertainty must have shape (height, width, 5)")

    full_image = Image.fromarray(packed_uncertainty[..., :3])
    full_image.save(paths.resize_uncertainty / f"{image_name}.png")
    Image.fromarray(
        np.repeat(packed_uncertainty[..., 3:4], 3, axis=2)
    ).save(paths.resize_uncertainty / f"{image_name}_disc.png")
    Image.fromarray(
        np.repeat(packed_uncertainty[..., 4:5], 3, axis=2)
    ).save(paths.resize_uncertainty / f"{image_name}_cup.png")

    # The legacy save/open -> PIL resize -> ToTensor -> save_image round-trip
    # does not change any uint8 samples.  Resizing the identical in-memory RGB
    # image therefore preserves both pixels and deterministic PNG bytes.
    full_image.resize((width, height)).save(
        paths.raw_uncertainty / f"{image_name}.png"
    )

    encoded_float = _encode_label_legacy(decoded)
    if not cv2.imwrite(str(paths.resized / f"{image_name}.png"), encoded_float):
        raise OSError(f"Failed to write resized disc/cup mask for {image_name}")
    raw = cv2.resize(encoded_float, (width, height), interpolation=cv2.INTER_NEAREST)
    if not cv2.imwrite(str(paths.raw / f"{image_name}.png"), raw):
        raise OSError(f"Failed to write raw disc/cup mask for {image_name}")
    return time.perf_counter() - started


class ThreadedPredictionWriter:
    """Overlap CPU image encoding with the following GPU inference batch."""

    def __init__(self, paths: DiscCupOutputPaths, *, threads: int = 4) -> None:
        if threads < 1:
            raise ValueError("threads must be at least one")
        self.paths = paths
        self.threads = threads
        self._executor = ThreadPoolExecutor(
            max_workers=threads, thread_name_prefix="m2-disc-output"
        )
        self._futures: list[Future[float]] = []
        self._closed = False
        self._first_submit_at: float | None = None
        self.submit_seconds = 0.0
        self.wait_seconds = 0.0
        self.cpu_seconds = 0.0
        self.wall_span_seconds = 0.0

    def submit(
        self,
        decoded: torch.Tensor,
        uncertainty: torch.Tensor,
        batch: PreparedBatch,
    ) -> None:
        """Quantize one batch, copy it once to CPU, and enqueue per-image IO."""

        if self._closed:
            raise RuntimeError("prediction writer is already closed")
        if decoded.device.type != "cpu":
            raise ValueError(
                "decoded must already be on CPU to preserve the one-D2H output contract"
            )
        batch_length = len(batch.names)
        if (
            decoded.shape[0] != batch_length
            or uncertainty.shape[0] != batch_length
            or len(batch.original_sizes) != batch_length
        ):
            raise ValueError("prediction and metadata batch lengths do not match")

        started = time.perf_counter()
        if self._first_submit_at is None:
            self._first_submit_at = started
        packed = quantize_uncertainty_batch_legacy(uncertainty, consume=True)
        decoded_array = decoded.detach().numpy()
        for index, image_name in enumerate(batch.names):
            self._futures.append(
                self._executor.submit(
                    _write_quantized_prediction,
                    decoded_array[index],
                    packed[index],
                    image_name,
                    batch.original_sizes[index],
                    self.paths,
                )
            )
        self.submit_seconds += time.perf_counter() - started

    def close(self) -> None:
        """Wait for every write and surface worker failures before postprocess."""

        if self._closed:
            return
        self._closed = True
        started = time.perf_counter()
        first_error: BaseException | None = None
        try:
            for future in self._futures:
                try:
                    self.cpu_seconds += future.result()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self.wait_seconds = time.perf_counter() - started
            if self._first_submit_at is not None:
                self.wall_span_seconds = time.perf_counter() - self._first_submit_at
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "ThreadedPredictionWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def intersection_reference(
    mask: np.ndarray, vessel: np.ndarray, row: int, column: int
) -> tuple[np.ndarray, float]:
    """The original per-pixel eight-neighbour test, kept as an oracle."""

    row_less = max(0, row - 1)
    column_less = max(0, column - 1)
    row_more = min(vessel.shape[0] - 1, row + 1)
    column_more = min(vessel.shape[1] - 1, column + 1)
    active_neighbours = (
        (vessel[row_less, column_less] > 0).astype("float")
        + (vessel[row_less, column] > 0).astype("float")
        + (vessel[row_less, column_more] > 0).astype("float")
        + (vessel[row, column_less] > 0).astype("float")
        + (vessel[row, column_more] > 0).astype("float")
        + (vessel[row_more, column_less] > 0).astype("float")
        + (vessel[row_more, column] > 0).astype("float")
        + (vessel[row_more, column_more] > 0).astype("float")
    )
    if active_neighbours > 2:
        cv2.circle(
            mask,
            (column, row),
            radius=1,
            color=(0, 0, 0),
            thickness=-1,
        )
    return mask, float(active_neighbours)


def build_intersection_mask_reference(vessel: np.ndarray) -> np.ndarray:
    """Run the exact legacy nested loops for tests and fallback execution."""

    if vessel.ndim != 2:
        raise ValueError("intersection input must be a two-dimensional array")
    mask = np.ones(vessel.shape)
    for row in range(1, mask.shape[0] - 1):
        for column in range(1, mask.shape[1] - 1):
            if vessel[row, column] > 0:
                intersection_reference(mask, vessel, row, column)
    return mask


def build_intersection_mask_vectorized(vessel: np.ndarray) -> np.ndarray:
    """Vectorize candidate detection while retaining exact OpenCV circle rasterization.

    The legacy loop never evaluates border centres. Candidate centres are
    emitted in the same row-major order, and cv2.circle remains the authority
    for its radius-one footprint (rather than assuming cross or 3x3).
    """

    if vessel.ndim != 2:
        raise ValueError("intersection input must be a two-dimensional array")
    positive = vessel > 0
    byte = positive.astype(np.uint8, copy=False)
    neighbours = (
        byte[:-2, :-2]
        + byte[:-2, 1:-1]
        + byte[:-2, 2:]
        + byte[1:-1, :-2]
        + byte[1:-1, 2:]
        + byte[2:, :-2]
        + byte[2:, 1:-1]
        + byte[2:, 2:]
    )
    candidates = positive[1:-1, 1:-1] & (neighbours > 2)
    mask = np.ones(vessel.shape)
    for relative_row, relative_column in np.argwhere(candidates):
        row = int(relative_row) + 1
        column = int(relative_column) + 1
        cv2.circle(
            mask,
            (column, row),
            radius=1,
            color=(0, 0, 0),
            thickness=-1,
        )
    return mask


def remove_skeleton_intersections(
    vessel: np.ndarray,
    backend: Literal["reference", "vectorized"] = "vectorized",
) -> np.ndarray:
    if backend == "reference":
        mask = build_intersection_mask_reference(vessel)
    elif backend == "vectorized":
        mask = build_intersection_mask_vectorized(vessel)
    else:
        raise ValueError(f"Unknown intersection backend: {backend}")
    return vessel * mask


_MODALITY_SUFFIXES: dict[str, str] = {
    "binary_process": "binary_process",
    "artery_process": "artery_process",
    "vein_process": "vein_process",
    "binary_skeleton": "binary_skeleton",
    "artery_skeleton": "artery_skeleton",
    "vein_skeleton": "vein_skeleton",
}


def _source_mask_paths(data_root: Path, filename: str) -> dict[str, Path]:
    m2_root = data_root / "Results" / "M2"
    binary_root = m2_root / "binary_vessel"
    artery_vein_root = m2_root / "artery_vein"
    return {
        "binary_process": binary_root / "binary_process" / filename,
        "artery_process": artery_vein_root / "artery_binary_process" / filename,
        "vein_process": artery_vein_root / "vein_binary_process" / filename,
        "binary_skeleton": binary_root / "binary_skeleton" / filename,
        "artery_skeleton": artery_vein_root / "artery_binary_skeleton" / filename,
        "vein_skeleton": artery_vein_root / "vein_binary_skeleton" / filename,
    }


def _zone_output_directories(
    data_root: Path,
) -> dict[tuple[str, str | None, str], Path]:
    m2_root = data_root / "Results" / "M2"
    binary_root = m2_root / "binary_vessel"
    artery_vein_root = m2_root / "artery_vein"
    result: dict[tuple[str, str | None, str], Path] = {}
    for centre in ("disc", "macular"):
        for zone in (None, "B", "C"):
            for modality, suffix in _MODALITY_SUFFIXES.items():
                root = binary_root if modality.startswith("binary") else artery_vein_root
                if zone is None:
                    name = f"{centre}_centred_{suffix}"
                elif centre == "disc":
                    name = f"Zone_{zone}_disc_centred_{suffix}"
                else:
                    name = f"macular_Zone_{zone}_centred_{suffix}"
                path = root / name
                path.mkdir(parents=True, exist_ok=True)
                result[(centre, zone, modality)] = path
    return result


def _read_source_masks(data_root: Path, filename: str) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for modality, path in _source_mask_paths(data_root, filename).items():
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Required M2 mask is unreadable: {path}")
        result[modality] = image[..., 0]
    return result


def _copy_whole_masks(
    data_root: Path,
    filename: str,
    centre: Literal["disc", "macular"],
    output_dirs: dict[tuple[str, str | None, str], Path],
) -> None:
    for modality, source in _source_mask_paths(data_root, filename).items():
        shutil.copy(source, output_dirs[(centre, None, modality)] / filename)


def _write_zone_masks(
    arrays: dict[str, np.ndarray],
    mask: np.ndarray,
    filename: str,
    centre: Literal["disc", "macular"],
    zone: Literal["B", "C"],
    output_dirs: dict[tuple[str, str | None, str], Path],
) -> None:
    for modality, array in arrays.items():
        destination = output_dirs[(centre, zone, modality)] / filename
        if not cv2.imwrite(str(destination), array * mask):
            raise OSError(f"Failed to write zone mask: {destination}")


def _append_invalid_macular(
    *,
    data_root: Path,
    filename: str,
    output_dirs: dict[tuple[str, str | None, str], Path],
    macular_names: list[str],
    macular_vertical_disc: list[float],
    macular_horizontal_disc: list[float],
    macular_vertical_cup: list[float],
    macular_horizontal_cup: list[float],
    macular_vertical_cdr: list[float],
    macular_horizontal_cdr: list[float],
) -> None:
    macular_names.append(filename)
    _copy_whole_masks(data_root, filename, "macular", output_dirs)
    macular_vertical_disc.append(-1)
    macular_horizontal_disc.append(-1)
    macular_vertical_cup.append(-1)
    macular_horizontal_cup.append(-1)
    macular_vertical_cdr.append(-1)
    macular_horizontal_cdr.append(-1)


def optic_disc_centre(
    data_root: Path,
    *,
    result_path: Path | None = None,
    intersection_backend: Literal["reference", "vectorized"] = "vectorized",
    progress: ProgressTask | None = None,
) -> dict[str, Any]:
    """Reproduce legacy disc/macular routing, M3 CSVs, and all zone masks."""

    started = time.perf_counter()
    data_root = Path(data_root).resolve()
    if result_path is None:
        result_path = (
            data_root / "Results" / "M2" / "optic_disc_cup" / "resized"
        )
    result_path = Path(result_path).resolve()
    if not result_path.is_dir():
        raise FileNotFoundError(f"Disc/cup resized output not found: {result_path}")

    disc_results = data_root / "Results" / "M3" / "Disc_centred"
    macular_results = data_root / "Results" / "M3" / "Macular_centred"
    disc_results.mkdir(parents=True, exist_ok=True)
    macular_results.mkdir(parents=True, exist_ok=True)
    output_dirs = _zone_output_directories(data_root)

    filenames = sorted(
        name for name in os.listdir(result_path) if name != ".ipynb_checkpoints"
    )
    resolution_path = data_root / "Results" / "M0" / "crop_info.csv"
    resolution_table = pd.read_csv(resolution_path)

    optic_names: list[str] = []
    optic_vertical_disc: list[float] = []
    optic_horizontal_disc: list[float] = []
    optic_vertical_cup: list[float] = []
    optic_horizontal_cup: list[float] = []
    optic_vertical_cdr: list[float] = []
    optic_horizontal_cdr: list[float] = []
    macular_names: list[str] = []
    macular_vertical_disc: list[float] = []
    macular_horizontal_disc: list[float] = []
    macular_vertical_cup: list[float] = []
    macular_horizontal_cup: list[float] = []
    macular_vertical_cdr: list[float] = []
    macular_horizontal_cdr: list[float] = []
    invalid_count = 0

    for filename in filenames:
        disc_cup = cv2.imread(str(result_path / filename))
        if disc_cup is None:
            raise OSError(f"Unreadable disc/cup mask: {result_path / filename}")
        disc_cup_912 = cv2.resize(
            disc_cup, (912, 912), interpolation=cv2.INTER_NEAREST
        )
        resolution_values = resolution_table.loc[
            resolution_table["Name"] == filename, "Scale_resolution"
        ].values
        if len(resolution_values) == 0:
            raise KeyError(f"No Scale_resolution for {filename} in {resolution_path}")
        resolution_scale = resolution_values[0]

        try:
            disc = disc_cup_912[..., 2]
            cup = disc_cup_912[..., 0]

            disc_mask = measure.label(disc)
            regions = measure.regionprops(disc_mask)
            regions.sort(key=lambda region: region.area, reverse=True)
            if len(regions) > 1:
                for region in regions[2:]:
                    disc_mask[region.coords[:, 0], region.coords[:, 1]] = 0
            disc[disc_mask != 0] = 255

            cup_mask = measure.label(cup)
            regions = measure.regionprops(cup_mask)
            regions.sort(key=lambda region: region.area, reverse=True)
            if len(regions) > 1:
                for region in regions[2:]:
                    cup_mask[region.coords[:, 0], region.coords[:, 1]] = 0
            cup[cup_mask != 0] = 255

            disc_index = np.where(disc > 0)
            disc_width_index = disc_index[1]
            disc_height_index = disc_index[0]
            disc_horizontal_width = np.max(disc_width_index) - np.min(
                disc_width_index
            )
            disc_vertical_height = np.max(disc_height_index) - np.min(
                disc_height_index
            )
            cup_index = np.where(cup > 0)
            cup_width_index = cup_index[1]
            cup_height_index = cup_index[0]
            cup_horizontal_width = np.max(cup_width_index) - np.min(cup_width_index)
            cup_vertical_height = np.max(cup_height_index) - np.min(cup_height_index)
            cup_width_centre = np.mean(cup_width_index)
            cup_height_centre = np.mean(cup_height_index)

            valid = (
                disc_horizontal_width < (disc.shape[0] / 3)
                and disc_vertical_height < (disc.shape[1] / 3)
                and cup_width_centre <= np.max(disc_width_index)
                and cup_width_centre >= np.min(disc_width_index)
                and cup_height_centre <= np.max(disc_height_index)
                and cup_height_centre >= np.min(disc_height_index)
                and cup_vertical_height < disc_vertical_height
                and cup_horizontal_width < disc_horizontal_width
            )
            if not valid:
                invalid_count += 1
                _append_invalid_macular(
                    data_root=data_root,
                    filename=filename,
                    output_dirs=output_dirs,
                    macular_names=macular_names,
                    macular_vertical_disc=macular_vertical_disc,
                    macular_horizontal_disc=macular_horizontal_disc,
                    macular_vertical_cup=macular_vertical_cup,
                    macular_horizontal_cup=macular_horizontal_cup,
                    macular_vertical_cdr=macular_vertical_cdr,
                    macular_horizontal_cdr=macular_horizontal_cdr,
                )
                if progress is not None:
                    progress.update()
                continue

            whole_index = np.where(disc_cup_912 > 0)
            whole_width = whole_index[1]
            whole_height = whole_index[0]
            horizontal_distance = np.absolute(
                np.mean(whole_height) - disc_cup_912.shape[1] / 2
            )
            vertical_distance = np.absolute(
                np.mean(whole_width) - disc_cup_912.shape[0] / 2
            )
            distance = np.sqrt(
                np.square(horizontal_distance) + np.square(vertical_distance)
            )

            arrays = _read_source_masks(data_root, filename)
            for modality in (
                "binary_skeleton",
                "artery_skeleton",
                "vein_skeleton",
            ):
                arrays[modality] = remove_skeleton_intersections(
                    arrays[modality], intersection_backend
                )

            zone_mask_b = np.zeros(arrays["binary_process"].shape)
            zone_mask_c = np.zeros(arrays["binary_process"].shape)
            zone_centre = (int(np.mean(whole_width)), int(np.mean(whole_height)))
            radius = max(
                int(disc_horizontal_width / 2), int(disc_vertical_height / 2)
            )
            cv2.circle(
                zone_mask_b,
                zone_centre,
                radius=3 * radius,
                color=(255, 255, 255),
                thickness=-1,
            )
            cv2.circle(
                zone_mask_b,
                zone_centre,
                radius=2 * radius,
                color=(0, 0, 0),
                thickness=-1,
            )
            zone_mask_b = zone_mask_b / 255
            cv2.circle(
                zone_mask_c,
                zone_centre,
                radius=5 * radius,
                color=(255, 255, 255),
                thickness=-1,
            )
            cv2.circle(
                zone_mask_c,
                zone_centre,
                radius=2 * radius,
                color=(0, 0, 0),
                thickness=-1,
            )
            zone_mask_c = zone_mask_c / 255

            if (distance / disc_cup_912.shape[1]) < 0.1:
                centre: Literal["disc", "macular"] = "disc"
                optic_names.append(filename)
                _write_zone_masks(
                    arrays, zone_mask_b, filename, centre, "B", output_dirs
                )
                _write_zone_masks(
                    arrays, zone_mask_c, filename, centre, "C", output_dirs
                )
                _copy_whole_masks(data_root, filename, centre, output_dirs)
                optic_vertical_disc.append(disc_vertical_height * resolution_scale)
                optic_horizontal_disc.append(
                    disc_horizontal_width * resolution_scale
                )
                optic_vertical_cup.append(cup_vertical_height * resolution_scale)
                optic_horizontal_cup.append(cup_horizontal_width * resolution_scale)
                optic_vertical_cdr.append(cup_vertical_height / disc_vertical_height)
                optic_horizontal_cdr.append(
                    cup_horizontal_width / disc_horizontal_width
                )
            else:
                centre = "macular"
                macular_names.append(filename)
                _write_zone_masks(
                    arrays, zone_mask_b, filename, centre, "B", output_dirs
                )
                _write_zone_masks(
                    arrays, zone_mask_c, filename, centre, "C", output_dirs
                )
                _copy_whole_masks(data_root, filename, centre, output_dirs)
                macular_vertical_disc.append(disc_vertical_height * resolution_scale)
                macular_horizontal_disc.append(
                    disc_horizontal_width * resolution_scale
                )
                macular_vertical_cup.append(cup_vertical_height * resolution_scale)
                macular_horizontal_cup.append(
                    cup_horizontal_width * resolution_scale
                )
                macular_vertical_cdr.append(
                    cup_vertical_height / disc_vertical_height
                )
                macular_horizontal_cdr.append(
                    cup_horizontal_width / disc_horizontal_width
                )
        except Exception:
            invalid_count += 1
            _append_invalid_macular(
                data_root=data_root,
                filename=filename,
                output_dirs=output_dirs,
                macular_names=macular_names,
                macular_vertical_disc=macular_vertical_disc,
                macular_horizontal_disc=macular_horizontal_disc,
                macular_vertical_cup=macular_vertical_cup,
                macular_horizontal_cup=macular_horizontal_cup,
                macular_vertical_cdr=macular_vertical_cdr,
                macular_horizontal_cdr=macular_horizontal_cdr,
            )
        if progress is not None:
            progress.update()

    columns = {
        "Name": optic_names,
        "Disc_height": optic_vertical_disc,
        "Disc_width": optic_horizontal_disc,
        "Cup_height": optic_vertical_cup,
        "Cup_width": optic_horizontal_cup,
        "CDR_vertical": optic_vertical_cdr,
        "CDR_horizontal": optic_horizontal_cdr,
    }
    pd.DataFrame(columns).to_csv(
        disc_results / "Disc_cup_results.csv", index=None, encoding="utf8"
    )
    columns = {
        "Name": macular_names,
        "Disc_height": macular_vertical_disc,
        "Disc_width": macular_horizontal_disc,
        "Cup_height": macular_vertical_cup,
        "Cup_width": macular_horizontal_cup,
        "CDR_vertical": macular_vertical_cdr,
        "CDR_horizontal": macular_horizontal_cdr,
    }
    pd.DataFrame(columns).to_csv(
        macular_results / "Disc_cup_results.csv", index=None, encoding="utf8"
    )
    return {
        "images": len(filenames),
        "disc_centred": len(optic_names),
        "macular_centred": len(macular_names),
        "invalid": invalid_count,
        "intersection_backend": intersection_backend,
        "wall_seconds": time.perf_counter() - started,
    }


def run_disc_cup(
    config: DiscCupConfig,
    *,
    get_arch: Callable[..., torch.nn.Module] | None = None,
) -> dict[str, Any]:
    """Run strict M2 disc/cup inference and optional downstream routing."""

    total_started = time.perf_counter()
    device = resolve_device(config.device)
    discovery_started = time.perf_counter()
    entries = discover_legacy_images(
        config.input_dir
        or (config.data_root / "Results" / "M1" / "Good_quality")
    )
    discovery_seconds = time.perf_counter() - discovery_started
    progress = ProgressReporter("M2", config.progress)
    paths = DiscCupOutputPaths.from_data_root(config.data_root)
    paths.create()
    loader = ThreadedBatchLoader(
        entries,
        batch_size=config.batch_size,
        threads=config.preprocess_threads,
        image_size=config.image_size,
        lookahead_batches=config.loader_lookahead_batches,
    )
    loader.start()

    cuda_index = device.index if device.index is not None else 0
    if device.type == "cuda":
        # Required by PyTorch 2.7 on Windows in a fresh process.
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(cuda_index)
    model_load_started = time.perf_counter()
    models: list[torch.nn.Module] = []
    inference_seconds = 0.0
    batch_count = 0
    writer = ThreadedPredictionWriter(paths, threads=config.postprocess_threads)
    try:
        with progress.task(
            "disc-cup-model-load", total=len(config.checkpoint_ids), unit="models"
        ) as model_progress:
            models = load_ensemble_models(
                config,
                device,
                get_arch=get_arch,
                progress=model_progress,
            )
        _sync_device(device)
    except BaseException:
        # Preprocessing starts before checkpoint I/O to preserve the existing
        # overlap, so a load error must still release both background pools.
        loader.close()
        writer.close()
        raise
    model_load_seconds = time.perf_counter() - model_load_started
    with progress.task(
        "disc-cup-inference-write", total=len(entries), unit="images"
    ) as inference_progress:
        try:
            for batch in loader:
                inference_started = time.perf_counter()
                decoded, uncertainty = ensemble_predict_legacy(
                    models,
                    batch.images,
                    device,
                    aggregation=config.aggregation,
                )
                _sync_device(device)
                inference_seconds += time.perf_counter() - inference_started

                # Submission performs one packed D2H copy. The image writes
                # continue in CPU workers while the next batch runs.
                writer.submit(decoded, uncertainty, batch)
                batch_count += 1
                inference_progress.update(len(batch.names))
                del decoded, uncertainty
        finally:
            loader.close()
            # optic_disc_centre consumes these files, so every future must finish
            # (and propagate any error) before model release and zone processing.
            writer.close()

    peak_allocated: int | None = None
    peak_reserved: int | None = None
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(cuda_index))
        peak_reserved = int(torch.cuda.max_memory_reserved(cuda_index))

    # Release eight WNets before the CPU-only zone pass or another M2 branch.
    models.clear()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        empty_cache = getattr(torch.mps, "empty_cache", None)
        if empty_cache is not None:
            empty_cache()

    postprocess: dict[str, Any] | None = None
    if config.run_optic_postprocess:
        with progress.task(
            "disc-cup-routing", total=len(entries), unit="images"
        ) as routing_progress:
            postprocess = optic_disc_centre(
                config.data_root,
                result_path=paths.resized,
                intersection_backend=config.intersection_backend,
                progress=routing_progress,
            )

    return {
        "module": "M2_disc_cup",
        "device": str(device),
        "images": len(entries),
        "batches": batch_count,
        "batch_size": config.batch_size,
        "preprocess_threads": config.preprocess_threads,
        "postprocess_threads": config.postprocess_threads,
        "loader": "thread",
        "loader_lookahead_batches": config.loader_lookahead_batches,
        "checkpoint_ids": list(config.checkpoint_ids),
        "aggregation": config.aggregation,
        "timing_seconds": {
            "discovery": discovery_seconds,
            "model_load": model_load_seconds,
            "preprocess_wait": loader.wait_seconds,
            "inference": inference_seconds,
            "output_encoding": writer.submit_seconds + writer.wait_seconds,
            "output_submit": writer.submit_seconds,
            "output_wait": writer.wait_seconds,
            "output_cpu_sum": writer.cpu_seconds,
            "output_wall_span": writer.wall_span_seconds,
            "optic_postprocess": None
            if postprocess is None
            else postprocess["wall_seconds"],
            "total": time.perf_counter() - total_started,
        },
        "peak_memory_bytes": {
            "cuda_allocated": peak_allocated,
            "cuda_reserved": peak_reserved,
        },
        "postprocess": postprocess,
    }
