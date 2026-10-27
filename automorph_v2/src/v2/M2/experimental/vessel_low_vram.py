"""Optional strict low-VRAM engine for the M2 binary-vessel ensemble.

This module changes only model residency.  It keeps the legacy checkpoint
order, FP32 tensors, batches, ensemble arithmetic, quantization, and output
writers from :mod:`v2.M2.vessel`.

The regular strict engine keeps all ten Segmenters on the accelerator and
runs the ensemble batch-by-batch.  This engine instead:

1. caches the strict preprocessed batches on CPU;
2. visits checkpoints in the fixed legacy order and keeps one model resident;
3. caches each model's sigmoid probabilities on CPU; and
4. reconstructs each batch on the target device with the original
   left-associated mean and variance operation order.

The tradeoff is substantially more host memory and repeated host/device
transfers in exchange for a much smaller accelerator-resident model set.
"""

from __future__ import annotations

import gc
import importlib.util
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

try:  # Normal package import used by the future orchestrator integration.
    from .. import vessel as _vessel
except ImportError:  # Standalone loading keeps the CPU test isolated.
    _VESSEL_PATH = Path(__file__).resolve().parents[1] / "vessel.py"
    _VESSEL_MODULE_NAME = "_automorph_v2_m2_vessel_for_low_vram"
    _VESSEL_SPEC = importlib.util.spec_from_file_location(
        _VESSEL_MODULE_NAME,
        _VESSEL_PATH,
    )
    if _VESSEL_SPEC is None or _VESSEL_SPEC.loader is None:
        raise ImportError(f"Could not load strict vessel module: {_VESSEL_PATH}")
    _vessel = importlib.util.module_from_spec(_VESSEL_SPEC)
    sys.modules[_VESSEL_MODULE_NAME] = _vessel
    _VESSEL_SPEC.loader.exec_module(_vessel)

try:
    from ...progress import ProgressReporter, ProgressTask
except ImportError:  # standalone loading from the direct stage runner
    _V2_ROOT = str(Path(__file__).resolve().parents[2])
    if _V2_ROOT not in sys.path:
        sys.path.insert(0, _V2_ROOT)
    from progress import ProgressReporter, ProgressTask


VesselConfig = _vessel.VesselConfig
ModelBuilder = Callable[[Path, int], torch.nn.Module]
ModelEventHook = Callable[[str, int, Path, int], None]


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _validate_cpu_batches(
    batches: Sequence[_vessel.VesselBatch],
) -> None:
    if not batches:
        raise ValueError("At least one preprocessed vessel batch is required")
    for batch_index, batch in enumerate(batches):
        images = batch.images
        if images.device.type != "cpu":
            raise ValueError(
                f"Preprocessed batch {batch_index} must be cached on CPU"
            )
        if images.dtype != torch.float32:
            raise ValueError(
                f"Preprocessed batch {batch_index} must be FP32; got {images.dtype}"
            )
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "Expected strict BCHW RGB input, got "
                f"batch {batch_index} shape {tuple(images.shape)}"
            )
        count = int(images.shape[0])
        if not (count == len(batch.names) == len(batch.widths) == len(batch.heights)):
            raise ValueError(f"Batch {batch_index} metadata length mismatch")


def _validate_probability_cache(
    probabilities: Sequence[torch.Tensor],
) -> None:
    if len(probabilities) != len(_vessel.STRICT_SEEDS):
        raise ValueError(
            "Strict low-VRAM vessel ensemble requires "
            f"{len(_vessel.STRICT_SEEDS)} cached predictions"
        )
    expected_shape = tuple(probabilities[0].shape)
    for model_index, probability in enumerate(probabilities):
        if probability.device.type != "cpu":
            raise ValueError(
                f"Cached probability {model_index} must reside on CPU"
            )
        if probability.dtype != torch.float32:
            raise ValueError(
                f"Cached probability {model_index} must be FP32; "
                f"got {probability.dtype}"
            )
        if tuple(probability.shape) != expected_shape:
            raise ValueError(
                "Cached probability shapes differ: "
                f"{expected_shape} versus {tuple(probability.shape)}"
            )


def strict_statistics_from_cpu_cache(
    probabilities: Sequence[torch.Tensor],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the legacy ensemble arithmetic while streaming CPU tensors.

    Only the storage location changes.  The expression tree is deliberately
    the same as ``vessel.strict_ensemble_statistics``: predictions 0 and 1
    seed the sum, predictions 2 through 9 are added left-to-right, and the
    variance uses the same two-term seed followed by left-to-right additions.
    No stacked reduction, online variance, or algebraic rewrite is used.
    """

    _validate_probability_cache(probabilities)
    target = torch.device(device)

    probability_0 = probabilities[0].to(
        device=target,
        dtype=torch.float32,
    )
    probability_1 = probabilities[1].to(
        device=target,
        dtype=torch.float32,
    )
    mean = probability_0 + probability_1
    del probability_0, probability_1
    for probability_cpu in probabilities[2:]:
        probability = probability_cpu.to(
            device=target,
            dtype=torch.float32,
        )
        mean = mean + probability
        del probability
    mean = mean / len(probabilities)

    probability_0 = probabilities[0].to(
        device=target,
        dtype=torch.float32,
    )
    probability_1 = probabilities[1].to(
        device=target,
        dtype=torch.float32,
    )
    variance_sum = torch.square(mean - probability_0) + torch.square(
        mean - probability_1
    )
    del probability_0, probability_1
    for probability_cpu in probabilities[2:]:
        probability = probability_cpu.to(
            device=target,
            dtype=torch.float32,
        )
        variance_sum = variance_sum + torch.square(mean - probability)
        del probability
    uncertainty = torch.sqrt(variance_sum / len(probabilities))
    return mean, uncertainty


def _stream_model_probabilities(
    batches: Sequence[_vessel.VesselBatch],
    *,
    paths: Sequence[Path],
    device: torch.device,
    model_builder: ModelBuilder,
    event_hook: ModelEventHook | None = None,
    progress: ProgressTask | None = None,
) -> tuple[list[list[torch.Tensor]], dict[str, Any]]:
    """Run checkpoint-major inference and return batch-major CPU predictions.

    ``model_builder`` must return a CPU model loaded from the supplied
    checkpoint.  It is injectable solely so CPU fake-model tests can exercise
    the complete residency/order logic without loading real checkpoints.
    """

    _validate_cpu_batches(batches)
    if len(paths) != len(_vessel.STRICT_SEEDS):
        raise ValueError(
            f"Expected {len(_vessel.STRICT_SEEDS)} checkpoints; got {len(paths)}"
        )

    cache: list[list[torch.Tensor]] = [[] for _ in batches]
    active_device_models = 0
    max_active_device_models = 0
    probability_cache_bytes = 0
    load_transfer_seconds = 0.0
    inference_cache_seconds = 0.0
    per_model_wall_seconds: list[float] = []

    for model_index, raw_path in enumerate(paths):
        checkpoint = Path(raw_path)
        model_started = time.perf_counter()
        model: torch.nn.Module | None = None
        resident = False
        try:
            load_started = time.perf_counter()
            model = model_builder(checkpoint, model_index)
            model.eval()
            model.to(device=device, dtype=torch.float32)
            _vessel._synchronize(device)
            load_transfer_seconds += time.perf_counter() - load_started

            active_device_models += 1
            resident = True
            max_active_device_models = max(
                max_active_device_models,
                active_device_models,
            )
            if active_device_models != 1:
                raise RuntimeError(
                    "Low-VRAM invariant violated: more than one model resident"
                )
            if event_hook is not None:
                event_hook(
                    "resident",
                    model_index,
                    checkpoint,
                    active_device_models,
                )

            inference_started = time.perf_counter()
            with torch.inference_mode():
                for batch_index, batch in enumerate(batches):
                    images = batch.images.to(
                        device=device,
                        dtype=torch.float32,
                    )
                    logits = model(images)
                    if not isinstance(logits, torch.Tensor):
                        raise TypeError(
                            "Legacy Segmenter must return one logits tensor"
                        )
                    if logits.dtype != torch.float32:
                        raise RuntimeError(
                            "Strict low-VRAM inference requires FP32 logits; "
                            f"got {logits.dtype}"
                        )
                    if logits.ndim != 4 or logits.shape[0] != images.shape[0]:
                        raise RuntimeError(
                            "Unexpected Segmenter output shape "
                            f"{tuple(logits.shape)} for input {tuple(images.shape)}"
                        )

                    probability = torch.sigmoid(logits)
                    probability_cpu = probability.to(
                        device="cpu",
                        dtype=torch.float32,
                    )
                    if probability_cpu.device.type != "cpu":
                        raise RuntimeError("Probability D2H transfer failed")
                    cache[batch_index].append(probability_cpu)
                    probability_cache_bytes += _tensor_bytes(probability_cpu)
                    del images, logits, probability, probability_cpu
            _vessel._synchronize(device)
            inference_cache_seconds += time.perf_counter() - inference_started
        finally:
            if model is not None:
                # All D2H probability copies are blocking and inference
                # tensors were deleted above. Synchronize, drop the caller's
                # sole model reference, collect it, and only then release this
                # process's unused CUDA cache. The next checkpoint cannot be
                # constructed before this sequence finishes.
                _vessel._synchronize(device)
                model = None
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if resident:
                active_device_models -= 1
                if event_hook is not None:
                    event_hook(
                        "released",
                        model_index,
                        checkpoint,
                        active_device_models,
                    )
            per_model_wall_seconds.append(time.perf_counter() - model_started)
        if progress is not None:
            progress.update()

    if active_device_models != 0:
        raise RuntimeError("Low-VRAM model residency counter did not return to zero")
    for probabilities in cache:
        _validate_probability_cache(probabilities)

    metrics: dict[str, Any] = {
        "checkpoint_order": [str(Path(path)) for path in paths],
        "batch_sizes": [int(batch.images.shape[0]) for batch in batches],
        "models_streamed": len(paths),
        "max_simultaneous_device_models": max_active_device_models,
        "probability_cache_cpu_bytes": probability_cache_bytes,
        "model_load_and_h2d_s": load_transfer_seconds,
        "model_inference_and_probability_d2h_s": inference_cache_seconds,
        "per_model_wall_s": per_model_wall_seconds,
    }
    return cache, metrics


def _production_model_builder(
    config: VesselConfig,
) -> ModelBuilder:
    segmenter = _vessel._import_legacy_segmenter(config.legacy_dir)

    def build(checkpoint: Path, model_index: int) -> torch.nn.Module:
        del model_index
        model = segmenter(
            input_channels=3,
            n_filters=32,
            n_classes=1,
            bilinear=False,
        )
        state = _vessel._load_checkpoint(checkpoint)
        model.load_state_dict(state)
        del state
        return model

    return build


def run_vessel_low_vram(
    config: VesselConfig,
    *,
    _model_builder: ModelBuilder | None = None,
    _event_hook: ModelEventHook | None = None,
) -> Mapping[str, Any]:
    """Run the strict vessel ensemble with one device-resident model.

    The public integration contract intentionally accepts the existing
    ``VesselConfig``.  Underscored injection points are for CPU fake-model
    tests; production callers should pass only ``config``.
    """

    total_started = time.perf_counter()
    _vessel._validate_config(config)

    discovery_started = time.perf_counter()
    image_paths = _vessel.discover_images(config.input_dir)
    outputs = _vessel.OutputPaths.create(config.output_dir)
    discovery_seconds = time.perf_counter() - discovery_started
    progress = ProgressReporter("M2", config.progress)

    device = _vessel.resolve_device(config.device)
    cuda_index = device.index if device.index is not None else 0
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(cuda_index)

    preprocessing_started = time.perf_counter()
    loader = _vessel.OrderedThreadBatchLoader(
        image_paths,
        batch_size=config.batch_size,
        threads=config.preprocess_threads,
        image_size=config.image_size,
        threshold=config.threshold,
    )
    batches: list[_vessel.VesselBatch] = []
    with progress.task(
        "vessel-low-vram-preprocess", total=len(image_paths), unit="images"
    ) as preprocess_progress:
        try:
            for batch in loader:
                batches.append(batch)
                preprocess_progress.update(len(batch.names))
        finally:
            loader.close()
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    _validate_cpu_batches(batches)

    paths = _vessel.checkpoint_paths(config)
    missing = [path for path in paths if not path.is_file()]
    if missing and _model_builder is None:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing M2 vessel checkpoints:\n{formatted}")
    builder = _model_builder or _production_model_builder(config)

    with progress.task(
        "vessel-low-vram-model-inference", total=len(paths), unit="models"
    ) as model_progress:
        cache, stream_metrics = _stream_model_probabilities(
            batches,
            paths=paths,
            device=device,
            model_builder=builder,
            event_hook=_event_hook,
            progress=model_progress,
        )

    post_executor = ThreadPoolExecutor(
        max_workers=config.postprocess_threads,
        thread_name_prefix="m2-vessel-low-vram-postprocess",
    )
    pending: deque[Future[Mapping[str, float]]] = deque()
    max_pending = max(config.postprocess_threads * 2, config.batch_size)
    postprocess_sum = 0.0
    morphology_sum = 0.0
    aggregation_quantize_transfer_seconds = 0.0
    output_progress = progress.task(
        "vessel-low-vram-aggregate-write", total=len(image_paths), unit="images"
    )

    def collect_one() -> None:
        nonlocal postprocess_sum, morphology_sum
        result = pending.popleft().result()
        postprocess_sum += float(result["postprocess_s"])
        morphology_sum += float(result["morphology_s"])
        output_progress.update()

    try:
        for batch_index, batch in enumerate(batches):
            aggregation_started = time.perf_counter()
            probabilities = cache[batch_index]
            with torch.inference_mode():
                mean, uncertainty = strict_statistics_from_cpu_cache(
                    probabilities,
                    device,
                )
                for sample_index, name in enumerate(batch.names):
                    mean_rgb, binary_rgb, uncertainty_rgb = (
                        _vessel._quantize_sample_outputs(
                            mean[sample_index, ...],
                            uncertainty[sample_index, ...],
                        )
                    )
                    pending.append(
                        post_executor.submit(
                            _vessel._write_sample_outputs,
                            outputs,
                            name=name,
                            width=batch.widths[sample_index],
                            height=batch.heights[sample_index],
                            mean_rgb=mean_rgb,
                            binary_rgb=binary_rgb,
                            uncertainty_rgb=uncertainty_rgb,
                        )
                    )
                    if len(pending) >= max_pending:
                        collect_one()
            _vessel._synchronize(device)
            aggregation_quantize_transfer_seconds += (
                time.perf_counter() - aggregation_started
            )
            cache[batch_index] = []
            del probabilities, mean, uncertainty

        postprocess_tail_started = time.perf_counter()
        while pending:
            collect_one()
        postprocess_tail_seconds = time.perf_counter() - postprocess_tail_started
    except BaseException:
        output_progress.close("error")
        raise
    else:
        output_progress.close()
    finally:
        post_executor.shutdown(wait=True)

    _vessel._synchronize(device)
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(cuda_index))
        peak_reserved = int(torch.cuda.max_memory_reserved(cuda_index))
    else:
        peak_allocated = None
        peak_reserved = None

    preprocessed_cache_bytes = sum(
        _tensor_bytes(batch.images) for batch in batches
    )
    summary: dict[str, Any] = {
        "module": "M2_Vessel_seg",
        "mode": "strict",
        "engine": "low_vram_model_streaming",
        "input_dir": str(Path(config.input_dir).resolve()),
        "output_dir": str(Path(config.output_dir).resolve()),
        "legacy_dir": str(Path(config.legacy_dir).resolve()),
        "device": str(device),
        "images": len(image_paths),
        "batches": len(batches),
        "batch_sizes": stream_metrics["batch_sizes"],
        "batch_size": config.batch_size,
        "image_size": list(config.image_size),
        "threshold": config.threshold,
        "preprocess_threads": config.preprocess_threads,
        "postprocess_threads": config.postprocess_threads,
        "checkpoint_seeds": list(config.seeds),
        "checkpoint_order": stream_metrics["checkpoint_order"],
        "fp32": True,
        "ensemble_reduction": "legacy_left_associated",
        "models_streamed": stream_metrics["models_streamed"],
        "max_simultaneous_device_models": stream_metrics[
            "max_simultaneous_device_models"
        ],
        "preprocessed_cache_cpu_bytes": preprocessed_cache_bytes,
        "probability_cache_cpu_bytes": stream_metrics[
            "probability_cache_cpu_bytes"
        ],
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "timings_s": {
            "discovery_and_output_setup": discovery_seconds,
            "preprocess_and_cpu_cache": preprocessing_seconds,
            "model_load_and_h2d": stream_metrics["model_load_and_h2d_s"],
            "model_inference_and_probability_d2h": stream_metrics[
                "model_inference_and_probability_d2h_s"
            ],
            "per_model_wall": stream_metrics["per_model_wall_s"],
            "aggregation_quantize_transfer": (
                aggregation_quantize_transfer_seconds
            ),
            "postprocess_tail_wait": postprocess_tail_seconds,
            "postprocess_task_sum": postprocess_sum,
            "morphology_task_sum": morphology_sum,
            "total_wall": time.perf_counter() - total_started,
        },
    }
    if config.write_summary:
        _vessel._write_json_atomic(
            Path(config.output_dir) / "run_summary.json",
            summary,
        )
    return summary


__all__ = [
    "VesselConfig",
    "run_vessel_low_vram",
    "strict_statistics_from_cpu_cache",
]
