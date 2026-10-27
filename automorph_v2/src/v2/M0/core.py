"""Parallel M0 preprocessing with an optional CUDA/MPS array backend.

The geometry detector intentionally follows ``M0_Preprocess/fundus_prep.py``
operation-for-operation.  Keeping that portion on OpenCV/NumPy makes the crop
metadata and decoded PNG pixels compatible with the legacy implementation.
The optional Torch backend accelerates the large mask/crop/pad operation while
CPU threads overlap image decoding, geometry work, and PNG encoding.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

try:
    from ..progress import ProgressReporter
except ImportError:  # imported as ``M0.core`` by the direct stage runner
    import sys

    _V2_ROOT = str(Path(__file__).resolve().parents[1])
    if _V2_ROOT not in sys.path:
        sys.path.insert(0, _V2_ROOT)
    from progress import ProgressReporter


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CROP_FIELDS = ["Name", "centre_w", "centre_h", "radius", "Scale", "Scale_resolution"]
METRIC_FIELDS = [
    "input_name", "output_name", "status", "error", "resolution",
    "centre_w", "centre_h", "radius", "Scale", "Scale_resolution",
    "decode_s", "geometry_s", "array_s", "encode_s", "total_s",
    "input_width", "input_height", "output_width", "output_height",
]


@dataclass(frozen=True)
class RunConfig:
    input_dir: Path
    output_dir: Path
    resolution_csv: Path
    device: str = "cpu"
    workers: int = 0
    png_compression: int = 1
    force: bool = False
    limit: int = 0
    progress: str = "off"


@dataclass
class ImageResult:
    input_name: str
    output_name: str
    status: str
    error: str = ""
    resolution: float = math.nan
    centre_w: int | None = None
    centre_h: int | None = None
    radius: int | None = None
    Scale: float | None = None
    Scale_resolution: float | None = None
    decode_s: float = 0.0
    geometry_s: float = 0.0
    array_s: float = 0.0
    encode_s: float = 0.0
    total_s: float = 0.0
    input_width: int | None = None
    input_height: int | None = None
    output_width: int | None = None
    output_height: int | None = None

    def crop_row(self) -> dict[str, Any]:
        return {
            'Name': self.output_name,
            'centre_w': self.centre_w,
            'centre_h': self.centre_h,
            'radius': self.radius,
            'Scale': self.Scale,
            'Scale_resolution': self.Scale_resolution,
        }


def legacy_output_name(input_name: str) -> str:
    """Preserve the original first-dot filename behavior."""
    return input_name.split(".")[0] + ".png"


def read_rgb(path: Path) -> np.ndarray:
    """Read via imdecode so non-ASCII paths also work reliably on Windows."""
    payload = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    bgr = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _get_mask_bz(gray_img: np.ndarray) -> np.ndarray:
    threshold = float(np.mean(gray_img)) / 3.0 - 5.0
    _, mask = cv2.threshold(gray_img, max(5.0, threshold), 1, cv2.THRESH_BINARY)
    flood_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
    new_mask = (1 - mask).astype(np.uint8)
    _, new_mask, _, _ = cv2.floodFill(
        new_mask, flood_mask, (0, 0), 0, cv2.FLOODFILL_MASK_ONLY
    )
    _, new_mask, _, _ = cv2.floodFill(
        new_mask,
        flood_mask,
        (new_mask.shape[1] - 1, new_mask.shape[0] - 1),
        0,
        cv2.FLOODFILL_MASK_ONLY,
    )
    mask = mask + new_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    mask = cv2.erode(mask, kernel)
    return cv2.dilate(mask, kernel)


def legacy_geometry(image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], int, int, int]:
    """Return circle mask, bbox, legacy centre fields, and radius."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    rough_mask = _get_mask_bz(normalized)

    row_sums = rough_mask.sum(axis=1)
    col_sums = rough_mask.sum(axis=0)
    row_candidates = np.where(row_sums > row_sums.max() * 0.95)[0]
    col_candidates = np.where(col_sums > col_sums.max() * 0.95)[0]
    if not len(row_candidates) or not len(col_candidates):
        raise ValueError("fundus centre could not be estimated")
    center_row = float(row_candidates.mean())
    center_col = float(col_candidates.mean())

    ksize = max(rough_mask.shape[1] // 400 * 2 + 1, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    edge = cv2.morphologyEx(rough_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    edge_rows, edge_cols = np.where(edge > 0)
    if not len(edge_rows):
        raise ValueError("fundus radius could not be estimated")
    distances = np.sqrt((edge_rows - center_row) ** 2 + (edge_cols - center_col) ** 2)
    counts = np.bincount(np.ceil(distances).astype(int))
    radius_candidates = np.where(counts > counts.max() * 0.995)[0]
    if not len(radius_candidates):
        raise ValueError("fundus radius histogram is empty")
    radius = int(radius_candidates.max())

    start_h = max(0, int(center_row - radius))
    start_w = max(0, int(center_col - radius))
    bbox = (
        start_h,
        start_w,
        min(height - start_h, 2 * radius),
        min(width - start_w, 2 * radius),
    )
    circle = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(circle, (int(center_col), int(center_row)), radius, 1, -1)

    # The upstream CSV labels these two values in this order even though the
    # first is the row coordinate.  Preserve it for downstream compatibility.
    centre_w = int(center_row)
    centre_h = int(center_col)
    return circle, bbox, centre_w, centre_h, radius


def _mask_crop_pad_cpu(
    image_rgb: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]
) -> np.ndarray:
    masked = image_rgb.copy()
    masked[mask <= 0, ...] = 0
    start_h, start_w, crop_h, crop_w = bbox
    crop = masked[start_h : start_h + crop_h, start_w : start_w + crop_w, ...]
    side = max(crop_h, crop_w)
    output = np.zeros((side, side, image_rgb.shape[2]), dtype=image_rgb.dtype)
    top = int(side / 2 - crop_h / 2)
    left = int(side / 2 - crop_w / 2)
    output[top : top + crop_h, left : left + crop_w, ...] = crop
    return output


class ArrayBackend:
    """Exact CPU backend or Torch CUDA/MPS backend for large-array operations."""

    def __init__(self, requested: str):
        requested = requested.lower()
        if requested not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        self.requested = requested
        self.torch = None
        self.device = "cpu"
        self._lock: threading.Lock | None = None

        if requested == "cpu":
            return
        try:
            import torch
        except ImportError:
            if requested != "auto":
                raise RuntimeError(f"device={requested} requires PyTorch")
            return

        resolved = requested
        if requested == "auto":
            if torch.cuda.is_available():
                resolved = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                resolved = "mps"
            else:
                resolved = "cpu"
        if resolved == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        if resolved == "mps" and not (
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS requested but the PyTorch MPS backend is unavailable")
        if resolved != "cpu":
            self.torch = torch
            self.device = resolved
            # MPS command submission is kept serialized for stability.  The
            # surrounding decode/geometry/encode work still runs concurrently.
            if resolved == "mps":
                self._lock = threading.Lock()

    @property
    def label(self) -> str:
        return "opencv:cpu" if self.device == "cpu" else f"torch:{self.device}+opencv:cpu"

    def apply(
        self, image_rgb: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> np.ndarray:
        if self.device == "cpu":
            return _mask_crop_pad_cpu(image_rgb, mask, bbox)
        if self._lock is None:
            return self._apply_torch(image_rgb, mask, bbox)
        with self._lock:
            return self._apply_torch(image_rgb, mask, bbox)

    def _apply_torch(
        self, image_rgb: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> np.ndarray:
        assert self.torch is not None
        torch = self.torch
        start_h, start_w, crop_h, crop_w = bbox
        with torch.inference_mode():
            image_t = torch.from_numpy(np.ascontiguousarray(image_rgb)).to(self.device)
            mask_t = torch.from_numpy(np.ascontiguousarray(mask)).to(self.device)
            masked = image_t * mask_t.unsqueeze(-1)
            crop = masked[start_h : start_h + crop_h, start_w : start_w + crop_w, :]
            side = max(crop_h, crop_w)
            top = int(side / 2 - crop_h / 2)
            left = int(side / 2 - crop_w / 2)
            output = torch.zeros((side, side, image_t.shape[2]), dtype=image_t.dtype, device=self.device)
            output[top : top + crop_h, left : left + crop_w, :] = crop
            return output.cpu().numpy()

    def peak_memory_mb(self) -> float:
        if self.device == "cuda" and self.torch is not None:
            return float(self.torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        return 0.0


def _encode_png_atomic(path: Path, image_rgb: np.ndarray, compression: int) -> None:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, int(compression)]
    )
    if not ok:
        raise OSError(f"OpenCV failed to encode {path}")
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("wb") as handle:
        handle.write(encoded.tobytes())
    os.replace(temp_path, path)


def _process_one(
    input_path: Path,
    output_path: Path,
    resolution: float,
    backend: ArrayBackend,
    compression: int,
) -> ImageResult:
    total_start = time.perf_counter()
    result = ImageResult(
        input_name=input_path.name,
        output_name=output_path.name,
        status="failed",
        resolution=resolution,
    )
    try:
        start = time.perf_counter()
        image = read_rgb(input_path)
        result.decode_s = time.perf_counter() - start
        result.input_height, result.input_width = image.shape[:2]

        start = time.perf_counter()
        mask, bbox, result.centre_w, result.centre_h, result.radius = legacy_geometry(image)
        result.geometry_s = time.perf_counter() - start

        start = time.perf_counter()
        output = backend.apply(image, mask, bbox)
        result.array_s = time.perf_counter() - start
        result.output_height, result.output_width = output.shape[:2]

        start = time.perf_counter()
        _encode_png_atomic(output_path, output, compression)
        result.encode_s = time.perf_counter() - start

        result.Scale = result.radius * 2.0 / 912.0
        result.Scale_resolution = resolution * result.Scale * 1000.0
        result.status = "ok"
    except Exception as exc:  # retain the batch while making failures visible
        result.error = f"{type(exc).__name__}: {exc}"
    result.total_s = time.perf_counter() - total_start
    return result


def _read_resolutions(path: Path) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"fundus", "res"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: fundus,res")
        values: dict[str, float] = {}
        for row in reader:
            values[row["fundus"]] = float(row["res"])
    return values


def _read_previous_crop(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return {row["Name"]: row for row in csv.DictReader(handle)}


def _write_csv_atomic(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def _skipped_result(input_name: str, row: dict[str, str], resolution: float) -> ImageResult:
    return ImageResult(
        input_name=input_name,
        output_name=row["Name"],
        status="skipped",
        resolution=resolution,
        centre_w=int(float(row["centre_w"])),
        centre_h=int(float(row["centre_h"])),
        radius=int(float(row["radius"])),
        Scale=float(row["Scale"]),
        Scale_resolution=float(row["Scale_resolution"]),
    )


def run_m0(config: RunConfig) -> dict[str, Any]:
    """Process one M0 batch and return the same summary written to JSON."""
    input_dir = Path(config.input_dir).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve()
    resolution_csv = Path(config.resolution_csv).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    if not 0 <= config.png_compression <= 9:
        raise ValueError("png_compression must be between 0 and 9")

    files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if config.limit > 0:
        files = files[: config.limit]
    output_names = [legacy_output_name(path.name) for path in files]
    if len(set(output_names)) != len(output_names):
        raise ValueError("input filenames collide under legacy first-dot PNG naming")

    resolutions = _read_resolutions(resolution_csv)
    missing = [path.name for path in files if path.name not in resolutions]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"resolution_information.csv is missing {len(missing)} images: {preview}")

    output_dir.mkdir(parents=True, exist_ok=True)
    crop_path = output_dir.parent / "crop_info.csv"
    metric_path = output_dir.parent / "run_metrics.csv"
    summary_path = output_dir.parent / "run_summary.json"
    previous = _read_previous_crop(crop_path)

    worker_count = config.workers or min(8, max(1, os.cpu_count() or 1))
    worker_count = max(1, worker_count)
    # Avoid multiplying OpenCV's own thread pool by our image-level workers.
    cv2.setNumThreads(1 if worker_count > 1 else max(1, cv2.getNumberOfCPUs()))
    backend = ArrayBackend(config.device)

    started = time.perf_counter()
    results: dict[str, ImageResult] = {}
    pending: list[tuple[Path, Path, float]] = []
    for source, output_name in zip(files, output_names):
        output = output_dir / output_name
        resolution = resolutions[source.name]
        if not config.force and output.exists() and output_name in previous:
            results[source.name] = _skipped_result(source.name, previous[output_name], resolution)
        else:
            pending.append((source, output, resolution))

    progress = ProgressReporter("M0", config.progress)
    with progress.task("preprocess", total=len(files), unit="images") as task:
        skipped_count = len(files) - len(pending)
        if skipped_count:
            task.update(skipped_count)
        if pending:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="m0") as pool:
                futures = {
                    pool.submit(
                        _process_one, source, output, resolution, backend, config.png_compression
                    ): source.name
                    for source, output, resolution in pending
                }
                for future in as_completed(futures):
                    item = future.result()
                    results[item.input_name] = item
                    # Futures complete on worker threads, but result collection
                    # and all progress output stay on this owning main thread.
                    task.update()

    ordered = [results[path.name] for path in files]
    successful = [item for item in ordered if item.status in {"ok", "skipped"}]
    _write_csv_atomic(crop_path, CROP_FIELDS, (item.crop_row() for item in successful))
    _write_csv_atomic(metric_path, METRIC_FIELDS, (asdict(item) for item in ordered))

    wall_s = time.perf_counter() - started
    failures = [item for item in ordered if item.status == "failed"]
    summary = {
        "module": "M0_v2",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "image_count": len(ordered),
        "processed_count": sum(item.status == "ok" for item in ordered),
        "skipped_count": sum(item.status == "skipped" for item in ordered),
        "failure_count": len(failures),
        "wall_s": wall_s,
        "throughput_images_s": len(pending) / wall_s if wall_s and pending else 0.0,
        "workers": worker_count,
        "device_requested": config.device,
        "device_resolved": backend.device,
        "backend": backend.label,
        "png_compression": config.png_compression,
        "gpu_peak_allocated_mb": backend.peak_memory_mb(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "failures": [{"name": item.input_name, "error": item.error} for item in failures],
    }
    temp_summary = summary_path.with_name(summary_path.name + ".tmp")
    temp_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temp_summary, summary_path)
    return summary
