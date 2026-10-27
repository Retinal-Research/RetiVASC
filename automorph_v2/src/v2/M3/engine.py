"""Unified, ordered, CPU-only AutoMorph M3 feature engine."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import configparser
from dataclasses import asdict, dataclass
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd

from .contract import (
    LEGACY_GROUP_ORDER,
    VESSEL_TYPES,
    MeasurementGroup,
    VesselType,
    discover_pairs,
    preflight,
)
from .merge import _write_csv_atomic, merge_feature_tables
from .metrics import MetricResult, evaluate_paths

try:
    from ..progress import ProgressReporter, ProgressTask
except ImportError:  # imported as ``M3.engine`` by the direct stage runner
    _V2_ROOT = str(Path(__file__).resolve().parents[1])
    if _V2_ROOT not in sys.path:
        sys.path.insert(0, _V2_ROOT)
    from progress import ProgressReporter, ProgressTask


RELEASE_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = RELEASE_ROOT / "runtime"
DEFAULT_CONFIGURATION = (
    RUNTIME_ROOT
    / "M3_feature_whole_pic"
    / "retipy"
    / "resources"
    / "retipy.config"
)

BASE_NAMES = (
    "Fractal_dimension",
    "Vessel_density",
    "Average_width",
    "Distance_tortuosity",
    "Squared_curvature_tortuosity",
    "Tortuosity_density",
)


@dataclass(frozen=True)
class MetricSettings:
    pixels_per_window: int = 15
    sampling_size: int = 6
    r2_threshold: float = 0.96


@dataclass(frozen=True)
class M3Config:
    data_root: Path
    workers: int = 4
    configuration: Path = DEFAULT_CONFIGURATION
    allow_existing_output: bool = False
    # Legacy drivers convert a per-image metric exception to an all--1 row and
    # continue.  Preserve that public result contract while recording every
    # traceback in run_errors.json.  Validation jobs may opt into fail-fast.
    allow_metric_failures: bool = True
    start_method: str = "spawn"
    quiet: bool = False
    progress: str = "off"


@dataclass(frozen=True)
class EvaluationTask:
    index: int
    centre: str
    region: str
    vessel_type: VesselType
    name: str
    skeleton_path: str
    process_path: str
    resolution: object
    settings: MetricSettings

    @property
    def is_zone(self) -> bool:
        return self.region != "whole"


@dataclass(frozen=True)
class EvaluationResult:
    index: int
    centre: str
    region: str
    vessel_type: VesselType
    name: str
    metrics: MetricResult
    wall_s: float
    error: str | None = None


def load_settings(configuration: str | Path) -> MetricSettings:
    path = Path(configuration).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"M3 configuration does not exist: {path}")
    parser = configparser.ConfigParser()
    read = parser.read(path)
    if not read or "General" not in parser:
        raise ValueError(f"M3 configuration has no [General] section: {path}")
    section = parser["General"]
    return MetricSettings(
        pixels_per_window=section.getint("PixelsPerWindow"),
        sampling_size=section.getint("SamplingSize"),
        r2_threshold=section.getfloat("R2Threshold"),
    )


def _worker_init() -> None:
    cv2.setNumThreads(1)


def _sentinel_metrics() -> MetricResult:
    return MetricResult(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1)


def _evaluate_task(task: EvaluationTask) -> EvaluationResult:
    started = time.perf_counter()
    try:
        metrics = evaluate_paths(
            task.skeleton_path,
            task.process_path,
            task.resolution,
            is_zone=task.is_zone,
            vessel_type=task.vessel_type,
            min_pixels_per_vessel=task.settings.pixels_per_window,
            sampling_size=task.settings.sampling_size,
            r2_threshold=task.settings.r2_threshold,
        )
        error = None
    except Exception:
        metrics = _sentinel_metrics()
        error = traceback.format_exc()
    return EvaluationResult(
        index=task.index,
        centre=task.centre,
        region=task.region,
        vessel_type=task.vessel_type,
        name=task.name,
        metrics=metrics,
        wall_s=time.perf_counter() - started,
        error=error,
    )


def _load_resolutions(data_root: Path) -> dict[str, object]:
    frame = pd.read_csv(data_root / "Results" / "M0" / "crop_info.csv")
    names = frame["Name"].astype(str).to_numpy()
    values = frame["Scale_resolution"].to_numpy()
    # Retipy obtains one value from ``Series.values[0]``.  Preserve the NumPy
    # scalar type as well as its bits for the strict arithmetic path.
    return dict(zip(names.tolist(), values, strict=True))


def _build_tasks(
    data_root: Path,
    settings: MetricSettings,
) -> list[EvaluationTask]:
    resolutions = _load_resolutions(data_root)
    tasks: list[EvaluationTask] = []
    index = 0
    for group in LEGACY_GROUP_ORDER:
        for vessel_type in VESSEL_TYPES:
            for pair in discover_pairs(data_root, group, vessel_type):
                tasks.append(
                    EvaluationTask(
                        index=index,
                        centre=group.centre,
                        region=group.region,
                        vessel_type=vessel_type,
                        name=pair.name,
                        skeleton_path=str(pair.skeleton),
                        process_path=str(pair.process),
                        resolution=resolutions[pair.name],
                        settings=settings,
                    )
                )
                index += 1
    return tasks


_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _evaluate_all(
    tasks: list[EvaluationTask],
    *,
    workers: int,
    start_method: str,
    quiet: bool,
    progress: ProgressTask | None = None,
) -> list[EvaluationResult]:
    if workers < 1:
        raise ValueError("workers must be at least one")
    if workers == 1:
        _worker_init()
        results = []
        for task in tasks:
            results.append(_evaluate_task(task))
            if progress is not None:
                progress.update()
        return results

    previous = {name: os.environ.get(name) for name in _THREAD_ENVIRONMENT}
    for name in _THREAD_ENVIRONMENT:
        os.environ[name] = "1"
    try:
        context = multiprocessing.get_context(start_method)
        ordered: list[EvaluationResult | None] = [None] * len(tasks)
        completed = 0
        next_report = max(1, len(tasks) // 20)
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_worker_init,
        ) as executor:
            future_to_index = {
                executor.submit(_evaluate_task, task): task.index for task in tasks
            }
            for future in as_completed(future_to_index):
                result = future.result()
                ordered[result.index] = result
                completed += 1
                if progress is not None:
                    # as_completed is consumed by the parent process; workers
                    # never write to the terminal or unified run log.
                    progress.update()
                elif not quiet and (
                    completed == len(tasks) or completed % next_report == 0
                ):
                    print(f"M3 measurements: {completed}/{len(tasks)}", flush=True)
        if any(result is None for result in ordered):
            raise RuntimeError("M3 worker pool returned an incomplete result set")
        return [result for result in ordered if result is not None]
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _result_index(
    results: list[EvaluationResult],
) -> dict[tuple[str, str, str, str], EvaluationResult]:
    return {
        (result.centre, result.region, result.vessel_type, result.name): result
        for result in results
    }


def _metric_columns(vessel_type: VesselType) -> list[str]:
    prefix = "" if vessel_type == "binary" else f"{vessel_type.title()}_"
    return [prefix + name for name in BASE_NAMES]


def _metric_values(result: EvaluationResult) -> list[object]:
    values = list(result.metrics.base_values)
    if result.region != "whole" and result.vessel_type == "artery":
        values.extend([result.metrics.crae_hubbard, result.metrics.crae_knudtson])
    elif result.region != "whole" and result.vessel_type == "vein":
        values.extend([result.metrics.crve_hubbard, result.metrics.crve_knudtson])
    return values


def _measurement_frame(
    data_root: Path,
    group: MeasurementGroup,
    indexed: dict[tuple[str, str, str, str], EvaluationResult],
) -> pd.DataFrame:
    disc_path = (
        data_root
        / "Results"
        / "M3"
        / group.centre_directory
        / "Disc_cup_results.csv"
    )
    disc_frame = pd.read_csv(disc_path).astype({"Name": "object"})
    frames: dict[VesselType, pd.DataFrame] = {}
    for vessel_type in VESSEL_TYPES:
        pairs = discover_pairs(data_root, group, vessel_type)
        columns = _metric_columns(vessel_type)
        if group.is_zone and vessel_type == "artery":
            columns.extend(["CRAE_Hubbard", "CRAE_Knudtson"])
        elif group.is_zone and vessel_type == "vein":
            columns.extend(["CRVE_Hubbard", "CRVE_Knudtson"])
        rows = []
        for pair in pairs:
            result = indexed[(group.centre, group.region, vessel_type, pair.name)]
            rows.append([pair.name, *_metric_values(result)])
        frames[vessel_type] = pd.DataFrame(rows, columns=["Name", *columns]).astype(
            {"Name": "object"}
        )

    disc_binary = pd.merge(disc_frame, frames["binary"], how="outer", on=["Name"])
    artery_vein = pd.merge(
        frames["artery"], frames["vein"], how="outer", on=["Name"]
    )
    combined = pd.merge(disc_binary, artery_vein, how="outer", on=["Name"])
    if group.is_zone:
        combined["AVR_Hubbard"] = combined["CRAE_Hubbard"] / combined["CRVE_Hubbard"]
        combined["AVR_Knudtson"] = (
            combined["CRAE_Knudtson"] / combined["CRVE_Knudtson"]
        )
    return combined


def _existing_outputs(data_root: Path) -> list[Path]:
    root = data_root / "Results" / "M3"
    paths = [root / "Disc_Features.csv", root / "Macular_Features.csv"]
    for group in LEGACY_GROUP_ORDER:
        paths.append(root / group.centre_directory / group.output_name)
    return [path for path in paths if path.exists()]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json_atomic(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_m3(config: M3Config) -> dict[str, Any]:
    """Run all six M3 feature groups and assemble the public CSV contract."""

    data_root = Path(config.data_root).expanduser().resolve()
    existing = _existing_outputs(data_root)
    if existing and not config.allow_existing_output:
        rendered = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "M3 output already exists; use a fresh v2 data root or pass "
            f"--allow-existing-output:\n{rendered}"
        )
    progress_mode = "off" if config.quiet else config.progress
    progress = ProgressReporter("M3", progress_mode)
    with progress.task("preflight", total=3, unit="steps") as preflight_progress:
        structure = preflight(data_root)
        preflight_progress.update()
        settings = load_settings(config.configuration)
        preflight_progress.update()
        tasks = _build_tasks(data_root, settings)
        preflight_progress.update()

    started = time.perf_counter()
    with progress.task(
        "measurements", total=len(tasks), unit="evaluations"
    ) as measurement_progress:
        results = _evaluate_all(
            tasks,
            workers=config.workers,
            start_method=config.start_method,
            quiet=config.quiet,
            progress=measurement_progress,
        )
    measurement_wall_s = time.perf_counter() - started
    errors = [result for result in results if result.error]
    m3_root = data_root / "Results" / "M3"
    if errors:
        error_payload = [
            {
                "centre": result.centre,
                "region": result.region,
                "vessel_type": result.vessel_type,
                "name": result.name,
                "error": result.error,
            }
            for result in errors
        ]
        _write_json_atomic(error_payload, m3_root / "run_errors.json")
        if not config.allow_metric_failures:
            raise RuntimeError(
                f"M3 rejected {len(errors)} metric failures; see "
                f"{m3_root / 'run_errors.json'}"
            )
    else:
        # Never leave a stale failure report beside a successful rerun.
        _write_json_atomic([], m3_root / "run_errors.json")

    indexed = _result_index(results)
    intermediate_outputs: list[str] = []
    group_wall_s: dict[str, float] = {}
    with progress.task(
        "group-tables", total=len(LEGACY_GROUP_ORDER), unit="tables"
    ) as table_progress:
        for group in LEGACY_GROUP_ORDER:
            frame = _measurement_frame(data_root, group, indexed)
            destination = m3_root / group.centre_directory / group.output_name
            _write_csv_atomic(frame, destination)
            intermediate_outputs.append(str(destination))
            group_wall_s[f"{group.centre}:{group.region}"] = sum(
                result.wall_s
                for result in results
                if result.centre == group.centre and result.region == group.region
            )
            table_progress.update()

    with progress.task("final-merge", total=1, unit="merge") as merge_progress:
        final_outputs = merge_feature_tables(m3_root)
        merge_progress.update()
    total_wall_s = time.perf_counter() - started
    summary: dict[str, Any] = {
        "module": "M3",
        "contract": "legacy-73-column-strict-cpu",
        "data_root": data_root,
        "configuration": Path(config.configuration).expanduser().resolve(),
        "workers": config.workers,
        "start_method": config.start_method,
        "device": "cpu",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "opencv": cv2.__version__,
        "preflight": structure,
        "settings": asdict(settings),
        "measurement_wall_s": measurement_wall_s,
        "total_wall_s": total_wall_s,
        "worker_task_sum_s": sum(result.wall_s for result in results),
        "group_task_sum_s": group_wall_s,
        "metric_failures": len(errors),
        "intermediate_outputs": intermediate_outputs,
        "final_outputs": {key: str(value) for key, value in final_outputs.items()},
    }
    _write_json_atomic(summary, m3_root / "run_summary.json")
    return summary
