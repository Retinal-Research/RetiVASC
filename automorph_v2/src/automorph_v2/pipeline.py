"""Safe orchestration and resumability for the standalone AutoMorph v2 tree."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence, TextIO
from uuid import uuid4

from v2.progress import parse_progress_line


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
PIPELINE_STAGES = ("m0", "m1", "m2", "m3")
MANIFEST_SCHEMA = 1
DEFAULT_RESOLUTION_MM_PER_PIXEL = 0.008
M0_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

M3_MEASUREMENT_OUTPUTS = (
    "Disc_centred/Disc_Zone_B_Measurement.csv",
    "Disc_centred/Disc_Zone_C_Measurement.csv",
    "Disc_centred/Disc_Measurement.csv",
    "Macular_centred/Macular_Zone_B_Measurement.csv",
    "Macular_centred/Macular_Zone_C_Measurement.csv",
    "Macular_centred/Macular_Measurement.csv",
    "Disc_Features.csv",
    "Macular_Features.csv",
    "run_summary.json",
    "run_errors.json",
)


class PipelineError(RuntimeError):
    """Base error displayed by the public orchestration CLI."""


class ExistingOutputError(PipelineError):
    """Raised when fresh or resume safety checks reject existing output."""


class StageFailedError(PipelineError):
    """Raised when a child stage exits unsuccessfully or is incomplete."""


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    data_root: Path
    device: str = "auto"
    threads: int = 4
    m0_device: str = "cpu"
    m0_workers: int | None = None
    m1_threads: int | None = None
    m2_threads: int | None = None
    m3_workers: int | None = None
    png_compression: int = 1
    vessel_engine: str = "resident"
    resolution_mm_per_pixel: float = DEFAULT_RESOLUTION_MM_PER_PIXEL
    require_calibrated_resolution: bool = False
    fail_on_m0_image_error: bool = False
    progress: str = "auto"
    from_stage: str = "m0"
    to_stage: str = "m3"
    mode: str = "fresh"

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root).expanduser().resolve())
        if self.threads < 1:
            raise ValueError("threads must be at least one")
        for name in ("m0_workers", "m1_threads", "m2_threads", "m3_workers"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least one")
        if self.from_stage not in PIPELINE_STAGES or self.to_stage not in PIPELINE_STAGES:
            raise ValueError(f"stage names must be one of {PIPELINE_STAGES}")
        if PIPELINE_STAGES.index(self.from_stage) > PIPELINE_STAGES.index(self.to_stage):
            raise ValueError("from_stage must not come after to_stage")
        if self.mode not in {"fresh", "resume", "overwrite"}:
            raise ValueError("mode must be fresh, resume, or overwrite")
        if not 0 <= self.png_compression <= 9:
            raise ValueError("png_compression must be between zero and nine")
        if self.vessel_engine not in {"resident", "low-vram"}:
            raise ValueError("vessel_engine must be resident or low-vram")
        if (
            not math.isfinite(self.resolution_mm_per_pixel)
            or self.resolution_mm_per_pixel <= 0
        ):
            raise ValueError("resolution_mm_per_pixel must be finite and positive")
        if self.progress not in {"auto", "bar", "plain", "off"}:
            raise ValueError("progress must be auto, bar, plain, or off")

    def worker_count(self, stage: str) -> int:
        override = {
            "m0": self.m0_workers,
            "m1": self.m1_threads,
            "m2": self.m2_threads,
            "m3": self.m3_workers,
        }[stage]
        return override if override is not None else self.threads


def selected_stages(config: PipelineConfig) -> tuple[str, ...]:
    start = PIPELINE_STAGES.index(config.from_stage)
    stop = PIPELINE_STAGES.index(config.to_stage)
    return PIPELINE_STAGES[start : stop + 1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _metadata_root(data_root: Path) -> Path:
    return data_root / ".automorph_v2"


def _manifest_path(data_root: Path, stage: str) -> Path:
    return _metadata_root(data_root) / "stages" / f"{stage}.json"


def _log_path(data_root: Path, stage: str) -> Path:
    return _metadata_root(data_root) / "logs" / f"{stage}.log"


def _run_log_path(data_root: Path) -> Path:
    return data_root / "run.log"


def _pipeline_manifest_path(data_root: Path) -> Path:
    return _metadata_root(data_root) / "pipeline.json"


def _path_has_content(path: Path) -> bool:
    if path.is_file() or path.is_symlink():
        return True
    return path.is_dir() and next(path.iterdir(), None) is not None


def _m3_owned_paths(data_root: Path) -> tuple[Path, ...]:
    root = data_root / "Results" / "M3"
    return tuple(root / relative for relative in M3_MEASUREMENT_OUTPUTS)


def owned_paths(data_root: Path, stage: str) -> tuple[Path, ...]:
    results = data_root / "Results"
    if stage == "m0":
        return (results / "M0",)
    if stage == "m1":
        return (results / "M1",)
    if stage == "m2":
        # Disc/cup creates the initial M3 centring tables as part of M2.
        return (results / "M2", results / "M3")
    if stage == "m3":
        return _m3_owned_paths(data_root)
    raise ValueError(f"unknown stage: {stage}")


def _required_outputs(data_root: Path, stage: str) -> tuple[Path, ...]:
    results = data_root / "Results"
    if stage == "m0":
        return (
            results / "M0" / "images",
            results / "M0" / "crop_info.csv",
            results / "M0" / "run_summary.json",
        )
    if stage == "m1":
        return (
            results / "M1" / "results_ensemble.csv",
            results / "M1" / "M1_quality_final.csv",
            results / "M1" / "M1_quality_summary.csv",
            results / "M1" / "Good_quality",
            results / "M1" / "Usable_quality",
            results / "M1" / "Bad_quality",
            results / "M1" / "run_summary.json",
        )
    if stage == "m2":
        return (
            results / "M2" / "binary_vessel",
            results / "M2" / "artery_vein",
            results / "M2" / "optic_disc_cup",
            results / "M2" / "run_summary.json",
            results / "M3" / "Disc_centred" / "Disc_cup_results.csv",
            results / "M3" / "Macular_centred" / "Disc_cup_results.csv",
        )
    if stage == "m3":
        return _m3_owned_paths(data_root)
    raise ValueError(f"unknown stage: {stage}")


def _output_fingerprint_paths(data_root: Path, stage: str) -> Mapping[str, Path]:
    results = data_root / "Results"
    if stage == "m0":
        return {"m0": results / "M0"}
    if stage == "m1":
        return {"m1": results / "M1"}
    if stage == "m2":
        return {
            "m2": results / "M2",
            "disc-centres": results / "M3" / "Disc_centred" / "Disc_cup_results.csv",
            "macular-centres": results / "M3" / "Macular_centred" / "Disc_cup_results.csv",
        }
    return {
        relative: results / "M3" / relative
        for relative in M3_MEASUREMENT_OUTPUTS
    }


def _input_paths(data_root: Path, stage: str) -> Mapping[str, Path]:
    results = data_root / "Results"
    if stage == "m0":
        return {
            "images": data_root / "images",
            "resolution": data_root / "resolution_information.csv",
        }
    if stage == "m1":
        return {"m0-images": results / "M0" / "images"}
    if stage == "m2":
        return {
            "good-quality": results / "M1" / "Good_quality",
            "crop-info": results / "M0" / "crop_info.csv",
        }
    if stage == "m3":
        return {
            "crop-info": results / "M0" / "crop_info.csv",
            "binary-vessel": results / "M2" / "binary_vessel",
            "artery-vein": results / "M2" / "artery_vein",
            "disc-centres": results / "M3" / "Disc_centred" / "Disc_cup_results.csv",
            "macular-centres": results / "M3" / "Macular_centred" / "Disc_cup_results.csv",
        }
    raise ValueError(f"unknown stage: {stage}")


def _model_paths(stage: str) -> Mapping[str, Path]:
    if stage == "m1":
        return {
            "quality-checkpoints": RUNTIME_ROOT
            / "M1_Retinal_Image_quality_EyePACS"
            / "Retinal_quality"
            / "EyePACS_quality"
            / "efficientnet"
        }
    if stage == "m2":
        return {
            "vessel-model": RUNTIME_ROOT / "M2_Vessel_seg" / "model.py",
            "vessel-checkpoints": RUNTIME_ROOT / "M2_Vessel_seg" / "Saved_model",
            "artery-vein-model": RUNTIME_ROOT / "M2_Artery_vein" / "scripts" / "model.py",
            "artery-vein-checkpoints": RUNTIME_ROOT / "M2_Artery_vein" / "ALL-AV",
            "disc-cup-models": RUNTIME_ROOT / "M2_lwnet_disc_cup" / "models",
            "disc-cup-checkpoints": RUNTIME_ROOT / "M2_lwnet_disc_cup" / "experiments",
        }
    return {}


def _implementation_paths(stage: str) -> Mapping[str, Path]:
    module = stage.upper()
    module_root = SOURCE_ROOT / "v2" / module
    paths: dict[str, Path] = {
        "wrapper": SOURCE_ROOT / "v2" / f"run_{stage}.py",
    }
    for path in sorted(module_root.rglob("*.py")):
        relative = path.relative_to(module_root).as_posix()
        paths[f"module:{relative}"] = path
    return paths


def _m3_configuration_path() -> Path:
    return (
        RUNTIME_ROOT
        / "M3_feature_whole_pic"
        / "retipy"
        / "resources"
        / "retipy.config"
    )


def _hash_file(digest: Any, path: Path) -> tuple[int, int]:
    size = path.stat().st_size
    digest.update(str(size).encode("ascii"))
    digest.update(b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return 1, size


def _fingerprint_paths(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Return a content SHA-256 over deterministic labels and relative paths."""

    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    missing: list[str] = []
    for label in sorted(paths):
        root = Path(paths[label])
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        if not root.exists():
            digest.update(b"MISSING\0")
            missing.append(label)
            continue
        if root.is_file():
            digest.update(b"FILE\0")
            count, byte_count = _hash_file(digest, root)
            files += count
            total_bytes += byte_count
            continue
        digest.update(b"DIRECTORY\0")
        entries = sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix.lower() not in {".pyc", ".pyo"}
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        for path in entries:
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            count, byte_count = _hash_file(digest, path)
            files += count
            total_bytes += byte_count
    return {
        "sha256": digest.hexdigest(),
        "files": files,
        "bytes": total_bytes,
        "missing": missing,
    }


def _runtime_contract() -> dict[str, str]:
    distributions = (
        "numpy",
        "pandas",
        "opencv-python",
        "opencv-python-headless",
        "Pillow",
        "scikit-image",
        "torch",
        "torchvision",
        "efficientnet-pytorch",
        "tqdm",
    )
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
    }
    for distribution in distributions:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "missing"
    return versions


def _stage_settings(config: PipelineConfig, stage: str) -> dict[str, Any]:
    common: dict[str, Any] = {
        "stage": stage,
        "workers": config.worker_count(stage),
        "runtime": _runtime_contract(),
    }
    if stage == "m0":
        common.update(
            device=config.m0_device,
            png_compression=config.png_compression,
            fail_on_image_error=config.fail_on_m0_image_error,
        )
    elif stage in {"m1", "m2"}:
        common["device"] = config.device
    else:
        common.update(
            device="cpu",
            start_method="spawn",
            fail_on_metric_error=False,
            retipy_configuration=_fingerprint_paths(
                {"retipy.config": _m3_configuration_path()}
            ),
        )
    if stage == "m2":
        common["vessel_engine"] = config.vessel_engine
    return common


def _configuration_fingerprint(config: PipelineConfig, stage: str) -> dict[str, Any]:
    settings = _stage_settings(config, stage)
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "settings": settings}


def _build_fingerprints(config: PipelineConfig, stage: str) -> dict[str, Any]:
    inputs = _fingerprint_paths(_input_paths(config.data_root, stage))
    models = _fingerprint_paths(_model_paths(stage))
    implementation = _fingerprint_paths(_implementation_paths(stage))
    return {
        "input": inputs,
        "configuration": _configuration_fingerprint(config, stage),
        "model": models,
        "implementation": implementation,
    }


def _assert_required_inputs(config: PipelineConfig, stage: str) -> None:
    missing = [
        f"{label}: {path}"
        for label, path in _input_paths(config.data_root, stage).items()
        if not path.exists()
    ]
    model_missing = [
        f"{label}: {path}"
        for label, path in _model_paths(stage).items()
        if not path.exists()
    ]
    missing.extend(model_missing)
    implementation_missing = [
        f"{label}: {path}"
        for label, path in _implementation_paths(stage).items()
        if not path.exists()
    ]
    missing.extend(implementation_missing)
    if stage == "m3" and not _m3_configuration_path().is_file():
        missing.append(f"retipy-config: {_m3_configuration_path()}")
    if missing:
        raise PipelineError(
            f"{stage} preflight is missing required input/model paths:\n  - "
            + "\n  - ".join(missing)
        )


def _stage_command(config: PipelineConfig, stage: str) -> list[str]:
    wrapper = SOURCE_ROOT / "v2" / f"run_{stage}.py"
    data = str(config.data_root)
    workers = str(config.worker_count(stage))
    child_progress = "off" if config.progress == "off" else "protocol"
    if stage == "m0":
        command = [
            sys.executable,
            str(wrapper),
            data,
            "--device",
            config.m0_device,
            "--workers",
            workers,
            "--png-compression",
            str(config.png_compression),
            "--progress",
            child_progress,
        ]
        if config.fail_on_m0_image_error:
            command.append("--fail-on-image-error")
        return command
    if stage == "m1":
        weights = _model_paths("m1")["quality-checkpoints"]
        return [
            sys.executable,
            str(wrapper),
            data,
            "--device",
            config.device,
            "--threads",
            workers,
            "--weights-dir",
            str(weights),
            "--progress",
            child_progress,
        ]
    if stage == "m2":
        models = _model_paths("m2")
        return [
            sys.executable,
            str(wrapper),
            data,
            "--device",
            config.device,
            "--threads",
            workers,
            "--vessel-engine",
            config.vessel_engine,
            "--vessel-weights-dir",
            str(models["vessel-checkpoints"]),
            "--artery-vein-weights-dir",
            str(models["artery-vein-checkpoints"]),
            "--disc-cup-module-dir",
            str(RUNTIME_ROOT / "M2_lwnet_disc_cup"),
            "--progress",
            child_progress,
        ]
    if stage == "m3":
        configuration = _m3_configuration_path()
        return [
            sys.executable,
            str(wrapper),
            data,
            "--device",
            "cpu",
            "--workers",
            workers,
            "--start-method",
            "spawn",
            "--configuration",
            str(configuration),
            "--progress",
            child_progress,
        ]
    raise ValueError(f"unknown stage: {stage}")


def _validate_data_root(data_root: Path) -> Path:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise PipelineError(f"DATA_ROOT is not an existing directory: {root}")
    if root.parent == root or root == Path(root.anchor):
        raise PipelineError("refusing to use a filesystem root as DATA_ROOT")
    home = Path.home().resolve()
    if root == home:
        raise PipelineError("refusing to use the user home directory as DATA_ROOT")
    if root == PROJECT_ROOT or PROJECT_ROOT.is_relative_to(root):
        raise PipelineError(
            "DATA_ROOT must not be the release directory or one of its ancestors"
        )
    return root


def _assert_safe_delete_target(data_root: Path, target: Path) -> None:
    resolved_root = data_root.resolve()
    resolved_target = target.resolve()
    try:
        relative = resolved_target.relative_to(resolved_root)
    except ValueError as error:
        raise PipelineError(f"refusing deletion outside DATA_ROOT: {target}") from error
    if not relative.parts or relative.parts[0] not in {"Results", ".automorph_v2"}:
        raise PipelineError(f"refusing deletion of non-owned path: {target}")


def _remove_owned_path(data_root: Path, target: Path) -> None:
    if not target.exists() and not target.is_symlink():
        return
    _assert_safe_delete_target(data_root, target)
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target)
    else:
        target.unlink()


def _prepare_overwrite(config: PipelineConfig, stages: Sequence[str]) -> None:
    targets: list[Path] = []
    for stage in stages:
        targets.extend(owned_paths(config.data_root, stage))
        # A range-limited overwrite must not discard trusted manifests and
        # logs for stages outside that range. They may still be needed for a
        # later verified resume. Remove only the selected stage state plus the
        # invocation-wide summary, which will be replaced by this run.
        targets.append(_manifest_path(config.data_root, stage))
        targets.append(_log_path(config.data_root, stage))
    targets.append(_pipeline_manifest_path(config.data_root))
    # Parent paths subsume children. Delete shallow paths first and only once.
    unique: list[Path] = []
    for target in sorted(set(targets), key=lambda item: len(item.parts)):
        if any(target == parent or parent in target.parents for parent in unique):
            continue
        unique.append(target)
    for target in unique:
        _remove_owned_path(config.data_root, target)


def _fresh_conflicts(config: PipelineConfig, stages: Sequence[str]) -> list[Path]:
    conflicts: list[Path] = []
    for stage in stages:
        conflicts.extend(
            path for path in owned_paths(config.data_root, stage) if _path_has_content(path)
        )
    metadata_root = _metadata_root(config.data_root)
    if _path_has_content(metadata_root):
        conflicts.append(metadata_root)
    return conflicts


def _ensure_fresh(config: PipelineConfig, stages: Sequence[str]) -> None:
    conflicts = _fresh_conflicts(config, stages)
    if conflicts:
        rendered = "\n  - ".join(str(path) for path in conflicts)
        raise ExistingOutputError(
            "fresh mode refuses existing selected-stage output or run metadata. "
            "Choose a new DATA_ROOT, --resume, or explicit --overwrite:\n  - "
            + rendered
        )


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExistingOutputError(f"cannot trust unreadable manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExistingOutputError(f"cannot trust non-object manifest: {path}")
    return value


def _outputs_complete(data_root: Path, stage: str) -> tuple[bool, list[str]]:
    missing = [str(path) for path in _required_outputs(data_root, stage) if not path.exists()]
    return not missing, missing


def _resume_decision(
    config: PipelineConfig,
    stage: str,
    fingerprints: Mapping[str, Any],
) -> bool:
    """Return True only when a completed, unchanged stage can be skipped."""

    manifest_path = _manifest_path(config.data_root, stage)
    manifest = _load_manifest(manifest_path)
    existing = [path for path in owned_paths(config.data_root, stage) if _path_has_content(path)]
    if manifest is None:
        if existing:
            raise ExistingOutputError(
                f"cannot resume {stage}: output exists without a completed manifest; "
                "use --overwrite after checking the data root"
            )
        return False
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ExistingOutputError(f"cannot resume {stage}: unsupported manifest schema")
    if manifest.get("status") != "completed":
        if existing:
            raise ExistingOutputError(
                f"cannot resume {stage}: manifest status is {manifest.get('status')!r} "
                "and partial output exists; use --overwrite"
            )
        return False
    if manifest.get("fingerprints") != fingerprints:
        raise ExistingOutputError(
            f"cannot resume {stage}: input/configuration/model fingerprints changed; "
            "use a fresh root or --overwrite"
        )
    complete, missing = _outputs_complete(config.data_root, stage)
    if not complete:
        raise ExistingOutputError(
            f"cannot resume {stage}: completed manifest has missing outputs: {missing}"
        )
    current_output = _fingerprint_paths(_output_fingerprint_paths(config.data_root, stage))
    if manifest.get("output_fingerprint") != current_output:
        raise ExistingOutputError(
            f"cannot resume {stage}: completed outputs changed after the manifest was written"
        )
    return True


def _write_run_event(
    run_log: TextIO,
    message: str,
    *,
    label: str = "INFO",
    echo: bool = False,
) -> None:
    """Write one durable, timestamped event to the unified run log."""

    normalized = message.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n") or [""]
    for line in lines:
        run_log.write(f"{_utc_now()} [{label}] {line}\n")
    run_log.flush()
    if echo:
        print(message, flush=True)


def _write_stage_output(run_log: TextIO, stage: str, output: str) -> None:
    """Make carriage-return progress output readable in the persistent log."""

    normalized = output.replace("\r\n", "\n").replace("\r", "\n")
    for line in normalized.splitlines():
        _write_run_event(run_log, line, label=stage.upper())


def _progress_duration(seconds: object) -> str:
    if not isinstance(seconds, (int, float)) or not math.isfinite(float(seconds)):
        return "--:--"
    rounded = max(0, int(float(seconds) + 0.5))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


@dataclass
class _RelayedProgress:
    total: int
    unit: str
    current: int = 0
    last_percent_bucket: int = -1
    last_logged_elapsed: float = -math.inf
    bar: Any | None = None


class _ProgressRelay:
    """Render child JSON progress once and persist clean, ANSI-free snapshots."""

    def __init__(
        self,
        *,
        stage: str,
        mode: str,
        stage_log: TextIO,
        run_log: TextIO | None,
    ) -> None:
        self.stage = stage
        self.stage_log = stage_log
        self.run_log = run_log
        self.tasks: dict[str, _RelayedProgress] = {}
        requested = mode
        interactive = bool(getattr(sys.stderr, "isatty", lambda: False)())
        use_bar = requested == "bar" or (requested == "auto" and interactive)
        self.tqdm_type: Any | None = None
        if use_bar:
            try:
                from tqdm import tqdm
            except ImportError:
                requested = "plain"
            else:
                self.tqdm_type = tqdm
                requested = "bar"
        elif requested == "auto":
            requested = "plain"
        self.mode = requested

    def _record(self, message: str, *, echo: bool) -> None:
        rendered = "progress: " + message
        self.stage_log.write(rendered + "\n")
        self.stage_log.flush()
        if self.run_log is not None:
            _write_run_event(
                self.run_log,
                message,
                label=f"{self.stage.upper()} PROGRESS",
            )
        if echo:
            print(message, flush=True)

    def _message(
        self,
        phase: str,
        state: _RelayedProgress,
        payload: Mapping[str, Any],
        event: str,
    ) -> str:
        total = state.total
        percent = 100.0 * state.current / total if total else 100.0
        elapsed = _progress_duration(payload.get("elapsed_s"))
        eta = _progress_duration(payload.get("eta_s"))
        rate_value = payload.get("rate")
        rate = float(rate_value) if isinstance(rate_value, (int, float)) else 0.0
        suffix = "" if event in {"start", "progress"} else f" [{event}]"
        return (
            f"{self.stage.upper()} {phase}: {state.current}/{total} {state.unit} "
            f"({percent:.1f}%) | elapsed {elapsed} | "
            f"rate {rate:.2f} {state.unit}/s | ETA {eta}{suffix}"
        )

    def handle(self, line: str) -> bool:
        payload = parse_progress_line(line)
        if payload is None:
            return False
        try:
            event = str(payload["event"])
            stage = str(payload["stage"])
            phase = str(payload["phase"])
            current = int(payload["current"])
            total = int(payload["total"])
            unit = str(payload["unit"])
            elapsed = float(payload.get("elapsed_s", 0.0))
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if stage.lower() != self.stage.lower() or current < 0 or total < 0:
            return False

        if event == "start" or phase not in self.tasks:
            state = _RelayedProgress(total=total, unit=unit, current=current)
            self.tasks[phase] = state
            if self.mode == "bar" and self.tqdm_type is not None:
                state.bar = self.tqdm_type(
                    total=total,
                    initial=current,
                    desc=f"{self.stage.upper()} {phase}",
                    unit=unit,
                    file=sys.stderr,
                    dynamic_ncols=True,
                    leave=True,
                )
        else:
            state = self.tasks[phase]
            previous = state.current
            state.current = min(current, total) if total else current
            state.total = total
            state.unit = unit
            if state.bar is not None:
                state.bar.update(max(0, state.current - previous))

        state.current = min(current, total) if total else current
        percent_bucket = int((100 * state.current / total) // 5) if total else 20
        terminal_event = event not in {"start", "progress"}
        should_log = (
            event == "start"
            or terminal_event
            or percent_bucket > state.last_percent_bucket
            or elapsed - state.last_logged_elapsed >= 30.0
        )
        if should_log:
            self._record(
                self._message(phase, state, payload, event),
                echo=self.mode == "plain",
            )
            state.last_percent_bucket = percent_bucket
            state.last_logged_elapsed = elapsed
        if terminal_event and state.bar is not None:
            if event != "complete":
                state.bar.set_postfix_str(event)
            state.bar.close()
            state.bar = None
        return True

    def close(self) -> None:
        for state in self.tasks.values():
            if state.bar is not None:
                state.bar.close()
                state.bar = None


class _DataRootRunLock:
    """An OS-released cross-platform lock preventing two writers per DATA_ROOT."""

    def __init__(self, data_root: Path) -> None:
        self.path = data_root / ".automorph_v2.run.lock"
        self.stream: Any | None = None
        self.run_id = uuid4().hex

    def acquire(self) -> None:
        if self.path.is_symlink():
            raise PipelineError(f"refusing to follow a run-lock symlink: {self.path}")
        if self.path.exists() and not self.path.is_file():
            raise PipelineError(f"run-lock path is not a regular file: {self.path}")
        if self.path.exists() and self.path.stat().st_nlink > 1:
            raise PipelineError(f"refusing to write a hard-linked run lock: {self.path}")
        try:
            stream = self.path.open("a+b")
        except OSError as error:
            raise PipelineError(f"cannot open DATA_ROOT run lock {self.path}: {error}") from error
        try:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, ImportError) as error:
            stream.close()
            owner = ""
            try:
                owner = self.path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                pass
            detail = f"; owner={owner}" if owner else ""
            raise ExistingOutputError(
                f"another AutoMorph v2 run already owns DATA_ROOT {self.path.parent}{detail}"
            ) from error
        payload = {
            "run_id": self.run_id,
            "pid": os.getpid(),
            "host": platform.node(),
            "acquired_at": _utc_now(),
        }
        try:
            encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
            stream.seek(0)
            stream.truncate()
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
        except BaseException:
            self._unlock(stream)
            stream.close()
            raise
        self.stream = stream

    @staticmethod
    def _unlock(stream: Any) -> None:
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except (OSError, ImportError):
            pass

    def release(self) -> None:
        if self.stream is None:
            return
        stream = self.stream
        self.stream = None
        self._unlock(stream)
        stream.close()

    def __enter__(self) -> "_DataRootRunLock":
        if self.stream is None:
            self.acquire()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> bool:
        self.release()
        return False


def _resolution_csv_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_resolution_provenance(data_root: Path) -> dict[str, Any] | None:
    path = _metadata_root(data_root) / "resolution_provenance.json"
    if path.is_symlink():
        raise ExistingOutputError(
            f"refusing to follow a resolution provenance symlink: {path}"
        )
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExistingOutputError(
            f"cannot trust unreadable resolution provenance {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ExistingOutputError(
            f"cannot trust non-object resolution provenance: {path}"
        )
    return value


def _ensure_resolution_information(
    config: PipelineConfig,
    run_log: TextIO,
    previous_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep a provided calibration or create the original 0.008 compatibility CSV."""

    resolution_path = config.data_root / "resolution_information.csv"
    provenance_path = _metadata_root(config.data_root) / "resolution_provenance.json"
    image_dir = config.data_root / "images"
    if not image_dir.is_dir():
        raise PipelineError(f"M0 input directory is missing: {image_dir}")
    if resolution_path.is_symlink():
        raise PipelineError(
            f"refusing to follow a resolution_information.csv symlink: {resolution_path}"
        )
    image_names = sorted(
        path.name
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in M0_IMAGE_EXTENSIONS
    )
    if resolution_path.exists():
        if not resolution_path.is_file():
            raise PipelineError(
                f"resolution_information.csv is not a regular file: {resolution_path}"
            )
        try:
            with resolution_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or not {"fundus", "res"}.issubset(
                    reader.fieldnames
                ):
                    raise PipelineError(
                        f"{resolution_path} must contain columns: fundus,res"
                    )
                names: set[str] = set()
                duplicates: list[str] = []
                invalid: list[str] = []
                for row in reader:
                    name = str(row.get("fundus", ""))
                    if name in names:
                        duplicates.append(name)
                    names.add(name)
                    try:
                        value = float(row.get("res", ""))
                    except (TypeError, ValueError):
                        invalid.append(name)
                    else:
                        if not math.isfinite(value) or value <= 0:
                            invalid.append(name)
        except OSError as error:
            raise PipelineError(f"cannot read {resolution_path}: {error}") from error
        if duplicates:
            raise PipelineError(
                "resolution_information.csv contains duplicate fundus values: "
                + ", ".join(duplicates[:5])
            )
        if invalid:
            raise PipelineError(
                "resolution_information.csv contains non-finite or non-positive res "
                "values for: "
                + ", ".join(invalid[:5])
            )
        missing = sorted(set(image_names) - names)
        if missing:
            raise PipelineError(
                f"resolution_information.csv is missing {len(missing)} images: "
                + ", ".join(missing[:5])
            )
        current_hash = _resolution_csv_sha256(resolution_path)
        prior_auto = bool(
            previous_provenance
            and previous_provenance.get("auto_generated") is True
        )
        previous_hash = (
            previous_provenance.get("csv_sha256")
            if previous_provenance is not None
            else None
        )
        previous_source = (
            previous_provenance.get("source")
            if previous_provenance is not None
            else None
        )
        known_unchanged_auto = bool(
            prior_auto
            and previous_source
            in {"legacy_uniform_default", "uniform_config_value"}
            and isinstance(previous_hash, str)
            and len(previous_hash) == 64
            and current_hash == previous_hash
        )
        if known_unchanged_auto and config.require_calibrated_resolution:
            raise PipelineError(
                "--require-calibrated-resolution rejects the existing "
                "resolution_information.csv because AutoMorph v2 previously "
                "generated it from a uniform assumption; provide a calibrated "
                "table and remove the old resolution provenance before retrying"
            )

        previous_resolution: float | None = None
        if previous_provenance is not None:
            try:
                candidate_resolution = float(
                    previous_provenance.get("resolution_mm_per_pixel")
                )
            except (TypeError, ValueError):
                pass
            else:
                if math.isfinite(candidate_resolution) and candidate_resolution > 0:
                    previous_resolution = candidate_resolution

        requested_uniform_changed = bool(
            known_unchanged_auto
            and previous_resolution is not None
            and not math.isclose(
                previous_resolution,
                config.resolution_mm_per_pixel,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )
        if requested_uniform_changed:
            if config.mode != "overwrite":
                raise ExistingOutputError(
                    "resolution_information.csv was auto-generated with "
                    f"{previous_resolution} mm/pixel, but this run requests "
                    f"{config.resolution_mm_per_pixel} mm/pixel; use --overwrite "
                    "to regenerate the known auto-generated table"
                )
            temporary = resolution_path.with_name(
                f".{resolution_path.name}.{uuid4().hex}.tmp"
            )
            try:
                with temporary.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream, lineterminator="\n")
                    writer.writerow(("fundus", "res"))
                    writer.writerows(
                        (name, str(config.resolution_mm_per_pixel))
                        for name in image_names
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, resolution_path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            current_hash = _resolution_csv_sha256(resolution_path)
            known_unchanged_auto = True

        if known_unchanged_auto:
            uses_legacy_default = (
                config.resolution_mm_per_pixel == DEFAULT_RESOLUTION_MM_PER_PIXEL
            )
            assumption = (
                "legacy Topcon-compatible default"
                if uses_legacy_default
                else "configured uniform value"
            )
            warning = (
                "Physical resolution was not supplied. The existing table is a "
                f"known AutoMorph v2 {assumption} of "
                f"{config.resolution_mm_per_pixel} mm/pixel for every image. "
                "Size-dependent M2/M3 measurements use this assumed scale."
            )
            provenance = {
                "schema_version": 1,
                "source": (
                    "legacy_uniform_default"
                    if uses_legacy_default
                    else "uniform_config_value"
                ),
                "auto_generated": True,
                "path": str(resolution_path),
                "created_at": (
                    previous_provenance.get("created_at")
                    if previous_provenance is not None
                    else None
                ),
                "recorded_at": _utc_now(),
                "image_count": len(image_names),
                "resolution_mm_per_pixel": config.resolution_mm_per_pixel,
                "units": "mm/pixel in the original image",
                "csv_sha256": current_hash,
                "warning": warning,
            }
            _atomic_json(provenance_path, provenance)
            action = "regenerated" if requested_uniform_changed else "reused"
            _write_run_event(
                run_log,
                f"WARNING: {warning} The known auto-generated table was {action}: "
                f"{resolution_path}.",
                label="INPUT",
                echo=True,
            )
            return provenance

        if prior_auto:
            warning = (
                "resolution_information.csv differs from the table previously "
                "auto-generated by AutoMorph v2. It was preserved, but it is not "
                "treated as calibrated automatically. Verify the values; after "
                "supplying a calibrated table, remove resolution_provenance.json "
                "to explicitly adopt it."
            )
            if config.require_calibrated_resolution:
                raise PipelineError(
                    "--require-calibrated-resolution rejects "
                    "resolution_information.csv because it was modified after "
                    "automatic generation; verify the table and remove the old "
                    "resolution provenance before retrying"
                )
            provenance = {
                "schema_version": 1,
                "source": "modified_after_auto_generation",
                "auto_generated": True,
                "path": str(resolution_path),
                "recorded_at": _utc_now(),
                "row_count": len(names),
                "csv_sha256": current_hash,
                "previous_csv_sha256": previous_hash,
                "warning": warning,
            }
            _atomic_json(provenance_path, provenance)
            _write_run_event(
                run_log,
                f"WARNING: {warning} Existing table: {resolution_path}.",
                label="INPUT",
                echo=True,
            )
            return provenance

        provenance: dict[str, Any] = {
            "schema_version": 1,
            "source": "provided",
            "auto_generated": False,
            "path": str(resolution_path),
            "recorded_at": _utc_now(),
            "row_count": len(names),
            "csv_sha256": current_hash,
            "note": "Existing calibration file preserved without modification.",
        }
        _atomic_json(provenance_path, provenance)
        _write_run_event(
            run_log,
            f"using existing physical-resolution table: {resolution_path}",
            label="INPUT",
        )
        return provenance

    if config.require_calibrated_resolution:
        raise PipelineError(
            "resolution_information.csv is required by "
            "--require-calibrated-resolution but is missing: "
            f"{resolution_path}"
        )

    temporary = resolution_path.with_name(
        f".{resolution_path.name}.{uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("fundus", "res"))
            writer.writerows(
                (name, str(config.resolution_mm_per_pixel)) for name in image_names
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, resolution_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    uses_legacy_default = (
        config.resolution_mm_per_pixel == DEFAULT_RESOLUTION_MM_PER_PIXEL
    )
    assumption = (
        "legacy Topcon-compatible default"
        if uses_legacy_default
        else "configured uniform value"
    )
    warning = (
        "Physical resolution was not supplied. AutoMorph v2 generated the "
        f"{assumption} {config.resolution_mm_per_pixel} mm/pixel for every image. "
        "Size-dependent M2/M3 measurements use this assumed scale."
    )
    provenance = {
        "schema_version": 1,
        "source": (
            "legacy_uniform_default" if uses_legacy_default else "uniform_config_value"
        ),
        "auto_generated": True,
        "path": str(resolution_path),
        "created_at": _utc_now(),
        "image_count": len(image_names),
        "resolution_mm_per_pixel": config.resolution_mm_per_pixel,
        "units": "mm/pixel in the original image",
        "csv_sha256": _resolution_csv_sha256(resolution_path),
        "warning": warning,
    }
    _atomic_json(provenance_path, provenance)
    _write_run_event(
        run_log,
        f"WARNING: {warning} Generated {resolution_path} with {len(image_names)} rows.",
        label="INPUT",
        echo=True,
    )
    return provenance


def _run_stage_subprocess(
    command: Sequence[str],
    log_path: Path,
    *,
    run_log: TextIO | None = None,
    stage: str | None = None,
    progress_mode: str = "auto",
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    # Every stage command has the resolved DATA_ROOT as its first positional
    # argument. Export it as well for any retained legacy helper that still
    # observes the original environment contract.
    environment["AUTOMORPH_DATA"] = str(command[2])
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        rendered_command = subprocess.list2cmdline(list(command))
        log.write("command: " + rendered_command + "\n")
        log.flush()
        if run_log is not None:
            _write_run_event(
                run_log,
                "command: " + rendered_command,
                label=(stage or "stage").upper(),
            )
        relay = _ProgressRelay(
            stage=stage or "stage",
            mode=progress_mode,
            stage_log=log,
            run_log=run_log,
        )
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                if relay.handle(line):
                    continue
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                if run_log is not None:
                    _write_stage_output(run_log, stage or "stage", line)
            return process.wait()
        except BaseException:
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    if process.poll() is None:
                        try:
                            process.kill()
                        except OSError:
                            pass
                        process.wait()
            raise
        finally:
            relay.close()
            process.stdout.close()


def _execute_stage(
    config: PipelineConfig,
    stage: str,
    fingerprints: Mapping[str, Any],
    *,
    run_log: TextIO | None = None,
) -> dict[str, Any]:
    command = _stage_command(config, stage)
    manifest_path = _manifest_path(config.data_root, stage)
    log_path = _log_path(config.data_root, stage)
    started_at = _utc_now()
    started = time.perf_counter()
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "stage": stage,
        "status": "running",
        "started_at": started_at,
        "command": list(command),
        "log": str(log_path),
        "run_log": str(_run_log_path(config.data_root)),
        "fingerprints": dict(fingerprints),
    }
    _atomic_json(manifest_path, manifest)
    try:
        return_code = _run_stage_subprocess(
            command,
            log_path,
            run_log=run_log,
            stage=stage,
            progress_mode=config.progress,
        )
        if return_code:
            raise StageFailedError(
                f"{stage} exited with code {return_code}; see {log_path}"
            )
        complete, missing = _outputs_complete(config.data_root, stage)
        if not complete:
            raise StageFailedError(
                f"{stage} exited successfully but required outputs are missing: {missing}"
            )
        output_fingerprint = _fingerprint_paths(
            _output_fingerprint_paths(config.data_root, stage)
        )
    except BaseException as error:
        failed = {
            **manifest,
            "status": "failed",
            "finished_at": _utc_now(),
            "wall_s": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
        }
        _atomic_json(manifest_path, failed)
        raise
    completed = {
        **manifest,
        "status": "completed",
        "finished_at": _utc_now(),
        "wall_s": time.perf_counter() - started,
        "return_code": 0,
        "output_fingerprint": output_fingerprint,
    }
    _atomic_json(manifest_path, completed)
    return completed


def _public_config(config: PipelineConfig) -> dict[str, Any]:
    value = asdict(config)
    value["data_root"] = str(config.data_root)
    return value


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    """Run a strict contiguous stage range and return its persisted summary."""

    validated_root = _validate_data_root(config.data_root)
    if validated_root != config.data_root:
        config = replace(config, data_root=validated_root)
    stages = selected_stages(config)
    # Resolve every prospective write/delete target before creating metadata.
    # This rejects a Results or metadata symlink that escapes DATA_ROOT even
    # during a normal fresh/resume invocation, not only during --overwrite.
    boundary_targets = [
        path for stage in stages for path in owned_paths(config.data_root, stage)
    ]
    boundary_targets.append(_metadata_root(config.data_root))
    for target in boundary_targets:
        _assert_safe_delete_target(config.data_root, target)
    run_log_path = _run_log_path(config.data_root)
    if run_log_path.is_symlink():
        raise PipelineError(f"refusing to follow a run.log symlink: {run_log_path}")
    if run_log_path.exists() and not run_log_path.is_file():
        raise PipelineError(f"run.log is not a regular file: {run_log_path}")
    if run_log_path.exists() and run_log_path.stat().st_nlink > 1:
        raise PipelineError(f"refusing to truncate a hard-linked run.log: {run_log_path}")
    run_lock = _DataRootRunLock(config.data_root)
    pipeline_path = _pipeline_manifest_path(config.data_root)
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "run_id": run_lock.run_id,
        "status": "running",
        "started_at": _utc_now(),
        "config": _public_config(config),
        "selected_stages": list(stages),
        "completed_stages": [],
        "skipped_stages": [],
        "stage_manifests": {},
        "run_log": str(run_log_path),
    }
    pipeline_written = False
    run_lock.acquire()
    try:
        run_log_stream = run_log_path.open("w", encoding="utf-8", newline="\n")
    except OSError as error:
        run_lock.release()
        raise PipelineError(f"cannot create run log {run_log_path}: {error}") from error

    with run_lock, run_log_stream as run_log:
        _write_run_event(run_log, "AutoMorph v2 run started", label="RUN")
        _write_run_event(run_log, f"run_id: {run_lock.run_id}", label="RUN")
        _write_run_event(
            run_log,
            "argv: " + json.dumps(sys.argv, ensure_ascii=False),
            label="RUN",
        )
        _write_run_event(
            run_log,
            "config: " + json.dumps(_public_config(config), sort_keys=True),
            label="RUN",
        )
        _write_run_event(
            run_log,
            f"platform: {platform.platform()}; python: {platform.python_version()}",
            label="RUN",
        )
        try:
            previous_resolution_provenance = (
                _load_resolution_provenance(config.data_root)
                if "m0" in stages
                else None
            )
            if config.mode == "overwrite":
                _write_run_event(
                    run_log,
                    "overwrite mode: removing selected pipeline-owned outputs",
                    label="RUN",
                )
                _prepare_overwrite(config, stages)
            elif config.mode == "fresh":
                _ensure_fresh(config, stages)

            if "m0" in stages:
                summary["resolution_information"] = _ensure_resolution_information(
                    config,
                    run_log,
                    previous_provenance=previous_resolution_provenance,
                )
            _atomic_json(pipeline_path, summary)
            pipeline_written = True
            for stage in stages:
                _assert_required_inputs(config, stage)
                fingerprints = _build_fingerprints(config, stage)
                if config.mode == "resume" and _resume_decision(
                    config, stage, fingerprints
                ):
                    _write_run_event(
                        run_log,
                        f"AutoMorph v2 [{stage}] resume: verified and skipped",
                        label=stage.upper(),
                        echo=True,
                    )
                    summary["skipped_stages"].append(stage)
                    summary["stage_manifests"][stage] = str(
                        _manifest_path(config.data_root, stage)
                    )
                else:
                    if config.mode == "resume":
                        existing = [
                            path
                            for path in owned_paths(config.data_root, stage)
                            if _path_has_content(path)
                        ]
                        if existing:
                            raise ExistingOutputError(
                                f"cannot safely retry {stage} with partial output: "
                                f"{existing}; use --overwrite"
                            )
                    _write_run_event(
                        run_log,
                        f"AutoMorph v2 [{stage}] starting",
                        label=stage.upper(),
                        echo=True,
                    )
                    _execute_stage(
                        config,
                        stage,
                        fingerprints,
                        run_log=run_log,
                    )
                    summary["completed_stages"].append(stage)
                    summary["stage_manifests"][stage] = str(
                        _manifest_path(config.data_root, stage)
                    )
                    _write_run_event(
                        run_log,
                        f"AutoMorph v2 [{stage}] complete",
                        label=stage.upper(),
                        echo=True,
                    )
                summary["updated_at"] = _utc_now()
                _atomic_json(pipeline_path, summary)
        except BaseException as error:
            summary.update(
                status="failed",
                finished_at=_utc_now(),
                wall_s=time.perf_counter() - started,
                error=f"{type(error).__name__}: {error}",
            )
            if pipeline_written:
                _atomic_json(pipeline_path, summary)
            label = "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "ERROR"
            _write_run_event(
                run_log,
                f"run failed after {summary['wall_s']:.3f}s: "
                f"{type(error).__name__}: {error}",
                label=label,
            )
            raise
        try:
            summary.update(
                status="completed",
                finished_at=_utc_now(),
                wall_s=time.perf_counter() - started,
            )
            _atomic_json(pipeline_path, summary)
            completed = ", ".join(summary["completed_stages"]) or "none"
            skipped = ", ".join(summary["skipped_stages"]) or "none"
            _write_run_event(
                run_log,
                f"run completed: completed={completed}; skipped={skipped}; "
                f"wall={summary['wall_s']:.3f}s",
                label="DONE",
            )
            return summary
        except BaseException as error:
            summary.update(
                status="failed",
                finished_at=_utc_now(),
                wall_s=time.perf_counter() - started,
                error=f"{type(error).__name__}: {error}",
            )
            try:
                _atomic_json(pipeline_path, summary)
            except BaseException as manifest_error:
                _write_run_event(
                    run_log,
                    f"could not persist failed pipeline manifest: "
                    f"{type(manifest_error).__name__}: {manifest_error}",
                    label="ERROR",
                )
            _write_run_event(
                run_log,
                f"run finalization failed after {summary['wall_s']:.3f}s: "
                f"{type(error).__name__}: {error}",
                label="ERROR",
            )
            raise


__all__ = [
    "ExistingOutputError",
    "PIPELINE_STAGES",
    "PipelineConfig",
    "PipelineError",
    "StageFailedError",
    "owned_paths",
    "run_pipeline",
    "selected_stages",
]
