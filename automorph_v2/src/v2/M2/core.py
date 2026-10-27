"""One-process orchestrator for the three strict AutoMorph M2 branches."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any, Mapping

import torch

from .artery_vein import ArteryVeinConfig, run_artery_vein
from .disc_cup import DiscCupConfig, run_disc_cup
from .vessel import VesselConfig, run_vessel
from .experimental.vessel_low_vram import run_vessel_low_vram

try:
    from ..progress import ProgressReporter
except ImportError:  # imported as ``M2.core`` by the direct stage runner
    import sys

    _V2_ROOT = str(Path(__file__).resolve().parents[1])
    if _V2_ROOT not in sys.path:
        sys.path.insert(0, _V2_ROOT)
    from progress import ProgressReporter


RELEASE_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = RELEASE_ROOT / "runtime"
# Kept as the public compatibility name used by run_m2 and older callers.
AUTOMORPH_ROOT = RUNTIME_ROOT
ALL_MODULES = ("vessel", "artery-vein", "disc-cup")


@dataclass(frozen=True, slots=True)
class M2Config:
    """Configuration for a strict, isolated M2 run."""

    data_root: Path
    device: str = "auto"
    threads: int = 4
    modules: tuple[str, ...] = ALL_MODULES
    automorph_root: Path = AUTOMORPH_ROOT
    vessel_weights_dir: Path | None = None
    vessel_engine: str = "resident"
    artery_vein_weights_dir: Path | None = None
    disc_cup_module_dir: Path | None = None
    artery_vein_engine: str = "strict"
    intersection_backend: str = "vectorized"
    allow_existing_outputs: bool = False
    write_summary: bool = True
    progress: str = "off"
    input_quality_file: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root).resolve())
        object.__setattr__(self, "automorph_root", Path(self.automorph_root).resolve())
        object.__setattr__(self, "modules", tuple(self.modules))
        if self.threads < 1:
            raise ValueError("threads must be at least one")
        if self.vessel_engine not in {"resident", "low-vram"}:
            raise ValueError("vessel_engine must be resident or low-vram")
        unknown = [name for name in self.modules if name not in ALL_MODULES]
        if unknown:
            raise ValueError(f"unknown M2 modules: {unknown}")
        if len(set(self.modules)) != len(self.modules):
            raise ValueError("M2 modules cannot be repeated")
        canonical = tuple(name for name in ALL_MODULES if name in self.modules)
        if self.modules != canonical:
            raise ValueError(f"M2 modules must preserve pipeline order: {ALL_MODULES}")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _cuda_peak_from_stage(summary: Mapping[str, Any]) -> tuple[int, int]:
    allocated = summary.get("peak_memory_allocated_bytes")
    reserved = summary.get("peak_memory_reserved_bytes")
    if allocated is None:
        allocated = summary.get("device_peak_allocated_bytes")
        reserved = summary.get("device_peak_reserved_bytes")
    if allocated is None:
        memory = summary.get("peak_memory_bytes", {})
        if isinstance(memory, Mapping):
            allocated = memory.get("cuda_allocated")
            reserved = memory.get("cuda_reserved")
    return int(allocated or 0), int(reserved or 0)


def _contains_files(path: Path) -> bool:
    return path.is_file() or (
        path.is_dir() and any(item.is_file() for item in path.rglob("*"))
    )


def _disc_owned_paths(results_root: Path) -> list[Path]:
    paths = [results_root / "optic_disc_cup", results_root.parent / "M3"]
    prefixes = (
        "disc_centred_",
        "macular_centred_",
        "Zone_B_disc_centred_",
        "Zone_C_disc_centred_",
        "macular_Zone_B_centred_",
        "macular_Zone_C_centred_",
    )
    for branch in (results_root / "binary_vessel", results_root / "artery_vein"):
        if branch.is_dir():
            paths.extend(
                item
                for item in branch.iterdir()
                if item.is_dir() and item.name.startswith(prefixes)
            )
    return paths


def _ensure_fresh_stage(name: str, results_root: Path) -> None:
    if name == "vessel":
        owned = [results_root / "binary_vessel"]
    elif name == "artery-vein":
        owned = [results_root / "artery_vein"]
    else:
        owned = _disc_owned_paths(results_root)
    stale = [str(path) for path in owned if _contains_files(path)]
    if stale:
        raise FileExistsError(
            "Refusing to mix a strict M2 run with existing stage outputs. "
            "Use a fresh data root or explicitly pass --allow-existing-output. "
            f"Stage {name!r} owns: {stale}"
        )


_DEFAULT_M2_QUALITY_DIRS = ("Good_quality", "Usable_quality", "Bad_quality")


def _read_m2_quality_dirs(m1_root: Path, config_file: Path | None) -> list[str]:
    """Read M1 subdirectory names from a manual config, defaulting to Good/Usable/Bad."""
    if config_file is not None and config_file.is_file():
        dirs: list[str] = []
        for line in config_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.lstrip("-").strip()  # tolerate "- Good_quality" yaml lists
            if line:
                dirs.append(line)
        if dirs:
            return dirs
    return list(_DEFAULT_M2_QUALITY_DIRS)


def _build_combined_m2_input(m1_root: Path, quality_dirs: list[str], metadata_root: Path) -> Path:
    """Aggregate images from the configured M1 quality dirs into one flat input dir."""
    combined = metadata_root / "m2_combined_input"
    if combined.exists():
        shutil.rmtree(combined)
    combined.mkdir(parents=True, exist_ok=True)
    count = 0
    for qdir in quality_dirs:
        src = m1_root / qdir
        if not src.is_dir():
            continue
        for item in src.iterdir():
            if item.is_file() and not item.name.startswith("."):
                target = combined / item.name
                if not target.exists():
                    try:
                        os.link(item, target)
                    except OSError:
                        shutil.copy2(item, target)
                    count += 1
    if count == 0:
        present = [q for q in quality_dirs if (m1_root / q).is_dir()]
        raise FileNotFoundError(
            f"No images found in configured M1 quality dirs {quality_dirs} "
            f"(present: {present}) under {m1_root}"
        )
    return combined


def run_m2(config: M2Config) -> dict[str, Any]:
    """Run selected M2 branches in their fixed production order."""

    started = time.perf_counter()
    data_root = config.data_root
    m1_root = data_root / "Results" / "M1"
    if not m1_root.is_dir():
        raise FileNotFoundError(f"M1 results directory not found: {m1_root}")
    quality_dirs = _read_m2_quality_dirs(
        m1_root,
        config.input_quality_file
        or (m1_root / "m2_input_quality.txt"),
    )
    image_dir = _build_combined_m2_input(
        m1_root, quality_dirs, data_root / ".automorph_v2"
    )

    automorph_root = config.automorph_root
    vessel_module = automorph_root / "M2_Vessel_seg"
    av_module = automorph_root / "M2_Artery_vein"
    disc_module = Path(
        config.disc_cup_module_dir
        or (automorph_root / "M2_lwnet_disc_cup")
    ).resolve()
    vessel_weights = Path(
        config.vessel_weights_dir or (vessel_module / "Saved_model")
    ).resolve()
    av_weights = Path(
        config.artery_vein_weights_dir or (av_module / "ALL-AV")
    ).resolve()

    results_root = data_root / "Results" / "M2"
    results_root.mkdir(parents=True, exist_ok=True)
    stages: dict[str, Any] = {}
    branch_progress = ProgressReporter("M2", config.progress).task(
        "branches", total=len(config.modules), unit="branches"
    )

    for name in config.modules:
        if not config.allow_existing_outputs:
            _ensure_fresh_stage(name, results_root)
        stage_started = time.perf_counter()
        print(f"M2 v2 [{name}] starting", flush=True)
        if name == "vessel":
            vessel_config = VesselConfig(
                input_dir=image_dir,
                output_dir=results_root / "binary_vessel",
                legacy_dir=vessel_module,
                checkpoint_root=vessel_weights,
                device=config.device,
                preprocess_threads=config.threads,
                postprocess_threads=config.threads,
                write_summary=False,
                progress=config.progress,
            )
            vessel_runner = (
                run_vessel
                if config.vessel_engine == "resident"
                else run_vessel_low_vram
            )
            stage = vessel_runner(vessel_config)
        elif name == "artery-vein":
            stage = run_artery_vein(
                ArteryVeinConfig(
                    input_dir=image_dir,
                    output_dir=results_root / "artery_vein",
                    weights_dir=av_weights,
                    device=config.device,
                    preprocess_threads=config.threads,
                    engine=config.artery_vein_engine,
                    write_summary=False,
                    progress=config.progress,
                )
            )
        else:
            required = (
                results_root / "binary_vessel" / "binary_process",
                results_root / "binary_vessel" / "binary_skeleton",
                results_root / "artery_vein" / "artery_binary_process",
                results_root / "artery_vein" / "vein_binary_process",
                results_root / "artery_vein" / "artery_binary_skeleton",
                results_root / "artery_vein" / "vein_binary_skeleton",
            )
            missing = [str(path) for path in required if not path.is_dir()]
            crop_info = data_root / "Results" / "M0" / "crop_info.csv"
            if not crop_info.is_file():
                missing.append(str(crop_info))
            if missing:
                raise FileNotFoundError(
                    "disc-cup postprocessing requires vessel and artery-vein "
                    f"outputs; missing: {missing}"
                )
            input_stems = [
                Path(name).stem
                for name in os.listdir(image_dir)
                if not name.startswith(".") and (image_dir / name).is_file()
            ]
            missing_masks = [
                str(directory / f"{stem}.png")
                for directory in required
                for stem in input_stems
                if not (directory / f"{stem}.png").is_file()
            ]
            if missing_masks:
                preview = missing_masks[:5]
                raise FileNotFoundError(
                    "disc-cup requires six prerequisite masks for every input; "
                    f"missing {len(missing_masks)} files, first: {preview}"
                )
            stage = run_disc_cup(
                DiscCupConfig(
                    data_root=data_root,
                    input_dir=image_dir,
                    legacy_module_root=disc_module,
                    device=config.device,
                    preprocess_threads=config.threads,
                    postprocess_threads=config.threads,
                    intersection_backend=config.intersection_backend,
                    progress=config.progress,
                )
            )
        stage = dict(stage)
        stage["orchestrator_wall_s"] = time.perf_counter() - stage_started
        stages[name] = stage
        print(
            f"M2 v2 [{name}] complete in {stage['orchestrator_wall_s']:.3f}s",
            flush=True,
        )
        branch_progress.update()

    branch_progress.close()

    peaks = [_cuda_peak_from_stage(value) for value in stages.values()]
    summary: dict[str, Any] = {
        "module": "M2",
        "contract": "strict-fp32-legacy-output",
        "data_root": str(data_root),
        "input_dir": str(image_dir),
        "results_dir": str(results_root),
        "device_requested": config.device,
        "threads": config.threads,
        "vessel_engine": config.vessel_engine,
        "modules": list(config.modules),
        "stage_summaries": stages,
        "stage_wall_s": {
            name: value["orchestrator_wall_s"] for name, value in stages.items()
        },
        "total_wall_s": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": max((item[0] for item in peaks), default=0),
        "peak_cuda_reserved_bytes": max((item[1] for item in peaks), default=0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    if config.write_summary:
        _atomic_json(results_root / "run_summary.json", summary)
    return summary


__all__ = ["ALL_MODULES", "AUTOMORPH_ROOT", "M2Config", "run_m2"]
