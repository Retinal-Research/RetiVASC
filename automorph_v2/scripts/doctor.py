#!/usr/bin/env python3
"""Validate an AutoMorph v2 installation without reading user images/results."""

from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
from typing import Any, Sequence


PYTHON_MIN = (3, 10)
PYTHON_MAX_EXCLUSIVE = (3, 13)
EXPECTED_PACKAGES: dict[str, tuple[str, str]] = {
    "numpy": ("numpy", "1.26.4"),
    "opencv-python-headless": ("cv2", "4.10.0.84"),
    "pandas": ("pandas", "2.3.3"),
    "Pillow": ("PIL", "10.4.0"),
    "scipy": ("scipy", "1.13.1"),
    "scikit-image": ("skimage", "0.24.0"),
    "efficientnet_pytorch": ("efficientnet_pytorch", "0.7.1"),
    "tqdm": ("tqdm", "4.67.1"),
}
OPENCV_DISTRIBUTIONS = {
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
}
M1_DIRS = (
    "0_seed_42",
    "1_seed_40",
    "2_seed_38",
    "3_seed_36",
    "4_seed_34",
    "5_seed_32",
    "6_seed_30",
    "7_seed_28",
)
VESSEL_SEEDS = (24, 26, 28, 30, 32, 34, 36, 38, 40, 42)
ENSEMBLE_SEEDS = (28, 30, 32, 34, 36, 38, 40, 42)
AV_CHECKPOINTS = ("CP_best_F1_A.pth", "CP_best_F1_all.pth", "CP_best_F1_V.pth")
SUPPORT_ASSET_PATHS = (
    "runtime/M2_Vessel_seg/model.py",
    "runtime/M2_Artery_vein/scripts/model.py",
    "runtime/M2_lwnet_disc_cup/models/get_model.py",
    "runtime/M2_lwnet_disc_cup/models/res_unet_adrian.py",
    "runtime/M3_feature_whole_pic/retipy/resources/retipy.config",
)


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


class Audit:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.records.append({"check": name, "passed": bool(passed), "detail": detail})

    @property
    def passed(self) -> bool:
        return all(record["passed"] for record in self.records)


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _check_python(audit: Audit) -> None:
    current = sys.version_info[:2]
    valid = PYTHON_MIN <= current < PYTHON_MAX_EXCLUSIVE
    audit.add(
        "python-version",
        valid,
        f"{platform.python_version()} (required >=3.10,<3.13)",
    )


def _check_opencv_distribution(audit: Audit) -> None:
    installed: list[str] = []
    for distribution in metadata.distributions():
        raw_name = distribution.metadata.get("Name", "")
        canonical = _canonical_name(raw_name)
        if canonical in OPENCV_DISTRIBUTIONS:
            installed.append(f"{canonical}=={distribution.version}")
    installed.sort()
    expected = ["opencv-python-headless==4.10.0.84"]
    audit.add(
        "single-opencv-distribution",
        installed == expected,
        ", ".join(installed) if installed else "none installed",
    )


def _check_imports(audit: Audit) -> None:
    for distribution, (module_name, expected_version) in EXPECTED_PACKAGES.items():
        installed_version = _distribution_version(distribution)
        try:
            importlib.import_module(module_name)
        except BaseException as error:
            audit.add(
                f"import:{module_name}",
                False,
                f"{type(error).__name__}: {error}",
            )
            continue
        audit.add(
            f"import:{module_name}",
            installed_version == expected_version,
            f"distribution {distribution}=={installed_version}; expected {expected_version}",
        )

    for module_name in ("torch", "torchvision", "automorph_v2", "automorph_v2.cli"):
        try:
            importlib.import_module(module_name)
        except BaseException as error:
            audit.add(
                f"import:{module_name}",
                False,
                f"{type(error).__name__}: {error}",
            )
        else:
            audit.add(f"import:{module_name}", True, "ok")


def _public_version(version: str | None) -> str | None:
    return None if version is None else version.split("+", 1)[0]


def _check_backend(audit: Audit, backend: str) -> None:
    torch_version = _distribution_version("torch")
    vision_version = _distribution_version("torchvision")
    version_ok = _public_version(torch_version) == "2.7.1" and _public_version(vision_version) == "0.22.1"

    system = platform.system()
    if backend == "cu118":
        version_ok = version_ok and torch_version == "2.7.1+cu118" and vision_version == "0.22.1+cu118"
    elif backend == "cpu" and system != "Darwin":
        version_ok = version_ok and torch_version == "2.7.1+cpu" and vision_version == "0.22.1+cpu"
    elif backend == "mps":
        version_ok = version_ok and system == "Darwin"

    audit.add(
        "torch-version-pair",
        version_ok,
        f"torch={torch_version}, torchvision={vision_version}, backend={backend}",
    )

    try:
        import torch
    except BaseException as error:
        audit.add("device", False, f"torch import failed: {type(error).__name__}: {error}")
        return

    try:
        if backend == "cpu":
            result = float(torch.ones(1, device="cpu").add_(1).item())
            audit.add("device", result == 2.0, f"CPU tensor probe={result}")
        elif backend == "cu118":
            available = system in {"Windows", "Linux"} and torch.version.cuda == "11.8" and torch.cuda.is_available()
            detail = f"system={system}, built_cuda={torch.version.cuda}, available={torch.cuda.is_available()}"
            if available:
                detail += f", device={torch.cuda.get_device_name(0)}"
            audit.add("device", available, detail)
        else:
            mps = getattr(torch.backends, "mps", None)
            built = bool(mps is not None and mps.is_built())
            available = bool(mps is not None and mps.is_available())
            probe_ok = False
            if system == "Darwin" and built and available:
                probe_ok = float(torch.ones(1, device="mps").cpu().item()) == 1.0
            audit.add(
                "device",
                system == "Darwin" and built and available and probe_ok,
                f"system={system}, mps_built={built}, mps_available={available}, probe={probe_ok}",
            )
    except BaseException as error:
        audit.add("device", False, f"{type(error).__name__}: {error}")


def _expected_weights(root: Path) -> set[Path]:
    expected: set[Path] = set()
    runtime_root = root / "runtime"
    m1_root = runtime_root / "M1_Retinal_Image_quality_EyePACS" / "Retinal_quality" / "EyePACS_quality" / "efficientnet"
    for directory in M1_DIRS:
        expected.add(m1_root / directory / "best_loss_checkpoint.pth")

    vessel_root = runtime_root / "M2_Vessel_seg" / "Saved_model" / "train_on_ALL-SIX"
    for seed in VESSEL_SEEDS:
        expected.add(
            vessel_root
            / f"20210630_uniform_thres40_ALL-SIX_savebest_randomseed_{seed}"
            / "G_best_F1_epoch.pth"
        )

    av_root = runtime_root / "M2_Artery_vein" / "ALL-AV"
    for seed in ENSEMBLE_SEEDS:
        checkpoint_root = av_root / f"20210724_ALL-AV_randomseed_{seed}" / "Discriminator_unet"
        for filename in AV_CHECKPOINTS:
            expected.add(checkpoint_root / filename)

    disc_root = runtime_root / "M2_lwnet_disc_cup" / "experiments" / "wnet_All_three_1024_disc_cup"
    for seed in ENSEMBLE_SEEDS:
        expected.add(disc_root / str(seed) / "model_checkpoint.pth")
    return expected


def _check_weights(audit: Audit, root: Path) -> None:
    expected = _expected_weights(root)
    runtime_root = root / "runtime"
    model_roots = (
        runtime_root / "M1_Retinal_Image_quality_EyePACS",
        runtime_root / "M2_Vessel_seg",
        runtime_root / "M2_Artery_vein",
        runtime_root / "M2_lwnet_disc_cup",
    )
    actual = {
        path
        for model_root in model_roots
        if model_root.is_dir()
        for path in model_root.rglob("*.pth")
        if path.is_file()
    }
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    audit.add(
        "weight-inventory",
        len(expected) == 50 and actual == expected,
        f"expected=50, found={len(actual)}, missing={len(missing)}, unexpected={len(unexpected)}",
    )
    if missing:
        audit.add("weight-inventory:missing", False, "; ".join(str(path.relative_to(root)) for path in missing[:5]))
    if unexpected:
        audit.add(
            "weight-inventory:unexpected",
            False,
            "; ".join(str(path.relative_to(root)) for path in unexpected[:5]),
        )

    invalid: list[str] = []
    lfs_prefix = b"version https://git-lfs.github.com/spec/v1"
    for path in sorted(expected & actual):
        try:
            if path.stat().st_size <= 1024:
                invalid.append(f"{path.relative_to(root)} (too small)")
                continue
            with path.open("rb") as stream:
                if stream.read(len(lfs_prefix)) == lfs_prefix:
                    invalid.append(f"{path.relative_to(root)} (Git LFS pointer)")
        except OSError as error:
            invalid.append(f"{path.relative_to(root)} ({error})")
    audit.add(
        "weight-payloads",
        not invalid,
        "all 50 checkpoint payloads present" if not invalid else "; ".join(invalid[:5]),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_model_manifest(audit: Audit, root: Path) -> None:
    """Verify every checkpoint and support asset against the frozen manifest."""

    manifest_path = root / "models" / "manifest.json"
    expected_weights = {
        path.relative_to(root).as_posix() for path in _expected_weights(root)
    }
    expected_paths = expected_weights | set(SUPPORT_ASSET_PATHS)
    errors: list[str] = []
    manifest: dict[str, Any] | None = None
    if manifest_path.is_symlink() or not manifest_path.is_file():
        errors.append("manifest missing or not a regular file")
    else:
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            errors.append("manifest is unreadable")
        else:
            if isinstance(value, dict):
                manifest = value
            else:
                errors.append("manifest root is not an object")

    entries: dict[str, dict[str, Any]] = {}
    if manifest is not None:
        raw_files = manifest.get("files")
        if not isinstance(raw_files, list):
            errors.append("files is not a list")
        else:
            for raw_entry in raw_files:
                if not isinstance(raw_entry, dict):
                    errors.append("non-object file entry")
                    continue
                relative = raw_entry.get("path")
                if not isinstance(relative, str) or relative in entries:
                    errors.append("invalid or duplicate file path")
                    continue
                entries[relative] = raw_entry
        if manifest.get("schema_version") != 1:
            errors.append("schema version mismatch")
        if manifest.get("checkpoint_count") != len(expected_weights):
            errors.append("checkpoint count mismatch")
        if manifest.get("support_asset_count") != len(SUPPORT_ASSET_PATHS):
            errors.append("support asset count mismatch")
        if set(entries) != expected_paths:
            errors.append("manifest file set mismatch")

    actual_checkpoint_bytes = 0
    actual_total_bytes = 0
    for relative in sorted(expected_paths):
        entry = entries.get(relative)
        path = root.joinpath(*relative.split("/"))
        if entry is None or path.is_symlink() or not path.is_file():
            errors.append(f"missing or unsafe payload: {relative}")
            continue
        try:
            byte_count = path.stat().st_size
            digest = _sha256_file(path)
        except OSError:
            errors.append(f"unreadable payload: {relative}")
            continue
        actual_total_bytes += byte_count
        if relative in expected_weights:
            actual_checkpoint_bytes += byte_count
        if entry.get("bytes") != byte_count or entry.get("sha256") != digest:
            errors.append(f"payload hash or size mismatch: {relative}")

    if manifest is not None:
        if manifest.get("total_checkpoint_bytes") != actual_checkpoint_bytes:
            errors.append("checkpoint byte total mismatch")
        if manifest.get("total_manifested_bytes") != actual_total_bytes:
            errors.append("manifested byte total mismatch")
    audit.add(
        "model-manifest",
        not errors,
        (
            f"verified {len(expected_weights)} checkpoints and "
            f"{len(SUPPORT_ASSET_PATHS)} support assets by SHA-256"
            if not errors
            else "; ".join(errors[:5])
        ),
    )


def _check_writeability(audit: Audit, write_root: Path) -> None:
    if not write_root.is_dir():
        audit.add("writeability", False, f"directory does not exist: {write_root}")
        return
    try:
        with tempfile.TemporaryDirectory(prefix=".automorph_v2_doctor_", dir=write_root) as directory:
            probe = Path(directory) / "probe"
            probe.write_bytes(b"automorph-v2")
            valid = probe.read_bytes() == b"automorph-v2"
        audit.add(
            "writeability",
            valid,
            f"temporary probe created and removed under {write_root}; existing files were not opened",
        )
    except OSError as error:
        audit.add("writeability", False, f"{type(error).__name__}: {error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="AutoMorph_v2 release root (default: inferred from this script)",
    )
    parser.add_argument("--backend", choices=("cpu", "cu118", "mps"), default="cpu")
    parser.add_argument(
        "--write-root",
        type=Path,
        help=(
            "existing output/work directory to probe; the doctor creates and removes "
            "one unique temporary directory and never opens existing user files"
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    write_root = (args.write_root or root).expanduser().resolve()
    audit = Audit()

    audit.add("release-root", root.is_dir(), str(root))
    _check_python(audit)
    _check_opencv_distribution(audit)
    _check_imports(audit)
    _check_backend(audit, args.backend)
    _check_weights(audit, root)
    _check_model_manifest(audit, root)
    _check_writeability(audit, write_root)

    payload = {
        "passed": audit.passed,
        "backend": args.backend,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "root": str(root),
        "checks": audit.records,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for record in audit.records:
            marker = "PASS" if record["passed"] else "FAIL"
            print(f"[{marker}] {record['check']}: {record['detail']}")
        passed_count = sum(bool(record["passed"]) for record in audit.records)
        print(f"\nDoctor: {'PASS' if audit.passed else 'FAIL'} ({passed_count}/{len(audit.records)} checks)")
    return 0 if audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
