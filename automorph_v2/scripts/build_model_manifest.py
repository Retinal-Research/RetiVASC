#!/usr/bin/env python3
"""Build and verify the deterministic AutoMorph v2 model inventory.

The manifest is intentionally generated from an explicit allow-list.  A
missing, renamed, or unexpected checkpoint is therefore a release error rather
than something silently omitted from the inventory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence


EXPECTED_CHECKPOINT_COUNT = 50
HASH_CHUNK_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """One required release artifact, relative to the release root."""

    module: str
    kind: str
    path: str


def _checkpoint_specs() -> tuple[ArtifactSpec, ...]:
    specs: list[ArtifactSpec] = []

    for checkpoint_dir in (
        "0_seed_42",
        "1_seed_40",
        "2_seed_38",
        "3_seed_36",
        "4_seed_34",
        "5_seed_32",
        "6_seed_30",
        "7_seed_28",
    ):
        specs.append(
            ArtifactSpec(
                module="M1_quality",
                kind="checkpoint",
                path=(
                    "runtime/M1_Retinal_Image_quality_EyePACS/Retinal_quality/"
                    "EyePACS_quality/efficientnet/"
                    f"{checkpoint_dir}/best_loss_checkpoint.pth"
                ),
            )
        )

    for seed in range(24, 43, 2):
        specs.append(
            ArtifactSpec(
                module="M2_vessel",
                kind="checkpoint",
                path=(
                    "runtime/M2_Vessel_seg/Saved_model/train_on_ALL-SIX/"
                    "20210630_uniform_thres40_ALL-SIX_savebest_randomseed_"
                    f"{seed}/G_best_F1_epoch.pth"
                ),
            )
        )

    for seed in range(28, 43, 2):
        checkpoint_dir = (
            "runtime/M2_Artery_vein/ALL-AV/"
            f"20210724_ALL-AV_randomseed_{seed}/Discriminator_unet"
        )
        for filename in (
            "CP_best_F1_all.pth",
            "CP_best_F1_A.pth",
            "CP_best_F1_V.pth",
        ):
            specs.append(
                ArtifactSpec(
                    module="M2_artery_vein",
                    kind="checkpoint",
                    path=f"{checkpoint_dir}/{filename}",
                )
            )

    for seed in range(28, 43, 2):
        specs.append(
            ArtifactSpec(
                module="M2_disc_cup",
                kind="checkpoint",
                path=(
                    "runtime/M2_lwnet_disc_cup/experiments/"
                    f"wnet_All_three_1024_disc_cup/{seed}/model_checkpoint.pth"
                ),
            )
        )

    result = tuple(specs)
    if len(result) != EXPECTED_CHECKPOINT_COUNT:
        raise AssertionError(
            f"internal checkpoint allow-list has {len(result)} entries; "
            f"expected {EXPECTED_CHECKPOINT_COUNT}"
        )
    return result


CHECKPOINT_SPECS = _checkpoint_specs()

SUPPORT_ASSET_SPECS = (
    ArtifactSpec("M2_vessel", "model_source", "runtime/M2_Vessel_seg/model.py"),
    ArtifactSpec(
        "M2_artery_vein",
        "model_source",
        "runtime/M2_Artery_vein/scripts/model.py",
    ),
    ArtifactSpec(
        "M2_disc_cup",
        "model_source",
        "runtime/M2_lwnet_disc_cup/models/get_model.py",
    ),
    ArtifactSpec(
        "M2_disc_cup",
        "model_source",
        "runtime/M2_lwnet_disc_cup/models/res_unet_adrian.py",
    ),
    ArtifactSpec(
        "M3_features",
        "configuration",
        "runtime/M3_feature_whole_pic/retipy/resources/retipy.config",
    ),
)

CHECKPOINT_SEARCH_ROOTS = (
    "runtime/M1_Retinal_Image_quality_EyePACS",
    "runtime/M2_Vessel_seg",
    "runtime/M2_Artery_vein",
    "runtime/M2_lwnet_disc_cup",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _found_checkpoint_paths(root: Path) -> set[str]:
    found: set[str] = set()
    for relative_root in CHECKPOINT_SEARCH_ROOTS:
        search_root = root / relative_root
        if not search_root.is_dir():
            raise FileNotFoundError(f"required model directory is missing: {search_root}")
        found.update(
            _relative_posix(path, root)
            for path in search_root.rglob("*.pth")
            if path.is_file()
        )
    return found


def _format_path_error(label: str, paths: Iterable[str]) -> str:
    ordered = sorted(paths)
    preview = "\n".join(f"  - {path}" for path in ordered[:10])
    suffix = "" if len(ordered) <= 10 else f"\n  ... and {len(ordered) - 10} more"
    return f"{label} ({len(ordered)}):\n{preview}{suffix}"


def build_manifest(root: Path) -> dict[str, object]:
    root = root.expanduser().resolve()
    expected_checkpoints = {spec.path for spec in CHECKPOINT_SPECS}
    found_checkpoints = _found_checkpoint_paths(root)
    missing = expected_checkpoints - found_checkpoints
    unexpected = found_checkpoints - expected_checkpoints
    errors: list[str] = []
    if missing:
        errors.append(_format_path_error("missing checkpoints", missing))
    if unexpected:
        errors.append(_format_path_error("unexpected checkpoints", unexpected))
    if len(found_checkpoints) != EXPECTED_CHECKPOINT_COUNT:
        errors.append(
            "checkpoint count mismatch: "
            f"found {len(found_checkpoints)}, expected {EXPECTED_CHECKPOINT_COUNT}"
        )

    all_specs = CHECKPOINT_SPECS + SUPPORT_ASSET_SPECS
    for spec in all_specs:
        path = root / Path(spec.path)
        if not path.is_file():
            errors.append(f"missing required {spec.kind}:\n  - {spec.path}")
    if errors:
        raise FileNotFoundError("release inventory is incomplete:\n" + "\n".join(errors))

    files: list[dict[str, object]] = []
    for spec in all_specs:
        path = root / Path(spec.path)
        files.append(
            {
                "module": spec.module,
                "kind": spec.kind,
                "path": spec.path,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    checkpoint_bytes = sum(
        int(item["bytes"]) for item in files if item["kind"] == "checkpoint"
    )
    return {
        "schema_version": 1,
        "description": "AutoMorph v2 required model and model-support inventory",
        "generated_by": "scripts/build_model_manifest.py",
        "path_base": "release_root",
        "checkpoint_count": EXPECTED_CHECKPOINT_COUNT,
        "support_asset_count": len(SUPPORT_ASSET_SPECS),
        "total_checkpoint_bytes": checkpoint_bytes,
        "total_manifested_bytes": sum(int(item["bytes"]) for item in files),
        "distribution_notice": (
            "Checksums identify tested artifacts; they do not grant checkpoint "
            "redistribution rights. See THIRD_PARTY_NOTICES.md."
        ),
        "files": files,
    }


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify the AutoMorph v2 model manifest"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="release root (default: parent of this script directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/manifest.json"),
        help="manifest path, relative to --root unless absolute",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the existing manifest instead of replacing it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    generated = build_manifest(root)

    if args.check:
        if not output.is_file():
            raise FileNotFoundError(f"manifest does not exist: {output}")
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != generated:
            raise RuntimeError(
                f"manifest is stale or does not match release artifacts: {output}"
            )
        action = "verified"
    else:
        _write_json_atomic(output, generated)
        action = "wrote"

    print(
        f"{action} {output}: "
        f"{generated['checkpoint_count']} checkpoints, "
        f"{generated['support_asset_count']} support assets, "
        f"{generated['total_manifested_bytes']} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
