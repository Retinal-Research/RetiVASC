"""Cross-platform paths and structural preflight for AutoMorph M3 v2."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd


Centre = Literal["disc", "macular"]
Region = Literal["whole", "b", "c"]
VesselType = Literal["binary", "artery", "vein"]

VESSEL_TYPES: tuple[VesselType, ...] = ("binary", "artery", "vein")

REQUIRED_DISC_COLUMNS: frozenset[str] = frozenset(
    {
        "Name",
        "Disc_height",
        "Disc_width",
        "Cup_height",
        "Cup_width",
        "CDR_vertical",
        "CDR_horizontal",
    }
)


@dataclass(frozen=True)
class MeasurementGroup:
    centre: Centre
    region: Region

    @property
    def centre_directory(self) -> str:
        return "Disc_centred" if self.centre == "disc" else "Macular_centred"

    @property
    def output_prefix(self) -> str:
        return "Disc" if self.centre == "disc" else "Macular"

    @property
    def input_stem(self) -> str:
        if self.centre == "disc":
            stems = {
                "whole": "disc_centred",
                "b": "Zone_B_disc_centred",
                "c": "Zone_C_disc_centred",
            }
        else:
            stems = {
                "whole": "macular_centred",
                "b": "macular_Zone_B_centred",
                "c": "macular_Zone_C_centred",
            }
        return stems[self.region]

    @property
    def output_name(self) -> str:
        suffix = {"whole": "", "b": "_Zone_B", "c": "_Zone_C"}[self.region]
        return f"{self.output_prefix}{suffix}_Measurement.csv"

    @property
    def is_zone(self) -> bool:
        return self.region != "whole"


# Preserve the public legacy launch order for progress and intermediate files.
LEGACY_GROUP_ORDER: tuple[MeasurementGroup, ...] = (
    MeasurementGroup("disc", "b"),
    MeasurementGroup("disc", "c"),
    MeasurementGroup("macular", "b"),
    MeasurementGroup("macular", "c"),
    MeasurementGroup("macular", "whole"),
    MeasurementGroup("disc", "whole"),
)


@dataclass(frozen=True)
class InputPair:
    name: str
    skeleton: Path
    process: Path


def input_directories(
    data_root: Path,
    group: MeasurementGroup,
    vessel_type: VesselType,
) -> tuple[Path, Path]:
    branch = "binary_vessel" if vessel_type == "binary" else "artery_vein"
    base = data_root / "Results" / "M2" / branch
    skeleton = base / f"{group.input_stem}_{vessel_type}_skeleton"
    process = base / f"{group.input_stem}_{vessel_type}_process"
    return skeleton, process


def _png_names(directory: Path) -> list[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"required M3 input directory is missing: {directory}")
    return sorted(path.name for path in directory.glob("*.png") if path.is_file())


def _gradable_image_names(data_root: Path) -> set[str]:
    """Return the set of image basenames marked Gradable in M1_quality_final.csv."""
    csv_path = data_root / "Results" / "M1" / "M1_quality_final.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"M1 quality CSV not found: {csv_path}")
    names: set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = pd.read_csv(fh)
        for _, row in reader.iterrows():
            gradable = str(row.get("Gradable", "")).strip().lower()
            if gradable in ("true", "1", "yes", "y"):
                names.add(os.path.basename(str(row.get("Name", ""))))
    return names


def discover_pairs(
    data_root: Path,
    group: MeasurementGroup,
    vessel_type: VesselType,
) -> list[InputPair]:
    skeleton_dir, process_dir = input_directories(data_root, group, vessel_type)
    gradable = _gradable_image_names(data_root)
    skeleton_names = [n for n in _png_names(skeleton_dir) if n in gradable]
    process_names = [n for n in _png_names(process_dir) if n in gradable]
    if skeleton_names != process_names:
        missing_process = sorted(set(skeleton_names) - set(process_names))
        missing_skeleton = sorted(set(process_names) - set(skeleton_names))
        raise ValueError(
            f"M3 skeleton/process mismatch for {group}/{vessel_type}: "
            f"missing process={missing_process[:5]}, "
            f"missing skeleton={missing_skeleton[:5]}"
        )
    return [
        InputPair(name, skeleton_dir / name, process_dir / name)
        for name in skeleton_names
    ]


def preflight(data_root: str | Path) -> dict[str, object]:
    """Validate every M3 input before any output directory is created."""

    root = Path(data_root).expanduser().resolve()
    crop_path = root / "Results" / "M0" / "crop_info.csv"
    if not crop_path.is_file():
        raise FileNotFoundError(f"required crop metadata is missing: {crop_path}")
    crop = pd.read_csv(crop_path)
    required_crop = {"Name", "Scale_resolution"}
    missing_columns = sorted(required_crop - set(crop.columns))
    if missing_columns:
        raise ValueError(f"crop_info.csv is missing columns: {missing_columns}")
    if crop["Name"].duplicated().any():
        duplicates = crop.loc[crop["Name"].duplicated(), "Name"].tolist()
        raise ValueError(f"crop_info.csv contains duplicate names: {duplicates[:5]}")
    numeric_resolutions = pd.to_numeric(crop["Scale_resolution"], errors="coerce")
    invalid_resolution = ~np.isfinite(numeric_resolutions) | (numeric_resolutions <= 0)
    if invalid_resolution.any():
        invalid_names = crop.loc[invalid_resolution, "Name"].astype(str).tolist()
        raise ValueError(
            "crop_info.csv contains non-finite or non-positive Scale_resolution "
            f"values for: {invalid_names[:5]}"
        )
    crop_names = set(crop["Name"].astype(str))

    group_counts: dict[str, int] = {}
    group_metadata_rows: dict[str, int] = {}
    group_metadata_only_rows: dict[str, int] = {}
    all_names: set[str] = set()
    for group in LEGACY_GROUP_ORDER:
        names_by_type: dict[str, list[str]] = {}
        for vessel_type in VESSEL_TYPES:
            pairs = discover_pairs(root, group, vessel_type)
            names_by_type[vessel_type] = [pair.name for pair in pairs]
        reference_names = names_by_type["binary"]
        for vessel_type in ("artery", "vein"):
            if names_by_type[vessel_type] != reference_names:
                raise ValueError(
                    f"M3 vessel-name mismatch for {group}: "
                    f"binary={len(reference_names)}, "
                    f"{vessel_type}={len(names_by_type[vessel_type])}"
                )

        centre_dir = root / "Results" / "M3" / group.centre_directory
        disc_path = centre_dir / "Disc_cup_results.csv"
        if not disc_path.is_file():
            raise FileNotFoundError(f"required disc/cup table is missing: {disc_path}")
        disc_frame = pd.read_csv(disc_path)
        missing_disc_columns = sorted(REQUIRED_DISC_COLUMNS - set(disc_frame.columns))
        if missing_disc_columns:
            raise ValueError(
                f"disc/cup table is missing columns {missing_disc_columns}: {disc_path}"
            )
        if disc_frame["Name"].duplicated().any():
            duplicates = disc_frame.loc[
                disc_frame["Name"].duplicated(), "Name"
            ].astype(str).tolist()
            raise ValueError(
                f"disc/cup table contains duplicate names {duplicates[:5]}: {disc_path}"
            )
        disc_names = sorted(
            n
            for n in disc_frame["Name"].astype(str)
            if n in _gradable_image_names(root)
        )
        key = f"{group.centre}:{group.region}"
        if group.is_zone:
            # Legacy M2 deliberately records invalid disc/cup detections in the
            # macular metadata table and copies their whole-image masks, but it
            # cannot construct Zone B/C masks for them. Legacy M3 measures the
            # masks that exist and outer-merges those rows with disc metadata,
            # leaving the unavailable zone metrics blank. Preserve that
            # contract while still rejecting an orphan mask with no metadata.
            orphan_masks = sorted(set(reference_names) - set(disc_names))
            if orphan_masks:
                raise ValueError(
                    f"M3 zone masks have no disc/cup metadata for {group}: "
                    f"orphan masks={orphan_masks[:5]}"
                )
        elif disc_names != reference_names:
            raise ValueError(
                f"M3 disc/cup names do not match masks for {group}: "
                f"disc rows={len(disc_names)}, masks={len(reference_names)}"
            )

        missing_crop = sorted(set(reference_names) - crop_names)
        if missing_crop:
            raise ValueError(
                f"crop_info.csv has no resolution for {group}: {missing_crop[:5]}"
            )
        group_counts[key] = len(reference_names)
        group_metadata_rows[key] = len(disc_names)
        group_metadata_only_rows[key] = len(set(disc_names) - set(reference_names))
        all_names.update(reference_names)

    return {
        "data_root": str(root),
        "crop_info": str(crop_path),
        "centred_image_count": len(all_names),
        "group_counts": group_counts,
        "group_metadata_rows": group_metadata_rows,
        "group_metadata_only_rows": group_metadata_only_rows,
        "retipy_evaluations": sum(group_counts.values()) * len(VESSEL_TYPES),
    }
