from __future__ import annotations

import csv
from pathlib import Path

import pytest

from v2.M3.contract import (
    LEGACY_GROUP_ORDER,
    VESSEL_TYPES,
    MeasurementGroup,
    input_directories,
    preflight,
)
from v2.M3.engine import EvaluationResult, _measurement_frame
from v2.M3.metrics import MetricResult


def _write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _build_inputs(root: Path) -> None:
    disc_name = "disc.png"
    macular_name = "macular.png"
    invalid_name = "invalid-disc-cup.png"
    for group in LEGACY_GROUP_ORDER:
        if group.centre == "disc":
            names = (disc_name,)
        elif group.region == "whole":
            names = (macular_name, invalid_name)
        else:
            names = (macular_name,)
        for vessel_type in VESSEL_TYPES:
            skeleton, process = input_directories(root, group, vessel_type)
            skeleton.mkdir(parents=True, exist_ok=True)
            process.mkdir(parents=True, exist_ok=True)
            for name in names:
                (skeleton / name).write_bytes(b"fixture")
                (process / name).write_bytes(b"fixture")

    columns = (
        "Name",
        "Disc_height",
        "Disc_width",
        "Cup_height",
        "Cup_width",
        "CDR_vertical",
        "CDR_horizontal",
    )
    _write_csv(
        root / "Results" / "M3" / "Disc_centred" / "Disc_cup_results.csv",
        columns,
        [(disc_name, 1, 1, 0.5, 0.5, 0.5, 0.5)],
    )
    _write_csv(
        root / "Results" / "M3" / "Macular_centred" / "Disc_cup_results.csv",
        columns,
        [
            (macular_name, 1, 1, 0.5, 0.5, 0.5, 0.5),
            (invalid_name, -1, -1, -1, -1, -1, -1),
        ],
    )
    _write_csv(
        root / "Results" / "M0" / "crop_info.csv",
        ("Name", "Scale_resolution"),
        [(disc_name, 8), (macular_name, 8), (invalid_name, 8)],
    )


def test_preflight_allows_legacy_metadata_only_zone_rows(tmp_path: Path) -> None:
    _build_inputs(tmp_path)
    result = preflight(tmp_path)
    assert result["group_counts"] == {
        "disc:b": 1,
        "disc:c": 1,
        "macular:b": 1,
        "macular:c": 1,
        "macular:whole": 2,
        "disc:whole": 1,
    }
    assert result["group_metadata_only_rows"] == {
        "disc:b": 0,
        "disc:c": 0,
        "macular:b": 1,
        "macular:c": 1,
        "macular:whole": 0,
        "disc:whole": 0,
    }
    assert result["retipy_evaluations"] == 21


def test_preflight_rejects_zone_mask_without_disc_metadata(tmp_path: Path) -> None:
    _build_inputs(tmp_path)
    group = MeasurementGroup("macular", "b")
    for vessel_type in VESSEL_TYPES:
        skeleton, process = input_directories(tmp_path, group, vessel_type)
        (skeleton / "orphan.png").write_bytes(b"fixture")
        (process / "orphan.png").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="zone masks have no disc/cup metadata"):
        preflight(tmp_path)


def test_zone_measurement_retains_metadata_only_row_with_blank_metrics(
    tmp_path: Path,
) -> None:
    _build_inputs(tmp_path)
    group = MeasurementGroup("macular", "b")
    indexed = {}
    for index, vessel_type in enumerate(VESSEL_TYPES):
        metrics = MetricResult(
            fractal_dimension=1.0,
            vessel_density=2.0,
            average_width=3.0,
            distance_tortuosity=4.0,
            squared_curvature_tortuosity=5.0,
            tortuosity_density=6.0,
            crae_hubbard=7.0 if vessel_type == "artery" else -1,
            crae_knudtson=8.0 if vessel_type == "artery" else -1,
            crve_hubbard=14.0 if vessel_type == "vein" else -1,
            crve_knudtson=16.0 if vessel_type == "vein" else -1,
        )
        indexed[("macular", "b", vessel_type, "macular.png")] = EvaluationResult(
            index=index,
            centre="macular",
            region="b",
            vessel_type=vessel_type,
            name="macular.png",
            metrics=metrics,
            wall_s=0.0,
        )

    frame = _measurement_frame(tmp_path, group, indexed)

    assert len(frame) == 2
    assert set(frame["Name"]) == {"macular.png", "invalid-disc-cup.png"}
    measured = frame.loc[frame["Name"] == "macular.png"].iloc[0]
    metadata_only = frame.loc[frame["Name"] == "invalid-disc-cup.png"].iloc[0]
    assert measured["AVR_Hubbard"] == 0.5
    assert measured["AVR_Knudtson"] == 0.5
    metric_columns = [column for column in frame.columns if column not in {
        "Name", "Disc_height", "Disc_width", "Cup_height", "Cup_width",
        "CDR_vertical", "CDR_horizontal",
    }]
    assert metadata_only[metric_columns].isna().all()
