#!/usr/bin/env python3
"""Privacy-safe, strict AutoMorph v2 release acceptance gate.

The comparator keeps image identifiers and filesystem roots in memory only.
Persisted artifacts contain aggregate counts, exactness results, timings, and a
small allow-list of environment fields.  Any required mismatch makes the
command return exit status 1 after writing the report artifacts.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np
from PIL import Image


EXPECTED_CANDIDATE_IMAGES = 21
SHIP_EXPECTED_INPUT_IMAGES = 100
SHIP_EXPECTED_STAGE_IMAGES = {
    "m0": 99,
    "m1": 99,
    "m2": 21,
    "m3": 21,
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
M1_HEADER = (
    "Name",
    "softmax_good",
    "softmax_usable",
    "softmax_bad",
    "good_sd",
    "usable_sd",
    "bad_sd",
    "Prediction",
)

M3_INPUT_TABLES = (
    "Disc_centred/Disc_cup_results.csv",
    "Macular_centred/Disc_cup_results.csv",
)
M3_OUTPUT_TABLES = (
    "Disc_centred/Disc_Zone_B_Measurement.csv",
    "Disc_centred/Disc_Zone_C_Measurement.csv",
    "Macular_centred/Macular_Zone_B_Measurement.csv",
    "Macular_centred/Macular_Zone_C_Measurement.csv",
    "Macular_centred/Macular_Measurement.csv",
    "Disc_centred/Disc_Measurement.csv",
    "Disc_Features.csv",
    "Macular_Features.csv",
)

M3_LEGACY_INTERMEDIATE_TABLES = {
    relative: Path(relative).name
    for relative in M3_OUTPUT_TABLES
    if "/" in relative
}

DEFAULT_LEGACY_TIMINGS: Mapping[str, Mapping[str, object]] = {
    "m0": {
        "seconds": 23.230382399866357,
        "images": 100,
        "provenance": (
            "accepted M0 legacy benchmark recorded 2026-09-03; "
            "one completed 100-image CPU run"
        ),
    },
    "m1": {
        "seconds": 24.388,
        "images": 99,
        "provenance": (
            "accepted M1 legacy benchmark recorded 2026-09-05; one completed "
            "99-image Torch 2.7.1+cu118 CUDA FP32 run"
        ),
    },
    "m2": {
        "seconds": 230.767665,
        "images": 21,
        "provenance": (
            "accepted M2 legacy CUDA-compatible benchmark recorded "
            "2026-09-05; one completed 21-image run"
        ),
    },
    "m3": {
        "seconds": 470.685056099901,
        "images": 21,
        "provenance": (
            "accepted M3 legacy serial CPU oracle recorded 2026-09-05; "
            "one completed 21-image run"
        ),
    },
}

SUMMARY_PATHS = {
    "m0": "Results/M0/run_summary.json",
    "m1": "Results/M1/run_summary.json",
    "m2": "Results/M2/run_summary.json",
    "m3": "Results/M3/run_summary.json",
}

MODULE_WALL_FIELDS = {
    "m0": "wall_s",
    "m1": "wall_s",
    "m2": "total_wall_s",
    "m3": "total_wall_s",
}

ENVIRONMENT_FIELDS = {
    "m0": ("python", "platform", "backend"),
    "m1": (
        "python",
        "platform",
        "torch",
        "device_resolved",
        "precision",
        "engine",
    ),
    "m2": (
        "python",
        "platform",
        "torch",
        "device_requested",
        "threads",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
        "contract",
    ),
    "m3": (
        "python",
        "platform",
        "numpy",
        "pandas",
        "opencv",
        "device",
        "workers",
        "start_method",
        "contract",
    ),
}


@dataclass(frozen=True, slots=True)
class Metric:
    stage: str
    check: str
    actual: object
    expected: object
    passed: bool
    required: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "check": self.check,
            "actual": self.actual,
            "expected": self.expected,
            "status": "PASS" if self.passed else "FAIL",
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class CsvData:
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class PixelComparison:
    exact: bool
    changed_pixels: int
    max_abs_delta: int


def _add_metric(
    metrics: list[Metric],
    stage: str,
    check: str,
    actual: object,
    expected: object,
    *,
    passed: bool | None = None,
    required: bool = True,
) -> None:
    metrics.append(
        Metric(
            stage=stage,
            check=check,
            actual=actual,
            expected=expected,
            passed=(actual == expected if passed is None else bool(passed)),
            required=required,
        )
    )


def portable_basename(value: str) -> str:
    """Extract a basename from either POSIX or Windows text on every OS."""

    return value.replace("\\", "/").rsplit("/", 1)[-1]


def sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def compare_decoded_pixels(candidate: Path, oracle: Path) -> PixelComparison:
    """Compare decoded visible pixels independent of the image encoding."""

    try:
        with Image.open(candidate) as candidate_image:
            candidate_pixels = np.asarray(candidate_image.convert("RGBA"))
        with Image.open(oracle) as oracle_image:
            oracle_pixels = np.asarray(oracle_image.convert("RGBA"))
    except (OSError, ValueError):
        return PixelComparison(False, 0, 0)
    if candidate_pixels.shape != oracle_pixels.shape:
        return PixelComparison(False, 0, 0)
    delta = np.abs(
        candidate_pixels.astype(np.int16) - oracle_pixels.astype(np.int16)
    )
    changed = int(np.count_nonzero(np.any(delta != 0, axis=-1)))
    maximum = int(delta.max()) if delta.size else 0
    return PixelComparison(changed == 0, changed, maximum)


def _flat_image_index(directory: Path) -> tuple[dict[str, Path], int, int]:
    if not directory.is_dir():
        return {}, 0, 0
    result: dict[str, Path] = {}
    duplicates = 0
    unsupported = 0
    for path in directory.iterdir():
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            unsupported += 1
            continue
        if path.name in result:
            duplicates += 1
        else:
            result[path.name] = path
    return result, duplicates, unsupported


def _read_csv(path: Path) -> CsvData | None:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = [tuple(row) for row in csv.reader(handle)]
    except (OSError, UnicodeError, csv.Error):
        return None
    if not rows:
        return CsvData((), ())
    return CsvData(rows[0], tuple(rows[1:]))


def _rows_by_basename(
    rows: Sequence[tuple[str, ...]],
) -> tuple[dict[str, tuple[str, ...]], int, int]:
    result: dict[str, tuple[str, ...]] = {}
    duplicates = 0
    malformed = 0
    for row in rows:
        if not row or not row[0]:
            malformed += 1
            continue
        basename = portable_basename(row[0])
        if basename in result:
            duplicates += 1
        else:
            result[basename] = row
    return result, duplicates, malformed


def compare_m0(
    candidate_root: Path,
    oracle_root: Path,
    metrics: list[Metric],
    *,
    expected_images: int = EXPECTED_CANDIDATE_IMAGES,
) -> tuple[dict[str, object], set[str]]:
    candidate_dir = candidate_root / "Results" / "M0" / "images"
    oracle_dir = oracle_root / "Results" / "M0" / "images"
    candidate, candidate_duplicates, candidate_unsupported = _flat_image_index(
        candidate_dir
    )
    oracle, oracle_duplicates, _oracle_unsupported = _flat_image_index(oracle_dir)
    candidate_names = set(candidate)
    covered = candidate_names & set(oracle)

    _add_metric(
        metrics,
        "m0",
        "candidate decoded image count",
        len(candidate),
        expected_images,
    )
    _add_metric(metrics, "m0", "candidate duplicate basenames", candidate_duplicates, 0)
    _add_metric(metrics, "m0", "candidate unsupported visible files", candidate_unsupported, 0)
    _add_metric(metrics, "m0", "oracle duplicate basenames", oracle_duplicates, 0)
    _add_metric(metrics, "m0", "oracle coverage for candidate", len(covered), len(candidate))

    decoded_exact = 0
    changed_pixels = 0
    maximum_delta = 0
    for name in sorted(covered):
        comparison = compare_decoded_pixels(candidate[name], oracle[name])
        decoded_exact += int(comparison.exact)
        changed_pixels += comparison.changed_pixels
        maximum_delta = max(maximum_delta, comparison.max_abs_delta)
    _add_metric(metrics, "m0", "decoded-pixel exact images", decoded_exact, len(candidate))
    _add_metric(metrics, "m0", "changed decoded pixels", changed_pixels, 0)
    _add_metric(metrics, "m0", "maximum decoded pixel delta", maximum_delta, 0)

    candidate_csv = _read_csv(candidate_root / "Results" / "M0" / "crop_info.csv")
    oracle_csv = _read_csv(oracle_root / "Results" / "M0" / "crop_info.csv")
    candidate_header = candidate_csv.header if candidate_csv else ()
    oracle_header = oracle_csv.header if oracle_csv else ()
    candidate_rows = candidate_csv.rows if candidate_csv else ()
    oracle_rows = oracle_csv.rows if oracle_csv else ()
    candidate_index, crop_duplicates, crop_malformed = _rows_by_basename(candidate_rows)
    oracle_index, oracle_crop_duplicates, oracle_crop_malformed = _rows_by_basename(
        oracle_rows
    )
    crop_values_exact = sum(
        candidate_index[name][1:] == oracle_index[name][1:]
        for name in candidate_names & set(candidate_index) & set(oracle_index)
    )

    _add_metric(metrics, "m0", "crop table schema exact", candidate_header, oracle_header)
    _add_metric(metrics, "m0", "candidate crop row count", len(candidate_rows), len(candidate))
    _add_metric(metrics, "m0", "candidate crop duplicate basenames", crop_duplicates, 0)
    _add_metric(metrics, "m0", "candidate malformed crop rows", crop_malformed, 0)
    _add_metric(metrics, "m0", "oracle crop duplicate basenames", oracle_crop_duplicates, 0)
    _add_metric(metrics, "m0", "oracle malformed crop rows", oracle_crop_malformed, 0)
    _add_metric(
        metrics,
        "m0",
        "candidate crop rows cover candidate images",
        len(set(candidate_index) & candidate_names),
        len(candidate),
        passed=set(candidate_index) == candidate_names,
    )
    _add_metric(
        metrics,
        "m0",
        "oracle crop coverage for candidate",
        len(set(oracle_index) & candidate_names),
        len(candidate),
    )
    _add_metric(
        metrics,
        "m0",
        "crop values exact by basename",
        crop_values_exact,
        len(candidate),
    )

    public = {
        "candidate_images": len(candidate),
        "oracle_selected_images": len(covered),
        "decoded_pixel_exact_images": decoded_exact,
        "changed_decoded_pixels": changed_pixels,
        "maximum_decoded_pixel_delta": maximum_delta,
        "crop_rows": len(candidate_rows),
        "crop_rows_exact_by_basename": crop_values_exact == len(candidate),
    }
    return public, candidate_names


def _route_membership(
    m1_root: Path,
) -> tuple[dict[str, tuple[str, Path]], int, int]:
    membership: dict[str, tuple[str, Path]] = {}
    duplicates = 0
    unsupported = 0
    for route, directory_name in (("good", "Good_quality"), ("bad", "Bad_quality")):
        indexed, local_duplicates, local_unsupported = _flat_image_index(
            m1_root / directory_name
        )
        duplicates += local_duplicates
        unsupported += local_unsupported
        for name, path in indexed.items():
            if name in membership:
                duplicates += 1
            else:
                membership[name] = (route, path)
    return membership, duplicates, unsupported


def compare_m1(
    candidate_root: Path,
    oracle_root: Path,
    expected_names: set[str],
    metrics: list[Metric],
    *,
    expected_images: int = EXPECTED_CANDIDATE_IMAGES,
) -> tuple[dict[str, object], set[str], int]:
    candidate_m1 = candidate_root / "Results" / "M1"
    oracle_m1 = oracle_root / "Results" / "M1"
    candidate_csv = _read_csv(candidate_m1 / "results_ensemble.csv")
    oracle_csv = _read_csv(oracle_m1 / "results_ensemble.csv")
    candidate_header = candidate_csv.header if candidate_csv else ()
    oracle_header = oracle_csv.header if oracle_csv else ()
    candidate_rows = candidate_csv.rows if candidate_csv else ()
    oracle_rows = oracle_csv.rows if oracle_csv else ()
    candidate_index, candidate_duplicates, candidate_malformed = _rows_by_basename(
        candidate_rows
    )
    oracle_index, oracle_duplicates, oracle_malformed = _rows_by_basename(oracle_rows)
    candidate_names = set(candidate_index)
    covered = candidate_names & set(oracle_index)

    _add_metric(metrics, "m1", "candidate CSV header", candidate_header, M1_HEADER)
    _add_metric(metrics, "m1", "oracle CSV header", oracle_header, M1_HEADER)
    _add_metric(metrics, "m1", "candidate row count", len(candidate_rows), expected_images)
    _add_metric(metrics, "m1", "candidate duplicate basenames", candidate_duplicates, 0)
    _add_metric(metrics, "m1", "candidate malformed rows", candidate_malformed, 0)
    _add_metric(metrics, "m1", "oracle duplicate basenames", oracle_duplicates, 0)
    _add_metric(metrics, "m1", "oracle malformed rows", oracle_malformed, 0)
    _add_metric(
        metrics,
        "m1",
        "candidate rows match M0 basenames",
        len(candidate_names & expected_names),
        len(expected_names),
        passed=candidate_names == expected_names,
    )
    _add_metric(metrics, "m1", "oracle row coverage for candidate", len(covered), len(candidate_rows))

    compared_fields = 0
    exact_fields = 0
    for name in sorted(covered):
        candidate_row = candidate_index[name]
        oracle_row = oracle_index[name]
        for column in range(1, len(M1_HEADER)):
            compared_fields += 1
            candidate_value = candidate_row[column] if column < len(candidate_row) else None
            oracle_value = oracle_row[column] if column < len(oracle_row) else None
            exact_fields += int(candidate_value == oracle_value)
    expected_field_count = len(candidate_rows) * (len(M1_HEADER) - 1)
    _add_metric(metrics, "m1", "non-path CSV fields compared", compared_fields, expected_field_count)
    _add_metric(metrics, "m1", "non-path CSV fields exact strings", exact_fields, expected_field_count)

    candidate_routes, route_duplicates, route_unsupported = _route_membership(candidate_m1)
    oracle_routes, oracle_route_duplicates, _oracle_route_unsupported = _route_membership(
        oracle_m1
    )
    _add_metric(metrics, "m1", "candidate duplicate route membership", route_duplicates, 0)
    _add_metric(metrics, "m1", "candidate unsupported route files", route_unsupported, 0)
    _add_metric(metrics, "m1", "oracle duplicate route membership", oracle_route_duplicates, 0)
    _add_metric(
        metrics,
        "m1",
        "candidate route set exact",
        len(set(candidate_routes) & candidate_names),
        len(candidate_names),
        passed=set(candidate_routes) == candidate_names,
    )

    route_exact = 0
    copied_sha_exact = 0
    selected_oracle_membership = 0
    for name in sorted(candidate_names):
        candidate_route = candidate_routes.get(name)
        oracle_route = oracle_routes.get(name)
        if oracle_route is not None:
            selected_oracle_membership += 1
        if candidate_route is None or oracle_route is None:
            continue
        if candidate_route[0] == oracle_route[0]:
            route_exact += 1
        candidate_hash = sha256_file(candidate_route[1])
        oracle_hash = sha256_file(oracle_route[1])
        if candidate_hash is not None and candidate_hash == oracle_hash:
            copied_sha_exact += 1
    _add_metric(
        metrics,
        "m1",
        "oracle route coverage for candidate",
        selected_oracle_membership,
        len(candidate_names),
    )
    _add_metric(metrics, "m1", "route membership exact", route_exact, len(candidate_names))
    _add_metric(metrics, "m1", "copied file SHA-256 exact", copied_sha_exact, len(candidate_names))

    good_count = sum(1 for route, _path in candidate_routes.values() if route == "good")
    bad_count = sum(1 for route, _path in candidate_routes.values() if route == "bad")
    public = {
        "candidate_rows": len(candidate_rows),
        "oracle_selected_rows": len(covered),
        "non_path_fields_compared": compared_fields,
        "non_path_fields_exact": exact_fields,
        "route_membership_exact": route_exact,
        "copied_sha256_exact": copied_sha_exact,
        "good_route_count": good_count,
        "bad_route_count": bad_count,
    }
    return public, candidate_names, good_count


def _is_m2_metadata(relative: str, path: Path) -> bool:
    parts = {part.lower() for part in Path(relative).parts}
    return (
        path.name.lower() == "run_summary.json"
        or path.suffix.lower() == ".log"
        or "logs" in parts
        or "log" in parts
    )


def _m2_file_index(data_root: Path) -> dict[str, Path]:
    root = data_root / "Results" / "M2"
    if not root.is_dir():
        return {}
    result: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if not _is_m2_metadata(relative, path):
            result[relative] = path
    return result


def _m2_unique_image_count(files: Mapping[str, Path]) -> int:
    return len(
        {
            Path(relative).name
            for relative, path in files.items()
            if relative.startswith("binary_vessel/raw/")
            and path.suffix.lower() in IMAGE_SUFFIXES
        }
    )


def compare_m2(
    candidate_root: Path,
    oracle_root: Path,
    metrics: list[Metric],
    *,
    expected_images: int = EXPECTED_CANDIDATE_IMAGES,
) -> tuple[dict[str, object], set[str]]:
    candidate = _m2_file_index(candidate_root)
    oracle = _m2_file_index(oracle_root)
    candidate_set = set(candidate)
    oracle_set = set(oracle)
    intersection = candidate_set & oracle_set
    relative_set_exact = candidate_set == oracle_set
    candidate_non_images = sum(
        path.suffix.lower() not in IMAGE_SUFFIXES for path in candidate.values()
    )
    oracle_non_images = sum(
        path.suffix.lower() not in IMAGE_SUFFIXES for path in oracle.values()
    )
    candidate_image_count = _m2_unique_image_count(candidate)
    oracle_image_count = _m2_unique_image_count(oracle)

    _add_metric(metrics, "m2", "candidate deterministic file count", len(candidate), len(oracle))
    _add_metric(metrics, "m2", "deterministic relative file set exact", relative_set_exact, True)
    _add_metric(metrics, "m2", "candidate non-image deterministic files", candidate_non_images, 0)
    _add_metric(metrics, "m2", "oracle non-image deterministic files", oracle_non_images, 0)
    _add_metric(metrics, "m2", "candidate unique image basenames", candidate_image_count, expected_images)
    _add_metric(metrics, "m2", "oracle unique image basenames", oracle_image_count, expected_images)

    sha_exact = 0
    decoded_exact = 0
    changed_pixels = 0
    maximum_delta = 0
    for relative in sorted(intersection):
        candidate_path = candidate[relative]
        oracle_path = oracle[relative]
        candidate_hash = sha256_file(candidate_path)
        oracle_hash = sha256_file(oracle_path)
        sha_exact += int(candidate_hash is not None and candidate_hash == oracle_hash)
        if candidate_path.suffix.lower() in IMAGE_SUFFIXES:
            comparison = compare_decoded_pixels(candidate_path, oracle_path)
            decoded_exact += int(comparison.exact)
            changed_pixels += comparison.changed_pixels
            maximum_delta = max(maximum_delta, comparison.max_abs_delta)
    _add_metric(metrics, "m2", "SHA-256 exact deterministic files", sha_exact, len(oracle))
    _add_metric(metrics, "m2", "decoded-pixel exact deterministic files", decoded_exact, len(oracle))
    _add_metric(metrics, "m2", "changed decoded pixels", changed_pixels, 0)
    _add_metric(metrics, "m2", "maximum decoded pixel delta", maximum_delta, 0)

    candidate_m3 = candidate_root / "Results" / "M3"
    oracle_m3 = oracle_root / "Results" / "M3"
    input_tables_present = 0
    input_tables_byte_exact = 0
    for relative in M3_INPUT_TABLES:
        candidate_path = candidate_m3 / Path(relative)
        oracle_path = oracle_m3 / Path(relative)
        if candidate_path.is_file() and oracle_path.is_file():
            input_tables_present += 1
            try:
                input_tables_byte_exact += int(
                    candidate_path.read_bytes() == oracle_path.read_bytes()
                )
            except OSError:
                pass
    _add_metric(metrics, "m2", "M3 centring input tables present", input_tables_present, len(M3_INPUT_TABLES))
    _add_metric(metrics, "m2", "M3 centring input tables byte exact", input_tables_byte_exact, len(M3_INPUT_TABLES))

    private_names = {path.name for path in candidate.values()} | {
        path.name for path in oracle.values()
    }
    public = {
        "deterministic_files": len(candidate),
        "oracle_deterministic_files": len(oracle),
        "relative_file_set_exact": relative_set_exact,
        "sha256_exact_files": sha_exact,
        "decoded_pixel_exact_files": decoded_exact,
        "changed_decoded_pixels": changed_pixels,
        "maximum_decoded_pixel_delta": maximum_delta,
        "centring_input_tables_byte_exact": input_tables_byte_exact,
        "candidate_image_count": candidate_image_count,
    }
    return public, private_names


def _m3_output_index(data_root: Path) -> dict[str, Path]:
    root = data_root / "Results" / "M3"
    if not root.is_dir():
        return {}
    excluded = set(M3_INPUT_TABLES)
    result: dict[str, Path] = {}
    for path in root.rglob("*.csv"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative not in excluded:
            result[relative] = path
    # The pristine legacy merge moves its six intermediate tables out of
    # Results/M3. Accepted oracle captures preserve them in this read-only
    # archive; map them back to their original relative contract for comparison.
    legacy_intermediate = data_root / "oracle_artifacts" / "intermediate_csv"
    for relative, basename in M3_LEGACY_INTERMEDIATE_TABLES.items():
        archived = legacy_intermediate / basename
        if relative not in result and archived.is_file():
            result[relative] = archived
    return result


def _shape_valid(table: CsvData | None) -> bool:
    if table is None or not table.header:
        return False
    return all(len(row) == len(table.header) for row in table.rows)


def _blank_mask(table: CsvData | None) -> tuple[tuple[bool, ...], ...]:
    if table is None:
        return ()
    return tuple(tuple(value == "" for value in row) for row in table.rows)


def _numeric_max_delta(
    candidate: CsvData | None,
    oracle: CsvData | None,
) -> float:
    if candidate is None or oracle is None:
        return math.inf
    maximum = Decimal(0)
    for candidate_row, oracle_row in zip(candidate.rows, oracle.rows):
        for candidate_value, oracle_value in zip(candidate_row[1:], oracle_row[1:]):
            if candidate_value == "" or oracle_value == "":
                continue
            try:
                candidate_number = Decimal(candidate_value)
                oracle_number = Decimal(oracle_value)
            except InvalidOperation:
                continue
            if not candidate_number.is_finite() or not oracle_number.is_finite():
                continue
            maximum = max(maximum, abs(candidate_number - oracle_number))
    return float(maximum)


def compare_csv_table(candidate_path: Path, oracle_path: Path) -> dict[str, object]:
    candidate = _read_csv(candidate_path)
    oracle = _read_csv(oracle_path)
    candidate_rows = candidate.rows if candidate else ()
    oracle_rows = oracle.rows if oracle else ()
    schema_exact = (
        candidate is not None
        and oracle is not None
        and candidate.header == oracle.header
        and _shape_valid(candidate)
        and _shape_valid(oracle)
    )
    row_order_exact = (
        len(candidate_rows) == len(oracle_rows)
        and tuple(row[0] if row else None for row in candidate_rows)
        == tuple(row[0] if row else None for row in oracle_rows)
    )
    blank_mask_exact = _blank_mask(candidate) == _blank_mask(oracle)
    cell_strings_exact = (
        candidate is not None
        and oracle is not None
        and candidate.header == oracle.header
        and candidate.rows == oracle.rows
    )
    return {
        "schema_exact": schema_exact,
        "row_order_exact": row_order_exact,
        "blank_mask_exact": blank_mask_exact,
        "cell_strings_exact": cell_strings_exact,
        "numeric_max_delta": _numeric_max_delta(candidate, oracle),
        "candidate_rows": len(candidate_rows),
        "oracle_rows": len(oracle_rows),
    }


def compare_m3(
    candidate_root: Path,
    oracle_root: Path,
    metrics: list[Metric],
    *,
    expected_images: int = EXPECTED_CANDIDATE_IMAGES,
) -> tuple[dict[str, object], set[str]]:
    candidate = _m3_output_index(candidate_root)
    oracle = _m3_output_index(oracle_root)
    expected = set(M3_OUTPUT_TABLES)
    candidate_set = set(candidate)
    oracle_set = set(oracle)
    _add_metric(metrics, "m3", "candidate output table set", len(candidate_set), len(expected), passed=candidate_set == expected)
    _add_metric(metrics, "m3", "oracle output table set", len(oracle_set), len(expected), passed=oracle_set == expected)
    _add_metric(metrics, "m3", "candidate/oracle relative table set exact", candidate_set == oracle_set, True)

    schemas = 0
    row_orders = 0
    blank_masks = 0
    cell_strings = 0
    maximum_delta = 0.0
    candidate_rows = 0
    oracle_rows = 0
    private_names: set[str] = set()
    for relative in M3_OUTPUT_TABLES:
        candidate_path = candidate.get(relative)
        oracle_path = oracle.get(relative)
        if candidate_path is None or oracle_path is None:
            maximum_delta = math.inf
            continue
        table_result = compare_csv_table(candidate_path, oracle_path)
        schemas += int(bool(table_result["schema_exact"]))
        row_orders += int(bool(table_result["row_order_exact"]))
        blank_masks += int(bool(table_result["blank_mask_exact"]))
        cell_strings += int(bool(table_result["cell_strings_exact"]))
        maximum_delta = max(maximum_delta, float(table_result["numeric_max_delta"]))
        candidate_rows += int(table_result["candidate_rows"])
        oracle_rows += int(table_result["oracle_rows"])
        candidate_csv = _read_csv(candidate_path)
        oracle_csv = _read_csv(oracle_path)
        for table in (candidate_csv, oracle_csv):
            if table is not None:
                private_names.update(
                    portable_basename(row[0])
                    for row in table.rows
                    if row and row[0]
                )

    _add_metric(metrics, "m3", "schemas exact", schemas, len(M3_OUTPUT_TABLES))
    _add_metric(metrics, "m3", "row/name order exact", row_orders, len(M3_OUTPUT_TABLES))
    _add_metric(metrics, "m3", "blank masks exact", blank_masks, len(M3_OUTPUT_TABLES))
    _add_metric(metrics, "m3", "cell strings exact", cell_strings, len(M3_OUTPUT_TABLES))
    _add_metric(metrics, "m3", "maximum absolute numeric delta", maximum_delta, 0.0)

    candidate_unique = len(
        {
            portable_basename(row[0])
            for path in candidate.values()
            for table in [_read_csv(path)]
            if table is not None
            for row in table.rows
            if row and row[0]
        }
    )
    oracle_unique = len(
        {
            portable_basename(row[0])
            for path in oracle.values()
            for table in [_read_csv(path)]
            if table is not None
            for row in table.rows
            if row and row[0]
        }
    )
    _add_metric(metrics, "m3", "candidate unique image rows", candidate_unique, expected_images)
    _add_metric(metrics, "m3", "oracle unique image rows", oracle_unique, expected_images)

    public = {
        "output_tables": len(candidate_set),
        "relative_table_set_exact": candidate_set == oracle_set == expected,
        "schemas_exact": schemas,
        "row_orders_exact": row_orders,
        "blank_masks_exact": blank_masks,
        "cell_strings_exact": cell_strings,
        "maximum_absolute_numeric_delta": _json_value(maximum_delta),
        "candidate_total_rows_across_tables": candidate_rows,
        "oracle_total_rows_across_tables": oracle_rows,
        "candidate_image_count": candidate_unique,
    }
    return public, private_names


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _finite_nonnegative(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0 else None


def _safe_environment_value(value: object) -> object | None:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def _regular_file(path: Path) -> bool:
    """Return true only for a directly stored regular file."""

    return path.is_file() and not path.is_symlink()


def _path_matches(value: object, expected: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        return Path(value).expanduser().resolve() == expected.expanduser().resolve()
    except (OSError, RuntimeError):
        return False


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return true when either resolved tree contains the other."""

    left = left.expanduser().resolve()
    right = right.expanduser().resolve()
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _strict_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def candidate_pipeline_artifacts(
    candidate_root: Path,
    expected_input_images: int,
    expected_stage_images: Mapping[str, int],
    metrics: list[Metric],
) -> tuple[dict[str, object], set[str]]:
    """Validate unified-run evidence without persisting private details."""

    stages = ("m0", "m1", "m2", "m3")
    images_root = candidate_root / "images"
    source_names = (
        {
            path.name
            for path in images_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        if images_root.is_dir()
        else set()
    )
    _add_metric(
        metrics,
        "release",
        "source input image count",
        len(source_names),
        expected_input_images,
    )

    metadata_root = candidate_root / ".automorph_v2"
    pipeline_path = metadata_root / "pipeline.json"
    run_log_path = candidate_root / "run.log"
    pipeline = _read_json_object(pipeline_path) if _regular_file(pipeline_path) else None
    config_value = pipeline.get("config") if pipeline else None
    config = config_value if isinstance(config_value, dict) else None
    run_id = pipeline.get("run_id") if pipeline else None
    refs_value = pipeline.get("stage_manifests") if pipeline else None
    stage_manifest_refs = refs_value if isinstance(refs_value, dict) else None
    progress_mode = config.get("progress") if config else None
    pipeline_ok = bool(
        pipeline
        and pipeline.get("schema_version") == 1
        and pipeline.get("status") == "completed"
        and isinstance(run_id, str)
        and re.fullmatch(r"[0-9a-f]{32}", run_id)
        and pipeline.get("selected_stages") == list(stages)
        and pipeline.get("completed_stages") == list(stages)
        and pipeline.get("skipped_stages") == []
        and config
        and _path_matches(config.get("data_root"), candidate_root)
        and config.get("from_stage") == "m0"
        and config.get("to_stage") == "m3"
        and progress_mode in {"auto", "bar", "plain", "off"}
        and _path_matches(pipeline.get("run_log"), run_log_path)
        and stage_manifest_refs
        and set(stage_manifest_refs) == set(stages)
        and all(
            _path_matches(
                stage_manifest_refs.get(stage),
                metadata_root / "stages" / f"{stage}.json",
            )
            for stage in stages
        )
    )
    _add_metric(metrics, "release", "pipeline manifest contract", pipeline_ok, True)

    completed_stage_manifests = 0
    for stage in stages:
        manifest_path = metadata_root / "stages" / f"{stage}.json"
        manifest = (
            _read_json_object(manifest_path) if _regular_file(manifest_path) else None
        )
        if bool(
            manifest
            and manifest.get("schema_version") == 1
            and manifest.get("stage") == stage
            and manifest.get("status") == "completed"
            and manifest.get("return_code") == 0
            and _path_matches(manifest.get("run_log"), run_log_path)
        ):
            completed_stage_manifests += 1
    _add_metric(
        metrics,
        "release",
        "completed stage manifest set",
        completed_stage_manifests,
        len(stages),
    )

    m0_summary_path = candidate_root / Path(SUMMARY_PATHS["m0"])
    m0_summary = (
        _read_json_object(m0_summary_path) if _regular_file(m0_summary_path) else None
    )
    m0_image_count = (
        _strict_nonnegative_int(m0_summary.get("image_count")) if m0_summary else None
    )
    m0_processed = (
        _strict_nonnegative_int(m0_summary.get("processed_count")) if m0_summary else None
    )
    m0_skipped = (
        _strict_nonnegative_int(m0_summary.get("skipped_count")) if m0_summary else None
    )
    m0_failures = (
        _strict_nonnegative_int(m0_summary.get("failure_count")) if m0_summary else None
    )
    m0_accounting_ok = bool(
        m0_image_count == expected_input_images
        and m0_processed == expected_stage_images["m0"]
        and m0_skipped is not None
        and m0_failures is not None
        and m0_processed + m0_skipped + m0_failures == m0_image_count
    )
    _add_metric(metrics, "release", "M0 source accounting", m0_accounting_ok, True)
    _add_metric(metrics, "release", "M0 skipped source images", m0_skipped, 0)
    _add_metric(
        metrics,
        "release",
        "M0 recorded source failures",
        m0_failures,
        expected_input_images - expected_stage_images["m0"],
    )

    log_text: str | None = None
    if _regular_file(run_log_path):
        try:
            log_text = run_log_path.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeError):
            log_text = None
    parsed_log: list[tuple[str, str]] = []
    log_lines = log_text.split("\n") if log_text is not None else []
    if log_lines and log_lines[-1] == "":
        log_lines.pop()
    log_pattern = re.compile(r"^\S+ \[([A-Z0-9][A-Z0-9 _-]*)\] (.+)$")
    for line in log_lines:
        match = log_pattern.fullmatch(line)
        if match:
            parsed_log.append((match.group(1), match.group(2)))
    run_id_lines = [
        message.removeprefix("run_id: ")
        for label, message in parsed_log
        if label == "RUN" and message.startswith("run_id: ")
    ]
    log_structure_ok = bool(
        log_text
        and log_lines
        and len(parsed_log) == len(log_lines)
        and "\x1b" not in log_text
        and "\r" not in log_text
        and "AUTOMORPH_PROGRESS" not in log_text
        and sum(
            label == "RUN" and message == "AutoMorph v2 run started"
            for label, message in parsed_log
        )
        == 1
        and len(run_id_lines) == 1
        and run_id_lines[0] == run_id
        and parsed_log[-1][0] == "DONE"
        and parsed_log[-1][1].startswith("run completed:")
        and not any(
            re.search(r"\b(?:ERROR|INTERRUPTED)\b", label)
            for label, _message in parsed_log
        )
    )
    _add_metric(metrics, "release", "complete root run.log contract", log_structure_ok, True)

    lifecycle_ok = all(
        any(
            label == stage.upper()
            and message == f"AutoMorph v2 [{stage}] starting"
            for label, message in parsed_log
        )
        and any(
            label == stage.upper()
            and message == f"AutoMorph v2 [{stage}] complete"
            for label, message in parsed_log
        )
        for stage in stages
    )
    _add_metric(metrics, "release", "run.log stage lifecycle coverage", lifecycle_ok, True)
    progress_counts = {
        stage: sum(label == f"{stage.upper()} PROGRESS" for label, _ in parsed_log)
        for stage in stages
    }
    progress_ok = (
        all(value == 0 for value in progress_counts.values())
        if progress_mode == "off"
        else progress_mode in {"auto", "bar", "plain"}
        and all(value > 0 for value in progress_counts.values())
    )
    _add_metric(metrics, "release", "run.log progress coverage", progress_ok, True)

    resolution_path = candidate_root / "resolution_information.csv"
    resolution_names: set[str] = set()
    resolution_rows = 0
    resolution_csv_ok = False
    if _regular_file(resolution_path):
        try:
            with resolution_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
            header_ok = bool(rows) and tuple(rows[0]) == ("fundus", "res")
            data_rows = rows[1:] if rows else []
            values_ok = True
            duplicate = False
            for row in data_rows:
                if len(row) != 2:
                    values_ok = False
                    continue
                name, raw_value = row
                if name in resolution_names:
                    duplicate = True
                resolution_names.add(name)
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    values_ok = False
                else:
                    values_ok = values_ok and math.isfinite(value) and value > 0
            resolution_rows = len(data_rows)
            resolution_csv_ok = bool(
                header_ok
                and values_ok
                and not duplicate
                and resolution_names == source_names
                and resolution_rows == expected_input_images
            )
        except (OSError, UnicodeError, csv.Error):
            resolution_csv_ok = False
    _add_metric(
        metrics,
        "release",
        "resolution table covers source inputs",
        resolution_csv_ok,
        True,
    )

    provenance_path = metadata_root / "resolution_provenance.json"
    provenance = (
        _read_json_object(provenance_path) if _regular_file(provenance_path) else None
    )
    source = provenance.get("source") if provenance else None
    auto_generated = provenance.get("auto_generated") if provenance else None
    allowed_source_mapping = {
        "legacy_uniform_default": True,
        "uniform_config_value": True,
        "provided": False,
    }
    provenance_trusted = bool(
        provenance
        and provenance.get("schema_version") == 1
        and source in allowed_source_mapping
        and auto_generated is allowed_source_mapping.get(source)
        and _path_matches(provenance.get("path"), resolution_path)
    )
    _add_metric(metrics, "release", "trusted resolution provenance", provenance_trusted, True)
    expected_count_field = "image_count" if auto_generated is True else "row_count"
    csv_hash = sha256_file(resolution_path) if _regular_file(resolution_path) else None
    provenance_csv_ok = bool(
        provenance_trusted
        and provenance.get("csv_sha256") == csv_hash
        and provenance.get(expected_count_field) == expected_input_images
        and resolution_rows == expected_input_images
    )
    _add_metric(metrics, "release", "resolution provenance matches CSV", provenance_csv_ok, True)
    embedded_ok = bool(
        pipeline
        and provenance
        and pipeline.get("resolution_information") == provenance
    )
    _add_metric(metrics, "release", "pipeline embeds resolution provenance", embedded_ok, True)
    calibrated_required = (
        config.get("require_calibrated_resolution") if config else None
    )
    calibration_ok = bool(
        isinstance(calibrated_required, bool)
        and (not calibrated_required or auto_generated is False)
    )
    _add_metric(metrics, "release", "calibrated-resolution policy honored", calibration_ok, True)

    public = {
        "source_input_image_count": len(source_names),
        "completed_stage_manifests": completed_stage_manifests,
        "m0_processed_count": m0_processed,
        "m0_skipped_count": m0_skipped,
        "m0_failure_count": m0_failures,
        "run_log_lines": len(log_lines),
        "run_log_progress_events": sum(progress_counts.values()),
        "resolution_rows": resolution_rows,
        "resolution_source": source if source in allowed_source_mapping else "untrusted",
        "resolution_auto_generated": (
            auto_generated if isinstance(auto_generated, bool) else None
        ),
    }
    return public, source_names


def candidate_metadata(
    candidate_root: Path,
    metrics: list[Metric],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    timings: dict[str, dict[str, object]] = {}
    environment: dict[str, dict[str, object]] = {}
    for stage in ("m0", "m1", "m2", "m3"):
        manifest = _read_json_object(
            candidate_root / ".automorph_v2" / "stages" / f"{stage}.json"
        )
        summary = _read_json_object(candidate_root / Path(SUMMARY_PATHS[stage]))
        manifest_ok = (
            manifest is not None
            and manifest.get("stage") == stage
            and manifest.get("status") == "completed"
        )
        summary_ok = summary is not None
        _add_metric(metrics, "release", f"{stage} completed stage manifest", manifest_ok, True)
        _add_metric(metrics, "release", f"{stage} run summary readable", summary_ok, True)

        stage_wall = _finite_nonnegative(manifest.get("wall_s")) if manifest else None
        module_wall = (
            _finite_nonnegative(summary.get(MODULE_WALL_FIELDS[stage]))
            if summary
            else None
        )
        _add_metric(
            metrics,
            "release",
            f"{stage} stage wall time recorded",
            stage_wall is not None,
            True,
        )
        _add_metric(
            metrics,
            "release",
            f"{stage} module wall time recorded",
            module_wall is not None,
            True,
        )
        timings[stage] = {
            "stage_wall_s": stage_wall,
            "module_wall_s": module_wall,
            "stage_timing_source": "candidate completed stage manifest",
            "module_timing_source": "candidate module run summary",
        }
        stage_environment: dict[str, object] = {}
        if summary:
            for field in ENVIRONMENT_FIELDS[stage]:
                if field in summary:
                    safe = _safe_environment_value(summary[field])
                    if safe is not None:
                        stage_environment[field] = safe
        environment[stage] = stage_environment

    m3_summary = _read_json_object(candidate_root / Path(SUMMARY_PATHS["m3"]))
    m3_metric_failures = (
        m3_summary.get("metric_failures") if m3_summary is not None else None
    )
    _add_metric(metrics, "release", "M3 metric failures", m3_metric_failures, 0)
    return timings, environment


def legacy_timings_from_args(
    args: argparse.Namespace,
    candidate_samples: Mapping[str, int],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for stage in ("m0", "m1", "m2", "m3"):
        default = DEFAULT_LEGACY_TIMINGS[stage]
        seconds_override = getattr(args, f"legacy_{stage}_seconds")
        images_override = getattr(args, f"legacy_{stage}_images")
        overridden = seconds_override is not None or images_override is not None
        seconds = float(
            seconds_override if seconds_override is not None else default["seconds"]
        )
        images = int(images_override if images_override is not None else default["images"])
        candidate_images = int(candidate_samples[stage])
        normalized = seconds * candidate_images / images
        result[stage] = {
            "accepted_wall_s": seconds,
            "accepted_sample_images": images,
            "candidate_sample_images": candidate_images,
            "candidate_normalized_legacy_s": normalized,
            "normalization": (
                "same-sample measured wall time"
                if images == candidate_images
                else "linear per-image projection; not a same-sample rerun"
            ),
            "provenance": (
                "operator CLI override; not independently verified"
                if overridden
                else str(default["provenance"])
            ),
        }
    return result


def _stage_verdict(metrics: Sequence[Metric], stage: str) -> bool:
    selected = [metric for metric in metrics if metric.stage == stage and metric.required]
    return bool(selected) and all(metric.passed for metric in selected)


def _json_value(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sanitized_metric(metric: Metric) -> dict[str, object]:
    value = metric.as_dict()
    value["actual"] = _json_value(value["actual"])
    value["expected"] = _json_value(value["expected"])
    return value


def _timing_summary(
    candidate_timings: Mapping[str, Mapping[str, object]],
    legacy_timings: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    stages: dict[str, dict[str, object]] = {}
    candidate_total = 0.0
    legacy_total = 0.0
    complete = True
    for stage in ("m0", "m1", "m2", "m3"):
        candidate_wall = candidate_timings[stage].get("stage_wall_s")
        legacy_wall = float(legacy_timings[stage]["candidate_normalized_legacy_s"])
        if candidate_wall is None:
            complete = False
            speedup = None
        else:
            candidate_value = float(candidate_wall)
            candidate_total += candidate_value
            speedup = legacy_wall / candidate_value if candidate_value else None
        legacy_total += legacy_wall
        stages[stage] = {
            **candidate_timings[stage],
            **legacy_timings[stage],
            "speedup_vs_candidate_normalized_legacy": speedup,
        }
    return {
        "stages": stages,
        "candidate_total_stage_wall_s": candidate_total if complete else None,
        "candidate_normalized_legacy_total_s": legacy_total,
        "overall_speedup": (
            legacy_total / candidate_total if complete and candidate_total else None
        ),
    }


def _privacy_guard(
    texts: Iterable[str],
    roots: Sequence[Path],
    private_names: Iterable[str],
) -> None:
    combined = "\n".join(texts)
    root_tokens: set[str] = set()
    for root in roots:
        value = str(root.expanduser().resolve())
        root_tokens.update({value, value.replace("\\", "/"), value.replace("/", "\\")})
    leaked_roots = [token for token in root_tokens if token and token in combined]
    leaked_names = [
        name
        for name in set(private_names)
        if len(name) >= 3 and name in combined
    ]
    if leaked_roots or leaked_names:
        raise RuntimeError("privacy guard rejected identifying input text")
    if re.search(r"(?i)(?:[a-z]:[\\/]|\\\\[^\\\s]+[\\/])", combined):
        raise RuntimeError("privacy guard rejected an absolute Windows path")


def _format_number(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _make_report(
    summary: Mapping[str, object],
    metrics: Sequence[Metric],
) -> str:
    verdict = str(summary["verdict"])
    checks = summary["checks"]
    timing = summary["timing"]
    environment = summary["environment"]
    assert isinstance(checks, Mapping)
    assert isinstance(timing, Mapping)
    assert isinstance(environment, Mapping)
    timing_stages = timing["stages"]
    assert isinstance(timing_stages, Mapping)

    expected_stages = summary["expected_stage_image_counts"]
    observed_stages = summary["observed_stage_image_counts"]
    assert isinstance(expected_stages, Mapping)
    assert isinstance(observed_stages, Mapping)
    lines = [
        "# AutoMorph v2 strict release acceptance",
        "",
        "## Outcome",
        "",
        f"**{verdict}** — all persisted evidence is aggregate-only and privacy-safe.",
        "",
        f"The release contract starts from {summary['expected_input_image_count']} "
        "source images and expects strict stage outputs "
        f"M0={expected_stages['m0']}, M1={expected_stages['m1']}, "
        f"M2={expected_stages['m2']}, M3={expected_stages['m3']}. "
        "Observed stage outputs were "
        f"M0={observed_stages['m0']}, M1={observed_stages['m1']}, "
        f"M2={observed_stages['m2']}, M3={observed_stages['m3']}. "
        "A source that M0 cannot decode is recorded and does not enter M1; "
        "images routed Bad_quality by M1 do not enter M2 or M3.",
        "",
        "| Stage | Required checks passed | Status |",
        "|---|---:|---|",
    ]
    for stage in ("m0", "m1", "m2", "m3", "release"):
        stage_metrics = [m for m in metrics if m.stage == stage and m.required]
        passed = sum(metric.passed for metric in stage_metrics)
        lines.append(
            f"| {stage.upper()} | {passed}/{len(stage_metrics)} | "
            f"{'PASS' if passed == len(stage_metrics) and stage_metrics else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "## Exactness gate",
            "",
            "| Stage | Check | Actual | Expected | Status |",
            "|---|---|---:|---:|---|",
        ]
    )
    for metric in metrics:
        if metric.stage == "release":
            continue
        lines.append(
            f"| {metric.stage.upper()} | {metric.check} | "
            f"{_format_number(_json_value(metric.actual), 6)} | "
            f"{_format_number(_json_value(metric.expected), 6)} | "
            f"{'PASS' if metric.passed else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "## Unified-run artifact gate",
            "",
            "| Check | Actual | Expected | Status |",
            "|---|---:|---:|---|",
        ]
    )
    for metric in metrics:
        if metric.stage != "release":
            continue
        lines.append(
            f"| {metric.check} | "
            f"{_format_number(_json_value(metric.actual), 6)} | "
            f"{_format_number(_json_value(metric.expected), 6)} | "
            f"{'PASS' if metric.passed else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "## Performance",
            "",
            "![Accepted legacy and candidate runtime](performance.png)",
            "",
            "| Stage | Accepted legacy sample | Legacy normalized to candidate | Candidate stage wall | Speedup |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for stage in ("m0", "m1", "m2", "m3"):
        item = timing_stages[stage]
        assert isinstance(item, Mapping)
        lines.append(
            f"| {stage.upper()} | {_format_number(item['accepted_wall_s'])} s / "
            f"{item['accepted_sample_images']} images | "
            f"{_format_number(item['candidate_normalized_legacy_s'])} s | "
            f"{_format_number(item['stage_wall_s'])} s | "
            f"{_format_number(item['speedup_vs_candidate_normalized_legacy'], 2)}x |"
        )
    lines.extend(
        [
            "",
            "Each timing row declares both its accepted legacy sample size and "
            "whether normalization was required for the candidate cohort. "
            "Timing does not affect the strict output verdict.",
            "",
            "### Timing provenance",
            "",
        ]
    )
    for stage in ("m0", "m1", "m2", "m3"):
        item = timing_stages[stage]
        assert isinstance(item, Mapping)
        lines.append(f"- {stage.upper()}: {item['provenance']}; {item['normalization']}.")

    lines.extend(
        [
            "",
            "## Candidate environment evidence",
            "",
            "Only allow-listed, non-path fields from module summaries are retained.",
            "",
            "| Stage | Python | Platform | Device/backend | Torch |",
            "|---|---|---|---|---|",
        ]
    )
    for stage in ("m0", "m1", "m2", "m3"):
        item = environment.get(stage, {})
        assert isinstance(item, Mapping)
        device = item.get("device_resolved", item.get("device_requested", item.get("device", item.get("backend", "n/a"))))
        lines.append(
            f"| {stage.upper()} | {item.get('python', 'n/a')} | "
            f"{item.get('platform', 'n/a')} | {device} | "
            f"{item.get('torch', 'n/a')} |"
        )

    lines.extend(
        [
            "",
            "## Machine-readable evidence",
            "",
            "- `summary.json`: aggregate verdict, exactness, timing, and environment",
            "- `metrics.csv`: one aggregate acceptance check per row",
            "- `performance.png`: accepted legacy reference versus candidate runtime",
            "",
            "No input root, absolute path, per-image row, image basename, or "
            "mismatch example is persisted by this report.",
            "",
        ]
    )
    return "\n".join(lines)


def _metrics_csv(metrics: Sequence[Metric]) -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("stage", "check", "actual", "expected", "status", "required"))
    for metric in metrics:
        writer.writerow(
            (
                metric.stage,
                metric.check,
                _format_number(_json_value(metric.actual), 9),
                _format_number(_json_value(metric.expected), 9),
                "PASS" if metric.passed else "FAIL",
                "yes" if metric.required else "no",
            )
        )
    return stream.getvalue()


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _performance_chart(
    path: Path,
    timing: Mapping[str, object],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = ("m0", "m1", "m2", "m3")
    timing_stages = timing["stages"]
    assert isinstance(timing_stages, Mapping)
    legacy = [
        float(timing_stages[stage]["candidate_normalized_legacy_s"])  # type: ignore[index]
        for stage in stages
    ]
    candidate = [
        float(timing_stages[stage]["stage_wall_s"] or 0.0)  # type: ignore[index]
        for stage in stages
    ]
    legacy_total = sum(legacy)
    candidate_total = sum(candidate)
    overall_speedup = legacy_total / candidate_total if candidate_total else None
    total_caption = (
        f"{overall_speedup:.2f}x faster"
        if overall_speedup is not None
        else "candidate timing incomplete"
    )

    # Give every stage its own linear scale. M0 and M1 are otherwise visually
    # flattened by the much longer M2/M3 legacy runtimes, while a shared log
    # scale makes substantial gains look deceptively small.
    figure, axes = plt.subplots(
        1,
        len(stages),
        figsize=(18.0, 5.4),
        constrained_layout=True,
    )
    figure.suptitle(
        "AutoMorph v2 performance — wall time (lower is better)\n"
        f"Total: {legacy_total:.2f} s → {candidate_total:.2f} s  |  "
        + total_caption,
        fontsize=16,
        fontweight="semibold",
    )

    legacy_color = "#808080"
    candidate_color = "#2a9d8f"
    for axis, stage, legacy_s, candidate_s in zip(
        axes, stages, legacy, candidate
    ):
        speedup = legacy_s / candidate_s if candidate_s else None
        bars = axis.bar(
            (0, 1),
            (legacy_s, candidate_s),
            width=0.72,
            color=(legacy_color, candidate_color),
        )
        axis.bar_label(
            bars,
            labels=(f"{legacy_s:.2f} s", f"{candidate_s:.2f} s"),
            padding=4,
            fontsize=10,
            fontweight="semibold",
        )
        axis.set_title(
            f"{stage.upper()}\n"
            + (f"{speedup:.2f}x faster" if speedup is not None else "timing unavailable"),
            fontsize=13,
            fontweight="semibold",
        )
        axis.set_xticks((0, 1), ("Legacy", "AutoMorph v2"))
        axis.set_ylabel("Seconds")
        axis.set_ylim(0.0, max(legacy_s, candidate_s, 1.0) * 1.20)
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", labelsize=10)
        axis.tick_params(axis="y", labelsize=9)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.tmp.png")
    try:
        figure.savefig(temporary, dpi=160)
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def run_acceptance(args: argparse.Namespace) -> dict[str, object]:
    candidate_root = args.candidate.expanduser().resolve()
    m0_oracle = args.m0_oracle.expanduser().resolve()
    m1_oracle = args.m1_oracle.expanduser().resolve()
    m2_oracle = args.m2_oracle.expanduser().resolve()
    m3_oracle = args.m3_oracle.expanduser().resolve()
    output = args.output.expanduser().resolve()
    roots = (candidate_root, m0_oracle, m1_oracle, m2_oracle, m3_oracle)
    for oracle in roots[1:]:
        if _paths_overlap(candidate_root, oracle):
            raise ValueError("candidate and oracle roots must be disjoint")
    if any(_paths_overlap(output, root) for root in roots):
        raise ValueError("acceptance output must be disjoint from candidate and oracles")
    expected_stage_images = {
        stage: int(
            getattr(
                args,
                f"expected_{stage}_images",
                SHIP_EXPECTED_STAGE_IMAGES[stage],
            )
        )
        for stage in ("m0", "m1", "m2", "m3")
    }
    expected_input_images = int(
        getattr(args, "expected_input_images", SHIP_EXPECTED_INPUT_IMAGES)
    )

    metrics: list[Metric] = []
    m0, m0_names = compare_m0(
        candidate_root,
        m0_oracle,
        metrics,
        expected_images=expected_stage_images["m0"],
    )
    m1, m1_names, m1_good_count = compare_m1(
        candidate_root,
        m1_oracle,
        m0_names,
        metrics,
        expected_images=expected_stage_images["m1"],
    )
    m2, m2_private_names = compare_m2(
        candidate_root,
        m2_oracle,
        metrics,
        expected_images=expected_stage_images["m2"],
    )
    m3, m3_private_names = compare_m3(
        candidate_root,
        m3_oracle,
        metrics,
        expected_images=expected_stage_images["m3"],
    )
    pipeline_artifacts, source_private_names = candidate_pipeline_artifacts(
        candidate_root,
        expected_input_images,
        expected_stage_images,
        metrics,
    )
    _add_metric(
        metrics,
        "release",
        "M1 good-quality handoff to M2 and M3",
        bool(
            m1_good_count == expected_stage_images["m2"]
            and expected_stage_images["m2"] == expected_stage_images["m3"]
        ),
        True,
    )
    candidate_timings, environment = candidate_metadata(candidate_root, metrics)

    candidate_timing_samples = {
        "m0": expected_input_images,
        "m1": int(m1["candidate_rows"]),
        "m2": int(m2["candidate_image_count"]),
        "m3": int(m3["candidate_image_count"]),
    }
    observed_stage_images = {
        "m0": int(m0["candidate_images"]),
        "m1": int(m1["candidate_rows"]),
        "m2": int(m2["candidate_image_count"]),
        "m3": int(m3["candidate_image_count"]),
    }
    legacy_timings = legacy_timings_from_args(args, candidate_timing_samples)
    timing = _timing_summary(candidate_timings, legacy_timings)
    required_metrics = [metric for metric in metrics if metric.required]
    verdict = "PASS" if required_metrics and all(m.passed for m in required_metrics) else "FAIL"
    checks = {
        stage: {
            "passed": _stage_verdict(metrics, stage),
            "required_checks": sum(
                metric.required and metric.stage == stage for metric in metrics
            ),
            "passed_required_checks": sum(
                metric.required and metric.stage == stage and metric.passed
                for metric in metrics
            ),
        }
        for stage in ("m0", "m1", "m2", "m3", "release")
    }
    summary: dict[str, object] = {
        "schema_version": 1,
        "verdict": verdict,
        "privacy_contract": "aggregate-only; no input roots or image identifiers",
        "expected_input_image_count": expected_input_images,
        "expected_stage_image_counts": expected_stage_images,
        "expected_candidate_image_count": expected_stage_images["m0"],
        "observed_candidate_image_count": int(m0["candidate_images"]),
        "observed_stage_image_counts": observed_stage_images,
        "checks": checks,
        "exactness": {"m0": m0, "m1": m1, "m2": m2, "m3": m3},
        "pipeline_artifacts": pipeline_artifacts,
        "timing": timing,
        "environment": environment,
        "metrics": [_sanitized_metric(metric) for metric in metrics],
        "artifacts": ("report.md", "summary.json", "metrics.csv", "performance.png"),
    }
    report = _make_report(summary, metrics)
    summary_text = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    metrics_text = _metrics_csv(metrics)
    private_names = (
        set(m0_names)
        | set(m1_names)
        | set(m2_private_names)
        | set(m3_private_names)
        | set(source_private_names)
    )
    _privacy_guard((report, summary_text, metrics_text), roots, private_names)

    _write_text_atomic(output / "report.md", report)
    _write_text_atomic(output / "summary.json", summary_text)
    _write_text_atomic(output / "metrics.csv", metrics_text)
    _performance_chart(output / "performance.png", timing)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict, privacy-safe AutoMorph v2 release acceptance gate"
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--m0-oracle", type=Path, required=True)
    parser.add_argument("--m1-oracle", type=Path, required=True)
    parser.add_argument("--m2-oracle", type=Path, required=True)
    parser.add_argument("--m3-oracle", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/acceptance/current"),
        help="aggregate-only report directory (default: docs/acceptance/current)",
    )
    parser.add_argument(
        "--expected-input-images",
        type=int,
        default=SHIP_EXPECTED_INPUT_IMAGES,
        help="expected source image count before M0",
    )
    for stage in ("m0", "m1", "m2", "m3"):
        parser.add_argument(
            f"--expected-{stage}-images",
            dest=f"expected_{stage}_images",
            type=int,
            default=SHIP_EXPECTED_STAGE_IMAGES[stage],
            help=f"expected strict output image count for {stage.upper()}",
        )
    for stage in ("m0", "m1", "m2", "m3"):
        parser.add_argument(
            f"--legacy-{stage}-seconds",
            dest=f"legacy_{stage}_seconds",
            type=float,
            help="override the accepted legacy wall time for this stage",
        )
        parser.add_argument(
            f"--legacy-{stage}-images",
            dest=f"legacy_{stage}_images",
            type=int,
            help="sample size associated with the legacy wall-time override",
        )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.expected_input_images <= 0:
        parser.error("--expected-input-images must be positive")
    for stage in ("m0", "m1", "m2", "m3"):
        if getattr(args, f"expected_{stage}_images") <= 0:
            parser.error(f"--expected-{stage}-images must be positive")
        seconds = getattr(args, f"legacy_{stage}_seconds")
        images = getattr(args, f"legacy_{stage}_images")
        if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
            parser.error(f"--legacy-{stage}-seconds must be finite and positive")
        if images is not None and images <= 0:
            parser.error(f"--legacy-{stage}-images must be positive")
    if args.expected_m0_images > args.expected_input_images:
        parser.error("--expected-m0-images cannot exceed --expected-input-images")
    if args.expected_m1_images != args.expected_m0_images:
        parser.error("--expected-m1-images must equal --expected-m0-images")
    if args.expected_m2_images > args.expected_m1_images:
        parser.error("--expected-m2-images cannot exceed --expected-m1-images")
    if args.expected_m3_images != args.expected_m2_images:
        parser.error("--expected-m3-images must equal --expected-m2-images")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_acceptance(args)
    except Exception as error:
        print(f"Release acceptance failed safely: {type(error).__name__}")
        return 1
    checks = summary["metrics"]
    assert isinstance(checks, list)
    required = [item for item in checks if item["required"]]
    passed = sum(item["status"] == "PASS" for item in required)
    print(
        f"Release acceptance: {summary['verdict']} "
        f"({passed}/{len(required)} required aggregate checks passed)"
    )
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
