"""Strict, cross-platform comparison of legacy and v2 M3 CSV outputs.

The legacy acceptance oracle archives its six intermediate tables in a flat
``oracle_artifacts/intermediate_csv`` directory because the original merge
script deletes them.  The two public feature tables remain in ``Results/M3``.
This module understands that layout and compares it with the normal nested v2
``Results/M3`` layout without loading values through floating-point libraries.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4


@dataclass(frozen=True)
class TableSpec:
    """One required M3 output and its canonical candidate-relative path."""

    key: str
    kind: str
    filename: str
    candidate_relative: Path


TABLE_SPECS: tuple[TableSpec, ...] = (
    TableSpec(
        "disc_whole",
        "intermediate",
        "Disc_Measurement.csv",
        Path("Disc_centred/Disc_Measurement.csv"),
    ),
    TableSpec(
        "disc_zone_b",
        "intermediate",
        "Disc_Zone_B_Measurement.csv",
        Path("Disc_centred/Disc_Zone_B_Measurement.csv"),
    ),
    TableSpec(
        "disc_zone_c",
        "intermediate",
        "Disc_Zone_C_Measurement.csv",
        Path("Disc_centred/Disc_Zone_C_Measurement.csv"),
    ),
    TableSpec(
        "macular_whole",
        "intermediate",
        "Macular_Measurement.csv",
        Path("Macular_centred/Macular_Measurement.csv"),
    ),
    TableSpec(
        "macular_zone_b",
        "intermediate",
        "Macular_Zone_B_Measurement.csv",
        Path("Macular_centred/Macular_Zone_B_Measurement.csv"),
    ),
    TableSpec(
        "macular_zone_c",
        "intermediate",
        "Macular_Zone_C_Measurement.csv",
        Path("Macular_centred/Macular_Zone_C_Measurement.csv"),
    ),
    TableSpec(
        "disc_features",
        "final",
        "Disc_Features.csv",
        Path("Disc_Features.csv"),
    ),
    TableSpec(
        "macular_features",
        "final",
        "Macular_Features.csv",
        Path("Macular_Features.csv"),
    ),
)


@dataclass(frozen=True)
class CsvDocument:
    path: Path
    sha256: str
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    @property
    def malformed_rows(self) -> int:
        return sum(len(row) != len(self.header) for row in self.rows)


def _absolute(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _first_existing(candidates: Sequence[Path]) -> Path:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve(strict=False)
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return next((path for path in ordered if path.is_file()), ordered[0])


def _normal_m3_candidates(root: Path, relative: Path) -> list[Path]:
    candidates: list[Path] = []
    if root.name.casefold() == "m3":
        candidates.append(root / relative)
    candidates.extend(
        (
            root / "Results" / "M3" / relative,
            root / "M3" / relative,
            root / relative,
        )
    )
    return candidates


def resolve_candidate_path(root: str | Path, spec: TableSpec) -> Path:
    """Resolve a v2 data root, Results root, or Results/M3 root."""

    root = _absolute(root)
    return _first_existing(_normal_m3_candidates(root, spec.candidate_relative))


def resolve_oracle_path(root: str | Path, spec: TableSpec) -> Path:
    """Resolve both an oracle run root and an ``oracle_artifacts`` root."""

    root = _absolute(root)
    candidates: list[Path] = []
    if spec.kind == "intermediate":
        candidates.extend(
            (
                root / "oracle_artifacts" / "intermediate_csv" / spec.filename,
                root / "intermediate_csv" / spec.filename,
            )
        )
        if root.name.casefold() == "intermediate_csv":
            candidates.append(root / spec.filename)
    else:
        candidates.extend(
            (
                root / "oracle_artifacts" / "final_csv" / spec.filename,
                root / "final_csv" / spec.filename,
            )
        )

    candidates.extend(_normal_m3_candidates(root, spec.candidate_relative))
    if root.name.casefold() == "oracle_artifacts":
        candidates.extend(
            _normal_m3_candidates(root.parent, spec.candidate_relative)
        )
    elif root.name.casefold() == "intermediate_csv" and root.parent.name.casefold() == "oracle_artifacts":
        candidates.extend(
            _normal_m3_candidates(root.parent.parent, spec.candidate_relative)
        )
    return _first_existing(candidates)


def _read_csv(path: Path) -> CsvDocument:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        matrix = [tuple(row) for row in csv.reader(stream, strict=True)]
    header = matrix[0] if matrix else ()
    return CsvDocument(path, _sha256(path), header, tuple(matrix[1:]))


def _names(document: CsvDocument) -> tuple[str | None, ...] | None:
    if document.header.count("Name") != 1:
        return None
    index = document.header.index("Name")
    return tuple(row[index] if index < len(row) else None for row in document.rows)


def _finite_decimal(value: str) -> Decimal | None:
    try:
        number = Decimal(value.strip())
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def _matrix_mismatch_count(
    oracle: Sequence[Sequence[str]], candidate: Sequence[Sequence[str]]
) -> int:
    missing = object()
    count = 0
    for row_index in range(max(len(oracle), len(candidate))):
        left = oracle[row_index] if row_index < len(oracle) else ()
        right = candidate[row_index] if row_index < len(candidate) else ()
        for column_index in range(max(len(left), len(right))):
            left_value: object = left[column_index] if column_index < len(left) else missing
            right_value: object = right[column_index] if column_index < len(right) else missing
            count += left_value != right_value
    return count


def _blank_mismatch_count(oracle: CsvDocument, candidate: CsvDocument) -> int:
    missing = object()
    count = 0
    for row_index in range(max(len(oracle.rows), len(candidate.rows))):
        left = oracle.rows[row_index] if row_index < len(oracle.rows) else ()
        right = candidate.rows[row_index] if row_index < len(candidate.rows) else ()
        for column_index in range(max(len(left), len(right))):
            left_value: object = left[column_index] if column_index < len(left) else missing
            right_value: object = right[column_index] if column_index < len(right) else missing
            if left_value is missing or right_value is missing:
                count += 1
            else:
                count += (left_value == "") != (right_value == "")
    return count


def _name_mismatch_count(
    oracle: tuple[str | None, ...] | None,
    candidate: tuple[str | None, ...] | None,
) -> int:
    if oracle is None or candidate is None:
        return 1
    missing = object()
    return sum(
        (oracle[index] if index < len(oracle) else missing)
        != (candidate[index] if index < len(candidate) else missing)
        for index in range(max(len(oracle), len(candidate)))
    )


def _display_cell(row: Sequence[str], index: int) -> tuple[str | None, bool]:
    if index >= len(row):
        return None, True
    return row[index], False


def _column_label(
    oracle: CsvDocument, candidate: CsvDocument, index: int
) -> str | None:
    if index < len(oracle.header):
        return oracle.header[index]
    if index < len(candidate.header):
        return candidate.header[index]
    return None


def _append_example(
    examples: list[dict[str, Any]],
    limit: int,
    *,
    kind: str,
    row: int | None = None,
    column_index: int | None = None,
    column: str | None = None,
    oracle: str | None = None,
    candidate: str | None = None,
    oracle_missing: bool = False,
    candidate_missing: bool = False,
    abs_delta: str | None = None,
) -> None:
    if len(examples) >= limit:
        return
    examples.append(
        {
            "kind": kind,
            "row": row,
            "csv_line": None if row is None else row + 1,
            "column_index": column_index,
            "column": column,
            "oracle": oracle,
            "candidate": candidate,
            "oracle_missing": oracle_missing,
            "candidate_missing": candidate_missing,
            "abs_delta": abs_delta,
        }
    )


def _unreadable_result(
    spec: TableSpec,
    oracle_path: Path,
    candidate_path: Path,
    oracle_error: str | None,
    candidate_error: str | None,
    max_examples: int,
) -> dict[str, Any]:
    oracle_present = oracle_path.is_file()
    candidate_present = candidate_path.is_file()
    examples: list[dict[str, Any]] = []
    if not oracle_present:
        _append_example(
            examples,
            max_examples,
            kind="missing_file",
            oracle=str(oracle_path),
            oracle_missing=True,
        )
    if not candidate_present:
        _append_example(
            examples,
            max_examples,
            kind="missing_file",
            candidate=str(candidate_path),
            candidate_missing=True,
        )
    if oracle_error:
        _append_example(
            examples, max_examples, kind="csv_read_error", oracle=oracle_error
        )
    if candidate_error:
        _append_example(
            examples, max_examples, kind="csv_read_error", candidate=candidate_error
        )

    false_fields = {
        "schema_set_exact": False,
        "column_order_exact": False,
        "row_count_exact": False,
        "name_set_exact": False,
        "name_order_exact": False,
        "row_order_exact": False,
        "blank_mask_exact": False,
        "cell_strings_exact": False,
        "decimal_numeric_exact": False,
        "byte_exact": False,
        "contract_exact": False,
        "strict_exact": False,
    }
    return {
        "key": spec.key,
        "kind": spec.kind,
        "filename": spec.filename,
        "oracle_path": str(oracle_path),
        "candidate_path": str(candidate_path),
        "oracle_present": oracle_present,
        "candidate_present": candidate_present,
        "oracle_readable": oracle_error is None and oracle_present,
        "candidate_readable": candidate_error is None and candidate_present,
        "oracle_error": oracle_error,
        "candidate_error": candidate_error,
        "oracle_sha256": _sha256(oracle_path) if oracle_present else None,
        "candidate_sha256": _sha256(candidate_path) if candidate_present else None,
        "oracle_rows": None,
        "candidate_rows": None,
        "oracle_columns": None,
        "candidate_columns": None,
        "oracle_malformed_rows": None,
        "candidate_malformed_rows": None,
        "name_mismatch_count": None,
        "blank_mask_mismatch_count": None,
        "emitted_cell_mismatch_count": None,
        "decimal_mismatch_count": None,
        "numeric_parse_mismatch_count": None,
        "numeric_cells_compared": 0,
        "max_abs_delta": None,
        "mismatch_examples": examples,
        **false_fields,
    }


def compare_table(
    spec: TableSpec,
    oracle_path: Path,
    candidate_path: Path,
    *,
    max_examples: int = 25,
) -> dict[str, Any]:
    """Compare one CSV without binary floating-point conversion."""

    if max_examples < 0:
        raise ValueError("max_examples must be non-negative")
    oracle_path = _absolute(oracle_path)
    candidate_path = _absolute(candidate_path)
    oracle_document: CsvDocument | None = None
    candidate_document: CsvDocument | None = None
    oracle_error: str | None = None
    candidate_error: str | None = None
    if oracle_path.is_file():
        try:
            oracle_document = _read_csv(oracle_path)
        except (OSError, UnicodeError, csv.Error) as error:
            oracle_error = f"{type(error).__name__}: {error}"
    if candidate_path.is_file():
        try:
            candidate_document = _read_csv(candidate_path)
        except (OSError, UnicodeError, csv.Error) as error:
            candidate_error = f"{type(error).__name__}: {error}"
    if oracle_document is None or candidate_document is None:
        return _unreadable_result(
            spec,
            oracle_path,
            candidate_path,
            oracle_error,
            candidate_error,
            max_examples,
        )

    oracle = oracle_document
    candidate = candidate_document
    schema_set_exact = Counter(oracle.header) == Counter(candidate.header)
    column_order_exact = oracle.header == candidate.header
    row_count_exact = len(oracle.rows) == len(candidate.rows)
    oracle_names = _names(oracle)
    candidate_names = _names(candidate)
    name_set_exact = bool(
        oracle_names is not None
        and candidate_names is not None
        and Counter(oracle_names) == Counter(candidate_names)
    )
    name_order_exact = bool(
        oracle_names is not None
        and candidate_names is not None
        and oracle_names == candidate_names
    )
    blank_mismatches = _blank_mismatch_count(oracle, candidate)
    matrix_oracle = (oracle.header, *oracle.rows)
    matrix_candidate = (candidate.header, *candidate.rows)
    emitted_mismatches = _matrix_mismatch_count(matrix_oracle, matrix_candidate)
    cell_strings_exact = matrix_oracle == matrix_candidate
    structural_exact = bool(
        column_order_exact
        and row_count_exact
        and oracle.malformed_rows == 0
        and candidate.malformed_rows == 0
    )
    blank_mask_exact = structural_exact and blank_mismatches == 0

    examples: list[dict[str, Any]] = []
    for column_index in range(max(len(oracle.header), len(candidate.header))):
        left, left_missing = _display_cell(oracle.header, column_index)
        right, right_missing = _display_cell(candidate.header, column_index)
        if left_missing or right_missing or left != right:
            _append_example(
                examples,
                max_examples,
                kind="column_order",
                column_index=column_index,
                column=left or right,
                oracle=left,
                candidate=right,
                oracle_missing=left_missing,
                candidate_missing=right_missing,
            )

    for row_index in range(max(len(oracle.rows), len(candidate.rows))):
        left_row = oracle.rows[row_index] if row_index < len(oracle.rows) else ()
        right_row = candidate.rows[row_index] if row_index < len(candidate.rows) else ()
        for column_index in range(max(len(left_row), len(right_row))):
            left, left_missing = _display_cell(left_row, column_index)
            right, right_missing = _display_cell(right_row, column_index)
            if left_missing or right_missing or left != right:
                _append_example(
                    examples,
                    max_examples,
                    kind="cell_string",
                    row=row_index + 1,
                    column_index=column_index,
                    column=_column_label(oracle, candidate, column_index),
                    oracle=left,
                    candidate=right,
                    oracle_missing=left_missing,
                    candidate_missing=right_missing,
                )

    name_mismatches = _name_mismatch_count(oracle_names, candidate_names)
    decimal_numeric_exact = structural_exact and oracle.header.count("Name") == 1
    decimal_mismatches = 0
    numeric_parse_mismatches = 0
    numeric_cells_compared = 0
    max_delta = Decimal(0)

    if decimal_numeric_exact:
        name_index = oracle.header.index("Name")
        for row_index, (left_row, right_row) in enumerate(
            zip(oracle.rows, candidate.rows, strict=True), start=1
        ):
            for column_index, (left, right) in enumerate(
                zip(left_row, right_row, strict=True)
            ):
                if column_index == name_index:
                    continue
                if left == "" and right == "":
                    continue
                if (left == "") != (right == ""):
                    decimal_numeric_exact = False
                    numeric_parse_mismatches += 1
                    continue
                left_decimal = _finite_decimal(left)
                right_decimal = _finite_decimal(right)
                if left_decimal is None or right_decimal is None:
                    decimal_numeric_exact = False
                    numeric_parse_mismatches += 1
                    _append_example(
                        examples,
                        max_examples,
                        kind="numeric_parse",
                        row=row_index,
                        column_index=column_index,
                        column=oracle.header[column_index],
                        oracle=left,
                        candidate=right,
                    )
                    continue
                numeric_cells_compared += 1
                delta = abs(left_decimal - right_decimal)
                if delta > max_delta:
                    max_delta = delta
                if left_decimal != right_decimal:
                    decimal_numeric_exact = False
                    decimal_mismatches += 1
                    _append_example(
                        examples,
                        max_examples,
                        kind="decimal_numeric",
                        row=row_index,
                        column_index=column_index,
                        column=oracle.header[column_index],
                        oracle=left,
                        candidate=right,
                        abs_delta=str(delta),
                    )

    contract_exact = bool(
        schema_set_exact
        and column_order_exact
        and row_count_exact
        and name_set_exact
        and name_order_exact
        and blank_mask_exact
    )
    byte_exact = oracle.sha256 == candidate.sha256
    strict_exact = bool(
        contract_exact and cell_strings_exact and decimal_numeric_exact
    )
    return {
        "key": spec.key,
        "kind": spec.kind,
        "filename": spec.filename,
        "oracle_path": str(oracle.path),
        "candidate_path": str(candidate.path),
        "oracle_present": True,
        "candidate_present": True,
        "oracle_readable": True,
        "candidate_readable": True,
        "oracle_error": None,
        "candidate_error": None,
        "oracle_sha256": oracle.sha256,
        "candidate_sha256": candidate.sha256,
        "oracle_rows": len(oracle.rows),
        "candidate_rows": len(candidate.rows),
        "oracle_columns": list(oracle.header),
        "candidate_columns": list(candidate.header),
        "oracle_malformed_rows": oracle.malformed_rows,
        "candidate_malformed_rows": candidate.malformed_rows,
        "schema_set_exact": schema_set_exact,
        "column_order_exact": column_order_exact,
        "row_count_exact": row_count_exact,
        "name_set_exact": name_set_exact,
        "name_order_exact": name_order_exact,
        "row_order_exact": name_order_exact,
        "blank_mask_exact": blank_mask_exact,
        "cell_strings_exact": cell_strings_exact,
        "decimal_numeric_exact": decimal_numeric_exact,
        "byte_exact": byte_exact,
        "contract_exact": contract_exact,
        "strict_exact": strict_exact,
        "name_mismatch_count": name_mismatches,
        "blank_mask_mismatch_count": blank_mismatches,
        "emitted_cell_mismatch_count": emitted_mismatches,
        "decimal_mismatch_count": decimal_mismatches,
        "numeric_parse_mismatch_count": numeric_parse_mismatches,
        "numeric_cells_compared": numeric_cells_compared,
        "max_abs_delta": str(max_delta) if numeric_cells_compared else None,
        "mismatch_examples": examples,
    }


def compare_m3(
    oracle_root: str | Path,
    candidate_root: str | Path,
    *,
    max_examples_per_table: int = 25,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare all six intermediate and both final M3 tables."""

    if max_examples_per_table < 0:
        raise ValueError("max_examples_per_table must be non-negative")
    oracle_root = _absolute(oracle_root)
    candidate_root = _absolute(candidate_root)
    tables = [
        compare_table(
            spec,
            resolve_oracle_path(oracle_root, spec),
            resolve_candidate_path(candidate_root, spec),
            max_examples=max_examples_per_table,
        )
        for spec in TABLE_SPECS
    ]
    examples = [
        {"table": table["key"], **example}
        for table in tables
        for example in table["mismatch_examples"]
    ]
    numeric_deltas = [
        Decimal(table["max_abs_delta"])
        for table in tables
        if table["max_abs_delta"] is not None
    ]
    all_tables_present = all(
        table["oracle_present"] and table["candidate_present"] for table in tables
    )
    all_tables_readable = all(
        table["oracle_readable"] and table["candidate_readable"] for table in tables
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "oracle_root": str(oracle_root),
        "candidate_root": str(candidate_root),
        "tables_expected": len(TABLE_SPECS),
        "tables_compared": sum(
            bool(table["oracle_readable"] and table["candidate_readable"])
            for table in tables
        ),
        "all_tables_present": all_tables_present,
        "all_tables_readable": all_tables_readable,
        "all_contract_exact": all(table["contract_exact"] for table in tables),
        "all_cell_strings_exact": all(
            table["cell_strings_exact"] for table in tables
        ),
        "all_decimal_numeric_exact": all(
            table["decimal_numeric_exact"] for table in tables
        ),
        "all_byte_exact": all(table["byte_exact"] for table in tables),
        "max_abs_delta": str(max(numeric_deltas, default=Decimal(0))),
        "total_name_mismatches": sum(
            int(table["name_mismatch_count"] or 0) for table in tables
        ),
        "total_blank_mask_mismatches": sum(
            int(table["blank_mask_mismatch_count"] or 0) for table in tables
        ),
        "total_emitted_cell_mismatches": sum(
            int(table["emitted_cell_mismatch_count"] or 0) for table in tables
        ),
        "total_decimal_mismatches": sum(
            int(table["decimal_mismatch_count"] or 0) for table in tables
        ),
        "total_numeric_parse_mismatches": sum(
            int(table["numeric_parse_mismatch_count"] or 0) for table in tables
        ),
        "mismatch_examples_emitted": len(examples),
        "max_examples_per_table": max_examples_per_table,
        "tables": tables,
    }
    summary["strict_exact"] = bool(
        all_tables_present
        and all_tables_readable
        and summary["all_contract_exact"]
        and summary["all_cell_strings_exact"]
        and summary["all_decimal_numeric_exact"]
    )
    return summary, tables, examples


def _atomic_text(path: Path, text: str, *, newline: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline=newline) as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _write_rows(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    empty_fields: Sequence[str] = (),
) -> None:
    fields = sorted({key for row in rows for key in row} or set(empty_fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if fields:
                writer.writeheader()
                writer.writerows(
                    {key: _csv_value(value) for key, value in row.items()}
                    for row in rows
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_comparison(
    output_dir: str | Path,
    summary: dict[str, Any],
    tables: Sequence[dict[str, Any]],
    examples: Sequence[dict[str, Any]],
) -> dict[str, Path]:
    """Write deterministic machine-readable acceptance artifacts."""

    output_dir = _absolute(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    table_path = output_dir / "per_table.csv"
    mismatch_path = output_dir / "mismatch_examples.csv"
    _atomic_text(
        summary_path,
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        newline="\n",
    )
    _write_rows(table_path, tables)
    _write_rows(
        mismatch_path,
        examples,
        empty_fields=(
            "table",
            "kind",
            "row",
            "csv_line",
            "column_index",
            "column",
            "oracle",
            "candidate",
            "oracle_missing",
            "candidate_missing",
            "abs_delta",
        ),
    )
    return {
        "summary": summary_path,
        "per_table": table_path,
        "mismatch_examples": mismatch_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly compare all legacy-oracle and v2 M3 CSV outputs"
    )
    parser.add_argument(
        "oracle",
        type=Path,
        help="legacy oracle run root or its oracle_artifacts directory",
    )
    parser.add_argument(
        "candidate",
        type=Path,
        help="v2 data root, Results root, or Results/M3 directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-examples-per-table",
        type=int,
        default=25,
        help="maximum detailed mismatch examples retained for each table",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_examples_per_table < 0:
        raise SystemExit("--max-examples-per-table must be non-negative")
    summary, tables, examples = compare_m3(
        args.oracle,
        args.candidate,
        max_examples_per_table=args.max_examples_per_table,
    )
    write_comparison(args.output_dir, summary, tables, examples)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if summary["strict_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
