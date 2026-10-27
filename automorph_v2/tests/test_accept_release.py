from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "accept_release.py"
SPEC = importlib.util.spec_from_file_location("_automorph_accept_release", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
accept_release = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = accept_release
SPEC.loader.exec_module(accept_release)


NAMES = tuple(f"private-subject-{index:02d}.png" for index in range(21))
EXTRA_NAME = "private-oracle-extra.png"


def _pixel_value(name: str) -> int:
    return sum(name.encode("utf-8")) % 251


def _write_png(path: Path, name: str, *, adjustment: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = (_pixel_value(name) + adjustment) % 256
    array = np.full((3, 4, 3), value, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def _write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _populate_root(root: Path, names: tuple[str, ...]) -> None:
    m0_images = root / "Results" / "M0" / "images"
    for name in names:
        _write_png(m0_images / name, name)
        source = root / "images" / name
        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(m0_images / name, source)
    _write_csv(
        root / "Results" / "M0" / "crop_info.csv",
        ("Name", "centre_w", "centre_h", "radius", "Scale", "Scale_resolution"),
        [
            (
                name,
                str(_pixel_value(name)),
                str(_pixel_value(name) + 1),
                "9",
                "2.0",
                "16.0",
            )
            for name in names
        ],
    )

    m1_root = root / "Results" / "M1"
    _write_csv(
        m1_root / "results_ensemble.csv",
        accept_release.M1_HEADER,
        [
            (
                str(root / "Results" / "M0" / "images" / name),
                "0.1",
                "0.8",
                "0.1",
                "0.01",
                "0.02",
                "0.03",
                "1",
            )
            for name in names
        ],
    )
    for name in names:
        destination = m1_root / "Good_quality" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(m0_images / name, destination)
    (m1_root / "Bad_quality").mkdir(parents=True, exist_ok=True)

    for name in names:
        _write_png(root / "Results" / "M2" / "binary_vessel" / "raw" / name, name)
    (root / "Results" / "M2" / "run_summary.json").write_text(
        '{"ignored": true}\n', encoding="utf-8"
    )

    m3_root = root / "Results" / "M3"
    centre_rows = [(name, "1", "2") for name in names]
    for relative in accept_release.M3_INPUT_TABLES:
        _write_csv(m3_root / Path(relative), ("Name", "x", "y"), centre_rows)
    for table_index, relative in enumerate(accept_release.M3_OUTPUT_TABLES):
        _write_csv(
            m3_root / Path(relative),
            ("Name", "metric", "optional"),
            [
                (name, f"{table_index}.{index}", "" if index % 2 else "1")
                for index, name in enumerate(names)
            ],
        )


def _candidate_metadata(
    root: Path,
    *,
    source_names: tuple[str, ...] = NAMES,
    processed_count: int | None = None,
    failure_count: int = 0,
) -> None:
    run_log = root / "run.log"
    processed = len(source_names) if processed_count is None else processed_count
    summaries = {
        "m0": {
            "wall_s": 1.0,
            "image_count": len(source_names),
            "processed_count": processed,
            "skipped_count": 0,
            "failure_count": failure_count,
            "python": "3.12.7",
            "platform": "SyntheticOS",
            "backend": "cpu",
        },
        "m1": {
            "wall_s": 2.0,
            "python": "3.12.7",
            "platform": "SyntheticOS",
            "torch": "2.7.1",
            "device_resolved": "cuda",
            "precision": "fp32",
            "engine": "optimized",
        },
        "m2": {
            "total_wall_s": 3.0,
            "python": "3.12.7",
            "platform": "SyntheticOS",
            "torch": "2.7.1",
            "device_requested": "cuda",
            "threads": 4,
            "peak_cuda_allocated_bytes": 100,
            "peak_cuda_reserved_bytes": 200,
            "contract": "strict-fp32-legacy-output",
        },
        "m3": {
            "total_wall_s": 4.0,
            "python": "3.12.7",
            "platform": "SyntheticOS",
            "numpy": "1.26.4",
            "pandas": "2.3.3",
            "opencv": "4.12.0",
            "device": "cpu",
            "workers": 4,
            "start_method": "spawn",
            "contract": "legacy-73-column-strict-cpu",
            "metric_failures": 0,
        },
    }
    for stage, summary in summaries.items():
        summary_path = root / Path(accept_release.SUMMARY_PATHS[stage])
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
        manifest_path = root / ".automorph_v2" / "stages" / f"{stage}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage": stage,
                    "status": "completed",
                    "return_code": 0,
                    "run_log": str(run_log.resolve()),
                    "wall_s": float(stage[-1]) + 1.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    resolution_path = root / "resolution_information.csv"
    _write_csv(
        resolution_path,
        ("fundus", "res"),
        [(name, "0.008") for name in source_names],
    )
    resolution_hash = hashlib.sha256(resolution_path.read_bytes()).hexdigest()
    provenance = {
        "schema_version": 1,
        "source": "provided",
        "auto_generated": False,
        "path": str(resolution_path.resolve()),
        "recorded_at": "2026-09-06T00:00:00+00:00",
        "row_count": len(source_names),
        "csv_sha256": resolution_hash,
        "note": "Synthetic calibrated fixture.",
    }
    provenance_path = root / ".automorph_v2" / "resolution_provenance.json"
    provenance_path.write_text(json.dumps(provenance) + "\n", encoding="utf-8")

    run_id = "a" * 32
    pipeline = {
        "schema_version": 1,
        "status": "completed",
        "run_id": run_id,
        "selected_stages": ["m0", "m1", "m2", "m3"],
        "completed_stages": ["m0", "m1", "m2", "m3"],
        "skipped_stages": [],
        "config": {
            "data_root": str(root.resolve()),
            "from_stage": "m0",
            "to_stage": "m3",
            "progress": "plain",
            "require_calibrated_resolution": False,
        },
        "run_log": str(run_log.resolve()),
        "stage_manifests": {
            stage: str(
                (root / ".automorph_v2" / "stages" / f"{stage}.json").resolve()
            )
            for stage in ("m0", "m1", "m2", "m3")
        },
        "resolution_information": provenance,
    }
    (root / ".automorph_v2" / "pipeline.json").write_text(
        json.dumps(pipeline) + "\n",
        encoding="utf-8",
    )
    lines = [
        "2026-09-06T00:00:00+00:00 [RUN] AutoMorph v2 run started",
        f"2026-09-06T00:00:00+00:00 [RUN] run_id: {run_id}",
    ]
    for stage in ("m0", "m1", "m2", "m3"):
        lines.extend(
            [
                f"2026-09-06T00:00:00+00:00 [{stage.upper()}] "
                f"AutoMorph v2 [{stage}] starting",
                f"2026-09-06T00:00:01+00:00 [{stage.upper()} PROGRESS] "
                f"{stage.upper()} synthetic: 1/1 items (100.0%)",
                f"2026-09-06T00:00:02+00:00 [{stage.upper()}] "
                f"AutoMorph v2 [{stage}] complete",
            ]
        )
    lines.append(
        "2026-09-06T00:00:03+00:00 [DONE] "
        "run completed: completed=m0, m1, m2, m3; skipped=none; wall=10.000s"
    )
    run_log.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")


def _fixture(tmp_path: Path) -> argparse.Namespace:
    candidate = tmp_path / "candidate-private-root"
    m0_oracle = tmp_path / "m0-private-oracle"
    m1_oracle = tmp_path / "m1-private-oracle"
    m2_oracle = tmp_path / "m2-private-oracle"
    m3_oracle = tmp_path / "m3-private-oracle"
    _populate_root(candidate, NAMES)
    _populate_root(m0_oracle, (EXTRA_NAME,) + NAMES)
    _populate_root(m1_oracle, (EXTRA_NAME,) + NAMES)
    _populate_root(m2_oracle, NAMES)
    _populate_root(m3_oracle, NAMES)
    _candidate_metadata(candidate)
    return argparse.Namespace(
        candidate=candidate,
        m0_oracle=m0_oracle,
        m1_oracle=m1_oracle,
        m2_oracle=m2_oracle,
        m3_oracle=m3_oracle,
        output=tmp_path / "public-report",
        expected_input_images=21,
        expected_m0_images=21,
        expected_m1_images=21,
        expected_m2_images=21,
        expected_m3_images=21,
        legacy_m0_seconds=None,
        legacy_m0_images=None,
        legacy_m1_seconds=None,
        legacy_m1_images=None,
        legacy_m2_seconds=None,
        legacy_m2_images=None,
        legacy_m3_seconds=None,
        legacy_m3_images=None,
    )


def _staged_fixture(tmp_path: Path) -> argparse.Namespace:
    """Build the ship cohort shape: 100 source -> 99 -> 99 -> 21 -> 21."""

    extra_bad = tuple(f"private-bad-{index:02d}.png" for index in range(78))
    stage_m0_names = NAMES + extra_bad
    invalid_source = "private-invalid-source.tiff"
    candidate = tmp_path / "candidate-private-root"
    m0_oracle = tmp_path / "m0-private-oracle"
    m1_oracle = tmp_path / "m1-private-oracle"
    m2_oracle = tmp_path / "m2-private-oracle"
    m3_oracle = tmp_path / "m3-private-oracle"
    _populate_root(candidate, stage_m0_names)
    _populate_root(m0_oracle, stage_m0_names)
    _populate_root(m1_oracle, stage_m0_names)
    _populate_root(m2_oracle, NAMES)
    _populate_root(m3_oracle, NAMES)

    for root in (candidate, m1_oracle):
        for name in extra_bad:
            source = root / "Results" / "M1" / "Good_quality" / name
            target = root / "Results" / "M1" / "Bad_quality" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
    for name in extra_bad:
        (
            candidate / "Results" / "M2" / "binary_vessel" / "raw" / name
        ).unlink()
    shutil.rmtree(candidate / "Results" / "M3")
    shutil.copytree(m3_oracle / "Results" / "M3", candidate / "Results" / "M3")
    (candidate / "images" / invalid_source).write_bytes(b"")
    _candidate_metadata(
        candidate,
        source_names=stage_m0_names + (invalid_source,),
        processed_count=len(stage_m0_names),
        failure_count=1,
    )
    return argparse.Namespace(
        candidate=candidate,
        m0_oracle=m0_oracle,
        m1_oracle=m1_oracle,
        m2_oracle=m2_oracle,
        m3_oracle=m3_oracle,
        output=tmp_path / "public-report",
        expected_input_images=100,
        expected_m0_images=99,
        expected_m1_images=99,
        expected_m2_images=21,
        expected_m3_images=21,
        legacy_m0_seconds=None,
        legacy_m0_images=None,
        legacy_m1_seconds=None,
        legacy_m1_images=None,
        legacy_m2_seconds=None,
        legacy_m2_images=None,
        legacy_m3_seconds=None,
        legacy_m3_images=None,
    )


def _metric(metrics: list[accept_release.Metric], check: str) -> accept_release.Metric:
    return next(metric for metric in metrics if metric.check == check)


def test_release_cli_defaults_to_the_official_staged_cohort(tmp_path: Path) -> None:
    args = accept_release.parse_args(
        [
            "--candidate",
            str(tmp_path / "candidate"),
            "--m0-oracle",
            str(tmp_path / "m0"),
            "--m1-oracle",
            str(tmp_path / "m1"),
            "--m2-oracle",
            str(tmp_path / "m2"),
            "--m3-oracle",
            str(tmp_path / "m3"),
        ]
    )
    assert args.expected_input_images == 100
    assert (
        args.expected_m0_images,
        args.expected_m1_images,
        args.expected_m2_images,
        args.expected_m3_images,
    ) == (99, 99, 21, 21)


def test_acceptance_rejects_candidate_oracle_self_comparison(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    args.m0_oracle = args.candidate
    try:
        accept_release.run_acceptance(args)
    except ValueError as error:
        assert str(error) == "candidate and oracle roots must be disjoint"
    else:
        raise AssertionError("self-comparison must fail before producing evidence")


def test_full_acceptance_passes_and_persists_no_private_identifiers(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    summary = accept_release.run_acceptance(args)
    assert summary["verdict"] == "PASS"
    for name in ("report.md", "summary.json", "metrics.csv", "performance.png"):
        assert (args.output / name).is_file()

    persisted = "\n".join(
        (args.output / name).read_text(encoding="utf-8")
        for name in ("report.md", "summary.json", "metrics.csv")
    )
    assert str(args.candidate) not in persisted
    assert str(args.m0_oracle) not in persisted
    assert all(name not in persisted for name in NAMES)
    assert EXTRA_NAME not in persisted


def test_m0_detects_decoded_pixel_change_with_superset_oracle(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    _write_png(
        args.candidate / "Results" / "M0" / "images" / NAMES[0],
        NAMES[0],
        adjustment=1,
    )
    metrics: list[accept_release.Metric] = []
    result, _names = accept_release.compare_m0(
        args.candidate, args.m0_oracle, metrics
    )
    assert result["decoded_pixel_exact_images"] == 20
    assert not _metric(metrics, "decoded-pixel exact images").passed
    assert _metric(metrics, "crop values exact by basename").passed


def test_m0_crop_comparison_is_keyed_for_superset_oracle(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    oracle_csv = args.m0_oracle / "Results" / "M0" / "crop_info.csv"
    table = accept_release._read_csv(oracle_csv)
    assert table is not None
    _write_csv(oracle_csv, table.header, tuple(reversed(table.rows)))
    metrics: list[accept_release.Metric] = []
    result, _names = accept_release.compare_m0(
        args.candidate, args.m0_oracle, metrics
    )
    assert result["crop_rows_exact_by_basename"]
    assert _metric(metrics, "crop values exact by basename").passed


def test_m1_compares_portable_basename_fields_routes_and_copy_hash(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    candidate_csv = args.candidate / "Results" / "M1" / "results_ensemble.csv"
    table = accept_release._read_csv(candidate_csv)
    assert table is not None
    changed = list(table.rows)
    row = list(changed[0])
    row[0] = "C:\\private\\folder\\" + NAMES[0]
    row[1] = "0.100000"
    changed[0] = tuple(row)
    _write_csv(candidate_csv, table.header, changed)

    source = args.candidate / "Results" / "M1" / "Good_quality" / NAMES[1]
    target = args.candidate / "Results" / "M1" / "Bad_quality" / NAMES[1]
    source.replace(target)
    _write_png(
        args.candidate / "Results" / "M1" / "Good_quality" / NAMES[2],
        NAMES[2],
        adjustment=1,
    )
    metrics: list[accept_release.Metric] = []
    result, _names, _good = accept_release.compare_m1(
        args.candidate, args.m1_oracle, set(NAMES), metrics
    )
    assert result["non_path_fields_exact"] == 146
    assert result["route_membership_exact"] == 20
    assert result["copied_sha256_exact"] == 20
    assert not _metric(metrics, "non-path CSV fields exact strings").passed
    assert not _metric(metrics, "route membership exact").passed
    assert not _metric(metrics, "copied file SHA-256 exact").passed


def test_m2_detects_file_set_hash_pixels_and_centring_input(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    _write_png(
        args.candidate
        / "Results"
        / "M2"
        / "binary_vessel"
        / "raw"
        / NAMES[0],
        NAMES[0],
        adjustment=1,
    )
    (args.candidate / "Results" / "M2" / "unexpected.png").write_bytes(b"bad")
    input_table = (
        args.candidate
        / "Results"
        / "M3"
        / "Disc_centred"
        / "Disc_cup_results.csv"
    )
    input_table.write_bytes(input_table.read_bytes() + b"\n")
    metrics: list[accept_release.Metric] = []
    result, _private = accept_release.compare_m2(
        args.candidate, args.m2_oracle, metrics
    )
    assert not result["relative_file_set_exact"]
    assert result["sha256_exact_files"] == 20
    assert result["decoded_pixel_exact_files"] == 20
    assert result["centring_input_tables_byte_exact"] == 1
    assert not _metric(metrics, "deterministic relative file set exact").passed
    assert not _metric(metrics, "M3 centring input tables byte exact").passed


def test_m2_sample_count_uses_canonical_raw_outputs(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    for root in (args.candidate, args.m2_oracle):
        derived = root / "Results" / "M2" / "binary_vessel" / "resize"
        _write_png(derived / "derived-output-name.png", "derived-output-name.png")
    metrics: list[accept_release.Metric] = []
    result, _private = accept_release.compare_m2(
        args.candidate, args.m2_oracle, metrics
    )
    assert result["candidate_image_count"] == len(NAMES)
    assert _metric(metrics, "candidate unique image basenames").passed


def test_m3_resolves_pristine_legacy_intermediate_archive(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    archive = args.m3_oracle / "oracle_artifacts" / "intermediate_csv"
    archive.mkdir(parents=True)
    for relative in accept_release.M3_LEGACY_INTERMEDIATE_TABLES:
        source = args.m3_oracle / "Results" / "M3" / Path(relative)
        source.replace(archive / source.name)
    indexed = accept_release._m3_output_index(args.m3_oracle)
    assert set(indexed) == set(accept_release.M3_OUTPUT_TABLES)


def test_csv_comparator_distinguishes_strings_blanks_and_numeric_delta(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.csv"
    oracle = tmp_path / "oracle.csv"
    _write_csv(candidate, ("Name", "value", "optional"), [("one", "1.00", "")])
    _write_csv(oracle, ("Name", "value", "optional"), [("one", "1.0", "")])
    result = accept_release.compare_csv_table(candidate, oracle)
    assert result["schema_exact"]
    assert result["row_order_exact"]
    assert result["blank_mask_exact"]
    assert not result["cell_strings_exact"]
    assert result["numeric_max_delta"] == 0.0

    _write_csv(candidate, ("Name", "value", "optional"), [("one", "1.25", "x")])
    result = accept_release.compare_csv_table(candidate, oracle)
    assert not result["blank_mask_exact"]
    assert result["numeric_max_delta"] == 0.25


def test_full_mismatch_still_writes_aggregate_fail_report(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    _write_png(
        args.candidate / "Results" / "M0" / "images" / NAMES[0],
        NAMES[0],
        adjustment=1,
    )
    summary = accept_release.run_acceptance(args)
    assert summary["verdict"] == "FAIL"
    persisted = (args.output / "summary.json").read_text(encoding="utf-8")
    assert NAMES[0] not in persisted
    assert str(args.candidate) not in persisted


def test_missing_m3_table_writes_fail_report_instead_of_nonfinite_json(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    missing = (
        args.candidate
        / "Results"
        / "M3"
        / Path(accept_release.M3_OUTPUT_TABLES[0])
    )
    missing.unlink()
    summary = accept_release.run_acceptance(args)
    assert summary["verdict"] == "FAIL"
    import json

    persisted = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    assert persisted["exactness"]["m3"]["maximum_absolute_numeric_delta"] is None


def test_cli_returns_zero_for_pass_and_one_for_mismatch(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    argv = [
        "--candidate",
        str(args.candidate),
        "--m0-oracle",
        str(args.m0_oracle),
        "--m1-oracle",
        str(args.m1_oracle),
        "--m2-oracle",
        str(args.m2_oracle),
        "--m3-oracle",
        str(args.m3_oracle),
        "--output",
        str(args.output),
        "--expected-input-images",
        "21",
        "--expected-m0-images",
        "21",
        "--expected-m1-images",
        "21",
        "--expected-m2-images",
        "21",
        "--expected-m3-images",
        "21",
    ]
    assert accept_release.main(argv) == 0
    _write_png(
        args.candidate / "Results" / "M0" / "images" / NAMES[0],
        NAMES[0],
        adjustment=1,
    )
    assert accept_release.main(argv) == 1


def test_ship_cohort_100_to_99_to_21_passes_with_recorded_source_failure(
    tmp_path: Path,
) -> None:
    args = _staged_fixture(tmp_path)
    summary = accept_release.run_acceptance(args)
    assert summary["verdict"] == "PASS"
    assert summary["expected_input_image_count"] == 100
    assert summary["observed_stage_image_counts"] == {
        "m0": 99,
        "m1": 99,
        "m2": 21,
        "m3": 21,
    }
    assert summary["pipeline_artifacts"]["m0_failure_count"] == 1
    persisted = (args.output / "summary.json").read_text(encoding="utf-8")
    assert "private-invalid-source.tiff" not in persisted


def test_pipeline_and_run_log_run_ids_must_match(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    path = args.candidate / ".automorph_v2" / "pipeline.json"
    pipeline = json.loads(path.read_text(encoding="utf-8"))
    pipeline["run_id"] = "b" * 32
    path.write_text(json.dumps(pipeline) + "\n", encoding="utf-8")
    summary = accept_release.run_acceptance(args)
    assert summary["verdict"] == "FAIL"
    metric = next(
        item for item in summary["metrics"]
        if item["check"] == "complete root run.log contract"
    )
    assert metric["status"] == "FAIL"


def test_run_log_rejects_raw_progress_protocol(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    path = args.candidate / "run.log"
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace(
            "[M0 PROGRESS] M0 synthetic:",
            "[M0 PROGRESS] AUTOMORPH_PROGRESS M0 synthetic:",
            1,
        ),
        encoding="utf-8",
        newline="",
    )
    summary = accept_release.run_acceptance(args)
    assert summary["verdict"] == "FAIL"
    metric = next(
        item for item in summary["metrics"]
        if item["check"] == "complete root run.log contract"
    )
    assert metric["status"] == "FAIL"


def test_resolution_csv_hash_must_match_provenance(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    path = args.candidate / "resolution_information.csv"
    path.write_text(
        path.read_text(encoding="utf-8").replace("0.008", "0.009", 1),
        encoding="utf-8",
        newline="",
    )
    summary = accept_release.run_acceptance(args)
    assert summary["verdict"] == "FAIL"
    metric = next(
        item for item in summary["metrics"]
        if item["check"] == "resolution provenance matches CSV"
    )
    assert metric["status"] == "FAIL"


def test_m0_source_accounting_must_balance(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    path = args.candidate / "Results" / "M0" / "run_summary.json"
    summary_data = json.loads(path.read_text(encoding="utf-8"))
    summary_data["failure_count"] = 1
    path.write_text(json.dumps(summary_data) + "\n", encoding="utf-8")
    summary = accept_release.run_acceptance(args)
    assert summary["verdict"] == "FAIL"
    metric = next(
        item for item in summary["metrics"]
        if item["check"] == "M0 source accounting"
    )
    assert metric["status"] == "FAIL"


def test_private_provenance_warning_is_never_persisted(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    provenance_path = (
        args.candidate / ".automorph_v2" / "resolution_provenance.json"
    )
    pipeline_path = args.candidate / ".automorph_v2" / "pipeline.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    secret = f"{args.candidate}::{NAMES[0]}"
    provenance["warning"] = secret
    provenance_path.write_text(json.dumps(provenance) + "\n", encoding="utf-8")
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    pipeline["resolution_information"] = provenance
    pipeline_path.write_text(json.dumps(pipeline) + "\n", encoding="utf-8")
    summary = accept_release.run_acceptance(args)
    assert summary["verdict"] == "PASS"
    persisted = "\n".join(
        (args.output / name).read_text(encoding="utf-8")
        for name in ("report.md", "summary.json", "metrics.csv")
    )
    assert secret not in persisted
    assert NAMES[0] not in persisted
