from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

from automorph_v2 import cli, pipeline
from automorph_v2.pipeline import (
    ExistingOutputError,
    PipelineConfig,
    _DataRootRunLock,
    _atomic_json,
    _build_fingerprints,
    _fingerprint_paths,
    _implementation_paths,
    _manifest_path,
    _output_fingerprint_paths,
    _prepare_overwrite,
    _required_outputs,
    _resume_decision,
    _run_stage_subprocess,
    _stage_command,
    run_pipeline,
    selected_stages,
)


def _data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    (root / "images").mkdir(parents=True)
    (root / "images" / "one.png").write_bytes(b"image")
    (root / "resolution_information.csv").write_text(
        "fundus,res\none.png,0.01\n", encoding="utf-8"
    )
    return root


def _materialize_m0_outputs(root: Path) -> None:
    required = _required_outputs(root, "m0")
    required[0].mkdir(parents=True, exist_ok=True)
    (required[0] / "one.png").write_bytes(b"processed")
    required[1].write_text("Name,Scale_resolution\none.png,1\n", encoding="utf-8")
    required[2].write_text("{}\n", encoding="utf-8")


def _install_synthetic_m0(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_subprocess(
        _command: object,
        _log_path: Path,
        **_kwargs: object,
    ) -> int:
        _materialize_m0_outputs(root)
        return 0

    monkeypatch.setattr(pipeline, "_build_fingerprints", lambda *_args: {})
    monkeypatch.setattr(pipeline, "_run_stage_subprocess", fake_subprocess)


def test_public_parser_defaults_to_auto_and_accepts_eight_threads(tmp_path: Path) -> None:
    root = _data_root(tmp_path)
    defaults = cli.build_parser().parse_args(["run", str(root)])
    args = cli.build_parser().parse_args(["run", str(root), "--threads", "8"])
    assert defaults.threads == "auto"
    assert args.command == "run"
    assert args.device == "auto"
    assert args.threads == 8
    assert args.m0_device == "cpu"
    assert defaults.fail_on_m0_image_error is False
    assert args.from_stage == "m0"
    assert args.to_stage == "m3"
    assert args.mode == "fresh"


@pytest.mark.parametrize(
    ("logical_cpus", "expected"),
    ((32, 8), (16, 8), (15, 4), (8, 4), (7, 7), (2, 2), (1, 1), (0, 1)),
)
def test_recommended_thread_count(logical_cpus: int, expected: int) -> None:
    assert cli.recommended_thread_count(logical_cpus) == expected


def test_auto_threads_are_resolved_and_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _data_root(tmp_path)
    captured: dict[str, PipelineConfig] = {}

    def fake_run(config: PipelineConfig) -> dict[str, object]:
        captured["config"] = config
        return {"completed_stages": [], "skipped_stages": [], "wall_s": 0.0}

    monkeypatch.setattr(cli.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(cli, "run_pipeline", fake_run)

    assert cli.main(["run", str(root)]) == 0
    assert captured["config"].threads == 8
    assert captured["config"].m0_workers is None
    assert "threads=8 (auto)" in capsys.readouterr().out


def test_explicit_threads_are_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _data_root(tmp_path)
    captured: dict[str, PipelineConfig] = {}

    def fake_run(config: PipelineConfig) -> dict[str, object]:
        captured["config"] = config
        return {"completed_stages": [], "skipped_stages": [], "wall_s": 0.0}

    monkeypatch.setattr(cli.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(cli, "run_pipeline", fake_run)

    assert cli.main(
        [
            "run",
            str(root),
            "--threads",
            "4",
            "--m3-workers",
            "2",
            "--fail-on-m0-image-error",
        ]
    ) == 0
    assert captured["config"].threads == 4
    assert captured["config"].worker_count("m2") == 4
    assert captured["config"].worker_count("m3") == 2
    assert captured["config"].fail_on_m0_image_error is True


def test_stage_range_is_contiguous(tmp_path: Path) -> None:
    config = PipelineConfig(
        data_root=_data_root(tmp_path), from_stage="m1", to_stage="m3"
    )
    assert selected_stages(config) == ("m1", "m2", "m3")


def test_fresh_refuses_existing_output_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _data_root(tmp_path)
    (root / "Results" / "M0").mkdir(parents=True)
    (root / "Results" / "M0" / "old.txt").write_text("old", encoding="utf-8")
    monkeypatch.setattr(
        pipeline,
        "_run_stage_subprocess",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not start"),
    )
    with pytest.raises(ExistingOutputError, match="fresh mode refuses"):
        run_pipeline(
            PipelineConfig(data_root=root, from_stage="m0", to_stage="m0")
        )
    run_log = (root / "run.log").read_text(encoding="utf-8")
    assert "[ERROR]" in run_log
    assert "ExistingOutputError" in run_log
    assert "fresh mode refuses" in run_log


def test_run_log_contains_config_stage_output_and_final_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _data_root(tmp_path)

    def fake_subprocess(
        _command: object,
        log_path: Path,
        *,
        run_log: object = None,
        stage: str | None = None,
        progress_mode: str = "auto",
    ) -> int:
        assert progress_mode == "auto"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("synthetic child stdout\n", encoding="utf-8")
        assert run_log is not None
        pipeline._write_stage_output(run_log, stage or "stage", "synthetic child stdout\n")
        _materialize_m0_outputs(root)
        return 0

    monkeypatch.setattr(pipeline, "_build_fingerprints", lambda *_args: {})
    monkeypatch.setattr(pipeline, "_run_stage_subprocess", fake_subprocess)

    summary = run_pipeline(
        PipelineConfig(
            data_root=root,
            threads=8,
            from_stage="m0",
            to_stage="m0",
        )
    )

    assert summary["status"] == "completed"
    assert summary["run_log"] == str(root / "run.log")
    run_log = (root / "run.log").read_text(encoding="utf-8")
    assert "[RUN] AutoMorph v2 run started" in run_log
    assert '"threads": 8' in run_log
    assert "[M0] AutoMorph v2 [m0] starting" in run_log
    assert "[M0] synthetic child stdout" in run_log
    assert "[DONE] run completed" in run_log
    assert (root / ".automorph_v2" / "logs" / "m0.log").read_text(
        encoding="utf-8"
    ) == "synthetic child stdout\n"


def test_missing_resolution_csv_is_generated_with_legacy_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _data_root(tmp_path)
    (root / "resolution_information.csv").unlink()
    (root / "images" / "two.TIFF").write_bytes(b"image")
    (root / "images" / "ignore.txt").write_text("not an image", encoding="utf-8")

    def fake_subprocess(
        _command: object,
        _log_path: Path,
        **_kwargs: object,
    ) -> int:
        _materialize_m0_outputs(root)
        return 0

    monkeypatch.setattr(pipeline, "_build_fingerprints", lambda *_args: {})
    monkeypatch.setattr(pipeline, "_run_stage_subprocess", fake_subprocess)

    summary = run_pipeline(
        PipelineConfig(data_root=root, from_stage="m0", to_stage="m0")
    )

    with (root / "resolution_information.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {"fundus": "one.png", "res": "0.008"},
        {"fundus": "two.TIFF", "res": "0.008"},
    ]
    provenance = json.loads(
        (root / ".automorph_v2" / "resolution_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance["auto_generated"] is True
    assert provenance["resolution_mm_per_pixel"] == 0.008
    assert summary["resolution_information"]["source"] == "legacy_uniform_default"
    assert "WARNING" in (root / "run.log").read_text(encoding="utf-8")


def test_strict_calibration_mode_refuses_missing_resolution_csv(tmp_path: Path) -> None:
    root = _data_root(tmp_path)
    (root / "resolution_information.csv").unlink()

    with pytest.raises(pipeline.PipelineError, match="require-calibrated-resolution"):
        run_pipeline(
            PipelineConfig(
                data_root=root,
                require_calibrated_resolution=True,
                from_stage="m0",
                to_stage="m0",
            )
        )

    assert not (root / "resolution_information.csv").exists()
    run_log = (root / "run.log").read_text(encoding="utf-8")
    assert "[ERROR]" in run_log
    assert "require-calibrated-resolution" in run_log


def test_auto_resolution_provenance_survives_resume_and_strict_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _data_root(tmp_path)
    (root / "resolution_information.csv").unlink()
    _install_synthetic_m0(root, monkeypatch)

    first = run_pipeline(
        PipelineConfig(data_root=root, from_stage="m0", to_stage="m0")
    )
    first_provenance = first["resolution_information"]
    assert first_provenance["auto_generated"] is True
    assert len(first_provenance["csv_sha256"]) == 64

    previous = pipeline._load_resolution_provenance(root)
    assert previous is not None
    resumed = pipeline._ensure_resolution_information(
        PipelineConfig(
            data_root=root,
            from_stage="m0",
            to_stage="m0",
            mode="resume",
        ),
        io.StringIO(),
        previous_provenance=previous,
    )
    assert resumed["auto_generated"] is True
    assert resumed["source"] == "legacy_uniform_default"
    assert resumed["csv_sha256"] == first_provenance["csv_sha256"]

    with pytest.raises(pipeline.PipelineError, match="previously generated"):
        pipeline._ensure_resolution_information(
            PipelineConfig(
                data_root=root,
                require_calibrated_resolution=True,
                from_stage="m0",
                to_stage="m0",
                mode="resume",
            ),
            io.StringIO(),
            previous_provenance=pipeline._load_resolution_provenance(root),
        )


def test_overwrite_regenerates_known_auto_resolution_for_new_uniform_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _data_root(tmp_path)
    (root / "resolution_information.csv").unlink()
    _install_synthetic_m0(root, monkeypatch)
    run_pipeline(PipelineConfig(data_root=root, from_stage="m0", to_stage="m0"))

    summary = run_pipeline(
        PipelineConfig(
            data_root=root,
            resolution_mm_per_pixel=0.01,
            from_stage="m0",
            to_stage="m0",
            mode="overwrite",
        )
    )

    rows = list(
        csv.DictReader(
            io.StringIO(
                (root / "resolution_information.csv").read_text(encoding="utf-8")
            )
        )
    )
    assert rows == [{"fundus": "one.png", "res": "0.01"}]
    provenance = summary["resolution_information"]
    assert provenance["source"] == "uniform_config_value"
    assert provenance["auto_generated"] is True
    assert provenance["resolution_mm_per_pixel"] == 0.01
    assert "regenerated" in (root / "run.log").read_text(encoding="utf-8")


def test_changed_uniform_value_requires_overwrite_for_known_auto_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _data_root(tmp_path)
    (root / "resolution_information.csv").unlink()
    _install_synthetic_m0(root, monkeypatch)
    run_pipeline(PipelineConfig(data_root=root, from_stage="m0", to_stage="m0"))
    original = (root / "resolution_information.csv").read_bytes()

    with pytest.raises(ExistingOutputError, match="use --overwrite"):
        pipeline._ensure_resolution_information(
            PipelineConfig(
                data_root=root,
                resolution_mm_per_pixel=0.01,
                from_stage="m0",
                to_stage="m0",
                mode="resume",
            ),
            io.StringIO(),
            previous_provenance=pipeline._load_resolution_provenance(root),
        )
    assert (root / "resolution_information.csv").read_bytes() == original


def test_modified_auto_resolution_is_untrusted_until_explicitly_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _data_root(tmp_path)
    (root / "resolution_information.csv").unlink()
    _install_synthetic_m0(root, monkeypatch)
    run_pipeline(PipelineConfig(data_root=root, from_stage="m0", to_stage="m0"))
    previous = pipeline._load_resolution_provenance(root)
    assert previous is not None
    changed_csv = b"fundus,res\none.png,0.012\n"
    (root / "resolution_information.csv").write_bytes(changed_csv)

    strict = PipelineConfig(
        data_root=root,
        require_calibrated_resolution=True,
        from_stage="m0",
        to_stage="m0",
        mode="resume",
    )
    with pytest.raises(pipeline.PipelineError, match="modified after"):
        pipeline._ensure_resolution_information(
            strict,
            io.StringIO(),
            previous_provenance=previous,
        )

    preserved = pipeline._ensure_resolution_information(
        PipelineConfig(
            data_root=root,
            from_stage="m0",
            to_stage="m0",
            mode="resume",
        ),
        io.StringIO(),
        previous_provenance=previous,
    )
    assert preserved["source"] == "modified_after_auto_generation"
    assert preserved["auto_generated"] is True
    assert (root / "resolution_information.csv").read_bytes() == changed_csv

    reloaded = pipeline._load_resolution_provenance(root)
    assert reloaded is not None
    rechecked = pipeline._ensure_resolution_information(
        PipelineConfig(
            data_root=root,
            from_stage="m0",
            to_stage="m0",
            mode="resume",
        ),
        io.StringIO(),
        previous_provenance=reloaded,
    )
    assert rechecked["source"] == "modified_after_auto_generation"
    with pytest.raises(pipeline.PipelineError, match="modified after"):
        pipeline._ensure_resolution_information(
            strict,
            io.StringIO(),
            previous_provenance=reloaded,
        )

    (root / ".automorph_v2" / "resolution_provenance.json").unlink()
    adopted = pipeline._ensure_resolution_information(
        strict,
        io.StringIO(),
        previous_provenance=None,
    )
    assert adopted["source"] == "provided"
    assert adopted["auto_generated"] is False


def test_stage_subprocess_relays_protocol_and_logs_clean_unicode_progress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _data_root(tmp_path)
    child = tmp_path / "progress_child.py"
    child.write_text(
        "import json, sys\n"
        "prefix = 'AUTOMORPH_PROGRESS '\n"
        "base = {'stage':'M0','phase':'preprocess','total':2,'unit':'images'}\n"
        "print('ordinary retina ✓')\n"
        "print('child stderr', file=sys.stderr)\n"
        "print(prefix + json.dumps(dict(base, event='start', current=0, elapsed_s=0.0, rate=0.0, eta_s=None)))\n"
        "print(prefix + json.dumps(dict(base, event='progress', current=1, elapsed_s=1.0, rate=1.0, eta_s=1.0)))\n"
        "print(prefix + json.dumps(dict(base, event='complete', current=2, elapsed_s=2.0, rate=1.0, eta_s=0.0)))\n",
        encoding="utf-8",
    )
    stage_log = tmp_path / "m0.log"
    run_log = io.StringIO()

    result = _run_stage_subprocess(
        [sys.executable, str(child), str(root)],
        stage_log,
        run_log=run_log,
        stage="m0",
        progress_mode="plain",
    )

    assert result == 0
    persisted = stage_log.read_text(encoding="utf-8")
    assert "ordinary retina ✓" in persisted
    assert "child stderr" in persisted
    assert "progress: M0 preprocess: 2/2 images" in persisted
    assert "AUTOMORPH_PROGRESS" not in persisted
    unified = run_log.getvalue()
    assert "[M0] ordinary retina ✓" in unified
    assert "[M0 PROGRESS] M0 preprocess: 2/2 images" in unified
    terminal = capsys.readouterr().out
    assert "ordinary retina ✓" in terminal
    assert "M0 preprocess: 2/2 images" in terminal


def test_data_root_lock_rejects_a_second_writer_and_releases_cleanly(
    tmp_path: Path,
) -> None:
    root = _data_root(tmp_path)
    first = _DataRootRunLock(root)
    first.acquire()
    try:
        with pytest.raises(ExistingOutputError, match="another AutoMorph v2 run"):
            _DataRootRunLock(root).acquire()
    finally:
        first.release()

    third = _DataRootRunLock(root)
    third.acquire()
    third.release()
    owner = json.loads((root / ".automorph_v2.run.lock").read_text(encoding="utf-8"))
    assert owner["run_id"] == third.run_id
    assert owner["pid"] > 0


def test_overwrite_removes_only_selected_owned_paths(tmp_path: Path) -> None:
    root = _data_root(tmp_path)
    m0 = root / "Results" / "M0"
    m1 = root / "Results" / "M1"
    metadata = root / ".automorph_v2"
    for directory in (m0, m1):
        directory.mkdir(parents=True)
        (directory / "sentinel.txt").write_text("x", encoding="utf-8")
    selected_manifest = metadata / "stages" / "m0.json"
    selected_log = metadata / "logs" / "m0.log"
    retained_manifest = metadata / "stages" / "m1.json"
    retained_log = metadata / "logs" / "m1.log"
    pipeline_manifest = metadata / "pipeline.json"
    for path in (
        selected_manifest,
        selected_log,
        retained_manifest,
        retained_log,
        pipeline_manifest,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("keep", encoding="utf-8")

    config = PipelineConfig(
        data_root=root, from_stage="m0", to_stage="m0", mode="overwrite"
    )
    _prepare_overwrite(config, ("m0",))

    assert not m0.exists()
    assert not selected_manifest.exists()
    assert not selected_log.exists()
    assert not pipeline_manifest.exists()
    assert retained_manifest.read_text(encoding="utf-8") == "x"
    assert retained_log.read_text(encoding="utf-8") == "x"
    assert (m1 / "sentinel.txt").read_text(encoding="utf-8") == "x"
    assert (root / "images" / "one.png").exists()
    assert outside.read_text(encoding="utf-8") == "keep"


def test_resume_skips_only_completed_matching_manifest(tmp_path: Path) -> None:
    root = _data_root(tmp_path)
    config = PipelineConfig(
        data_root=root, from_stage="m0", to_stage="m0", mode="resume"
    )
    _materialize_m0_outputs(root)
    fingerprints = _build_fingerprints(config, "m0")
    manifest = {
        "schema_version": pipeline.MANIFEST_SCHEMA,
        "stage": "m0",
        "status": "completed",
        "fingerprints": fingerprints,
        "output_fingerprint": _fingerprint_paths(
            _output_fingerprint_paths(root, "m0")
        ),
    }
    _atomic_json(_manifest_path(root, "m0"), manifest)
    assert _resume_decision(config, "m0", fingerprints) is True

    (root / "resolution_information.csv").write_text(
        "fundus,res\none.png,0.02\n", encoding="utf-8"
    )
    changed = _build_fingerprints(config, "m0")
    with pytest.raises(ExistingOutputError, match="fingerprints changed"):
        _resume_decision(config, "m0", changed)


def test_implementation_fingerprint_excludes_documentation() -> None:
    paths = _implementation_paths("m0")
    assert paths
    assert all(path.suffix == ".py" for path in paths.values())
    assert all(path.name != "README.md" for path in paths.values())


def test_m3_command_is_cpu_spawn_and_fail_fast(tmp_path: Path) -> None:
    config = PipelineConfig(
        data_root=_data_root(tmp_path),
        device="cuda:1",
        threads=8,
        from_stage="m3",
        to_stage="m3",
    )
    command = _stage_command(config, "m3")
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--workers") + 1] == "8"
    assert command[command.index("--start-method") + 1] == "spawn"
    assert "--fail-on-metric-error" in command


def test_m0_command_continues_per_image_failures_unless_fail_fast_requested(
    tmp_path: Path,
) -> None:
    compatible = _stage_command(
        PipelineConfig(
            data_root=_data_root(tmp_path / "compatible"),
            from_stage="m0",
            to_stage="m0",
        ),
        "m0",
    )
    strict = _stage_command(
        PipelineConfig(
            data_root=_data_root(tmp_path / "strict"),
            fail_on_m0_image_error=True,
            from_stage="m0",
            to_stage="m0",
        ),
        "m0",
    )
    assert "--fail-on-image-error" not in compatible
    assert "--fail-on-image-error" in strict


def test_module_help_runs_without_ml_imports() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "automorph_v2", "--help"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "run" in result.stdout
