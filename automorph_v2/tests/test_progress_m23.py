from __future__ import annotations

import io
from pathlib import Path

from v2.M2 import core as m2_core
from v2.M3 import engine as m3_engine
from v2.progress import ProgressReporter, parse_progress_line
from v2.run_m2 import parse_args as parse_m2_args
from v2.run_m3 import build_parser as build_m3_parser


def _events(stream: io.StringIO) -> list[dict[str, object]]:
    return [
        event
        for line in stream.getvalue().splitlines()
        if (event := parse_progress_line(line)) is not None
    ]


def _m3_task(index: int) -> m3_engine.EvaluationTask:
    return m3_engine.EvaluationTask(
        index=index,
        centre="disc",
        region="whole",
        vessel_type="binary",
        name=f"image-{index}.png",
        skeleton_path="unused-skeleton",
        process_path="unused-process",
        resolution=1.0,
        settings=m3_engine.MetricSettings(),
    )


def _m3_result(task: m3_engine.EvaluationTask) -> m3_engine.EvaluationResult:
    return m3_engine.EvaluationResult(
        index=task.index,
        centre=task.centre,
        region=task.region,
        vessel_type=task.vessel_type,
        name=task.name,
        metrics=m3_engine._sentinel_metrics(),
        wall_s=0.0,
    )


def test_direct_stage_progress_defaults_and_legacy_quiet() -> None:
    assert parse_m2_args([]).progress == "auto"
    assert parse_m2_args(["--progress", "protocol"]).progress == "protocol"
    m3_parser = build_m3_parser()
    assert m3_parser.parse_args([]).progress == "auto"
    args = m3_parser.parse_args(["--progress", "bar", "--quiet"])
    assert args.progress == "bar"
    assert args.quiet is True
    assert m2_core.M2Config(Path(".")).progress == "off"
    assert m3_engine.M3Config(Path(".")).progress == "off"


def test_m2_orchestrator_emits_branch_protocol(monkeypatch, tmp_path, capsys) -> None:
    (tmp_path / "Results" / "M1" / "Good_quality").mkdir(parents=True)
    received = []

    def fake_vessel(config):
        received.append(config)
        return {"images": 1}

    monkeypatch.setattr(m2_core, "run_vessel", fake_vessel)
    summary = m2_core.run_m2(
        m2_core.M2Config(
            data_root=tmp_path,
            automorph_root=tmp_path / "runtime",
            modules=("vessel",),
            progress="protocol",
            write_summary=False,
        )
    )

    events = [
        event
        for line in capsys.readouterr().out.splitlines()
        if (event := parse_progress_line(line)) is not None
    ]
    branch_events = [event for event in events if event["phase"] == "branches"]
    assert [event["event"] for event in branch_events] == [
        "start",
        "progress",
        "complete",
    ]
    assert branch_events[-1]["current"] == 1
    assert summary["modules"] == ["vessel"]
    assert received[0].progress == "protocol"


def test_m3_single_worker_progress_updates_in_parent(monkeypatch) -> None:
    tasks = [_m3_task(0), _m3_task(1)]
    monkeypatch.setattr(m3_engine, "_evaluate_task", _m3_result)
    stream = io.StringIO()
    reporter = ProgressReporter("M3", "protocol", stream=stream)
    with reporter.task("measurements", len(tasks), "evaluations") as progress:
        results = m3_engine._evaluate_all(
            tasks,
            workers=1,
            start_method="spawn",
            quiet=True,
            progress=progress,
        )

    assert [result.index for result in results] == [0, 1]
    assert _events(stream)[-1]["current"] == 2


def test_m3_pool_progress_updates_from_as_completed(monkeypatch) -> None:
    class FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class FakeExecutor:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, function, task):
            return FakeFuture(function(task))

    tasks = [_m3_task(0), _m3_task(1), _m3_task(2)]
    monkeypatch.setattr(m3_engine, "_evaluate_task", _m3_result)
    monkeypatch.setattr(m3_engine, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(m3_engine, "as_completed", lambda futures: list(futures))
    stream = io.StringIO()
    reporter = ProgressReporter("M3", "protocol", stream=stream)
    with reporter.task("measurements", len(tasks), "evaluations") as progress:
        results = m3_engine._evaluate_all(
            tasks,
            workers=2,
            start_method="spawn",
            quiet=True,
            progress=progress,
        )

    assert [result.index for result in results] == [0, 1, 2]
    assert _events(stream)[-1]["current"] == 3
