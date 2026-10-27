from __future__ import annotations

import csv
import io
import json
import threading
from pathlib import Path

import pytest

from v2 import progress as progress_module
from v2.M0 import core as m0_core
from v2.M1 import core as m1_core
from v2.progress import PROTOCOL_PREFIX, ProgressReporter, parse_progress_line
from v2 import run_m0, run_m1


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_protocol_lines_are_prefixed_json_with_rate_and_eta() -> None:
    stream = io.StringIO()
    clock = Clock()
    reporter = ProgressReporter('M1', 'protocol', stream=stream, clock=clock)
    with reporter.task('inference', total=3, unit='images') as task:
        clock.value = 1.0
        task.update(2)

    lines = stream.getvalue().splitlines()
    assert all(line.startswith(PROTOCOL_PREFIX) for line in lines)
    payloads = [parse_progress_line(line) for line in lines]
    assert [payload['event'] for payload in payloads] == [
        'start', 'progress', 'complete',
    ]
    assert payloads[1] == {
        'event': 'progress',
        'stage': 'M1',
        'phase': 'inference',
        'current': 2,
        'total': 3,
        'unit': 'images',
        'elapsed_s': 1.0,
        'rate': 2.0,
        'eta_s': 0.5,
    }


def test_parse_progress_line_rejects_ordinary_or_invalid_output() -> None:
    assert parse_progress_line('ordinary stage output') is None
    assert parse_progress_line(PROTOCOL_PREFIX + 'not-json') is None
    assert parse_progress_line(PROTOCOL_PREFIX + json.dumps([1, 2])) is None


def test_plain_output_is_rate_limited_and_log_friendly() -> None:
    stream = io.StringIO()
    clock = Clock()
    reporter = ProgressReporter(
        'M0', 'plain', stream=stream, min_interval=1.0, clock=clock
    )
    task = reporter.task('preprocess', total=5, unit='images')
    task.update()
    clock.value = 0.5
    task.update()
    clock.value = 1.0
    task.update()
    task.close()

    lines = stream.getvalue().splitlines()
    assert len(lines) == 3
    assert 'M0 preprocess: 0/5 images' in lines[0]
    assert 'M0 preprocess: 3/5 images' in lines[1]
    assert 'elapsed 00:01' in lines[1]
    assert 'rate 3.00 images/s' in lines[1]
    assert 'ETA 00:01' in lines[1]
    assert lines[2].endswith('[complete]')


def test_auto_uses_tqdm_on_tty_and_updates_the_real_bar_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    class FakeTqdm:
        def __init__(self, **kwargs) -> None:
            calls.append(('init', kwargs))

        def update(self, amount: int) -> None:
            calls.append(('update', amount))

        def set_postfix_str(self, value: str) -> None:
            calls.append(('postfix', value))

        def close(self) -> None:
            calls.append(('close', None))

    monkeypatch.setattr(progress_module, '_load_tqdm', lambda: FakeTqdm)
    reporter = ProgressReporter('M2', 'auto', stream=TTYBuffer())
    with reporter.task('vessel', total=4, unit='images') as task:
        task.update(2)

    assert calls[0][0] == 'init'
    assert calls[0][1]['total'] == 4
    assert ('update', 2) in calls
    assert calls[-1] == ('close', None)


def test_forced_bar_falls_back_to_plain_without_tqdm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.StringIO()
    monkeypatch.setattr(progress_module, '_load_tqdm', lambda: None)
    with ProgressReporter('M3', 'bar', stream=stream).task(
        'features', total=1, unit='images'
    ) as task:
        task.update()
    assert 'M3 features:' in stream.getvalue()
    assert '[complete]' in stream.getvalue()


def test_off_is_silent_and_cross_thread_updates_are_rejected() -> None:
    stream = io.StringIO()
    task = ProgressReporter('M0', 'off', stream=stream).task(
        'preprocess', total=1, unit='images'
    )
    errors: list[Exception] = []

    def update_from_worker() -> None:
        try:
            task.update()
        except Exception as error:  # test captures the thread exception
            errors.append(error)

    worker = threading.Thread(target=update_from_worker)
    worker.start()
    worker.join()
    task.close()

    assert stream.getvalue() == ''
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert 'main thread' in str(errors[0])


def test_direct_stage_clis_default_to_auto_and_accept_protocol(tmp_path: Path) -> None:
    assert run_m0.build_parser().parse_args([str(tmp_path)]).progress == 'auto'
    assert run_m0.build_parser().parse_args(
        [str(tmp_path), '--progress', 'protocol']
    ).progress == 'protocol'
    assert run_m1.parse_args([str(tmp_path)]).progress == 'auto'
    assert run_m1.parse_args(
        [str(tmp_path), '--progress', 'protocol']
    ).progress == 'protocol'


def test_m0_cli_records_image_failures_but_only_fails_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = {
        'image_count': 2,
        'failure_count': 1,
        'wall_s': 1.0,
        'throughput_images_s': 2.0,
        'backend': 'opencv:cpu',
    }
    monkeypatch.setattr(run_m0, 'run_m0', lambda _config: summary)

    assert run_m0.main([str(tmp_path), '--progress', 'off']) == 0
    assert run_m0.main(
        [str(tmp_path), '--progress', 'off', '--fail-on-image-error']
    ) == 1


def test_m0_parallel_completion_emits_per_image_progress_on_main_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_dir = tmp_path / 'images'
    input_dir.mkdir()
    names = ['a.png', 'b.png', 'c.png']
    for name in names:
        (input_dir / name).write_bytes(b'fixture')
    resolution_csv = tmp_path / 'resolution_information.csv'
    resolution_csv.write_text(
        'fundus,res\n' + ''.join(f'{name},0.01\n' for name in names),
        encoding='utf-8',
    )

    worker_threads: set[int] = set()

    def fake_process(
        input_path: Path,
        output_path: Path,
        resolution: float,
        _backend,
        _compression: int,
    ) -> m0_core.ImageResult:
        worker_threads.add(threading.get_ident())
        return m0_core.ImageResult(
            input_name=input_path.name,
            output_name=output_path.name,
            status='ok',
            resolution=resolution,
        )

    monkeypatch.setattr(m0_core, '_process_one', fake_process)
    summary = m0_core.run_m0(m0_core.RunConfig(
        input_dir=input_dir,
        output_dir=tmp_path / 'Results' / 'M0' / 'images',
        resolution_csv=resolution_csv,
        workers=2,
        progress='protocol',
    ))

    payloads = [
        parse_progress_line(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(PROTOCOL_PREFIX)
    ]
    assert summary['processed_count'] == 3
    assert payloads[-1]['event'] == 'complete'
    assert payloads[-1]['current'] == 3
    assert payloads[-1]['total'] == 3
    assert threading.get_ident() not in worker_threads


def test_m1_routing_progress_is_updated_by_caller_thread(tmp_path: Path) -> None:
    sources = tmp_path / 'sources'
    sources.mkdir()
    rows = []
    for index, prediction in enumerate((0, 1, 2)):
        path = sources / f'{index}.png'
        path.write_bytes(b'image')
        rows.append({
            'Name': str(path),
            'Prediction': prediction,
            'softmax_bad': 0.1,
        })

    stream = io.StringIO()
    reporter = ProgressReporter('M1', 'protocol', stream=stream)
    with reporter.task('routing', total=3, unit='images') as task:
        good, usable, bad = m1_core.materialize_classification(
            rows, tmp_path / 'M1', workers=2, progress=task
        )

    payloads = [parse_progress_line(line) for line in stream.getvalue().splitlines()]
    assert (good, usable, bad) == (1, 1, 1)
    assert payloads[-1]['current'] == 3
    assert (tmp_path / 'M1' / 'Good_quality' / '0.png').is_file()
    assert (tmp_path / 'M1' / 'Usable_quality' / '1.png').is_file()
    assert (tmp_path / 'M1' / 'Bad_quality' / '2.png').is_file()


def test_m1_writes_explicit_class_and_gradability_artifacts(tmp_path: Path) -> None:
    rows = [
        {
            'Name': f'image-{index}.png',
            'softmax_good': good,
            'softmax_usable': usable,
            'softmax_bad': bad,
            'good_sd': 0.01,
            'usable_sd': 0.02,
            'bad_sd': 0.03,
            'Prediction': prediction,
        }
        for index, (good, usable, bad, prediction) in enumerate((
            (0.8, 0.1, 0.1, 0),
            (0.2, 0.6, 0.2, 1),
            (0.1, 0.6, 0.3, 1),
            (0.1, 0.1, 0.8, 2),
        ))
    ]

    counts = m1_core.write_gradability_artifacts(tmp_path, rows)

    assert counts == {
        'prediction_good_count': 1,
        'prediction_usable_count': 2,
        'prediction_bad_count': 1,
        'gradable_count': 2,
        'ungradable_count': 2,
    }
    with (tmp_path / 'M1_quality_final.csv').open(newline='', encoding='utf8') as handle:
        final_rows = list(csv.DictReader(handle))
    assert [row['Gradable'] for row in final_rows] == ['True', 'True', 'False', 'False']
    with (tmp_path / 'M1_quality_summary.csv').open(newline='', encoding='utf8') as handle:
        summary = next(csv.DictReader(handle))
    assert summary['total_images'] == '4'
    assert summary['good_count'] == '1'
    assert summary['usable_count'] == '2'
    assert summary['bad_count'] == '1'
    assert summary['gradable_count'] == '2'
    assert summary['ungradable_count'] == '2'


def test_m1_cli_distinguishes_prediction_classes_from_gradability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(run_m1, 'run_m1', lambda _config: {
        'image_count': 10,
        'prediction_good_count': 3,
        'prediction_usable_count': 2,
        'prediction_bad_count': 5,
        'gradable_count': 4,
        'ungradable_count': 6,
        'wall_s': 1.0,
        'device_resolved': 'cpu',
        'precision': 'fp32',
    })

    assert run_m1.main([str(tmp_path), '--progress', 'off']) == 0
    output = capsys.readouterr().out
    assert 'classes: good=3, usable=2, bad=5' in output
    assert 'gradability: gradable=4, ungradable=6' in output
