from __future__ import annotations

import io
import json

from aassr_v2.autonomous_experiment import run_autonomous_experiment
from aassr_v2.progress import ProgressReporter, format_duration


def test_progress_reporter_writes_console_log_and_machine_status(tmp_path) -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(
        3,
        tmp_path,
        every_items=1,
        every_seconds=3600.0,
        console=True,
        stream=stream,
    )
    reporter.start({"jobs": 1})
    reporter.advance({"job": "1/1", "phase": "training", "episode": "1/3"})
    reporter.advance({"job": "1/1", "phase": "training", "episode": "2/3"})
    reporter.advance({"job": "1/1", "phase": "evaluation", "episode": "1/1"})
    reporter.finish({"rows": 3})

    status = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert status["event"] == "finish"
    assert status["completed"] == 3
    assert status["total"] == 3
    assert status["percent"] == 100.0
    assert status["eta_seconds"] == 0.0
    assert "ETA" in stream.getvalue()
    assert "phase=training" in (tmp_path / "progress.log").read_text(encoding="utf-8")
    assert len((tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()) >= 5


def test_quiet_runner_still_writes_progress_files(tmp_path) -> None:
    config = {
        "name": "progress_tiny",
        "runner": "autonomous_main",
        "seeds": [1],
        "train_episodes": 8,
        "eval_episodes": 2,
        "progress": {"every_episodes": 2, "every_seconds": 3600.0},
        "environments": [{"name": "opaque_l2", "length": 2}],
        "conditions": [
            {
                "name": "full",
                "use_imagination": True,
                "minimum_holdout_count": 2,
            }
        ],
    }
    artifacts = run_autonomous_experiment(
        config,
        output_dir=tmp_path / "run",
        overwrite=True,
        progress_console=False,
    )
    status = json.loads(
        (artifacts.output_dir / "progress.json").read_text(encoding="utf-8")
    )
    assert status["event"] == "finish"
    assert status["completed"] == 10
    assert artifacts.row_count == 10
    assert (artifacts.output_dir / "progress.log").exists()
    assert (artifacts.output_dir / "progress.jsonl").exists()
    assert (artifacts.output_dir / "episodes.csv").exists()


def test_duration_format_supports_long_runs() -> None:
    assert format_duration(None) == "--:--:--"
    assert format_duration(65.0) == "00:01:05"
    assert format_duration(90_061.0) == "1d 01:01:01"


def test_progress_status_retries_windows_reader_collision(
    tmp_path, monkeypatch
) -> None:
    path_type = type(tmp_path)
    original = path_type.replace
    attempts = 0

    def flaky_replace(self, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("simulated Windows reader lock")
        return original(self, target)

    monkeypatch.setattr(path_type, "replace", flaky_replace)
    reporter = ProgressReporter(
        1,
        tmp_path,
        every_items=1,
        every_seconds=3600.0,
        console=False,
    )
    reporter.start({"jobs": 1})
    assert attempts >= 3
    assert json.loads(
        (tmp_path / "progress.json").read_text(encoding="utf-8")
    )["event"] == "start"
