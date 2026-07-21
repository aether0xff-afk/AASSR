from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apassr_tool.dmp import StepRecord
from apassr_tool.gui import _format_duration, _generic_episode_row, _prepare_log_paths, _write_gui_summary
from apassr_tool.novelty import NoveltyMemory
from apassr_tool.policy import PolicyABC
from apassr_tool.prophecy import TableProphecyModel


class GuiLoggingTests(unittest.TestCase):
    def test_format_duration_keeps_eta_readable(self) -> None:
        self.assertEqual(_format_duration(3.25), "3.2s")
        self.assertEqual(_format_duration(75), "1m 15s")
        self.assertEqual(_format_duration(7320), "2h 02m")

    def test_prepare_log_paths_creates_timestamped_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepare_log_paths(tmp)

            self.assertTrue(paths["dir"].exists())
            self.assertEqual(paths["records"].name, "juice_train_records.jsonl")
            self.assertEqual(paths["episodes"].name, "juice_train_episodes.jsonl")
            self.assertEqual(paths["summary"].name, "juice_train_summary.json")
            self.assertEqual(paths["checkpoint"].name, "checkpoint_latest.json")
            self.assertTrue(str(paths["dir"]).startswith(str(Path(tmp))))

    def test_gui_summary_and_checkpoint_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _prepare_log_paths(tmp)
            policy = PolicyABC()
            prophecy = TableProphecyModel()
            novelty = NoveltyMemory()
            record = StepRecord(
                step=0,
                action="GET /",
                template="HTTP_GET_PATH",
                what="HTTP_GET",
                how="NORMAL",
                where="KK_PATH",
                status=200,
                new_kv=3,
                reward=3.1,
                flag_found=False,
            )
            row = _generic_episode_row(episode=0, records=[record], prophecy=prophecy, novelty=novelty)

            _write_gui_summary(
                paths,
                policy=policy,
                prophecy=prophecy,
                novelty=novelty,
                reward_observer=None,
                episode_rows=[row],
                started_at=0.0,
                run_config={"plugin": "juice-shop-full", "episodes_completed": 1},
            )

            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            checkpoint = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
            self.assertEqual(summary["run"]["source"], "gui")
            self.assertEqual(summary["episodes"][0]["new_kv_total"], 3)
            self.assertEqual(checkpoint["run"]["plugin"], "juice-shop-full")


if __name__ == "__main__":
    unittest.main()
