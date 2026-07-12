import csv
import tempfile
import unittest
from pathlib import Path

from aassr.analysis import analyze_results, condition_seed_stats, summary_table


FIELDNAMES = [
    "condition",
    "seed",
    "episode",
    "success",
    "steps_to_flag",
    "total_reward",
    "external_reward",
    "semantic_gain_total",
    "prophecy_error_mean",
    "repeat_count",
    "error_count",
    "knowledge_reuse_count",
    "unique_action_count",
]


class AnalysisTests(unittest.TestCase):
    def test_steps_to_flag_summary_uses_successful_episodes_only(self) -> None:
        rows = [
            episode("C0", 0, 0, True, 5),
            episode("C0", 0, 1, False, 100),
            episode("C0", 1, 0, True, 7),
            episode("C0", 1, 1, False, 100),
        ]

        stats = condition_seed_stats(rows)
        summary = summary_table(stats, bootstrap_samples=20)

        self.assertEqual(summary[0].condition, "C0")
        self.assertEqual(summary[0].steps_to_flag_mean, 6.0)
        self.assertEqual(summary[0].success_rate_mean, 0.5)

    def test_condition_stats_computes_repeat_and_error_rates(self) -> None:
        rows = [
            episode("C3", 0, 0, True, 10, repeat_count=2, error_count=1),
            episode("C3", 0, 1, True, 20, repeat_count=4, error_count=2),
        ]

        stats = condition_seed_stats(rows)

        self.assertEqual(len(stats), 1)
        self.assertAlmostEqual(stats[0].repeat_rate_mean, 0.2)
        self.assertAlmostEqual(stats[0].error_rate_mean, 0.1)

    def test_analysis_writes_tables_figures_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "all"
            output_dir = input_dir / "analysis"
            write_fake_experiment(input_dir)

            analyze_results(
                input_dir=input_dir,
                output_dir=output_dir,
                bootstrap_samples=20,
                learning_window=1,
            )

            expected = [
                "summary_table.csv",
                "condition_stats.csv",
                "learning_curve.csv",
                "figure_success_rate.png",
                "figure_steps_to_flag.png",
                "figure_semantic_gain.png",
                "figure_repeat_error_rate.png",
                "figure_learning_curve.png",
                "report.md",
            ]
            for filename in expected:
                self.assertTrue((output_dir / filename).exists(), filename)

            with (output_dir / "summary_table.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(
            {row["condition"] for row in rows},
            {"C0", "C1", "C2", "C3", "C4", "C5", "QLEARN", "DQN_PARTIAL", "ORACLE_MDP"},
        )

    def test_report_displays_baseline_condition_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "all"
            output_dir = input_dir / "analysis"
            write_fake_experiment(input_dir)

            analyze_results(
                input_dir=input_dir,
                output_dir=output_dir,
                bootstrap_samples=20,
                learning_window=1,
            )

            report = (output_dir / "report.md").read_text(encoding="utf-8")

        self.assertIn("Q-learning baseline", report)
        self.assertIn("DQN partial-observation baseline", report)
        self.assertIn("Oracle MDP, full-map upper bound", report)
        self.assertIn("C4 PolicyABC + Sequence Prophecy variant + Imagination", report)
        self.assertIn("C5 Improved APASSR", report)


def episode(
    condition,
    seed,
    episode_index,
    success,
    steps_to_flag,
    *,
    semantic_gain=10,
    repeat_count=0,
    error_count=0,
    prophecy_error=0.0,
    total_reward=1.0,
):
    return {
        "condition": condition,
        "seed": seed,
        "episode": episode_index,
        "success": success,
        "steps_to_flag": steps_to_flag,
        "total_reward": total_reward,
        "external_reward": 1.0 if success else 0.0,
        "semantic_gain_total": semantic_gain,
        "prophecy_error_mean": prophecy_error,
        "repeat_count": repeat_count,
        "error_count": error_count,
        "knowledge_reuse_count": 1,
        "unique_action_count": max(1, steps_to_flag - repeat_count),
    }


def write_fake_experiment(input_dir: Path) -> None:
    conditions = ("C0", "C1", "C2", "C3", "C4", "C5", "QLEARN", "DQN_PARTIAL", "ORACLE_MDP")
    for index, condition in enumerate(conditions):
        condition_dir = input_dir / condition
        condition_dir.mkdir(parents=True)
        rows = [
            episode(condition, 0, 0, index >= 2, 8 + index, repeat_count=index, error_count=1),
            episode(condition, 0, 1, True, 6 + index, repeat_count=index, error_count=0),
            episode(condition, 1, 0, True, 5 + index, repeat_count=0, error_count=index),
            episode(condition, 1, 1, False, 50, repeat_count=2, error_count=2),
        ]
        with (condition_dir / "gridworld_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
