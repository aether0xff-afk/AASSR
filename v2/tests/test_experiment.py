import csv
import tempfile
import unittest
from pathlib import Path

from aassr import CellKind, KK
from aassr.experiment import (
    ExperimentComponents,
    ExperimentCondition,
    ExperimentSpec,
    ProphecyKind,
    c5_imagination_config,
    default_output_dir,
    run_all_conditions,
    run_experiment,
    run_experiment_spec,
)
from aassr.gridworld import ActionCandidate, ActionName
from aassr.metrics import action_signature
from aassr.prophecy import SequenceProphecyModel, TableProphecyModel, TransformerProphecyModel
from aassr.v2_compare import run_v2_comparison
from aassr.worlds import WorldKind, make_world


class ExperimentRunnerTests(unittest.TestCase):
    def test_c0_experiment_returns_required_episode_metrics(self) -> None:
        _, episodes, summaries = run_experiment(
            condition=ExperimentCondition.C0,
            episodes=1,
            seeds=1,
            step_limit=10,
        )

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].condition, "C0")
        self.assertIn(episodes[0].success, {True, False})
        self.assertGreaterEqual(episodes[0].steps_to_flag, 0)
        self.assertEqual(summaries[0].condition, "C0")
        self.assertEqual(summaries[0].episodes, 1)

    def test_c2_records_prophecy_error_in_step_metrics(self) -> None:
        steps, _, _ = run_experiment(
            condition=ExperimentCondition.C2,
            episodes=1,
            seeds=1,
            step_limit=10,
        )

        self.assertTrue(steps)
        self.assertTrue(any(step.prophecy_error >= 0.0 for step in steps))

    def test_c3_records_imagination_candidate_counts(self) -> None:
        steps, _, _ = run_experiment(
            condition=ExperimentCondition.C3,
            episodes=1,
            seeds=1,
            step_limit=10,
        )

        self.assertTrue(steps)
        self.assertTrue(any(step.imagination_candidate_count > 0 for step in steps))

    def test_c4_uses_optional_sequence_prophecy_and_imagination(self) -> None:
        steps, episodes, summaries = run_experiment(
            condition=ExperimentCondition.C4,
            episodes=1,
            seeds=1,
            step_limit=10,
        )

        self.assertEqual(summaries[0].condition, "C4")
        self.assertEqual(episodes[0].condition, "C4")
        self.assertTrue(any(step.imagination_candidate_count > 0 for step in steps))

    def test_c3_remains_table_prophecy_main_condition_while_c4_is_optional_variant(self) -> None:
        c3 = ExperimentComponents.for_condition(ExperimentCondition.C3, seed=0)
        c4 = ExperimentComponents.for_condition(ExperimentCondition.C4, seed=0)

        self.assertIsInstance(c3.prophecy, TableProphecyModel)
        self.assertIsInstance(c4.prophecy, SequenceProphecyModel)
        self.assertTrue(c3.use_imagination)
        self.assertTrue(c4.use_imagination)

    def test_c5_is_improved_apassr_condition_without_replacing_c3(self) -> None:
        c3 = ExperimentComponents.for_condition(ExperimentCondition.C3, seed=0)
        c5 = ExperimentComponents.for_condition(ExperimentCondition.C5, seed=0)

        self.assertIsInstance(c3.prophecy, TableProphecyModel)
        self.assertIsInstance(c5.prophecy, TableProphecyModel)
        self.assertTrue(c3.use_imagination)
        self.assertTrue(c5.use_imagination)
        self.assertEqual(c3.imagination_config.knowledge_weight, 2.0)
        self.assertEqual(c3.imagination_config.policy_prior_weight, 0.2)
        self.assertEqual(c5.imagination_config.knowledge_weight, 0.0)
        self.assertEqual(c5.imagination_config.policy_prior_weight, c3.imagination_config.policy_prior_weight)
        self.assertEqual(c5.imagination_config.repeat_weight, c3.imagination_config.repeat_weight)
        self.assertEqual(c5.imagination_config.error_weight, c3.imagination_config.error_weight)

    def test_c5_imagination_config_matches_ablation_improvement(self) -> None:
        config = c5_imagination_config()

        self.assertEqual(config.knowledge_weight, 0.0)
        self.assertGreater(config.policy_prior_weight, 0.0)
        self.assertGreater(config.repeat_weight, 0.0)
        self.assertGreater(config.error_weight, 0.0)

    def test_experiment_writes_required_csv_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_experiment(
                condition="C1",
                episodes=1,
                seeds=1,
                step_limit=10,
                output_dir=tmpdir,
            )

            output = Path(tmpdir)
            expected_files = [
                output / "gridworld_steps.csv",
                output / "gridworld_episodes.csv",
                output / "gridworld_summary.csv",
            ]
            for path in expected_files:
                self.assertTrue(path.exists(), path)

            with (output / "gridworld_episodes.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertIn("condition", rows[0])
        self.assertIn("steps_to_flag", rows[0])
        self.assertIn("semantic_gain_total", rows[0])

    def test_default_output_dir_is_condition_safe(self) -> None:
        self.assertEqual(default_output_dir("C2"), "runs/gridworld/C2")

    def test_action_signature_ignores_current_position(self) -> None:
        first = ActionCandidate(
            name=ActionName.MOVE_TOWARD,
            template="MOVE_TOWARD {KK_FLAG_CELL}",
            required_kk_slots=(KK.CURRENT_POS, KK.FLAG_CELL),
            bindings={KK.CURRENT_POS: (1, 1), KK.FLAG_CELL: (5, 2)},
        )
        second = ActionCandidate(
            name=ActionName.MOVE_TOWARD,
            template="MOVE_TOWARD {KK_FLAG_CELL}",
            required_kk_slots=(KK.CURRENT_POS, KK.FLAG_CELL),
            bindings={KK.CURRENT_POS: (2, 1), KK.FLAG_CELL: (5, 2)},
        )

        self.assertEqual(action_signature(first), action_signature(second))

    def test_random_world_changes_with_seed(self) -> None:
        first = make_world(WorldKind.RANDOM_KEY_DOOR, seed=1)
        second = make_world(WorldKind.RANDOM_KEY_DOOR, seed=2)

        self.assertNotEqual(first.cells, second.cells)
        self.assertIn(CellKind.FLAG, set(first.cells.values()))
        self.assertIn(CellKind.KEY, set(first.cells.values()))
        self.assertIn(CellKind.DOOR, set(first.cells.values()))

    def test_experiment_accepts_random_world(self) -> None:
        _, episodes, _ = run_experiment(
            condition="C0",
            world=WorldKind.RANDOM_FLAG,
            episodes=1,
            seeds=1,
            step_limit=10,
        )

        self.assertEqual(len(episodes), 1)

    def test_all_condition_runner_writes_combined_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, summaries = run_all_conditions(
                episodes=1,
                seeds=1,
                step_limit=5,
                output_dir=tmpdir,
            )

            output = Path(tmpdir)
            self.assertTrue((output / "combined_summary.csv").exists())
            for condition in ("C0", "C1", "C2", "C3", "C4", "C5"):
                self.assertTrue((output / condition / "gridworld_summary.csv").exists())

            with (output / "combined_summary.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(summaries), 6)
        self.assertEqual({row["condition"] for row in rows}, {"C0", "C1", "C2", "C3", "C4", "C5"})

    def test_custom_experiment_spec_writes_ablation_condition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = ExperimentSpec(
                condition="A1_TRANSFORMER_C3",
                prophecy_kind=ProphecyKind.TRANSFORMER,
                prophecy_beta=0.3,
                use_imagination=True,
            )
            steps, episodes, summaries = run_experiment_spec(
                spec=spec,
                episodes=1,
                seeds=1,
                step_limit=5,
                output_dir=tmpdir,
            )

            self.assertTrue(steps)
            self.assertEqual(episodes[0].condition, "A1_TRANSFORMER_C3")
            self.assertEqual(summaries[0].condition, "A1_TRANSFORMER_C3")
            self.assertTrue((Path(tmpdir) / "gridworld_episodes.csv").exists())

    def test_experiment_spec_can_select_transformer_prophecy(self) -> None:
        spec = ExperimentSpec(
            condition="A1_TRANSFORMER_C3",
            prophecy_kind=ProphecyKind.TRANSFORMER,
            use_imagination=True,
        )

        components = ExperimentComponents.for_spec(spec, seed=0)

        self.assertIsInstance(components.prophecy, TransformerProphecyModel)
        self.assertTrue(components.use_imagination)

    def test_v2_compare_can_include_full_and_calibrated_full_in_combined_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = run_v2_comparison(
                episodes=1,
                seeds=1,
                step_limit=4,
                output_dir=tmpdir,
                world=WorldKind.V2_COMPLEX,
                analyze=False,
                include_apassr_full=True,
                include_apassr_full_cal=True,
            )

            with (output / "combined_summary.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        conditions = {row["condition"] for row in rows}
        self.assertIn("APASSR_FULL", conditions)
        self.assertIn("APASSR_FULL_CAL", conditions)


if __name__ == "__main__":
    unittest.main()
