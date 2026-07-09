import tempfile
import unittest
from pathlib import Path

from aassr.ablation import (
    imagination_depth_branch_suite,
    parse_worlds,
    prophecy_implementation_suite,
    prophecy_reward_suite,
    run_ablation_suite,
)
from aassr.experiment import ProphecyKind
from aassr.worlds import WorldKind


class AblationSuiteTests(unittest.TestCase):
    def test_prophecy_implementation_suite_compares_table_and_transformer(self) -> None:
        suite = prophecy_implementation_suite()

        kinds = {spec.condition: spec.prophecy_kind for spec in suite.specs}

        self.assertEqual(kinds["A1_TABLE_C3"], ProphecyKind.TABLE)
        self.assertEqual(kinds["A1_TRANSFORMER_C3"], ProphecyKind.TRANSFORMER)

    def test_prophecy_reward_suite_toggles_beta_only(self) -> None:
        suite = prophecy_reward_suite()
        by_condition = {spec.condition: spec for spec in suite.specs}

        self.assertEqual(by_condition["A2_REWARD_ON"].prophecy_beta, 0.3)
        self.assertEqual(by_condition["A2_REWARD_OFF"].prophecy_beta, 0.0)
        self.assertEqual(by_condition["A2_REWARD_ON"].prophecy_kind, by_condition["A2_REWARD_OFF"].prophecy_kind)
        self.assertEqual(
            by_condition["A2_REWARD_ON"].imagination_config,
            by_condition["A2_REWARD_OFF"].imagination_config,
        )

    def test_imagination_suite_expands_depth_and_branch_grid(self) -> None:
        suite = imagination_depth_branch_suite(depths=(1, 2), branches=(1, 3))

        conditions = {spec.condition for spec in suite.specs}

        self.assertEqual(conditions, {"A3_D1_B1", "A3_D1_B3", "A3_D2_B1", "A3_D2_B3"})

    def test_parse_worlds_keeps_environment_sweep_separate_from_ablation(self) -> None:
        self.assertEqual(parse_worlds("v2_complex,locked_bottleneck"), (WorldKind.V2_COMPLEX, WorldKind.LOCKED_BOTTLENECK))
        self.assertIn(WorldKind.RANDOM_KEY_DOOR, parse_worlds("all"))

    def test_run_ablation_suite_writes_condition_outputs_and_analysis(self) -> None:
        suite = prophecy_reward_suite()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = run_ablation_suite(
                suite=suite,
                world=WorldKind.RANDOM_KEY_DOOR,
                episodes=1,
                seeds=1,
                step_limit=5,
                output_dir=tmpdir,
            )

            for condition in ("A2_REWARD_ON", "A2_REWARD_OFF"):
                self.assertTrue((output / condition / "gridworld_episodes.csv").exists(), condition)
            self.assertTrue((output / "analysis" / "summary_table.csv").exists())
            self.assertTrue((Path(tmpdir) / "README.md").exists())


if __name__ == "__main__":
    unittest.main()
