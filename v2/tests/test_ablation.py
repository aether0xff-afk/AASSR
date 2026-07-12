import tempfile
import unittest
from pathlib import Path

from aassr.ablation import (
    imagination_depth_branch_suite,
    imagination_mechanism_suite,
    parse_worlds,
    prophecy_implementation_suite,
    prophecy_reward_suite,
    prophecy_score_component_suite,
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

    def test_imagination_mechanism_suite_removes_one_mechanism_at_a_time(self) -> None:
        suite = imagination_mechanism_suite()
        by_condition = {spec.condition: spec for spec in suite.specs}

        self.assertIn("A4_FULL_C3", by_condition)
        self.assertEqual(by_condition["A4_NO_DEPENDENCY"].imagination_config.dependency_weight, 0.0)
        self.assertEqual(by_condition["A4_NO_REPEAT_PENALTY"].imagination_config.repeat_weight, 0.0)
        self.assertEqual(by_condition["A4_NO_POLICY_PRIOR"].imagination_config.policy_prior_weight, 0.0)
        self.assertEqual(by_condition["A4_NO_ROLLOUT_VALUE"].imagination_config.rollout_discount, 0.0)
        self.assertEqual(by_condition["A4_ONE_STEP_NO_DEP"].imagination_config.rollout_depth, 1)
        self.assertEqual(by_condition["A4_ONE_STEP_NO_DEP"].imagination_config.dependency_weight, 0.0)

    def test_prophecy_score_component_suite_removes_score_terms(self) -> None:
        suite = prophecy_score_component_suite()
        by_condition = {spec.condition: spec for spec in suite.specs}

        self.assertEqual(by_condition["A5_NO_KNOWLEDGE_GAIN"].imagination_config.knowledge_weight, 0.0)
        self.assertEqual(by_condition["A5_NO_FLAG_PROB"].imagination_config.flag_weight, 0.0)
        self.assertEqual(by_condition["A5_NO_ERROR_AVOIDANCE"].imagination_config.error_weight, 0.0)

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
