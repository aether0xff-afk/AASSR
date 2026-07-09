import tempfile
import unittest
from pathlib import Path

from aassr import CellKind, GridWorld, GridWorldDMP
from aassr.dqn_baseline import partial_state_action_features, run_dqn_partial_baseline
from aassr.traditional_baselines import run_q_learning_baseline


class TraditionalBaselineTests(unittest.TestCase):
    def test_q_learning_baseline_runs_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, episodes, summaries = run_q_learning_baseline(
                episodes=1,
                seeds=1,
                step_limit=10,
                output_dir=tmpdir,
                epsilon=0.0,
            )

            self.assertEqual(len(episodes), 1)
            self.assertEqual(summaries[0].condition, "QLEARN")
            self.assertTrue((Path(tmpdir) / "gridworld_summary.csv").exists())

    def test_dqn_partial_baseline_runs_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, episodes, summaries = run_dqn_partial_baseline(
                episodes=1,
                seeds=1,
                step_limit=10,
                output_dir=tmpdir,
                epsilon=0.0,
            )

            self.assertEqual(len(episodes), 1)
            self.assertEqual(summaries[0].condition, "DQN_PARTIAL")
            self.assertTrue((Path(tmpdir) / "gridworld_summary.csv").exists())

    def test_dqn_partial_features_use_knowledge_state_and_candidate(self) -> None:
        world = GridWorld(width=4, height=2, start=(0, 0), cells={(3, 1): CellKind.FLAG})
        dmp = GridWorldDMP(world)
        candidate = dmp.generate_candidates()[0]

        features = partial_state_action_features(dmp, candidate)

        self.assertGreater(features.size, world.width * world.height)
        self.assertGreater(float(features.sum()), 0.0)

if __name__ == "__main__":
    unittest.main()
