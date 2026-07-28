from __future__ import annotations

import unittest

from apassr_tool.actions import ActionTemplate, generate_candidates
from apassr_tool.dmp import APASSRToolDMP
from apassr_tool.knowledge import seed_knowledge
from apassr_tool.prophecy import TableProphecyModel
from apassr_tool.reward import RewardSignal
from apassr_tool.tools import ToolExecutor, ToolResult


def candidate():
    return next(
        row for row in generate_candidates(seed_knowledge("http://127.0.0.1:8088"))
        if row.template == ActionTemplate.HTTP_GET_PATH
    )


class PositiveSignalTests(unittest.TestCase):
    def test_success_trajectory_gets_decaying_hindsight_credit(self) -> None:
        model = TableProphecyModel(hindsight_decay=0.8, hindsight_window=4)
        row = candidate()
        for solved in (0, 0, 0, 1):
            model.update(row, reward=float(solved), new_kv=0, solved_delta=solved, status=200)
        model.finalize_episode(0)
        credits = [sample.eventual_solve_credit for sample in model.replay]
        self.assertEqual(credits[-1], 1.0)
        self.assertGreater(credits[-2], credits[-3])
        self.assertGreater(credits[-3], credits[-4])
        self.assertTrue(all(sample.positive for sample in model.replay))

    def test_progress_without_solve_is_positive(self) -> None:
        model = TableProphecyModel()
        model.update(candidate(), reward=1.0, new_kv=1, solved_delta=0, status=200, progress=0.5)
        model.finalize_episode(0)
        self.assertTrue(model.replay[0].positive)
        self.assertGreater(model.predict(candidate()).progress_probability, 0.0)

    def test_balanced_batch_includes_sparse_positive(self) -> None:
        model = TableProphecyModel(minimum_positive_ratio=0.25)
        row = candidate()
        model.update(row, reward=1.0, new_kv=1, solved_delta=0, status=200, progress=0.5)
        for _ in range(20):
            model.update(row, reward=-1.0, new_kv=0, solved_delta=0, status=400)
        batch = model.sample_batch(8)
        self.assertGreaterEqual(sum(sample.positive for sample in batch), 1)

    def test_bootstrap_prevents_permanent_all_zero_predictions(self) -> None:
        prediction = TableProphecyModel().predict(candidate())
        self.assertGreater(prediction.immediate_solve_probability, 0.0)
        self.assertGreater(prediction.progress_probability, 0.0)
        self.assertGreater(prediction.eventual_solve_probability, 0.0)

    def test_prediction_targets_are_distinct(self) -> None:
        model = TableProphecyModel()
        row = candidate()
        model.update(row, reward=1.0, new_kv=1, solved_delta=0, status=200, progress=0.75)
        prediction = model.predict(row)
        self.assertNotEqual(prediction.immediate_solve_probability, prediction.progress_probability)
        self.assertGreater(prediction.expected_progress, 0.0)

    def test_dmp_finalizes_hindsight_into_shared_replay(self) -> None:
        class Executor(ToolExecutor):
            def execute(self, call):
                return ToolResult(tool="CURL_GET", command=[], status=200, stdout='{"ok":true}')

        class Observer:
            def reset(self): pass
            def observe(self): return RewardSignal(new_solved=("fixture",), solved_total=1, challenge_total=1)

        model = TableProphecyModel()
        dmp = APASSRToolDMP(
            base_url="http://127.0.0.1:8088", executor=Executor(prefer_curl=False),
            reward_observer=Observer(), prophecy_model=model, step_limit=1,
        )
        dmp.run()
        self.assertEqual(len(model.replay), 1)
        self.assertEqual(model.replay[0].eventual_solve_credit, 1.0)
        self.assertEqual(model.replay[0].distance_to_solve, 0)


if __name__ == "__main__":
    unittest.main()
