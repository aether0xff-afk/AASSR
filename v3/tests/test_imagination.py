from __future__ import annotations

import unittest

from apassr_tool.actions import ActionTemplate, generate_candidates
from apassr_tool.dmp import APASSRToolDMP
from apassr_tool.imagination import ImaginationCycle
from apassr_tool.knowledge import KK, seed_knowledge
from apassr_tool.prophecy import TableProphecyModel
from apassr_tool.tools import ToolExecutor


class ImaginationTests(unittest.TestCase):
    def test_table_prophecy_predicts_from_prior_execution_data(self) -> None:
        store = seed_knowledge("http://127.0.0.1:8088")
        candidates = generate_candidates(store)
        candidate = next(row for row in candidates if row.template == ActionTemplate.HTTP_GET_PATH)
        prophecy = TableProphecyModel()

        neutral = prophecy.predict(candidate)
        self.assertEqual(neutral.support, 0)

        prophecy.update(candidate, reward=5.0, new_kv=4, solved_delta=1, status=200)
        prediction = prophecy.predict(candidate)

        self.assertGreater(prediction.support, 0)
        self.assertGreater(prediction.expected_reward, 0.0)
        self.assertGreater(prediction.expected_knowledge, 0.0)
        self.assertGreater(prediction.solved_rate, 0.0)

    def test_imagination_multiplier_is_neutral_until_experience_exists(self) -> None:
        store = seed_knowledge("http://127.0.0.1:8088")
        candidate = next(
            row for row in generate_candidates(store) if row.template == ActionTemplate.HTTP_GET_PATH
        )
        prophecy = TableProphecyModel()
        imagination = ImaginationCycle(prophecy)

        multiplier, prediction = imagination.score_multiplier(candidate)
        self.assertEqual(multiplier, 1.0)
        self.assertEqual(prediction.support, 0)

        prophecy.update(candidate, reward=5.0, new_kv=4, solved_delta=0, status=200)
        learned_multiplier, learned_prediction = imagination.score_multiplier(candidate)
        self.assertGreater(learned_multiplier, 1.0)
        self.assertGreater(learned_prediction.support, 0)

    def test_dmp_score_uses_learned_imagination_memory(self) -> None:
        base_url = "http://127.0.0.1:8088"
        prophecy = TableProphecyModel()
        dmp = APASSRToolDMP(
            base_url=base_url,
            executor=ToolExecutor(prefer_curl=False),
            prophecy_model=prophecy,
            step_limit=1,
        )
        dmp.store.add(KK.ENDPOINT, "/api/a", source="test")
        dmp.store.add(KK.ENDPOINT, "/api/b", source="test")
        candidates = generate_candidates(dmp.store)
        a_candidate = next(
            row for row in candidates if row.template == ActionTemplate.HTTP_GET_API and row.bindings[KK.ENDPOINT] == "/api/a"
        )
        b_candidate = next(
            row for row in candidates if row.template == ActionTemplate.HTTP_GET_API and row.bindings[KK.ENDPOINT] == "/api/b"
        )

        before_a = dmp._candidate_score(a_candidate)
        before_b = dmp._candidate_score(b_candidate)
        self.assertEqual(before_a, before_b)

        prophecy.update(a_candidate, reward=8.0, new_kv=5, solved_delta=1, status=200)

        self.assertGreater(dmp._candidate_score(a_candidate), dmp._candidate_score(b_candidate))


if __name__ == "__main__":
    unittest.main()
