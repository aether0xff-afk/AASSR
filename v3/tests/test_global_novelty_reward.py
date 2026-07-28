from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from apassr_tool.actions import ActionTemplate, generate_candidates
from apassr_tool.dmp import APASSRToolDMP
from apassr_tool.knowledge import KK, seed_knowledge
from apassr_tool.novelty import NoveltyMemory, normalize_response
from apassr_tool.reward import RewardSignal
from apassr_tool.tools import ToolExecutor, ToolResult


def get_candidate(query_value: str | None = None):
    store = seed_knowledge("http://127.0.0.1:8088")
    if query_value is None:
        return next(row for row in generate_candidates(store) if row.template == ActionTemplate.HTTP_GET_PATH)
    store.add(KK.ENDPOINT, "/api/Deliverys", source="test")
    store.add(KK.PARAM_NAME, "id", source="test")
    store.add(KK.PROBE_VALUE, query_value, source="test")
    return next(
        row for row in generate_candidates(store)
        if row.template == ActionTemplate.HTTP_QUERY_PROBE and row.bindings[KK.PROBE_VALUE] == query_value
    )


class SequenceExecutor(ToolExecutor):
    def __init__(self, results: list[ToolResult]):
        super().__init__(prefer_curl=False)
        self.results = results

    def execute(self, call):
        return self.results.pop(0)


class SolveObserver:
    def reset(self): pass

    def observe(self):
        return RewardSignal(new_solved=("fixture",), solved_total=1, challenge_total=1)


class GlobalNoveltyTests(unittest.TestCase):
    def test_same_response_is_novel_only_once_across_episodes(self) -> None:
        memory = NoveltyMemory()
        candidate = get_candidate()
        first = memory.update(candidate, status=200, response_body='{"a":1}', semantic_items=((KK.PARAM_NAME, "a"),))
        second = memory.update(candidate, status=200, response_body='{"a":1}', semantic_items=((KK.PARAM_NAME, "a"),))
        self.assertGreater(first.semantic_novelty, 0)
        self.assertEqual(second.semantic_novelty, 0)
        self.assertEqual(second.bonus, 0.0)
        # A new DMP/episode can share the same run-scoped memory.
        third = memory.update(candidate, status=200, response_body='{"a":1}', semantic_items=((KK.PARAM_NAME, "a"),))
        self.assertTrue(third.repeated_response)
        self.assertEqual(third.bonus, 0.0)

    def test_json_order_and_volatile_fields_are_deduplicated(self) -> None:
        left = '{"requestId":"abc","timestamp":"2026-01-01T00:00:00Z","data":{"b":2,"a":1}}'
        right = '{"data":{"a":1,"b":2},"timestamp":"2027-02-02T03:04:05Z","requestId":"xyz"}'
        self.assertEqual(normalize_response(left), normalize_response(right))
        self.assertNotEqual(normalize_response('{"authenticated":false}'), normalize_response('{"authenticated":true}'))

    def test_schema_change_creates_semantic_novelty(self) -> None:
        memory = NoveltyMemory()
        candidate = get_candidate()
        memory.update(candidate, status=200, response_body='{"a":1}', semantic_items=((KK.PARAM_NAME, "a"),))
        changed = memory.update(candidate, status=200, response_body='{"a":1,"b":2}', semantic_items=((KK.PARAM_NAME, "a"), (KK.PARAM_NAME, "b")))
        self.assertGreater(changed.semantic_novelty, 0)
        self.assertTrue(changed.meaningful_transition)

    def test_parameter_values_share_canonical_action_but_status_transition_is_meaningful(self) -> None:
        memory = NoveltyMemory()
        first = get_candidate("0")
        second = get_candidate("1969196030")
        self.assertEqual(memory.signature(first), memory.signature(second))
        memory.update(first, status=401, response_body='{"error":"auth"}')
        update = memory.update(second, status=200, response_body='{"data":[]}', semantic_items=((KK.ROLE, "user"),))
        self.assertTrue(update.repeated_action)
        self.assertTrue(update.meaningful_transition)
        self.assertGreater(update.semantic_novelty, 0)

    def test_persistence_is_run_isolated_and_resettable(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            candidate = get_candidate()
            one = NoveltyMemory(persistence_path=folder, run_id="one")
            one.update(candidate, status=200, response_body="same")
            loaded = NoveltyMemory(persistence_path=folder, run_id="one")
            isolated = NoveltyMemory(persistence_path=folder, run_id="two")
            self.assertEqual(loaded.predict(candidate).signature_count, 1)
            self.assertEqual(isolated.predict(candidate).signature_count, 0)
            loaded.reset(delete_persisted=True)
            reset = NoveltyMemory(persistence_path=folder, run_id="one")
            self.assertEqual(reset.predict(candidate).signature_count, 0)
            self.assertGreater(reset.update(candidate, status=200, response_body="same").semantic_novelty, 0)

    def test_corrupt_persistence_fails_safe(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "broken.json").write_text("{not json", encoding="utf-8")
            memory = NoveltyMemory(persistence_path=folder, run_id="broken")
            self.assertEqual(memory.signature_counts, {})


class RewardDecompositionTests(unittest.TestCase):
    def _dmp(self, bodies_statuses, observer=None):
        results = [ToolResult(tool="CURL_GET", command=[], status=status, stdout=body) for body, status in bodies_statuses]
        return APASSRToolDMP(
            base_url="http://127.0.0.1:8088", executor=SequenceExecutor(results),
            reward_observer=observer, novelty_memory=NoveltyMemory(), step_limit=len(results),
        )

    def test_solve_dominates_large_json_and_components_sum(self) -> None:
        body = "{" + ",".join(f'\"k{i}\":{i}' for i in range(200)) + "}"
        observation = self._dmp([(body, 200)]).run().records[0]
        solved = self._dmp([('{"ok":true}', 200)], SolveObserver()).run().records[0]
        self.assertGreater(solved.reward_challenge_solved, observation.reward * 10)
        terms = (
            solved.reward_challenge_solved + solved.reward_challenge_progress
            + solved.reward_semantic_novelty + solved.reward_useful_observation
            - solved.penalty_repeated_action - solved.penalty_repeated_response
            - solved.penalty_invalid_action - solved.penalty_no_progress
        )
        self.assertAlmostEqual(solved.reward, terms)
        self.assertEqual(solved.reward, solved.reward_total)

    def test_repeated_response_cannot_generate_positive_reward(self) -> None:
        dmp = self._dmp([('{"a":1}', 200), ('{"a":1}', 200)])
        candidate = get_candidate()
        first, _ = dmp.execute_candidate(0, candidate)
        second, _ = dmp.execute_candidate(1, candidate)
        self.assertGreater(first.reward, second.reward)
        self.assertLessEqual(second.reward, 0.0)
        self.assertEqual(second.reward_semantic_novelty, 0.0)
        self.assertGreater(second.penalty_repeated_action, 0.0)
        self.assertGreater(second.penalty_repeated_response, 0.0)

    def test_new_failed_hypothesis_beats_repeat_and_error_penalty_grows(self) -> None:
        dmp = self._dmp([('{"error":"bad input"}', 400), ('{"error":"bad input"}', 400)])
        candidate = get_candidate()
        first, _ = dmp.execute_candidate(0, candidate)
        second, _ = dmp.execute_candidate(1, candidate)
        self.assertGreater(first.reward, second.reward)
        self.assertGreater(second.penalty_invalid_action, first.penalty_invalid_action)

    def test_semantic_reward_is_capped_and_explicit_solve_reward_is_preserved(self) -> None:
        body = "{" + ",".join(f'\"k{i}\":{i}' for i in range(100)) + "}"
        observation = self._dmp([(body, 200)]).run().records[0]
        self.assertLessEqual(observation.reward_semantic_novelty, 2.0)
        dmp = APASSRToolDMP(
            base_url="http://127.0.0.1:8088",
            executor=SequenceExecutor([ToolResult(tool="CURL_GET", command=[], status=200, stdout='{"ok":true}')]),
            reward_observer=SolveObserver(), solved_reward=7.0, step_limit=1,
        )
        solved = dmp.run().records[0]
        self.assertEqual(solved.reward_challenge_solved, 7.0)


if __name__ == "__main__":
    unittest.main()
