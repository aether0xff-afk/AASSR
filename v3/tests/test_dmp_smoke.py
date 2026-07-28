from __future__ import annotations

import threading
import unittest

from apassr_tool.actions import ActionTemplate, generate_candidates
from apassr_tool.dmp import APASSRToolDMP, candidate_syntax_penalty
from apassr_tool.knowledge import KK
from apassr_tool.policy import How, PolicyABC, PolicyView, What, Where
from apassr_tool.sandbox_server import make_server
from apassr_tool.tools import ToolExecutor
from apassr_tool.tools import ToolCall, ToolName, ToolResult


class UnavailableToolExecutor(ToolExecutor):
    def execute(self, call: ToolCall) -> ToolResult:
        if call.tool == ToolName.WHATWEB_SCAN:
            return ToolResult(
                tool=call.tool.value,
                command=["whatweb"],
                status=127,
                stdout="",
                stderr="whatweb is not installed",
                unavailable=True,
            )
        return super().execute(call)


class RecordingObserver:
    def __init__(self) -> None:
        self.candidates = 0
        self.steps = 0
        self.top_label = ""

    def on_candidates_scored(self, *, scored, **kwargs) -> None:
        self.candidates += 1
        self.top_label = scored[0][0].label
        self.assert_score_shape(scored[0][1])

    def on_step(self, *, record, **kwargs) -> None:
        self.steps += 1

    def assert_score_shape(self, score) -> None:
        for key in ["final_score", "policy_score", "imagination_score", "predicted_reward"]:
            if key not in score:
                raise AssertionError(f"missing score key: {key}")


class DmpSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = make_server("127.0.0.1", 0, quiet=True)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_apassr_finds_flag(self) -> None:
        base_url = f"http://127.0.0.1:{self.port}"
        executor = ToolExecutor(prefer_curl=False)
        dmp = APASSRToolDMP(base_url=base_url, executor=executor, step_limit=40)
        result = dmp.run()
        self.assertTrue(result.success)
        self.assertEqual(result.flag, "FLAG{LOCAL_TOOL_APASSR_CHAIN}")
        self.assertGreaterEqual(result.steps, 4)

    def test_breadth_score_moves_beyond_overused_get_path(self) -> None:
        base_url = f"http://127.0.0.1:{self.port}"
        dmp = APASSRToolDMP(base_url=base_url, executor=ToolExecutor(prefer_curl=False), step_limit=40)
        dmp.store.add(KK.ENDPOINT, "/api/users", source="test")
        dmp.store.add(KK.PARAM_NAME, "id", source="test")
        dmp.store.add(KK.PROBE_VALUE, "7", source="observed:test")
        dmp.template_counts[ActionTemplate.HTTP_GET_PATH.value] = 50
        dmp.what_counts["HTTP_GET"] = 50
        candidate = dmp.choose_candidate()
        self.assertIsNotNone(candidate)
        self.assertNotEqual(candidate.template, ActionTemplate.HTTP_GET_PATH)

    def test_endpoint_breadth_moves_beyond_overused_endpoint(self) -> None:
        base_url = f"http://127.0.0.1:{self.port}"
        dmp = APASSRToolDMP(base_url=base_url, executor=ToolExecutor(prefer_curl=False), step_limit=40)
        dmp.store.add(KK.ENDPOINT, "/api/a", source="test")
        dmp.store.add(KK.ENDPOINT, "/api/b", source="test")
        dmp.store.add(KK.PARAM_NAME, "id", source="test")
        dmp.store.add(KK.PROBE_VALUE, "7", source="observed:test")
        dmp.endpoint_counts["/api/a"] = 100
        candidates = generate_candidates(dmp.store)
        a_candidate = next(
            candidate
            for candidate in candidates
            if candidate.template == ActionTemplate.HTTP_QUERY_PROBE
            and candidate.bindings.get(KK.ENDPOINT) == "/api/a"
        )
        b_candidate = next(
            candidate
            for candidate in candidates
            if candidate.template == ActionTemplate.HTTP_QUERY_PROBE
            and candidate.bindings.get(KK.ENDPOINT) == "/api/b"
        )
        self.assertLess(dmp._candidate_score(a_candidate), dmp._candidate_score(b_candidate))

    def test_unavailable_tool_is_removed_from_future_candidates(self) -> None:
        base_url = f"http://127.0.0.1:{self.port}"
        dmp = APASSRToolDMP(base_url=base_url, executor=UnavailableToolExecutor(prefer_curl=False), step_limit=40)
        whatweb = next(
            candidate
            for candidate in generate_candidates(dmp.store)
            if candidate.template == ActionTemplate.WEB_FINGERPRINT
        )
        dmp.execute_candidate(0, whatweb)
        self.assertIn(ToolName.WHATWEB_SCAN.value, dmp.unavailable_tools)
        self.assertTrue(
            all(
                candidate.tool_call.tool != ToolName.WHATWEB_SCAN
                for candidate in generate_candidates(dmp.store)
                if candidate.tool_call.tool.value not in dmp.unavailable_tools
            )
        )

    def test_dmp_observer_receives_candidate_scores_and_steps(self) -> None:
        base_url = f"http://127.0.0.1:{self.port}"
        observer = RecordingObserver()
        dmp = APASSRToolDMP(
            base_url=base_url,
            executor=ToolExecutor(prefer_curl=False),
            observer=observer,
            step_limit=1,
        )

        result = dmp.run()

        self.assertEqual(result.steps, 1)
        self.assertGreater(observer.candidates, 0)
        self.assertEqual(observer.steps, 1)
        self.assertTrue(observer.top_label)

    def test_dmp_uses_policy_sample_before_candidate_binding(self) -> None:
        base_url = f"http://127.0.0.1:{self.port}"
        policy_view = PolicyView(What.QUERY_PROBE, How.PROBE_VALUE, Where.KK_PARAM_NAME)
        policy = PolicyABC(
            min_prob=0.0,
            seed=0,
            what_probs={policy_view.what: 1.0},
            how_probs={policy_view.how: 1.0},
            where_probs={policy_view.where: 1.0},
        )
        dmp = APASSRToolDMP(
            base_url=base_url,
            executor=ToolExecutor(prefer_curl=False),
            policy=policy,
            step_limit=1,
            policy_sampling_attempts=1,
        )
        dmp.store.add(KK.ENDPOINT, "/api/users", source="test")
        dmp.store.add(KK.PARAM_NAME, "id", source="test")
        dmp.store.add(KK.PROBE_VALUE, "7", source="test")
        all_candidates = generate_candidates(dmp.store)

        selected = dmp.choose_candidate()

        self.assertIsNotNone(selected)
        self.assertEqual(dmp.last_sampled_policy, policy_view)
        self.assertEqual(selected.policy, policy_view)
        self.assertLess(dmp.last_candidate_count, len(all_candidates))

    def test_syntax_penalty_suppresses_directory_append_to_file(self) -> None:
        base_url = f"http://127.0.0.1:{self.port}"
        dmp = APASSRToolDMP(base_url=base_url, executor=ToolExecutor(prefer_curl=False), step_limit=1)
        dmp.store.add(KK.PATH, "styles.css/package.json", source="test")
        candidates = generate_candidates(dmp.store)
        bad = next(candidate for candidate in candidates if candidate.label == "GET styles.css/package.json")

        score = dmp._candidate_score_details(bad)

        self.assertGreaterEqual(candidate_syntax_penalty(bad), 0.9)
        self.assertLessEqual(float(score["syntax_multiplier"]), 0.1)

    def test_syntax_penalty_does_not_hit_valid_file_fetch(self) -> None:
        base_url = f"http://127.0.0.1:{self.port}"
        dmp = APASSRToolDMP(base_url=base_url, executor=ToolExecutor(prefer_curl=False), step_limit=1)
        dmp.store.add(KK.PATH, "styles.css", source="test")
        candidates = generate_candidates(dmp.store)
        good = next(candidate for candidate in candidates if candidate.label == "GET styles.css")

        self.assertEqual(candidate_syntax_penalty(good), 0.0)


if __name__ == "__main__":
    unittest.main()
