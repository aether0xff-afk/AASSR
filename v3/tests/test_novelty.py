from __future__ import annotations

import threading
import unittest

from apassr_tool.actions import ActionTemplate, generate_candidates
from apassr_tool.dmp import APASSRToolDMP
from apassr_tool.experiment import _objective_settings
from apassr_tool.knowledge import seed_knowledge
from apassr_tool.novelty import NoveltyMemory
from apassr_tool.sandbox_server import make_server
from apassr_tool.tools import ToolExecutor


class NoveltyTests(unittest.TestCase):
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

    def test_novelty_score_declines_after_repeating_same_candidate(self) -> None:
        store = seed_knowledge("http://127.0.0.1:8088")
        candidate = next(
            row for row in generate_candidates(store) if row.template == ActionTemplate.HTTP_GET_PATH
        )
        memory = NoveltyMemory()

        before = memory.predict(candidate)
        memory.update(candidate, status=200, new_kv=3, solved_delta=0)
        after = memory.predict(candidate)

        self.assertGreater(before.score, after.score)
        self.assertEqual(after.signature_count, 1)

    def test_dmp_applies_novelty_bonus_to_reward(self) -> None:
        base_url = f"http://127.0.0.1:{self.port}"
        dmp = APASSRToolDMP(
            base_url=base_url,
            executor=ToolExecutor(prefer_curl=False),
            novelty_memory=NoveltyMemory(),
            novelty_reward=1.0,
            novelty_score_weight=1.0,
            step_limit=1,
        )

        result = dmp.run()

        self.assertEqual(len(result.records), 1)
        self.assertGreater(result.records[0].novelty_bonus, 0.0)
        self.assertGreater(result.records[0].novelty_score, 0.0)
        self.assertTrue(result.records[0].novelty_signature)
        self.assertAlmostEqual(
            result.records[0].reward,
            result.records[0].policy_reward + result.records[0].novelty_bonus,
        )

    def test_weird_objective_reduces_raw_knowledge_reward(self) -> None:
        balanced = _objective_settings("balanced")
        weird = _objective_settings("weird")

        self.assertEqual(balanced["knowledge_reward_scale"], 1.0)
        self.assertLess(weird["knowledge_reward_scale"], balanced["knowledge_reward_scale"])
        self.assertLess(weird["knowledge_reward_cap"], balanced["knowledge_reward_cap"])
        self.assertGreater(weird["novelty_reward"], balanced["novelty_reward"])


if __name__ == "__main__":
    unittest.main()
