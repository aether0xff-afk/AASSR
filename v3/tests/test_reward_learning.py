from __future__ import annotations

import threading
import unittest
from unittest.mock import Mock, patch

import requests

from apassr_tool.dmp import APASSRToolDMP
from apassr_tool.policy import PolicyABC
from apassr_tool.reward import JuiceShopChallengeObserver, RewardSignal
from apassr_tool.sandbox_server import make_server
from apassr_tool.tools import ToolExecutor


class OneSolvedObserver:
    def __init__(self) -> None:
        self.reset_count = 0
        self.observe_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def observe(self) -> RewardSignal:
        self.observe_count += 1
        if self.observe_count == 1:
            return RewardSignal(new_solved=("syntheticChallenge",), solved_total=1)
        return RewardSignal(new_solved=(), solved_total=1)


class RewardLearningTests(unittest.TestCase):
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

    def test_solved_observer_adds_learning_reward_without_task_staging(self) -> None:
        base_url = f"http://127.0.0.1:{self.port}"
        observer = OneSolvedObserver()
        policy = PolicyABC()
        dmp = APASSRToolDMP(
            base_url=base_url,
            executor=ToolExecutor(prefer_curl=False),
            policy=policy,
            reward_observer=observer,
            solved_reward=20.0,
            step_limit=1,
        )

        result = dmp.run()

        self.assertTrue(result.success)
        self.assertEqual(result.solved_challenges, ["syntheticChallenge"])
        self.assertEqual(observer.reset_count, 1)
        self.assertEqual(result.records[0].solved_delta, 1)
        self.assertGreaterEqual(result.records[0].reward, 20.0)

    def test_juice_shop_observer_reports_new_solved_and_total(self) -> None:
        observer = JuiceShopChallengeObserver("http://127.0.0.1:3000")
        first = Mock()
        first.json.return_value = {
            "data": [
                {"key": "a", "solved": True},
                {"key": "b", "solved": False},
            ]
        }
        first.raise_for_status.return_value = None
        second = Mock()
        second.json.return_value = {
            "data": [
                {"key": "a", "solved": True},
                {"key": "b", "solved": True},
            ]
        }
        second.raise_for_status.return_value = None

        with patch("apassr_tool.reward.requests.get", side_effect=[first, second]):
            observer.reset()
            signal = observer.observe()

        self.assertEqual(signal.new_solved, ("b",))
        self.assertEqual(signal.solved_total, 2)
        self.assertEqual(signal.challenge_total, 2)
        self.assertEqual(observer.solved_keys, ("a", "b"))

    def test_juice_shop_observer_keeps_progress_when_scoreboard_is_unreachable(self) -> None:
        observer = JuiceShopChallengeObserver("http://127.0.0.1:3000")
        observer._known_solved = {"a"}
        observer._last_challenge_total = 10

        with patch("apassr_tool.reward.requests.get", side_effect=requests.RequestException("down")):
            signal = observer.observe()

        self.assertEqual(signal.new_solved, ())
        self.assertEqual(signal.solved_total, 1)
        self.assertEqual(signal.challenge_total, 10)


if __name__ == "__main__":
    unittest.main()
