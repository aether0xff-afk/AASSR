from __future__ import annotations

import unittest

from apassr_tool.policy import How, PolicyABC, PolicyView, What, Where


class PolicyABCTests(unittest.TestCase):
    def test_positive_reward_increases_selected_axes(self) -> None:
        policy = PolicyABC(lr=0.1)
        view = PolicyView(What.FORM_POST, How.AUTH_ATTEMPT, Where.KK_USERNAME)
        before = policy.score(view)
        policy.update(view, reward=3.0)
        after = policy.score(view)
        self.assertGreater(after, before)

    def test_min_probability_floor_keeps_axes_available(self) -> None:
        policy = PolicyABC(lr=0.5, min_prob=0.02)
        selected = PolicyView(What.HTTP_GET, How.NORMAL, Where.KK_PATH)
        for _ in range(30):
            policy.update(selected, reward=2.0)
        self.assertTrue(all(value > 0.0 for value in policy.what_probs.values()))
        self.assertTrue(all(value > 0.0 for value in policy.how_probs.values()))
        self.assertTrue(all(value > 0.0 for value in policy.where_probs.values()))


if __name__ == "__main__":
    unittest.main()
