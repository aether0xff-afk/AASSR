import unittest

from aassr import GridWorld, GridWorldDMP
from aassr.policy import PolicyABC, candidate_axes


class PolicyABCTests(unittest.TestCase):
    def test_candidate_axes_decompose_what_how_where(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))
        candidate = dmp.generate_candidates()[0]

        what, how, where = candidate_axes(candidate)

        self.assertEqual(what, candidate.name.value)
        self.assertEqual(how, candidate.strategy)
        self.assertTrue(where.startswith("KK_"))

    def test_positive_reward_increases_selected_axis_probabilities(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))
        candidate = dmp.generate_candidates()[0]
        policy = PolicyABC.uniform_gridworld(learning_rate=1.0, seed=0)
        what, how, where = candidate_axes(candidate)
        before = (
            policy.policy_a[what],
            policy.policy_b[how],
            policy.policy_c[where],
        )

        policy.update(candidate, reward=1.0)

        self.assertGreater(policy.policy_a[what], before[0])
        self.assertGreater(policy.policy_b[how], before[1])
        self.assertGreater(policy.policy_c[where], before[2])

    def test_negative_reward_decreases_selected_axis_probabilities(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))
        candidate = dmp.generate_candidates()[0]
        policy = PolicyABC.uniform_gridworld(learning_rate=1.0, seed=0)
        what, how, where = candidate_axes(candidate)
        before = (
            policy.policy_a[what],
            policy.policy_b[how],
            policy.policy_c[where],
        )

        policy.update(candidate, reward=-1.0)

        self.assertLess(policy.policy_a[what], before[0])
        self.assertLess(policy.policy_b[how], before[1])
        self.assertLess(policy.policy_c[where], before[2])

    def test_dmp_updates_policyabc_after_step(self) -> None:
        policy = PolicyABC.uniform_gridworld(learning_rate=1.0, seed=0)
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)), scorer=policy)
        candidate = dmp.generate_candidates()[0]
        what, how, where = candidate_axes(candidate)
        before = (
            policy.policy_a[what],
            policy.policy_b[how],
            policy.policy_c[where],
        )

        dmp.execute(candidate)

        self.assertNotEqual(policy.policy_a[what], before[0])
        self.assertNotEqual(policy.policy_b[how], before[1])
        self.assertNotEqual(policy.policy_c[where], before[2])

    def test_policyabc_keeps_minimum_probability_floor(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))
        candidate = dmp.generate_candidates()[0]
        policy = PolicyABC.uniform_gridworld(learning_rate=10.0, min_prob=0.02, seed=0)

        for _ in range(20):
            policy.update(candidate, reward=-1.0)

        self.assertGreaterEqual(min(policy.policy_a.values()), 0.02 / 1.08)
        self.assertGreaterEqual(min(policy.policy_b.values()), 0.02 / 1.12)
        self.assertGreaterEqual(min(policy.policy_c.values()), 0.02 / 1.16)

    def test_policyabc_samples_what_how_where_before_binding(self) -> None:
        policy = PolicyABC(
            policy_a={"INSPECT_CELL": 1.0},
            policy_b={"random": 1.0},
            policy_c={"KK_UNKNOWN_NEIGHBOR": 1.0},
            min_prob=0.0,
            seed=0,
        )

        self.assertEqual(
            policy.sample_axes(),
            ("INSPECT_CELL", "random", "KK_UNKNOWN_NEIGHBOR"),
        )


if __name__ == "__main__":
    unittest.main()
