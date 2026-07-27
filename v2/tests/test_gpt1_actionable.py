import unittest

from aassr.actionable import (
    ActionableGridWorldDMP,
    semantic_candidate_signature,
)
from aassr.gridworld import ActionCandidate, ActionName, CellKind, GridWorld
from aassr.knowledge import KK


class ActionableRewardTests(unittest.TestCase):
    def test_newly_unlocked_action_receives_positive_reward(self) -> None:
        world = GridWorld(
            width=3,
            height=1,
            start=(0, 0),
            cells={(1, 0): CellKind.KEY, (2, 0): CellKind.FLAG},
        )
        dmp = ActionableGridWorldDMP(world, step_limit=20)
        inspect_key = next(
            candidate
            for candidate in dmp.generate_candidates()
            if candidate.bindings.get(KK.UNKNOWN_NEIGHBOR) == (1, 0)
        )

        result = dmp.execute(inspect_key)

        diagnostics = dmp.actionable_reward.last_diagnostics
        self.assertGreater(diagnostics.newly_unlocked_actions, 0)
        self.assertGreater(result.intrinsic_reward, 0.0)

    def test_repeating_same_semantic_action_without_change_is_penalized(self) -> None:
        dmp = ActionableGridWorldDMP(GridWorld(width=3, height=1, start=(0, 0)))
        candidate = next(iter(dmp.generate_candidates()))
        dmp.execute(candidate)

        result = dmp.execute(candidate)

        diagnostics = dmp.actionable_reward.last_diagnostics
        self.assertTrue(diagnostics.semantic_repeat)
        self.assertLess(result.intrinsic_reward, 0.0)

    def test_signature_deduplicates_how_labels(self) -> None:
        base = ActionCandidate(
            name=ActionName.MOVE_TOWARD,
            template="MOVE_TOWARD {KK_FRONTIER_CELL}",
            required_kk_slots=(KK.CURRENT_POS, KK.FRONTIER_CELL),
            bindings={KK.CURRENT_POS: (0, 0), KK.FRONTIER_CELL: (1, 0)},
            strategy="nearest",
        )
        alternate = ActionCandidate(
            name=base.name,
            template=base.template,
            required_kk_slots=base.required_kk_slots,
            bindings={KK.CURRENT_POS: (5, 5), KK.FRONTIER_CELL: (1, 0)},
            strategy="random",
        )

        self.assertEqual(
            semantic_candidate_signature(base),
            semantic_candidate_signature(alternate),
        )


if __name__ == "__main__":
    unittest.main()
