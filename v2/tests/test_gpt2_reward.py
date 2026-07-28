import unittest

from aassr.gpt2_reward import (
    ActionErrorKind,
    ActionableGridWorldDMP,
    ActionableRewardModule,
    semantic_candidate_signature,
)
from aassr.gridworld import ActionCandidate, ActionName, DMPConfig, GridWorld
from aassr.knowledge import KK, KV, KnowledgeDelta, KnowledgeStatus, ValueType


class GPT2RewardTests(unittest.TestCase):
    def test_semantic_signature_ignores_how_and_current_position(self) -> None:
        first = ActionCandidate(
            name=ActionName.MOVE_TOWARD,
            template="MOVE_TOWARD {KK_FRONTIER_CELL}",
            required_kk_slots=(KK.CURRENT_POS, KK.FRONTIER_CELL),
            bindings={KK.CURRENT_POS: (0, 0), KK.FRONTIER_CELL: (2, 0)},
            strategy="nearest",
        )
        second = ActionCandidate(
            name=ActionName.MOVE_TOWARD,
            template=first.template,
            required_kk_slots=first.required_kk_slots,
            bindings={KK.CURRENT_POS: (1, 0), KK.FRONTIER_CELL: (2, 0)},
            strategy="random",
        )
        self.assertEqual(semantic_candidate_signature(first), semantic_candidate_signature(second))

    def test_unlock_reward_is_positive_without_named_kk_weight(self) -> None:
        module = ActionableRewardModule()
        reward = module.compute(
            delta_k=KnowledgeDelta(),
            newly_unlocked_actions=2,
            newly_locked_actions=0,
            error_kind=ActionErrorKind.NONE,
            repeated=False,
            semantic_repeat=False,
            cycle_repeat=False,
            flag_found=False,
        )
        self.assertGreater(reward.total_reward, 0.0)

    def test_stagnant_repeat_is_penalized(self) -> None:
        module = ActionableRewardModule()
        reward = module.compute(
            delta_k=KnowledgeDelta(),
            newly_unlocked_actions=0,
            newly_locked_actions=0,
            error_kind=ActionErrorKind.NONE,
            repeated=True,
            semantic_repeat=True,
            cycle_repeat=False,
            flag_found=False,
        )
        self.assertLess(reward.total_reward, 0.0)

    def test_any_resolved_lifecycle_transition_receives_progress(self) -> None:
        kv = KV(
            value=(1, 1),
            type=ValueType.CELL_COORD,
            status=KnowledgeStatus.BLOCKED,
        )
        module = ActionableRewardModule()
        reward = module.compute(
            delta_k=KnowledgeDelta(status_changed=((KK.WALL_CELL, kv),)),
            newly_unlocked_actions=0,
            newly_locked_actions=0,
            error_kind=ActionErrorKind.NONE,
            repeated=False,
            semantic_repeat=False,
            cycle_repeat=False,
            flag_found=False,
        )
        self.assertGreater(reward.total_reward, 0.0)
        self.assertEqual(module.last_diagnostics.lifecycle_progress_count, 1)

    def test_actionable_dmp_forces_curiosity_off(self) -> None:
        dmp = ActionableGridWorldDMP(
            GridWorld(width=3, height=3, start=(1, 1)),
            config=DMPConfig(prediction_error_mode="curiosity"),
        )
        self.assertEqual(dmp.config.prediction_error_mode, "disabled")


if __name__ == "__main__":
    unittest.main()
