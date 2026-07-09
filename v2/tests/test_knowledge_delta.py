import unittest

from aassr import KK, KnowledgeStore, ValueType
from aassr.reward import RewardModule


class KnowledgeDeltaTests(unittest.TestCase):
    def test_usage_only_update_is_not_semantic_information_gain(self) -> None:
        store = KnowledgeStore()
        store.add(KK.KNOWN_CELL, (1, 1), ValueType.CELL_COORD)
        before = store.snapshot_items()

        store.mark_used(KK.KNOWN_CELL, (1, 1), success=True, step=1)
        delta = store.delta_since(before)

        self.assertEqual(delta.semantic_information_gain(), 0)
        self.assertEqual(delta.usage_change_count(), 1)
        self.assertFalse(delta.has_semantic_changes())
        self.assertEqual(delta.semantic_changed_kk(), set())

    def test_reward_ignores_usage_only_update(self) -> None:
        store = KnowledgeStore()
        store.add(KK.KNOWN_CELL, (1, 1), ValueType.CELL_COORD)
        before = store.snapshot_items()
        store.mark_used(KK.KNOWN_CELL, (1, 1), success=True, step=1)
        delta = store.delta_since(before)

        reward = RewardModule().compute(
            delta_k=delta,
            error=False,
            repeated=False,
            flag_found=False,
        )

        self.assertEqual(reward.intrinsic_reward, 0.0)
        self.assertEqual(reward.total_reward, 0.0)

    def test_current_pos_update_is_not_semantic_information_gain(self) -> None:
        store = KnowledgeStore()
        store.set_singleton(KK.CURRENT_POS, (1, 1), ValueType.CELL_COORD)
        before = store.snapshot_items()

        store.set_singleton(KK.CURRENT_POS, (1, 2), ValueType.CELL_COORD, step=1)
        delta = store.delta_since(before)

        self.assertEqual(delta.semantic_information_gain(), 0)
        self.assertIn(KK.CURRENT_POS, delta.semantic_changed_kk())


if __name__ == "__main__":
    unittest.main()
