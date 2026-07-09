import unittest

from aassr import CellKind, GridWorld, GridWorldDMP, KK, KnowledgeSource, KnowledgeStatus
from aassr.gridworld import ActionName


class GridWorldDMPTests(unittest.TestCase):
    def test_initial_seed_generates_inspect_candidates(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))

        candidates = dmp.generate_candidates()

        self.assertTrue(any(candidate.name == ActionName.INSPECT_CELL for candidate in candidates))
        self.assertTrue(dmp.store.has_active(KK.CURRENT_POS))
        self.assertEqual(len(dmp.store.values(KK.CURRENT_POS)), 1)
        self.assertTrue(dmp.store.has_active(KK.DIRECTION))
        self.assertTrue(dmp.store.has_active(KK.SELF))


    def test_frontier_discovery_creates_move_toward_candidate(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))
        inspect = next(
            candidate
            for candidate in dmp.generate_candidates()
            if candidate.name == ActionName.INSPECT_CELL
        )

        result = dmp.execute(inspect)

        self.assertFalse(result.delta_k.is_empty())
        self.assertTrue(
            any(
                candidate.name == ActionName.MOVE_TOWARD
                and KK.FRONTIER_CELL in candidate.bindings
                for candidate in dmp.generate_candidates()
            )
        )

    def test_current_pos_stays_singleton_after_movement(self) -> None:
        world = GridWorld(width=3, height=3, start=(1, 1), cells={(1, 2): CellKind.KEY})
        dmp = GridWorldDMP(world)
        inspect_key = next(
            candidate
            for candidate in dmp.generate_candidates()
            if candidate.bindings.get(KK.UNKNOWN_NEIGHBOR) == (1, 2)
        )
        dmp.execute(inspect_key)
        move_to_key = next(
            candidate
            for candidate in dmp.generate_candidates()
            if candidate.name == ActionName.MOVE_TOWARD
            and candidate.bindings.get(KK.KEY_CELL) == (1, 2)
        )

        dmp.execute(move_to_key)

        current_positions = dmp.store.values(KK.CURRENT_POS, include_inactive=True)
        self.assertEqual(len(current_positions), 1)
        self.assertEqual(current_positions[0].value, (1, 2))
        self.assertTrue(dmp.store.has_active(KK.VISITED_CELL))


    def test_confirmed_wall_is_excluded_from_movement_candidates(self) -> None:
        world = GridWorld(width=3, height=3, start=(1, 1), cells={(1, 0): CellKind.WALL})
        dmp = GridWorldDMP(world)
        wall_inspect = next(
            candidate
            for candidate in dmp.generate_candidates()
            if candidate.bindings.get(KK.UNKNOWN_NEIGHBOR) == (1, 0)
        )

        result = dmp.execute(wall_inspect)

        self.assertIn(KK.WALL_CELL, result.delta_k.changed_kk())
        self.assertFalse(
            any(
                candidate.name == ActionName.MOVE_TOWARD
                and (1, 0) in candidate.bindings.values()
                for candidate in dmp.generate_candidates()
            )
        )
        wall_values = dmp.store.values(KK.WALL_CELL, include_inactive=True)
        self.assertEqual(wall_values[0].value, (1, 0))
        self.assertEqual(wall_values[0].status, KnowledgeStatus.BLOCKED)


    def test_hint_creates_hint_cell_and_inferred_flag_candidate(self) -> None:
        world = GridWorld(
            width=4,
            height=1,
            start=(0, 0),
            cells={(1, 0): CellKind.HINT},
            hints={(1, 0): (3, 0)},
        )
        dmp = GridWorldDMP(world)
        hint_inspect = next(iter(dmp.generate_candidates()))

        dmp.execute(hint_inspect)
        self.assertTrue(dmp.store.has_active(KK.HINT_VALUE))
        follow_hint = next(
            candidate
            for candidate in dmp.generate_candidates()
            if candidate.name == ActionName.FOLLOW_HINT
        )
        dmp.execute(follow_hint)

        self.assertTrue(dmp.store.has_active(KK.HINT_CELL))
        flag_values = dmp.store.values(KK.FLAG_CELL)
        self.assertEqual(flag_values[0].value, (3, 0))
        self.assertEqual(flag_values[0].source, KnowledgeSource.INFERRED)
        self.assertEqual(
            dmp.store.values(KK.HINT_VALUE, include_inactive=True)[0].status,
            KnowledgeStatus.CONSUMED,
        )


    def test_key_acquire_and_door_open_lifecycle(self) -> None:
        world = GridWorld(
            width=4,
            height=1,
            start=(0, 0),
            cells={(1, 0): CellKind.KEY, (2, 0): CellKind.DOOR, (3, 0): CellKind.FLAG},
        )
        dmp = GridWorldDMP(world)
        dmp.execute(next(iter(dmp.generate_candidates())))

        move_to_key = next(
            candidate
            for candidate in dmp.generate_candidates()
            if candidate.name == ActionName.MOVE_TOWARD
            and candidate.bindings.get(KK.KEY_CELL) == (1, 0)
        )
        dmp.execute(move_to_key)

        self.assertTrue(dmp.store.has_active(KK.KEY_OBJECT))
        self.assertEqual(
            dmp.store.values(KK.KEY_CELL, include_inactive=True)[0].status,
            KnowledgeStatus.CONSUMED,
        )

        door_inspect = next(
            candidate
            for candidate in dmp.generate_candidates()
            if candidate.name == ActionName.INSPECT_CELL
            and candidate.bindings.get(KK.UNKNOWN_NEIGHBOR) == (2, 0)
        )
        dmp.execute(door_inspect)

        use_key = next(
            candidate
            for candidate in dmp.generate_candidates()
            if candidate.name == ActionName.USE_OBJECT
        )
        result = dmp.execute(use_key)

        self.assertGreater(result.total_reward, 0.0)
        self.assertEqual(
            dmp.store.values(KK.DOOR_CELL, include_inactive=True)[0].status,
            KnowledgeStatus.CONSUMED,
        )
        self.assertFalse(
            any(
                candidate.name == ActionName.USE_OBJECT
                for candidate in dmp.generate_candidates()
            )
        )

    def test_policy_opens_door_before_reaching_flag(self) -> None:
        world = GridWorld(
            width=4,
            height=1,
            start=(0, 0),
            cells={(1, 0): CellKind.KEY, (2, 0): CellKind.DOOR, (3, 0): CellKind.FLAG},
        )
        dmp = GridWorldDMP(world)
        action_names = []

        for _ in range(20):
            candidate = dmp.choose_candidate("nearest")
            self.assertIsNotNone(candidate)
            action_names.append(candidate.name)
            dmp.execute(candidate)
            if dmp.position == (3, 0):
                break

        self.assertIn(ActionName.USE_OBJECT, action_names)
        self.assertLess(action_names.index(ActionName.USE_OBJECT), len(action_names) - 1)
        self.assertEqual(dmp.position, (3, 0))
        self.assertEqual(
            dmp.store.values(KK.DOOR_CELL, include_inactive=True)[0].status,
            KnowledgeStatus.CONSUMED,
        )

    def test_repeat_signature_ignores_current_position(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=6, height=4, start=(1, 1)))
        first = next(candidate for candidate in dmp.generate_candidates() if KK.FRONTIER_CELL in candidate.bindings)
        second = type(first)(
            name=first.name,
            template=first.template,
            required_kk_slots=first.required_kk_slots,
            bindings={**first.bindings, KK.CURRENT_POS: (2, 1)},
            strategy=first.strategy,
        )

        self.assertEqual(dmp._signature(first), dmp._signature(second))


if __name__ == "__main__":
    unittest.main()
