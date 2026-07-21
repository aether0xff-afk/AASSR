import unittest

import numpy as np

from aassr import CellKind, DMPConfig, GridWorld, GridWorldDMP, KK, KnowledgeStatus
from aassr.experiment import ExperimentComponents, ExperimentCondition
from aassr.gridworld import ActionName
from aassr.imagination import ImaginationConfig, PredictedStateImaginationCycle
from aassr.policy import PolicyABC
from aassr.prophecy import ProphecyPrediction, gridworld_knowledge_state_signature


class RuleProphecy:
    def __init__(self) -> None:
        self.predict_calls = []

    def predict(self, state_signature, candidate):
        self.predict_calls.append((state_signature, candidate))
        probs = {kk: 0.0 for kk in KK}
        if candidate.name == ActionName.INSPECT_CELL:
            probs[KK.KEY_OBJECT] = 1.0
        if candidate.name == ActionName.USE_OBJECT:
            return ProphecyPrediction(probs, error_prob=0.0, flag_prob=1.0, confidence=1.0)
        return ProphecyPrediction(probs, error_prob=0.0, flag_prob=0.0, confidence=1.0)

    def update(self, *args, **kwargs):
        raise AssertionError("full imagination test should not execute prophecy updates")


class SentinelWorld(GridWorld):
    @property
    def flag_position(self):
        raise AssertionError("hidden flag position was read")

    @property
    def key_position(self):
        raise AssertionError("hidden key position was read")

    def kind_at(self, cell):
        raise AssertionError("hidden kind_at was read")

    def inspect(self, cell):
        raise AssertionError("future action execution was attempted")


class APASSRFullTests(unittest.TestCase):
    def _door_setup(self, world_type=GridWorld) -> GridWorldDMP:
        world = world_type(width=3, height=2, start=(0, 0))
        dmp = GridWorldDMP(world, config=DMPConfig(independent_policy_axes=True))
        dmp.store.add(KK.DOOR_CELL, (1, 0), value_type=dmp.store.values(KK.CURRENT_POS)[0].type)
        return dmp

    def test_future_candidate_is_regenerated_after_predicted_key_gain(self) -> None:
        dmp = self._door_setup()
        initial = dmp.generate_candidates()
        self.assertFalse(any(candidate.name == ActionName.USE_OBJECT for candidate in initial))
        prophecy = RuleProphecy()
        imagination = PredictedStateImaginationCycle(
            prophecy,
            ImaginationConfig(rollout_depth=2, predicted_delta_threshold=0.5, policy_prior_weight=0.0),
        )

        trace = imagination.choose("ignored", initial, dmp=dmp)

        selected_trajectory = next(item for item in trace.trajectories if item.steps[0].action == trace.selected)
        future_actions = [step.action.name for step in selected_trajectory.steps[1:]]
        self.assertIn(ActionName.USE_OBJECT, future_actions)

    def test_imagined_state_signature_changes_after_key_gain(self) -> None:
        dmp = self._door_setup()
        initial = dmp.generate_candidates()
        imagination = PredictedStateImaginationCycle(
            RuleProphecy(),
            ImaginationConfig(rollout_depth=2, predicted_delta_threshold=0.5),
        )

        trace = imagination.choose("ignored", initial, dmp=dmp)
        trajectory = next(item for item in trace.trajectories if len(item.steps) >= 2)
        before = gridworld_knowledge_state_signature(
            trajectory.steps[0].state_before.knowledge,
            position=trajectory.steps[0].state_before.position_or_position_belief,
            width=dmp.world.width,
            height=dmp.world.height,
        )
        after = gridworld_knowledge_state_signature(
            trajectory.steps[0].state_after.knowledge,
            position=trajectory.steps[0].state_after.position_or_position_belief,
            width=dmp.world.width,
            height=dmp.world.height,
            last_semantic_delta=trajectory.steps[0].state_after.last_semantic_delta,
        )

        self.assertNotEqual(before, after)
        self.assertIn(("has_key", False), before)
        self.assertIn(("has_key", True), after)

    def test_prophecy_receives_changed_state_each_rollout_depth(self) -> None:
        dmp = self._door_setup()
        prophecy = RuleProphecy()
        imagination = PredictedStateImaginationCycle(prophecy, ImaginationConfig(rollout_depth=2))

        imagination.choose("ignored", dmp.generate_candidates(), dmp=dmp)

        signatures = [call[0] for call in prophecy.predict_calls]
        self.assertGreater(len(set(signatures)), 1)

    def test_full_imagination_does_not_read_hidden_map(self) -> None:
        dmp = self._door_setup(SentinelWorld)
        prophecy = RuleProphecy()
        imagination = PredictedStateImaginationCycle(prophecy, ImaginationConfig(rollout_depth=2))

        trace = imagination.choose("ignored", dmp.generate_candidates(), dmp=dmp)

        self.assertTrue(trace.scores)

    def test_policy_b_changes_how_selection_frequency(self) -> None:
        policy = PolicyABC.uniform_gridworld(seed=4)
        policy.policy_b.update({"least_tried": 0.85, "random": 0.01, "nearest": 0.05, "high_uncertainty": 0.05})
        policy._normalize(policy.policy_b)
        dmp = GridWorldDMP(
            GridWorld(width=3, height=3, start=(1, 1)),
            scorer=policy,
            config=DMPConfig(independent_policy_axes=True),
        )
        candidates = [
            candidate
            for candidate in dmp.generate_candidates()
            if candidate.name == ActionName.INSPECT_CELL and KK.UNKNOWN_NEIGHBOR in candidate.bindings
        ]
        self.assertGreater(len({candidate.strategy for candidate in candidates}), 1)

        counts = {"least_tried": 0, "random": 0}
        for _ in range(300):
            selected = policy.choose(candidates, dmp)
            if selected.strategy in counts:
                counts[selected.strategy] += 1

        self.assertGreater(counts["least_tried"], counts["random"])

    def test_sequence_context_resets_between_episodes_but_weights_remain(self) -> None:
        components = ExperimentComponents.for_condition(ExperimentCondition.C4, seed=7)
        dmp = components.make_dmp(GridWorld(width=3, height=3, start=(1, 1)), step_limit=5)
        candidate = dmp.choose_candidate("scorer")
        dmp.execute(candidate)
        self.assertGreater(float(np.linalg.norm(components.prophecy._context)), 0.0)
        weights_before = components.prophecy._wk.copy()

        components.make_dmp(GridWorld(width=3, height=3, start=(1, 1)), step_limit=5)

        self.assertEqual(float(np.linalg.norm(components.prophecy._context)), 0.0)
        self.assertTrue(np.array_equal(weights_before, components.prophecy._wk))

    def test_multiple_keys_get_unique_objects_and_consumable_lifecycle(self) -> None:
        world = GridWorld(
            width=5,
            height=1,
            start=(0, 0),
            cells={(1, 0): CellKind.KEY, (2, 0): CellKind.KEY, (3, 0): CellKind.DOOR, (4, 0): CellKind.FLAG},
        )
        dmp = GridWorldDMP(world)
        dmp.execute(next(candidate for candidate in dmp.generate_candidates() if candidate.bindings.get(KK.UNKNOWN_NEIGHBOR) == (1, 0)))
        dmp.execute(next(candidate for candidate in dmp.generate_candidates() if candidate.bindings.get(KK.KEY_CELL) == (1, 0)))
        dmp.execute(next(candidate for candidate in dmp.generate_candidates() if candidate.bindings.get(KK.UNKNOWN_NEIGHBOR) == (2, 0)))
        dmp.execute(next(candidate for candidate in dmp.generate_candidates() if candidate.bindings.get(KK.KEY_CELL) == (2, 0)))

        keys = dmp.store.values(KK.KEY_OBJECT, include_inactive=True)
        self.assertEqual(len({key.value for key in keys}), 2)

        dmp.execute(next(candidate for candidate in dmp.generate_candidates() if candidate.bindings.get(KK.UNKNOWN_NEIGHBOR) == (3, 0)))
        dmp.execute(next(candidate for candidate in dmp.generate_candidates() if candidate.name == ActionName.USE_OBJECT))

        statuses = [key.status for key in dmp.store.values(KK.KEY_OBJECT, include_inactive=True)]
        self.assertIn(KnowledgeStatus.CONSUMED, statuses)
        self.assertIn(KnowledgeStatus.ACTIVE, statuses)

    def test_apassr_full_condition_enables_full_structure(self) -> None:
        components = ExperimentComponents.for_condition(ExperimentCondition.APASSR_FULL, seed=0)
        dmp = components.make_dmp(GridWorld(width=3, height=3, start=(1, 1)), step_limit=5)

        self.assertTrue(components.use_prophecy)
        self.assertTrue(components.use_imagination)
        self.assertTrue(components.full_imagination)
        self.assertTrue(components.independent_policy_axes)
        self.assertIsInstance(dmp.imagination, PredictedStateImaginationCycle)
        self.assertGreater(components.imagination_config.rollout_depth, 1)
        self.assertFalse(components.imagination_config.calibrated_imagination_enabled)
        self.assertFalse(components.imagination_config.candidate_dedup_enabled)

    def test_apassr_full_cal_condition_is_isolated_from_full_and_c_conditions(self) -> None:
        full = ExperimentComponents.for_condition(ExperimentCondition.APASSR_FULL, seed=0)
        calibrated = ExperimentComponents.for_condition(ExperimentCondition.APASSR_FULL_CAL, seed=0)
        c3 = ExperimentComponents.for_condition(ExperimentCondition.C3, seed=0)
        c5 = ExperimentComponents.for_condition(ExperimentCondition.C5, seed=0)

        self.assertTrue(calibrated.full_imagination)
        self.assertTrue(calibrated.imagination_config.calibrated_imagination_enabled)
        self.assertTrue(calibrated.imagination_config.candidate_dedup_enabled)
        self.assertFalse(full.imagination_config.calibrated_imagination_enabled)
        self.assertFalse(full.imagination_config.candidate_dedup_enabled)
        self.assertFalse(c3.full_imagination)
        self.assertFalse(c3.imagination_config.calibrated_imagination_enabled)
        self.assertFalse(c5.full_imagination)
        self.assertFalse(c5.imagination_config.calibrated_imagination_enabled)


if __name__ == "__main__":
    unittest.main()
