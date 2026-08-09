from __future__ import annotations

from types import MethodType

import pytest

torch = pytest.importorskip("torch")

from aassr_v2.action_plugins import PluginOutcome
from aassr_v2.current_entrypoint import build_current_pentest_aassr_core
from aassr_v2.current_generation import (
    CurrentRelationalPolicy,
    relational_action_key,
    relational_state_vector,
)
from aassr_v2.current_hardware import (
    HardwareCurrentRelationalPolicy,
    HardwareRelationalInvariantDQN,
)
import aassr_v2.current_generation as current_generation_module
import aassr_v2.current_hardware as current_hardware_module
from aassr_v2.current_runtime import FullyRelationalNeuralDeltaProphecy
from aassr_v2.neural_delta_prophecy import NeuralDeltaConfig
from aassr_v2.pentest_agent_main_test import ACTION_FEATURE_SIZE, HttpAgentCodec
from aassr_v2.pentest_curriculum_env import relational_action_features
from aassr_v2.pentest_transfer_stages import (
    TRANSFER_STAGES,
    TransferDiagnosticWorld,
)
from aassr_v2.replay import ReplayTransition


def _world_state(seed: int = 90_001):
    world = TransferDiagnosticWorld(seed, stage=TRANSFER_STAGES[0])
    state = world.snapshot()
    return world, state


def _small_neural(*, state_capacity: int = 4, input_capacity: int = 8):
    return FullyRelationalNeuralDeltaProphecy(
        HttpAgentCodec(),
        config=NeuralDeltaConfig(
            action_feature_size=ACTION_FEATURE_SIZE,
            hidden_units=8,
            ensemble_size=1,
            replay_capacity=8,
            batch_size=2,
            warmup_steps=2,
            gradient_steps_per_observation=1,
        ),
        seed=7,
        device="cpu",
        state_encoding_cache_capacity=state_capacity,
        model_input_cache_capacity=input_capacity,
    )


def _reference_pair_scores(dqn, states, actions):
    keys = tuple(
        dqn.encode_state(state) + relational_action_features(state, action)
        for state, action in zip(states, actions, strict=True)
    )
    unique = []
    index_by_key = {}
    inverse = []
    for key in keys:
        index = index_by_key.get(key)
        if index is None:
            index = len(unique)
            index_by_key[key] = index
            unique.append(key)
        inverse.append(index)
    with dqn.torch.no_grad():
        values = dqn.online(dqn._tensor(unique)).squeeze(1).detach().cpu().tolist()
    return tuple(float(values[index]) for index in inverse)


def _reference_policy_rank(policy, state, *, limit):
    actions = tuple(state.available_actions)
    primitive = tuple(action for action in actions if action.verb_name != "skill")
    primitive_values = policy.dqn.score_actions(state, primitive) if primitive else ()
    by_signature = {
        action.signature: value
        for action, value in zip(primitive, primitive_values, strict=True)
    }
    rows = []
    for action in actions:
        if action.verb_name == "skill":
            entry = policy._skill_values.get(str(action.target))
            base = 0.0 if entry is None else entry.mean
        else:
            base = by_signature[action.signature] + policy._information_entry(
                state,
                action,
            ).mean
        rows.append((action, float(base)))
    rows.sort(key=lambda item: (-item[1], item[0].signature))
    return tuple((item[0].signature, item[1]) for item in rows[:limit])


def test_neural_identity_cache_matches_scalar_reference_and_never_aliases_equal_clones():
    neural = _small_neural()
    world, first_state = _world_state()
    second_state = world.snapshot()
    assert second_state == first_state
    assert second_state is not first_state
    first_action = first_state.available_actions[0]
    second_action = second_state.available_actions[0]
    assert second_action == first_action
    assert second_action is not first_action

    expected = relational_state_vector(first_state) + relational_action_key(
        first_state,
        first_action,
    )
    first = neural._input(first_state, first_action)
    repeated = neural._input(first_state, first_action)
    cloned = neural._input(second_state, second_action)

    assert first == expected
    assert repeated is first
    assert cloned == expected
    diagnostics = neural.diagnostics()
    assert diagnostics["model_input_cache_hits"] == 1
    assert diagnostics["model_input_cache_misses"] == 2
    assert diagnostics["state_encoding_cache_hits"] == 0
    assert diagnostics["state_encoding_cache_misses"] == 2


def test_neural_identity_cache_eviction_is_bounded_and_recomputes_exactly():
    neural = _small_neural(state_capacity=2, input_capacity=2)
    world, _ = _world_state()
    pairs = []
    for _ in range(3):
        state = world.snapshot()
        pairs.append((state, state.available_actions[0]))

    expected = []
    for state, action in pairs:
        expected.append(
            relational_state_vector(state) + relational_action_key(state, action)
        )
        assert neural._input(state, action) == expected[-1]

    diagnostics = neural.diagnostics()
    assert diagnostics["state_encoding_cache_entries"] == 2
    assert diagnostics["model_input_cache_entries"] == 2
    assert diagnostics["state_encoding_cache_evictions"] == 1
    assert diagnostics["model_input_cache_evictions"] == 1

    misses_before = diagnostics["model_input_cache_misses"]
    assert neural._input(*pairs[0]) == expected[0]
    diagnostics = neural.diagnostics()
    assert diagnostics["model_input_cache_misses"] == misses_before + 1
    assert diagnostics["state_encoding_cache_entries"] <= 2
    assert diagnostics["model_input_cache_entries"] <= 2


def test_fast_validator_reuses_exact_model_inputs_after_model_revision():
    agent = build_current_pentest_aassr_core(
        seed=11,
        train_transitions=256,
        device="cpu",
    )
    world, _ = _world_state()
    rows = []
    for index in range(70):
        state = world.snapshot()
        rows.append(
            ReplayTransition(
                state,
                state.available_actions[0],
                state,
                trace_id=f"cache-row-{index}",
            )
        )

    neural = agent.base_neural_prophecy
    neural.observations = neural.config.warmup_steps
    first = agent.evaluator.validator.evaluate(agent.prophecy, rows)
    diagnostics_before = neural.diagnostics()
    neural.gradient_updates += 1
    second = agent.evaluator.validator.evaluate(agent.prophecy, rows)
    diagnostics_after = neural.diagnostics()

    assert first.count == second.count == 64
    assert second.mean_similarity == pytest.approx(
        first.mean_similarity,
        abs=0.0,
        rel=0.0,
    )
    assert diagnostics_after["model_input_cache_hits"] >= (
        diagnostics_before["model_input_cache_hits"] + 64
    )
    validator = agent.evaluator.validator.runtime_diagnostics()
    assert validator["cache_misses"] == 2


def test_hardware_pair_batch_encodes_identical_state_once_and_matches_reference():
    optimized = HardwareRelationalInvariantDQN(
        17,
        train_transitions=256,
        device="cpu",
    )
    reference = HardwareRelationalInvariantDQN(
        17,
        train_transitions=256,
        device="cpu",
    )
    world, state = _world_state()
    clone = world.snapshot()
    action = state.available_actions[0]
    clone_action = clone.available_actions[0]
    states = (state, state, clone, state)
    actions = (action, action, clone_action, action)
    expected = _reference_pair_scores(reference, states, actions)

    original = optimized.encode_state
    encoded_objects = []

    def counted(self, item):
        encoded_objects.append(item)
        return original(item)

    optimized.encode_state = MethodType(counted, optimized)
    actual = optimized.score_state_action_batch(states, actions)

    assert actual == expected
    assert len(encoded_objects) == 2
    assert encoded_objects[0] is state
    assert encoded_objects[1] is clone
    stats = optimized.model_stats()
    assert stats["pair_state_encoding_rows"] == 4
    assert stats["pair_state_unique_encodings"] == 2


def test_hardware_dqn_same_seed_actions_and_updates_remain_exact():
    left = HardwareRelationalInvariantDQN(
        23,
        train_transitions=256,
        device="cpu",
    )
    right = HardwareRelationalInvariantDQN(
        23,
        train_transitions=256,
        device="cpu",
    )
    world, before = _world_state(90_023)
    action = before.available_actions[0]
    after = world.snapshot()

    left_decision = left.select_action(before, transition=0, training=False)
    right_decision = right.select_action(before, transition=0, training=False)
    assert left_decision.action.signature == right_decision.action.signature

    outcome = PluginOutcome(snapshot=after)
    for index in range(left.warmup_steps):
        reward = float(index == left.warmup_steps - 1)
        left.observe(before, action, outcome, reward=reward)
        right.observe(before, action, outcome, reward=reward)

    assert left.gradient_updates == right.gradient_updates == 1
    for name, tensor in left.online.state_dict().items():
        assert torch.equal(tensor, right.online.state_dict()[name]), name
    assert tuple(left.replay) == tuple(right.replay)


def test_policy_rank_reuses_one_relational_state_key_and_matches_scalar_reference(
    monkeypatch,
):
    world, entry = _world_state(90_031)
    state = world.step(entry.available_actions[0]).snapshot
    assert len(state.available_actions) > 1

    reference = CurrentRelationalPolicy(
        HardwareRelationalInvariantDQN(31, train_transitions=256, device="cpu")
    )
    optimized = CurrentRelationalPolicy(
        HardwareRelationalInvariantDQN(31, train_transitions=256, device="cpu")
    )
    chosen = state.available_actions[0]
    reference.observe_information_return(state, chosen, 0.75)
    optimized.observe_information_return(state, chosen, 0.75)
    expected = _reference_policy_rank(reference, state, limit=len(state.available_actions))

    original = current_generation_module.relational_state_key
    calls = 0

    def counted(item):
        nonlocal calls
        calls += 1
        return original(item)

    monkeypatch.setattr(current_generation_module, "relational_state_key", counted)
    actual_rows = optimized.rank(
        state,
        limit=len(state.available_actions),
    )
    actual = tuple((item.action.signature, item.score) for item in actual_rows)

    assert actual == expected
    assert calls == 1


def test_hardware_policy_batch_reuses_one_state_key_per_frontier_state(
    monkeypatch,
):
    world, entry = _world_state(90_032)
    state = world.step(entry.available_actions[0]).snapshot
    states = (state, state)
    limits = (len(state.available_actions), len(state.available_actions))
    memories = (None, None)

    reference = CurrentRelationalPolicy(
        HardwareRelationalInvariantDQN(32, train_transitions=256, device="cpu")
    )
    optimized = HardwareCurrentRelationalPolicy(
        HardwareRelationalInvariantDQN(32, train_transitions=256, device="cpu")
    )
    expected = tuple(
        _reference_policy_rank(reference, item, limit=limit)
        for item, limit in zip(states, limits, strict=True)
    )

    original = current_hardware_module.relational_state_key
    calls = 0

    def counted(item):
        nonlocal calls
        calls += 1
        return original(item)

    monkeypatch.setattr(current_hardware_module, "relational_state_key", counted)
    actual_rows = optimized.rank_batch(states, limits, memories)
    actual = tuple(
        tuple((item.action.signature, item.score) for item in rows)
        for rows in actual_rows
    )

    assert actual == expected
    assert calls == len(states)
