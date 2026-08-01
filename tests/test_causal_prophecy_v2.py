from __future__ import annotations

from aassr_v2.causal_dependency_world import CausalDependencyWorldV2
from aassr_v2.causal_agent_v2 import CausalAASSRAgent
from aassr_v2.causal_prophecy import EmpiricalCausalProphecy
from aassr_v2.causal_representation import (
    ObservableTransition,
    RelationalEffectEncoder,
)
from aassr_v2.paper_v2_protocol import clone_agent_from_checkpoint


def transition_for(world: CausalDependencyWorldV2, action: str) -> ObservableTransition:
    before = world.observe()
    outcome = world.step(action)
    return ObservableTransition(
        before,
        action,
        outcome.observation,
        outcome.action_succeeded,
        outcome.inventory_delta,
        len(outcome.facts_added),
        len(outcome.facts_removed),
        len(outcome.unlocked_actions),
        outcome.resource_cost,
        outcome.damage,
        outcome.spatial_change is not None,
        outcome.reward,
    )


def action_for(world: CausalDependencyWorldV2, key: str) -> str:
    return next(
        action
        for action in world.observe().available_actions
        if world.private_action_key(action) == key
    )


def test_v20_learns_observable_heads() -> None:
    world = CausalDependencyWorldV2(world_seed=82001)
    prophecy = EmpiricalCausalProphecy(RelationalEffectEncoder())
    action = action_for(world, "collect_wood")
    transition = transition_for(world, action)
    prophecy.observe_transition(transition)
    prediction = prophecy.predict_v20(transition.before, action)
    assert prediction.visit_count == 1
    assert prediction.observable_effect_delta["inventory_change"] == 1.0
    assert prediction.next_observable_state == transition.after


def test_success_head_uses_only_observed_terminal_outcome() -> None:
    success_world = CausalDependencyWorldV2(world_seed=82001)
    prophecy = EmpiricalCausalProphecy(RelationalEffectEncoder(), gamma=1.0)
    action = action_for(success_world, "scan")
    transition = transition_for(success_world, action)
    prophecy.observe_transition(transition)
    prophecy.finish_episode(True)
    success_probability = prophecy.predict_v20(transition.before, action).terminal_return_probability
    assert success_probability == 1.0

    failed_world = CausalDependencyWorldV2(world_seed=82001)
    failed_action = action_for(failed_world, "scan")
    failed_transition = transition_for(failed_world, failed_action)
    prophecy.observe_transition(failed_transition)
    prophecy.finish_episode(False)
    probability = prophecy.predict_v20(failed_transition.before, failed_action).terminal_return_probability
    assert probability == 0.5


def test_v21_uncertainty_falls_with_observable_evidence() -> None:
    prophecy = EmpiricalCausalProphecy(RelationalEffectEncoder())
    predictions = []
    for _ in range(8):
        world = CausalDependencyWorldV2(world_seed=82001)
        action = action_for(world, "scan")
        before = world.observe()
        predictions.append(prophecy.predict_v21(before, action).uncertainty)
        prophecy.observe_transition(transition_for(world, action))
        prophecy.finish_episode(False)
    final_world = CausalDependencyWorldV2(world_seed=82001)
    final_action = action_for(final_world, "scan")
    assert prophecy.predict_v21(final_world.observe(), final_action).uncertainty < predictions[0]


def test_unseen_action_is_high_ood_and_cannot_be_confident() -> None:
    world = CausalDependencyWorldV2(world_seed=82001)
    prophecy = EmpiricalCausalProphecy(RelationalEffectEncoder())
    action = world.observe().available_actions[0]
    prediction = prophecy.predict_v21(world.observe(), action)
    assert prediction.ood_score == 1.0
    assert prediction.calibration_confidence == 0.0


def test_full_checkpoint_clones_policy_prophecy_and_relational_memory() -> None:
    agent = CausalAASSRAgent(RelationalEffectEncoder, seed=11)
    world = CausalDependencyWorldV2(world_seed=82001)
    action = action_for(world, "scan")
    transition = transition_for(world, action)
    agent.observe_transition(transition)
    agent.finish_episode(False)
    clone, _ = clone_agent_from_checkpoint(
        agent, lambda: CausalAASSRAgent(RelationalEffectEncoder, seed=11)
    )
    assert clone.prophecy.total_updates == 1
    assert clone.policy.encoder.memory.update_count == 1
