from __future__ import annotations

from aassr_v2.causal_dependency_world import CausalDependencyWorldV2
from aassr_v2.causal_representation import (
    IdentityEncoder,
    ObservableTransition,
    RelationalEffectEncoder,
    RepresentedReturnAgent,
)
from aassr_v2.representation_diagnostic import (
    run_diagnostic_two_b,
    run_episode,
)


def _action_for(world: CausalDependencyWorldV2, key: str) -> str:
    return next(
        action
        for action in world.observe().available_actions
        if world.private_action_key(action) == key
    )


def _observable_transition(world: CausalDependencyWorldV2, key: str) -> ObservableTransition:
    before = world.observe()
    action = _action_for(world, key)
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


def test_identity_changes_but_relational_state_preserves_isomorphism() -> None:
    first = CausalDependencyWorldV2(
        world_seed=82001, token_seed=92001, observation_seed=82001
    )
    second = CausalDependencyWorldV2(
        world_seed=83001, token_seed=92001, observation_seed=83001
    )
    identity = IdentityEncoder()
    relational = RelationalEffectEncoder()
    assert identity.state_key(first.observe()) != identity.state_key(second.observe())
    assert relational.state_key(first.observe()) == relational.state_key(second.observe())


def test_learned_effect_profile_ignores_action_token_name() -> None:
    first = CausalDependencyWorldV2(world_seed=82001, token_seed=92001)
    second = CausalDependencyWorldV2(world_seed=82001, token_seed=93001)
    left = RelationalEffectEncoder()
    right = RelationalEffectEncoder()
    left_transition = _observable_transition(first, "scan")
    right_transition = _observable_transition(second, "scan")
    left.observe(left_transition)
    right.observe(right_transition)
    assert left.action_key(left_transition.after, left_transition.action) == right.action_key(
        right_transition.after, right_transition.action
    )


def test_action_remap_branches_share_exact_start_checkpoint() -> None:
    rows = run_diagnostic_two_b(
        research_seeds=[2003],
        train_world_seeds=[82001, 82002],
        adaptation_world_seeds=[85001, 85002],
        training_episodes=60,
        evaluation_episodes=5,
        budgets=[1, 4],
    )
    for representation in {row.representation for row in rows}:
        starts = {
            row.start_checkpoint_fingerprint
            for row in rows
            if row.representation == representation
        }
        assert len(starts) == 1
    assert all(
        row.evaluation_checkpoint_fingerprint == row.final_checkpoint_fingerprint
        for row in rows
    )


def test_encoders_receive_same_raw_observation_instance() -> None:
    world = CausalDependencyWorldV2(world_seed=82001)
    observation = world.observe()
    identity = IdentityEncoder()
    relational = RelationalEffectEncoder()
    identity.state_key(observation)
    relational.state_key(observation)
    assert observation is observation


def test_unknown_action_value_survives_learned_effect_key_transition() -> None:
    world = CausalDependencyWorldV2(world_seed=82001, token_seed=92001)
    transition = _observable_transition(world, "scan")
    encoder = RelationalEffectEncoder()
    agent = RepresentedReturnAgent(encoder, seed=7, learning_rate=1.0)
    unknown_key = encoder.action_key(transition.before, transition.action)
    assert unknown_key.startswith("unknown:")

    agent.observe_transition(transition)
    agent.finish_episode(True, gamma=1.0)

    learned_key = encoder.action_key(transition.before, transition.action)
    assert learned_key.startswith("effect:")
    assert learned_key != unknown_key
    assert agent.q_value(transition.before, transition.action) == 1.0
