from __future__ import annotations

from aassr_v2.causal_dependency_world import (
    CAUSAL_LAW_SHA256,
    CausalDependencyWorldV2,
    certify_world,
)
from aassr_v2.causal_diagnostic import diagnostic_one_gates, run_diagnostic_one


def test_tokens_change_but_causal_law_is_stable() -> None:
    first = CausalDependencyWorldV2(world_seed=82001)
    second = CausalDependencyWorldV2(world_seed=83001)
    assert first.causal_law_sha256 == second.causal_law_sha256 == CAUSAL_LAW_SHA256
    assert first.action_token_sha256 != second.action_token_sha256
    assert first.observation_token_sha256 != second.observation_token_sha256


def test_strict_sparse_observation_does_not_leak_private_state() -> None:
    world = CausalDependencyWorldV2(world_seed=82001)
    payload = repr(world.observe().to_dict()).lower()
    for forbidden in (
        "viable",
        "solution_family",
        "optimal_plan",
        "oracle_transition",
        "latent_risk",
        "goal_progress",
    ):
        assert forbidden not in payload


def test_world_solver_certifies_nontrivial_sparse_environment() -> None:
    certification = certify_world(
        CausalDependencyWorldV2(world_seed=82001), random_rollouts=400
    )
    assert certification.adequate, certification.issues
    assert certification.minimum_plan_length is not None
    assert certification.valid_solution_count >= 2
    assert certification.causal_family_count >= 3
    assert certification.dead_end_count >= 1
    assert certification.random_policy_success_estimate <= 0.10


def test_oracle_transition_matches_real_clone_transition() -> None:
    world = CausalDependencyWorldV2(world_seed=82001)
    action = world.observe().available_actions[0]
    predicted = world.oracle_transition(action)
    actual = world.clone().step(action).observation
    assert predicted == actual


def test_diagnostic_one_uses_frozen_checkpoint_clone() -> None:
    results = run_diagnostic_one(
        research_seeds=[2003],
        train_world_seeds=[82001, 82002],
        training_episodes=500,
        evaluation_episodes=50,
    )
    gates = diagnostic_one_gates(results)
    assert gates["checkpoint_immutable"]
    assert gates["evaluation_learning_calls_zero"]
    assert gates["contextual_training_above_random"]
    assert gates["contextual_frozen_above_random"]
    assert gates["contextual_replay_gap_within_0_10"]
    assert gates["full_replay_gap_within_0_10"]
