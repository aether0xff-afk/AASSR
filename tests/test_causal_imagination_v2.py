from __future__ import annotations

from aassr_v2.causal_dependency_world import CausalDependencyWorldV2
from aassr_v2.causal_imagination import (
    CausalImaginationPlanner,
    ImaginationGateConfig,
    OracleReturnModel,
    RandomReturnModel,
)
from aassr_v2.causal_representation import RelationalEffectEncoder, RepresentedReturnAgent


def test_low_confidence_high_ood_blocks_intervention() -> None:
    world = CausalDependencyWorldV2(world_seed=82001)
    policy = RepresentedReturnAgent(RelationalEffectEncoder(), seed=1)
    planner = CausalImaginationPlanner(RandomReturnModel(3), gated=True)
    record = planner.decide(world.observe(), policy)
    assert not record.intervened
    assert "blocked" in record.intervention_reason or record.policy_only_action == record.final_selected_action


def test_oracle_transition_is_exact() -> None:
    world = CausalDependencyWorldV2(world_seed=82001)
    action = world.observe().available_actions[0]
    estimate = OracleReturnModel().estimate(world.observe(), action, world=world)
    assert estimate.transition_exact
    assert estimate.next_observation == world.oracle_transition(action)


def test_depth_and_branching_change_search_structure() -> None:
    world = CausalDependencyWorldV2(world_seed=82001)
    policy = RepresentedReturnAgent(RelationalEffectEncoder(), seed=1)
    shallow = CausalImaginationPlanner(
        OracleReturnModel(),
        config=ImaginationGateConfig(maximum_depth=1, branching_factor=1),
    ).decide(world.observe(), policy, world=world)
    deep = CausalImaginationPlanner(
        OracleReturnModel(),
        config=ImaginationGateConfig(maximum_depth=3, branching_factor=4),
    ).decide(world.observe(), policy, world=world)
    assert deep.imagined_nodes > shallow.imagined_nodes
    assert deep.maximum_depth_reached > shallow.maximum_depth_reached


def test_policy_and_model_q_are_discounted_return_scale() -> None:
    world = CausalDependencyWorldV2(world_seed=82001)
    policy = RepresentedReturnAgent(RelationalEffectEncoder(), seed=1)
    record = CausalImaginationPlanner(OracleReturnModel()).decide(
        world.observe(), policy, world=world
    )
    assert all(0.0 <= value <= 1.0 for value in record.root_policy_q.values())
    assert all(0.0 <= value <= 1.0 for value in record.root_model_q.values())
