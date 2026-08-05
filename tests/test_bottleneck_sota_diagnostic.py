from __future__ import annotations

from copy import deepcopy

from aassr_v2.baseline_efficiency_portable import (
    CHOICE_ACTIONS,
    BenchmarkGridPushWorld,
    solvable_map_seeds,
)
from aassr_v2.bottleneck_sota_portable import (
    BenchmarkOracleProgressScorer,
    BenchmarkOracleProphecy,
    OracleBFSBenchmarkAgent,
    remaining_oracle_steps,
    world_from_snapshot,
)


def _first_solvable_seed() -> int:
    return solvable_map_seeds(7000, 1)[0]


def test_snapshot_world_roundtrip_preserves_state() -> None:
    world = BenchmarkGridPushWorld(_first_solvable_seed())
    reconstructed = world_from_snapshot(world.snapshot())
    assert reconstructed.snapshot().vector == world.snapshot().vector
    assert reconstructed.snapshot().facts == world.snapshot().facts
    assert reconstructed.snapshot().available_actions == world.snapshot().available_actions


def test_oracle_prophecy_matches_real_transition() -> None:
    world = BenchmarkGridPushWorld(_first_solvable_seed())
    prophecy = BenchmarkOracleProphecy()
    for action in CHOICE_ACTIONS:
        expected = deepcopy(world)
        actual = expected.step(action).snapshot
        predicted = prophecy.predict(world.snapshot(), action, samples=1)[0]
        assert predicted.next_state.vector == actual.vector
        assert predicted.next_state.facts == actual.facts
        assert predicted.next_state.available_actions == actual.available_actions
        assert predicted.next_state.goal_progress == actual.goal_progress


def test_oracle_progress_scorer_prefers_a_shortest_path_action() -> None:
    world = BenchmarkGridPushWorld(_first_solvable_seed())
    state = world.snapshot()
    before_distance = remaining_oracle_steps(state)
    assert before_distance is not None and before_distance > 0
    prophecy = BenchmarkOracleProphecy()
    scorer = BenchmarkOracleProgressScorer()
    values = []
    distances = []
    for action in CHOICE_ACTIONS:
        after = prophecy.predict(state, action, samples=1)[0].next_state
        values.append(scorer.score(state, action, after))
        distances.append(remaining_oracle_steps(after))
    assert max(values) > 0.0
    assert before_distance - 1 in distances


def test_oracle_bfs_agent_reaches_goal_in_shortest_steps() -> None:
    seed = _first_solvable_seed()
    world = BenchmarkGridPushWorld(seed)
    expected_steps = remaining_oracle_steps(world.snapshot())
    assert expected_steps is not None
    agent = OracleBFSBenchmarkAgent(seed, train_episodes=1)
    steps = 0
    while not world.success and not world.failed:
        decision = agent.select_action(
            world.snapshot(),
            episode=0,
            training=False,
        )
        world.step(decision.action)
        steps += 1
    assert world.success
    assert steps == expected_steps
