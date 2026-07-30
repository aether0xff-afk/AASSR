from __future__ import annotations

import json

from aassr_v2.autonomous_agent import AutonomousAgentConfig, AutonomousLearningAgent, ContextualPolicy
from aassr_v2.escape_imagination_capture import (
    ImaginationEventStream,
    capture_imaginations,
    serialize_imagination_result,
)
from aassr_v2.imagination_tree import ImaginationConfig, ImaginationTree
from aassr_v2.tabular_prophecy import TabularProphecy
from aassr_v2.types import Action, StateSnapshot


def _state(*, goal: float = 0.0, actions: tuple[Action, ...] = ()) -> StateSnapshot:
    return StateSnapshot(
        vector=(goal,),
        facts=frozenset({f"goal={goal}"}),
        available_actions=actions,
        goal_progress=goal,
        metadata={"position": (1, 2), "steps": 7},
    )


def _result():
    action = Action("advance")
    before = _state(actions=(action,))
    after = _state(goal=1.0)
    prophecy = TabularProphecy()
    prophecy.learn(before, action, after)
    planner = ImaginationTree(
        ContextualPolicy(),
        prophecy,
        config=ImaginationConfig(
            branching_factor=1,
            maximum_depth=3,
            beam_width=4,
            outcome_samples=1,
            minimum_path_confidence=0.0,
            uncertainty_penalty=0.0,
            update_policy=False,
        ),
    )
    return planner.plan(before)


def test_complete_imagination_tree_is_serialized() -> None:
    payload = serialize_imagination_result(_result())
    assert payload["chosen_action"]["signature"] == "advance|_|_|_"
    assert payload["root_evaluations"][0]["chosen"] is True
    assert payload["nodes"][0]["node_id"] == 0
    assert payload["nodes"][-1]["terminal_reason"] == "goal"
    assert payload["nodes"][-1]["state"]["goal_progress"] == 1.0


def test_every_imagination_is_flushed_to_jsonl(tmp_path) -> None:
    received = []
    stream = ImaginationEventStream(tmp_path, callback=received.append)
    with capture_imaginations(stream.record):
        _result()
        _result()
    stream.close()

    rows = [json.loads(line) for line in (tmp_path / "imaginations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["sequence"] for row in rows] == [1, 2]
    assert all(row["root_step"] == 7 for row in rows)
    assert all(row["nodes"] for row in rows)
    assert len(received) == 2
    summary = json.loads((tmp_path / "imagination_summary.json").read_text(encoding="utf-8"))
    assert summary["events"] == 2
    assert summary["total_nodes"] >= 4


def test_epsilon_random_step_skips_imagination_even_with_interval_one() -> None:
    action = Action("advance")
    state = _state(actions=(action,))
    agent = AutonomousLearningAgent(
        TabularProphecy(),
        config=AutonomousAgentConfig(
            epsilon_start=1.0,
            epsilon_end=1.0,
            use_imagination=True,
            imagination_interval=1,
            imagination_minimum_coverage=0.0,
        ),
        seed=3,
    )
    assert agent.select_action(state, episode=0, explore=True).used_imagination is False


def test_interval_one_imagines_on_every_eligible_nonrandom_step() -> None:
    action = Action("advance")
    state = _state(actions=(action,))
    agent = AutonomousLearningAgent(
        TabularProphecy(),
        config=AutonomousAgentConfig(
            epsilon_start=0.0,
            epsilon_end=0.0,
            use_imagination=True,
            imagination_interval=1,
            imagination_minimum_coverage=0.0,
        ),
        seed=3,
    )
    assert agent.select_action(state, episode=0, explore=True).used_imagination is True
    assert agent.select_action(state, episode=0, explore=True).used_imagination is True


def test_interval_two_imagines_only_every_second_eligible_decision() -> None:
    action = Action("advance")
    state = _state(actions=(action,))
    agent = AutonomousLearningAgent(
        TabularProphecy(),
        config=AutonomousAgentConfig(
            epsilon_start=0.0,
            epsilon_end=0.0,
            use_imagination=True,
            imagination_interval=2,
            imagination_minimum_coverage=0.0,
        ),
        seed=3,
    )
    assert agent.select_action(state, episode=0, explore=True).used_imagination is False
    assert agent.select_action(state, episode=0, explore=True).used_imagination is True
