from __future__ import annotations

from types import SimpleNamespace

from aassr_v2.state_distance_goal_executor import StateDistanceGoalExecutor
from aassr_v2.types import Action, ActionVerb, Prediction, StateSnapshot


class LinearProphecy:
    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        direction = action.parameters["direction"]
        delta = 1.0 if direction == "east" else -1.0
        next_state = StateSnapshot(
            (state.vector[0] + delta, *state.vector[1:]),
            state.facts,
            state.available_actions,
            state.goal_progress,
        )
        return tuple(Prediction(next_state, 1.0, "test") for _ in range(samples))


def test_executor_chooses_prediction_closer_to_maker_waypoint() -> None:
    east = Action(ActionVerb.MOVE, parameters={"direction": "east"})
    west = Action(ActionVerb.MOVE, parameters={"direction": "west"})
    state = StateSnapshot((0.0, 0.0), frozenset(), (east, west), 0.0)
    proposal = SimpleNamespace(
        desired_state=StateSnapshot((2.0, 0.0), frozenset(), (), 0.0)
    )

    executor = StateDistanceGoalExecutor(LinearProphecy(), samples=2)
    chosen, evaluations = executor.choose(state, proposal)

    assert len(evaluations) == 2
    assert chosen.action.signature == east.signature
    assert chosen.distance < next(
        item.distance
        for item in evaluations
        if item.action.signature == west.signature
    )


def test_executor_uses_no_direction_specific_target_rule() -> None:
    east = Action(ActionVerb.MOVE, parameters={"direction": "east"})
    west = Action(ActionVerb.MOVE, parameters={"direction": "west"})
    state = StateSnapshot((0.0, 0.0), frozenset(), (east, west), 0.0)
    proposal = SimpleNamespace(
        desired_state=StateSnapshot((-2.0, 0.0), frozenset(), (), 0.0)
    )

    executor = StateDistanceGoalExecutor(LinearProphecy(), samples=1)
    chosen, _ = executor.choose(state, proposal)

    assert chosen.action.signature == west.signature
