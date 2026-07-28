from aassr_v2 import (
    Action,
    ActionVerb,
    GridWorldEnv,
    GridWorldSpec,
    KnowledgeStore,
    TabularProphecy,
    TransitionEvaluator,
)
from aassr_v2.trace import TraceLedger


def test_observation_unlocks_pickup() -> None:
    environment = GridWorldEnv(
        GridWorldSpec(
            width=2,
            height=1,
            start=(0, 0),
            goal=(1, 0),
            keys=(((0, 0), "blue"),),
            required_inventory_at_goal=frozenset({"blue"}),
        )
    )

    assert all(
        action.verb is not ActionVerb.PICKUP
        for action in environment.snapshot().available_actions
    )

    result = environment.step(
        Action(ActionVerb.OBSERVE, target="0,0")
    )

    assert any(
        action.verb is ActionVerb.PICKUP
        for action in result.unlocked_actions
    )


def test_matching_key_opens_door_and_reaches_goal() -> None:
    environment = GridWorldEnv(
        GridWorldSpec(
            width=3,
            height=1,
            start=(0, 0),
            goal=(2, 0),
            keys=(((0, 0), "blue"),),
            doors=(((1, 0), "blue"),),
            required_inventory_at_goal=frozenset({"blue"}),
        )
    )

    environment.step(Action(ActionVerb.OBSERVE, target="0,0"))
    environment.step(Action(ActionVerb.PICKUP, target="0,0"))
    environment.step(Action(ActionVerb.OBSERVE, target="1,0"))
    environment.step(
        Action(ActionVerb.USE, target="1,0", tool="blue")
    )
    environment.step(Action(ActionVerb.MOVE, destination="east"))
    result = environment.step(Action(ActionVerb.MOVE, destination="east"))

    assert result.goal_reached


def test_evaluator_records_prediction_improvement_and_knowledge() -> None:
    environment = GridWorldEnv(
        GridWorldSpec(
            width=2,
            height=1,
            start=(0, 0),
            goal=(1, 0),
            keys=(((0, 0), "blue"),),
        )
    )
    evaluator = TransitionEvaluator(TabularProphecy())
    knowledge = KnowledgeStore()
    ledger = TraceLedger()

    evaluated = evaluator.execute(
        environment,
        Action(ActionVerb.OBSERVE, target="0,0"),
        knowledge,
        ledger,
    )

    assert evaluated.prediction_score_after > evaluated.prediction_score_before
    assert evaluated.information_value.unlocked_action_value == 1.0
    assert knowledge.get("cell:0,0=key:blue") is not None
    assert len(ledger.all()) == 1
