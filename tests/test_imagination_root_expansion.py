from aassr_v2.imagination_tree import ImaginationConfig, ImaginationTree
from aassr_v2.policy import WeightedPolicy
from aassr_v2.tabular_prophecy import TabularProphecy
from aassr_v2.types import Action, StateSnapshot


def test_root_expands_all_actions_even_when_deeper_branching_is_one() -> None:
    actions = tuple(Action(f"choice_{index}") for index in range(5))
    state = StateSnapshot((0.0,), frozenset({"start"}), actions, 0.0)
    prophecy = TabularProphecy()
    for index, action in enumerate(actions):
        next_state = StateSnapshot(
            (float(index + 1),),
            frozenset({f"after:{index}"}),
            (),
            0.0,
        )
        prophecy.learn(state, action, next_state)

    planner = ImaginationTree(
        WeightedPolicy({action.signature: float(-index) for index, action in enumerate(actions)}),
        prophecy,
        config=ImaginationConfig(
            branching_factor=1,
            maximum_depth=1,
            beam_width=1,
            outcome_samples=1,
            minimum_path_confidence=0.0,
            update_policy=False,
            expand_all_root_actions=True,
        ),
    )

    result = planner.plan(state)

    assert {item.action.signature for item in result.root_evaluations} == {
        action.signature for action in actions
    }


def test_root_expansion_can_be_disabled_for_ablation() -> None:
    actions = tuple(Action(f"choice_{index}") for index in range(4))
    state = StateSnapshot((0.0,), frozenset(), actions, 0.0)
    prophecy = TabularProphecy()
    for action in actions:
        prophecy.learn(state, action, StateSnapshot((1.0,), frozenset(), (), 0.0))

    planner = ImaginationTree(
        WeightedPolicy({action.signature: float(-index) for index, action in enumerate(actions)}),
        prophecy,
        config=ImaginationConfig(
            branching_factor=1,
            maximum_depth=1,
            beam_width=1,
            outcome_samples=1,
            minimum_path_confidence=0.0,
            update_policy=False,
            expand_all_root_actions=False,
        ),
    )

    result = planner.plan(state)

    assert len(result.root_evaluations) == 1
