from aassr_v2 import (
    Action,
    ActionVerb,
    AdaptiveDepthController,
    ImaginationConfig,
    ImaginationTree,
    Prediction,
    StateSnapshot,
    WeightedPolicy,
)


class ScriptedProphecy:
    name = "scripted"

    def __init__(self, transitions: dict[tuple[tuple[float, ...], str], StateSnapshot]):
        self.transitions = transitions

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        del samples
        next_state = self.transitions[(state.vector, action.signature)]
        return (Prediction(next_state, 1.0, "scripted:exact"),)

    def learn(self, *args: object) -> None:
        del args


def make_state(
    value: int,
    progress: float,
    actions: tuple[Action, ...],
) -> StateSnapshot:
    return StateSnapshot(
        (float(value),),
        available_actions=actions,
        goal_progress=progress,
    )


def test_tree_prefers_delayed_goal_over_immediate_bait() -> None:
    bait = Action(ActionVerb.OBSERVE, target="bait")
    setup = Action(ActionVerb.OBSERVE, target="setup")
    wait = Action(ActionVerb.MOVE, destination="west")
    finish = Action(ActionVerb.MOVE, destination="east")

    root = make_state(0, 0.0, (bait, setup))
    bait_state = make_state(1, 0.4, (wait,))
    setup_state = make_state(2, 0.0, (finish,))
    dead_end = make_state(3, 0.4, (wait,))
    goal = make_state(4, 1.0, ())

    prophecy = ScriptedProphecy(
        {
            (root.vector, bait.signature): bait_state,
            (root.vector, setup.signature): setup_state,
            (bait_state.vector, wait.signature): dead_end,
            (setup_state.vector, finish.signature): goal,
        }
    )
    policy = WeightedPolicy(
        {
            bait.signature: 1.0,
            setup.signature: 0.0,
        }
    )
    planner = ImaginationTree(
        policy,
        prophecy,
        config=ImaginationConfig(
            branching_factor=2,
            maximum_depth=2,
            beam_width=4,
        ),
    )

    result = planner.plan(root)

    assert result.chosen_action == setup
    assert result.maximum_depth_reached == 2
    assert policy.weight(setup) > 0.0


def test_parallel_branches_keep_independent_policy_memories() -> None:
    left = Action(ActionVerb.MOVE, destination="west")
    right = Action(ActionVerb.MOVE, destination="east")
    root = make_state(0, 0.0, (left, right))
    left_state = make_state(1, 0.1, ())
    right_state = make_state(2, 0.2, ())
    prophecy = ScriptedProphecy(
        {
            (root.vector, left.signature): left_state,
            (root.vector, right.signature): right_state,
        }
    )

    result = ImaginationTree(
        WeightedPolicy(),
        prophecy,
        config=ImaginationConfig(maximum_depth=1, branching_factor=2),
    ).plan(root)

    children = [node for node in result.nodes if node.depth == 1]
    assert len(children) == 2
    for child in children:
        assert set(child.policy_memory.deltas) == {
            child.action_from_parent.signature
        }


def test_adaptive_depth_grows_slowly_and_shrinks_quickly() -> None:
    controller = AdaptiveDepthController(
        minimum_depth=1,
        maximum_depth=4,
        window_size=2,
        grow_threshold=0.9,
        shrink_threshold=0.6,
        grow_streak=2,
    )

    for _ in range(4):
        controller.observe(0.95)
    assert controller.depth == 2

    controller.observe(0.0)
    controller.observe(0.0)
    assert controller.depth == 1


def test_unseen_prediction_stops_on_low_confidence() -> None:
    action = Action(ActionVerb.OBSERVE, target="x")
    root = make_state(0, 0.0, (action,))
    same = make_state(0, 0.0, (action,))

    class UnseenProphecy:
        def predict(
            self,
            state: StateSnapshot,
            action: Action,
            *,
            samples: int,
        ) -> tuple[Prediction, ...]:
            del state, action, samples
            return (Prediction(same, 1.0, "tabular:unseen"),)

    result = ImaginationTree(
        WeightedPolicy(),
        UnseenProphecy(),
        config=ImaginationConfig(
            maximum_depth=4,
            minimum_path_confidence=0.1,
        ),
    ).plan(root)

    child = next(node for node in result.nodes if node.depth == 1)
    assert child.terminal_reason == "low_confidence"
    assert result.maximum_depth_reached == 1


def test_state_without_actions_becomes_terminal_leaf() -> None:
    action = Action(ActionVerb.OBSERVE, target="x")
    root = make_state(0, 0.0, (action,))
    empty = make_state(1, 0.0, ())
    prophecy = ScriptedProphecy(
        {(root.vector, action.signature): empty}
    )

    result = ImaginationTree(
        WeightedPolicy(),
        prophecy,
        config=ImaginationConfig(maximum_depth=3),
    ).plan(root)

    leaf = next(node for node in result.nodes if node.depth == 1)
    assert leaf.terminal_reason == "no_actions"
    assert result.root_evaluations[0].best_path == (action.signature,)
