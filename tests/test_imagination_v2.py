from __future__ import annotations

from types import SimpleNamespace

import pytest

from aassr_v2.autonomous_agent_core import ContextualPolicy
from aassr_v2.imagination_tree import ImaginationConfig, ImaginationTree
from aassr_v2.imagination_v2 import (
    ImaginationV2Agent,
    LegacyProphecyGRUCriticAgent,
    NeuralPolicyOnlyAgent,
    make_imagination_v2_agent,
)
from aassr_v2.types import Action, Prediction, StateSnapshot


ACTIONS = (Action("left"), Action("right"))


class _DeterministicProphecy:
    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        del samples
        delta = 1.0 if action.verb_name == "left" else 2.0
        value = state.vector[0] + delta
        return (
            Prediction(
                StateSnapshot(
                    (value,),
                    frozenset({f"value:{value}"}),
                    ACTIONS,
                ),
                1.0,
                source="test:exact",
            ),
        )


class _CountingCritic:
    value_mode = "absolute"

    def initial_memory(self) -> tuple[str, ...]:
        return ()

    def score_step(
        self,
        before: StateSnapshot,
        action: Action,
        after: StateSnapshot,
        *,
        memory: tuple[str, ...] | None,
        prophecy_confidence: float,
    ) -> SimpleNamespace:
        del before, after, prophecy_confidence
        history = tuple(memory or ()) + (action.signature,)
        value = 0.1 * len(history)
        if action.verb_name == "left":
            value += 0.01
        return SimpleNamespace(value=value, memory=history)


def test_stateful_critic_memory_is_branch_local_and_absolute() -> None:
    tree = ImaginationTree(
        ContextualPolicy(),
        _DeterministicProphecy(),
        config=ImaginationConfig(
            branching_factor=2,
            maximum_depth=2,
            beam_width=4,
            outcome_samples=1,
            minimum_path_confidence=0.0,
            uncertainty_penalty=0.0,
            update_policy=False,
        ),
        scorer=_CountingCritic(),
    )
    result = tree.plan(StateSnapshot((0.0,), frozenset(), ACTIONS))

    depth_one = [node for node in result.nodes if node.depth == 1]
    depth_two = [node for node in result.nodes if node.depth == 2]
    assert len(depth_one) == 2
    assert all(len(node.scorer_memory) == 1 for node in depth_one)
    assert all(len(node.scorer_memory) == 2 for node in depth_two)

    left_left = next(
        node
        for node in depth_two
        if tuple(item.split("|", 1)[0] for item in node.action_path)
        == ("left", "left")
    )
    assert left_left.cumulative_value == pytest.approx(0.21)


def test_imagination_v2_factory_wires_neural_prophecy_and_gru_critic() -> None:
    agent = make_imagination_v2_agent(
        "imagination_v2",
        7,
        train_episodes=10,
    )
    assert isinstance(agent, ImaginationV2Agent)
    assert agent.agent.planner.scorer is agent.critic
    assert not agent.critic_ready

    critic_only = make_imagination_v2_agent(
        "legacy_gru_critic",
        7,
        train_episodes=10,
    )
    assert isinstance(critic_only, LegacyProphecyGRUCriticAgent)
    assert critic_only.agent.planner.scorer is critic_only.critic


def test_neural_policy_only_control_permanently_disables_imagination() -> None:
    agent = make_imagination_v2_agent(
        "neural_policy_only",
        7,
        train_episodes=10,
    )
    assert isinstance(agent, NeuralPolicyOnlyAgent)
    assert not agent.agent.config.use_imagination
