from __future__ import annotations

from dataclasses import replace
from typing import Any

from . import toolgrid_factorial_masked as env
from .toolgrid_imagination_fix_patch import install_toolgrid_imagination_fix


base = env.base


def _terminal_class(state: Any) -> int:
    if state.available_actions:
        return 0
    if state.goal_progress >= 1.0 or "success" in state.facts:
        return 1
    return 2


def install_outcome_diversity_gate() -> None:
    """Install a rule-agnostic critical-decision gate for the debug run.

    The underlying representation, calibration, and replay fixes are installed
    first. Imagination is then allowed only when one-step model predictions for
    the currently available actions disagree about qualitative outcome class:
    nonterminal, terminal success, or terminal failure. Ordinary states where
    every action predicts the same class use the learned policy directly.
    """

    install_toolgrid_imagination_fix("categorical_tool_replay")
    parent = base.ToolGridHybridAgent

    class OutcomeDiversityGateHybridAgent(parent):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            original_select = self.agent.select_action

            def gated_select(
                state: Any,
                *,
                episode: int,
                explore: bool,
            ) -> Any:
                original = self.agent.config
                if not original.use_imagination or len(state.available_actions) < 2:
                    return original_select(state, episode=episode, explore=explore)

                outcome_classes = {
                    _terminal_class(
                        self.agent.prophecy.predict(state, action, samples=1)[0].next_state
                    )
                    for action in state.available_actions
                }
                if len(outcome_classes) > 1:
                    return original_select(state, episode=episode, explore=explore)

                self.agent.config = replace(original, use_imagination=False)
                try:
                    return original_select(state, episode=episode, explore=explore)
                finally:
                    self.agent.config = original

            self.agent.select_action = gated_select

    base.ToolGridHybridAgent = OutcomeDiversityGateHybridAgent
