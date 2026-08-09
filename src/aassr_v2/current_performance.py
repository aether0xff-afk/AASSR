from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from .native_batching import DepthBatchedImaginationTree
from .prophecy import ProphecyStep
from .skills import SKILL_VERB
from .types import Action, Prediction, StateSnapshot


class CurrentDepthBatchedProphecyView:
    """Depth-batch the current Neural Delta while preserving symbolic context.

    This is intentionally independent of the historical TorchGRU-specific
    `DepthBatchedProphecyView`. Primitive branches share one Neural-Delta ensemble
    forward pass. Relational Skill macros keep their scalar variable-length
    rollout. Pre-existing Knowledge is composed after the neural batch using the
    exact same public facts/action bindings as the scalar current runtime.
    """

    name = "current-neural-delta-depth-batched"

    def __init__(self, agent: object) -> None:
        self.agent = agent
        self.view = agent.prophecy
        self._batch_calls = 0
        self._batch_rows = 0
        self._skill_fallback_rows = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.view, name)

    @staticmethod
    def _with_knowledge(
        prediction: Prediction,
        state: StateSnapshot,
        knowledge: object,
    ) -> Prediction:
        entries = tuple(knowledge.values())
        known_facts = frozenset(
            entry.key
            for entry in entries
            if bool(entry.value) and entry.confidence > 0.0
        )
        enabled = {
            signature
            for entry in entries
            for signature in entry.enabled_action_signatures
        }
        current_actions = {
            item.signature: item for item in state.available_actions
        }
        next_state = prediction.next_state
        action_map = {
            item.signature: item for item in next_state.available_actions
        }
        for signature in sorted(enabled):
            item = current_actions.get(signature)
            if item is not None:
                action_map[signature] = item
        return replace(
            prediction,
            next_state=replace(
                next_state,
                facts=next_state.facts | known_facts,
                available_actions=tuple(
                    action_map[key] for key in sorted(action_map)
                ),
            ),
        )

    def predict_step_batch(
        self,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
        memories: Sequence[Any],
        *,
        samples: int,
    ) -> tuple[ProphecyStep, ...]:
        if not (len(states) == len(actions) == len(memories)):
            raise ValueError("states/actions/memories batch length mismatch")
        if not states:
            return ()

        results: list[ProphecyStep | None] = [None] * len(states)
        primitive_indices = [
            index
            for index, action in enumerate(actions)
            if action.verb_name != SKILL_VERB
        ]
        if primitive_indices:
            primitive_states = tuple(states[index] for index in primitive_indices)
            primitive_actions = tuple(actions[index] for index in primitive_indices)
            rows = self.agent.calibrated_prophecy.predict_batch(
                primitive_states,
                primitive_actions,
                samples=samples,
            )
            self._batch_calls += 1
            self._batch_rows += len(primitive_indices)
            for output_index, state, predictions in zip(
                primitive_indices,
                primitive_states,
                rows,
                strict=True,
            ):
                contextual = tuple(
                    self._with_knowledge(
                        prediction,
                        state,
                        self.agent.knowledge,
                    )
                    for prediction in predictions
                )
                augmented = tuple(
                    replace(
                        prediction,
                        next_state=self.agent.skills.augment_state(
                            prediction.next_state
                        ),
                    )
                    for prediction in contextual
                )
                results[output_index] = ProphecyStep(
                    augmented,
                    memories[output_index],
                )

        for index, (state, action, memory) in enumerate(
            zip(states, actions, memories, strict=True)
        ):
            if results[index] is not None:
                continue
            self._skill_fallback_rows += 1
            results[index] = self.agent.skill_prophecy.predict_step(
                state,
                action,
                memory=memory,
                samples=samples,
            )

        if any(item is None for item in results):
            raise RuntimeError("current depth batch left an unresolved branch")
        return tuple(item for item in results if item is not None)

    def runtime_diagnostics(self) -> dict[str, int]:
        return {
            "current_imagination_batch_calls": self._batch_calls,
            "current_imagination_batch_rows": self._batch_rows,
            "current_imagination_skill_fallback_rows": self._skill_fallback_rows,
        }


def enable_current_depth_batching(agent: object) -> object:
    """Install semantics-preserving current-generation depth batching."""

    wrapped = CurrentDepthBatchedProphecyView(agent)
    old = agent.core.planner
    planner = DepthBatchedImaginationTree(
        old.policy,
        wrapped,
        config=old.config,
        scorer=old.scorer,
    )
    planner._state_key = old._state_key
    agent.core.planner = planner
    agent.current_batched_prophecy = wrapped
    agent.current_depth_batching = True
    return agent
