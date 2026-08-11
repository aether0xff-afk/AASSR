from __future__ import annotations

from collections import Counter
from types import MethodType
from typing import Any, Sequence

from .current_generation import relational_action_key
from .current_relational_state_v3 import relational_state_key_v3
from .prophecy import ProphecyStep
from .skills import SKILL_VERB


def _action_key(state: object, action: object) -> tuple[Any, ...]:
    if getattr(action, "verb_name", None) == SKILL_VERB:
        return ("skill", str(getattr(action, "target", "")))
    return ("primitive", *relational_action_key(state, action))


def install_structural_root_dedup(agent: object) -> object:
    """Compute identifier-equivalent roots once, then fan values to aliases.

    Root evaluation coverage remains concrete so the final action is always a real
    environment action. Only expensive model/Critic computation is deduplicated.
    """
    if getattr(agent, "current_structural_root_dedup", False):
        return agent

    counters: Counter[str] = Counter()
    prophecy_view = agent.current_batched_prophecy
    original_predict_step_batch = prophecy_view.predict_step_batch

    def dedup_predict_step_batch(
        self_view: object,
        states: Sequence[Any],
        actions: Sequence[Any],
        memories: Sequence[Any],
        *,
        samples: int,
    ) -> tuple[ProphecyStep, ...]:
        if not (len(states) == len(actions) == len(memories)):
            raise ValueError("states/actions/memories batch length mismatch")
        if not states:
            return ()

        unique_states = []
        unique_actions = []
        unique_memories = []
        unique_index: dict[tuple[Any, ...], int] = {}
        inverse: list[int] = []

        for index, (state, action, memory) in enumerate(
            zip(states, actions, memories, strict=True)
        ):
            if getattr(action, "verb_name", None) == SKILL_VERB:
                key = ("skill-row", index)
            else:
                key = (
                    relational_state_key_v3(state),
                    _action_key(state, action),
                )
            target = unique_index.get(key)
            if target is None:
                target = len(unique_states)
                unique_index[key] = target
                unique_states.append(state)
                unique_actions.append(action)
                unique_memories.append(memory)
            inverse.append(target)

        rows = original_predict_step_batch(
            tuple(unique_states),
            tuple(unique_actions),
            tuple(unique_memories),
            samples=samples,
        )
        if len(rows) != len(unique_states):
            raise RuntimeError("deduplicated Prophecy returned wrong row count")

        counters["prophecy_requested_rows"] += len(states)
        counters["prophecy_unique_rows"] += len(unique_states)
        counters["prophecy_alias_rows_removed"] += len(states) - len(unique_states)

        output = []
        for index, target in enumerate(inverse):
            row = rows[target]
            # Primitive Prophecy leaves recurrent memory unchanged. Preserve the
            # caller's original memory object when fanning one structural result
            # back to several concrete aliases.
            if getattr(actions[index], "verb_name", None) == SKILL_VERB:
                output.append(row)
            else:
                output.append(ProphecyStep(tuple(row.predictions), memories[index]))
        return tuple(output)

    prophecy_view.predict_step_batch = MethodType(
        dedup_predict_step_batch,
        prophecy_view,
    )

    planner = agent.planner
    original_score_candidates = planner._score_candidates

    def dedup_score_candidates(
        self_planner: object,
        candidates: Sequence[tuple[Any, Any, Any, float]],
    ) -> tuple[Any, ...]:
        if not candidates:
            return ()

        unique = []
        unique_index: dict[tuple[Any, ...], int] = {}
        inverse: list[int] = []
        for index, candidate in enumerate(candidates):
            node, scored_action, after, confidence = candidate
            if int(getattr(node, "depth", -1)) != 0:
                key = ("nonroot", index)
            else:
                key = (
                    relational_state_key_v3(node.state),
                    _action_key(node.state, scored_action.action),
                    relational_state_key_v3(after),
                    round(float(confidence), 8),
                )
            target = unique_index.get(key)
            if target is None:
                target = len(unique)
                unique_index[key] = target
                unique.append(candidate)
            inverse.append(target)

        rows = original_score_candidates(tuple(unique))
        if len(rows) != len(unique):
            raise RuntimeError("deduplicated Critic returned wrong row count")

        root_rows = sum(int(getattr(candidate[0], "depth", -1) == 0) for candidate in candidates)
        root_unique = sum(int(getattr(candidate[0], "depth", -1) == 0) for candidate in unique)
        counters["critic_root_requested_rows"] += root_rows
        counters["critic_root_unique_rows"] += root_unique
        counters["critic_root_alias_rows_removed"] += root_rows - root_unique
        return tuple(rows[target] for target in inverse)

    planner._score_candidates = MethodType(dedup_score_candidates, planner)

    original_diagnostics = agent.diagnostics

    def diagnostics_with_dedup(self_agent: object) -> dict[str, Any]:
        output = dict(original_diagnostics())
        output["structural_root_dedup"] = {
            "enabled": True,
            **dict(counters),
        }
        repairs = dict(output.get("current_repairs", {}))
        repairs.update(
            {
                "structural_root_compute_dedup": True,
                "concrete_root_execution_preserved": True,
            }
        )
        output["current_repairs"] = repairs
        return output

    agent.diagnostics = MethodType(diagnostics_with_dedup, agent)
    agent._structural_root_dedup_diagnostics = counters
    agent.current_structural_root_dedup = True
    return agent
