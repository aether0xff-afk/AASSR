from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .types import Action, Prediction, StateSnapshot, TransitionTrace


def action_to_dict(action: Action) -> dict[str, Any]:
    return {
        "verb": action.verb_name,
        "target": action.target,
        "tool": action.tool,
        "destination": action.destination,
        "parameters": dict(action.parameters),
        "metadata": dict(action.metadata),
        "signature": action.signature,
    }


def state_to_dict(state: StateSnapshot) -> dict[str, Any]:
    return {
        "vector": list(state.vector),
        "facts": sorted(state.facts),
        "available_actions": [
            action_to_dict(action)
            for action in state.available_actions
        ],
        "goal_progress": state.goal_progress,
        "metadata": dict(state.metadata),
    }


def prediction_to_dict(
    prediction: Prediction,
) -> dict[str, Any]:
    return {
        "next_state": state_to_dict(prediction.next_state),
        "probability": prediction.probability,
        "source": prediction.source,
    }


def trace_to_dict(
    trace: TransitionTrace,
) -> dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "before": state_to_dict(trace.before),
        "action": action_to_dict(trace.action),
        "predictions": [
            prediction_to_dict(item)
            for item in trace.predictions
        ],
        "after": state_to_dict(trace.after),
        "added_facts": sorted(trace.added_facts),
        "removed_facts": sorted(trace.removed_facts),
        "unlocked_actions": [
            action_to_dict(action)
            for action in trace.unlocked_actions
        ],
        "error": trace.error,
        "real_reward": trace.real_reward,
        "goal_ids": list(trace.goal_ids),
    }


class JsonlLedgerWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def append(
        self,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        record = {
            "event_type": event_type,
            **dict(payload),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )

    def append_trace(
        self,
        trace: TransitionTrace,
        **metrics: Any,
    ) -> None:
        self.append(
            "transition",
            {
                "trace": trace_to_dict(trace),
                "metrics": metrics,
            },
        )
