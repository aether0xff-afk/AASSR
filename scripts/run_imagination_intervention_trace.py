from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import MethodType
from typing import Any, Iterable, Sequence

import run_imagination_gate_ablation as gate
from aassr_v2.pentest_curriculum_schedule import semantic_fingerprint
from aassr_v2.pentest_transfer_stages import TRANSFER_STAGES
from aassr_v2.types import Action, Prediction, StateSnapshot, TransitionTrace


TRACE_VERSION = "imagination-intervention-decision-trace-v1"
DEFAULT_MARGIN = 0.05
TRACE_SEEDS = tuple(range(94_001, 94_005))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    try:
        return asdict(value)
    except (TypeError, ValueError):
        return repr(value)


def _action(action: Action | None) -> dict[str, Any] | None:
    if action is None:
        return None
    return {
        "signature": action.signature,
        "verb": action.verb_name,
        "target": action.target,
        "tool": action.tool,
        "destination": action.destination,
        "parameters": _jsonable(dict(action.parameters)),
        "metadata": _jsonable(dict(action.metadata)),
    }


def _semantic(state: StateSnapshot) -> Any:
    return _jsonable(semantic_fingerprint(state))


def _state(state: StateSnapshot) -> dict[str, Any]:
    return {
        "semantic_fingerprint": _semantic(state),
        "goal_progress": float(state.goal_progress),
        "facts": sorted(state.facts),
        "available_action_signatures": [
            action.signature for action in state.available_actions
        ],
        "vector": [float(value) for value in state.vector],
        "metadata": _jsonable(dict(state.metadata)),
    }


def _prediction(prediction: Prediction, before: StateSnapshot) -> dict[str, Any]:
    predicted = prediction.next_state
    return {
        "probability": float(prediction.probability),
        "source": prediction.source,
        "next_state": _state(predicted),
        "added_facts_vs_current": sorted(predicted.facts - before.facts),
        "removed_facts_vs_current": sorted(before.facts - predicted.facts),
    }


def _trace(trace: TransitionTrace) -> dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "action": _action(trace.action),
        "before": _state(trace.before),
        "predictions": [
            _prediction(prediction, trace.before) for prediction in trace.predictions
        ],
        "after": _state(trace.after),
        "actual_added_facts": sorted(trace.added_facts),
        "actual_removed_facts": sorted(trace.removed_facts),
        "actual_unlocked_actions": [_action(action) for action in trace.unlocked_actions],
        "error": bool(trace.error),
        "real_reward": float(trace.real_reward),
        "semantic_changed": _semantic(trace.before) != _semantic(trace.after),
    }


def _find_action(state: StateSnapshot, signature: str) -> Action | None:
    if not signature:
        return None
    return next(
        (action for action in state.available_actions if action.signature == signature),
        None,
    )


def _safe_confidence(agent: object, state: StateSnapshot, action: Action | None) -> float | None:
    if action is None:
        return None
    try:
        return float(agent.skill_prophecy.confidence(state, action))
    except Exception:
        return None


def _safe_policy_value(agent: object, state: StateSnapshot, action: Action | None) -> float | None:
    if action is None:
        return None
    try:
        return float(agent.policy.value(state, action))
    except Exception:
        return None


def _safe_predictions(
    agent: object,
    state: StateSnapshot,
    action: Action | None,
) -> list[dict[str, Any]]:
    if action is None:
        return []
    try:
        predictions = agent.prophecy.predict_with_context(
            state,
            action,
            knowledge=agent.knowledge,
            samples=1,
        )
    except Exception as exc:
        return [{"logging_error": repr(exc)}]
    return [_prediction(prediction, state) for prediction in predictions]


class DetailedTraceCapture:
    def __init__(self, agent: object) -> None:
        self.agent = agent
        self.events: list[dict[str, Any]] = []
        self.episodes: list[dict[str, Any]] = []
        self._current_episode: dict[str, Any] | None = None
        self._last_event: dict[str, Any] | None = None
        self._last_plan: Any = None
        self._original_begin_episode = agent.begin_episode
        self._original_core_select = agent._core_select_action
        self._original_step = agent.step
        self._original_plan = agent.planner.plan

    def install(self) -> None:
        capture = self

        def recording_begin_episode(self_agent: object, *args: Any, **kwargs: Any) -> Any:
            result = capture._original_begin_episode(*args, **kwargs)
            episode = {
                "capture_episode_index": len(capture.episodes),
                "events": [],
            }
            capture.episodes.append(episode)
            capture._current_episode = episode
            capture._last_event = None
            return result

        def recording_plan(self_planner: object, state: StateSnapshot) -> Any:
            plan = capture._original_plan(state)
            capture._last_plan = plan
            return plan

        def recording_core_select(
            self_agent: object,
            state: StateSnapshot,
            *,
            episode: int,
            explore: bool,
        ) -> Any:
            capture._last_plan = None
            decision = capture._original_core_select(
                state,
                episode=episode,
                explore=explore,
            )
            policy_action = _find_action(state, decision.policy_action_signature)
            preferred_action = _find_action(
                state,
                decision.imagination_preferred_action_signature,
            )
            executed_action = _find_action(state, decision.action.signature) or decision.action

            roots = []
            plan = capture._last_plan
            if plan is not None:
                for root in getattr(plan, "root_evaluations", ()):
                    root_action = getattr(root, "action", None)
                    roots.append(
                        {
                            "action": _action(root_action),
                            "planner_aggregate_value": float(
                                getattr(root, "aggregate_value", 0.0)
                            ),
                            "policy_value": _safe_policy_value(
                                self_agent,
                                state,
                                root_action,
                            ),
                            "prophecy_confidence": _safe_confidence(
                                self_agent,
                                state,
                                root_action,
                            ),
                        }
                    )
                roots.sort(
                    key=lambda row: (
                        -float(row["planner_aggregate_value"]),
                        str((row.get("action") or {}).get("signature", "")),
                    )
                )

            event = {
                "event_index": len(capture.events),
                "decision_episode_argument": int(episode),
                "explore": bool(explore),
                "state": _state(state),
                "policy_action": _action(policy_action),
                "preferred_action": _action(preferred_action),
                "executed_action": _action(executed_action),
                "policy_action_policy_value": _safe_policy_value(
                    self_agent,
                    state,
                    policy_action,
                ),
                "preferred_action_policy_value": _safe_policy_value(
                    self_agent,
                    state,
                    preferred_action,
                ),
                "policy_action_prophecy_confidence": _safe_confidence(
                    self_agent,
                    state,
                    policy_action,
                ),
                "preferred_action_prophecy_confidence": _safe_confidence(
                    self_agent,
                    state,
                    preferred_action,
                ),
                "policy_action_one_step_prophecy": _safe_predictions(
                    self_agent,
                    state,
                    policy_action,
                ),
                "preferred_action_one_step_prophecy": _safe_predictions(
                    self_agent,
                    state,
                    preferred_action,
                ),
                "used_imagination": bool(decision.used_imagination),
                "imagined_nodes": int(decision.imagined_nodes),
                "imagination_depth": int(decision.imagination_depth),
                "model_coverage": float(decision.model_coverage),
                "gate_reason": decision.imagination_gate_reason,
                "switch_candidate": bool(decision.imagination_switch_candidate),
                "intervention_allowed": bool(decision.imagination_intervention_allowed),
                "changed_action": bool(decision.imagination_changed_action),
                "planner_policy_value": float(decision.imagination_policy_value),
                "planner_preferred_value": float(decision.imagination_preferred_value),
                "planner_advantage": float(decision.imagination_advantage),
                "required_advantage": float(decision.imagination_required_advantage),
                "root_imagined_value": float(decision.root_imagined_value),
                "planner_root_evaluations": roots,
                "real_traces": [],
                "world_after_step": None,
            }
            capture.events.append(event)
            capture._last_event = event
            if capture._current_episode is not None:
                capture._current_episode["events"].append(event)
            return decision

        def recording_step(
            self_agent: object,
            environment: object,
            *,
            episode: int,
            training: bool,
            primitive_budget: int | None = None,
        ) -> Any:
            result = capture._original_step(
                environment,
                episode=episode,
                training=training,
                primitive_budget=primitive_budget,
            )
            event = capture._last_event
            if event is not None:
                event["real_traces"] = [_trace(trace) for trace in result.traces]
                event["world_after_step"] = {
                    "success": bool(getattr(environment, "success", False)),
                    "proof_acquired": bool(getattr(environment, "proof_acquired", False)),
                    "failed": bool(getattr(environment, "failed", False)),
                    "locked": bool(getattr(environment, "locked", False)),
                    "rate_limited": bool(getattr(environment, "rate_limited", False)),
                    "terminal": bool(getattr(environment, "terminal", False)),
                }
            return result

        self.agent.begin_episode = MethodType(recording_begin_episode, self.agent)
        self.agent._core_select_action = MethodType(recording_core_select, self.agent)
        self.agent.step = MethodType(recording_step, self.agent)
        self.agent.planner.plan = MethodType(recording_plan, self.agent.planner)

    def uninstall(self) -> None:
        self.agent.begin_episode = self._original_begin_episode
        self.agent._core_select_action = self._original_core_select
        self.agent.step = self._original_step
        self.agent.planner.plan = self._original_plan

    def annotate_rows(self, rows: Sequence[Any]) -> None:
        if len(rows) != len(self.episodes):
            raise AssertionError(
                "detailed trace episode count mismatch: "
                f"rows={len(rows)} captured={len(self.episodes)}"
            )
        for row, episode in zip(rows, self.episodes, strict=True):
            episode["episode_result"] = {
                "phase": row.phase,
                "condition": row.condition,
                "research_seed": int(row.research_seed),
                "scenario_seed": int(row.scenario_seed),
                "level": int(row.curriculum_level),
                "stage": row.curriculum_stage,
                "status": row.status,
                "success": int(row.success),
                "failure": int(row.failure),
                "stalled": int(row.stalled),
                "truncation": int(row.truncation),
                "primitive_transitions": int(row.primitive_transitions),
                "reward": float(row.reward),
                "aseq_guard_events": int(row.aseq_guard_events),
                "imagination_runs": int(row.imagination_runs),
                "imagination_interventions": int(row.imagination_interventions),
                "imagination_changed_actions": int(row.imagination_changed_actions),
            }
            for event in episode["events"]:
                event["episode_result"] = dict(episode["episode_result"])


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str))
            handle.write("\n")


def _write_trace_files(
    output: Path,
    *,
    condition: str,
    capture: DetailedTraceCapture,
) -> None:
    safe = condition.replace("/", "_")
    _write_jsonl(output / f"decision_trace_{safe}.jsonl", capture.events)
    _write_jsonl(output / f"episode_trace_{safe}.jsonl", capture.episodes)
    switch_events = [event for event in capture.events if event["switch_candidate"]]
    intervention_events = [
        event for event in capture.events if event["intervention_allowed"]
    ]
    _write_jsonl(output / f"switch_trace_{safe}.jsonl", switch_events)
    _write_jsonl(output / f"intervention_trace_{safe}.jsonl", intervention_events)


def _tracing_evaluate_variant(
    agent: object,
    *,
    research_seed: int,
    diagnostic_seeds: Sequence[int],
    diagnostic_stage_indices: Sequence[int],
    transition_budget: int,
    condition: str,
    use_imagination: bool,
    uncertainty_margin: float,
) -> dict[str, Any]:
    margin = float(uncertainty_margin)
    original_config = agent.config
    original_evaluate = _ORIGINAL_EVALUATE_VARIANT

    if use_imagination:
        agent.config = replace(
            original_config,
            imagination_intervention_margin=margin,
            imagination_uncertainty_margin=0.0,
        )
        evaluated_condition = f"aassr_gate_margin_{str(margin).replace('.', 'p')}"
    else:
        evaluated_condition = "aassr_no_imagination"

    capture = DetailedTraceCapture(agent)
    capture.install()
    try:
        result = original_evaluate(
            agent,
            research_seed=research_seed,
            diagnostic_seeds=diagnostic_seeds,
            diagnostic_stage_indices=diagnostic_stage_indices,
            transition_budget=transition_budget,
            condition=evaluated_condition,
            use_imagination=use_imagination,
            uncertainty_margin=0.0,
        )
    finally:
        capture.uninstall()
        agent.config = original_config

    capture.annotate_rows(result["rows"])
    result["detailed_trace_capture"] = capture
    if use_imagination:
        result["calibration_margin"] = margin
        result["gate_formula"] = "required_advantage = calibration_margin"
    return result


_ORIGINAL_EVALUATE_VARIANT = gate._evaluate_variant


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one frozen-checkpoint Imagination gate comparison with full decision-level traces."
    )
    parser.add_argument("--output-dir", default="runs/imagination_intervention_trace")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--transitions", type=int, default=2048)
    parser.add_argument("--block-target", type=int, default=512)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--max-level", type=int, default=4)
    parser.add_argument("--seed-count", type=int, default=len(TRACE_SEEDS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--allow-critic-not-ready", action="store_true")
    args = parser.parse_args()

    if args.margin < 0.0:
        parser.error("--margin must be non-negative")
    if not 0 <= args.max_level < len(TRANSFER_STAGES):
        parser.error("--max-level outside transfer-stage range")
    if not 1 <= args.seed_count <= len(TRACE_SEEDS):
        parser.error("--seed-count outside trace seed pool")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    gate._evaluate_variant = _tracing_evaluate_variant
    result = gate.run_gate_ablation(
        output,
        research_seed=int(args.seed),
        transition_budget=int(args.transitions),
        block_target=int(args.block_target),
        diagnostic_seeds=TRACE_SEEDS[: args.seed_count],
        diagnostic_stage_indices=tuple(range(args.max_level + 1)),
        uncertainty_margins=(float(args.margin),),
        device=args.device,
        allow_tf32=not args.no_tf32,
        require_critic_ready=not args.allow_critic_not_ready,
    )

    captures = []
    for variant in result["variants"]:
        capture = variant.pop("detailed_trace_capture", None)
        if capture is None:
            continue
        _write_trace_files(
            output,
            condition=variant["condition"],
            capture=capture,
        )
        captures.append(
            {
                "condition": variant["condition"],
                "decisions": len(capture.events),
                "episodes": len(capture.episodes),
                "switch_candidates": sum(
                    bool(event["switch_candidate"]) for event in capture.events
                ),
                "interventions": sum(
                    bool(event["intervention_allowed"]) for event in capture.events
                ),
            }
        )

    result["version"] = TRACE_VERSION
    result["trace_contract"] = {
        "margin": float(args.margin),
        "coverage_uncertainty_lambda": 0.0,
        "trace_seeds": list(TRACE_SEEDS[: args.seed_count]),
        "stage_indices": list(range(args.max_level + 1)),
        "captures": captures,
        "records": [
            "decision state and semantic fingerprint",
            "policy/preferred/executed action with full structured fields",
            "Policy values and Prophecy confidence for compared actions",
            "one-step Prophecy predictions for Policy and preferred actions",
            "exact planner root evaluations used by the decision",
            "coverage, advantage, threshold, and gate reason",
            "real executed transition prediction versus actual next state",
            "added/removed facts, unlocked actions, error and reward",
            "episode seed, level, final status and transition count",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(json.dumps(result["trace_contract"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
