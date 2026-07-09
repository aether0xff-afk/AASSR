from __future__ import annotations

from typing import Any

from .gridworld import ActionCandidate, GridWorldDMP, StepResult
from .knowledge import KK


def candidate_label(candidate: ActionCandidate) -> str:
    bindings = ", ".join(f"{kk.value}={value}" for kk, value in candidate.bindings.items())
    return f"{candidate.template}\n-> {candidate.name.value}({bindings})"


def candidate_rows(dmp: GridWorldDMP) -> list[dict[str, Any]]:
    return [
        {
            "WHAT": candidate.name.value,
            "HOW": candidate.strategy,
            "WHERE": where_slot(candidate),
            "required_KK": ", ".join(kk.value for kk in candidate.required_kk_slots),
            "bound_KV": bound_kv(candidate),
            "executable_action": executable_command(candidate),
        }
        for candidate in dmp.generate_candidates()
    ]


def binding_rows(dmp: GridWorldDMP) -> list[dict[str, Any]]:
    rows = []
    for candidate in dmp.generate_candidates():
        for kk in candidate.required_kk_slots:
            value = dmp.position if kk == KK.CURRENT_POS else candidate.bindings.get(kk)
            rows.append(
                {
                    "template": candidate.template,
                    "KK slot": kk.value,
                    "KV bound": format_table_value(value),
                    "generated command": executable_command(candidate),
                }
            )
    return rows


def action_template_rows() -> list[dict[str, str]]:
    return [
        {
            "template": "MOVE_TOWARD {KK_FRONTIER_CELL | KK_HINT_CELL | KK_KEY_CELL | KK_DOOR_CELL | KK_FLAG_CELL}",
            "role": "Bind a target cell from Knowledge Storage and move one step toward it.",
        },
        {
            "template": "INSPECT_CELL {KK_UNKNOWN_NEIGHBOR | KK_FRONTIER_CELL}",
            "role": "Bind an unknown or frontier cell and convert observation into new KV.",
        },
        {
            "template": "USE_OBJECT {KK_KEY_OBJECT} ON {KK_DOOR_CELL}",
            "role": "Bind an acquired object and a target door.",
        },
        {
            "template": "FOLLOW_HINT {KK_HINT_VALUE}",
            "role": "Bind hint content and create inferred target KV.",
        },
    ]


def paper_project_comparison_rows() -> list[dict[str, str]]:
    return [
        {
            "axis": "Domain",
            "Original APASSR / prior setting": "Security or pentesting task space with command templates.",
            "This project": "Deterministic GridWorld with inspect, move, hint, key, door, and flag tasks.",
        },
        {
            "axis": "Knowledge Storage",
            "Original APASSR / prior setting": "Stores discovered values such as target IP, port, path, service, credential, and vulnerability evidence.",
            "This project": "Stores typed KV candidates such as current position, frontier cell, hint value, key object, door cell, and flag cell.",
        },
        {
            "axis": "KK",
            "Original APASSR / prior setting": "Abstract command parameter slots such as TARGET_IP, PORT, PATH, SERVICE, or TOOL option.",
            "This project": "Abstract action parameter slots such as KK_FRONTIER_CELL, KK_UNKNOWN_NEIGHBOR, KK_KEY_OBJECT, and KK_DOOR_CELL.",
        },
        {
            "axis": "KV",
            "Original APASSR / prior setting": "Concrete values discovered from observations and bound into command slots.",
            "This project": "Concrete cell coordinates, directions, object instances, and hint targets bound into action slots.",
        },
        {
            "axis": "Action template",
            "Original APASSR / prior setting": "Command forms such as nmap {WHAT_OPTION} {PORT} {TARGET_IP}.",
            "This project": "Action forms such as MOVE_TOWARD {KK_FRONTIER_CELL}, INSPECT_CELL {KK_UNKNOWN_NEIGHBOR}, and USE_OBJECT {KK_KEY_OBJECT} ON {KK_DOOR_CELL}.",
        },
        {
            "axis": "PolicyABC",
            "Original APASSR / prior setting": "Policy decomposes command generation into WHAT, HOW, and WHERE choices.",
            "This project": "PolicyABC learns WHAT action, HOW binding strategy, and WHERE KK slot probability tables.",
        },
        {
            "axis": "Prophecy",
            "Original APASSR / prior setting": "Predicts likely knowledge change, error, and goal-related outcome for candidate actions.",
            "This project": "Prophecy Module predicts possible delta-KK, error, and flag relevance. TableProphecyModel is the current lightweight C3 implementation.",
        },
        {
            "axis": "Imagination",
            "Original APASSR / prior setting": "Evaluates candidate actions before execution using predicted outcomes.",
            "This project": "ImaginationCycle performs depth-limited candidate rollout from ProphecyPrediction without reading hidden GridWorld map state.",
        },
        {
            "axis": "Reward target",
            "Original APASSR / prior setting": "Task success plus useful knowledge gain and prediction quality.",
            "This project": "Sparse flag reward, semantic knowledge gain, error/repeat penalties, and optional prophecy prediction error reward.",
        },
        {
            "axis": "Experiment conditions",
            "Original APASSR / prior setting": "Baseline, policy, prophecy, and imagination ablations.",
            "This project": "C0 Random, C1 PolicyABC, C2 PolicyABC + Prophecy, C3 main framework with Prophecy + Imagination, C4 optional sequence-prophecy variant.",
        },
    ]


def implementation_status_rows() -> list[dict[str, str]]:
    return [
        {"module": "Knowledge Storage", "status": "Implemented", "evidence": "KK/KV store, metadata, lifecycle, semantic KnowledgeDelta"},
        {"module": "Action Binding", "status": "Implemented", "evidence": "Executable candidates generated by binding KV values into KK slots"},
        {"module": "PolicyABC", "status": "Implemented", "evidence": "WHAT/HOW/WHERE probability tables with reward update and min-prob floor"},
        {"module": "Prophecy", "status": "Implemented", "evidence": "ProphecyModule interface with TableProphecyModel as lightweight implementation and SequenceProphecyModel as optional variant"},
        {"module": "Imagination", "status": "Implemented", "evidence": "Depth-limited Prophecy-based candidate rollout before execution"},
        {"module": "ExperimentRunner", "status": "Implemented", "evidence": "C0/C1/C2/C3/C4 runner, randomized worlds, combined summary CSV"},
        {"module": "Analysis", "status": "Implemented", "evidence": "Seed-level bootstrap CI, paper-facing plots, report.md"},
        {"module": "Full paper result", "status": "Pending run", "evidence": "Requires medium/paper-candidate experiment execution and interpretation"},
    ]


def policy_probability_rows(dmp: GridWorldDMP) -> list[dict[str, Any]]:
    snapshot = getattr(dmp.scorer, "snapshot", lambda: {})()
    rows = []
    for axis, table in snapshot.items():
        for key, probability in table.items():
            rows.append({"axis": axis, "key": key, "probability": round(probability, 4)})
    return rows


def imagination_score_rows(result: StepResult | None) -> list[dict[str, Any]]:
    if result is None or result.imagination_trace is None:
        return []
    rows = []
    selected = result.imagination_trace.selected
    for score in sorted(result.imagination_trace.scores, key=lambda item: item.score, reverse=True):
        rows.append(
            {
                "selected": score.candidate == selected,
                "candidate": executable_command(score.candidate),
                "score": round(score.score, 4),
                "expected_kk_gain": round(score.expected_kk_gain, 4),
                "flag_prob": round(score.predicted_flag_prob, 4),
                "error_prob": round(score.predicted_error_prob, 4),
                "rollout_value": round(score.rollout_value, 4),
                "rollout_depth": score.rollout_depth,
                "repeat_penalty": round(score.repeat_penalty, 4),
                "policy_prior": round(score.policy_prior, 4),
            }
        )
    return rows


def knowledge_rows(dmp: GridWorldDMP) -> list[dict[str, Any]]:
    rows = []
    for kk, entries in dmp.store.snapshot().items():
        for item in entries:
            rows.append(
                {
                    "KK": kk,
                    "KV": format_table_value(item["value"]),
                    "type": item["type"],
                    "source": item["source"],
                    "confidence": item["confidence"],
                    "status": item["status"],
                    "used": item["used_count"],
                    "success": item["success_count"],
                    "updated": item["last_updated"],
                }
            )
    return rows


def trace_row(
    *,
    result: StepResult,
    candidate: ActionCandidate,
    selected_by: str,
    pos_before: tuple[int, int],
    pos_after: tuple[int, int],
) -> dict[str, Any]:
    return {
        "step": result.step,
        "selected_by": selected_by,
        "WHAT": candidate.name.value,
        "HOW": candidate.strategy,
        "WHERE": where_slot(candidate),
        "command": executable_command(candidate),
        "pos_before": format_table_value(pos_before),
        "pos_after": format_table_value(pos_after),
        "delta_added": len(result.delta_k.added),
        "delta_updated": len(result.delta_k.updated),
        "delta_status": len(result.delta_k.status_changed),
        "delta_removed": len(result.delta_k.removed),
        "delta_usage": len(result.delta_k.usage_updated),
        "semantic_gain": result.delta_k.semantic_information_gain(),
        "reward": result.total_reward,
        "prophecy_error": result.prophecy_error,
        "prophecy_loss": result.prophecy_loss,
        "imagination_score": result.to_dict()["imagination_selected_score"],
        "imagination_candidates": result.to_dict()["imagination_candidate_count"],
        "done": result.done,
        "result": compact_result(result),
    }


def where_slot(candidate: ActionCandidate) -> str:
    for kk in candidate.required_kk_slots:
        if kk != KK.CURRENT_POS:
            return kk.value
    return candidate.required_kk_slots[0].value


def bound_kv(candidate: ActionCandidate) -> str:
    return ", ".join(
        f"{kk.value}={format_table_value(value)}" for kk, value in candidate.bindings.items()
    )


def executable_command(candidate: ActionCandidate) -> str:
    if candidate.name.value == "USE_OBJECT":
        return (
            f"USE {format_table_value(candidate.bindings[KK.KEY_OBJECT])} "
            f"ON {format_table_value(candidate.bindings[KK.DOOR_CELL])}"
        )
    target = next((value for kk, value in candidate.bindings.items() if kk != KK.CURRENT_POS), None)
    return f"{candidate.name.value} {format_table_value(target)}"


def compact_result(result: StepResult) -> str:
    observation = result.observation
    if "cell" in observation:
        kind = observation["kind"].value if hasattr(observation["kind"], "value") else observation["kind"]
        return f"observed {kind} at {format_table_value(observation['cell'])}"
    if "target" in observation:
        return f"target={format_table_value(observation['target'])}, moved={observation['moved']}"
    if "door" in observation:
        return f"door={format_table_value(observation['door'])}, opened={observation['opened']}"
    if "hint" in observation:
        return f"hint={format_table_value(observation['hint'])}"
    return str(observation)


def format_table_value(value: Any) -> str:
    if isinstance(value, tuple):
        return "(" + ", ".join(str(part) for part in value) + ")"
    if isinstance(value, list):
        return "[" + ", ".join(format_table_value(part) for part in value) + "]"
    return str(value)
