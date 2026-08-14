from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import MethodType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from scripts import run_post10k_checkpoint_diagnostic as base
from scripts import run_post10k_step2_probe_priority as step2


VERSION = "post10k-step3-negative-route-profile-memory-v1"


def _install_negative_route_profile_memory(agent: object) -> Counter[str]:
    """Diagnostic only: carry public route/profile rejection across object IDs."""
    counters: Counter[str] = Counter()
    invalid_pairs: set[tuple[str, str]] = set()

    old_begin = agent.begin_episode
    old_selection = agent._selection_state
    old_step = agent.step

    def begin(self_agent: object, *args: Any, **kwargs: Any):
        invalid_pairs.clear()
        return old_begin(*args, **kwargs)

    def selection(self_agent: object, state: object):
        filtered, semantic, guarded, fallback = old_selection(state)
        if not invalid_pairs:
            return filtered, semantic, guarded, fallback

        kept = []
        removed = 0
        for action in filtered.available_actions:
            if getattr(action, "verb_name", "") != "request_object":
                kept.append(action)
                continue
            route_id = str(action.parameters.get("route_id", ""))
            profile_id = str(action.parameters.get("profile_id", ""))
            if (route_id, profile_id) in invalid_pairs:
                removed += 1
                continue
            kept.append(action)

        if removed <= 0:
            return filtered, semantic, guarded, fallback
        if not kept:
            counters["all_actions_would_be_removed_fallback"] += 1
            return filtered, semantic, guarded, fallback

        counters["filter_events"] += 1
        counters["filtered_actions"] += removed
        return (
            type(filtered)(
                vector=filtered.vector,
                facts=filtered.facts,
                available_actions=tuple(kept),
                goal_progress=filtered.goal_progress,
                metadata=filtered.metadata,
            ),
            semantic,
            guarded,
            fallback,
        )

    def step(
        self_agent: object,
        environment: object,
        *,
        episode: int,
        training: bool = True,
        primitive_budget: int | None = None,
    ):
        response_count = len(getattr(environment, "responses", ()))
        result = old_step(
            environment,
            episode=episode,
            training=training,
            primitive_budget=primitive_budget,
        )
        if training or not result.traces:
            return result

        responses = tuple(getattr(environment, "responses", ()))[response_count:]
        for trace, response in zip(result.traces, responses):
            action = trace.action
            if getattr(action, "verb_name", "") != "request_object":
                continue
            facts = set(getattr(response, "body_facts", ()))
            if "request_profile_not_applicable" not in facts:
                continue
            route_id = str(action.parameters.get("route_id", ""))
            profile_id = str(action.parameters.get("profile_id", ""))
            pair = (route_id, profile_id)
            counters["negative_response_events"] += 1
            if pair not in invalid_pairs:
                invalid_pairs.add(pair)
                counters["new_invalid_pairs"] += 1
        counters["active_invalid_pairs"] = len(invalid_pairs)
        return result

    agent.begin_episode = MethodType(begin, agent)
    agent._selection_state = MethodType(selection, agent)
    agent.step = MethodType(step, agent)
    agent._diagnostic_negative_profile_memory = counters
    return counters


def _parse_ints(text: str) -> tuple[int, ...]:
    return base._parse_ints(text)


def _load_step2_reference(path: Path, seeds: tuple[int, ...]) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ref_seeds = tuple(int(value) for value in data.get("scenario_seeds", ()))
    if ref_seeds != seeds:
        raise ValueError(f"step2 seed mismatch: {ref_seeds!r} != {seeds!r}")
    experimental = data.get("experimental")
    if not isinstance(experimental, dict):
        raise ValueError("step2 reference lacks experimental aggregate")
    return dict(experimental)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Step-3 post-10k diagnosis: keep Iteration-2 probe priority, then "
            "add only episode-local memory for public request_profile_not_applicable "
            "route/profile evidence across object IDs."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--step2-reference",
        type=Path,
        default=REPO_ROOT / "docs/experiments/post10k_l2_step2_reference.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/post10k_step3_negative_profile_memory"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument(
        "--scenario-seeds",
        type=_parse_ints,
        default=base.TRANSFER_DIAGNOSTIC_SEEDS,
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if not args.step2_reference.is_file():
        parser.error(f"step2 reference not found: {args.step2_reference}")

    seeds = tuple(int(value) for value in args.scenario_seeds)
    step2_reference = _load_step2_reference(args.step2_reference, seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    agent, payload = base._restore(
        args.checkpoint,
        device=args.device,
        allow_tf32=not args.no_tf32,
    )
    learning_before = base._learning_fingerprint(agent)
    agent.requested_imagination = False

    # Iteration 2 is held fixed as the experimental scaffold. The only new
    # scientific variable in Iteration 3 is cross-object route/profile rejection
    # memory derived from the actually returned public response body.
    probe_counters = step2._install_probe_priority(agent)
    memory_counters = _install_negative_route_profile_memory(agent)

    stage = base.TRANSFER_STAGES[2]
    rows = [
        base._run_episode(
            agent,
            stage=stage,
            seed=seed,
            condition="step3_probe_priority_plus_negative_profile_memory",
        )
        for seed in seeds
    ]

    if base._learning_fingerprint(agent) != learning_before:
        raise AssertionError("step-3 frozen diagnostic mutated learned state")

    experimental = base._aggregate(rows)
    failures = sum(row.status == "failure" for row in rows)
    experimental["failures"] = failures

    result = {
        "version": VERSION,
        "checkpoint_git_commit": payload.get("git_commit"),
        "checkpoint": str(args.checkpoint),
        "step2_reference": str(args.step2_reference),
        "device": args.device,
        "scenario_seeds": list(seeds),
        "comparison_contract": (
            "Iteration 2 probe-class priority is held fixed. The sole Iteration-3 "
            "delta is episode-local memory of public request_profile_not_applicable "
            "evidence at route/profile granularity across object IDs."
        ),
        "single_changed_variable": (
            "after a request_object action receives public response fact "
            "request_profile_not_applicable, suppress later request_object actions "
            "using the same route/profile pair with different object IDs for the "
            "rest of that episode, while alternatives remain"
        ),
        "unchanged_from_iteration2": [
            "environment",
            "L2 stage",
            "checkpoint weights",
            "state neural encoding",
            "action neural encoding",
            "Policy values",
            "probe-class priority scaffold",
            "reward",
            "ASEQ implementation",
            "Prophecy",
            "Critic",
            "Imagination disabled",
            "scenario seeds",
            "transition cap",
        ],
        "step2_reference_aggregate": step2_reference,
        "experimental": experimental,
        "delta_vs_step2": {
            "successes": int(experimental.get("successes", 0))
            - int(step2_reference.get("successes", 0)),
            "success_rate": float(experimental.get("success_rate", 0.0))
            - float(step2_reference.get("success_rate", 0.0)),
            "mean_transitions": float(experimental.get("mean_transitions", 0.0))
            - float(step2_reference.get("mean_transitions", 0.0)),
            "mean_aseq_guards": float(experimental.get("mean_aseq_guards", 0.0))
            - float(step2_reference.get("mean_aseq_guards", 0.0)),
            "mean_unknown_attempts": float(
                experimental.get("mean_unknown_attempts", 0.0)
            ) - float(step2_reference.get("mean_unknown_attempts", 0.0)),
        },
        "probe_diagnostics": dict(probe_counters),
        "negative_profile_memory_diagnostics": dict(memory_counters),
        "frozen_learning_state_asserted": True,
        "interpretation_rule": (
            "If success rises and failures/unsafe repeated probing fall, discarded "
            "route/profile negative-response semantics are a major L2 defect. If "
            "filtering fires strongly but success remains 0/8, preserve the finding "
            "but move next to positive profile-role discovery/retention. If filtering "
            "rarely fires, reject this hypothesis."
        ),
    }

    base._write_csv(args.output_dir / "episodes.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
