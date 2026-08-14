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
from scripts import run_post10k_step3_negative_profile_memory as step3
from scripts import run_post10k_step6_positive_profile_applicability as step6

VERSION = "post10k-step7-positive-applicability-until-target-v1"


def _has_observed_target(state: object) -> bool:
    return any(
        str(fact).startswith("observed_target_object:")
        for fact in getattr(state, "facts", ())
    )


def _install_positive_until_target(agent: object) -> Counter[str]:
    """Iteration 6 positive applicability, but stop its priority after target discovery.

    Relative to Iteration 6, the only behavioral delta is the handoff boundary:
    once the public observation contains observed_target_object:<id>, positive-pair
    priority no longer overrides Iteration-3 selection. Positive evidence collection,
    negative route/profile memory, probe priority, Policy values and all learned
    weights remain unchanged.
    """

    counters: Counter[str] = Counter()
    positive_pairs: set[tuple[str, str]] = set()

    policy = agent.policy
    iteration3_select = policy.select
    old_begin = agent.begin_episode
    old_step = agent.step

    def begin(self_agent: object, *args: Any, **kwargs: Any):
        positive_pairs.clear()
        return old_begin(*args, **kwargs)

    def select(
        self_policy: object,
        state: object,
        *,
        randomizer: object,
        epsilon: float,
        exploration_bonus: float,
    ):
        if float(epsilon) != 0.0 or not positive_pairs:
            return iteration3_select(
                state,
                randomizer=randomizer,
                epsilon=epsilon,
                exploration_bonus=exploration_bonus,
            )

        # Sole Iteration-7 delta versus Iteration 6.
        if _has_observed_target(state):
            counters["target_handoff_states"] += 1
            return iteration3_select(
                state,
                randomizer=randomizer,
                epsilon=epsilon,
                exploration_bonus=exploration_bonus,
            )

        candidates = tuple(
            action
            for action in state.available_actions
            if getattr(action, "verb_name", "") == "request_object"
            and step6._pair(action) in positive_pairs
        )
        if not candidates:
            counters["positive_pair_no_available_candidate"] += 1
            return iteration3_select(
                state,
                randomizer=randomizer,
                epsilon=epsilon,
                exploration_bonus=exploration_bonus,
            )

        counters["positive_pair_priority_opportunities"] += 1
        counters["positive_pair_candidates_total"] += len(candidates)
        counters["positive_pair_candidates_max"] = max(
            int(counters.get("positive_pair_candidates_max", 0)),
            len(candidates),
        )

        canonical = iteration3_select(
            state,
            randomizer=randomizer,
            epsilon=epsilon,
            exploration_bonus=exploration_bonus,
        )
        if canonical in candidates:
            counters["iteration3_already_positive_pair"] += 1
            return canonical

        candidate_signatures = {action.signature for action in candidates}
        ranked = self_policy.rank(state, limit=len(state.available_actions))
        selected = next(
            (item.action for item in ranked if item.action.signature in candidate_signatures),
            None,
        )
        if selected is None:
            raise RuntimeError("positive route/profile candidates vanished from Policy ranking")

        counters["positive_pair_priority_overrides"] += 1
        return selected

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
            if not base._unknown_profile(trace.before, action):
                continue
            if not step6._public_positive_object_semantics(response):
                continue

            pair = step6._pair(action)
            counters["positive_response_events"] += 1
            if pair not in positive_pairs:
                positive_pairs.add(pair)
                counters["new_positive_pairs"] += 1
            counters["active_positive_pairs"] = len(positive_pairs)

        return result

    policy.select = MethodType(select, policy)
    agent.begin_episode = MethodType(begin, agent)
    agent.step = MethodType(step, agent)
    agent._diagnostic_positive_until_target = counters
    return counters


def _parse_ints(text: str) -> tuple[int, ...]:
    return base._parse_ints(text)


def _load_reference(path: Path, seeds: tuple[int, ...]) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ref_seeds = tuple(int(value) for value in data.get("scenario_seeds", ()))
    if ref_seeds != seeds:
        raise ValueError(f"reference seed mismatch: {ref_seeds!r} != {seeds!r}")
    experimental = data.get("experimental")
    if not isinstance(experimental, dict):
        raise ValueError("reference lacks experimental aggregate")
    return dict(experimental)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Step-7 post-10k diagnosis: reproduce Iteration 6, but stop only its "
            "positive route/profile priority after public target-object discovery."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--step6-reference",
        type=Path,
        default=REPO_ROOT / "docs/experiments/post10k_l2_step6_reference.json",
    )
    parser.add_argument(
        "--step3-reference",
        type=Path,
        default=REPO_ROOT / "docs/experiments/post10k_l2_step3_reference.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/post10k_step7_positive_until_target"),
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
    if not args.step6_reference.is_file():
        parser.error(f"step6 reference not found: {args.step6_reference}")
    if not args.step3_reference.is_file():
        parser.error(f"step3 reference not found: {args.step3_reference}")

    seeds = tuple(int(value) for value in args.scenario_seeds)
    step6_reference = _load_reference(args.step6_reference, seeds)
    step3_reference = _load_reference(args.step3_reference, seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    agent, payload = base._restore(
        args.checkpoint,
        device=args.device,
        allow_tf32=not args.no_tf32,
    )
    learning_before = base._learning_fingerprint(agent)
    agent.requested_imagination = False

    # Hold Iteration 3 fixed.
    probe_counters = step2._install_probe_priority(agent)
    negative_counters = step3._install_negative_route_profile_memory(agent)

    # Reproduce Iteration 6 positive evidence semantics, changing only the
    # post-target handoff boundary.
    positive_counters = _install_positive_until_target(agent)

    stage = base.TRANSFER_STAGES[2]
    rows = [
        base._run_episode(
            agent,
            stage=stage,
            seed=seed,
            condition="step7_iter6_positive_priority_until_target",
        )
        for seed in seeds
    ]

    if base._learning_fingerprint(agent) != learning_before:
        raise AssertionError("step-7 frozen diagnostic mutated learned state")

    experimental = base._aggregate(rows)
    experimental["failures"] = sum(row.status == "failure" for row in rows)

    result = {
        "version": VERSION,
        "checkpoint_git_commit": payload.get("git_commit"),
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "scenario_seeds": list(seeds),
        "comparison_contract": (
            "Iteration 6 is reproduced exactly except that once the public state "
            "contains observed_target_object:<id>, positive route/profile priority "
            "stops and selection returns to Iteration-3 behavior."
        ),
        "single_changed_variable": (
            "stop Iteration-6 positive route/profile priority after public target "
            "object discovery; keep all pre-target positive evidence behavior unchanged"
        ),
        "unchanged_from_iteration6": [
            "environment dynamics and L2 stage",
            "checkpoint weights",
            "state/action neural encodings",
            "Policy values",
            "Iteration-2 probe-class priority",
            "Iteration-3 negative route/profile memory",
            "positive evidence definition",
            "positive pair ranking before target discovery",
            "reward",
            "ASEQ implementation",
            "Prophecy",
            "Critic",
            "Imagination disabled",
            "scenario seeds",
            "transition cap",
        ],
        "step6_reference_aggregate": step6_reference,
        "step3_reference_aggregate": step3_reference,
        "experimental": experimental,
        "delta_vs_step6": {
            "successes": int(experimental.get("successes", 0)) - int(step6_reference.get("successes", 0)),
            "success_rate": float(experimental.get("success_rate", 0.0)) - float(step6_reference.get("success_rate", 0.0)),
            "failures": int(experimental.get("failures", 0)) - int(step6_reference.get("failures", 0)),
            "truncations": int(experimental.get("truncations", 0)) - int(step6_reference.get("truncations", 0)),
            "mean_transitions": float(experimental.get("mean_transitions", 0.0)) - float(step6_reference.get("mean_transitions", 0.0)),
            "mean_aseq_guards": float(experimental.get("mean_aseq_guards", 0.0)) - float(step6_reference.get("mean_aseq_guards", 0.0)),
        },
        "delta_vs_step3": {
            "successes": int(experimental.get("successes", 0)) - int(step3_reference.get("successes", 0)),
            "success_rate": float(experimental.get("success_rate", 0.0)) - float(step3_reference.get("success_rate", 0.0)),
        },
        "probe_diagnostics": dict(probe_counters),
        "negative_profile_memory_diagnostics": dict(negative_counters),
        "positive_profile_memory_diagnostics": dict(positive_counters),
        "frozen_learning_state_asserted": True,
        "interpretation_rule": (
            "If target_handoff_states fires and success recovers above Iteration 6, "
            "positive applicability is useful only before target discovery and its "
            "post-target persistence was harmful. If success returns merely to the "
            "Iteration-3 1/8 baseline, reject positive applicability as a material "
            "remaining repair. If target_handoff never fires, target discovery itself "
            "remains the bottleneck."
        ),
    }

    base._write_csv(args.output_dir / "episodes.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
