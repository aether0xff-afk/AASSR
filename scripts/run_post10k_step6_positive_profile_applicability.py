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


VERSION = "post10k-step6-positive-route-profile-applicability-v1"


def _pair(action: object) -> tuple[str, str]:
    return (
        str(action.parameters.get("route_id", "")),
        str(action.parameters.get("profile_id", "")),
    )


def _public_positive_object_semantics(response: object) -> bool:
    facts = {str(fact) for fact in getattr(response, "body_facts", ())}
    return bool(
        "authorization_boundary_missing" in facts
        or any(fact.startswith("object:") for fact in facts)
    )


def _install_positive_applicability_memory(agent: object) -> Counter[str]:
    """Keep public evidence that one route/profile pair actually serves object semantics.

    Iteration 3 already remembers the negative public relation
    request_profile_not_applicable. This iteration adds only the complementary
    positive relation: when an unknown-profile request_object receives a public
    object-semantic response, remember that concrete route/profile pair for the
    episode and prefer remaining request_object actions using that pair.

    No hidden scenario role, target identity, oracle action, reward or model weight
    is consulted.
    """

    counters: Counter[str] = Counter()
    positive_pairs: set[tuple[str, str]] = set()

    policy = agent.policy
    step2_select = policy.select
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
            return step2_select(
                state,
                randomizer=randomizer,
                epsilon=epsilon,
                exploration_bonus=exploration_bonus,
            )

        candidates = tuple(
            action
            for action in state.available_actions
            if getattr(action, "verb_name", "") == "request_object"
            and _pair(action) in positive_pairs
        )
        if not candidates:
            counters["positive_pair_no_available_candidate"] += 1
            return step2_select(
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

        # Measure the Iteration-3 choice, then change only the class priority when
        # it would leave the already-supported route/profile pair.
        iteration3_choice = step2_select(
            state,
            randomizer=randomizer,
            epsilon=epsilon,
            exploration_bonus=exploration_bonus,
        )
        if iteration3_choice in candidates:
            counters["iteration3_already_positive_pair"] += 1
            return iteration3_choice

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
            # Only learn applicability while the profile was still unknown before
            # the executed action. Once a role is public, canonical semantics apply.
            if not base._unknown_profile(trace.before, action):
                continue
            if not _public_positive_object_semantics(response):
                continue

            pair = _pair(action)
            counters["positive_response_events"] += 1
            if pair not in positive_pairs:
                positive_pairs.add(pair)
                counters["new_positive_pairs"] += 1
            counters["active_positive_pairs"] = len(positive_pairs)

        return result

    policy.select = MethodType(select, policy)
    agent.begin_episode = MethodType(begin, agent)
    agent.step = MethodType(step, agent)
    agent._diagnostic_positive_profile_memory = counters
    return counters


def _parse_ints(text: str) -> tuple[int, ...]:
    return base._parse_ints(text)


def _load_step3_reference(path: Path, seeds: tuple[int, ...]) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ref_seeds = tuple(int(value) for value in data.get("scenario_seeds", ()))
    if ref_seeds != seeds:
        raise ValueError(f"step3 seed mismatch: {ref_seeds!r} != {seeds!r}")
    experimental = data.get("experimental")
    if not isinstance(experimental, dict):
        raise ValueError("step3 reference lacks experimental aggregate")
    return dict(experimental)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Step-6 post-10k diagnosis: return to Iteration 3 and add only "
            "episode-local positive route/profile applicability memory from public "
            "object-semantic responses."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--step3-reference",
        type=Path,
        default=REPO_ROOT / "docs/experiments/post10k_l2_step3_reference.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/post10k_step6_positive_profile_applicability"),
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
    if not args.step3_reference.is_file():
        parser.error(f"step3 reference not found: {args.step3_reference}")

    seeds = tuple(int(value) for value in args.scenario_seeds)
    step3_reference = _load_step3_reference(args.step3_reference, seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    agent, payload = base._restore(
        args.checkpoint,
        device=args.device,
        allow_tf32=not args.no_tf32,
    )
    learning_before = base._learning_fingerprint(agent)
    agent.requested_imagination = False

    # Reproduce Iteration 3 first: forced probe priority + public negative
    # route/profile memory. Iterations 4 and 5 are rejected and not carried.
    probe_counters = step2._install_probe_priority(agent)
    negative_counters = step3._install_negative_route_profile_memory(agent)

    # Sole Iteration-6 addition: retain positive route/profile applicability from
    # public object-semantic responses and prefer that supported pair thereafter.
    positive_counters = _install_positive_applicability_memory(agent)

    stage = base.TRANSFER_STAGES[2]
    rows = [
        base._run_episode(
            agent,
            stage=stage,
            seed=seed,
            condition="step6_iter3_plus_positive_profile_applicability",
        )
        for seed in seeds
    ]

    if base._learning_fingerprint(agent) != learning_before:
        raise AssertionError("step-6 frozen diagnostic mutated learned state")

    experimental = base._aggregate(rows)
    experimental["failures"] = sum(row.status == "failure" for row in rows)

    result = {
        "version": VERSION,
        "checkpoint_git_commit": payload.get("git_commit"),
        "checkpoint": str(args.checkpoint),
        "step3_reference": str(args.step3_reference),
        "device": args.device,
        "scenario_seeds": list(seeds),
        "comparison_contract": (
            "Iterations 4 and 5 are rejected and not carried. Iteration 3 probe "
            "priority and negative route/profile memory are held fixed. The sole "
            "Iteration-6 delta is episode-local retention and use of public positive "
            "route/profile applicability evidence from object-semantic responses."
        ),
        "single_changed_variable": (
            "after an unknown-profile request_object receives public object-semantic "
            "evidence (object:<id> or authorization_boundary_missing), remember that "
            "route/profile pair as applicable for the episode and prioritize remaining "
            "request_object actions using that pair"
        ),
        "unchanged_from_iteration3": [
            "environment dynamics and L2 stage",
            "checkpoint weights",
            "state/action neural encodings",
            "Policy values and ordering inside the positive pair",
            "Iteration-2 probe-class priority scaffold before positive evidence",
            "Iteration-3 negative route/profile memory",
            "no hidden profile role or target identity exposed",
            "reward",
            "ASEQ implementation",
            "Prophecy",
            "Critic",
            "Imagination disabled",
            "scenario seeds",
            "transition cap",
        ],
        "step3_reference_aggregate": step3_reference,
        "experimental": experimental,
        "delta_vs_step3": {
            "successes": int(experimental.get("successes", 0))
            - int(step3_reference.get("successes", 0)),
            "success_rate": float(experimental.get("success_rate", 0.0))
            - float(step3_reference.get("success_rate", 0.0)),
            "mean_transitions": float(experimental.get("mean_transitions", 0.0))
            - float(step3_reference.get("mean_transitions", 0.0)),
            "mean_aseq_guards": float(experimental.get("mean_aseq_guards", 0.0))
            - float(step3_reference.get("mean_aseq_guards", 0.0)),
            "mean_unknown_attempts": float(experimental.get("mean_unknown_attempts", 0.0))
            - float(step3_reference.get("mean_unknown_attempts", 0.0)),
        },
        "probe_diagnostics": dict(probe_counters),
        "negative_profile_memory_diagnostics": dict(negative_counters),
        "positive_profile_memory_diagnostics": dict(positive_counters),
        "frozen_learning_state_asserted": True,
        "interpretation_rule": (
            "If positive_response_events/new_positive_pairs fire and success rises "
            "materially over Iteration 3, discarded positive route/profile "
            "applicability is a major remaining L2 defect. If positive memory fires "
            "strongly but success stays near 1/8, retain the semantic finding but "
            "move downstream to target/workflow use. If positive evidence rarely "
            "appears, the remaining bottleneck is earlier in reaching a valid "
            "route/profile pair."
        ),
    }

    base._write_csv(args.output_dir / "episodes.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
