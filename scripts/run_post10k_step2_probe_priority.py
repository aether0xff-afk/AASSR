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


VERSION = "post10k-step2-unknown-object-profile-probe-priority-v1"
DEFAULT_BASELINE = REPO_ROOT / "docs" / "experiments" / "post10k_l2_baseline_reference.json"


def _unknown_object_profile_probe(state: object, action: object) -> bool:
    return bool(
        getattr(action, "verb_name", "") == "request_object"
        and base._unknown_profile(state, action)
    )


def _install_probe_priority(agent: object) -> Counter[str]:
    """Diagnostic only: prioritize the unknown object-profile probe class."""
    counters: Counter[str] = Counter()
    policy = agent.policy
    original_select = policy.select

    def select(
        self_policy: object,
        state: object,
        *,
        randomizer: object,
        epsilon: float,
        exploration_bonus: float,
    ):
        if float(epsilon) != 0.0:
            return original_select(
                state,
                randomizer=randomizer,
                epsilon=epsilon,
                exploration_bonus=exploration_bonus,
            )

        actions = tuple(state.available_actions)
        if not actions:
            raise ValueError("cannot select from an empty action set")

        probes = tuple(
            action for action in actions if _unknown_object_profile_probe(state, action)
        )
        if not probes:
            return original_select(
                state,
                randomizer=randomizer,
                epsilon=epsilon,
                exploration_bonus=exploration_bonus,
            )

        counters["probe_opportunities"] += 1
        counters["probe_candidates_total"] += len(probes)
        counters["probe_candidates_max"] = max(
            int(counters.get("probe_candidates_max", 0)), len(probes)
        )

        canonical = original_select(
            state,
            randomizer=randomizer,
            epsilon=epsilon,
            exploration_bonus=exploration_bonus,
        )
        if _unknown_object_profile_probe(state, canonical):
            counters["canonical_already_probe"] += 1
            return canonical

        probe_signatures = {action.signature for action in probes}
        ranked = self_policy.rank(state, limit=len(actions))
        selected = next(
            (item.action for item in ranked if item.action.signature in probe_signatures),
            None,
        )
        if selected is None:
            raise RuntimeError("unknown probe subset vanished from Policy ranking")

        counters["forced_probe_choices"] += 1
        counters["overridden_nonprobe_choices"] += 1
        return selected

    policy.select = MethodType(select, policy)
    agent._diagnostic_probe_priority = counters
    return counters


def _parse_ints(text: str) -> tuple[int, ...]:
    return base._parse_ints(text)


def _load_baseline(path: Path, seeds: tuple[int, ...]) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    baseline_seeds = tuple(int(value) for value in data.get("scenario_seeds", ()))
    if baseline_seeds != seeds:
        raise ValueError(f"baseline seed mismatch: {baseline_seeds!r} != {seeds!r}")
    baseline = data.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("baseline reference does not contain a baseline object")
    return dict(baseline)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Step-2 post-10k diagnosis: when response-causal unknown-profile "
            "request_object probes exist, prioritize that class while keeping "
            "the frozen Policy's ranking inside the class unchanged."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/post10k_step2_probe_priority"),
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
    if not args.baseline_summary.is_file():
        parser.error(f"baseline reference not found: {args.baseline_summary}")

    seeds = tuple(int(value) for value in args.scenario_seeds)
    baseline = _load_baseline(args.baseline_summary, seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    agent, payload = base._restore(
        args.checkpoint,
        device=args.device,
        allow_tf32=not args.no_tf32,
    )
    learning_before = base._learning_fingerprint(agent)
    agent.requested_imagination = False
    probe_counters = _install_probe_priority(agent)

    stage = base.TRANSFER_STAGES[2]
    rows = [
        base._run_episode(
            agent,
            stage=stage,
            seed=seed,
            condition="step2_unknown_object_profile_probe_priority",
        )
        for seed in seeds
    ]

    if base._learning_fingerprint(agent) != learning_before:
        raise AssertionError("step-2 frozen diagnostic mutated learned state")

    experimental = base._aggregate(rows)
    baseline_successes = int(baseline.get("successes", 0))
    experimental_successes = int(experimental.get("successes", 0))

    result = {
        "version": VERSION,
        "checkpoint_git_commit": payload.get("git_commit"),
        "checkpoint": str(args.checkpoint),
        "baseline_summary": str(args.baseline_summary),
        "device": args.device,
        "scenario_seeds": list(seeds),
        "single_changed_variable": (
            "when one or more response-causal unknown-profile request_object "
            "actions are available, execute the highest existing frozen-Policy "
            "ranked action inside that subset instead of the global Policy winner"
        ),
        "unchanged": [
            "environment",
            "state representation",
            "action representation",
            "learned weights",
            "Policy scores",
            "Policy ordering inside the probe subset",
            "no candidate answer or raw target identity exposed",
            "no tried-candidate memory added",
            "no randomization added",
            "no reward or information bonus added",
            "ASEQ",
            "Prophecy",
            "Critic",
            "Imagination disabled",
            "scenario seeds",
            "transition cap",
        ],
        "baseline": baseline,
        "experimental": experimental,
        "delta": {
            "successes": experimental_successes - baseline_successes,
            "success_rate": float(experimental.get("success_rate", 0.0))
            - float(baseline.get("success_rate", 0.0)),
            "mean_aseq_guards": float(experimental.get("mean_aseq_guards", 0.0))
            - float(baseline.get("mean_aseq_guards", 0.0)),
            "mean_unknown_attempts": float(
                experimental.get("mean_unknown_attempts", 0.0)
            )
            - float(baseline.get("mean_unknown_attempts", 0.0)),
            "mean_transitions": float(experimental.get("mean_transitions", 0.0))
            - float(baseline.get("mean_transitions", 0.0)),
        },
        "probe_diagnostics": dict(probe_counters),
        "frozen_learning_state_asserted": True,
        "interpretation_rule": (
            "If unknown attempts and success rise, class-level Policy ranking of "
            "response-causal profile probes is a primary L2 bottleneck. If probes "
            "are forced often and unknown attempts rise but success stays 0/8, "
            "move next to within-candidate relational identity/memory or downstream "
            "response learning. If the intervention rarely fires, reject this "
            "hypothesis and inspect action-surface/state construction earlier."
        ),
    }

    base._write_csv(args.output_dir / "episodes.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
