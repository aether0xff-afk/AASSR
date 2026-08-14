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


VERSION = "post10k-step1-structural-tie-randomization-v1"


def _install_structural_tie_randomization(agent: object) -> Counter[str]:
    """Diagnostic only: randomize concrete representatives of one top structural tie."""
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
        # This experiment is frozen evaluation only. Preserve nonzero-epsilon
        # behavior exactly in case the helper is reused accidentally.
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

        ranked = self_policy.rank(state, limit=len(actions))
        if not ranked:
            raise RuntimeError("policy returned no ranked actions")

        top = ranked[0]
        top_score = float(top.score)
        top_structure = base._structure(agent, state, top.action)

        tied_same_structure = tuple(
            item.action
            for item in ranked
            if abs(float(item.score) - top_score) <= 1e-12
            and base._structure(agent, state, item.action) == top_structure
        )
        if len(tied_same_structure) <= 1:
            return top.action

        counters["structural_tie_events"] += 1
        counters["tied_concrete_actions"] += len(tied_same_structure)
        if any(base._unknown_profile(state, action) for action in tied_same_structure):
            counters["unknown_profile_tie_events"] += 1

        # Sorting makes the random draw reproducible for the exact same concrete
        # action set. The selected index is random, so lexicographic signature no
        # longer decides which structurally equivalent concrete action executes.
        choices = tuple(sorted(tied_same_structure, key=lambda action: action.signature))
        selected = randomizer.choice(choices)
        counters["randomized_choices"] += 1
        if selected.signature != top.action.signature:
            counters["changed_from_signature_winner"] += 1
        return selected

    policy.select = MethodType(select, policy)
    agent._diagnostic_structural_tie = counters
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
        raise ValueError("baseline summary does not contain a baseline object")
    return dict(baseline)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Step-1 post-10k diagnosis: remove raw-signature winner bias only "
            "inside exact top-score structural action ties."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("runs/post10k_l2_diagnostic_seed7/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/post10k_step1_structural_tie"),
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
        parser.error(f"baseline summary not found: {args.baseline_summary}")

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
    tie_counters = _install_structural_tie_randomization(agent)

    stage = base.TRANSFER_STAGES[2]
    rows = [
        base._run_episode(
            agent,
            stage=stage,
            seed=seed,
            condition="step1_structural_tie_randomization",
        )
        for seed in seeds
    ]

    if base._learning_fingerprint(agent) != learning_before:
        raise AssertionError("step-1 frozen diagnostic mutated learned state")

    experimental = base._aggregate(rows)
    baseline_successes = int(baseline.get("successes", 0))
    experimental_successes = int(experimental.get("successes", 0))
    success_delta = experimental_successes - baseline_successes
    guard_delta = float(experimental.get("mean_aseq_guards", 0.0)) - float(
        baseline.get("mean_aseq_guards", 0.0)
    )

    result = {
        "version": VERSION,
        "checkpoint_git_commit": payload.get("git_commit"),
        "checkpoint": str(args.checkpoint),
        "baseline_summary": str(args.baseline_summary),
        "device": args.device,
        "scenario_seeds": list(seeds),
        "single_changed_variable": (
            "when the frozen Policy's top-scoring action belongs to an exact "
            "same-score/same-structural-feature concrete tie, choose uniformly "
            "within that one structural equivalence class instead of letting "
            "raw action.signature lexicographic order choose the representative"
        ),
        "unchanged": [
            "environment",
            "state representation",
            "action representation",
            "learned weights",
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
            "successes": success_delta,
            "success_rate": float(experimental.get("success_rate", 0.0))
            - float(baseline.get("success_rate", 0.0)),
            "mean_aseq_guards": guard_delta,
            "mean_unknown_attempts": float(
                experimental.get("mean_unknown_attempts", 0.0)
            )
            - float(baseline.get("mean_unknown_attempts", 0.0)),
        },
        "tie_diagnostics": dict(tie_counters),
        "frozen_learning_state_asserted": True,
        "interpretation_rule": (
            "If success appears and/or ASEQ guard pressure drops materially, "
            "raw-signature representative bias contributes to the L2 cliff. "
            "If behavior stays near baseline, reject this as the primary cause "
            "and move to the next single-variable representation experiment."
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
