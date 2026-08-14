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
from scripts import run_post10k_step3_negative_profile_memory as step3


VERSION = "post10k-step5-read-profile-handoff-v1"


def _has_observed_read_profile(state: object) -> bool:
    return any(
        str(fact).startswith("observed_profile_role:")
        and str(fact).endswith(":read")
        for fact in getattr(state, "facts", ())
    )


def _unknown_object_profile_probe(state: object, action: object) -> bool:
    return bool(
        getattr(action, "verb_name", "") == "request_object"
        and base._unknown_profile(state, action)
    )


def _install_probe_until_read_handoff(agent: object) -> Counter[str]:
    """Iteration-2 probe priority, but hand control back after read-role discovery."""
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

        # Iteration 5 changes exactly one decision boundary relative to Iteration 3:
        # once the public observation has identified any profile as role=read,
        # stop the diagnostic forced-probe scaffold and return control to the
        # canonical frozen Policy. Before that milestone, behavior is identical to
        # Iteration 2/3 probe-class priority.
        if _has_observed_read_profile(state):
            counters["read_role_handoff_states"] += 1
            canonical = original_select(
                state,
                randomizer=randomizer,
                epsilon=epsilon,
                exploration_bonus=exploration_bonus,
            )
            if _unknown_object_profile_probe(state, canonical):
                counters["canonical_probe_after_read"] += 1
            else:
                counters["canonical_nonprobe_after_read"] += 1
            return canonical

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

        counters["forced_probe_choices_before_read"] += 1
        counters["overridden_nonprobe_choices_before_read"] += 1
        return selected

    policy.select = MethodType(select, policy)
    agent._diagnostic_probe_handoff = counters
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
            "Step-5 post-10k diagnosis: preserve Iteration-3 negative response "
            "memory, but stop the forced unknown-profile probe scaffold once a "
            "public read-profile role has been observed."
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
        default=Path("runs/post10k_step5_read_profile_handoff"),
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

    # Iteration 4 was rejected, so it is intentionally not carried forward.
    # Relative to Iteration 3, negative route/profile memory remains identical.
    # The sole delta is the probe-priority handoff after public read-role discovery.
    handoff_counters = _install_probe_until_read_handoff(agent)
    negative_counters = step3._install_negative_route_profile_memory(agent)

    stage = base.TRANSFER_STAGES[2]
    rows = [
        base._run_episode(
            agent,
            stage=stage,
            seed=seed,
            condition="step5_negative_memory_plus_read_profile_handoff",
        )
        for seed in seeds
    ]

    if base._learning_fingerprint(agent) != learning_before:
        raise AssertionError("step-5 frozen diagnostic mutated learned state")

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
            "Iteration 4 is rejected and not carried forward. Iteration 3 negative "
            "route/profile memory is held fixed. The sole Iteration-5 delta is "
            "ending the diagnostic forced unknown-profile probe priority after a "
            "public observed_profile_role:*:read fact appears."
        ),
        "single_changed_variable": (
            "before read-profile discovery, keep Iteration-3 probe-class priority; "
            "after any public read-profile role is observed, stop forced probing "
            "and return action selection to the canonical frozen Policy"
        ),
        "unchanged_from_iteration3": [
            "environment dynamics and L2 stage",
            "checkpoint weights",
            "state/action neural encodings",
            "Policy values",
            "negative route/profile memory",
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
        "handoff_diagnostics": dict(handoff_counters),
        "negative_profile_memory_diagnostics": dict(negative_counters),
        "frozen_learning_state_asserted": True,
        "interpretation_rule": (
            "If read_role_handoff_states fires and success rises materially over "
            "Iteration 3, the continuing forced-probe scaffold was masking a "
            "positive milestone and the canonical agent needs a general explore-to-"
            "exploit handoff signal rather than permanent candidate probing. If the "
            "handoff fires but success remains near 1/8, move downstream to whether "
            "workflow/target positive semantics are represented and selected. If the "
            "handoff rarely or never fires, the remaining bottleneck is earlier: "
            "finding the valid read-profile/target combination itself."
        ),
    }

    base._write_csv(args.output_dir / "episodes.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
