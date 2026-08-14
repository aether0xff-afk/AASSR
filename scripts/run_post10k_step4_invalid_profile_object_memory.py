from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from scripts import run_post10k_checkpoint_diagnostic as base
from scripts import run_post10k_step2_probe_priority as step2
from scripts import run_post10k_step3_negative_profile_memory as step3


VERSION = "post10k-step4-invalid-profile-does-not-mark-object-tried-v1"


class _ResponseMemoryProxy:
    """Delegate response memory while correcting one invalid-profile bookkeeping case."""

    def __init__(self, delegate: object, counters: Counter[str]) -> None:
        self._delegate = delegate
        self._counters = counters

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def observe(
        self,
        action: object,
        response: object | None,
        *,
        authenticated: bool,
    ) -> None:
        tried = getattr(self._delegate, "tried_objects")
        before = set(tried)
        self._delegate.observe(action, response, authenticated=authenticated)

        if getattr(action, "verb_name", "") != "request_object" or response is None:
            return
        facts = set(getattr(response, "body_facts", ()))
        if "request_profile_not_applicable" not in facts:
            self._counters["legitimate_object_request_events"] += 1
            return

        self._counters["invalid_profile_object_events"] += 1
        object_id = action.parameters.get("object_id")
        if object_id is None:
            return
        object_id = str(object_id)
        if object_id in before:
            self._counters["preexisting_tried_object_preserved"] += 1
            return
        if object_id in tried:
            tried.discard(object_id)
            self._counters["prevented_new_global_tried_object"] += 1


def _install_invalid_profile_object_memory_fix(agent: object) -> Counter[str]:
    """Diagnostic only: an invalid profile must not newly mark its object as tried."""
    counters: Counter[str] = Counter()
    runtime_plugin = getattr(agent, "runtime_plugin", None)
    old_factory = getattr(runtime_plugin, "environment_factory", None)
    if not callable(old_factory):
        raise RuntimeError("runtime plugin has no environment_factory")

    def factory(*, scenario_seed: int, stage: object):
        world = old_factory(scenario_seed=int(scenario_seed), stage=stage)
        memory = getattr(world, "response_memory", None)
        if memory is None:
            raise RuntimeError("diagnostic world has no response_memory")
        world.response_memory = _ResponseMemoryProxy(memory, counters)
        counters["worlds_wrapped"] += 1
        return world

    runtime_plugin.environment_factory = factory
    agent._diagnostic_invalid_profile_object_memory = counters
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
            "Step-4 post-10k diagnosis: hold Iterations 2+3 fixed and change only "
            "tried-object bookkeeping for request_profile_not_applicable responses."
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
        default=Path("runs/post10k_step4_invalid_profile_object_memory"),
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

    # Hold Iterations 2 and 3 fixed. The only Iteration-4 delta is whether an
    # object request made with a publicly rejected profile is allowed to pollute
    # the global tried_object relation.
    probe_counters = step2._install_probe_priority(agent)
    negative_counters = step3._install_negative_route_profile_memory(agent)
    object_memory_counters = _install_invalid_profile_object_memory_fix(agent)

    stage = base.TRANSFER_STAGES[2]
    rows = [
        base._run_episode(
            agent,
            stage=stage,
            seed=seed,
            condition="step4_negative_profile_memory_plus_clean_object_trials",
        )
        for seed in seeds
    ]

    if base._learning_fingerprint(agent) != learning_before:
        raise AssertionError("step-4 frozen diagnostic mutated learned state")

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
            "Iteration-2 probe priority and Iteration-3 negative route/profile memory "
            "are held fixed. The sole Iteration-4 delta is that a public "
            "request_profile_not_applicable response cannot newly mark its object_id "
            "as globally tried."
        ),
        "single_changed_variable": (
            "for request_object only, if the returned public body contains "
            "request_profile_not_applicable, undo only the newly-added tried_object "
            "for that object; preserve any tried_object state that existed before "
            "the invalid-profile attempt"
        ),
        "unchanged_from_iteration3": [
            "environment dynamics and L2 stage",
            "checkpoint weights",
            "state/action neural encodings",
            "Policy values",
            "Iteration-2 probe-class priority scaffold",
            "Iteration-3 negative route/profile memory scaffold",
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
            "failures": int(experimental.get("failures", 0))
            - int(step3_reference.get("failures", 0)),
            "truncations": int(experimental.get("truncations", 0))
            - int(step3_reference.get("truncations", 0)),
            "mean_transitions": float(experimental.get("mean_transitions", 0.0))
            - float(step3_reference.get("mean_transitions", 0.0)),
            "mean_aseq_guards": float(experimental.get("mean_aseq_guards", 0.0))
            - float(step3_reference.get("mean_aseq_guards", 0.0)),
            "mean_unknown_attempts": float(
                experimental.get("mean_unknown_attempts", 0.0)
            ) - float(step3_reference.get("mean_unknown_attempts", 0.0)),
        },
        "probe_diagnostics": dict(probe_counters),
        "negative_profile_memory_diagnostics": dict(negative_counters),
        "object_trial_memory_diagnostics": dict(object_memory_counters),
        "frozen_learning_state_asserted": True,
        "interpretation_rule": (
            "If success rises materially over 1/8 while prevented_new_global_tried_object "
            "fires, invalid-profile pollution of tried_object is a major L2 defect. "
            "If the counter fires strongly but success stays near 1/8, keep the semantic "
            "bug finding but move next to positive read-profile/target discovery retention. "
            "If the counter rarely fires, reject it as a major L2 explanation."
        ),
    }

    base._write_csv(args.output_dir / "episodes.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
