from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean
from types import MethodType
from typing import Any, Mapping, Sequence

from aassr_v2 import pentest_curriculum_schedule as schedule
from aassr_v2.current_checkpoint import (
    _load_trusted_local_checkpoint,
    _restore_validated_current_frozen_payload,
    _validate_exact_current_provenance,
)
from aassr_v2.pentest_curriculum_env import STALL_PATIENCE
from aassr_v2.pentest_transfer_stages import (
    TRANSFER_DIAGNOSTIC_SEEDS,
    TRANSFER_STAGES,
)
from aassr_v2.types import Action, StateSnapshot

VERSION = "post10k-checkpoint-diagnostic-v1"


@dataclass(frozen=True, slots=True)
class EpisodeRow:
    condition: str
    seed: int
    profile_decoys: int
    success: int
    status: str
    transitions: int
    requests: int
    aseq_guards: int
    unknown_attempts: int
    unique_unknown_concrete: int
    unique_unknown_structural: int
    max_unknown_concrete: int
    max_unknown_structural: int
    alias_states: int
    filter_events: int
    shadow_plans: int
    shadow_disagreements: int


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=check,
        capture_output=True,
        text=True,
        timeout=20,
    )


def _assert_runtime_matches(commit: str) -> None:
    """Allow scripts-only diagnostic commits, but never runtime drift."""
    commit = str(commit).strip()
    if not commit:
        raise ValueError("checkpoint git_commit is empty")
    _git("cat-file", "-e", f"{commit}^{{commit}}")
    status = _git(
        "status", "--porcelain", "--untracked-files=normal", "--", "src/aassr_v2"
    )
    if status.stdout.strip():
        raise RuntimeError("src/aassr_v2 worktree is dirty")
    diff = _git(
        "diff", "--quiet", commit, "HEAD", "--", "src/aassr_v2", check=False
    )
    if diff.returncode == 1:
        raise RuntimeError(
            "src/aassr_v2 differs from checkpoint commit; refusing diagnosis"
        )
    if diff.returncode != 0:
        raise RuntimeError(f"git diff failed with {diff.returncode}")


def _restore(path: Path, *, device: str, allow_tf32: bool):
    payload = _load_trusted_local_checkpoint(path, device=device)
    commit = str(payload.get("git_commit", ""))
    _assert_runtime_matches(commit)
    _validate_exact_current_provenance(payload, expected_git_commit=commit)
    agent = _restore_validated_current_frozen_payload(
        payload, device=device, allow_tf32=allow_tf32
    )
    return agent, payload


def _learning_fingerprint(agent: object) -> tuple[int, ...]:
    return (
        int(agent.dqn.environment_steps),
        int(agent.dqn.gradient_updates),
        int(agent.base_neural_prophecy.observations),
        int(agent.base_neural_prophecy.gradient_updates),
        int(agent.critic.episodes),
        int(agent.critic.transitions),
        int(agent.critic.gradient_updates),
    )


def _profile_role(state: StateSnapshot, profile_id: str) -> str | None:
    prefix = f"observed_profile_role:{profile_id}:"
    for fact in state.facts:
        if fact.startswith(prefix):
            return fact.rsplit(":", 1)[1]
    return None


def _unknown_profile(state: StateSnapshot, action: Action) -> bool:
    profile_id = str(action.parameters.get("profile_id", ""))
    return bool(
        profile_id
        and profile_id != "profile-browse"
        and action.verb_name in {"request", "request_object"}
        and _profile_role(state, profile_id) is None
    )


def _unknown_actions(state: StateSnapshot) -> tuple[Action, ...]:
    return tuple(a for a in state.available_actions if _unknown_profile(state, a))


def _structure(agent: object, state: StateSnapshot, action: Action) -> tuple[float, ...]:
    return tuple(float(v) for v in agent.representation.action_structure(state, action))


def _aseq_guards(agent: object) -> int:
    return int(agent.diagnostics().get("aseq", {}).get("guard_events", 0))


def _status(world: object, *, stalled: bool, exhausted: bool) -> str:
    if bool(getattr(world, "success", False)) and bool(
        getattr(world, "proof_acquired", False)
    ):
        return "success"
    if bool(getattr(world, "failed", False)) and bool(getattr(world, "locked", False)):
        return "failure"
    if stalled:
        return "stalled"
    if bool(getattr(world, "rate_limited", False)) or exhausted:
        return "truncation"
    return "truncation"


def _install_without_replacement(agent: object) -> Counter[str]:
    """Diagnostic only: remember unchanged concrete candidates within semantic S."""
    counters: Counter[str] = Counter()
    seen: set[tuple[Any, str]] = set()
    old_begin = agent.begin_episode
    old_selection = agent._selection_state
    old_step = agent.step

    def begin(self_agent: object, *args: Any, **kwargs: Any):
        seen.clear()
        return old_begin(*args, **kwargs)

    def selection(self_agent: object, state: StateSnapshot):
        filtered, semantic, guarded, fallback = old_selection(state)
        candidates = _unknown_actions(filtered)
        if len(candidates) < 2:
            return filtered, semantic, guarded, fallback
        candidate_sigs = {a.signature for a in candidates}
        untried = {sig for sig in candidate_sigs if (semantic, sig) not in seen}
        if not untried:
            counters["all_tried_fallback"] += 1
            return filtered, semantic, guarded, fallback
        kept = tuple(
            a
            for a in filtered.available_actions
            if a.signature not in candidate_sigs or a.signature in untried
        )
        removed = len(filtered.available_actions) - len(kept)
        if removed:
            counters["filter_events"] += 1
            counters["filtered_actions"] += removed
            filtered = replace(filtered, available_actions=kept)
        return filtered, semantic, guarded, fallback

    def step(
        self_agent: object,
        environment: object,
        *,
        episode: int,
        training: bool = True,
        primitive_budget: int | None = None,
    ):
        result = old_step(
            environment,
            episode=episode,
            training=training,
            primitive_budget=primitive_budget,
        )
        if training or not result.traces:
            return result
        selected = result.decision.action
        before, after = result.traces[0].before, result.traces[-1].after
        if (
            _unknown_profile(before, selected)
            and schedule.semantic_fingerprint(before)
            == schedule.semantic_fingerprint(after)
        ):
            seen.add((schedule.semantic_fingerprint(before), selected.signature))
        return result

    agent.begin_episode = MethodType(begin, agent)
    agent._selection_state = MethodType(selection, agent)
    agent.step = MethodType(step, agent)
    agent._diagnostic_filter = counters
    return counters


def _install_shadow(
    agent: object,
    events: list[dict[str, Any]],
    *,
    max_plans_per_episode: int,
) -> None:
    """Run planner on a few ambiguous states and never execute its choice."""
    old_begin = agent.begin_episode
    old_core = agent._core_select_action
    seen_states: set[Any] = set()
    used = 0

    def begin(self_agent: object, *args: Any, **kwargs: Any):
        nonlocal used
        seen_states.clear()
        used = 0
        return old_begin(*args, **kwargs)

    def core(
        self_agent: object,
        state: StateSnapshot,
        *,
        episode: int,
        explore: bool,
    ):
        nonlocal used
        decision = old_core(state, episode=episode, explore=explore)
        if explore:
            return decision
        policy_sig = str(decision.policy_action_signature)
        if decision.action.signature != policy_sig:
            raise RuntimeError(
                "canonical Imagination intervened; shadow mode requires Policy execution"
            )
        candidates = _unknown_actions(state)
        semantic = schedule.semantic_fingerprint(state)
        if (
            len(candidates) < 2
            or semantic in seen_states
            or used >= max_plans_per_episode
        ):
            return decision

        seen_states.add(semantic)
        used += 1
        started = time.perf_counter()
        plan = self_agent.planner.plan(state)
        roots = []
        for root in plan.root_evaluations:
            action = root.action
            confidence = float(self_agent.skill_prophecy.confidence(state, action))
            support_fn = getattr(self_agent.critic, "support_confidence", None)
            support = float(support_fn(state, action)) if callable(support_fn) else 0.0
            roots.append(
                {
                    "action_signature": action.signature,
                    "value": float(root.aggregate_value),
                    "confidence": confidence,
                    "critic_support": support,
                    "unknown_profile": _unknown_profile(state, action),
                }
            )

        threshold = float(self_agent.config.imagination_minimum_coverage)
        raw = max(roots, key=lambda r: (r["value"], r["action_signature"]), default=None)
        reliable = max(
            (r for r in roots if r["confidence"] >= threshold),
            key=lambda r: (r["value"], r["action_signature"]),
            default=None,
        )
        events.append(
            {
                "event_index": len(events),
                "gate_reason": str(decision.imagination_gate_reason),
                "model_coverage": float(decision.model_coverage),
                "policy_action": policy_sig,
                "raw_preferred": None if raw is None else raw["action_signature"],
                "reliable_preferred": None if reliable is None else reliable["action_signature"],
                "raw_disagreement": bool(
                    raw is not None and raw["action_signature"] != policy_sig
                ),
                "reliable_disagreement": bool(
                    reliable is not None and reliable["action_signature"] != policy_sig
                ),
                "unknown_concrete": len(candidates),
                "unknown_structural": len(
                    {_structure(self_agent, state, a) for a in candidates}
                ),
                "planner_nodes": len(plan.nodes),
                "planner_depth": int(plan.maximum_depth_reached),
                "wall_seconds": time.perf_counter() - started,
                "roots": roots,
            }
        )
        return decision

    agent.begin_episode = MethodType(begin, agent)
    agent._core_select_action = MethodType(core, agent)


def _run_episode(agent: object, *, stage: object, seed: int, condition: str) -> EpisodeRow:
    factory = getattr(getattr(agent, "runtime_plugin", None), "environment_factory", None)
    if not callable(factory):
        raise RuntimeError("runtime plugin has no environment_factory")
    world = factory(scenario_seed=int(seed), stage=stage)
    agent.begin_episode()

    guards_before = _aseq_guards(agent)
    filter_counter: Counter[str] = getattr(agent, "_diagnostic_filter", Counter())
    filter_before = int(filter_counter.get("filter_events", 0))
    shadow_events: list[dict[str, Any]] = getattr(agent, "_diagnostic_shadow", [])
    shadow_before = len(shadow_events)

    transitions = 0
    unchanged = 0
    stalled = False
    recent: deque[tuple[Any, str]] = deque(maxlen=STALL_PATIENCE)
    cap = max(24, int(stage.rate_limit) + STALL_PATIENCE)
    attempts = 0
    concrete: set[str] = set()
    structural: set[tuple[float, ...]] = set()
    max_concrete = max_structural = alias_states = 0

    while (
        transitions < cap
        and not bool(getattr(world, "success", False))
        and not (
            bool(getattr(world, "failed", False))
            and bool(getattr(world, "locked", False))
        )
        and not bool(getattr(world, "rate_limited", False))
    ):
        result = agent.step(
            world,
            episode=transitions,
            training=False,
            primitive_budget=cap - transitions,
        )
        if not result.traces:
            break
        for trace in result.traces:
            unknown = _unknown_actions(trace.before)
            structures = {_structure(agent, trace.before, a) for a in unknown}
            max_concrete = max(max_concrete, len(unknown))
            max_structural = max(max_structural, len(structures))
            alias_states += int(len(unknown) > len(structures))
            if _unknown_profile(trace.before, trace.action):
                attempts += 1
                concrete.add(trace.action.signature)
                structural.add(_structure(agent, trace.before, trace.action))

            before = schedule.semantic_fingerprint(trace.before)
            after = schedule.semantic_fingerprint(trace.after)
            unchanged = unchanged + 1 if before == after else 0
            recent.append((before, trace.action.signature))
            transitions += 1
            if unchanged >= STALL_PATIENCE:
                counts = Counter(recent)
                stalled = len(counts) <= 3 or max(counts.values(), default=0) >= 4
            if stalled or transitions >= cap:
                break
        if stalled:
            break

    status = _status(world, stalled=stalled, exhausted=transitions >= cap)
    agent.finish_episode(
        final_return=1.0 if status == "success" else -1.0 if status == "failure" else 0.0,
        training=False,
    )
    episode_shadow = shadow_events[shadow_before:]
    return EpisodeRow(
        condition=condition,
        seed=int(seed),
        profile_decoys=int(stage.extra_profile_count),
        success=int(status == "success"),
        status=status,
        transitions=transitions,
        requests=int(getattr(world, "request_count", 0)),
        aseq_guards=_aseq_guards(agent) - guards_before,
        unknown_attempts=attempts,
        unique_unknown_concrete=len(concrete),
        unique_unknown_structural=len(structural),
        max_unknown_concrete=max_concrete,
        max_unknown_structural=max_structural,
        alias_states=alias_states,
        filter_events=int(filter_counter.get("filter_events", 0)) - filter_before,
        shadow_plans=len(episode_shadow),
        shadow_disagreements=sum(
            bool(event["reliable_disagreement"]) for event in episode_shadow
        ),
    )


def _aggregate(rows: Sequence[EpisodeRow]) -> dict[str, Any]:
    if not rows:
        return {"episodes": 0, "successes": 0, "success_rate": 0.0}
    return {
        "episodes": len(rows),
        "successes": sum(r.success for r in rows),
        "success_rate": fmean(r.success for r in rows),
        "stalls": sum(r.status == "stalled" for r in rows),
        "truncations": sum(r.status == "truncation" for r in rows),
        "mean_transitions": fmean(r.transitions for r in rows),
        "mean_requests": fmean(r.requests for r in rows),
        "mean_aseq_guards": fmean(r.aseq_guards for r in rows),
        "mean_unknown_attempts": fmean(r.unknown_attempts for r in rows),
        "mean_unique_unknown_concrete": fmean(r.unique_unknown_concrete for r in rows),
        "mean_unique_unknown_structural": fmean(
            r.unique_unknown_structural for r in rows
        ),
        "alias_states": sum(r.alias_states for r in rows),
        "filter_events": sum(r.filter_events for r in rows),
        "shadow_plans": sum(r.shadow_plans for r in rows),
        "shadow_disagreements": sum(r.shadow_disagreements for r in rows),
    }


def _run_condition(
    checkpoint: Path,
    *,
    device: str,
    allow_tf32: bool,
    stage: object,
    seeds: Sequence[int],
    condition: str,
    without_replacement: bool = False,
    shadow: bool = False,
    max_shadow_plans: int = 4,
):
    agent, payload = _restore(checkpoint, device=device, allow_tf32=allow_tf32)
    learning_before = _learning_fingerprint(agent)
    imagination_before = dict(agent.imagination_diagnostics())
    agent.requested_imagination = bool(shadow)
    if without_replacement:
        _install_without_replacement(agent)

    shadow_events: list[dict[str, Any]] = []
    if shadow:
        agent._diagnostic_shadow = shadow_events
        _install_shadow(agent, shadow_events, max_plans_per_episode=max_shadow_plans)

    rows = [
        _run_episode(agent, stage=stage, seed=int(seed), condition=condition)
        for seed in seeds
    ]
    if _learning_fingerprint(agent) != learning_before:
        raise AssertionError("frozen diagnostic mutated learned state")

    diagnostics = agent.diagnostics()
    critic_support = dict(diagnostics.get("critic_support", {}))
    gate_delta = {
        str(key).removeprefix("gate:"): int(value)
        - int(imagination_before.get(key, 0))
        for key, value in diagnostics.get("imagination", {}).items()
        if str(key).startswith("gate:")
        and int(value) - int(imagination_before.get(key, 0)) != 0
    }
    metadata = {
        "checkpoint_git_commit": payload.get("git_commit"),
        "critic_ready": bool(agent.critic_ready),
        "critic_reliably_ready": bool(
            agent.critic_reliably_ready()
            if callable(getattr(agent, "critic_reliably_ready", None))
            else agent.critic_ready
        ),
        "recent_signed_support": dict(critic_support.get("recent_signed_support", {})),
        "gate_delta": gate_delta,
    }
    return rows, shadow_events, metadata


def _parse_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def _write_csv(path: Path, rows: Sequence[EpisodeRow]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose the 10k L2 cliff from one frozen checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/post10k_checkpoint_diagnostic"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--profile-counts", type=_parse_ints, default=(0, 1, 2, 4))
    parser.add_argument(
        "--scenario-seeds", type=_parse_ints, default=TRANSFER_DIAGNOSTIC_SEEDS
    )
    parser.add_argument("--skip-without-replacement", action="store_true")
    parser.add_argument("--skip-shadow", action="store_true")
    parser.add_argument("--shadow-max-plans-per-episode", type=int, default=4)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if any(x < 0 for x in args.profile_counts):
        parser.error("--profile-counts must be non-negative")
    if args.shadow_max_plans_per_episode <= 0:
        parser.error("--shadow-max-plans-per-episode must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    l2 = TRANSFER_STAGES[2]
    rows_all: list[EpisodeRow] = []
    shadow_all: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    sweep: dict[int, dict[str, Any]] = {}

    for count in args.profile_counts:
        stage = replace(
            l2,
            name=f"l2_profile_choices_decoys_{count}",
            extra_profile_count=int(count),
        )
        condition = f"profile_sweep_decoys_{count}"
        rows, events, meta = _run_condition(
            args.checkpoint,
            device=args.device,
            allow_tf32=not args.no_tf32,
            stage=stage,
            seeds=args.scenario_seeds,
            condition=condition,
        )
        rows_all.extend(rows)
        shadow_all.extend(events)
        sweep[int(count)] = _aggregate(rows)
        metadata[condition] = meta

    baseline_count = max(args.profile_counts)
    baseline_stage = replace(
        l2,
        name=f"l2_profile_choices_decoys_{baseline_count}",
        extra_profile_count=int(baseline_count),
    )
    baseline = sweep[int(baseline_count)]

    without = {"skipped": True, "successes": 0}
    if not args.skip_without_replacement:
        condition = f"profile_decoys_{baseline_count}_without_replacement"
        rows, events, meta = _run_condition(
            args.checkpoint,
            device=args.device,
            allow_tf32=not args.no_tf32,
            stage=baseline_stage,
            seeds=args.scenario_seeds,
            condition=condition,
            without_replacement=True,
        )
        rows_all.extend(rows)
        without = {"skipped": False, **_aggregate(rows)}
        metadata[condition] = meta

    shadow = {"skipped": True, "shadow_plans": 0}
    if not args.skip_shadow:
        condition = f"profile_decoys_{baseline_count}_shadow"
        rows, events, meta = _run_condition(
            args.checkpoint,
            device=args.device,
            allow_tf32=not args.no_tf32,
            stage=baseline_stage,
            seeds=args.scenario_seeds,
            condition=condition,
            shadow=True,
            max_shadow_plans=args.shadow_max_plans_per_episode,
        )
        rows_all.extend(rows)
        for event in events:
            event["condition"] = condition
        shadow_all.extend(events)
        shadow = {"skipped": False, **_aggregate(rows)}
        metadata[condition] = meta

    counts = sorted(sweep)
    rates = [float(sweep[c]["success_rate"]) for c in counts]
    monotonic = all(b <= a + 1e-12 for a, b in zip(rates, rates[1:]))
    wr_delta = int(without.get("successes", 0)) - int(baseline.get("successes", 0))
    hypothesis = {
        "profile_choice_scaling_signal": (
            "supported"
            if rates and rates[0] > rates[-1] and monotonic
            else "not_established"
        ),
        "without_replacement_memory_signal": (
            "supported"
            if not without.get("skipped") and wr_delta >= 2
            else "not_established"
        ),
        "without_replacement_success_delta": wr_delta,
        "shadow_imagination_observed": int(shadow.get("shadow_plans", 0)) > 0,
        "contract": (
            "diagnostic only: no learned weights change; profile sweep changes only "
            "extra_profile_count; without-replacement never reveals the correct "
            "profile; shadow planner never changes the executed action"
        ),
    }

    _write_csv(args.output_dir / "episodes.csv", rows_all)
    with (args.output_dir / "shadow_events.jsonl").open("w", encoding="utf-8") as handle:
        for event in shadow_all:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    result = {
        "version": VERSION,
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "scenario_seeds": list(map(int, args.scenario_seeds)),
        "stage": {
            "level": 2,
            "base_name": l2.name,
            "only_swept_axis": "extra_profile_count",
            "object_count": l2.object_count,
            "decoy_route_count": l2.decoy_route_count,
            "require_csrf": l2.require_csrf,
            "workflow_depth": l2.workflow_depth,
            "lockout_threshold": l2.lockout_threshold,
            "rate_limit": l2.rate_limit,
            "session_ttl": l2.session_ttl,
        },
        "profile_sweep": {str(k): sweep[k] for k in counts},
        "baseline": baseline,
        "without_replacement": without,
        "shadow": shadow,
        "hypothesis": hypothesis,
        "condition_metadata": metadata,
        "frozen_learning_state_asserted": True,
        "runtime_provenance": (
            "src/aassr_v2 must be clean and byte-identical to checkpoint commit"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
