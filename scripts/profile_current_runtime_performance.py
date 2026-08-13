from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from aassr_v2.current_entrypoint import build_current_pentest_aassr_core
from aassr_v2.current_hot_path_profile import current_hot_path_phase, current_hot_path_snapshot
from aassr_v2.current_protocol import run_current_episode
from aassr_v2.pentest_curriculum_env import STALL_PATIENCE
from aassr_v2.pentest_current_generation_smoke import _learning_counters
from aassr_v2.pentest_transfer_stages import TRANSFER_STAGES, TRANSFER_TRAIN_SEEDS


PROFILE_VERSION = "current-runtime-performance-profile-v1"
LEARNING_COUNTER_NAMES = (
    "dqn.environment_steps",
    "dqn.gradient_updates",
    "dqn.replay_size",
    "prophecy.observations",
    "prophecy.gradient_updates",
    "prophecy.replay_size",
    "critic.episodes",
    "critic.gradient_updates",
    "critic.replay_size",
    "evaluator.train_size",
    "evaluator.holdout_size",
    "policy.information_size",
    "policy.skill_values_size",
    "feature_memory_size",
    "skills_size",
    "predictor.bias",
    "predictor.weights",
)


def _sync(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    import torch

    torch.cuda.synchronize()


def _module_max_abs(left: object, right: object) -> float:
    left_state = left.state_dict()
    right_state = right.state_dict()
    if tuple(left_state) != tuple(right_state):
        raise AssertionError("profile comparison model state_dict keys differ")
    maximum = 0.0
    for key in left_state:
        a = left_state[key].detach()
        b = right_state[key].detach().to(a.device)
        if a.shape != b.shape:
            raise AssertionError(f"profile comparison tensor shape differs: {key}")
        if a.numel():
            maximum = max(maximum, float((a - b).abs().max().detach().cpu().item()))
    return maximum


def _parameter_difference(reference: object, optimized: object) -> dict[str, float]:
    prophecy = max(
        (
            _module_max_abs(left, right)
            for left, right in zip(
                reference.base_neural_prophecy.models,
                optimized.base_neural_prophecy.models,
                strict=True,
            )
        ),
        default=0.0,
    )
    return {
        "dqn_online_max_abs": _module_max_abs(reference.dqn.online, optimized.dqn.online),
        "dqn_target_max_abs": _module_max_abs(reference.dqn.target, optimized.dqn.target),
        "prophecy_ensemble_max_abs": prophecy,
        "critic_gru_max_abs": _module_max_abs(reference.critic.gru, optimized.critic.gru),
        "critic_output_max_abs": _module_max_abs(reference.critic.output, optimized.critic.output),
    }


def _run_training(
    *,
    research_seed: int,
    transition_budget: int,
    train_seeds: Sequence[int],
    device: str,
    performance: bool,
    label: str,
    progress_every: int,
) -> tuple[object, dict[str, Any]]:
    print(f"[{label}] starting {transition_budget} transitions on {device}", flush=True)
    agent = build_current_pentest_aassr_core(
        seed=int(research_seed),
        train_transitions=int(transition_budget),
        use_imagination=True,
        device=device,
        profile_hot_path=True,
        enable_performance_optimizations=bool(performance),
    )
    agent.requested_imagination = True

    transition_total = 0
    episode = 0
    rows = []
    _sync(device)
    started = time.perf_counter()
    next_progress = max(1, int(progress_every))
    with current_hot_path_phase(agent, "training"):
        while transition_total < transition_budget:
            stage = TRANSFER_STAGES[0]
            natural_cap = max(24, stage.rate_limit + STALL_PATIENCE)
            hard_left = transition_budget - transition_total
            cap = min(natural_cap, hard_left)
            scenario_seed = int(train_seeds[episode % len(train_seeds)])
            exploration = min(
                1000,
                int(1000 * transition_total / max(1, transition_budget)),
            )
            row, consumed = run_current_episode(
                agent,
                condition="aassr_current_performance_profile",
                research_seed=int(research_seed),
                stage_index=0,
                scenario_seed=scenario_seed,
                phase="performance_profile",
                block=0,
                episode=exploration,
                focus_level=0,
                transition_start=transition_total,
                transition_cap=cap,
                transition_budget=transition_budget,
                training=True,
                budget_cap=hard_left < natural_cap,
            )
            if consumed <= 0:
                raise RuntimeError("performance profile consumed no transitions")
            rows.append(row)
            transition_total += consumed
            episode += 1
            if transition_total >= next_progress or transition_total >= transition_budget:
                elapsed_so_far = time.perf_counter() - started
                print(
                    f"[{label}] {transition_total}/{transition_budget} transitions "
                    f"({elapsed_so_far:.1f}s elapsed)",
                    flush=True,
                )
                while next_progress <= transition_total:
                    next_progress += max(1, int(progress_every))
    _sync(device)
    elapsed = time.perf_counter() - started
    print(f"[{label}] complete in {elapsed:.1f}s", flush=True)

    diagnostics = agent.diagnostics()
    profile = current_hot_path_snapshot(
        agent,
        training_transitions=transition_total,
    )
    result = {
        "performance_enabled": bool(performance),
        "device": str(device),
        "wall_seconds": elapsed,
        "transitions": transition_total,
        "transitions_per_second": transition_total / elapsed if elapsed > 0.0 else 0.0,
        "episodes": episode,
        "episode_rows": [
            {
                "scenario_seed": int(row.scenario_seed),
                "status": row.status,
                "primitive_transitions": int(row.primitive_transitions),
                "success": int(row.success),
                "failure": int(row.failure),
                "stalled": int(row.stalled),
                "truncation": int(row.truncation),
            }
            for row in rows
        ],
        "learning_counters": list(_learning_counters(agent)),
        "dqn_updates": int(agent.dqn.gradient_updates),
        "prophecy_updates": int(agent.base_neural_prophecy.gradient_updates),
        "critic_updates": int(agent.critic.gradient_updates),
        "dqn_replay_size": len(agent.dqn.replay),
        "prophecy_replay_size": len(agent.base_neural_prophecy.replay),
        "critic_replay_size": len(agent.critic.replay),
        "runtime_performance_contract": getattr(
            agent,
            "current_runtime_performance_contract",
            "disabled-reference",
        ),
        "runtime_cuda_fast_path": bool(
            getattr(agent, "current_runtime_cuda_fast_path", False)
        ),
        "hot_path": profile,
        "hardware": diagnostics.get("hardware", {}),
        "prophecy_performance": {
            key: value
            for key, value in diagnostics.get("prophecy_current", {}).items()
            if key.startswith("performance_")
            or key in {
                "state_encode_cache_hits",
                "state_encode_cache_misses",
                "packed_host_transfer_batches",
                "training_metric_sync_batches",
                "deferred_ensemble_variance_sync",
                "vectorized_status_mixture_decode",
            }
        },
    }
    return agent, result


def _replay_equal(left: Iterable[Any], right: Iterable[Any]) -> bool:
    return tuple(left) == tuple(right)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare audited current AASSR runtime with performance fast paths OFF/ON "
            "without changing the scientific training contract."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--transitions", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress-every", type=int, default=64)
    parser.add_argument(
        "--output",
        default="runs/current_runtime_performance_profile.json",
    )
    args = parser.parse_args()
    if args.transitions <= 0:
        raise SystemExit("--transitions must be positive")
    if args.progress_every <= 0:
        raise SystemExit("--progress-every must be positive")

    seeds = TRANSFER_TRAIN_SEEDS[:4]
    reference_agent, reference = _run_training(
        research_seed=args.seed,
        transition_budget=args.transitions,
        train_seeds=seeds,
        device=args.device,
        performance=False,
        label="reference",
        progress_every=args.progress_every,
    )
    optimized_agent, optimized = _run_training(
        research_seed=args.seed,
        transition_budget=args.transitions,
        train_seeds=seeds,
        device=args.device,
        performance=True,
        label="optimized",
        progress_every=args.progress_every,
    )

    differences = _parameter_difference(reference_agent, optimized_agent)
    replay_match = {
        "dqn": _replay_equal(reference_agent.dqn.replay, optimized_agent.dqn.replay),
        "prophecy": _replay_equal(
            reference_agent.base_neural_prophecy.replay,
            optimized_agent.base_neural_prophecy.replay,
        ),
        "critic": _replay_equal(reference_agent.critic.replay, optimized_agent.critic.replay),
    }
    episode_rows_match = reference["episode_rows"] == optimized["episode_rows"]
    learning_counters_match = (
        reference["learning_counters"] == optimized["learning_counters"]
    )
    learning_counter_differences = {
        name: {
            "reference": reference["learning_counters"][index],
            "optimized": optimized["learning_counters"][index],
        }
        for index, name in enumerate(LEARNING_COUNTER_NAMES)
        if reference["learning_counters"][index]
        != optimized["learning_counters"][index]
    }
    exact_contract_match = bool(
        episode_rows_match and learning_counters_match and all(replay_match.values())
    )
    max_parameter_difference = max(differences.values(), default=0.0)
    # CUDA kernel formulation can introduce tiny roundoff while preserving the
    # exact sampled transitions. A larger drift is treated as a failed performance
    # patch rather than silently accepted.
    numerical_contract_match = max_parameter_difference <= 1e-5
    safe = bool(exact_contract_match and numerical_contract_match)

    comparison = {
        "exact_training_contract_match": exact_contract_match,
        "episode_rows_match": episode_rows_match,
        "learning_counters_match": learning_counters_match,
        "learning_counter_differences": learning_counter_differences,
        "numerical_parameter_contract_match": numerical_contract_match,
        "safe_for_scaling": safe,
        "replay_match": replay_match,
        "parameter_max_abs": differences,
        "maximum_parameter_abs_difference": max_parameter_difference,
        "wall_seconds_reference": reference["wall_seconds"],
        "wall_seconds_optimized": optimized["wall_seconds"],
        "speedup_x": (
            reference["wall_seconds"] / optimized["wall_seconds"]
            if optimized["wall_seconds"] > 0.0
            else 0.0
        ),
        "wall_reduction_fraction": (
            1.0 - optimized["wall_seconds"] / reference["wall_seconds"]
            if reference["wall_seconds"] > 0.0
            else 0.0
        ),
    }
    output = {
        "profile_version": PROFILE_VERSION,
        "seed": int(args.seed),
        "transition_budget_per_variant": int(args.transitions),
        "device": str(args.device),
        "reference": reference,
        "optimized": optimized,
        "comparison": comparison,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(comparison, indent=2, sort_keys=True))
    print(f"artifact: {path}")
    if not safe:
        raise SystemExit(
            "performance profile changed the audited training contract; do not run 10k"
        )


if __name__ == "__main__":
    main()
