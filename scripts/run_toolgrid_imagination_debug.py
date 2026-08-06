from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

from aassr_v2 import toolgrid_debug_clone as _toolgrid_debug_clone  # noqa: F401
from aassr_v2 import toolgrid_imagination_debug as debug
from aassr_v2 import toolgrid_factorial_masked as env


def _production_imagination_decision(agent, state):
    """Exercise the production gate while retaining full planner diagnostics."""

    original = agent.agent.config
    terminal_choice = (
        agent.use_imagination
        and agent.critic_ready
        and agent._predicted_terminal_choice(state)
    )
    agent.agent.config = replace(
        original,
        use_imagination=terminal_choice,
    )
    try:
        decision = agent.agent.select_action(state, episode=0, explore=False)
    finally:
        agent.agent.config = original

    if terminal_choice:
        return decision
    reason = (
        "critic_not_ready"
        if agent.use_imagination and not agent.critic_ready
        else "nonterminal_choice"
        if agent.use_imagination
        else "policy_only"
    )
    return replace(
        decision,
        imagination_opportunity=agent.use_imagination,
        imagination_gate_reason=reason,
    )


def _train_exact_budget(
    agent,
    *,
    seed: int,
    grid_size: int,
    action_count: int,
    transition_budget: int,
    train_map_count: int,
):
    cell_base = seed * 10_000_000 + grid_size * 100_000 + action_count * 1_000
    training_maps = tuple(cell_base + index for index in range(train_map_count))
    environment_steps = 0
    segment = 0
    completed_episodes = 0
    successes = 0
    started = time.perf_counter()

    while environment_steps < transition_budget:
        row, environment_steps = env._run_episode(
            agent,
            condition="imagination_v2",
            research_seed=seed,
            phase="diagnostic_training",
            grid_size=grid_size,
            action_count=action_count,
            checkpoint_transition_target=transition_budget,
            episode_index=segment,
            map_seed=training_maps[segment % len(training_maps)],
            training=True,
            environment_steps_total=environment_steps,
            schedule_horizon=transition_budget,
        )
        if row.termination != "budget_checkpoint":
            completed_episodes += 1
            successes += row.success
        segment += 1

    return {
        "training_transitions": environment_steps,
        "training_segments": segment,
        "training_episodes": completed_episodes,
        "training_successes": successes,
        "training_success_rate": (
            successes / completed_episodes if completed_episodes else 0.0
        ),
        "training_wall_seconds": time.perf_counter() - started,
    }


def _correct_diagnostic_labels(summary):
    """Avoid calling ended/not-ended agreement a three-class terminal metric."""

    components = summary.get("component_diagnostics", {})
    termination_accuracy = components.pop("prophecy_terminal_accuracy", None)
    if termination_accuracy is not None:
        components["prophecy_termination_status_accuracy"] = termination_accuracy
    components["prophecy_terminal_class_accuracy"] = None
    summary["config"]["terminal_metric_note"] = (
        "termination-status agreement is reported separately; success/failure "
        "calibration is enforced by the production calibrator"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train one corrected ToolGrid hybrid without training-time "
            "Imagination, then evaluate the identical checkpoint with the "
            "learned policy alone and with the production terminal-choice gate."
        )
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--grid-size",
        required=True,
        type=int,
        choices=env.GRID_SIZES,
    )
    parser.add_argument(
        "--action-count",
        required=True,
        type=int,
        choices=env.ACTION_COUNTS,
    )
    parser.add_argument("--transition-budget", type=int, default=5_000)
    parser.add_argument("--train-map-count", type=int, default=48)
    parser.add_argument("--evaluation-map-count", type=int, default=100)
    args = parser.parse_args()

    debug._imagination_decision = _production_imagination_decision
    debug._train_without_imagination = _train_exact_budget
    summary = debug.run_toolgrid_imagination_debug(
        args.output,
        seed=args.seed,
        grid_size=args.grid_size,
        action_count=args.action_count,
        transition_budget=args.transition_budget,
        train_map_count=args.train_map_count,
        evaluation_map_count=args.evaluation_map_count,
    )
    summary["config"]["debug_strategy"] = "production_corrected"
    summary["config"]["exact_transition_budget"] = True
    _correct_diagnostic_labels(summary)
    with open(f"{args.output}/summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=repr)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
