from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import threading
import time
import traceback
from typing import Any, Callable, Mapping

from .autonomous_agent import AutonomousAgentConfig, AutonomousLearningAgent
from .escape_gridworld import (
    EscapeGridSpec,
    EscapeGridWorld,
    generate_escape_grid,
    oracle_plan,
)
from .escape_reporting import (
    EscapeEpisodeRecord,
    EscapeSessionRecorder,
    build_statistics,
    serialize_action,
    serialize_snapshot,
    utc_now_iso,
)
from .tabular_prophecy import TabularProphecy


class TrainingMode(str, Enum):
    LIVE = "live"
    FAST = "fast"


class TrainingRuntime:
    """Thread-safe display mode shared by the GUI and training worker."""

    def __init__(self, mode: TrainingMode = TrainingMode.FAST) -> None:
        self._mode = TrainingMode(mode)
        self._lock = threading.Lock()

    @property
    def mode(self) -> TrainingMode:
        with self._lock:
            return self._mode

    def set_mode(self, mode: TrainingMode) -> None:
        with self._lock:
            self._mode = TrainingMode(mode)


@dataclass(frozen=True, slots=True)
class EscapeTrainingConfig:
    episodes: int = 2_000
    seed: int = 7
    color_count: int = 2
    distractor_boxes: int = 2
    gamma: float = 0.97
    policy_learning_rate: float = 0.2
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 1_500
    use_imagination: bool = True
    imagination_depth: int = 5
    imagination_branching_factor: int = 2
    imagination_beam_width: int = 16
    imagination_minimum_coverage: float = 0.35
    validated_gain_weight: float = 0.2
    repeat_penalty: float = 0.05
    error_penalty: float = 0.2
    minimum_holdout_count: int = 4
    validation_interval: int = 8
    imagination_interval: int = 1
    live_step_delay: float = 0.06
    fast_progress_interval: int = 25
    rolling_window: int = 100
    efficiency_bonus_scale: float = 1.0
    save_episode_checkpoints: bool = True
    checkpoint_interval: int = 100
    checkpoint_retention: int = 10
    step_flush_interval: int = 64
    max_step_log_bytes: int = 1_073_741_824

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 < self.policy_learning_rate <= 1.0:
            raise ValueError("policy_learning_rate must be in (0, 1]")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if self.epsilon_decay_episodes <= 0:
            raise ValueError("epsilon_decay_episodes must be positive")
        if self.imagination_depth <= 0:
            raise ValueError("imagination_depth must be positive")
        if self.imagination_branching_factor <= 0:
            raise ValueError("imagination_branching_factor must be positive")
        if self.imagination_beam_width <= 0:
            raise ValueError("imagination_beam_width must be positive")
        if self.live_step_delay < 0.0:
            raise ValueError("live_step_delay must be non-negative")
        if self.fast_progress_interval <= 0:
            raise ValueError("fast_progress_interval must be positive")
        if self.rolling_window <= 0:
            raise ValueError("rolling_window must be positive")
        if self.efficiency_bonus_scale < 0.0:
            raise ValueError("efficiency_bonus_scale must be non-negative")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if self.checkpoint_retention < 0:
            raise ValueError("checkpoint_retention must be non-negative")
        if self.step_flush_interval <= 0:
            raise ValueError("step_flush_interval must be positive")
        if self.max_step_log_bytes < 0:
            raise ValueError("max_step_log_bytes must be non-negative")


@dataclass(frozen=True, slots=True)
class EscapeRenderFrame:
    spec: EscapeGridSpec
    episode: int
    total_episodes: int
    step: int
    position: tuple[int, int]
    inventory: tuple[str, ...]
    open_boxes: frozenset[str]
    open_doors: frozenset[tuple[int, int]]
    action: str
    event: str
    success: bool
    episode_finished: bool
    epsilon: float
    rolling_success: float
    rolling_score: float
    total_successes: int
    total_score: float
    episode_score: float
    elapsed_seconds: float
    mode: TrainingMode
    used_imagination: bool = False
    imagined_nodes: int = 0
    prediction_score: float = 0.0
    holdout_gain: float = 0.0
    intrinsic_value: float = 0.0


@dataclass(frozen=True, slots=True)
class EscapeTrainingSummary:
    episodes: int
    successes: int
    success_rate: float
    rolling_success: float
    total_score: float
    mean_score: float
    rolling_score: float
    elapsed_seconds: float
    total_steps: int
    policy_entries: int
    oracle_steps: int
    imagination_decisions: int
    imagined_nodes: int
    stopped: bool
    output_dir: str
    episode_records: tuple[EscapeEpisodeRecord, ...]
    statistics: Mapping[str, Any]


FrameCallback = Callable[[EscapeRenderFrame], None]
CompleteCallback = Callable[[EscapeTrainingSummary], None]


def epsilon_for_episode(config: EscapeTrainingConfig, episode: int) -> float:
    fraction = min(1.0, max(0.0, episode / config.epsilon_decay_episodes))
    return config.epsilon_start + fraction * (
        config.epsilon_end - config.epsilon_start
    )


def success_score_multiplier(
    optimal_steps: int,
    actual_steps: int,
    *,
    bonus_scale: float = 1.0,
) -> float:
    """Score a completed escape without introducing intermediate rewards."""

    if optimal_steps <= 0 or actual_steps <= 0:
        raise ValueError("step counts must be positive")
    if bonus_scale < 0.0:
        raise ValueError("bonus_scale must be non-negative")
    efficiency = min(1.0, optimal_steps / actual_steps)
    return 1.0 + bonus_scale * efficiency


def _make_agent(config: EscapeTrainingConfig) -> AutonomousLearningAgent:
    agent_config = AutonomousAgentConfig(
        gamma=config.gamma,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay_episodes=config.epsilon_decay_episodes,
        policy_learning_rate=config.policy_learning_rate,
        use_imagination=config.use_imagination,
        imagination_depth=config.imagination_depth,
        imagination_branching_factor=config.imagination_branching_factor,
        imagination_beam_width=config.imagination_beam_width,
        imagination_minimum_coverage=config.imagination_minimum_coverage,
        validated_gain_weight=config.validated_gain_weight,
        repeat_penalty=config.repeat_penalty,
        error_penalty=config.error_penalty,
        minimum_holdout_count=config.minimum_holdout_count,
        validation_interval=config.validation_interval,
        imagination_interval=config.imagination_interval,
    )
    return AutonomousLearningAgent(
        TabularProphecy(),
        config=agent_config,
        seed=config.seed,
    )


def _policy_entry_count(agent: AutonomousLearningAgent) -> int:
    return len(getattr(agent.policy, "_local", {}))


def _prophecy_entry_count(agent: AutonomousLearningAgent) -> int:
    return len(getattr(agent.prophecy, "_exact", {}))


def _holdout_size(agent: AutonomousLearningAgent) -> int:
    return len(getattr(agent.holdout, "_items", ()))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _frame(
    environment: EscapeGridWorld,
    *,
    episode: int,
    total_episodes: int,
    action: str,
    event: str,
    episode_finished: bool,
    epsilon: float,
    rolling_success: float,
    rolling_score: float,
    total_successes: int,
    total_score: float,
    episode_score: float,
    elapsed_seconds: float,
    mode: TrainingMode,
    used_imagination: bool = False,
    imagined_nodes: int = 0,
    prediction_score: float = 0.0,
    holdout_gain: float = 0.0,
    intrinsic_value: float = 0.0,
) -> EscapeRenderFrame:
    return EscapeRenderFrame(
        spec=environment.spec,
        episode=episode,
        total_episodes=total_episodes,
        step=environment.steps,
        position=environment.position,
        inventory=tuple(sorted(environment.inventory)),
        open_boxes=frozenset(environment.open_boxes),
        open_doors=frozenset(environment.open_doors),
        action=action,
        event=event,
        success=environment.success,
        episode_finished=episode_finished,
        epsilon=epsilon,
        rolling_success=rolling_success,
        rolling_score=rolling_score,
        total_successes=total_successes,
        total_score=total_score,
        episode_score=episode_score,
        elapsed_seconds=elapsed_seconds,
        mode=mode,
        used_imagination=used_imagination,
        imagined_nodes=imagined_nodes,
        prediction_score=prediction_score,
        holdout_gain=holdout_gain,
        intrinsic_value=intrinsic_value,
    )


def train_escape_agent(
    config: EscapeTrainingConfig,
    *,
    mode: TrainingMode = TrainingMode.FAST,
    runtime: TrainingRuntime | None = None,
    on_frame: FrameCallback | None = None,
    on_complete: CompleteCallback | None = None,
    stop_event: threading.Event | None = None,
    output_dir: str | Path | None = None,
) -> EscapeTrainingSummary:
    """Train Full AASSR and durably save every step, episode, model and chart."""

    stop = stop_event or threading.Event()
    display = runtime or TrainingRuntime(mode)
    spec = generate_escape_grid(
        config.seed,
        color_count=config.color_count,
        distractor_boxes=config.distractor_boxes,
    )
    oracle_steps = len(oracle_plan(spec))
    agent = _make_agent(config)
    recorder = EscapeSessionRecorder(
        config=config,
        spec=spec,
        oracle_steps=oracle_steps,
        initial_mode=display.mode.value,
        output_dir=output_dir,
    )
    started = time.perf_counter()
    outcomes: list[int] = []
    scores: list[float] = []
    successes = 0
    total_score = 0.0
    imagination_decisions_total = 0
    imagined_nodes_total = 0
    last_mode = display.mode
    final_summary: EscapeTrainingSummary | None = None
    finalized = False

    try:
        for episode_index in range(config.episodes):
            if stop.is_set():
                break
            episode = episode_index + 1
            epsilon = agent.epsilon(episode_index)
            environment = EscapeGridWorld(spec)
            episode_started_perf = time.perf_counter()
            episode_started_utc = utc_now_iso()
            action_counts: Counter[str] = Counter()
            event_counts: Counter[str] = Counter()
            predictions: list[float] = []
            holdout_before_values: list[float] = []
            holdout_after_values: list[float] = []
            holdout_gains: list[float] = []
            intrinsic_values: list[float] = []
            errors = 0
            repeats = 0
            move_actions = 0
            interaction_actions = 0
            blocked_moves = 0
            found_keys = 0
            opened_doors = 0
            empty_boxes = 0
            episode_imagination_decisions = 0
            episode_imagined_nodes = 0
            maximum_imagination_depth = 0
            mode_seconds = {TrainingMode.LIVE: 0.0, TrainingMode.FAST: 0.0}
            latest_imagination = False
            latest_nodes = 0
            latest_prediction = 0.0
            latest_gain = 0.0
            latest_intrinsic = 0.0

            current_mode = display.mode
            if current_mode is not last_mode:
                recorder.record_mode_switch(
                    previous_mode=last_mode.value,
                    new_mode=current_mode.value,
                    session_elapsed_seconds=time.perf_counter() - started,
                    episode=episode,
                    step=0,
                )
                last_mode = current_mode
            if current_mode is TrainingMode.LIVE and on_frame is not None:
                recent_outcomes = outcomes[-config.rolling_window :]
                recent_scores = scores[-config.rolling_window :]
                on_frame(
                    _frame(
                        environment,
                        episode=episode,
                        total_episodes=config.episodes,
                        action="",
                        event="episode_started",
                        episode_finished=False,
                        epsilon=epsilon,
                        rolling_success=_mean([float(item) for item in recent_outcomes]),
                        rolling_score=_mean(recent_scores),
                        total_successes=successes,
                        total_score=total_score,
                        episode_score=0.0,
                        elapsed_seconds=time.perf_counter() - started,
                        mode=current_mode,
                    )
                )

            interrupted = False
            while not environment.success:
                if stop.is_set():
                    interrupted = True
                    break
                tick_started = time.perf_counter()
                current_mode = display.mode
                if current_mode is not last_mode:
                    recorder.record_mode_switch(
                        previous_mode=last_mode.value,
                        new_mode=current_mode.value,
                        session_elapsed_seconds=tick_started - started,
                        episode=episode,
                        step=environment.steps,
                    )
                    last_mode = current_mode

                before = environment.snapshot()
                decision = agent.select_action(
                    before,
                    episode=episode_index,
                    explore=True,
                )
                outcome = environment.step(decision.action)
                metrics = agent.observe(before, decision.action, outcome)
                compute_seconds = time.perf_counter() - tick_started

                action_counts[decision.action.signature] += 1
                event_counts[outcome.event] += 1
                predictions.append(metrics.prediction_score)
                holdout_before_values.append(metrics.holdout_before)
                holdout_after_values.append(metrics.holdout_after)
                holdout_gains.append(metrics.holdout_gain)
                intrinsic_values.append(metrics.intrinsic_value)
                errors += int(metrics.error)
                repeats += int(metrics.repeated)
                move_actions += int(decision.action.verb_name == "move")
                interaction_actions += int(decision.action.verb_name == "interact")
                blocked_moves += int(outcome.event == "blocked")
                found_keys += int(outcome.event.startswith("found_key:"))
                opened_doors += int(outcome.event.startswith("opened_door:"))
                empty_boxes += int(outcome.event == "empty_box")
                latest_imagination = decision.used_imagination
                latest_nodes = decision.imagined_nodes
                latest_prediction = metrics.prediction_score
                latest_gain = metrics.holdout_gain
                latest_intrinsic = metrics.intrinsic_value
                if decision.used_imagination:
                    episode_imagination_decisions += 1
                    episode_imagined_nodes += decision.imagined_nodes
                    maximum_imagination_depth = max(
                        maximum_imagination_depth,
                        decision.imagination_depth,
                    )
                    imagination_decisions_total += 1
                    imagined_nodes_total += decision.imagined_nodes

                if current_mode is TrainingMode.LIVE:
                    recent_outcomes = outcomes[-config.rolling_window :]
                    recent_scores = scores[-config.rolling_window :]
                    if on_frame is not None:
                        on_frame(
                            _frame(
                                environment,
                                episode=episode,
                                total_episodes=config.episodes,
                                action=decision.action.signature,
                                event=outcome.event,
                                episode_finished=environment.success,
                                epsilon=epsilon,
                                rolling_success=_mean(
                                    [float(item) for item in recent_outcomes]
                                ),
                                rolling_score=_mean(recent_scores),
                                total_successes=successes,
                                total_score=total_score,
                                episode_score=0.0,
                                elapsed_seconds=time.perf_counter() - started,
                                mode=current_mode,
                                used_imagination=decision.used_imagination,
                                imagined_nodes=decision.imagined_nodes,
                                prediction_score=metrics.prediction_score,
                                holdout_gain=metrics.holdout_gain,
                                intrinsic_value=metrics.intrinsic_value,
                            )
                        )
                    if config.live_step_delay and stop.wait(config.live_step_delay):
                        interrupted = True

                tick_ended = time.perf_counter()
                tick_wall_seconds = tick_ended - tick_started
                mode_seconds[current_mode] += tick_wall_seconds
                recorder.record_step(
                    {
                        "timestamp_utc": utc_now_iso(),
                        "episode": episode,
                        "step": environment.steps,
                        "session_elapsed_seconds": tick_ended - started,
                        "episode_elapsed_seconds": tick_ended - episode_started_perf,
                        "tick_wall_seconds": tick_wall_seconds,
                        "compute_seconds": compute_seconds,
                        "mode": current_mode.value,
                        "epsilon": epsilon,
                        "action": serialize_action(decision.action),
                        "event": outcome.event,
                        "error": outcome.error,
                        "goal_reached": outcome.goal_reached,
                        "external_reward": outcome.reward,
                        "added_facts": sorted(outcome.added_facts),
                        "removed_facts": sorted(outcome.removed_facts),
                        "unlocked_actions": [
                            serialize_action(action) for action in outcome.unlocked_actions
                        ],
                        "decision": {
                            "used_imagination": decision.used_imagination,
                            "imagined_nodes": decision.imagined_nodes,
                            "imagination_depth": decision.imagination_depth,
                            "root_imagined_value": decision.root_imagined_value,
                        },
                        "metrics": {
                            "prediction_score": metrics.prediction_score,
                            "holdout_before": metrics.holdout_before,
                            "holdout_after": metrics.holdout_after,
                            "holdout_gain": metrics.holdout_gain,
                            "intrinsic_value": metrics.intrinsic_value,
                            "repeated": metrics.repeated,
                            "error": metrics.error,
                        },
                        "before": serialize_snapshot(before),
                        "after": serialize_snapshot(outcome.snapshot),
                    }
                )
                if interrupted:
                    break

            episode_ended_perf = time.perf_counter()
            episode_ended_utc = utc_now_iso()
            duration_seconds = episode_ended_perf - episode_started_perf
            score = 0.0
            success = environment.success and not interrupted
            if success:
                score = success_score_multiplier(
                    oracle_steps,
                    environment.steps,
                    bonus_scale=config.efficiency_bonus_scale,
                )
                agent.finish_episode(final_return=score)
                successes += 1
                total_score += score
                outcomes.append(1)
                scores.append(score)
            else:
                agent.discard_episode()
                outcomes.append(0)
                scores.append(0.0)

            recent_scores = scores[-config.rolling_window :]
            rolling_score = _mean(recent_scores)
            record = EscapeEpisodeRecord(
                session_id=recorder.session_id,
                episode=episode,
                success=success,
                interrupted=interrupted,
                started_at_utc=episode_started_utc,
                ended_at_utc=episode_ended_utc,
                duration_seconds=duration_seconds,
                session_elapsed_seconds=episode_ended_perf - started,
                steps=environment.steps,
                optimal_steps=oracle_steps,
                efficiency=(oracle_steps / environment.steps) if environment.steps else 0.0,
                score=score,
                rolling_score=rolling_score,
                epsilon=epsilon,
                move_actions=move_actions,
                interaction_actions=interaction_actions,
                errors=errors,
                repeated_actions=repeats,
                blocked_moves=blocked_moves,
                found_keys=found_keys,
                opened_doors=opened_doors,
                empty_boxes=empty_boxes,
                imagination_decisions=episode_imagination_decisions,
                imagined_nodes=episode_imagined_nodes,
                maximum_imagination_depth=maximum_imagination_depth,
                mean_prediction_score=_mean(predictions),
                mean_holdout_before=_mean(holdout_before_values),
                mean_holdout_after=_mean(holdout_after_values),
                mean_holdout_gain=_mean(holdout_gains),
                positive_holdout_gain_total=sum(max(0.0, value) for value in holdout_gains),
                mean_intrinsic_value=_mean(intrinsic_values),
                intrinsic_value_total=sum(intrinsic_values),
                live_seconds=mode_seconds[TrainingMode.LIVE],
                fast_seconds=mode_seconds[TrainingMode.FAST],
                policy_entries=_policy_entry_count(agent),
                prophecy_exact_entries=_prophecy_entry_count(agent),
                holdout_size=_holdout_size(agent),
                action_counts=dict(action_counts),
                event_counts=dict(event_counts),
            )
            recorder.record_episode(record)
            if config.save_episode_checkpoints and (
                episode == 1
                or episode % config.checkpoint_interval == 0
            ):
                recorder.write_checkpoint(agent, episode=episode)

            recent_outcomes = outcomes[-config.rolling_window :]
            rolling_success = _mean([float(item) for item in recent_outcomes])
            current_mode = display.mode
            should_emit = (
                current_mode is TrainingMode.LIVE
                or episode == 1
                or episode == config.episodes
                or episode % config.fast_progress_interval == 0
                or interrupted
            )
            if should_emit and on_frame is not None:
                on_frame(
                    _frame(
                        environment,
                        episode=episode,
                        total_episodes=config.episodes,
                        action="",
                        event="success" if success else "interrupted",
                        episode_finished=True,
                        epsilon=epsilon,
                        rolling_success=rolling_success,
                        rolling_score=rolling_score,
                        total_successes=successes,
                        total_score=total_score,
                        episode_score=score,
                        elapsed_seconds=time.perf_counter() - started,
                        mode=current_mode,
                        used_imagination=latest_imagination,
                        imagined_nodes=latest_nodes,
                        prediction_score=latest_prediction,
                        holdout_gain=latest_gain,
                        intrinsic_value=latest_intrinsic,
                    )
                )
            if interrupted:
                break

        elapsed = time.perf_counter() - started
        final_outcomes = outcomes[-config.rolling_window :]
        final_scores = scores[-config.rolling_window :]
        statistics_payload = build_statistics(
            recorder.records,
            action_counts=recorder.action_counts,
            event_counts=recorder.event_counts,
        )
        final_summary = EscapeTrainingSummary(
            episodes=len(recorder.records),
            successes=successes,
            success_rate=(successes / len(recorder.records)) if recorder.records else 0.0,
            rolling_success=_mean([float(item) for item in final_outcomes]),
            total_score=total_score,
            mean_score=(total_score / successes) if successes else 0.0,
            rolling_score=_mean(final_scores),
            elapsed_seconds=elapsed,
            total_steps=sum(record.steps for record in recorder.records),
            policy_entries=_policy_entry_count(agent),
            oracle_steps=oracle_steps,
            imagination_decisions=imagination_decisions_total,
            imagined_nodes=imagined_nodes_total,
            stopped=stop.is_set(),
            output_dir=str(recorder.output_dir),
            episode_records=tuple(recorder.records),
            statistics=statistics_payload,
        )
        recorder.finalize(
            summary={
                "episodes": final_summary.episodes,
                "successes": final_summary.successes,
                "success_rate": final_summary.success_rate,
                "rolling_success": final_summary.rolling_success,
                "total_score": final_summary.total_score,
                "mean_score": final_summary.mean_score,
                "rolling_score": final_summary.rolling_score,
                "elapsed_seconds": final_summary.elapsed_seconds,
                "total_steps": final_summary.total_steps,
                "policy_entries": final_summary.policy_entries,
                "oracle_steps": final_summary.oracle_steps,
                "imagination_decisions": final_summary.imagination_decisions,
                "imagined_nodes": final_summary.imagined_nodes,
            },
            agent=agent,
            stopped=final_summary.stopped,
        )
        finalized = True
        if on_complete is not None:
            on_complete(final_summary)
        return final_summary
    except Exception:
        error_text = traceback.format_exc()
        if not finalized:
            recorder.finalize(
                summary={
                    "episodes": len(recorder.records),
                    "successes": successes,
                    "total_score": total_score,
                    "elapsed_seconds": time.perf_counter() - started,
                    "total_steps": sum(record.steps for record in recorder.records),
                },
                agent=agent,
                stopped=stop.is_set(),
                error=error_text,
            )
            finalized = True
        raise
    finally:
        if not finalized:
            recorder.close()
