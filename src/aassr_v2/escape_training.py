from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading
import time
from typing import Callable

from .autonomous_agent import AutonomousAgentConfig, AutonomousLearningAgent
from .escape_gridworld import (
    EscapeGridSpec,
    EscapeGridWorld,
    generate_escape_grid,
    oracle_plan,
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
    imagination_minimum_coverage: float = 0.75
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
    policy_entries: int
    oracle_steps: int
    imagination_decisions: int
    imagined_nodes: int
    stopped: bool


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
    """Score a completed escape without introducing intermediate rewards.

    A shortest-path success receives ``1 + bonus_scale``. Longer successful
    routes approach the base score 1.0. This function is called only after the
    exit is reached, so an unfinished trajectory never receives this score.
    """

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
    local = getattr(agent.policy, "_local", {})
    return len(local)


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
) -> EscapeTrainingSummary:
    """Train Full AASSR while allowing live/fast switching in one session.

    The environment has no per-episode tick limit. An episode ends only at the
    exit or when the whole session is manually stopped. Switching display mode
    changes only frame emission and sleep; it never resets the world, agent,
    random generator, episode, or learning state.
    """

    stop = stop_event or threading.Event()
    display = runtime or TrainingRuntime(mode)
    spec = generate_escape_grid(
        config.seed,
        color_count=config.color_count,
        distractor_boxes=config.distractor_boxes,
    )
    oracle_steps = len(oracle_plan(spec))
    agent = _make_agent(config)
    started = time.perf_counter()
    outcomes: list[int] = []
    scores: list[float] = []
    successes = 0
    total_score = 0.0
    completed_episodes = 0
    imagination_decisions = 0
    imagined_nodes_total = 0

    for episode_index in range(config.episodes):
        if stop.is_set():
            break
        episode = episode_index + 1
        epsilon = agent.epsilon(episode_index)
        environment = EscapeGridWorld(spec)
        latest_imagination = False
        latest_nodes = 0
        latest_prediction = 0.0
        latest_gain = 0.0
        latest_intrinsic = 0.0

        current_mode = display.mode
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
            state = environment.snapshot()
            decision = agent.select_action(
                state,
                episode=episode_index,
                explore=True,
            )
            outcome = environment.step(decision.action)
            metrics = agent.observe(state, decision.action, outcome)
            latest_imagination = decision.used_imagination
            latest_nodes = decision.imagined_nodes
            latest_prediction = metrics.prediction_score
            latest_gain = metrics.holdout_gain
            latest_intrinsic = metrics.intrinsic_value
            if decision.used_imagination:
                imagination_decisions += 1
                imagined_nodes_total += decision.imagined_nodes

            current_mode = display.mode
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
                    break

        if interrupted:
            agent.discard_episode()
            break

        episode_score = success_score_multiplier(
            oracle_steps,
            environment.steps,
            bonus_scale=config.efficiency_bonus_scale,
        )
        agent.finish_episode(final_return=episode_score)
        successes += 1
        total_score += episode_score
        outcomes.append(1)
        scores.append(episode_score)
        completed_episodes += 1
        recent_outcomes = outcomes[-config.rolling_window :]
        recent_scores = scores[-config.rolling_window :]
        rolling_success = _mean([float(item) for item in recent_outcomes])
        rolling_score = _mean(recent_scores)

        current_mode = display.mode
        should_emit = (
            current_mode is TrainingMode.LIVE
            or episode == 1
            or episode == config.episodes
            or episode % config.fast_progress_interval == 0
        )
        if should_emit and on_frame is not None:
            on_frame(
                _frame(
                    environment,
                    episode=episode,
                    total_episodes=config.episodes,
                    action="",
                    event="success",
                    episode_finished=True,
                    epsilon=epsilon,
                    rolling_success=rolling_success,
                    rolling_score=rolling_score,
                    total_successes=successes,
                    total_score=total_score,
                    episode_score=episode_score,
                    elapsed_seconds=time.perf_counter() - started,
                    mode=current_mode,
                    used_imagination=latest_imagination,
                    imagined_nodes=latest_nodes,
                    prediction_score=latest_prediction,
                    holdout_gain=latest_gain,
                    intrinsic_value=latest_intrinsic,
                )
            )

    elapsed = time.perf_counter() - started
    final_outcomes = outcomes[-config.rolling_window :]
    final_scores = scores[-config.rolling_window :]
    summary = EscapeTrainingSummary(
        episodes=completed_episodes,
        successes=successes,
        success_rate=(successes / completed_episodes) if completed_episodes else 0.0,
        rolling_success=_mean([float(item) for item in final_outcomes]),
        total_score=total_score,
        mean_score=(total_score / completed_episodes) if completed_episodes else 0.0,
        rolling_score=_mean(final_scores),
        elapsed_seconds=elapsed,
        policy_entries=_policy_entry_count(agent),
        oracle_steps=oracle_steps,
        imagination_decisions=imagination_decisions,
        imagined_nodes=imagined_nodes_total,
        stopped=stop.is_set(),
    )
    if on_complete is not None:
        on_complete(summary)
    return summary
