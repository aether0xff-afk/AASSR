from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import random
import threading
import time
from typing import Callable, Mapping

from .escape_gridworld import (
    EscapeGridSpec,
    EscapeGridWorld,
    generate_escape_grid,
    oracle_plan,
)
from .types import Action


class TrainingMode(str, Enum):
    LIVE = "live"
    FAST = "fast"


@dataclass(frozen=True, slots=True)
class EscapeTrainingConfig:
    episodes: int = 2_000
    seed: int = 7
    color_count: int = 2
    distractor_boxes: int = 2
    max_steps: int = 180
    learning_rate: float = 0.25
    discount: float = 0.97
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 1_500
    live_step_delay: float = 0.06
    fast_progress_interval: int = 25
    rolling_window: int = 100

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if self.epsilon_decay_episodes <= 0:
            raise ValueError("epsilon_decay_episodes must be positive")
        if self.live_step_delay < 0.0:
            raise ValueError("live_step_delay must be non-negative")
        if self.fast_progress_interval <= 0:
            raise ValueError("fast_progress_interval must be positive")
        if self.rolling_window <= 0:
            raise ValueError("rolling_window must be positive")


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
    total_successes: int
    elapsed_seconds: float
    mode: TrainingMode


@dataclass(frozen=True, slots=True)
class EscapeTrainingSummary:
    episodes: int
    successes: int
    success_rate: float
    rolling_success: float
    elapsed_seconds: float
    q_entries: int
    oracle_steps: int
    stopped: bool


FrameCallback = Callable[[EscapeRenderFrame], None]
CompleteCallback = Callable[[EscapeTrainingSummary], None]


class ContextualQLearner:
    """Small transparent baseline for exercising the environment and GUI.

    It deliberately stores values by the complete current context. This is not
    presented as the final AASSR result; it is the observable baseline that the
    Prophecy/Imagination conditions can later be compared against.
    """

    def __init__(
        self,
        *,
        learning_rate: float,
        discount: float,
        seed: int,
    ) -> None:
        self.learning_rate = learning_rate
        self.discount = discount
        self.randomizer = random.Random(seed)
        self._values: dict[tuple[tuple[object, ...], str], float] = {}

    def value(self, state_key: tuple[object, ...], action: Action) -> float:
        return self._values.get((state_key, action.signature), 0.0)

    def choose(
        self,
        environment: EscapeGridWorld,
        *,
        epsilon: float,
    ) -> Action:
        actions = environment.snapshot().available_actions
        if not actions:
            raise RuntimeError("cannot choose an action from a terminal state")
        if self.randomizer.random() < epsilon:
            return self.randomizer.choice(actions)
        key = environment.state_key()
        scored = [(self.value(key, action), action) for action in actions]
        best = max(score for score, _ in scored)
        candidates = [action for score, action in scored if score == best]
        return self.randomizer.choice(candidates)

    def learn(
        self,
        before_key: tuple[object, ...],
        action: Action,
        reward: float,
        after: EscapeGridWorld,
    ) -> None:
        current = self.value(before_key, action)
        next_actions = after.snapshot().available_actions
        bootstrap = 0.0
        if next_actions:
            after_key = after.state_key()
            bootstrap = max(self.value(after_key, item) for item in next_actions)
        target = reward + self.discount * bootstrap
        self._values[(before_key, action.signature)] = current + (
            self.learning_rate * (target - current)
        )

    def reinforce_successful_trajectory(
        self,
        trajectory: tuple[tuple[tuple[object, ...], Action], ...],
    ) -> None:
        """Propagate the sparse terminal reward through the successful route.

        This is a backward Monte-Carlo update using only the final reward. It
        makes the learning animation practical without adding handcrafted key or
        door rewards.
        """

        value = 1.0
        for state_key, action in reversed(trajectory):
            current = self.value(state_key, action)
            self._values[(state_key, action.signature)] = current + (
                self.learning_rate * (value - current)
            )
            value *= self.discount

    @property
    def entry_count(self) -> int:
        return len(self._values)

    def snapshot(self) -> Mapping[tuple[tuple[object, ...], str], float]:
        return dict(self._values)


def epsilon_for_episode(config: EscapeTrainingConfig, episode: int) -> float:
    progress = min(1.0, max(0.0, episode / config.epsilon_decay_episodes))
    if config.epsilon_start == 0.0:
        return 0.0
    ratio = config.epsilon_end / config.epsilon_start
    if ratio <= 0.0:
        return config.epsilon_start * (1.0 - progress)
    return config.epsilon_start * math.exp(math.log(ratio) * progress)


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
    total_successes: int,
    elapsed_seconds: float,
    mode: TrainingMode,
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
        total_successes=total_successes,
        elapsed_seconds=elapsed_seconds,
        mode=mode,
    )


def train_escape_agent(
    config: EscapeTrainingConfig,
    *,
    mode: TrainingMode = TrainingMode.FAST,
    on_frame: FrameCallback | None = None,
    on_complete: CompleteCallback | None = None,
    stop_event: threading.Event | None = None,
) -> EscapeTrainingSummary:
    """Train one contextual policy while optionally streaming immutable frames.

    LIVE and FAST execute exactly the same learning loop. LIVE emits every step
    and sleeps briefly; FAST emits only periodic episode summaries and never
    sleeps, so selecting a display mode cannot change the learned policy.
    """

    stop = stop_event or threading.Event()
    spec = generate_escape_grid(
        config.seed,
        color_count=config.color_count,
        distractor_boxes=config.distractor_boxes,
        max_steps=config.max_steps,
    )
    oracle_steps = len(oracle_plan(spec))
    learner = ContextualQLearner(
        learning_rate=config.learning_rate,
        discount=config.discount,
        seed=config.seed,
    )
    started = time.perf_counter()
    outcomes: list[int] = []
    successes = 0
    completed_episodes = 0

    for episode_index in range(config.episodes):
        if stop.is_set():
            break
        episode = episode_index + 1
        epsilon = epsilon_for_episode(config, episode_index)
        environment = EscapeGridWorld(spec)
        trajectory: list[tuple[tuple[object, ...], Action]] = []
        event = "episode_started"

        if mode is TrainingMode.LIVE and on_frame is not None:
            on_frame(
                _frame(
                    environment,
                    episode=episode,
                    total_episodes=config.episodes,
                    action="",
                    event=event,
                    episode_finished=False,
                    epsilon=epsilon,
                    rolling_success=(sum(outcomes[-config.rolling_window :]) / len(outcomes[-config.rolling_window :])) if outcomes else 0.0,
                    total_successes=successes,
                    elapsed_seconds=time.perf_counter() - started,
                    mode=mode,
                )
            )

        while not environment.done and not stop.is_set():
            before_key = environment.state_key()
            action = learner.choose(environment, epsilon=epsilon)
            outcome = environment.step(action)
            trajectory.append((before_key, action))
            learner.learn(before_key, action, outcome.reward, environment)
            event = outcome.event

            if mode is TrainingMode.LIVE:
                rolling = (
                    sum(outcomes[-config.rolling_window :])
                    / len(outcomes[-config.rolling_window :])
                    if outcomes
                    else 0.0
                )
                if on_frame is not None:
                    on_frame(
                        _frame(
                            environment,
                            episode=episode,
                            total_episodes=config.episodes,
                            action=action.signature,
                            event=event,
                            episode_finished=environment.done,
                            epsilon=epsilon,
                            rolling_success=rolling,
                            total_successes=successes,
                            elapsed_seconds=time.perf_counter() - started,
                            mode=mode,
                        )
                    )
                if config.live_step_delay:
                    time.sleep(config.live_step_delay)

        success = int(environment.success)
        if success:
            learner.reinforce_successful_trajectory(tuple(trajectory))
            successes += 1
        outcomes.append(success)
        completed_episodes += 1
        window = outcomes[-config.rolling_window :]
        rolling = sum(window) / len(window)

        should_emit = (
            mode is TrainingMode.LIVE
            or episode == 1
            or episode == config.episodes
            or episode % config.fast_progress_interval == 0
            or stop.is_set()
        )
        if should_emit and on_frame is not None:
            on_frame(
                _frame(
                    environment,
                    episode=episode,
                    total_episodes=config.episodes,
                    action="",
                    event="success" if success else "timeout",
                    episode_finished=True,
                    epsilon=epsilon,
                    rolling_success=rolling,
                    total_successes=successes,
                    elapsed_seconds=time.perf_counter() - started,
                    mode=mode,
                )
            )

    elapsed = time.perf_counter() - started
    final_window = outcomes[-config.rolling_window :]
    summary = EscapeTrainingSummary(
        episodes=completed_episodes,
        successes=successes,
        success_rate=(successes / completed_episodes) if completed_episodes else 0.0,
        rolling_success=(sum(final_window) / len(final_window)) if final_window else 0.0,
        elapsed_seconds=elapsed,
        q_entries=learner.entry_count,
        oracle_steps=oracle_steps,
        stopped=stop.is_set(),
    )
    if on_complete is not None:
        on_complete(summary)
    return summary
