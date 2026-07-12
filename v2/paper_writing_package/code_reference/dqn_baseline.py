from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .experiment import _run_seed_jobs, _world_seed
from .gridworld import ActionCandidate, ActionName, DMPConfig, GridWorldDMP
from .knowledge import KK
from .mdp_baseline import write_mdp_outputs
from .metrics import EpisodeMetric, StepMetric, episode_metric, step_metric, summary_metric
from .traditional_baselines import used_count
from .worlds import WorldKind, make_world


ACTION_ORDER = tuple(ActionName)
STRATEGY_ORDER = ("nearest", "least_tried", "high_uncertainty", "normal", "prophecy_best")
CELL_KKS = (
    KK.KNOWN_CELL,
    KK.VISITED_CELL,
    KK.UNKNOWN_NEIGHBOR,
    KK.FRONTIER_CELL,
    KK.WALL_CELL,
    KK.HINT_CELL,
    KK.KEY_CELL,
    KK.DOOR_CELL,
    KK.FLAG_CELL,
    KK.CURRENT_POS,
)


@dataclass(frozen=True)
class Transition:
    features: np.ndarray
    reward: float
    next_features: tuple[np.ndarray, ...]
    done: bool


class NumpyDQN:
    """Small MLP Q approximator used as a no-extra-dependency DQN baseline."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 32,
        learning_rate: float = 0.003,
        gamma: float = 0.9,
        replay_size: int = 2000,
        batch_size: int = 8,
        seed: int = 0,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        scale = 1.0 / max(1, input_dim) ** 0.5
        self.w1 = self.rng.normal(0.0, scale, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = self.rng.normal(0.0, 1.0 / hidden_dim ** 0.5, size=(hidden_dim, 1))
        self.b2 = np.zeros(1)
        self.replay: list[Transition] = []

    def predict(self, features: np.ndarray) -> float:
        hidden = np.maximum(features @ self.w1 + self.b1, 0.0)
        return float((hidden @ self.w2 + self.b2)[0])

    def remember(self, transition: Transition) -> None:
        self.replay.append(transition)
        if len(self.replay) > self.replay_size:
            self.replay.pop(0)

    def train_step(self) -> None:
        if len(self.replay) < self.batch_size:
            return
        indexes = self.rng.choice(len(self.replay), size=self.batch_size, replace=False)
        batch = [self.replay[int(index)] for index in indexes]
        x = np.stack([item.features for item in batch])
        targets = np.array([self._target(item) for item in batch], dtype=float).reshape(-1, 1)

        z1 = x @ self.w1 + self.b1
        hidden = np.maximum(z1, 0.0)
        pred = hidden @ self.w2 + self.b2
        grad = (pred - targets) * (2.0 / len(batch))

        grad_w2 = hidden.T @ grad
        grad_b2 = grad.sum(axis=0)
        grad_hidden = grad @ self.w2.T
        grad_z1 = grad_hidden * (z1 > 0.0)
        grad_w1 = x.T @ grad_z1
        grad_b1 = grad_z1.sum(axis=0)

        self.w2 -= self.learning_rate * grad_w2
        self.b2 -= self.learning_rate * grad_b2
        self.w1 -= self.learning_rate * grad_w1
        self.b1 -= self.learning_rate * grad_b1

    def _target(self, transition: Transition) -> float:
        if transition.done or not transition.next_features:
            return transition.reward
        next_best = max(self.predict(features) for features in transition.next_features)
        return transition.reward + self.gamma * next_best


def run_dqn_partial_baseline(
    *,
    episodes: int,
    seeds: int,
    step_limit: int,
    world: str | WorldKind = WorldKind.RANDOM_KEY_DOOR,
    output_dir: str | Path | None = None,
    condition: str = "DQN_PARTIAL",
    epsilon: float = 0.2,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.05,
    workers: int = 1,
    progress: bool = False,
    progress_label: str | None = None,
) -> tuple[list[StepMetric], list[EpisodeMetric], list[Any]]:
    world = WorldKind(world)

    args = [
        (seed, episodes, step_limit, world.value, condition, epsilon, epsilon_decay, min_epsilon)
        for seed in range(seeds)
    ]
    chunks = _run_seed_jobs(
        _run_dqn_seed,
        args,
        workers=workers,
        progress=progress,
        label=progress_label or condition,
    )

    all_steps = [row for steps, _ in chunks for row in steps]
    all_episodes = [row for _, episode_rows in chunks for row in episode_rows]

    summaries = [summary_metric(condition, all_episodes)]
    if output_dir is not None:
        write_mdp_outputs(output_dir, all_steps, all_episodes, summaries)
    return all_steps, all_episodes, summaries


def _run_dqn_seed(
    args: tuple[int, int, int, str, str, float, float, float],
) -> tuple[list[StepMetric], list[EpisodeMetric]]:
    seed, episodes, step_limit, world_value, condition, epsilon, epsilon_decay, min_epsilon = args
    world = WorldKind(world_value)
    rng = random.Random(seed)
    model: NumpyDQN | None = None
    current_epsilon = epsilon
    seed_steps: list[StepMetric] = []
    seed_episodes: list[EpisodeMetric] = []

    for episode in range(episodes):
        dmp = GridWorldDMP(
            make_world(world, seed=_world_seed(seed, episode)),
            config=DMPConfig(),
            step_limit=step_limit,
        )
        step_rows: list[StepMetric] = []
        while not dmp.done:
            candidates = dmp.generate_candidates()
            if not candidates:
                break
            state_features = partial_state_features(dmp, candidate_count=len(candidates))
            features = [
                state_action_features_from_state(dmp, candidate, state_features)
                for candidate in candidates
            ]
            if model is None:
                model = NumpyDQN(len(features[0]), seed=seed)
            candidate, chosen_features = choose_dqn_candidate(
                candidates,
                features,
                model,
                rng,
                epsilon=current_epsilon,
            )
            result = dmp.execute(candidate)
            next_candidates = dmp.generate_candidates()
            next_state_features = partial_state_features(dmp, candidate_count=len(next_candidates))
            next_features = tuple(
                state_action_features_from_state(dmp, item, next_state_features)
                for item in next_candidates
            )
            model.remember(
                Transition(
                    features=chosen_features,
                    reward=result.total_reward,
                    next_features=next_features,
                    done=result.done,
                )
            )
            if result.step % 4 == 0 or result.done:
                model.train_step()
            row = step_metric(
                condition=condition,
                seed=seed,
                episode=episode,
                result=result,
            )
            step_rows.append(row)
            seed_steps.append(row)
        seed_episodes.append(
            episode_metric(
                condition=condition,
                seed=seed,
                episode=episode,
                steps=step_rows,
                knowledge_reuse_count=int(dmp.metrics()["knowledge_reuse_count"]),
                step_limit=step_limit,
            )
        )
        current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)
    return seed_steps, seed_episodes


def choose_dqn_candidate(
    candidates: list[ActionCandidate],
    features: list[np.ndarray],
    model: NumpyDQN,
    rng: random.Random,
    *,
    epsilon: float,
) -> tuple[ActionCandidate, np.ndarray]:
    if rng.random() < epsilon:
        index = rng.randrange(len(candidates))
        return candidates[index], features[index]
    scores = [model.predict(feature) + rng.random() * 1e-9 for feature in features]
    index = max(range(len(candidates)), key=lambda item: scores[item])
    return candidates[index], features[index]


def partial_state_action_features(dmp: GridWorldDMP, candidate: ActionCandidate) -> np.ndarray:
    candidates = dmp.generate_candidates()
    return state_action_features_from_state(
        dmp,
        candidate,
        partial_state_features(dmp, candidate_count=len(candidates)),
    )


def partial_state_features(dmp: GridWorldDMP, *, candidate_count: int) -> np.ndarray:
    width = dmp.world.width
    height = dmp.world.height
    values: list[float] = []
    for kk in CELL_KKS:
        mask = [0.0] * (width * height)
        for kv in dmp.store.values(kk, include_inactive=True):
            if isinstance(kv.value, tuple) and len(kv.value) == 2:
                x, y = kv.value
                if 0 <= x < width and 0 <= y < height:
                    mask[y * width + x] = 1.0
        values.extend(mask)

    values.extend(
        [
            dmp.position[0] / max(1, width - 1),
            dmp.position[1] / max(1, height - 1),
            dmp.step_index / max(1, dmp.step_limit),
            1.0 if dmp.store.has_active(KK.KEY_OBJECT) else 0.0,
            len(dmp.world.opened_doors) / 4.0,
            candidate_count / 20.0,
        ]
    )
    return np.array(values, dtype=float)


def state_action_features_from_state(
    dmp: GridWorldDMP,
    candidate: ActionCandidate,
    state_features: np.ndarray,
) -> np.ndarray:
    width = dmp.world.width
    height = dmp.world.height
    values = list(state_features)
    values.extend(_one_hot(ACTION_ORDER, candidate.name))
    values.extend(_one_hot(STRATEGY_ORDER, candidate.strategy))
    for kk in KK:
        values.append(1.0 if kk in candidate.bindings and kk != KK.CURRENT_POS else 0.0)
    values.extend(
        [
            dmp.distance_for(candidate) / max(1, width + height),
            used_count(dmp, candidate) / 10.0,
            len(candidate.bindings) / 4.0,
        ]
    )
    return np.array(values, dtype=float)


def _one_hot(options: tuple[Any, ...], value: Any) -> list[float]:
    return [1.0 if item == value else 0.0 for item in options]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a partial-observation numpy DQN baseline for GridWorld.")
    parser.add_argument("--world", choices=[world.value for world in WorldKind], default=WorldKind.RANDOM_KEY_DOOR.value)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--step-limit", type=int, default=80)
    parser.add_argument("--output-dir", default="runs/dqn/DQN_PARTIAL")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, _, summaries = run_dqn_partial_baseline(
        episodes=args.episodes,
        seeds=args.seeds,
        step_limit=args.step_limit,
        world=args.world,
        output_dir=args.output_dir,
        workers=args.workers,
        progress=True,
    )
    summary = summaries[0]
    print(
        "condition={condition} episodes={episodes} seeds={seeds} "
        "success_rate={success_rate:.3f} steps_to_flag_mean={steps_to_flag_mean:.3f}".format(
            **summary.to_dict()
        )
    )
    print(f"wrote {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
