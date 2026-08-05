from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import elements
import embodied
import numpy as np

from .baseline_efficiency_portable import (
    CHOICE_ACTIONS,
    GRIDPUSH_OBSERVATION_SIZE,
    BenchmarkGridPushWorld,
    encode_gridpush_state,
    oracle_shortest_steps,
    solvable_map_seeds,
)


class BenchmarkEvaluationComplete(RuntimeError):
    """Internal control signal used after the requested evaluation episodes."""


class DreamerV3GridPushEnv(embodied.Env):
    """Official DreamerV3 Env adapter for the matched GridPush benchmark."""

    def __init__(
        self,
        mode: str,
        *,
        seed: int,
        train_map_count: int = 64,
        evaluation_map_count: int = 50,
        target_episodes: int = 0,
        episode_log: str | Path | None = None,
        worker_index: int = 0,
    ) -> None:
        if mode not in {"train", "seen", "unseen"}:
            raise ValueError(f"unknown DreamerV3 benchmark mode: {mode}")
        self.mode = mode
        self.seed = int(seed)
        self.worker_index = int(worker_index)
        self.target_episodes = int(target_episodes)
        self.training_maps = solvable_map_seeds(
            self.seed * 1_000_000,
            int(train_map_count),
        )
        self.unseen_maps = solvable_map_seeds(
            self.seed * 1_000_000 + 500_000,
            int(evaluation_map_count),
        )
        self.world: BenchmarkGridPushWorld | None = None
        self.done = True
        self.episode_index = 0
        self.completed_episodes = 0
        self._episode_log_path = Path(episode_log) if episode_log else None
        self._episode_log_handle = None
        if self._episode_log_path is not None:
            self._episode_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._episode_log_handle = self._episode_log_path.open(
                "a",
                encoding="utf-8",
                buffering=1,
            )

    @property
    def obs_space(self) -> dict[str, elements.Space]:
        return {
            "vector": elements.Space(
                np.float32,
                (GRIDPUSH_OBSERVATION_SIZE,),
            ),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
            "log/success": elements.Space(np.float32),
            "log/path_efficiency": elements.Space(np.float32),
            "log/episode_steps": elements.Space(np.float32),
        }

    @property
    def act_space(self) -> dict[str, elements.Space]:
        return {
            "reset": elements.Space(bool),
            "action": elements.Space(
                np.int32,
                (),
                0,
                len(CHOICE_ACTIONS),
            ),
        }

    def _map_pool(self) -> tuple[int, ...]:
        if self.mode in {"train", "seen"}:
            return self.training_maps
        return self.unseen_maps

    def _start_episode(self) -> dict[str, Any]:
        if (
            self.target_episodes > 0
            and self.completed_episodes >= self.target_episodes
        ):
            raise BenchmarkEvaluationComplete(
                f"completed {self.completed_episodes} evaluation episodes"
            )
        pool = self._map_pool()
        index = (
            self.worker_index + self.episode_index
        ) % len(pool)
        map_seed = pool[index]
        self.episode_index += 1
        self.world = BenchmarkGridPushWorld(map_seed)
        oracle = oracle_shortest_steps(map_seed)
        if oracle is None:
            raise RuntimeError(f"selected unsolvable benchmark map: {map_seed}")
        self.world.optimal_steps = oracle
        self.done = False
        return self._observation(
            reward=0.0,
            is_first=True,
            is_last=False,
            is_terminal=False,
        )

    def _log_episode(self) -> None:
        if self.world is None:
            return
        payload = {
            "algorithm": "dreamerv3_official_size1m",
            "mode": self.mode,
            "seed": self.seed,
            "worker_index": self.worker_index,
            "episode": self.completed_episodes,
            "map_seed": self.world.seed,
            "success": int(self.world.success),
            "reward": 1.0 if self.world.success else 0.0,
            "steps": self.world.steps,
            "optimal_steps": self.world.optimal_steps,
            "path_efficiency": (
                self.world.optimal_steps / self.world.steps
                if self.world.success and self.world.steps
                else 0.0
            ),
        }
        if self._episode_log_handle is not None:
            self._episode_log_handle.write(
                json.dumps(payload, sort_keys=True) + "\n"
            )
            self._episode_log_handle.flush()

    def _observation(
        self,
        *,
        reward: float,
        is_first: bool,
        is_last: bool,
        is_terminal: bool,
    ) -> dict[str, Any]:
        if self.world is None:
            vector = np.zeros(
                (GRIDPUSH_OBSERVATION_SIZE,),
                dtype=np.float32,
            )
            success = 0.0
            efficiency = 0.0
            steps = 0.0
        else:
            vector = np.asarray(
                encode_gridpush_state(self.world.snapshot()),
                dtype=np.float32,
            )
            success = float(self.world.success)
            efficiency = (
                self.world.optimal_steps / self.world.steps
                if self.world.success and self.world.steps
                else 0.0
            )
            steps = float(self.world.steps)
        return {
            "vector": vector,
            "reward": np.float32(reward),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
            "log/success": np.float32(success),
            "log/path_efficiency": np.float32(efficiency),
            "log/episode_steps": np.float32(steps),
        }

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        if bool(action.get("reset", False)) or self.done or self.world is None:
            return self._start_episode()

        index = int(np.asarray(action["action"]).item())
        if not 0 <= index < len(CHOICE_ACTIONS):
            raise ValueError(f"DreamerV3 action index out of range: {index}")
        outcome = self.world.step(CHOICE_ACTIONS[index])
        self.done = bool(self.world.success or self.world.failed)
        if self.done:
            self._log_episode()
            self.completed_episodes += 1
        return self._observation(
            reward=float(outcome.reward),
            is_first=False,
            is_last=self.done,
            is_terminal=self.done,
        )

    def close(self) -> None:
        if self._episode_log_handle is not None:
            self._episode_log_handle.close()
            self._episode_log_handle = None
