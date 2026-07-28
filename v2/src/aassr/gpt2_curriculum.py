from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from .worlds import WorldKind


class CurriculumBand(StrEnum):
    FOUNDATION = "foundation"
    CONTROL = "control"
    COMPOSITION = "composition"
    ADVERSARIAL = "adversarial"


@dataclass(frozen=True)
class CurriculumConfig:
    window_size: int = 12
    exploration_rate: float = 0.15
    target_mastery: float = 0.60
    progress_weight: float = 0.60
    learnable_zone_weight: float = 0.30
    diversity_weight: float = 0.10


@dataclass(frozen=True)
class CurriculumTask:
    band: CurriculumBand
    world_kind: WorldKind
    seed: int
    difficulty: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["band"] = self.band.value
        payload["world_kind"] = self.world_kind.value
        return payload


@dataclass(frozen=True)
class CurriculumOutcome:
    success: bool
    steps: int
    step_limit: int
    error_count: int = 0
    repeat_count: int = 0

    @property
    def score(self) -> float:
        efficiency = 0.0
        if self.success:
            efficiency = max(0.0, 1.0 - self.steps / max(1, self.step_limit))
        stability = 1.0 - min(
            1.0,
            (self.error_count + self.repeat_count) / max(1, self.steps),
        )
        return 0.70 * float(self.success) + 0.20 * efficiency + 0.10 * stability


class LearningProgressScheduler:
    """Selects procedural tasks by learning progress, not by stored solutions."""

    BANDS = (
        CurriculumBand.FOUNDATION,
        CurriculumBand.CONTROL,
        CurriculumBand.COMPOSITION,
        CurriculumBand.ADVERSARIAL,
    )

    def __init__(
        self,
        *,
        seed: int | None = None,
        config: CurriculumConfig | None = None,
    ) -> None:
        self.random = random.Random(seed)
        self.config = config or CurriculumConfig()
        self._scores: dict[CurriculumBand, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.config.window_size * 2)
        )
        self._task_index = 0
        self._history: list[dict[str, Any]] = []

    def next_task(self) -> CurriculumTask:
        band = self._choose_band()
        task_seed = self.random.randrange(1_000_000_000)
        task = CurriculumTask(
            band=band,
            world_kind=self._world_for_band(band, task_seed),
            seed=task_seed,
            difficulty=self.difficulty(band),
        )
        self._task_index += 1
        return task

    def observe(self, task: CurriculumTask, outcome: CurriculumOutcome) -> None:
        self._scores[task.band].append(outcome.score)
        self._history.append(
            {
                **task.to_dict(),
                "success": outcome.success,
                "steps": outcome.steps,
                "step_limit": outcome.step_limit,
                "error_count": outcome.error_count,
                "repeat_count": outcome.repeat_count,
                "score": outcome.score,
            }
        )

    def mastery(self, band: CurriculumBand) -> float:
        values = list(self._scores[band])[-self.config.window_size :]
        return sum(values) / len(values) if values else 0.0

    def learning_progress(self, band: CurriculumBand) -> float:
        values = list(self._scores[band])
        width = self.config.window_size
        if len(values) < width * 2:
            return 0.5
        previous = values[-2 * width : -width]
        current = values[-width:]
        return max(0.0, (sum(current) - sum(previous)) / width)

    def difficulty(self, band: CurriculumBand) -> float:
        structural = self.BANDS.index(band) / (len(self.BANDS) - 1)
        mastery_adjustment = 0.20 * (self.mastery(band) - self.config.target_mastery)
        return min(1.0, max(0.0, structural + mastery_adjustment))

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            band.value: {
                "mastery": self.mastery(band),
                "learning_progress": self.learning_progress(band),
                "samples": float(len(self._scores[band])),
                "difficulty": self.difficulty(band),
            }
            for band in self.BANDS
        }

    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def _choose_band(self) -> CurriculumBand:
        unseen = [band for band in self.BANDS if not self._scores[band]]
        if unseen:
            return unseen[0]
        if self.random.random() < self.config.exploration_rate:
            return self.random.choice(self.BANDS)

        def priority(band: CurriculumBand) -> float:
            progress = self.learning_progress(band)
            mastery = self.mastery(band)
            learnable_zone = 1.0 - min(
                1.0,
                abs(mastery - self.config.target_mastery) / max(0.01, self.config.target_mastery),
            )
            diversity_bonus = 1.0 / (1.0 + len(self._scores[band]))
            return (
                self.config.progress_weight * progress
                + self.config.learnable_zone_weight * learnable_zone
                + self.config.diversity_weight * diversity_bonus
            )

        return max(self.BANDS, key=priority)

    def _world_for_band(self, band: CurriculumBand, task_seed: int) -> WorldKind:
        if band == CurriculumBand.FOUNDATION:
            return WorldKind.RANDOM_FLAG
        if band == CurriculumBand.CONTROL:
            return WorldKind.RANDOM_WALL_FLAG
        if band == CurriculumBand.COMPOSITION:
            return WorldKind.RANDOM_KEY_DOOR
        return (
            WorldKind.V2_COMPLEX
            if task_seed % 2 == 0
            else WorldKind.LOCKED_BOTTLENECK
        )
