from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import StrEnum

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
    target_success_low: float = 0.35
    target_success_high: float = 0.80


@dataclass(frozen=True)
class CurriculumTask:
    band: CurriculumBand
    world_kind: WorldKind
    seed: int
    difficulty: float


class LearningProgressScheduler:
    """Chooses tasks by measured learning progress instead of a scripted solution order."""

    BANDS = (
        CurriculumBand.FOUNDATION,
        CurriculumBand.CONTROL,
        CurriculumBand.COMPOSITION,
        CurriculumBand.ADVERSARIAL,
    )

    def __init__(self, *, seed: int | None = None, config: CurriculumConfig | None = None) -> None:
        self.random = random.Random(seed)
        self.config = config or CurriculumConfig()
        self._scores: dict[CurriculumBand, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.config.window_size * 2)
        )
        self._task_index = 0

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

    def observe(self, task: CurriculumTask, *, success: bool, steps: int, step_limit: int) -> None:
        efficiency = 0.0
        if success:
            efficiency = max(0.0, 1.0 - (steps / max(1, step_limit)))
        score = 0.8 * float(success) + 0.2 * efficiency
        self._scores[task.band].append(score)

    def mastery(self, band: CurriculumBand) -> float:
        values = list(self._scores[band])[-self.config.window_size :]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def learning_progress(self, band: CurriculumBand) -> float:
        values = list(self._scores[band])
        width = self.config.window_size
        if len(values) < width * 2:
            return 0.5
        previous = values[-2 * width : -width]
        current = values[-width:]
        return max(0.0, (sum(current) / width) - (sum(previous) / width))

    def difficulty(self, band: CurriculumBand) -> float:
        base = self.BANDS.index(band) / (len(self.BANDS) - 1)
        mastery = self.mastery(band)
        if mastery > self.config.target_success_high:
            return min(1.0, base + 0.15)
        if 0.0 < mastery < self.config.target_success_low:
            return max(0.0, base - 0.10)
        return base

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            band.value: {
                "mastery": self.mastery(band),
                "learning_progress": self.learning_progress(band),
                "samples": float(len(self._scores[band])),
            }
            for band in self.BANDS
        }

    def _choose_band(self) -> CurriculumBand:
        unseen = [band for band in self.BANDS if not self._scores[band]]
        if unseen:
            return unseen[self._task_index % len(unseen)]
        if self.random.random() < self.config.exploration_rate:
            return self.random.choice(self.BANDS)

        def priority(band: CurriculumBand) -> float:
            progress = self.learning_progress(band)
            mastery = self.mastery(band)
            learnable_zone = 1.0 - min(1.0, abs(mastery - 0.60) / 0.60)
            diversity_bonus = 1.0 / (1.0 + len(self._scores[band]))
            return 0.60 * progress + 0.30 * learnable_zone + 0.10 * diversity_bonus

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
