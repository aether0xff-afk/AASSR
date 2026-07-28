from __future__ import annotations

from collections import deque
from statistics import fmean


class AdaptiveDepthController:
    """Grow imagination depth slowly and shrink it quickly after real errors."""

    def __init__(
        self,
        *,
        minimum_depth: int = 1,
        maximum_depth: int = 6,
        window_size: int = 8,
        grow_threshold: float = 0.90,
        shrink_threshold: float = 0.65,
        grow_streak: int = 2,
    ) -> None:
        if not 1 <= minimum_depth <= maximum_depth:
            raise ValueError("depth bounds are invalid")
        if window_size <= 0 or grow_streak <= 0:
            raise ValueError("window_size and grow_streak must be positive")
        if shrink_threshold > grow_threshold:
            raise ValueError("shrink_threshold cannot exceed grow_threshold")

        self.minimum_depth = minimum_depth
        self.maximum_depth = maximum_depth
        self.window_size = window_size
        self.grow_threshold = grow_threshold
        self.shrink_threshold = shrink_threshold
        self.grow_streak = grow_streak

        self._depth = minimum_depth
        self._scores: deque[float] = deque(maxlen=window_size)
        self._good_windows = 0

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def mean_score(self) -> float:
        return fmean(self._scores) if self._scores else 0.0

    def observe(self, prediction_score: float) -> int:
        """Record one real next-state prediction score and return new depth."""

        self._scores.append(max(-1.0, min(1.0, prediction_score)))
        if len(self._scores) < self.window_size:
            return self._depth

        mean_score = self.mean_score
        if mean_score < self.shrink_threshold:
            self._depth = max(self.minimum_depth, self._depth - 1)
            self._good_windows = 0
            return self._depth

        if mean_score >= self.grow_threshold:
            self._good_windows += 1
            if self._good_windows >= self.grow_streak:
                self._depth = min(self.maximum_depth, self._depth + 1)
                self._good_windows = 0
        else:
            self._good_windows = 0

        return self._depth
