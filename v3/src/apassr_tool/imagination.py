from __future__ import annotations

from dataclasses import dataclass
import math

from .actions import ActionCandidate
from .prophecy import ProphecyPrediction, TableProphecyModel


ExperienceMemory = TableProphecyModel
CandidatePrediction = ProphecyPrediction


@dataclass
class ImaginationCycle:
    prophecy_model: TableProphecyModel
    reward_weight: float = 0.02
    knowledge_weight: float = 0.02
    solved_weight: float = 0.4
    error_weight: float = 0.5
    max_multiplier: float = 3.0
    min_multiplier: float = 0.25

    def score_multiplier(self, candidate: ActionCandidate) -> tuple[float, ProphecyPrediction]:
        prediction = self.prophecy_model.predict(candidate)
        if prediction.support == 0:
            return 1.0, prediction
        raw = (
            self.reward_weight * prediction.expected_reward
            + self.knowledge_weight * prediction.expected_knowledge
            + self.solved_weight * prediction.solved_rate
            - self.error_weight * prediction.error_rate
        )
        confidence = math.sqrt(prediction.support) / (1.0 + math.sqrt(prediction.support))
        multiplier = math.exp(raw * confidence)
        multiplier = max(self.min_multiplier, min(self.max_multiplier, multiplier))
        return multiplier, prediction
