from __future__ import annotations

from dataclasses import dataclass

from .knowledge import KnowledgeDelta


@dataclass(frozen=True)
class RewardBreakdown:
    external_reward: float
    intrinsic_reward: float
    total_reward: float


class RewardModule:
    """Computes sparse external reward plus knowledge-change intrinsic signal."""

    def __init__(
        self,
        *,
        flag_reward: float = 1.0,
        knowledge_gain_weight: float = 0.1,
        error_penalty: float = -0.2,
        repeat_penalty: float = -0.05,
    ) -> None:
        self.flag_reward = flag_reward
        self.knowledge_gain_weight = knowledge_gain_weight
        self.error_penalty = error_penalty
        self.repeat_penalty = repeat_penalty

    def compute(
        self,
        *,
        delta_k: KnowledgeDelta,
        error: bool,
        repeated: bool,
        flag_found: bool,
    ) -> RewardBreakdown:
        external = self.flag_reward if flag_found else 0.0
        intrinsic = self.knowledge_gain_weight * delta_k.semantic_information_gain()
        if error:
            intrinsic += self.error_penalty
        if repeated:
            intrinsic += self.repeat_penalty
        return RewardBreakdown(
            external_reward=external,
            intrinsic_reward=intrinsic,
            total_reward=external + intrinsic,
        )
