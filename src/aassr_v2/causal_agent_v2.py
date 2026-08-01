from __future__ import annotations

from typing import Callable

from .causal_prophecy import EmpiricalCausalProphecy
from .causal_representation import (
    CausalEncoder,
    ObservableTransition,
    RepresentedReturnAgent,
)
from .paper_v2_types import FullAgentCheckpoint, RawCausalObservation


class CausalAASSRAgent:
    """Policy and observable Prophecy with independently serialized encoders."""

    def __init__(
        self,
        encoder_factory: Callable[[], CausalEncoder],
        *,
        seed: int,
        return_target_mode: str = "monte_carlo",
    ) -> None:
        self.encoder_factory = encoder_factory
        self.seed = int(seed)
        self.policy = RepresentedReturnAgent(encoder_factory(), seed=seed)
        self.prophecy = EmpiricalCausalProphecy(
            encoder_factory(), return_target_mode=return_target_mode
        )

    def policy_action(
        self, observation: RawCausalObservation, *, epsilon: float
    ) -> str:
        return self.policy.select_action(observation, epsilon=epsilon)

    def observe_transition(self, transition: ObservableTransition) -> None:
        self.policy.observe_transition(transition)
        self.prophecy.observe_transition(transition)

    def finish_episode(self, success: bool) -> None:
        self.policy.finish_episode(success)
        self.prophecy.finish_episode(success)

    def export_full_checkpoint(self) -> FullAgentCheckpoint:
        policy = self.policy.export_full_checkpoint()
        return FullAgentCheckpoint(
            policy=policy.policy,
            prophecy=self.prophecy.export(),
            holdout=policy.holdout,
            rng=policy.rng,
            planner_cache=policy.planner_cache,
            counters={
                **policy.counters,
                "prophecy_updates": self.prophecy.total_updates,
            },
            replay_buffer=policy.replay_buffer,
            normalization_state=policy.normalization_state,
            calibration_buffer=policy.calibration_buffer,
            relational_representation=policy.relational_representation,
        )

    def import_full_checkpoint(self, checkpoint: FullAgentCheckpoint) -> None:
        self.policy.import_full_checkpoint(checkpoint)
        self.prophecy.restore(checkpoint.prophecy)
