from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass
from typing import Sequence

from .autonomous_agent_core import AutonomousAgentConfig, AutonomousLearningAgent
from .baseline_efficiency_benchmark import (
    CHOICE_ACTIONS,
    BenchmarkGridPushWorld,
    encode_gridpush_state,
)
from .bottleneck_sota_diagnostic import (
    DQNPolicyAdapter,
    HybridDQNImaginationAgent,
    _current_scorer,
)
from .neural_delta_prophecy import (
    NeuralDeltaConfig,
    NeuralDeltaProphecy,
    StateCodec,
)
from .types import StateSnapshot


@dataclass(frozen=True, slots=True)
class BenchmarkGridPushCodec(StateCodec):
    """Representation codec only; contains no transition or reward rules."""

    @property
    def dimension(self) -> int:
        return 25

    def encode(self, state: StateSnapshot) -> tuple[float, ...]:
        return encode_gridpush_state(state)

    @staticmethod
    def _bounded(value: float) -> float:
        return min(1.0, max(0.0, float(value)))

    def decode(
        self,
        encoded: Sequence[float],
        *,
        scaffold: StateSnapshot,
        terminal_class: int,
        source: str,
    ) -> StateSnapshot:
        if len(encoded) != self.dimension:
            raise ValueError("benchmark neural state must have 25 values")
        values = [self._bounded(value) for value in encoded]
        # The public observation schema has normalized coordinates, a 0..6
        # phase, three binary flags, and nine binary used-cell indicators.
        coordinate_indexes = range(12)
        for index in coordinate_indexes:
            values[index] = round(values[index] * 2.0) / 2.0
        phase = min(6, max(0, int(round(values[12] * 6.0))))
        values[12] = phase / 6.0
        for index in range(13, 16):
            values[index] = float(values[index] >= 0.5)
        for index in range(16, 25):
            values[index] = float(values[index] >= 0.5)

        facts = {f"phase:{phase}"}
        for index, occupied in enumerate(values[16:25]):
            if occupied >= 0.5:
                x = index % BenchmarkGridPushWorld.grid_size
                y = index // BenchmarkGridPushWorld.grid_size
                facts.add(f"used:{x}:{y}")
        if values[13] >= 0.5:
            facts.add("bridge_built")
        if values[14] >= 0.5:
            facts.add("key_held")
        if values[15] >= 0.5:
            facts.add("door_open")

        if terminal_class == 1:
            facts.add("success")
        elif terminal_class == 2:
            facts.add("failed")
        available_actions = () if terminal_class else CHOICE_ACTIONS
        metadata = dict(scaffold.metadata)
        metadata.update(
            {
                "imagined_neural_delta": True,
                "imagined_neural_delta_source": source,
            }
        )
        return StateSnapshot(
            vector=tuple(float(value) for value in values[:16]),
            facts=frozenset(facts),
            available_actions=available_actions,
            goal_progress=1.0 if terminal_class == 1 else 0.0,
            metadata=metadata,
        )


class NeuralProphecyHybridAgent(HybridDQNImaginationAgent):
    """Identity-preserving DQN Policy + learned Prophecy + Imagination."""

    def __init__(
        self,
        seed: int,
        *,
        train_episodes: int,
        adaptive: bool = False,
    ) -> None:
        prophecy = NeuralDeltaProphecy(
            BenchmarkGridPushCodec(),
            config=NeuralDeltaConfig(
                hidden_units=128,
                ensemble_size=3,
                replay_capacity=50_000,
                batch_size=64,
                warmup_steps=128,
                learning_rate=1e-3,
                gradient_steps_per_observation=1,
                confidence_prior=256.0,
            ),
            seed=seed ^ 0x4E455552,
        )
        super().__init__(
            seed,
            train_episodes=train_episodes,
            prophecy=prophecy,
            scorer=_current_scorer(),
            depth=5,
            branching_factor=2,
            beam_width=16,
            outcome_samples=1,
            imagination_interval=1,
            use_effect_composition=False,
            name=(
                "hybrid_neural_prophecy_adaptive"
                if adaptive
                else "hybrid_neural_prophecy"
            ),
        )
        # Neural confidence already represents data support and ensemble
        # disagreement. Requiring minimum coverage prevents early hallucinated
        # rollouts without disabling imagination after the model becomes useful.
        self.agent.config.imagination_minimum_coverage = 0.30
        self.adaptive = bool(adaptive)
        self._original_depth = self.agent.planner.config.maximum_depth

    def select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        training: bool,
    ):
        if not self.adaptive:
            return super().select_action(
                state,
                episode=episode,
                training=training,
            )

        # Planning remains multi-step, but depth follows learned model
        # confidence. This removes fixed worst-case computation rather than the
        # AASSR requirement to compare counterfactual futures.
        coverage = self.agent.model_coverage(state)
        if coverage < 0.30:
            depth = 1
        elif coverage < 0.55:
            depth = 2
        elif coverage < 0.75:
            depth = 3
        else:
            depth = self._original_depth
        original = self.agent.planner.config
        from dataclasses import replace

        self.agent.planner.config = replace(original, maximum_depth=depth)
        try:
            return super().select_action(
                state,
                episode=episode,
                training=training,
            )
        finally:
            self.agent.planner.config = original

    def model_stats(self) -> dict[str, int | float]:
        stats = dict(super().model_stats())
        prophecy = self.agent.base_prophecy
        neural_stats = prophecy.stats()
        stats.update(
            {
                "prophecy_observations": neural_stats.observations,
                "prophecy_gradient_updates": neural_stats.gradient_updates,
                "prophecy_replay_size": neural_stats.replay_size,
                "prophecy_mean_training_loss": neural_stats.mean_training_loss,
                "prophecy_last_ensemble_variance": (
                    neural_stats.last_ensemble_variance
                ),
            }
        )
        stats["model_units"] = int(stats.get("model_units", 0)) + int(
            neural_stats.parameter_count
        )
        handle = io.BytesIO()
        self.dqn.torch.save(
            [model.state_dict() for model in prophecy.models],
            handle,
        )
        stats["model_bytes"] = int(stats.get("model_bytes", 0)) + len(
            handle.getvalue()
        )
        return stats
