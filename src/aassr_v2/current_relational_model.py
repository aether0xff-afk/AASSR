from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import fmean
from types import SimpleNamespace
from typing import Any, Sequence
import random

from .current_generation import relational_action_key, relational_state_vector
from .current_relational_codec import (
    ACTION_SLOT_COUNT,
    REL_DESCRIPTOR_SIZE,
    TERMINAL_CLASSES,
    decode_relational_state,
    transition_target,
)
from .pentest_agent_main_test import ACTION_FEATURE_SIZE, AGENT_STATE_SIZE
from .skills import SKILL_VERB
from .types import Action, Prediction, StateSnapshot


@dataclass(frozen=True, slots=True)
class RelationalPrediction(Prediction):
    """Prediction with outcome mass separated from model reliability.

    ``Prediction.probability`` is kept as the historical reliability field because
    the existing planner/Skill contracts already interpret it that way. Stochastic
    outcome frequency therefore lives in ``outcome_probability`` and can never
    accidentally become a Critic/value bonus.
    """

    outcome_probability: float = 1.0

    def __post_init__(self) -> None:
        Prediction.__post_init__(self)
        if not 0.0 <= self.outcome_probability <= 1.0:
            raise ValueError("outcome_probability must be between 0 and 1")

    @property
    def reliability(self) -> float:
        return float(self.probability)


@dataclass(frozen=True, slots=True)
class RelationalProphecyConfig:
    hidden_units: int = 128
    ensemble_size: int = 3
    replay_capacity: int = 50_000
    batch_size: int = 64
    warmup_steps: int = 128
    learning_rate: float = 1e-3
    gradient_steps_per_observation: int = 1
    confidence_prior: float = 256.0
    variance_scale: float = 4.0
    gradient_clip: float = 5.0


class RelationalStochasticProphecy:
    """Relational world model with explicit observed multi-outcome support.

    Neural ensemble members provide generalization to unseen relational states.
    When the *same* relational state/action input has produced multiple different
    relational outcomes in real experience, those outcomes are retained as a
    categorical empirical distribution and emitted as distinct imagined worlds.
    Reliability and empirical outcome probability are explicitly separate.
    """

    name = "current-relational-stochastic-world-model-v1"

    def __init__(
        self,
        *,
        seed: int,
        device: str = "cpu",
        config: RelationalProphecyConfig | None = None,
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("RelationalStochasticProphecy requires torch") from exc
        self.torch = torch
        self.nn = nn
        self.config = config or RelationalProphecyConfig()
        self.device = torch.device(device)
        torch.manual_seed(int(seed))
        self.input_size = AGENT_STATE_SIZE + ACTION_FEATURE_SIZE
        self.output_size = REL_DESCRIPTOR_SIZE + ACTION_SLOT_COUNT + TERMINAL_CLASSES
        self.models = [
            nn.Sequential(
                nn.Linear(self.input_size, self.config.hidden_units),
                nn.ReLU(),
                nn.Linear(self.config.hidden_units, self.config.hidden_units),
                nn.ReLU(),
                nn.Linear(self.config.hidden_units, self.output_size),
            ).to(self.device)
            for _ in range(self.config.ensemble_size)
        ]
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
            for model in self.models
        ]
        self.replay: deque[
            tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], int]
        ] = deque(maxlen=self.config.replay_capacity)
        self._outcomes: dict[
            tuple[float, ...],
            dict[tuple[tuple[float, ...], tuple[int, ...], int], int],
        ] = {}
        self.observations = 0
        self.gradient_updates = 0
        self._losses: deque[float] = deque(maxlen=512)
        self._last_ensemble_variance = 1.0
        self.batch_prediction_calls = 0
        self.batch_prediction_rows = 0
        self.empirical_multioutcome_rows = 0

    @property
    def training_stats(self) -> Any:
        return SimpleNamespace(updates=int(self.gradient_updates))

    def _tensor(self, values: Any, *, dtype: Any | None = None) -> Any:
        return self.torch.as_tensor(
            values,
            dtype=dtype or self.torch.float32,
            device=self.device,
        )

    def _input(self, state: StateSnapshot, action: Action) -> tuple[float, ...]:
        values = relational_state_vector(state) + relational_action_key(state, action)
        if len(values) != self.input_size:
            raise ValueError("relational world-model input size drift")
        return values

    @staticmethod
    def _outcome_key(
        next_descriptor: Sequence[float],
        next_mask: Sequence[float],
        next_terminal: int,
    ) -> tuple[tuple[float, ...], tuple[int, ...], int]:
        return (
            tuple(round(float(value), 8) for value in next_descriptor),
            tuple(int(float(value) >= 0.5) for value in next_mask),
            int(next_terminal),
        )

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        if action.verb_name == SKILL_VERB:
            return
        model_input, next_descriptor, next_mask, next_terminal = transition_target(
            state, action, actual_next_state
        )
        if model_input != self._input(state, action):
            raise AssertionError("relational Prophecy train input contract drift")
        self.replay.append(
            (model_input, next_descriptor, next_mask, next_terminal)
        )
        bucket = self._outcomes.setdefault(model_input, {})
        outcome = self._outcome_key(next_descriptor, next_mask, next_terminal)
        bucket[outcome] = bucket.get(outcome, 0) + 1

        self.observations += 1
        if len(self.replay) < max(self.config.batch_size, self.config.warmup_steps):
            return
        for _ in range(self.config.gradient_steps_per_observation):
            self._train_step()

    def _train_step(self) -> None:
        for model_index, (model, optimizer) in enumerate(
            zip(self.models, self.optimizers, strict=True)
        ):
            local = random.Random(
                (self.observations + 1) * 1_000_003
                + (self.gradient_updates + 1) * 97
                + model_index
            )
            batch = [
                self.replay[local.randrange(len(self.replay))]
                for _ in range(self.config.batch_size)
            ]
            output = model(self._tensor([row[0] for row in batch]))
            descriptor_logits = output[:, :REL_DESCRIPTOR_SIZE]
            mask_start = REL_DESCRIPTOR_SIZE
            mask_end = mask_start + ACTION_SLOT_COUNT
            mask_logits = output[:, mask_start:mask_end]
            terminal_logits = output[:, mask_end:]

            descriptor_loss = self.nn.functional.smooth_l1_loss(
                self.torch.sigmoid(descriptor_logits),
                self._tensor([row[1] for row in batch]),
            )
            mask_loss = self.nn.functional.binary_cross_entropy_with_logits(
                mask_logits,
                self._tensor([row[2] for row in batch]),
            )
            terminal_loss = self.nn.functional.cross_entropy(
                terminal_logits,
                self._tensor(
                    [row[3] for row in batch],
                    dtype=self.torch.int64,
                ),
            )
            loss = descriptor_loss + 0.35 * mask_loss + 0.25 * terminal_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
            optimizer.step()
            self._losses.append(float(loss.detach().cpu().item()))
        self.gradient_updates += 1

    def _forward(
        self,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
    ) -> Any:
        inputs = self._tensor(
            [self._input(s, a) for s, a in zip(states, actions, strict=True)]
        )
        with self.torch.no_grad():
            return self.torch.stack([model(inputs) for model in self.models], dim=0)

    def _confidence(self, outputs: Any) -> Any:
        mask_start = REL_DESCRIPTOR_SIZE
        mask_end = mask_start + ACTION_SLOT_COUNT
        parts = (
            self.torch.sigmoid(outputs[:, :, :REL_DESCRIPTOR_SIZE]),
            self.torch.sigmoid(outputs[:, :, mask_start:mask_end]),
            self.torch.softmax(outputs[:, :, mask_end:], dim=-1),
        )
        variance = sum(
            part.var(dim=0, unbiased=False).mean(dim=1) for part in parts
        )
        self._last_ensemble_variance = float(
            variance.mean().detach().cpu().item()
        )
        sample_confidence = self.observations / (
            self.observations + self.config.confidence_prior
        )
        return self.torch.clamp(
            sample_confidence
            * self.torch.exp(-self.config.variance_scale * variance),
            min=0.05,
            max=0.995,
        )

    def _empirical_predictions(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        confidence: float,
        samples: int,
    ) -> tuple[Prediction, ...]:
        bucket = self._outcomes.get(self._input(state, action), {})
        if len(bucket) < 2:
            return ()
        selected = sorted(
            bucket.items(),
            key=lambda item: (-item[1], item[0]),
        )[: max(1, int(samples))]
        selected_total = sum(count for _, count in selected)
        predictions = []
        for index, ((next_descriptor, next_mask, next_terminal), count) in enumerate(selected):
            source = f"{self.name}:empirical-outcome-{index}"
            predictions.append(
                RelationalPrediction(
                    decode_relational_state(
                        next_descriptor,
                        next_mask,
                        scaffold=state,
                        predicted_terminal=next_terminal,
                        source=source,
                    ),
                    max(0.0, min(1.0, float(confidence))),
                    source=source,
                    outcome_probability=(
                        float(count) / float(selected_total)
                        if selected_total > 0
                        else 1.0 / len(selected)
                    ),
                )
            )
        self.empirical_multioutcome_rows += 1
        return tuple(predictions)

    def predict_batch(
        self,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
        *,
        samples: int,
    ) -> tuple[tuple[Prediction, ...], ...]:
        if len(states) != len(actions):
            raise ValueError("states/actions batch length mismatch")
        if samples <= 0:
            raise ValueError("samples must be positive")
        if not states:
            return ()
        self.batch_prediction_calls += 1
        self.batch_prediction_rows += len(states)
        if self.observations < self.config.warmup_steps:
            return tuple(
                (
                    RelationalPrediction(
                        state,
                        0.0,
                        source=f"{self.name}:unseen",
                        outcome_probability=1.0,
                    ),
                )
                for state in states
            )

        outputs = self._forward(states, actions)
        confidence = self._confidence(outputs).detach().cpu().tolist()
        mask_start = REL_DESCRIPTOR_SIZE
        mask_end = mask_start + ACTION_SLOT_COUNT
        descriptors = self.torch.sigmoid(
            outputs[:, :, :REL_DESCRIPTOR_SIZE]
        ).detach().cpu().tolist()
        masks = self.torch.sigmoid(
            outputs[:, :, mask_start:mask_end]
        ).detach().cpu().tolist()
        terminals = self.torch.softmax(
            outputs[:, :, mask_end:], dim=-1
        ).detach().cpu().tolist()

        member_count = min(int(samples), self.config.ensemble_size)
        rows: list[tuple[Prediction, ...]] = []
        for row_index, (state, action) in enumerate(
            zip(states, actions, strict=True)
        ):
            empirical = self._empirical_predictions(
                state,
                action,
                confidence=float(confidence[row_index]),
                samples=samples,
            )
            if empirical:
                rows.append(empirical)
                continue

            predictions = []
            for member in range(member_count):
                predicted_terminal = max(
                    range(TERMINAL_CLASSES),
                    key=lambda index: terminals[member][row_index][index],
                )
                source = f"{self.name}:member-{member}"
                predictions.append(
                    RelationalPrediction(
                        decode_relational_state(
                            descriptors[member][row_index],
                            masks[member][row_index],
                            scaffold=state,
                            predicted_terminal=predicted_terminal,
                            source=source,
                        ),
                        float(confidence[row_index]),
                        source=source,
                        outcome_probability=1.0 / max(1, member_count),
                    )
                )
            rows.append(tuple(predictions))
        return tuple(rows)

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        return self.predict_batch((state,), (action,), samples=samples)[0]

    def confidence_batch(
        self,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
    ) -> tuple[float, ...]:
        if len(states) != len(actions):
            raise ValueError("states/actions batch length mismatch")
        if not states:
            return ()
        if self.observations < self.config.warmup_steps:
            return (0.0,) * len(states)
        return tuple(
            float(value)
            for value in self._confidence(
                self._forward(states, actions)
            ).detach().cpu().tolist()
        )

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        return self.confidence_batch((state,), (action,))[0]

    def diagnostics(self) -> dict[str, int | float | str]:
        multimodal_inputs = sum(len(bucket) > 1 for bucket in self._outcomes.values())
        distinct_outcomes = sum(len(bucket) for bucket in self._outcomes.values())
        return {
            "name": self.name,
            "device": str(self.device),
            "observations": self.observations,
            "gradient_updates": self.gradient_updates,
            "replay_size": len(self.replay),
            "mean_training_loss": fmean(self._losses) if self._losses else 0.0,
            "last_ensemble_variance": self._last_ensemble_variance,
            "parameter_count": sum(
                p.numel() for model in self.models for p in model.parameters()
            ),
            "batch_prediction_calls": self.batch_prediction_calls,
            "batch_prediction_rows": self.batch_prediction_rows,
            "state_input_relational": 1,
            "action_input_relational": 1,
            "prediction_output": "relational-descriptor+legal-mask+terminal",
            "ensemble_outcomes_not_mean_collapsed": 1,
            "reliability_outcome_probability_separated": 1,
            "empirical_multimodal_input_keys": multimodal_inputs,
            "empirical_distinct_outcomes": distinct_outcomes,
            "empirical_multioutcome_prediction_rows": self.empirical_multioutcome_rows,
        }
