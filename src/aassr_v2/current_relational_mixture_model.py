from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import exp
from statistics import fmean
from types import SimpleNamespace
from typing import Any, Sequence
import random

from .current_generation import relational_action_key
from .current_relational_codec import (
    ACTION_SLOT_COUNT,
    legal_action_mask,
    terminal_class,
)
from .current_relational_decode_v2 import decode_relational_state_v2
from .current_relational_model import (
    RelationalPrediction,
    RelationalProphecyConfig,
    RelationalStochasticProphecy,
)
from .current_relational_state import (
    REL_DESCRIPTOR_SIZE,
    relational_state_descriptor_v2,
    relational_state_vector_v2,
)
from .pentest_agent_main_test import ACTION_FEATURE_SIZE, AGENT_STATE_SIZE
from .skills import SKILL_VERB
from .types import Action, Prediction, StateSnapshot


@dataclass(frozen=True, slots=True)
class RelationalMixtureProphecyConfig(RelationalProphecyConfig):
    mixture_components: int = 3
    mixture_entropy_weight: float = 0.01
    mode_merge_distance: float = 0.035


class ConditionalMixtureRelationalProphecy(RelationalStochasticProphecy):
    """Conditional multi-head relational world model.

    One structural state/action can legitimately have several next outcomes under
    partial observability. Each ensemble member therefore predicts a categorical
    mixture of K semantic futures instead of one deterministic regression target.
    Exact repeated inputs still use empirical outcome frequencies when available;
    the learned mixture generalizes those modes to novel but related states.

    Model reliability is epistemic set disagreement between ensemble members.
    Aleatoric outcome mass is represented by the learned mixture weights and is
    never added to branch value.
    """

    name = "current-relational-conditional-mixture-world-model-v2"

    def __init__(
        self,
        *,
        seed: int,
        device: str = "cpu",
        config: RelationalMixtureProphecyConfig | None = None,
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "ConditionalMixtureRelationalProphecy requires torch"
            ) from exc
        self.torch = torch
        self.nn = nn
        self.config = config or RelationalMixtureProphecyConfig()
        if self.config.mixture_components <= 1:
            raise ValueError("mixture_components must be greater than one")
        self.device = torch.device(device)
        torch.manual_seed(int(seed))

        self.input_size = AGENT_STATE_SIZE + ACTION_FEATURE_SIZE
        self.component_size = (
            REL_DESCRIPTOR_SIZE + ACTION_SLOT_COUNT + 3
        )
        self.output_size = (
            self.config.mixture_components * self.component_size
            + self.config.mixture_components
        )
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
        self.mixture_prediction_rows = 0

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
        values = relational_state_vector_v2(state) + relational_action_key(state, action)
        if len(values) != self.input_size:
            raise ValueError("relational mixture input size drift")
        return values

    @staticmethod
    def _target(
        state: StateSnapshot,
        action: Action,
        next_state: StateSnapshot,
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], int]:
        return (
            relational_state_vector_v2(state) + relational_action_key(state, action),
            relational_state_descriptor_v2(next_state),
            legal_action_mask(next_state),
            terminal_class(next_state),
        )

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
        model_input, next_descriptor, next_mask, next_terminal = self._target(
            state,
            action,
            actual_next_state,
        )
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

    def _split_output(self, output: Any) -> tuple[Any, Any, Any, Any]:
        k = self.config.mixture_components
        component_end = k * self.component_size
        components = output[:, :component_end].reshape(
            output.shape[0],
            k,
            self.component_size,
        )
        mixture_logits = output[:, component_end:]
        descriptor_end = REL_DESCRIPTOR_SIZE
        mask_end = descriptor_end + ACTION_SLOT_COUNT
        descriptor_logits = components[:, :, :descriptor_end]
        mask_logits = components[:, :, descriptor_end:mask_end]
        terminal_logits = components[:, :, mask_end:]
        return descriptor_logits, mask_logits, terminal_logits, mixture_logits

    def _train_step(self) -> None:
        torch = self.torch
        nn = self.nn
        k = self.config.mixture_components
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
            descriptor_logits, mask_logits, terminal_logits, mixture_logits = (
                self._split_output(output)
            )

            descriptor_target = self._tensor([row[1] for row in batch]).unsqueeze(1)
            descriptor_target = descriptor_target.expand(-1, k, -1)
            descriptor_loss = nn.functional.smooth_l1_loss(
                torch.sigmoid(descriptor_logits),
                descriptor_target,
                reduction="none",
            ).mean(dim=2)

            mask_target = self._tensor([row[2] for row in batch]).unsqueeze(1)
            mask_target = mask_target.expand(-1, k, -1)
            mask_loss = nn.functional.binary_cross_entropy_with_logits(
                mask_logits,
                mask_target,
                reduction="none",
            ).mean(dim=2)

            terminal_target = self._tensor(
                [row[3] for row in batch],
                dtype=torch.int64,
            )
            terminal_target = terminal_target.unsqueeze(1).expand(-1, k)
            terminal_loss = nn.functional.cross_entropy(
                terminal_logits.reshape(-1, 3),
                terminal_target.reshape(-1),
                reduction="none",
            ).reshape(-1, k)

            reconstruction = (
                descriptor_loss
                + 0.35 * mask_loss
                + 0.25 * terminal_loss
            )
            log_mixture = torch.log_softmax(mixture_logits, dim=1)
            # Soft mixture likelihood: different heads can own different targets
            # for the exact same input instead of being forced toward their mean.
            nll = -torch.logsumexp(log_mixture - reconstruction, dim=1).mean()
            mixture = torch.softmax(mixture_logits, dim=1)
            entropy = -(mixture * log_mixture).sum(dim=1).mean()
            loss = nll - self.config.mixture_entropy_weight * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
            optimizer.step()
            self._losses.append(float(loss.detach().cpu().item()))
        self.gradient_updates += 1

    def _forward(self, states: Sequence[StateSnapshot], actions: Sequence[Action]) -> Any:
        inputs = self._tensor(
            [self._input(state, action) for state, action in zip(states, actions, strict=True)]
        )
        with self.torch.no_grad():
            return self.torch.stack([model(inputs) for model in self.models], dim=0)

    def _decoded_outputs(self, outputs: Any) -> tuple[Any, Any, Any, Any]:
        # outputs: [ensemble, batch, flat]
        descriptor_rows = []
        mask_rows = []
        terminal_rows = []
        mixture_rows = []
        for member in range(outputs.shape[0]):
            desc, mask, terminal, mixture = self._split_output(outputs[member])
            descriptor_rows.append(self.torch.sigmoid(desc))
            mask_rows.append(self.torch.sigmoid(mask))
            terminal_rows.append(self.torch.softmax(terminal, dim=2))
            mixture_rows.append(self.torch.softmax(mixture, dim=1))
        return (
            self.torch.stack(descriptor_rows, dim=0),
            self.torch.stack(mask_rows, dim=0),
            self.torch.stack(terminal_rows, dim=0),
            self.torch.stack(mixture_rows, dim=0),
        )

    def _set_disagreement(
        self,
        descriptors: Any,
        masks: Any,
        terminals: Any,
        mixtures: Any,
    ) -> Any:
        """Epistemic disagreement between predicted mode *sets*.

        Diversity within one member is aleatoric and should not reduce model
        reliability. Only failure of different ensemble members to agree on a
        nearby mode set is penalized.
        """
        pair_scores = []
        ensemble = descriptors.shape[0]
        for left in range(ensemble):
            for right in range(left + 1, ensemble):
                desc_distance = (
                    descriptors[left].unsqueeze(2)
                    - descriptors[right].unsqueeze(1)
                ).abs().mean(dim=3)
                mask_distance = (
                    masks[left].unsqueeze(2)
                    - masks[right].unsqueeze(1)
                ).abs().mean(dim=3)
                terminal_distance = (
                    terminals[left].unsqueeze(2)
                    - terminals[right].unsqueeze(1)
                ).abs().mean(dim=3)
                distance = (
                    0.55 * desc_distance
                    + 0.30 * mask_distance
                    + 0.15 * terminal_distance
                )
                left_nearest = distance.min(dim=2).values
                right_nearest = distance.min(dim=1).values
                left_score = (
                    left_nearest * mixtures[left]
                ).sum(dim=1)
                right_score = (
                    right_nearest * mixtures[right]
                ).sum(dim=1)
                pair_scores.append(0.5 * (left_score + right_score))
        if not pair_scores:
            return self.torch.zeros(
                descriptors.shape[1],
                dtype=self.torch.float32,
                device=self.device,
            )
        return self.torch.stack(pair_scores, dim=0).mean(dim=0)

    def _confidence_from_decoded(
        self,
        descriptors: Any,
        masks: Any,
        terminals: Any,
        mixtures: Any,
    ) -> Any:
        disagreement = self._set_disagreement(
            descriptors,
            masks,
            terminals,
            mixtures,
        )
        self._last_ensemble_variance = float(
            disagreement.mean().detach().cpu().item()
        )
        sample_confidence = self.observations / (
            self.observations + self.config.confidence_prior
        )
        return self.torch.clamp(
            sample_confidence
            * self.torch.exp(-self.config.variance_scale * disagreement),
            min=0.05,
            max=0.995,
        )

    @staticmethod
    def _mode_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
        descriptor_distance = fmean(
            abs(a - b)
            for a, b in zip(left["descriptor"], right["descriptor"], strict=True)
        )
        mask_distance = fmean(
            abs(a - b)
            for a, b in zip(left["mask"], right["mask"], strict=True)
        )
        terminal_distance = fmean(
            abs(a - b)
            for a, b in zip(left["terminal"], right["terminal"], strict=True)
        )
        return (
            0.55 * descriptor_distance
            + 0.30 * mask_distance
            + 0.15 * terminal_distance
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
        total = sum(count for _, count in selected)
        rows = []
        for index, ((next_descriptor, next_mask, next_terminal), count) in enumerate(selected):
            source = f"{self.name}:empirical-outcome-{index}"
            rows.append(
                RelationalPrediction(
                    decode_relational_state_v2(
                        next_descriptor,
                        next_mask,
                        scaffold=state,
                        predicted_terminal=next_terminal,
                        source=source,
                    ),
                    max(0.0, min(1.0, float(confidence))),
                    source=source,
                    outcome_probability=(float(count) / total if total else 1.0 / len(selected)),
                )
            )
        self.empirical_multioutcome_rows += 1
        return tuple(rows)

    def _mixture_predictions(
        self,
        *,
        state: StateSnapshot,
        row_index: int,
        descriptors: list[Any],
        masks: list[Any],
        terminals: list[Any],
        mixtures: list[Any],
        confidence: float,
        samples: int,
    ) -> tuple[Prediction, ...]:
        candidates: list[dict[str, Any]] = []
        ensemble = len(descriptors)
        for member in range(ensemble):
            for component in range(self.config.mixture_components):
                candidates.append(
                    {
                        "member": member,
                        "component": component,
                        "descriptor": descriptors[member][row_index][component],
                        "mask": masks[member][row_index][component],
                        "terminal": terminals[member][row_index][component],
                        "mass": mixtures[member][row_index][component] / ensemble,
                    }
                )
        candidates.sort(key=lambda item: (-float(item["mass"]), item["member"], item["component"]))

        clusters: list[dict[str, Any]] = []
        for candidate in candidates:
            nearest = None
            nearest_distance = float("inf")
            for cluster in clusters:
                distance = self._mode_distance(candidate, cluster["representative"])
                if distance < nearest_distance:
                    nearest = cluster
                    nearest_distance = distance
            if nearest is not None and nearest_distance <= self.config.mode_merge_distance:
                nearest["mass"] += float(candidate["mass"])
                continue
            clusters.append(
                {
                    "representative": candidate,
                    "mass": float(candidate["mass"]),
                }
            )

        clusters.sort(key=lambda item: -item["mass"])
        selected = clusters[: max(1, int(samples))]
        selected_mass = sum(item["mass"] for item in selected)
        rows = []
        for index, cluster in enumerate(selected):
            candidate = cluster["representative"]
            terminal = max(
                range(3),
                key=lambda terminal_index: candidate["terminal"][terminal_index],
            )
            source = (
                f"{self.name}:mixture-{candidate['member']}-{candidate['component']}"
            )
            rows.append(
                RelationalPrediction(
                    decode_relational_state_v2(
                        candidate["descriptor"],
                        candidate["mask"],
                        scaffold=state,
                        predicted_terminal=terminal,
                        source=source,
                    ),
                    max(0.0, min(1.0, float(confidence))),
                    source=source,
                    outcome_probability=(
                        cluster["mass"] / selected_mass
                        if selected_mass > 0.0
                        else 1.0 / len(selected)
                    ),
                )
            )
        self.mixture_prediction_rows += 1
        return tuple(rows)

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
        descriptors_t, masks_t, terminals_t, mixtures_t = self._decoded_outputs(outputs)
        confidence_t = self._confidence_from_decoded(
            descriptors_t,
            masks_t,
            terminals_t,
            mixtures_t,
        )
        descriptors = descriptors_t.detach().cpu().tolist()
        masks = masks_t.detach().cpu().tolist()
        terminals = terminals_t.detach().cpu().tolist()
        mixtures = mixtures_t.detach().cpu().tolist()
        confidences = confidence_t.detach().cpu().tolist()

        rows = []
        for row_index, (state, action) in enumerate(zip(states, actions, strict=True)):
            empirical = self._empirical_predictions(
                state,
                action,
                confidence=float(confidences[row_index]),
                samples=samples,
            )
            if empirical:
                rows.append(empirical)
                continue
            rows.append(
                self._mixture_predictions(
                    state=state,
                    row_index=row_index,
                    descriptors=descriptors,
                    masks=masks,
                    terminals=terminals,
                    mixtures=mixtures,
                    confidence=float(confidences[row_index]),
                    samples=samples,
                )
            )
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
        outputs = self._forward(states, actions)
        decoded = self._decoded_outputs(outputs)
        return tuple(
            float(value)
            for value in self._confidence_from_decoded(*decoded).detach().cpu().tolist()
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
                parameter.numel()
                for model in self.models
                for parameter in model.parameters()
            ),
            "batch_prediction_calls": self.batch_prediction_calls,
            "batch_prediction_rows": self.batch_prediction_rows,
            "state_input_relational": 1,
            "action_input_relational": 1,
            "prediction_output": "relational-descriptor-v2+legal-mask+terminal-mixture",
            "conditional_mixture_components": self.config.mixture_components,
            "mixture_training_objective": "soft-mixture-likelihood",
            "epistemic_confidence": "ensemble-mode-set-disagreement",
            "reliability_outcome_probability_separated": 1,
            "empirical_multimodal_input_keys": multimodal_inputs,
            "empirical_distinct_outcomes": distinct_outcomes,
            "empirical_multioutcome_prediction_rows": self.empirical_multioutcome_rows,
            "learned_mixture_prediction_rows": self.mixture_prediction_rows,
        }
