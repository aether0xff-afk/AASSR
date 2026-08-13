from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Sequence
import random

from .current_generation import relational_action_key
from .current_relational_codec import (
    ACTION_SLOT_COUNT,
    TERMINAL_CLASSES,
)
from .current_relational_decode_v2 import decode_relational_state_v2
from .current_relational_model import (
    RelationalPrediction,
    RelationalProphecyConfig,
    RelationalStochasticProphecy,
)
from .current_relational_state import (
    REL_DESCRIPTOR_SIZE,
    relational_state_vector_v2,
)
from .pentest_agent_main_test import ACTION_FEATURE_SIZE, AGENT_STATE_SIZE
from .types import Action, Prediction, StateSnapshot


@dataclass(frozen=True, slots=True)
class RelationalMixtureProphecyConfig(RelationalProphecyConfig):
    mixture_components: int = 3
    mixture_entropy_weight: float = 0.01
    # Audited current runtime has only ensemble_size * mixture_components learned
    # hypotheses (3 * 3 by default). Preserve all semantically distinct modes and
    # merge only exact duplicates; approximate averaging can erase sparse but
    # decision-critical public differences such as one unlocked action or one
    # workflow/control channel.
    mode_merge_distance: float = 0.0


class ConditionalMixtureRelationalProphecy(RelationalStochasticProphecy):
    """Conditional multi-head relational world model.

    One structural state/action can legitimately have several next outcomes under
    partial observability. Each ensemble member predicts a categorical mixture of
    K semantic futures instead of one deterministic regression target. The target
    contains only the relational next-state descriptor, relational legal-action
    mask, and four-way task terminal class (active/success/failure/truncation).

    ``Prediction.probability`` is epistemic reliability. Stochastic outcome mass
    lives separately in ``outcome_probability`` and is consumed by the planner's
    chance backup, never added to branch value.
    """

    name = "current-relational-conditional-mixture-world-model-v3"

    def __init__(
        self,
        *,
        seed: int,
        device: str = "cpu",
        config: RelationalMixtureProphecyConfig | None = None,
        representation: object | None = None,
    ) -> None:
        config = config or RelationalMixtureProphecyConfig()
        if config.mixture_components <= 1:
            raise ValueError("mixture_components must be greater than one")

        # Reuse the audited replay/input/outcome bookkeeping from the relational
        # base class, then replace only its deterministic heads with mixture heads.
        super().__init__(
            seed=int(seed),
            device=device,
            config=config,
            representation=representation,
        )
        self.config: RelationalMixtureProphecyConfig = config
        self.input_size = (
            representation.state_size + representation.action_feature_size
            if representation is not None
            else AGENT_STATE_SIZE + ACTION_FEATURE_SIZE
        )
        descriptor_size = (
            representation.descriptor_size
            if representation is not None
            else REL_DESCRIPTOR_SIZE
        )
        self.component_size = (
            descriptor_size + ACTION_SLOT_COUNT + TERMINAL_CLASSES
        )
        self.output_size = (
            self.config.mixture_components * self.component_size
            + self.config.mixture_components
        )
        self.models = [
            self.nn.Sequential(
                self.nn.Linear(self.input_size, self.config.hidden_units),
                self.nn.ReLU(),
                self.nn.Linear(self.config.hidden_units, self.config.hidden_units),
                self.nn.ReLU(),
                self.nn.Linear(self.config.hidden_units, self.output_size),
            ).to(self.device)
            for _ in range(self.config.ensemble_size)
        ]
        self.optimizers = [
            self.torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
            for model in self.models
        ]
        self.mixture_prediction_rows = 0

    def _input(self, state: StateSnapshot, action: Action) -> tuple[float, ...]:
        values = (
            self.representation.state_vector(state)
            + self.representation.action_structure(state, action)
            if self.representation is not None
            else relational_state_vector_v2(state) + relational_action_key(state, action)
        )
        if len(values) != self.input_size:
            raise ValueError("relational mixture input size drift")
        return values

    def _split_output(self, output: Any) -> tuple[Any, Any, Any, Any]:
        k = self.config.mixture_components
        component_end = k * self.component_size
        components = output[:, :component_end].reshape(
            output.shape[0],
            k,
            self.component_size,
        )
        mixture_logits = output[:, component_end:]
        descriptor_end = self.component_size - ACTION_SLOT_COUNT - TERMINAL_CLASSES
        mask_end = descriptor_end + ACTION_SLOT_COUNT
        descriptor_logits = components[:, :, :descriptor_end]
        mask_logits = components[:, :, descriptor_end:mask_end]
        terminal_logits = components[:, :, mask_end:]
        if terminal_logits.shape[-1] != TERMINAL_CLASSES:
            raise RuntimeError("mixture terminal head size drift")
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
            ).unsqueeze(1).expand(-1, k)
            terminal_loss = nn.functional.cross_entropy(
                terminal_logits.reshape(-1, TERMINAL_CLASSES),
                terminal_target.reshape(-1),
                reduction="none",
            ).reshape(-1, k)

            reconstruction = (
                descriptor_loss
                + 0.35 * mask_loss
                + 0.25 * terminal_loss
            )
            log_mixture = torch.log_softmax(mixture_logits, dim=1)
            # Soft mixture likelihood lets different components own legitimately
            # different targets for the same relational input.
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

    def _decoded_outputs(self, outputs: Any) -> tuple[Any, Any, Any, Any]:
        descriptor_rows = []
        mask_rows = []
        terminal_rows = []
        mixture_rows = []
        for member in range(outputs.shape[0]):
            descriptor, mask, terminal, mixture = self._split_output(outputs[member])
            descriptor_rows.append(self.torch.sigmoid(descriptor))
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
        """Epistemic disagreement between predicted mode sets.

        Diversity *within* one member is aleatoric and should not lower model
        reliability. Only disagreement between ensemble members' mode sets does.
        Sparse public differences must not disappear in a high-dimensional mean,
        so each field uses its largest coordinate disagreement before the existing
        descriptor/mask/terminal weighting.
        """
        pair_scores = []
        ensemble = descriptors.shape[0]
        for left in range(ensemble):
            for right in range(left + 1, ensemble):
                descriptor_distance = (
                    descriptors[left].unsqueeze(2)
                    - descriptors[right].unsqueeze(1)
                ).abs().amax(dim=3)
                mask_distance = (
                    masks[left].unsqueeze(2)
                    - masks[right].unsqueeze(1)
                ).abs().amax(dim=3)
                terminal_distance = (
                    terminals[left].unsqueeze(2)
                    - terminals[right].unsqueeze(1)
                ).abs().amax(dim=3)
                distance = (
                    0.55 * descriptor_distance
                    + 0.30 * mask_distance
                    + 0.15 * terminal_distance
                )
                left_nearest = distance.min(dim=2).values
                right_nearest = distance.min(dim=1).values
                left_score = (left_nearest * mixtures[left]).sum(dim=1)
                right_score = (right_nearest * mixtures[right]).sum(dim=1)
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
        candidates.sort(
            key=lambda item: (-float(item["mass"]), item["member"], item["component"])
        )

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
            predicted_terminal = max(
                range(TERMINAL_CLASSES),
                key=lambda terminal_index: candidate["terminal"][terminal_index],
            )
            source = (
                f"{self.name}:mixture-{candidate['member']}-{candidate['component']}"
            )
            rows.append(
                RelationalPrediction(
                    (
                        self.representation.decode_state
                        if self.representation is not None
                        else decode_relational_state_v2
                    )(
                        candidate["descriptor"],
                        candidate["mask"],
                        scaffold=state,
                        predicted_terminal=predicted_terminal,
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
        decoded = self._decoded_outputs(self._forward(states, actions))
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
            "prediction_output": (
                "relational-descriptor-v2+legal-mask+"
                "active-success-failure-truncation-mixture"
            ),
            "conditional_mixture_components": self.config.mixture_components,
            "terminal_classes": TERMINAL_CLASSES,
            "mixture_training_objective": "soft-mixture-likelihood",
            "epistemic_confidence": "ensemble-mode-set-sparse-max-disagreement",
            "mode_merge_distance": self.config.mode_merge_distance,
            "lossy_mode_merge_disabled": int(self.config.mode_merge_distance == 0.0),
            "reliability_outcome_probability_separated": 1,
            "empirical_multimodal_input_keys": multimodal_inputs,
            "empirical_distinct_outcomes": distinct_outcomes,
            "empirical_multioutcome_prediction_rows": self.empirical_multioutcome_rows,
            "learned_mixture_prediction_rows": self.mixture_prediction_rows,
        }
