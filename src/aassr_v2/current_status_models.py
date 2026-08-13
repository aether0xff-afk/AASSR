from __future__ import annotations

import random
from statistics import fmean
from typing import Any, Type

from .current_agent import CurrentProphecyView
from .current_relational_codec import ACTION_SLOT_COUNT, TERMINAL_CLASSES
from .current_relational_mixture_model import ConditionalMixtureRelationalProphecy
from .current_relational_model import RelationalPrediction, RelationalStochasticProphecy
from .current_relational_skill_prophecy import RelationalStochasticSkillProphecy
from .current_relational_state_v3 import (
    STATUS_CODES_V3,
    STATUS_SIZE,
    STATUS_START_INDEX,
    decode_relational_state_v3,
    latest_status_vector,
    semantic_prediction_score_v3,
)
from .current_semantic_calibration import (
    RelationalDepthBatchedProphecyView,
    SemanticCalibratedProphecy,
    SemanticPredictionValidator,
)
from .current_semantic_evaluator import RelationalAdvancedTransitionEvaluator
from .skills import SKILL_VERB
from .types import Action, Prediction, StateSnapshot


# Keep the overall status contribution unchanged from the first v3 repair so the
# new experiment changes *which* status objective is learned, not how much status
# dominates the total world-model loss.
STATUS_LOSS_WEIGHT = 0.5
STATUS_CLASS_WEIGHT_POWER = 0.5
STATUS_CLASS_WEIGHT_CAP = 4.0
STATUS_NO_OBSERVATION_BCE_WEIGHT = 0.25
STATUS_OBJECTIVE = "class-balanced-categorical-public-http-status-v2"


class _BalancedPublicStatusMixin:
    """General class balancing for mutually-exclusive public response outcomes.

    No HTTP code is assigned a hand-written meaning or penalty. We only use the
    empirical frequency of each publicly observed status class. Rare classes get
    a capped inverse-sqrt weight, normalized so the expected status-loss scale
    stays approximately constant as the class distribution changes.

    Active v3 models also preserve a complete empirical outcome distribution.
    Once the world model says an exact relational input has produced several real
    outcomes, none of their probability mass may be silently discarded before a
    planner claiming to compute expected sparse return sees it.
    """

    complete_outcome_distribution = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._status_observation_counts = [0 for _ in range(STATUS_SIZE)]
        self._status_no_observation_count = 0
        self._last_status_training_loss = 0.0
        self._last_status_training_accuracy = 0.0

    def _record_public_status(self, state: StateSnapshot) -> None:
        values = latest_status_vector(state)
        if values and max(values) >= 0.5:
            index = max(range(STATUS_SIZE), key=lambda item: (values[item], -item))
            self._status_observation_counts[index] += 1
        else:
            self._status_no_observation_count += 1

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        if action.verb_name != SKILL_VERB:
            self._record_public_status(actual_next_state)
        super().learn(state, action, actual_next_state)

    def _status_class_weights(self) -> tuple[float, ...]:
        counts = tuple(int(value) for value in self._status_observation_counts)
        present = [value for value in counts if value > 0]
        if not present:
            return (1.0,) * STATUS_SIZE
        maximum = max(present)
        raw = [
            min(
                STATUS_CLASS_WEIGHT_CAP,
                (maximum / max(1, count)) ** STATUS_CLASS_WEIGHT_POWER,
            )
            if count > 0
            else 1.0
            for count in counts
        ]
        total = sum(counts)
        expected = (
            sum(count * weight for count, weight in zip(counts, raw, strict=True))
            / max(1, total)
        )
        expected = max(expected, 1e-12)
        return tuple(weight / expected for weight in raw)

    def _status_class_weight_tensor(self) -> Any:
        return self._tensor(self._status_class_weights())

    def _status_loss_rows(
        self,
        status_logits: Any,
        status_target: Any,
    ) -> tuple[Any, float]:
        """Return per-row categorical loss and top-1 accuracy for observed status.

        Rows with no public status use only a small all-zero BCE fallback for
        compatibility with non-response historical rows. HTTP response rows are
        mutually exclusive categorical observations and never become a ninth
        invented "unknown status" outcome during active inference.
        """
        torch = self.torch
        nn = self.nn
        leading_shape = status_logits.shape[:-1]
        flat_logits = status_logits.reshape(-1, STATUS_SIZE)
        flat_target = status_target.reshape(-1, STATUS_SIZE)
        observed = flat_target.sum(dim=1) >= 0.5
        losses = torch.zeros(
            flat_logits.shape[0],
            dtype=torch.float32,
            device=self.device,
        )
        accuracy = 0.0

        if bool(observed.any()):
            logits = flat_logits[observed]
            target = flat_target[observed]
            classes = target.argmax(dim=1)
            log_probabilities = torch.log_softmax(logits, dim=1)
            class_weights = self._status_class_weight_tensor()
            picked = -log_probabilities.gather(1, classes.unsqueeze(1)).squeeze(1)
            losses[observed] = picked * class_weights[classes]
            predicted = logits.argmax(dim=1)
            accuracy = float(
                (predicted == classes).float().mean().detach().cpu().item()
            )

        missing = ~observed
        if bool(missing.any()):
            missing_loss = nn.functional.binary_cross_entropy_with_logits(
                flat_logits[missing],
                torch.zeros_like(flat_logits[missing]),
                reduction="none",
            ).mean(dim=1)
            losses[missing] = STATUS_NO_OBSERVATION_BCE_WEIGHT * missing_loss

        return losses.reshape(leading_shape), accuracy

    def _empirical_predictions(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        confidence: float,
        samples: int,
    ) -> tuple[Prediction, ...]:
        del samples
        bucket = self._outcomes.get(self._input(state, action), {})
        if len(bucket) < 2:
            return ()
        selected = sorted(
            bucket.items(),
            key=lambda item: (-item[1], item[0]),
        )
        total = sum(count for _, count in selected)
        predictions = []
        for index, ((next_descriptor, next_mask, next_terminal), count) in enumerate(selected):
            source = f"{self.name}:empirical-outcome-{index}"
            predictions.append(
                RelationalPrediction(
                    decode_relational_state_v3(
                        next_descriptor,
                        next_mask,
                        scaffold=state,
                        predicted_terminal=next_terminal,
                        source=source,
                    ),
                    max(0.0, min(1.0, float(confidence))),
                    source=source,
                    outcome_probability=(
                        float(count) / float(total)
                        if total > 0
                        else 1.0 / len(selected)
                    ),
                )
            )
        self.empirical_multioutcome_rows += 1
        return tuple(predictions)

    def _status_diagnostics(self) -> dict[str, int | float | str]:
        weights = self._status_class_weights()
        result: dict[str, int | float | str] = {
            "status_supervision": 1,
            "status_output_channels": STATUS_SIZE,
            "status_loss_weight": STATUS_LOSS_WEIGHT,
            "status_training_objective": STATUS_OBJECTIVE,
            "status_inference_objective": "softmax-categorical-realized-one-hot",
            "status_class_weighting": "inverse-sqrt-frequency-capped-normalized",
            "status_class_weight_power": STATUS_CLASS_WEIGHT_POWER,
            "status_class_weight_cap": STATUS_CLASS_WEIGHT_CAP,
            "status_no_observation_bce_weight": STATUS_NO_OBSERVATION_BCE_WEIGHT,
            "status_no_observation_count": self._status_no_observation_count,
            "last_status_training_loss": self._last_status_training_loss,
            "last_status_training_accuracy": self._last_status_training_accuracy,
            "complete_empirical_outcome_mass": 1,
        }
        for index, code in enumerate(STATUS_CODES_V3):
            result[f"status_count_{code}"] = self._status_observation_counts[index]
            result[f"status_class_weight_{code}"] = weights[index]
        return result


class StatusAwareRelationalStochasticProphecy(
    _BalancedPublicStatusMixin,
    RelationalStochasticProphecy,
):
    """Relational world model with balanced categorical public-status supervision."""

    name = "current-relational-stochastic-world-model-v5-status-categorical"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        descriptor_size = self.output_size - ACTION_SLOT_COUNT - TERMINAL_CLASSES
        if descriptor_size < STATUS_START_INDEX + STATUS_SIZE:
            raise ValueError(
                "status-aware Prophecy requires relational public state v3"
            )

    def _decoded_outputs(self, outputs: Any) -> tuple[Any, Any, Any]:
        """Decode every status-aware base-model path with categorical geometry."""
        descriptor_size = self.output_size - ACTION_SLOT_COUNT - TERMINAL_CLASSES
        mask_start = descriptor_size
        mask_end = mask_start + ACTION_SLOT_COUNT
        pre_status = self.torch.sigmoid(outputs[:, :, :STATUS_START_INDEX])
        status = self.torch.softmax(
            outputs[
                :,
                :,
                STATUS_START_INDEX : STATUS_START_INDEX + STATUS_SIZE,
            ],
            dim=-1,
        )
        descriptor_parts = [pre_status, status]
        if descriptor_size > STATUS_START_INDEX + STATUS_SIZE:
            descriptor_parts.append(
                self.torch.sigmoid(
                    outputs[
                        :,
                        :,
                        STATUS_START_INDEX + STATUS_SIZE : descriptor_size,
                    ]
                )
            )
        descriptors = self.torch.cat(descriptor_parts, dim=-1)
        masks = self.torch.sigmoid(outputs[:, :, mask_start:mask_end])
        terminals = self.torch.softmax(outputs[:, :, mask_end:], dim=-1)
        return descriptors, masks, terminals

    def _confidence(self, outputs: Any) -> Any:
        """Use the same categorical geometry for training and reliability."""
        descriptors, masks, terminals = self._decoded_outputs(outputs)
        parts = (descriptors, masks, terminals)
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

    def _train_step(self) -> None:
        status_losses: list[float] = []
        status_accuracies: list[float] = []
        descriptor_size = self.output_size - ACTION_SLOT_COUNT - TERMINAL_CLASSES
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
            descriptor_logits = output[:, :descriptor_size]
            mask_start = descriptor_size
            mask_end = mask_start + ACTION_SLOT_COUNT
            mask_logits = output[:, mask_start:mask_end]
            terminal_logits = output[:, mask_end:]

            descriptor_target = self._tensor([row[1] for row in batch])
            descriptor_loss = self.nn.functional.smooth_l1_loss(
                self.torch.sigmoid(descriptor_logits[:, :STATUS_START_INDEX]),
                descriptor_target[:, :STATUS_START_INDEX],
            )
            status_rows, status_accuracy = self._status_loss_rows(
                descriptor_logits[
                    :, STATUS_START_INDEX : STATUS_START_INDEX + STATUS_SIZE
                ],
                descriptor_target[
                    :, STATUS_START_INDEX : STATUS_START_INDEX + STATUS_SIZE
                ],
            )
            status_loss = status_rows.mean()
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
            loss = (
                descriptor_loss
                + STATUS_LOSS_WEIGHT * status_loss
                + 0.35 * mask_loss
                + 0.25 * terminal_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
            optimizer.step()
            self._losses.append(float(loss.detach().cpu().item()))
            status_losses.append(float(status_loss.detach().cpu().item()))
            status_accuracies.append(status_accuracy)
        self._last_status_training_loss = fmean(status_losses) if status_losses else 0.0
        self._last_status_training_accuracy = (
            fmean(status_accuracies) if status_accuracies else 0.0
        )
        self.gradient_updates += 1

    def predict_batch(
        self,
        states: tuple[StateSnapshot, ...] | list[StateSnapshot] | Any,
        actions: tuple[Action, ...] | list[Action] | Any,
        *,
        samples: int,
    ) -> tuple[tuple[Prediction, ...], ...]:
        """Mirror the base predictor without its descriptor-wide sigmoid.

        The class advertises a complete learned outcome distribution, so the
        caller's requested sample count may not truncate ensemble probability
        mass. Empirical multi-outcome rows are already complete above; learned
        rows therefore emit every ensemble member as an equal-mass hypothesis.
        """
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
        descriptors_t, masks_t, terminals_t = self._decoded_outputs(outputs)
        descriptors = descriptors_t.detach().cpu().tolist()
        masks = masks_t.detach().cpu().tolist()
        terminals = terminals_t.detach().cpu().tolist()

        member_count = int(self.config.ensemble_size)
        rows: list[tuple[Prediction, ...]] = []
        for row_index, (state, action) in enumerate(zip(states, actions, strict=True)):
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
                        decode_relational_state_v3(
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

    def diagnostics(self) -> dict[str, int | float | str]:
        return {
            **super().diagnostics(),
            **self._status_diagnostics(),
            "status_categorical_inference": 1,
            "complete_learned_ensemble_mass": 1,
        }


class StatusAwareConditionalMixtureRelationalProphecy(
    _BalancedPublicStatusMixin,
    ConditionalMixtureRelationalProphecy,
):
    """Conditional mixture with categorical status and decision-safe mode identity."""

    name = "current-relational-conditional-mixture-world-model-v6-status-categorical"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        descriptor_size = (
            self.component_size - ACTION_SLOT_COUNT - TERMINAL_CLASSES
        )
        if descriptor_size < STATUS_START_INDEX + STATUS_SIZE:
            raise ValueError(
                "status-aware mixture Prophecy requires relational public state v3"
            )

    def _train_step(self) -> None:
        torch = self.torch
        nn = self.nn
        k = self.config.mixture_components
        status_losses: list[float] = []
        status_accuracies: list[float] = []
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

            descriptor_target_single = self._tensor([row[1] for row in batch])
            descriptor_target = descriptor_target_single.unsqueeze(1).expand(-1, k, -1)
            descriptor_loss = nn.functional.smooth_l1_loss(
                torch.sigmoid(descriptor_logits[:, :, :STATUS_START_INDEX]),
                descriptor_target[:, :, :STATUS_START_INDEX],
                reduction="none",
            ).mean(dim=2)
            status_rows, status_accuracy = self._status_loss_rows(
                descriptor_logits[
                    :, :, STATUS_START_INDEX : STATUS_START_INDEX + STATUS_SIZE
                ],
                descriptor_target[
                    :, :, STATUS_START_INDEX : STATUS_START_INDEX + STATUS_SIZE
                ],
            )

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
                + STATUS_LOSS_WEIGHT * status_rows
                + 0.35 * mask_loss
                + 0.25 * terminal_loss
            )
            log_mixture = torch.log_softmax(mixture_logits, dim=1)
            nll = -torch.logsumexp(log_mixture - reconstruction, dim=1).mean()
            mixture = torch.softmax(mixture_logits, dim=1)
            entropy = -(mixture * log_mixture).sum(dim=1).mean()
            loss = nll - self.config.mixture_entropy_weight * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
            optimizer.step()
            self._losses.append(float(loss.detach().cpu().item()))
            status_losses.append(float(status_rows.mean().detach().cpu().item()))
            status_accuracies.append(status_accuracy)
        self._last_status_training_loss = fmean(status_losses) if status_losses else 0.0
        self._last_status_training_accuracy = (
            fmean(status_accuracies) if status_accuracies else 0.0
        )
        self.gradient_updates += 1

    def _decoded_outputs(self, outputs: Any) -> tuple[Any, Any, Any, Any]:
        """Decode the categorical status head with softmax, never sigmoid."""
        descriptor_rows = []
        mask_rows = []
        terminal_rows = []
        mixture_rows = []
        for member in range(outputs.shape[0]):
            descriptor_logits, mask, terminal, mixture = self._split_output(outputs[member])
            pre_status = self.torch.sigmoid(
                descriptor_logits[:, :, :STATUS_START_INDEX]
            )
            status = self.torch.softmax(
                descriptor_logits[
                    :,
                    :,
                    STATUS_START_INDEX : STATUS_START_INDEX + STATUS_SIZE,
                ],
                dim=2,
            )
            descriptor_parts = [pre_status, status]
            if descriptor_logits.shape[2] > STATUS_START_INDEX + STATUS_SIZE:
                descriptor_parts.append(
                    self.torch.sigmoid(
                        descriptor_logits[
                            :,
                            :,
                            STATUS_START_INDEX + STATUS_SIZE :,
                        ]
                    )
                )
            descriptor_rows.append(self.torch.cat(descriptor_parts, dim=2))
            mask_rows.append(self.torch.sigmoid(mask))
            terminal_rows.append(self.torch.softmax(terminal, dim=2))
            mixture_rows.append(self.torch.softmax(mixture, dim=1))
        return (
            self.torch.stack(descriptor_rows, dim=0),
            self.torch.stack(mask_rows, dim=0),
            self.torch.stack(terminal_rows, dim=0),
            self.torch.stack(mixture_rows, dim=0),
        )

    @staticmethod
    def _mode_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
        """Never merge modes that induce a different real decision surface."""
        left_descriptor = tuple(float(value) for value in left["descriptor"])
        right_descriptor = tuple(float(value) for value in right["descriptor"])
        left_status = max(
            range(STATUS_SIZE),
            key=lambda index: left_descriptor[STATUS_START_INDEX + index],
        )
        right_status = max(
            range(STATUS_SIZE),
            key=lambda index: right_descriptor[STATUS_START_INDEX + index],
        )
        if left_status != right_status:
            return 1.0

        left_mask = tuple(float(value) >= 0.5 for value in left["mask"])
        right_mask = tuple(float(value) >= 0.5 for value in right["mask"])
        if left_mask != right_mask:
            return 1.0

        left_terminal = max(range(TERMINAL_CLASSES), key=lambda i: left["terminal"][i])
        right_terminal = max(range(TERMINAL_CLASSES), key=lambda i: right["terminal"][i])
        if left_terminal != right_terminal:
            return 1.0

        base_left = left_descriptor[:STATUS_START_INDEX]
        base_right = right_descriptor[:STATUS_START_INDEX]
        if not base_left:
            return 0.0
        return fmean(
            abs(a - b) for a, b in zip(base_left, base_right, strict=True)
        )

    def _mixture_predictions(self, **kwargs: Any) -> tuple[Prediction, ...]:
        # At most ensemble_size * mixture_components learned components exist.
        # Request all of them so the generic clustering code does not condition on
        # a top-k subset and silently renormalize away tail probability mass.
        kwargs["samples"] = self.config.ensemble_size * self.config.mixture_components
        return super()._mixture_predictions(**kwargs)

    def diagnostics(self) -> dict[str, int | float | str]:
        return {
            **super().diagnostics(),
            **self._status_diagnostics(),
            "status_categorical_inference": 1,
            "decision_surface_preserving_mode_merge": 1,
            "complete_learned_mixture_mass": 1,
        }


def install_status_supervised_world_model(
    agent: object,
    *,
    seed: int,
    device: str,
    model_class: Type[RelationalStochasticProphecy] = StatusAwareRelationalStochasticProphecy,
) -> object:
    """Replace the just-built untrained current world model with a status-aware one.

    Builders call this immediately after ``install_current_repairs`` and before
    any environment transition is collected. Replay/evaluator auxiliaries are
    preserved, so the replacement changes only the world-model training objective
    and its dependent Prophecy views.
    """
    if getattr(agent, "current_status_supervised_world_model", False):
        return agent

    old_evaluator = agent.evaluator
    replay = old_evaluator.replay
    if int(getattr(agent.base_neural_prophecy, "observations", 0)) != 0:
        raise RuntimeError(
            "status-supervised world model must be installed before real training"
        )

    base = model_class(
        seed=int(seed) ^ 0x52454C41,
        device=device,
        representation=getattr(agent, "representation", None),
    )
    representation = getattr(agent, "representation", None)
    calibrated = SemanticCalibratedProphecy(
        base,
        replay,
        representation=representation,
    )
    skill = RelationalStochasticSkillProphecy(
        calibrated,
        agent.skills,
        agent.knowledge,
    )
    prophecy = CurrentProphecyView(skill)
    validator = SemanticPredictionValidator(
        samples=3,
        prediction_score=(
            representation.prediction_score
            if representation is not None
            else semantic_prediction_score_v3
        ),
    )
    evaluator = RelationalAdvancedTransitionEvaluator(
        prophecy,
        replay=replay,
        validator=validator,
        predictor=old_evaluator.predictor,
        logger=old_evaluator.logger,
        samples=3,
        intrinsic_cap=float(old_evaluator.intrinsic_cap),
        representation=representation,
    )

    agent.base_neural_prophecy = base
    agent.calibrated_prophecy = calibrated
    agent.knowledge_prophecy = calibrated
    agent.skill_prophecy = skill
    agent.prophecy = prophecy
    agent.evaluator = evaluator
    agent.current_semantic_validation = True
    agent.current_semantic_evaluator = True

    batched = RelationalDepthBatchedProphecyView(agent)
    batched.complete_outcome_distribution = bool(
        getattr(base, "complete_outcome_distribution", False)
    )
    agent.current_batched_prophecy = batched
    agent.planner.prophecy = batched
    agent.core.prophecy = prophecy
    agent.current_status_supervised_world_model = True
    return agent
