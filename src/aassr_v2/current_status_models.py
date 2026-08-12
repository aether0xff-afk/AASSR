from __future__ import annotations

import random
from statistics import fmean
from typing import Any, Type

from .current_agent import CurrentProphecyView
from .current_relational_codec import ACTION_SLOT_COUNT, TERMINAL_CLASSES
from .current_relational_mixture_model import ConditionalMixtureRelationalProphecy
from .current_relational_model import RelationalStochasticProphecy
from .current_relational_skill_prophecy import RelationalStochasticSkillProphecy
from .current_relational_state_v3 import (
    STATUS_CODES_V3,
    STATUS_SIZE,
    STATUS_START_INDEX,
    latest_status_vector,
)
from .current_semantic_calibration import (
    RelationalDepthBatchedProphecyView,
    SemanticCalibratedProphecy,
    SemanticPredictionValidator,
)
from .current_semantic_evaluator import RelationalAdvancedTransitionEvaluator
from .skills import SKILL_VERB
from .types import Action, StateSnapshot


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
    """

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

        Rows with no public status use only a small all-zero BCE fallback. This
        preserves the existing no-observation representation without inventing a
        ninth task-specific class or changing the 43-D public-state contract.
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

    def _status_diagnostics(self) -> dict[str, int | float | str]:
        weights = self._status_class_weights()
        result: dict[str, int | float | str] = {
            "status_supervision": 1,
            "status_output_channels": STATUS_SIZE,
            "status_loss_weight": STATUS_LOSS_WEIGHT,
            "status_training_objective": STATUS_OBJECTIVE,
            "status_class_weighting": "inverse-sqrt-frequency-capped-normalized",
            "status_class_weight_power": STATUS_CLASS_WEIGHT_POWER,
            "status_class_weight_cap": STATUS_CLASS_WEIGHT_CAP,
            "status_no_observation_bce_weight": STATUS_NO_OBSERVATION_BCE_WEIGHT,
            "status_no_observation_count": self._status_no_observation_count,
            "last_status_training_loss": self._last_status_training_loss,
            "last_status_training_accuracy": self._last_status_training_accuracy,
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

    name = "current-relational-stochastic-world-model-v4-status-balanced"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        descriptor_size = self.output_size - ACTION_SLOT_COUNT - TERMINAL_CLASSES
        if descriptor_size < STATUS_START_INDEX + STATUS_SIZE:
            raise ValueError(
                "status-aware Prophecy requires relational public state v3"
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
            # Status has its own categorical objective below. Excluding it from the
            # generic descriptor regression prevents the many negative channels of
            # a one-hot status vector from overwhelming a rare positive class.
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

    def diagnostics(self) -> dict[str, int | float | str]:
        return {
            **super().diagnostics(),
            **self._status_diagnostics(),
        }


class StatusAwareConditionalMixtureRelationalProphecy(
    _BalancedPublicStatusMixin,
    ConditionalMixtureRelationalProphecy,
):
    """Conditional mixture model with balanced categorical status supervision."""

    name = "current-relational-conditional-mixture-world-model-v5-status-balanced"

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

    def diagnostics(self) -> dict[str, int | float | str]:
        return {
            **super().diagnostics(),
            **self._status_diagnostics(),
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
    )
    calibrated = SemanticCalibratedProphecy(base, replay)
    skill = RelationalStochasticSkillProphecy(
        calibrated,
        agent.skills,
        agent.knowledge,
    )
    prophecy = CurrentProphecyView(skill)
    validator = SemanticPredictionValidator(samples=3)
    evaluator = RelationalAdvancedTransitionEvaluator(
        prophecy,
        replay=replay,
        validator=validator,
        predictor=old_evaluator.predictor,
        logger=old_evaluator.logger,
        samples=3,
        intrinsic_cap=float(old_evaluator.intrinsic_cap),
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
    agent.current_batched_prophecy = batched
    agent.planner.prophecy = batched
    agent.core.prophecy = prophecy
    agent.current_status_supervised_world_model = True
    return agent
