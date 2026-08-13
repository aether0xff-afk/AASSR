from __future__ import annotations

from collections import defaultdict
from math import exp
from types import MethodType
from typing import Any, Sequence
import random

from .current_generation import relational_action_key
from .current_relational_codec import ACTION_SLOT_COUNT, TERMINAL_CLASSES
from .current_relational_state import relational_state_vector_v2
from .current_relational_state_v3 import (
    STATUS_SIZE,
    STATUS_START_INDEX,
    relational_state_key_v3,
)
from .current_semantic_calibration import (
    CALIBRATION_CACHE_LIMIT,
    CALIBRATION_LOCALITY_SCALE,
    _public_state_distance,
    probability_weighted_semantic_score,
)
from .current_status_models import (
    STATUS_LOSS_WEIGHT,
    STATUS_NO_OBSERVATION_BCE_WEIGHT,
    StatusAwareConditionalMixtureRelationalProphecy,
)
from .types import Action, Prediction, StateSnapshot


PERFORMANCE_CONTRACT = "semantics-preserving-no-training-schedule-change-v1"


def _install_indexed_calibration(calibrated: object) -> None:
    """Avoid rescanning the complete holdout before every calibration cache hit."""

    calibrated._performance_holdout_index_key = None
    calibrated._performance_holdout_index = {}
    calibrated.performance_holdout_index_rebuilds = 0
    calibrated.performance_holdout_index_hits = 0

    def indexed_items(
        self: object,
        state: StateSnapshot,
        action: Action,
    ) -> tuple[Any, tuple[Any, ...]]:
        representation = getattr(self, "representation", None)
        action_structure = (
            representation.action_structure
            if representation is not None
            else relational_action_key
        )
        action_key = action_structure(state, action)
        frozen = getattr(self, "_frozen_holdout", None)
        if frozen is not None:
            source = frozen
            revision = (
                "frozen",
                id(frozen),
                len(frozen),
                id(frozen[-1]) if frozen else 0,
            )
        else:
            replay = self.replay
            # Internal read-only view: ReplayBuffer.holdout() creates a new tuple on
            # every call. The holdout changes only once per holdout_stride samples.
            source = replay._holdout
            revision = (
                "live",
                id(replay),
                int(replay._seen) // int(replay.holdout_stride),
                len(source),
            )

        if revision != self._performance_holdout_index_key:
            grouped: dict[Any, list[Any]] = defaultdict(list)
            for item in source:
                grouped[action_structure(item.state, item.action)].append(item)
            self._performance_holdout_index = {
                key: tuple(values) for key, values in grouped.items()
            }
            self._performance_holdout_index_key = revision
            self.performance_holdout_index_rebuilds += 1
        else:
            self.performance_holdout_index_hits += 1
        return action_key, self._performance_holdout_index.get(action_key, ())

    def calibration(self: object, state: StateSnapshot, action: Action) -> float:
        action_key, items = indexed_items(self, state, action)
        representation = getattr(self, "representation", None)
        state_key = (
            representation.state_key
            if representation is not None
            else relational_state_key_v3
        )
        prediction_score = (
            representation.prediction_score
            if representation is not None
            else None
        )
        revision = int(self.base.gradient_updates)
        cache_key = (
            action_key,
            state_key(state),
            len(items) // max(1, self.refresh_stride),
            revision // max(1, self.refresh_stride),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]
        self.refreshes += 1
        if len(items) < self.minimum_count:
            value = 0.0
        else:
            nearest = sorted(
                ((_public_state_distance(state, item.state), item) for item in items),
                key=lambda row: row[0],
            )[: self.evaluation_limit]
            selected = tuple(item for _, item in nearest)
            locality = tuple(
                exp(-CALIBRATION_LOCALITY_SCALE * max(0.0, distance))
                for distance, _ in nearest
            )
            rows = self.base.predict_batch(
                tuple(item.state for item in selected),
                tuple(item.action for item in selected),
                samples=self.base.config.ensemble_size,
            )
            self.batch_refreshes += 1
            self.batch_rows += len(selected)
            scores = tuple(
                probability_weighted_semantic_score(
                    predictions,
                    item.next_state,
                    prediction_score=prediction_score,
                )
                for item, predictions in zip(selected, rows, strict=True)
            )
            total = sum(locality)
            if total <= 1e-12:
                value = 0.0
            else:
                local_score = sum(
                    weight * score
                    for weight, score in zip(locality, scores, strict=True)
                ) / total
                value = local_score * max(locality)
            value = max(0.0, min(1.0, value))
        if len(self._cache) >= CALIBRATION_CACHE_LIMIT and cache_key not in self._cache:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = value
        return value

    calibrated._calibration = MethodType(calibration, calibrated)


def _status_loss_rows_device(model: object, status_logits: Any, status_target: Any) -> tuple[Any, Any]:
    """Same categorical status loss without Python boolean/device synchronizations."""

    torch = model.torch
    nn = model.nn
    leading_shape = status_logits.shape[:-1]
    flat_logits = status_logits.reshape(-1, STATUS_SIZE)
    flat_target = status_target.reshape(-1, STATUS_SIZE)
    observed = flat_target.sum(dim=1) >= 0.5
    classes = flat_target.argmax(dim=1)
    log_probabilities = torch.log_softmax(flat_logits, dim=1)
    class_weights = model._status_class_weight_tensor()
    picked = -log_probabilities.gather(1, classes.unsqueeze(1)).squeeze(1)
    observed_loss = picked * class_weights[classes]
    missing_loss = nn.functional.binary_cross_entropy_with_logits(
        flat_logits,
        torch.zeros_like(flat_logits),
        reduction="none",
    ).mean(dim=1)
    losses = torch.where(
        observed,
        observed_loss,
        STATUS_NO_OBSERVATION_BCE_WEIGHT * missing_loss,
    )
    predicted = flat_logits.argmax(dim=1)
    observed_count = observed.to(torch.float32).sum()
    correct = ((predicted == classes) & observed).to(torch.float32).sum()
    accuracy = correct / observed_count.clamp_min(1.0)
    return losses.reshape(leading_shape), accuracy


def _install_status_mixture_fast_path(base: object) -> None:
    if not isinstance(base, StatusAwareConditionalMixtureRelationalProphecy):
        return

    base.performance_state_encode_cache_hits = 0
    base.performance_state_encode_cache_misses = 0
    base.performance_host_transfer_batches = 0
    base.performance_training_metric_syncs = 0
    base._performance_pending_ensemble_variance = None
    original_diagnostics = base.diagnostics

    def forward(
        self: object,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
    ) -> Any:
        encoded: dict[int, tuple[StateSnapshot, tuple[float, ...]]] = {}
        inputs: list[tuple[float, ...]] = []
        representation = getattr(self, "representation", None)
        state_vector = (
            representation.state_vector
            if representation is not None
            else relational_state_vector_v2
        )
        action_structure = (
            representation.action_structure
            if representation is not None
            else relational_action_key
        )
        for state, action in zip(states, actions, strict=True):
            identity = id(state)
            cached = encoded.get(identity)
            if cached is None or cached[0] is not state:
                cached = (state, state_vector(state))
                encoded[identity] = cached
                self.performance_state_encode_cache_misses += 1
            else:
                self.performance_state_encode_cache_hits += 1
            values = cached[1] + action_structure(state, action)
            if len(values) != self.input_size:
                raise ValueError("relational mixture input size drift")
            inputs.append(values)
        tensor = self._tensor(inputs)
        with self.torch.no_grad():
            return self.torch.stack([model(tensor) for model in self.models], dim=0)

    def decoded_outputs(self: object, outputs: Any) -> tuple[Any, Any, Any, Any]:
        k = self.config.mixture_components
        component_end = k * self.component_size
        components = outputs[:, :, :component_end].reshape(
            outputs.shape[0],
            outputs.shape[1],
            k,
            self.component_size,
        )
        mixture_logits = outputs[:, :, component_end:]
        descriptor_size = self.component_size - ACTION_SLOT_COUNT - TERMINAL_CLASSES
        mask_end = descriptor_size + ACTION_SLOT_COUNT
        descriptor_logits = components[:, :, :, :descriptor_size]
        mask_logits = components[:, :, :, descriptor_size:mask_end]
        terminal_logits = components[:, :, :, mask_end:]

        pre_status = self.torch.sigmoid(
            descriptor_logits[:, :, :, :STATUS_START_INDEX]
        )
        status = self.torch.softmax(
            descriptor_logits[
                :,
                :,
                :,
                STATUS_START_INDEX : STATUS_START_INDEX + STATUS_SIZE,
            ],
            dim=-1,
        )
        parts = [pre_status, status]
        if descriptor_size > STATUS_START_INDEX + STATUS_SIZE:
            parts.append(
                self.torch.sigmoid(
                    descriptor_logits[
                        :,
                        :,
                        :,
                        STATUS_START_INDEX + STATUS_SIZE :,
                    ]
                )
            )
        descriptors = self.torch.cat(parts, dim=-1)
        masks = self.torch.sigmoid(mask_logits)
        terminals = self.torch.softmax(terminal_logits, dim=-1)
        mixtures = self.torch.softmax(mixture_logits, dim=-1)
        return descriptors, masks, terminals, mixtures

    def confidence_from_decoded(
        self: object,
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
        self._performance_pending_ensemble_variance = disagreement.mean().detach()
        sample_confidence = self.observations / (
            self.observations + self.config.confidence_prior
        )
        return self.torch.clamp(
            sample_confidence
            * self.torch.exp(-self.config.variance_scale * disagreement),
            min=0.05,
            max=0.995,
        )

    def train_step(self: object) -> None:
        torch = self.torch
        nn = self.nn
        k = self.config.mixture_components
        model_losses: list[Any] = []
        status_losses: list[Any] = []
        status_accuracies: list[Any] = []

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
            status_rows, status_accuracy = _status_loss_rows_device(
                self,
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
            model_losses.append(loss.detach())
            status_losses.append(status_rows.mean().detach())
            status_accuracies.append(status_accuracy.detach())

        # Diagnostics used to force several GPU->CPU barriers per ensemble member.
        # Materialize all scalar metrics with one transfer after all updates instead.
        metric_tensor = torch.stack(
            model_losses
            + [torch.stack(status_losses).mean(), torch.stack(status_accuracies).mean()]
        )
        host_metrics = metric_tensor.cpu().tolist()
        self.performance_training_metric_syncs += 1
        for value in host_metrics[: len(model_losses)]:
            self._losses.append(float(value))
        self._last_status_training_loss = float(host_metrics[-2])
        self._last_status_training_accuracy = float(host_metrics[-1])
        self.gradient_updates += 1

    def predict_batch(
        self: object,
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
                    self.RelationalPrediction(
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

        ensemble, batch_size, components, descriptor_size = descriptors_t.shape
        mask_size = masks_t.shape[-1]
        terminal_size = terminals_t.shape[-1]
        confidence_channel = confidence_t.view(1, batch_size, 1, 1).expand(
            ensemble,
            batch_size,
            components,
            1,
        )
        packed = self.torch.cat(
            (
                descriptors_t,
                masks_t,
                terminals_t,
                mixtures_t.unsqueeze(-1),
                confidence_channel,
            ),
            dim=-1,
        ).detach().cpu()
        self.performance_host_transfer_batches += 1
        descriptor_end = descriptor_size
        mask_end = descriptor_end + mask_size
        terminal_end = mask_end + terminal_size
        mixture_index = terminal_end
        confidence_index = terminal_end + 1
        descriptors = packed[:, :, :, :descriptor_end].tolist()
        masks = packed[:, :, :, descriptor_end:mask_end].tolist()
        terminals = packed[:, :, :, mask_end:terminal_end].tolist()
        mixtures = packed[:, :, :, mixture_index].tolist()
        confidences = packed[0, :, 0, confidence_index].tolist()

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

    def diagnostics(self: object) -> dict[str, Any]:
        pending = getattr(self, "_performance_pending_ensemble_variance", None)
        if pending is not None:
            self._last_ensemble_variance = float(pending.cpu().item())
            self._performance_pending_ensemble_variance = None
        result = dict(original_diagnostics())
        result.update(
            {
                "performance_contract": PERFORMANCE_CONTRACT,
                "state_encode_cache_hits": self.performance_state_encode_cache_hits,
                "state_encode_cache_misses": self.performance_state_encode_cache_misses,
                "packed_host_transfer_batches": self.performance_host_transfer_batches,
                "training_metric_sync_batches": self.performance_training_metric_syncs,
                "deferred_ensemble_variance_sync": 1,
                "vectorized_status_mixture_decode": 0,
            }
        )
        return result

    # RelationalPrediction is looked up by the optimized early warmup path without
    # importing a second copy or changing its runtime type.
    from .current_relational_model import RelationalPrediction
    base.RelationalPrediction = RelationalPrediction
    base._forward = MethodType(forward, base)
    # Keep member-by-member decoding. On CUDA, combining the ensemble dimension
    # can select a different reduction shape and perturb calibration at the last
    # few bits. Packed transfer and deferred diagnostic sync remain active.
    base._confidence_from_decoded = MethodType(confidence_from_decoded, base)
    base._train_step = MethodType(train_step, base)
    base.predict_batch = MethodType(predict_batch, base)
    base.diagnostics = MethodType(diagnostics, base)


def install_current_runtime_performance(agent: object) -> object:
    """Install only schedule- and semantics-preserving runtime optimizations."""

    if getattr(agent, "current_runtime_performance", False):
        return agent
    _install_indexed_calibration(agent.calibrated_prophecy)
    _install_status_mixture_fast_path(agent.base_neural_prophecy)
    agent.current_runtime_performance = True
    agent.current_runtime_performance_contract = PERFORMANCE_CONTRACT
    return agent
