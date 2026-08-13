from __future__ import annotations

from types import MethodType
from typing import Any, Sequence

from .current_generation import relational_action_key
from .current_relational_state import relational_state_vector_v2


PERFORMANCE_V2_CONTRACT = "semantics-preserving-ensemble-inference+critic-prepack-v2"
_LINEAR_POSITIONS = (0, 2, 4)


def _validate_fused_models(models: Sequence[Any]) -> None:
    if not models:
        raise ValueError("fused ensemble requires at least one model")
    for model in models:
        if len(model) != 5:
            raise RuntimeError("fused ensemble expects Linear/ReLU/Linear/ReLU/Linear")
        if tuple(
            index for index in _LINEAR_POSITIONS if hasattr(model[index], "weight")
        ) != _LINEAR_POSITIONS:
            raise RuntimeError("fused ensemble model layout drift")


def _stack_ensemble_parameters(
    torch: Any,
    models: Sequence[Any],
) -> tuple[tuple[Any, Any], ...]:
    """Materialize one inference-only ensemble parameter pack."""

    _validate_fused_models(models)
    return tuple(
        (
            torch.stack([model[index].weight for model in models], dim=0).detach(),
            torch.stack([model[index].bias for model in models], dim=0).detach(),
        )
        for index in _LINEAR_POSITIONS
    )


def _stacked_linear_forward_prepacked(
    torch: Any,
    parameters: Sequence[tuple[Any, Any]],
    tensor: Any,
) -> Any:
    """Evaluate `[ensemble, batch, input]` using already stacked parameters."""

    ensemble = int(parameters[0][0].shape[0])
    value = tensor.unsqueeze(0).expand(ensemble, -1, -1)
    for position, (weights, biases) in enumerate(parameters):
        value = torch.bmm(value, weights.transpose(1, 2)) + biases.unsqueeze(1)
        if position + 1 < len(parameters):
            value = torch.relu(value)
    return value


def _stacked_linear_forward(
    torch: Any,
    models: Sequence[Any],
    tensor: Any,
) -> Any:
    """Reference helper for one fused forward with a freshly packed ensemble.

    Runtime inference uses a revision-aware cached parameter pack so repeated
    Imagination calls do not re-copy every ensemble weight. Parameters remain
    owned by the original independent modules/optimizers.
    """

    parameters = _stack_ensemble_parameters(torch, models)
    return _stacked_linear_forward_prepacked(torch, parameters, tensor)


def _parameter_revision(models: Sequence[Any], gradient_updates: int) -> tuple[Any, ...]:
    """Detect optimizer steps and explicit state-dict/in-place parameter updates."""

    versions = tuple(
        int(getattr(parameter, "_version", 0))
        for model in models
        for parameter in model.parameters()
    )
    return (int(gradient_updates), *versions)


def install_fused_prophecy_ensemble_inference(base: object) -> object:
    """Replace sequential ensemble inference with cached ensemble-dimension GEMMs."""

    if getattr(base, "performance_fused_ensemble_inference", False):
        return base

    _validate_fused_models(base.models)
    original_diagnostics = base.diagnostics
    base.performance_fused_ensemble_forward_calls = 0
    base.performance_fused_ensemble_forward_rows = 0
    base.performance_fused_ensemble_pack_rebuilds = 0
    base.performance_fused_ensemble_pack_hits = 0
    base._performance_fused_ensemble_revision = None
    base._performance_fused_ensemble_parameters = None

    def fused_parameters(self: object) -> tuple[tuple[Any, Any], ...]:
        revision = _parameter_revision(self.models, self.gradient_updates)
        if (
            self._performance_fused_ensemble_parameters is None
            or revision != self._performance_fused_ensemble_revision
        ):
            with self.torch.no_grad():
                self._performance_fused_ensemble_parameters = _stack_ensemble_parameters(
                    self.torch,
                    self.models,
                )
            self._performance_fused_ensemble_revision = revision
            self.performance_fused_ensemble_pack_rebuilds += 1
        else:
            self.performance_fused_ensemble_pack_hits += 1
        return self._performance_fused_ensemble_parameters

    def fused_forward(
        self: object,
        states: Sequence[Any],
        actions: Sequence[Any],
    ) -> Any:
        encoded: dict[int, tuple[Any, tuple[float, ...]]] = {}
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
                if hasattr(self, "performance_state_encode_cache_misses"):
                    self.performance_state_encode_cache_misses += 1
            else:
                if hasattr(self, "performance_state_encode_cache_hits"):
                    self.performance_state_encode_cache_hits += 1
            values = cached[1] + action_structure(state, action)
            if len(values) != self.input_size:
                raise ValueError("relational mixture input size drift")
            inputs.append(values)

        tensor = self._tensor(inputs)
        with self.torch.no_grad():
            output = _stacked_linear_forward_prepacked(
                self.torch,
                self._fused_ensemble_parameters(),
                tensor,
            )
        self.performance_fused_ensemble_forward_calls += 1
        self.performance_fused_ensemble_forward_rows += len(states)
        return output

    def diagnostics(self: object) -> dict[str, Any]:
        result = dict(original_diagnostics())
        result.update(
            {
                "performance_v2_contract": PERFORMANCE_V2_CONTRACT,
                "fused_ensemble_inference": 1,
                "fused_ensemble_forward_calls": self.performance_fused_ensemble_forward_calls,
                "fused_ensemble_forward_rows": self.performance_fused_ensemble_forward_rows,
                "fused_ensemble_pack_rebuilds": self.performance_fused_ensemble_pack_rebuilds,
                "fused_ensemble_pack_hits": self.performance_fused_ensemble_pack_hits,
            }
        )
        return result

    base._fused_ensemble_parameters = MethodType(fused_parameters, base)
    base._forward = MethodType(fused_forward, base)
    base.diagnostics = MethodType(diagnostics, base)
    base.performance_fused_ensemble_inference = True
    return base


def install_prepacked_critic_training(critic: object) -> object:
    """Build padded Critic tensors once per update instead of once per time step."""

    if getattr(critic, "performance_prepacked_sequence_training", False):
        return critic

    critic.performance_prepacked_sequence_training = True
    critic.performance_prepacked_sequence_batches = 0
    critic.performance_prepacked_sequence_rows = 0

    def train_step(self: object) -> None:
        revision = int(self.suffix_sequences)
        if self._performance_replay_snapshot_revision != revision:
            self._performance_replay_snapshot = tuple(self.replay)
            self._performance_replay_snapshot_revision = revision
            self.performance_full_replay_copies += 1

        batch = self.randomizer.sample(
            self._performance_replay_snapshot,
            self.batch_size,
        )
        lengths = [len(encoded) for encoded, _ in batch]
        max_length = max(lengths)
        zero_features = (0.0,) * self.encoder.feature_size

        feature_rows = [
            list(encoded) + [zero_features] * (max_length - len(encoded))
            for encoded, _ in batch
        ]
        target_rows = [
            list(targets) + [0.0] * (max_length - len(targets))
            for _, targets in batch
        ]
        mask_rows = [
            [1.0] * length + [0.0] * (max_length - length)
            for length in lengths
        ]

        features = self._tensor(feature_rows)
        targets = self._tensor(target_rows)
        masks = self._tensor(mask_rows)
        lengths_tensor = self._tensor(lengths)
        hidden = self.torch.zeros(
            (len(batch), self.hidden_units),
            dtype=self.torch.float32,
            device=self.device,
        )
        loss_sums = self.torch.zeros(
            len(batch),
            dtype=self.torch.float32,
            device=self.device,
        )

        # GRUCell recurrence is intentionally retained: replacing it with another
        # recurrent formulation would be a scientific/numerical change. Only CPU
        # list construction and host->device tensor creation leave this loop.
        for step_index in range(max_length):
            hidden = self.gru(features[:, step_index, :], hidden)
            predicted = self.torch.tanh(self.output(hidden).squeeze(1))
            per_row = self.nn.functional.smooth_l1_loss(
                predicted,
                targets[:, step_index],
                reduction="none",
            )
            loss_sums = loss_sums + per_row * masks[:, step_index]

        loss = (loss_sums / lengths_tensor).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.nn.utils.clip_grad_norm_(
            tuple(self.gru.parameters()) + tuple(self.output.parameters()),
            5.0,
        )
        self.optimizer.step()
        self.gradient_updates += 1
        self._losses.append(float(loss.detach().cpu().item()))
        self.train_batch_calls += 1
        self.train_batch_time_steps += max_length
        self.train_batch_transition_rows += sum(lengths)
        self.performance_prepacked_sequence_batches += 1
        self.performance_prepacked_sequence_rows += sum(lengths)

    critic._train_step = MethodType(train_step, critic)
    return critic


def install_current_runtime_performance_v2(
    agent: object,
    *,
    enable_cuda_fusion: bool,
) -> object:
    """Install second-stage fast paths without changing the scientific contract."""

    if getattr(agent, "current_runtime_performance_v2", False):
        return agent

    if enable_cuda_fusion:
        install_fused_prophecy_ensemble_inference(agent.base_neural_prophecy)
    install_prepacked_critic_training(agent.critic)

    agent.current_runtime_performance_v2 = True
    agent.current_runtime_performance_v2_contract = PERFORMANCE_V2_CONTRACT
    agent.current_runtime_fused_ensemble = bool(enable_cuda_fusion)
    agent.current_runtime_prepacked_critic = True
    return agent
