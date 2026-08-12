from __future__ import annotations

from types import MethodType
from typing import Any, Sequence

from .current_generation import relational_action_key
from .current_relational_state import relational_state_vector_v2


PERFORMANCE_V2_CONTRACT = "semantics-preserving-ensemble-inference+critic-prepack-v2"


def _stacked_linear_forward(
    torch: Any,
    models: Sequence[Any],
    tensor: Any,
) -> Any:
    """Evaluate equal-shape 3-layer MLP ensemble as three batched GEMMs.

    Parameters remain owned by the original independent modules/optimizers. This
    path is inference-only, so training order and optimizer semantics are exactly
    unchanged. The only numerical change permitted by the scaling gate is normal
    floating-point kernel roundoff within the existing 1e-5 contract.
    """

    if not models:
        raise ValueError("fused ensemble requires at least one model")
    expected = (0, 2, 4)
    for model in models:
        if len(model) != 5:
            raise RuntimeError("fused ensemble expects Linear/ReLU/Linear/ReLU/Linear")
        if tuple(index for index in expected if hasattr(model[index], "weight")) != expected:
            raise RuntimeError("fused ensemble model layout drift")

    # [B, I] -> [E, B, I], then one batch matmul per linear layer.
    value = tensor.unsqueeze(0).expand(len(models), -1, -1)
    for position, layer_index in enumerate(expected):
        weights = torch.stack(
            [model[layer_index].weight for model in models],
            dim=0,
        )
        biases = torch.stack(
            [model[layer_index].bias for model in models],
            dim=0,
        )
        value = torch.bmm(value, weights.transpose(1, 2)) + biases.unsqueeze(1)
        if position + 1 < len(expected):
            value = torch.relu(value)
    return value


def install_fused_prophecy_ensemble_inference(base: object) -> object:
    """Replace sequential ensemble inference with ensemble-dimension GEMMs."""

    if getattr(base, "performance_fused_ensemble_inference", False):
        return base

    original_diagnostics = base.diagnostics
    base.performance_fused_ensemble_forward_calls = 0
    base.performance_fused_ensemble_forward_rows = 0

    def fused_forward(
        self: object,
        states: Sequence[Any],
        actions: Sequence[Any],
    ) -> Any:
        encoded: dict[int, tuple[Any, tuple[float, ...]]] = {}
        inputs: list[tuple[float, ...]] = []
        for state, action in zip(states, actions, strict=True):
            identity = id(state)
            cached = encoded.get(identity)
            if cached is None or cached[0] is not state:
                cached = (state, relational_state_vector_v2(state))
                encoded[identity] = cached
                if hasattr(self, "performance_state_encode_cache_misses"):
                    self.performance_state_encode_cache_misses += 1
            else:
                if hasattr(self, "performance_state_encode_cache_hits"):
                    self.performance_state_encode_cache_hits += 1
            values = cached[1] + relational_action_key(state, action)
            if len(values) != self.input_size:
                raise ValueError("relational mixture input size drift")
            inputs.append(values)

        tensor = self._tensor(inputs)
        with self.torch.no_grad():
            output = _stacked_linear_forward(self.torch, self.models, tensor)
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
            }
        )
        return result

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
