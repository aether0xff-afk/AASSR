from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .current_generation import RelationalInvariantDQN


@dataclass(frozen=True, slots=True)
class CurrentHardwareInfo:
    requested_device: str
    resolved_device: str
    device_type: str
    cuda_available: bool
    cuda_device_name: str | None
    cuda_capability: tuple[int, int] | None
    torch_version: str
    float_dtype: str
    tf32_enabled: bool
    deterministic_algorithms: bool
    dqn_gpu_sync_free_targets: bool = True
    imagination_depth_batching: bool = True
    torch_compile_enabled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def configure_current_hardware(
    device: str,
    *,
    allow_tf32: bool = True,
) -> CurrentHardwareInfo:
    """Configure the active float32 torch path without changing the algorithm.

    TF32 is a local hardware execution option only. The experiment artifact records
    whether it was enabled. `torch.compile` is deliberately not enabled here: the
    current workload has dynamic action cardinality and short online updates, for
    which compile/recompile overhead can dominate and complicate Windows runs.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("current-generation hardware path requires torch") from exc

    resolved = torch.device(device)
    cuda_available = bool(torch.cuda.is_available())
    device_name: str | None = None
    capability: tuple[int, int] | None = None
    tf32_enabled = False

    if resolved.type == "cuda":
        if not cuda_available:
            raise RuntimeError(
                f"CUDA device {device!r} requested but torch.cuda.is_available() is false"
            )
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(index)
        resolved = torch.device("cuda", index)
        device_name = torch.cuda.get_device_name(index)
        capability = tuple(int(value) for value in torch.cuda.get_device_capability(index))
        tf32_enabled = bool(allow_tf32)
        try:
            torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        except (AttributeError, RuntimeError):  # pragma: no cover - torch-version dependent
            pass
        try:
            torch.backends.cudnn.allow_tf32 = tf32_enabled
        except (AttributeError, RuntimeError):  # pragma: no cover
            pass
        setter = getattr(torch, "set_float32_matmul_precision", None)
        if callable(setter):
            setter("high" if tf32_enabled else "highest")

    deterministic = bool(torch.are_deterministic_algorithms_enabled())
    return CurrentHardwareInfo(
        requested_device=str(device),
        resolved_device=str(resolved),
        device_type=resolved.type,
        cuda_available=cuda_available,
        cuda_device_name=device_name,
        cuda_capability=capability,
        torch_version=str(torch.__version__),
        float_dtype="float32",
        tf32_enabled=tf32_enabled,
        deterministic_algorithms=deterministic,
    )


class HardwareRelationalInvariantDQN(RelationalInvariantDQN):
    """The current relational DQN on an explicit torch device.

    The historical relational DQN was CPU-only. Moving it naively to CUDA also
    introduced one `.item()` synchronization per replay row while reducing the
    next-action set. This implementation keeps the exact Bellman target but keeps
    those maxima as device tensors until the single loss computation.
    """

    name = "hardware-relational-invariant-dqn"

    def __init__(
        self,
        seed: int,
        *,
        train_transitions: int,
        device: str = "cpu",
    ) -> None:
        super().__init__(seed, train_transitions=train_transitions)
        self.device = self.torch.device(device)
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        self.online.to(self.device)
        self.target.to(self.device)
        # No optimizer step occurs before this replacement. Rebuilding here avoids
        # carrying any optimizer tensor on the construction-time CPU device.
        self.optimizer = self.torch.optim.Adam(
            self.online.parameters(),
            lr=learning_rate,
        )
        self.target.eval()
        self.device_target_reductions = 0

    def _tensor(self, values: Any) -> Any:
        return self.torch.as_tensor(
            values,
            dtype=self.torch.float32,
            device=self.device,
        )

    def _train_step(self) -> None:
        batch = self.randomizer.sample(list(self.replay), self.batch_size)
        inputs = self._tensor([item[0] + item[1] for item in batch])
        predicted = self.online(inputs).squeeze(1)

        flat: list[tuple[float, ...]] = []
        spans: list[tuple[int, int, int]] = []
        for index, (_, _, _, next_state, next_actions, terminal) in enumerate(batch):
            if terminal or not next_actions:
                continue
            start = len(flat)
            flat.extend(next_state + features for features in next_actions)
            spans.append((index, start, len(flat)))

        next_values = self.torch.zeros(
            len(batch),
            dtype=self.torch.float32,
            device=self.device,
        )
        if flat:
            with self.torch.no_grad():
                scored = self.target(self._tensor(flat)).squeeze(1)
                # Tensor-to-tensor assignment stays on device. In particular, do
                # not call .item() once per replay row on CUDA.
                for index, start, end in spans:
                    next_values[index] = scored[start:end].max()
                    self.device_target_reductions += 1

        rewards = self._tensor([item[2] for item in batch])
        terminals = self._tensor([float(item[5]) for item in batch])
        targets = rewards + self.gamma * (1.0 - terminals) * next_values
        loss = self.loss(predicted, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.gradient_updates += 1
        if self.gradient_updates % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

    def model_stats(self) -> dict[str, int | float | str]:
        stats = dict(super().model_stats())
        stats.update(
            {
                "device": str(self.device),
                "device_type": self.device.type,
                "device_target_reductions": self.device_target_reductions,
                "per_row_target_item_syncs": 0,
                "hardware_optimized": 1,
            }
        )
        return stats


def install_hardware_dqn(
    agent: object,
    *,
    seed: int,
    train_transitions: int,
    device: str,
    allow_tf32: bool = True,
) -> CurrentHardwareInfo:
    """Install the same device-aware DQN used by AASSR and the bare baseline."""

    from .current_generation import CurrentRelationalPolicy

    info = configure_current_hardware(device, allow_tf32=allow_tf32)
    dqn = HardwareRelationalInvariantDQN(
        int(seed) ^ 0xD1A6,
        train_transitions=int(train_transitions),
        device=info.resolved_device,
    )
    policy = CurrentRelationalPolicy(dqn)
    agent.dqn = dqn
    agent.policy = policy
    agent.planner.policy = policy
    agent.core.policy = policy
    agent.hardware_info = info
    return info


def hardware_diagnostics(agent: object) -> dict[str, Any]:
    info = getattr(agent, "hardware_info", None)
    if isinstance(info, CurrentHardwareInfo):
        output = info.as_dict()
    else:
        output = {}
    dqn = getattr(agent, "dqn", None)
    dqn_stats = getattr(dqn, "model_stats", None)
    if callable(dqn_stats):
        output["dqn"] = dict(dqn_stats())
    prophecy = getattr(agent, "base_neural_prophecy", None)
    prophecy_diagnostics = getattr(prophecy, "diagnostics", None)
    if callable(prophecy_diagnostics):
        output["prophecy"] = dict(prophecy_diagnostics())
    output["depth_batching"] = bool(getattr(agent, "current_depth_batching", False))
    return output
