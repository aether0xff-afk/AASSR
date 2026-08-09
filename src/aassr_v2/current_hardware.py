from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .branch_critic import BranchCriticStep, CriticTransition
from .current_generation import RelationalGRUBranchCritic, RelationalInvariantDQN
from .types import Action, StateSnapshot


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
    dqn_fused_next_action_reduce: bool = True
    imagination_depth_batching: bool = True
    critic_depth_batching: bool = True
    critic_on_shared_device: bool = True
    torch_compile_enabled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def configure_current_hardware(
    device: str,
    *,
    allow_tf32: bool = True,
) -> CurrentHardwareInfo:
    """Configure the active float32 torch path without changing the algorithm."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("current-generation hardware path requires torch") from exc

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # pragma: no cover
        torch.use_deterministic_algorithms(True)

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
        except (AttributeError, RuntimeError):  # pragma: no cover
            pass
        try:
            torch.backends.cudnn.allow_tf32 = tf32_enabled
        except (AttributeError, RuntimeError):  # pragma: no cover
            pass
        setter = getattr(torch, "set_float32_matmul_precision", None)
        if callable(setter):
            setter("high" if tf32_enabled else "highest")

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
        deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
    )


class HardwareRelationalInvariantDQN(RelationalInvariantDQN):
    """Current relational DQN with batched, sync-free Bellman target reduction."""

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
        self.optimizer = self.torch.optim.Adam(
            self.online.parameters(),
            lr=learning_rate,
        )
        self.target.eval()
        self.device_target_reductions = 0
        self.fused_target_reduce_calls = 0

    def _tensor(self, values: Any, *, dtype: Any | None = None) -> Any:
        return self.torch.as_tensor(
            values,
            dtype=dtype or self.torch.float32,
            device=self.device,
        )

    def _train_step(self) -> None:
        batch = self.randomizer.sample(list(self.replay), self.batch_size)
        inputs = self._tensor([item[0] + item[1] for item in batch])
        predicted = self.online(inputs).squeeze(1)

        flat: list[tuple[float, ...]] = []
        owners: list[int] = []
        for index, (_, _, _, next_state, next_actions, terminal) in enumerate(batch):
            if terminal or not next_actions:
                continue
            flat.extend(next_state + features for features in next_actions)
            owners.extend([index] * len(next_actions))
            self.device_target_reductions += 1

        next_values = self.torch.full(
            (len(batch),),
            float("-inf"),
            dtype=self.torch.float32,
            device=self.device,
        )
        if flat:
            with self.torch.no_grad():
                scored = self.target(self._tensor(flat)).squeeze(1)
                owner_tensor = self._tensor(owners, dtype=self.torch.int64)
                next_values.scatter_reduce_(
                    0,
                    owner_tensor,
                    scored,
                    reduce="amax",
                    include_self=True,
                )
                self.fused_target_reduce_calls += 1
        next_values = self.torch.where(
            self.torch.isfinite(next_values),
            next_values,
            self.torch.zeros_like(next_values),
        )

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
                "fused_target_reduce_calls": self.fused_target_reduce_calls,
                "per_row_target_item_syncs": 0,
                "fused_next_action_reduce": 1,
                "hardware_optimized": 1,
            }
        )
        return stats


class HardwareRelationalGRUBranchCritic(RelationalGRUBranchCritic):
    """Relational GRU Critic with one batched score call per tree depth."""

    name = "hardware-relational-gru-branch-critic"

    def __init__(self, seed: int, *, device: str = "cpu") -> None:
        super().__init__(seed)
        self.device = self.torch.device(device)
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        self.gru.to(self.device)
        self.output.to(self.device)
        self.optimizer = self.torch.optim.Adam(
            tuple(self.gru.parameters()) + tuple(self.output.parameters()),
            lr=learning_rate,
        )
        self.scalar_score_calls = 0
        self.batch_score_calls = 0
        self.batch_score_rows = 0

    def _tensor(self, values: Any) -> Any:
        return self.torch.as_tensor(
            values,
            dtype=self.torch.float32,
            device=self.device,
        )

    def initial_memory(self) -> Any:
        return self.torch.zeros(
            (1, self.hidden_units),
            dtype=self.torch.float32,
            device=self.device,
        )

    def _episode_loss(
        self,
        encoded: Sequence[tuple[float, ...]],
        target: float,
    ) -> Any:
        hidden = self.initial_memory()
        logits = []
        for item in encoded:
            hidden = self.gru(self._tensor(item).unsqueeze(0), hidden)
            logits.append(self.output(hidden)[0, 0])
        stacked = self.torch.stack(logits)
        targets = self.torch.full_like(stacked, target)
        return self.nn.functional.binary_cross_entropy_with_logits(stacked, targets)

    def score_step(
        self,
        before: StateSnapshot,
        action: Action,
        after: StateSnapshot,
        *,
        memory: Any = None,
        prophecy_confidence: float = 1.0,
    ) -> BranchCriticStep:
        self.scalar_score_calls += 1
        encoded = self.encoder.encode(
            CriticTransition(before, action, after, prophecy_confidence)
        )
        hidden = self.initial_memory() if memory is None else memory.to(self.device)
        with self.torch.no_grad():
            next_hidden = self.gru(self._tensor(encoded).unsqueeze(0), hidden)
            value = float(
                self.torch.sigmoid(self.output(next_hidden)[0, 0]).detach().cpu().item()
            )
        return BranchCriticStep(value, next_hidden.detach().clone())

    def score_step_batch(
        self,
        befores: Sequence[StateSnapshot],
        actions: Sequence[Action],
        afters: Sequence[StateSnapshot],
        memories: Sequence[Any],
        prophecy_confidences: Sequence[float],
    ) -> tuple[BranchCriticStep, ...]:
        length = len(befores)
        if not (
            length == len(actions)
            == len(afters)
            == len(memories)
            == len(prophecy_confidences)
        ):
            raise ValueError("critic batch inputs have different lengths")
        if not length:
            return ()

        encoded = tuple(
            self.encoder.encode(
                CriticTransition(before, action, after, confidence)
            )
            for before, action, after, confidence in zip(
                befores,
                actions,
                afters,
                prophecy_confidences,
                strict=True,
            )
        )
        hidden_rows = [
            self.initial_memory() if memory is None else memory.to(self.device)
            for memory in memories
        ]
        hidden = self.torch.cat(hidden_rows, dim=0)
        with self.torch.no_grad():
            next_hidden = self.gru(self._tensor(encoded), hidden)
            values = self.torch.sigmoid(self.output(next_hidden).squeeze(1))
            host_values = values.detach().cpu().tolist()
        self.batch_score_calls += 1
        self.batch_score_rows += length
        return tuple(
            BranchCriticStep(
                float(value),
                next_hidden[index : index + 1].detach().clone(),
            )
            for index, value in enumerate(host_values)
        )

    def hardware_stats(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "device_type": self.device.type,
            "hardware_optimized": 1,
            "scalar_score_calls": self.scalar_score_calls,
            "batch_score_calls": self.batch_score_calls,
            "batch_score_rows": self.batch_score_rows,
        }


def install_hardware_dqn(
    agent: object,
    *,
    seed: int,
    train_transitions: int,
    device: str,
    allow_tf32: bool = True,
) -> CurrentHardwareInfo:
    """Install shared-device Policy DQN and batched branch Critic for AASSR."""

    from .current_generation import CurrentRelationalPolicy

    info = configure_current_hardware(device, allow_tf32=allow_tf32)
    dqn = HardwareRelationalInvariantDQN(
        int(seed) ^ 0xD1A6,
        train_transitions=int(train_transitions),
        device=info.resolved_device,
    )
    policy = CurrentRelationalPolicy(dqn)
    critic = HardwareRelationalGRUBranchCritic(
        int(seed) ^ 0x43524954,
        device=info.resolved_device,
    )
    agent.dqn = dqn
    agent.policy = policy
    agent.critic = critic
    agent.planner.policy = policy
    agent.planner.scorer = critic
    agent.core.policy = policy
    agent.hardware_info = info
    return info


def hardware_diagnostics(agent: object) -> dict[str, Any]:
    info = getattr(agent, "hardware_info", None)
    output = info.as_dict() if isinstance(info, CurrentHardwareInfo) else {}
    dqn_stats = getattr(getattr(agent, "dqn", None), "model_stats", None)
    if callable(dqn_stats):
        output["dqn"] = dict(dqn_stats())
    prophecy_diagnostics = getattr(
        getattr(agent, "base_neural_prophecy", None),
        "diagnostics",
        None,
    )
    if callable(prophecy_diagnostics):
        output["prophecy"] = dict(prophecy_diagnostics())
    critic_hardware = getattr(getattr(agent, "critic", None), "hardware_stats", None)
    if callable(critic_hardware):
        output["critic"] = dict(critic_hardware())
    planner_diagnostics = getattr(getattr(agent, "planner", None), "runtime_diagnostics", None)
    if callable(planner_diagnostics):
        output["planner"] = dict(planner_diagnostics())
    output["depth_batching"] = bool(getattr(agent, "current_depth_batching", False))
    return output
