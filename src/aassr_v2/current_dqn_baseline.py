from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .action_plugins import PluginOutcome
from .current_hardware import (
    CurrentHardwareInfo,
    HardwareRelationalInvariantDQN,
    configure_current_hardware,
)
from .current_relational_state_v3 import install_status_aware_relational_contract
from .pentest_agent_main_test import DynamicActionDQN, action_features
from .pentest_curriculum_causal import OBSERVATION_CONTRACT
from .types import Action, StateSnapshot, TransitionTrace


RAW_DQN_CONDITION = "dqn_raw"
RELATIONAL_DQN_CONDITION = "dqn_relational"
# Compatibility alias for the first implementation of this control. New current
# experiments must report the explicit raw/relational names above.
BARE_DQN_CONDITION = RELATIONAL_DQN_CONDITION


class HardwareRawDynamicActionDQN(DynamicActionDQN):
    """True raw-input DQN with the same corrected/hardware execution contract.

    State input is the v3 snapshot vector exactly as exposed by the environment;
    action input is the historical stable raw-signature hash from `action_features`.
    No relational role abstraction is supplied. Known TD boundary and CUDA host-
    sync bugs are still corrected because a baseline should be plain, not broken.
    """

    name = "hardware-raw-dynamic-action-dqn"

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
        self._episode_boundary_pending = False
        self.forced_episode_boundaries = 0
        self.fused_target_reduce_calls = 0
        self.device_target_reductions = 0

    def _tensor(self, values: Any, *, dtype: Any | None = None) -> Any:
        return self.torch.as_tensor(
            values,
            dtype=dtype or self.torch.float32,
            device=self.device,
        )

    def mark_episode_boundary(self) -> None:
        self._episode_boundary_pending = True

    def _consume_episode_boundary(self) -> bool:
        boundary = self._episode_boundary_pending
        self._episode_boundary_pending = False
        if boundary:
            self.forced_episode_boundaries += 1
        return boundary

    def observe(
        self,
        before: StateSnapshot,
        action: Action,
        outcome: PluginOutcome,
        *,
        reward: float,
    ) -> None:
        terminal = self._consume_episode_boundary() or not outcome.snapshot.available_actions
        next_actions = tuple(
            action_features(item) for item in outcome.snapshot.available_actions
        )
        self.replay.append(
            (
                self.encode_state(before),
                action_features(action),
                float(reward),
                self.encode_state(outcome.snapshot),
                next_actions,
                terminal,
            )
        )
        self.environment_steps += 1
        if len(self.replay) >= max(self.batch_size, self.warmup_steps):
            self._train_step()

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
                "forced_episode_boundaries": self.forced_episode_boundaries,
                "device_target_reductions": self.device_target_reductions,
                "fused_target_reduce_calls": self.fused_target_reduce_calls,
                "per_row_target_item_syncs": 0,
                "fused_next_action_reduce": 1,
                "hardware_optimized": 1,
                "representation": "raw-v3-vector+raw-signature-action-hash",
            }
        )
        return stats


@dataclass(frozen=True, slots=True)
class DQNOnlyStep:
    traces: tuple[TransitionTrace, ...]


@dataclass(frozen=True, slots=True)
class _PendingDQNTransition:
    before: StateSnapshot
    action: Action
    after: StateSnapshot


class DQNOnlyAgent:
    """Minimal current-protocol wrapper around one DQN implementation."""

    def __init__(
        self,
        dqn: object,
        *,
        condition: str,
        representation: str,
        train_transitions: int,
        hardware_info: CurrentHardwareInfo,
    ) -> None:
        self.dqn = dqn
        self.condition = str(condition)
        self.representation = str(representation)
        self.train_transitions = int(train_transitions)
        self.hardware_info = hardware_info
        self._pending: _PendingDQNTransition | None = None
        self._trace_index = 0
        self._steps = 0

    def _validate_snapshot(self, state: StateSnapshot) -> None:
        actual = state.metadata.get("observation_contract")
        if actual != OBSERVATION_CONTRACT:
            raise ValueError(
                f"{self.condition} observation contract mismatch: "
                f"expected {OBSERVATION_CONTRACT!r}, got {actual!r}"
            )

    def begin_episode(self) -> None:
        if self._pending is not None:
            raise RuntimeError(
                f"{self.condition} episode began with an unflushed transition"
            )

    def _observe_pending(self, *, reward: float, terminal: bool) -> None:
        if self._pending is None:
            return
        if terminal:
            self.dqn.mark_episode_boundary()
        transition = self._pending
        self.dqn.observe(
            transition.before,
            transition.action,
            PluginOutcome(snapshot=transition.after),
            reward=float(reward),
        )
        self._pending = None

    def _normalized_transition(self, exploration_index: int) -> int:
        fraction = min(1.0, max(0.0, float(exploration_index) / 1000.0))
        return int(round(fraction * self.train_transitions))

    def step(
        self,
        environment: object,
        *,
        episode: int,
        training: bool = True,
        primitive_budget: int | None = None,
    ) -> DQNOnlyStep:
        if primitive_budget is not None and primitive_budget <= 0:
            raise ValueError("primitive_budget must be positive when supplied")
        if training:
            self._observe_pending(reward=0.0, terminal=False)

        before = environment.snapshot()
        self._validate_snapshot(before)
        decision = self.dqn.select_action(
            before,
            transition=self._normalized_transition(episode),
            training=training,
        )
        outcome = environment.step(decision.action)
        after = outcome.snapshot
        self._validate_snapshot(after)
        self._trace_index += 1
        trace = TransitionTrace(
            f"{self.condition}-{self._trace_index:08d}",
            before,
            decision.action,
            (),
            after,
            frozenset(getattr(outcome, "added_facts", after.facts - before.facts)),
            frozenset(getattr(outcome, "removed_facts", before.facts - after.facts)),
            tuple(getattr(outcome, "unlocked_actions", ())),
            bool(getattr(outcome, "error", False)),
            real_reward=0.0,
        )
        if training:
            self._pending = _PendingDQNTransition(before, decision.action, after)
        self._steps += 1
        return DQNOnlyStep((trace,))

    def finish_episode(
        self,
        *,
        final_return: float,
        training: bool = True,
    ) -> None:
        if training:
            self._observe_pending(
                reward=float(final_return),
                terminal=True,
            )
        else:
            self._pending = None

    def learning_counters(self) -> tuple[int, int, int]:
        return (
            int(self.dqn.environment_steps),
            int(self.dqn.gradient_updates),
            len(self.dqn.replay),
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "dqn_only": True,
            "representation": self.representation,
            "observation_contract": OBSERVATION_CONTRACT,
            "steps": self._steps,
            "aseq": {
                "guard_events": 0,
                "all_guarded_fallbacks": 0,
            },
            "imagination": {
                "runs": 0,
                "interventions": 0,
                "changed_actions": 0,
            },
            "skill_uses": 0,
            "promoted_skills": 0,
            "modules_absent": [
                "aseq",
                "knowledge",
                "prophecy",
                "imagination",
                "skills",
                "branch_critic",
                "feature_memory",
                "information_value_residual",
            ],
            "policy": {
                f"dqn:{key}": value
                for key, value in self.dqn.model_stats().items()
            },
            "hardware": {
                **self.hardware_info.as_dict(),
                "dqn": dict(self.dqn.model_stats()),
            },
        }


class BareRelationalDQNAgent(DQNOnlyAgent):
    pass


class RawDQNAgent(DQNOnlyAgent):
    pass


def build_relational_dqn_agent(
    *,
    seed: int,
    train_transitions: int,
    device: str = "cpu",
    allow_tf32: bool = True,
) -> BareRelationalDQNAgent:
    # The relational baseline must be self-contained.  Do not rely on an AASSR
    # builder or a pytest import side effect to install the active public v3
    # representation first; a fresh experiment process trains this baseline
    # before AASSR is constructed.
    install_status_aware_relational_contract()
    hardware_info = configure_current_hardware(device, allow_tf32=allow_tf32)
    dqn = HardwareRelationalInvariantDQN(
        int(seed) ^ 0xD1A6,
        train_transitions=int(train_transitions),
        device=hardware_info.resolved_device,
    )
    return BareRelationalDQNAgent(
        dqn,
        condition=RELATIONAL_DQN_CONDITION,
        representation="current-relational-public-state-v3+latest-http-status",
        train_transitions=int(train_transitions),
        hardware_info=hardware_info,
    )


def build_raw_dqn_agent(
    *,
    seed: int,
    train_transitions: int,
    device: str = "cpu",
    allow_tf32: bool = True,
) -> RawDQNAgent:
    hardware_info = configure_current_hardware(device, allow_tf32=allow_tf32)
    dqn = HardwareRawDynamicActionDQN(
        int(seed) ^ 0xD1A6,
        train_transitions=int(train_transitions),
        device=hardware_info.resolved_device,
    )
    return RawDQNAgent(
        dqn,
        condition=RAW_DQN_CONDITION,
        representation="raw-v3-vector+raw-signature-action-hash",
        train_transitions=int(train_transitions),
        hardware_info=hardware_info,
    )


def build_bare_dqn_agent(
    *,
    seed: int,
    train_transitions: int,
    device: str = "cpu",
    allow_tf32: bool = True,
) -> BareRelationalDQNAgent:
    """Compatibility alias for the relational DQN control."""

    return build_relational_dqn_agent(
        seed=seed,
        train_transitions=train_transitions,
        device=device,
        allow_tf32=allow_tf32,
    )
