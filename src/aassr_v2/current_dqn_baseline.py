from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .action_plugins import PluginOutcome
from .current_hardware import (
    CurrentHardwareInfo,
    HardwareRelationalInvariantDQN,
    configure_current_hardware,
)
from .pentest_curriculum_causal import OBSERVATION_CONTRACT
from .types import Action, StateSnapshot, TransitionTrace


BARE_DQN_CONDITION = "dqn_bare"


@dataclass(frozen=True, slots=True)
class BareDQNStep:
    traces: tuple[TransitionTrace, ...]


@dataclass(frozen=True, slots=True)
class _PendingDQNTransition:
    before: StateSnapshot
    action: Action
    after: StateSnapshot


class BareRelationalDQNAgent:
    """DQN-only control condition for the current pentest experiment.

    The baseline shares only the current relational state/action representation,
    sparse external reward, environment, action surface and budget. It does not
    instantiate ASEQ, Knowledge, Prophecy, Imagination, Skills, the branch Critic,
    feature memory, or the information-value residual.
    """

    def __init__(
        self,
        *,
        seed: int,
        train_transitions: int,
        device: str = "cpu",
        allow_tf32: bool = True,
    ) -> None:
        self.seed = int(seed)
        self.train_transitions = int(train_transitions)
        self.hardware_info: CurrentHardwareInfo = configure_current_hardware(
            device,
            allow_tf32=allow_tf32,
        )
        self.dqn = HardwareRelationalInvariantDQN(
            self.seed ^ 0xD1A6,
            train_transitions=self.train_transitions,
            device=self.hardware_info.resolved_device,
        )
        self._pending: _PendingDQNTransition | None = None
        self._trace_index = 0
        self._steps = 0

    def _validate_snapshot(self, state: StateSnapshot) -> None:
        actual = state.metadata.get("observation_contract")
        if actual != OBSERVATION_CONTRACT:
            raise ValueError(
                "bare DQN observation contract mismatch: "
                f"expected {OBSERVATION_CONTRACT!r}, got {actual!r}"
            )

    def begin_episode(self) -> None:
        if self._pending is not None:
            raise RuntimeError("bare DQN episode began with an unflushed transition")

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
    ) -> BareDQNStep:
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
            f"dqn-bare-{self._trace_index:08d}",
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
        return BareDQNStep((trace,))

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
            "condition": BARE_DQN_CONDITION,
            "dqn_only": True,
            "representation": "same-relational-state-action-input-as-current-policy",
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


def build_bare_dqn_agent(
    *,
    seed: int,
    train_transitions: int,
    device: str = "cpu",
    allow_tf32: bool = True,
) -> BareRelationalDQNAgent:
    return BareRelationalDQNAgent(
        seed=int(seed),
        train_transitions=int(train_transitions),
        device=device,
        allow_tf32=allow_tf32,
    )
