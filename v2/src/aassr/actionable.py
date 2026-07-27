from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import StrEnum
from math import log1p
from typing import Any

from .gridworld import ActionCandidate, ActionName, GridWorldDMP, StepResult
from .knowledge import KK, KnowledgeDelta, KnowledgeStatus
from .reward import RewardBreakdown


class ActionErrorKind(StrEnum):
    NONE = "none"
    BLOCKED_PATH = "blocked_path"
    INVALID_BINDING = "invalid_binding"
    EXECUTION_FAILURE = "execution_failure"


@dataclass(frozen=True)
class ActionableRewardConfig:
    """Goal-agnostic reward shaping based on consequences, not hand-authored plans."""

    flag_reward: float = 1.0
    unlock_weight: float = 0.03
    lifecycle_weight: float = 0.04
    minor_information_weight: float = 0.005
    information_cap: int = 3
    blocked_path_penalty: float = -0.02
    invalid_binding_penalty: float = -0.08
    execution_failure_penalty: float = -0.05
    semantic_repeat_penalty: float = -0.05
    cycle_penalty: float = -0.03
    repeated_error_multiplier: float = 1.5


@dataclass(frozen=True)
class ActionableRewardDiagnostics:
    newly_unlocked_actions: int = 0
    newly_locked_actions: int = 0
    semantic_change_count: int = 0
    lifecycle_progress_count: int = 0
    semantic_repeat: bool = False
    cycle_repeat: bool = False
    error_kind: ActionErrorKind = ActionErrorKind.NONE
    unlock_reward: float = 0.0
    lifecycle_reward: float = 0.0
    information_reward: float = 0.0
    error_adjustment: float = 0.0
    repeat_adjustment: float = 0.0
    cycle_adjustment: float = 0.0


_BOOKKEEPING_KK = {
    KK.CURRENT_POS,
    KK.DIRECTION,
    KK.SELF,
    KK.KNOWN_CELL,
    KK.VISITED_CELL,
    KK.UNKNOWN_NEIGHBOR,
    KK.FRONTIER_CELL,
}


class ActionableRewardModule:
    """Rewards reusable capability gains while avoiding target-specific KK weights."""

    def __init__(self, config: ActionableRewardConfig | None = None) -> None:
        self.config = config or ActionableRewardConfig()
        self.last_diagnostics = ActionableRewardDiagnostics()

    def compute(
        self,
        *,
        delta_k: KnowledgeDelta,
        newly_unlocked_actions: int,
        newly_locked_actions: int,
        error_kind: ActionErrorKind,
        repeated: bool,
        semantic_repeat: bool,
        cycle_repeat: bool,
        flag_found: bool,
    ) -> RewardBreakdown:
        semantic_change_count = _semantic_change_count(delta_k)
        lifecycle_progress_count = _lifecycle_progress_count(delta_k)

        unlock_reward = self.config.unlock_weight * log1p(max(0, newly_unlocked_actions))
        lifecycle_reward = self.config.lifecycle_weight * lifecycle_progress_count
        information_reward = self.config.minor_information_weight * min(
            semantic_change_count,
            self.config.information_cap,
        )

        error_adjustment = self._error_adjustment(error_kind, repeated=repeated)
        repeat_adjustment = self.config.semantic_repeat_penalty if semantic_repeat else 0.0
        cycle_adjustment = self.config.cycle_penalty if cycle_repeat else 0.0

        external = self.config.flag_reward if flag_found else 0.0
        intrinsic = (
            unlock_reward
            + lifecycle_reward
            + information_reward
            + error_adjustment
            + repeat_adjustment
            + cycle_adjustment
        )
        self.last_diagnostics = ActionableRewardDiagnostics(
            newly_unlocked_actions=newly_unlocked_actions,
            newly_locked_actions=newly_locked_actions,
            semantic_change_count=semantic_change_count,
            lifecycle_progress_count=lifecycle_progress_count,
            semantic_repeat=semantic_repeat,
            cycle_repeat=cycle_repeat,
            error_kind=error_kind,
            unlock_reward=unlock_reward,
            lifecycle_reward=lifecycle_reward,
            information_reward=information_reward,
            error_adjustment=error_adjustment,
            repeat_adjustment=repeat_adjustment,
            cycle_adjustment=cycle_adjustment,
        )
        return RewardBreakdown(
            external_reward=external,
            intrinsic_reward=intrinsic,
            total_reward=external + intrinsic,
        )

    def _error_adjustment(self, error_kind: ActionErrorKind, *, repeated: bool) -> float:
        if error_kind == ActionErrorKind.NONE:
            return 0.0
        if error_kind == ActionErrorKind.BLOCKED_PATH:
            value = self.config.blocked_path_penalty
        elif error_kind == ActionErrorKind.INVALID_BINDING:
            value = self.config.invalid_binding_penalty
        else:
            value = self.config.execution_failure_penalty
        if repeated:
            value *= self.config.repeated_error_multiplier
        return value


class ActionableGridWorldDMP(GridWorldDMP):
    """C5-compatible DMP that replaces legacy reward learning with consequence reward."""

    def __init__(
        self,
        *args: Any,
        actionable_reward_config: ActionableRewardConfig | None = None,
        **kwargs: Any,
    ) -> None:
        # The base reward is still computed inside GridWorldDMP for compatibility,
        # but its selector update is intercepted and replaced after diagnostics exist.
        super().__init__(*args, **kwargs)
        self.actionable_reward = ActionableRewardModule(actionable_reward_config)
        self._position_history: deque[tuple[int, int]] = deque([self.position], maxlen=4)

    def _update_selector(self, candidate: ActionCandidate, reward: float) -> None:
        # GridWorldDMP computes reward before post-transition candidate comparison.
        # Delay the only policy update until ActionableGridWorldDMP.execute finishes.
        return None

    def execute(self, candidate: ActionCandidate) -> StepResult:
        before_signatures = {
            semantic_candidate_signature(item)
            for item in self.generate_candidates()
        }
        repeated = self._signature(candidate) in self._executed_signatures

        legacy_result = super().execute(candidate)

        after_signatures = {
            semantic_candidate_signature(item)
            for item in self.generate_candidates()
        }
        newly_unlocked = len(after_signatures - before_signatures)
        newly_locked = len(before_signatures - after_signatures)

        self._position_history.append(self.position)
        semantic_repeat = repeated and not legacy_result.delta_k.has_semantic_changes()
        cycle_repeat = _is_two_cycle(self._position_history)
        error_kind = classify_error(candidate, legacy_result)

        reward = self.actionable_reward.compute(
            delta_k=legacy_result.delta_k,
            newly_unlocked_actions=newly_unlocked,
            newly_locked_actions=newly_locked,
            error_kind=error_kind,
            repeated=repeated,
            semantic_repeat=semantic_repeat,
            cycle_repeat=cycle_repeat,
            flag_found=legacy_result.flag_found,
        )

        prophecy_adjustment = legacy_result.total_reward - (
            legacy_result.external_reward + legacy_result.intrinsic_reward
        )
        total_reward = reward.total_reward + prophecy_adjustment
        update = getattr(self.scorer, "update", None)
        if callable(update):
            update(candidate, total_reward)

        if self.recent_transitions:
            transition = self.recent_transitions[-1]
            transition["reward"] = total_reward
            transition["actionable_reward"] = {
                "newly_unlocked_actions": newly_unlocked,
                "newly_locked_actions": newly_locked,
                "semantic_repeat": semantic_repeat,
                "cycle_repeat": cycle_repeat,
                "error_kind": error_kind.value,
            }

        return replace(
            legacy_result,
            external_reward=reward.external_reward,
            intrinsic_reward=reward.intrinsic_reward,
            total_reward=total_reward,
        )


def semantic_candidate_signature(candidate: ActionCandidate) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Execution-level signature: HOW labels and current position do not create duplicates."""

    return (
        candidate.template,
        tuple(
            sorted(
                (kk.value, repr(value))
                for kk, value in candidate.bindings.items()
                if kk != KK.CURRENT_POS
            )
        ),
    )


def classify_error(candidate: ActionCandidate, result: StepResult) -> ActionErrorKind:
    if not result.error:
        return ActionErrorKind.NONE
    if candidate.name == ActionName.MOVE_TOWARD:
        return ActionErrorKind.BLOCKED_PATH
    if candidate.name == ActionName.USE_OBJECT:
        return ActionErrorKind.INVALID_BINDING
    return ActionErrorKind.EXECUTION_FAILURE


def _semantic_change_count(delta_k: KnowledgeDelta) -> int:
    changed = 0
    for kk, _ in delta_k.added + delta_k.updated + delta_k.status_changed:
        if kk not in _BOOKKEEPING_KK:
            changed += 1
    for kk, _ in delta_k.removed:
        if kk not in _BOOKKEEPING_KK:
            changed += 1
    return changed


def _lifecycle_progress_count(delta_k: KnowledgeDelta) -> int:
    return sum(
        1
        for _, kv in delta_k.status_changed
        if kv.status == KnowledgeStatus.CONSUMED
    )


def _is_two_cycle(history: deque[tuple[int, int]]) -> bool:
    if len(history) < 4:
        return False
    a, b, c, d = tuple(history)
    return a == c and b == d and a != b
