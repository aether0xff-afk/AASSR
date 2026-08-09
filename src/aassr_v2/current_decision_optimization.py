from __future__ import annotations

from types import MethodType
from typing import Any, Iterable

from .autonomous_agent_core import ActionDecision
from .current_generation import relational_action_key
from .skills import SKILL_VERB
from .types import Action, StateSnapshot


def _coverage_cache_key(state: StateSnapshot, action: Action) -> tuple[Any, ...]:
    if action.verb_name == SKILL_VERB:
        return ("skill", str(action.target))
    return ("primitive", relational_action_key(state, action))


def _memoized_relational_coverage(
    self: object,
    state: StateSnapshot,
    actions: Iterable[Action],
) -> float:
    """Exact coverage average with repeated relational confidence memoized.

    Current Neural Delta, calibration, and confidence all use the same relational
    state/action identity. Concrete seed-renamed actions with the same relational
    key therefore have exactly the same confidence in one state. We still add one
    value per original action in original order, preserving the arithmetic shape
    of the previous average while avoiding repeated holdout scans/model calls.
    """

    materialized = tuple(actions)
    if not materialized:
        return 1.0
    cache: dict[tuple[Any, ...], float] = {}
    total = 0.0
    for action in materialized:
        key = _coverage_cache_key(state, action)
        if key not in cache:
            cache[key] = float(self.confidence(state, action))
        total += cache[key]
    return total / len(materialized)


def _fast_core_select_action(
    self: object,
    state: StateSnapshot,
    *,
    episode: int,
    explore: bool,
) -> ActionDecision:
    """Skip coverage when an earlier gate already makes it irrelevant.

    Gate order in the canonical current agent is:
      disabled -> training_suppressed -> critic_not_ready -> coverage -> eligible.
    Coverage cannot affect the selected real action in the first three cases, so
    evaluating it there is dead work. The eligible path delegates to the original
    implementation unchanged, where the memoized exact coverage above is used.
    """

    opportunity = bool(self.requested_imagination)
    if opportunity and not (explore and not self.training_imagination) and self.critic_ready:
        return self._current_original_core_select_action(
            state,
            episode=episode,
            explore=explore,
        )

    epsilon = self.epsilon(episode) if explore else 0.0
    policy_action = self.policy.select(
        state,
        randomizer=self.randomizer,
        epsilon=epsilon,
        exploration_bonus=0.0,
    )
    self._decision_index += 1

    if not opportunity:
        reason = "disabled"
    elif explore and not self.training_imagination:
        reason = "training_suppressed"
    else:
        reason = "critic_not_ready"

    self.current_coverage_skipped_decisions += 1
    return self._record_decision(
        ActionDecision(
            policy_action,
            False,
            policy_action_signature=policy_action.signature,
            imagination_opportunity=opportunity,
            imagination_eligible=False,
            imagination_gate_reason=reason,
            # Coverage was provably irrelevant to this decision. Keep a bounded
            # numeric diagnostic value without paying to compute the unused metric.
            model_coverage=0.0,
        )
    )


def install_current_decision_optimizations(agent: object) -> object:
    if hasattr(agent, "_current_original_core_select_action"):
        return agent

    agent._current_original_core_select_action = agent._core_select_action
    agent.current_coverage_skipped_decisions = 0
    agent.skill_prophecy.coverage = MethodType(
        _memoized_relational_coverage,
        agent.skill_prophecy,
    )
    agent._core_select_action = MethodType(_fast_core_select_action, agent)
    agent.current_decision_optimization = True
    return agent
