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
    """Average confidence once per unique relational action identity.

    Concrete aliases with the same public relational action are deliberately one
    model question everywhere else (Prophecy, root dedup, Critic support). Their
    multiplicity must therefore not become an accidental confidence weight. This
    makes Imagination eligibility invariant to identifier renaming and to how many
    equivalent object/route aliases happen to be present on the concrete surface.
    """

    materialized = tuple(actions)
    if not materialized:
        return 1.0
    cache: dict[tuple[Any, ...], float] = {}
    for action in materialized:
        representation = getattr(getattr(self, "base", None), "representation", None)
        key = (
            ("skill", str(action.target))
            if action.verb_name == SKILL_VERB
            else (
                "primitive",
                (
                    representation.action_structure(state, action)
                    if representation is not None
                    else relational_action_key(state, action)
                ),
            )
        )
        if key not in cache:
            cache[key] = float(self.confidence(state, action))
    return sum(cache.values()) / len(cache)


def _critic_is_reliably_ready(agent: object) -> bool:
    ready = getattr(agent, "critic_reliably_ready", None)
    if callable(ready):
        return bool(ready())
    return bool(agent.critic_ready)


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
    implementation unchanged, where unique-structural coverage above is used.
    """

    opportunity = bool(self.requested_imagination)
    critic_ready = _critic_is_reliably_ready(self)
    if opportunity and not (explore and not self.training_imagination) and critic_ready:
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
    agent.current_structural_coverage = True
    return agent
