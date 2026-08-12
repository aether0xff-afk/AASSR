from __future__ import annotations

from dataclasses import replace
from types import MethodType
from typing import Any, Iterable

from .autonomous_agent_core import ActionDecision
from .branch_critic import CriticTransition
from .current_generation import relational_action_key
from .skills import SKILL_VERB
from .types import Action, StateSnapshot


class ConfidenceIndependentCriticEncoder:
    """Keep the current Critic input shape while removing confidence as a value feature.

    Historical/current checkpoints expect the final scalar slot to exist. The
    current relational Critic used that slot for Prophecy confidence, which lets
    confidence influence branch value in addition to the planner's uncertainty
    penalty. Replace only that input with a constant so the network shape and
    batched hardware path stay compatible while confidence can no longer rank
    branches through the Critic.
    """

    def __init__(self, delegate: object) -> None:
        self.delegate = delegate
        self.feature_size = int(getattr(delegate, "feature_size"))

    def encode(self, transition: CriticTransition) -> tuple[float, ...]:
        neutral = CriticTransition(
            transition.before,
            transition.action,
            transition.after,
            1.0,
        )
        encoded = tuple(float(value) for value in self.delegate.encode(neutral))
        if len(encoded) != self.feature_size:
            raise ValueError("confidence-independent critic encoder size mismatch")
        return encoded


def _confidence_cache_key(state: StateSnapshot, action: Action) -> tuple[Any, ...]:
    if action.verb_name == SKILL_VERB:
        return ("skill", str(action.target))
    return ("primitive", relational_action_key(state, action))


def _action_confidences(
    agent: object,
    state: StateSnapshot,
    actions: Iterable[Action],
) -> dict[str, float]:
    """Return calibrated Prophecy confidence with relational duplicates memoized."""

    cache: dict[tuple[Any, ...], float] = {}
    output: dict[str, float] = {}
    for action in actions:
        key = _confidence_cache_key(state, action)
        if key not in cache:
            raw = float(agent.skill_prophecy.confidence(state, action))
            cache[key] = max(0.0, min(1.0, raw))
        output[action.signature] = cache[key]
    return output


def _choose_best(agent: object, state: StateSnapshot, evaluations: Iterable[Any]) -> Any:
    materialized = tuple(evaluations)
    if not materialized:
        return None
    best_value = max(float(item.aggregate_value) for item in materialized)
    tied = [
        item
        for item in materialized
        if abs(float(item.aggregate_value) - best_value) <= 1e-12
    ]
    return min(
        tied,
        key=lambda item: (
            -float(agent.policy.value(state, item.action)),
            item.action.signature,
        ),
    )


def _critic_is_reliably_ready(agent: object) -> bool:
    ready = getattr(agent, "critic_reliably_ready", None)
    if callable(ready):
        return bool(ready())
    return bool(agent.critic_ready)


def _confidence_gated_core_select_action(
    self: object,
    state: StateSnapshot,
    *,
    episode: int,
    explore: bool,
) -> ActionDecision:
    """Current Imagination decision rule with confidence used only as reliability.

    Confidence has exactly two roles here:
      1. the existing global model-coverage eligibility gate; and
      2. a per-root prediction reliability gate.

    Once a root is reliable, ranking and intervention advantage are Critic-only.
    Confidence does not alter planner value, Critic input, or the fixed
    intervention margin.
    """

    epsilon = self.epsilon(episode) if explore else 0.0
    policy_action = self.policy.select(
        state,
        randomizer=self.randomizer,
        epsilon=epsilon,
        exploration_bonus=0.0,
    )
    self._decision_index += 1
    coverage = self.skill_prophecy.coverage(state, state.available_actions)

    opportunity = bool(self.requested_imagination)
    if not opportunity:
        reason = "disabled"
    elif explore and not self.training_imagination:
        reason = "training_suppressed"
    elif not _critic_is_reliably_ready(self):
        reason = "critic_not_ready"
    elif coverage < self.config.imagination_minimum_coverage:
        reason = "coverage"
    else:
        reason = "eligible"

    if reason != "eligible":
        return self._record_decision(
            ActionDecision(
                policy_action,
                False,
                policy_action_signature=policy_action.signature,
                imagination_opportunity=opportunity,
                imagination_eligible=False,
                imagination_gate_reason=reason,
                model_coverage=coverage,
            )
        )

    plan = self.planner.plan(state)
    roots = tuple(plan.root_evaluations)
    if not roots:
        return self._record_decision(
            ActionDecision(
                policy_action,
                True,
                imagined_nodes=len(plan.nodes),
                imagination_depth=plan.maximum_depth_reached,
                policy_action_signature=policy_action.signature,
                imagination_opportunity=True,
                imagination_eligible=True,
                imagination_gate_reason="no_root_evaluation",
                model_coverage=coverage,
            )
        )

    by_signature = {item.action.signature: item for item in roots}
    policy_evaluation = by_signature.get(policy_action.signature)
    minimum_confidence = float(self.config.imagination_minimum_coverage)
    confidences = _action_confidences(
        self,
        state,
        tuple(item.action for item in roots) + (policy_action,),
    )
    reliable = tuple(
        item
        for item in roots
        if confidences.get(item.action.signature, 0.0) >= minimum_confidence
    )
    preferred = _choose_best(self, state, reliable)
    raw_preferred = _choose_best(self, state, roots)

    policy_value = (
        float(policy_evaluation.aggregate_value)
        if policy_evaluation is not None
        else 0.0
    )
    required_advantage = float(self.config.imagination_intervention_margin)
    policy_confidence = confidences.get(policy_action.signature, 0.0)

    # Fail closed when the Policy branch itself cannot be predicted reliably.
    # Without a reliable baseline, a Critic difference is not an apples-to-apples
    # reason to override the real Policy action.
    if policy_evaluation is None:
        displayed = preferred or raw_preferred
        displayed_value = float(displayed.aggregate_value) if displayed is not None else 0.0
        switch_candidate = bool(
            displayed is not None
            and displayed.action.signature != policy_action.signature
        )
        return self._record_decision(
            ActionDecision(
                policy_action,
                True,
                imagined_nodes=len(plan.nodes),
                imagination_depth=plan.maximum_depth_reached,
                root_imagined_value=policy_value,
                policy_action_signature=policy_action.signature,
                imagination_opportunity=True,
                imagination_eligible=True,
                imagination_gate_reason="policy_not_evaluated",
                model_coverage=coverage,
                imagination_preferred_action_signature=(
                    displayed.action.signature if displayed is not None else ""
                ),
                imagination_policy_value=policy_value,
                imagination_preferred_value=displayed_value,
                imagination_advantage=0.0,
                imagination_required_advantage=required_advantage,
                imagination_switch_candidate=switch_candidate,
                imagination_intervention_allowed=False,
            )
        )

    if policy_confidence < minimum_confidence:
        displayed = preferred or raw_preferred or policy_evaluation
        displayed_value = float(displayed.aggregate_value)
        switch_candidate = displayed.action.signature != policy_action.signature
        advantage = displayed_value - policy_value if switch_candidate else 0.0
        return self._record_decision(
            ActionDecision(
                policy_action,
                True,
                imagined_nodes=len(plan.nodes),
                imagination_depth=plan.maximum_depth_reached,
                root_imagined_value=policy_value,
                policy_action_signature=policy_action.signature,
                imagination_opportunity=True,
                imagination_eligible=True,
                imagination_gate_reason="policy_prediction_low_confidence",
                model_coverage=coverage,
                imagination_preferred_action_signature=displayed.action.signature,
                imagination_policy_value=policy_value,
                imagination_preferred_value=displayed_value,
                imagination_advantage=advantage,
                imagination_required_advantage=required_advantage,
                imagination_switch_candidate=switch_candidate,
                imagination_intervention_allowed=False,
            )
        )

    if preferred is None:
        return self._record_decision(
            ActionDecision(
                policy_action,
                True,
                imagined_nodes=len(plan.nodes),
                imagination_depth=plan.maximum_depth_reached,
                root_imagined_value=policy_value,
                policy_action_signature=policy_action.signature,
                imagination_opportunity=True,
                imagination_eligible=True,
                imagination_gate_reason="no_reliable_imagination",
                model_coverage=coverage,
                imagination_policy_value=policy_value,
                imagination_required_advantage=required_advantage,
            )
        )

    switch_candidate = preferred.action.signature != policy_action.signature
    preferred_value = float(preferred.aggregate_value)
    advantage = preferred_value - policy_value if switch_candidate else 0.0
    intervention_allowed = switch_candidate and advantage >= required_advantage

    if intervention_allowed:
        intervention_reason = "intervention"
    elif switch_candidate:
        intervention_reason = "insufficient_advantage"
    elif (
        raw_preferred is not None
        and raw_preferred.action.signature != policy_action.signature
        and confidences.get(raw_preferred.action.signature, 0.0) < minimum_confidence
    ):
        # A low-confidence raw winner was explicitly removed before Critic-only
        # comparison. Keep this visible in diagnostics without executing it.
        intervention_reason = "candidate_prediction_low_confidence"
    else:
        intervention_reason = "policy_agreement"

    executed_action = preferred.action if intervention_allowed else policy_action
    executed_value = preferred_value if intervention_allowed else policy_value
    return self._record_decision(
        ActionDecision(
            executed_action,
            True,
            imagined_nodes=len(plan.nodes),
            imagination_depth=plan.maximum_depth_reached,
            root_imagined_value=executed_value,
            policy_action_signature=policy_action.signature,
            imagination_opportunity=True,
            imagination_eligible=True,
            imagination_gate_reason=intervention_reason,
            imagination_changed_action=intervention_allowed,
            model_coverage=coverage,
            imagination_preferred_action_signature=preferred.action.signature,
            imagination_policy_value=policy_value,
            imagination_preferred_value=preferred_value,
            imagination_advantage=advantage,
            imagination_required_advantage=required_advantage,
            imagination_switch_candidate=switch_candidate,
            imagination_intervention_allowed=intervention_allowed,
        )
    )


def install_current_confidence_gate(agent: object) -> object:
    """Install the current confidence-as-reliability-only Imagination contract."""

    if getattr(agent, "current_confidence_gate", False):
        return agent

    # Planner branch ranking becomes Critic-only. Confidence may still terminate
    # an unreliable path through minimum_path_confidence, which is a gate rather
    # than a value bonus/penalty.
    agent.planner.config = replace(
        agent.planner.config,
        uncertainty_penalty=0.0,
    )
    agent.core.planner = agent.planner

    # The old coverage-dependent extra margin is intentionally retired. Keep the
    # config field for artifact compatibility but freeze its active value at zero.
    agent.config = replace(
        agent.config,
        imagination_uncertainty_margin=0.0,
    )

    # Preserve the final scalar input slot for model/batching compatibility, but
    # make it a constant so Critic values cannot depend on Prophecy confidence.
    if not isinstance(agent.critic.encoder, ConfidenceIndependentCriticEncoder):
        agent.critic.encoder = ConfidenceIndependentCriticEncoder(agent.critic.encoder)

    agent._confidence_gate_previous_core_select_action = agent._core_select_action
    agent._core_select_action = MethodType(_confidence_gated_core_select_action, agent)
    agent.current_confidence_gate = True
    agent.current_confidence_gate_threshold = float(
        agent.config.imagination_minimum_coverage
    )
    agent.current_imagination_value_contract = "critic_only_after_confidence_gate"
    return agent
