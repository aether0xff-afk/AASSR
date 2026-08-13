from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import replace
from math import exp
from types import MethodType
from typing import Any, Sequence

from .current_generation import relational_action_key
from .current_relational_state import (
    KNOWN_OBJECT_INDEX,
    KNOWN_PROFILE_INDEX,
    KNOWN_ROUTE_INDEX,
    ROLE_START_INDEX,
    WORKFLOW_PROGRESS_INDEX,
)
from .current_relational_state_v3 import (
    latest_status_code,
    relational_state_descriptor_v3,
)
from .pentest_curriculum_env import PROFILE_RELATIONS, ROUTE_RELATIONS
from .skills import SKILL_VERB
from .types import Action, StateSnapshot


SUPPORT_REPLAY_PER_ACTION = 512
DEFAULT_CRITIC_SUPPORT_THRESHOLD = 0.55
MIN_CRITIC_READY_EPISODES = 32
MIN_CRITIC_POSITIVE_RETURNS = 4
MIN_CRITIC_NEGATIVE_RETURNS = 4


def _action_key(state: StateSnapshot, action: Action) -> tuple[Any, ...]:
    if action.verb_name == SKILL_VERB:
        return ("skill", str(action.target))
    return ("primitive", *relational_action_key(state, action))


def _relative_count_distance(left: float, right: float) -> float:
    left_count = int(round(max(0.0, float(left)) * 32.0))
    right_count = int(round(max(0.0, float(right)) * 32.0))
    denominator = max(1, left_count, right_count)
    return abs(left_count - right_count) / denominator


def _semantic_support_distance(
    left: StateSnapshot,
    right_descriptor: Sequence[float],
    right_status: int | None,
) -> float:
    """Coarse public-state distance used only to detect Critic extrapolation.

    This is deliberately not a value feature. It ignores request-count timing and
    other nuisance scalars, while comparing the structural quantities that change
    the task regime: public controls, workflow progress, resource scale, semantic
    role distributions, object information, and latest observed HTTP status.
    """
    left_descriptor = relational_state_descriptor_v3(left)

    control_distance = max(
        abs(left_descriptor[index] - right_descriptor[index])
        for index in range(7)
    )
    workflow_distance = abs(
        left_descriptor[WORKFLOW_PROGRESS_INDEX]
        - right_descriptor[WORKFLOW_PROGRESS_INDEX]
    )
    count_distance = max(
        _relative_count_distance(left_descriptor[index], right_descriptor[index])
        for index in (KNOWN_ROUTE_INDEX, KNOWN_PROFILE_INDEX, KNOWN_OBJECT_INDEX)
    )

    route_end = ROLE_START_INDEX + len(ROUTE_RELATIONS)
    profile_end = route_end + len(PROFILE_RELATIONS)
    route_distance = 0.5 * sum(
        abs(left_descriptor[index] - right_descriptor[index])
        for index in range(ROLE_START_INDEX, route_end)
    )
    profile_distance = 0.5 * sum(
        abs(left_descriptor[index] - right_descriptor[index])
        for index in range(route_end, profile_end)
    )

    own_index = profile_end
    target_index = own_index + 1
    tried_index = target_index + 1
    object_distance = max(
        abs(left_descriptor[own_index] - right_descriptor[own_index]),
        abs(left_descriptor[target_index] - right_descriptor[target_index]),
        _relative_count_distance(
            left_descriptor[tried_index],
            right_descriptor[tried_index],
        ),
    )

    left_status = latest_status_code(left)
    status_distance = float(left_status != right_status)

    return max(
        control_distance,
        workflow_distance,
        count_distance,
        route_distance,
        profile_distance,
        object_distance,
        status_distance,
    )


def _recent_support_snapshot(agent: object) -> dict[str, int]:
    recent_support = getattr(agent.critic, "recent_signed_support", None)
    if callable(recent_support):
        live = {key: int(value) for key, value in recent_support().items()}
        if int(live.get("observed", 0)) > 0:
            return live
    # Frozen checkpoints restore the compact recent counts through _critic_counts;
    # they intentionally do not restore the Critic training replay because the
    # checkpoint scope is evaluation-only.
    positive = int(agent._critic_counts.get("recent_positive_returns", 0))
    negative = int(agent._critic_counts.get("recent_negative_returns", 0))
    zero = int(agent._critic_counts.get("recent_zero_returns", 0))
    observed = positive + negative + zero
    return {
        "window_capacity": int(agent._critic_counts.get("recent_return_window", 0)),
        "observed": observed,
        "positive": positive,
        "zero": zero,
        "negative": negative,
    }


def install_critic_support_gate(
    agent: object,
    *,
    threshold: float = DEFAULT_CRITIC_SUPPORT_THRESHOLD,
) -> object:
    """Fail closed when an override would rely on unsupported Critic extrapolation."""
    if getattr(agent, "current_critic_support_gate", False):
        return agent

    critic = agent.critic
    representation = getattr(agent, "representation", None)
    action_structure = (
        representation.action_structure
        if representation is not None
        else relational_action_key
    )
    state_descriptor = (
        representation.state_descriptor
        if representation is not None
        else relational_state_descriptor_v3
    )

    def action_key(state: StateSnapshot, action: Action) -> tuple[Any, ...]:
        if action.verb_name == SKILL_VERB:
            return ("skill", str(action.target))
        return ("primitive", *action_structure(state, action))
    support_rows: dict[
        tuple[Any, ...],
        deque[tuple[tuple[float, ...], int | None]],
    ] = defaultdict(lambda: deque(maxlen=SUPPORT_REPLAY_PER_ACTION))
    counters: Counter[str] = Counter()
    # Explicitly expose the mutable stores used by the closures below so a frozen
    # research checkpoint can restore *exactly the state the gate consults*.
    agent._critic_support_rows = support_rows

    original_observe_episode = critic.observe_episode

    def observe_episode_with_support(
        self_critic: object,
        trajectory: Sequence[Any],
        *,
        success: bool,
    ) -> None:
        for transition in trajectory:
            key = action_key(transition.before, transition.action)
            support_rows[key].append(
                (
                    tuple(state_descriptor(transition.before)),
                    latest_status_code(transition.before),
                )
            )
            counters["observed_real_transitions"] += 1
        return original_observe_episode(trajectory, success=success)

    critic.observe_episode = MethodType(observe_episode_with_support, critic)

    # The production CurrentAgent owns finish_episode(), but this installer is also
    # intentionally reusable with minimal agents in regression/diagnostic harnesses.
    # Signed-return accounting is therefore attached only when that lifecycle hook
    # exists; Critic local-support gating itself does not depend on it.
    original_finish_episode = getattr(agent, "finish_episode", None)
    if callable(original_finish_episode):

        def finish_episode_with_signed_support(
            self_agent: object,
            *,
            final_return: float,
            training: bool = True,
        ) -> None:
            contributed = training and bool(
                getattr(self_agent, "_critic_trajectory", ())
            )
            value = float(final_return)
            result = original_finish_episode(final_return=value, training=training)
            if not contributed:
                return result

            if value > 0.0:
                counters["critic_positive_return_episodes"] += 1
                self_agent._critic_counts["positive_returns"] += 1
            elif value < 0.0:
                counters["critic_negative_return_episodes"] += 1
                self_agent._critic_counts["negative_returns"] += 1
            else:
                counters["critic_zero_return_episodes"] += 1
                self_agent._critic_counts["zero_returns"] += 1
                if int(self_agent._critic_counts.get("non_successes", 0)) > 0:
                    self_agent._critic_counts["non_successes"] -= 1

            recent = getattr(self_agent.critic, "recent_signed_support", None)
            if callable(recent):
                support = recent()
                self_agent._critic_counts["recent_return_window"] = int(
                    support.get("window_capacity", 0)
                )
                self_agent._critic_counts["recent_positive_returns"] = int(
                    support.get("positive", 0)
                )
                self_agent._critic_counts["recent_zero_returns"] = int(
                    support.get("zero", 0)
                )
                self_agent._critic_counts["recent_negative_returns"] = int(
                    support.get("negative", 0)
                )
            return result

        agent.finish_episode = MethodType(finish_episode_with_signed_support, agent)
    else:
        counters["signed_episode_accounting_hook_unavailable"] += 1

    def critic_reliably_ready(self_agent: object) -> bool:
        """Require cumulative *and recent* signed sparse-return evidence."""
        if not bool(self_agent.critic_ready):
            return False
        support = _recent_support_snapshot(self_agent)
        return (
            int(support.get("positive", 0)) >= MIN_CRITIC_POSITIVE_RETURNS
            and int(support.get("negative", 0)) >= MIN_CRITIC_NEGATIVE_RETURNS
        )

    agent.critic_reliably_ready = MethodType(critic_reliably_ready, agent)

    def support_confidence(
        self_critic: object,
        state: StateSnapshot,
        action: Action,
    ) -> float:
        del self_critic
        rows = support_rows.get(action_key(state, action), ())
        if not rows:
            return 0.0
        nearest = min(
            _semantic_support_distance(state, descriptor, status)
            for descriptor, status in rows
        )
        proximity = exp(-4.0 * max(0.0, float(nearest)))
        sample_factor = len(rows) / float(len(rows) + 4)
        return max(0.0, min(1.0, proximity * sample_factor))

    critic.support_confidence = MethodType(support_confidence, critic)

    previous_select = agent._core_select_action
    threshold = max(0.0, min(1.0, float(threshold)))

    def reconcile_recorded_block(
        self_agent: object,
        decision: object,
        reason: str,
    ) -> None:
        diagnostics = getattr(self_agent, "_imagination_diagnostics", None)
        if diagnostics is None:
            return

        old_gate = f"gate:{getattr(decision, 'imagination_gate_reason', '')}"
        if int(diagnostics.get(old_gate, 0)) <= 0:
            return

        if bool(getattr(decision, "imagination_intervention_allowed", False)):
            diagnostics["interventions"] = max(
                0,
                int(diagnostics.get("interventions", 0)) - 1,
            )
        if bool(getattr(decision, "imagination_changed_action", False)):
            diagnostics["changed_actions"] = max(
                0,
                int(diagnostics.get("changed_actions", 0)) - 1,
            )
        if bool(getattr(decision, "imagination_switch_candidate", False)):
            diagnostics["suppressed_switches"] += 1

        diagnostics[old_gate] -= 1
        diagnostics[f"gate:{reason}"] += 1
        counters["reconciled_pre_support_interventions"] += 1

    def support_gated_select(
        self_agent: object,
        state: StateSnapshot,
        *,
        episode: int,
        explore: bool,
    ) -> Any:
        decision = previous_select(state, episode=episode, explore=explore)
        if not bool(getattr(decision, "imagination_intervention_allowed", False)):
            return decision

        policy_signature = str(getattr(decision, "policy_action_signature", ""))
        policy_action = next(
            (
                action
                for action in state.available_actions
                if action.signature == policy_signature
            ),
            None,
        )
        if policy_action is None:
            counters["blocked_policy_action_missing"] += 1
            return decision

        candidate = decision.action
        policy_support = float(critic.support_confidence(state, policy_action))
        candidate_support = float(critic.support_confidence(state, candidate))
        self_agent._last_critic_policy_support = policy_support
        self_agent._last_critic_candidate_support = candidate_support

        if policy_support < threshold:
            counters["blocked_policy_ood"] += 1
            reason = "critic_policy_out_of_distribution"
        elif candidate_support < threshold:
            counters["blocked_candidate_ood"] += 1
            reason = "critic_candidate_out_of_distribution"
        else:
            counters["supported_interventions"] += 1
            return decision

        reconcile_recorded_block(self_agent, decision, reason)
        return replace(
            decision,
            action=policy_action,
            root_imagined_value=float(
                getattr(decision, "imagination_policy_value", 0.0)
            ),
            imagination_gate_reason=reason,
            imagination_changed_action=False,
            imagination_intervention_allowed=False,
        )

    agent._critic_support_previous_core_select_action = previous_select
    agent._core_select_action = MethodType(support_gated_select, agent)
    agent.current_critic_support_gate = True
    agent.current_signed_critic_readiness = True
    agent.current_recent_signed_critic_readiness = True
    agent.current_critic_support_threshold = threshold
    agent._critic_support_diagnostics = counters

    original_diagnostics = agent.diagnostics

    def diagnostics_with_support(self_agent: object) -> dict[str, Any]:
        output = dict(original_diagnostics())
        recent = _recent_support_snapshot(self_agent)
        output["critic_support"] = {
            "enabled": True,
            "threshold": threshold,
            "structural_action_buckets": len(support_rows),
            "stored_support_rows": sum(len(rows) for rows in support_rows.values()),
            "signed_readiness": bool(self_agent.critic_ready),
            "reliable_recent_signed_readiness": bool(
                self_agent.critic_reliably_ready()
            ),
            "minimum_ready_episodes": MIN_CRITIC_READY_EPISODES,
            "minimum_positive_returns": MIN_CRITIC_POSITIVE_RETURNS,
            "minimum_negative_returns": MIN_CRITIC_NEGATIVE_RETURNS,
            "recent_signed_support": dict(recent),
            **dict(counters),
        }
        repairs = dict(output.get("current_repairs", {}))
        repairs.update(
            {
                "relational_public_state_v3": True,
                "public_http_status_preserved": True,
                "status_aware_semantic_calibration": True,
                "critic_local_support_gate": True,
                "critic_support_is_gate_not_value": True,
                "critic_signed_negative_return_readiness": True,
                "critic_recent_signed_return_readiness": True,
                "intervention_counters_are_post_support": True,
            }
        )
        output["current_repairs"] = repairs
        return output

    agent.diagnostics = MethodType(diagnostics_with_support, agent)
    return agent
