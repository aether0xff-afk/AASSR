from __future__ import annotations

from collections import Counter
from typing import Callable, Sequence

from .pentest_agent_main_test import AGENT_STATE_SIZE
from .pentest_curriculum_env import (
    PROFILE_RELATIONS,
    ROUTE_RELATIONS,
    relational_action_features,
)
from .skills import SKILL_VERB
from .types import StateSnapshot


CONTROL_SIZE = 7
AUDIT_PRESSURE_INDEX = 7
REQUEST_USAGE_INDEX = 8
SESSION_REMAINING_INDEX = 9
WORKFLOW_PROGRESS_INDEX = 10
KNOWN_ROUTE_INDEX = 11
KNOWN_PROFILE_INDEX = 12
KNOWN_OBJECT_INDEX = 13
ROLE_START_INDEX = 14

REL_DESCRIPTOR_SIZE = (
    CONTROL_SIZE
    + 4
    + 3
    + len(ROUTE_RELATIONS)
    + len(PROFILE_RELATIONS)
    + 2
    + 1
    + 4
)
ACTION_COUNT_INDEX = REL_DESCRIPTOR_SIZE - 4
UNIQUE_ACTION_COUNT_INDEX = REL_DESCRIPTOR_SIZE - 3
REQUEST_FRACTION_INDEX = REL_DESCRIPTOR_SIZE - 2
OBJECT_REQUEST_FRACTION_INDEX = REL_DESCRIPTOR_SIZE - 1


def _clamp01(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float(default)
    return max(0.0, min(1.0, result))


def _fact_values(facts: frozenset[str], prefix: str) -> tuple[str, ...]:
    return tuple(
        fact.removeprefix(prefix)
        for fact in facts
        if fact.startswith(prefix)
    )


def _entity_role_distribution(
    facts: frozenset[str],
    *,
    known_prefix: str,
    observed_prefix: str,
    roles: Sequence[str],
    implicit_role: Callable[[str], str | None] | None = None,
) -> tuple[float, ...]:
    """Role distribution over every public known entity, including unknowns."""
    known = set(_fact_values(facts, known_prefix))
    observed: dict[str, str] = {}
    for fact in facts:
        if not fact.startswith(observed_prefix):
            continue
        payload = fact.removeprefix(observed_prefix)
        try:
            identifier, role = payload.rsplit(":", 1)
        except ValueError:
            continue
        if role not in roles:
            role = "unknown"
        observed[identifier] = role
        known.add(identifier)

    counts = Counter()
    for identifier in known:
        role = observed.get(identifier)
        if role is None and implicit_role is not None:
            role = implicit_role(identifier)
        if role not in roles:
            role = "unknown"
        counts[role] += 1
    normalizer = float(max(1, len(known)))
    return tuple(counts[role] / normalizer for role in roles)


def workflow_progress_fraction(state: StateSnapshot) -> float:
    explicit = state.metadata.get("relational_workflow_progress")
    if explicit is not None:
        return _clamp01(explicit)
    if len(state.vector) > WORKFLOW_PROGRESS_INDEX:
        return _clamp01(state.vector[WORKFLOW_PROGRESS_INDEX])

    scale = state.metadata.get("workflow_progress_scale")
    for fact in state.facts:
        if not fact.startswith("workflow_progress:"):
            continue
        parts = fact.split(":")
        if len(parts) == 2 and scale is not None:
            try:
                return _clamp01(float(parts[1]) / max(1.0, float(scale)))
            except (TypeError, ValueError, ZeroDivisionError):
                continue
        if len(parts) == 3:
            try:
                return _clamp01(float(parts[1]) / max(1.0, float(parts[2])))
            except (TypeError, ValueError, ZeroDivisionError):
                continue
    return _clamp01(state.goal_progress)


def relational_state_descriptor_v2(state: StateSnapshot) -> tuple[float, ...]:
    """Rename-invariant audited public state for current transfer learners.

    Unknown-role mass is explicit: every known route/profile without an observed
    semantic role contributes to ``unknown`` rather than disappearing from the
    role distribution. ``profile-browse`` has an implicit public browse role.
    Imagined concrete multiplicity remains a scalar summary only.
    """
    facts = state.facts
    controls = tuple(
        _clamp01(state.vector[index] if index < len(state.vector) else 0.0)
        for index in range(CONTROL_SIZE)
    )
    audit_pressure = 0.0
    request_usage = _clamp01(
        state.vector[REQUEST_USAGE_INDEX]
        if len(state.vector) > REQUEST_USAGE_INDEX
        else 0.0
    )
    session_remaining = 0.0
    workflow_progress = workflow_progress_fraction(state)

    known_routes = len(_fact_values(facts, "known_route:"))
    known_profiles = len(_fact_values(facts, "known_profile:"))
    known_objects = len(_fact_values(facts, "known_object:"))
    tried_objects = len(_fact_values(facts, "tried_object:"))

    route_roles = _entity_role_distribution(
        facts,
        known_prefix="known_route:",
        observed_prefix="observed_route_role:",
        roles=ROUTE_RELATIONS,
    )
    profile_roles = _entity_role_distribution(
        facts,
        known_prefix="known_profile:",
        observed_prefix="observed_profile_role:",
        roles=PROFILE_RELATIONS,
        implicit_role=lambda identifier: (
            "browse" if identifier == "profile-browse" else None
        ),
    )

    actions = tuple(state.available_actions)
    relational_actions = {
        relational_action_features(state, action)
        for action in actions
        if action.verb_name != SKILL_VERB
    }
    observed_action_count = min(1.0, len(actions) / 128.0)
    observed_unique_count = min(1.0, len(relational_actions) / 32.0)
    observed_request_fraction = (
        sum(action.verb_name == "request" for action in actions) / len(actions)
        if actions
        else 0.0
    )
    observed_object_request_fraction = (
        sum(action.verb_name == "request_object" for action in actions) / len(actions)
        if actions
        else 0.0
    )
    action_count_fraction = _clamp01(
        state.metadata.get("relational_action_count_fraction"),
        observed_action_count,
    )
    unique_action_count_fraction = _clamp01(
        state.metadata.get("relational_unique_action_count_fraction"),
        observed_unique_count,
    )
    request_fraction = _clamp01(
        state.metadata.get("relational_request_fraction"),
        observed_request_fraction,
    )
    object_request_fraction = _clamp01(
        state.metadata.get("relational_object_request_fraction"),
        observed_object_request_fraction,
    )

    descriptor = (
        *controls,
        audit_pressure,
        request_usage,
        session_remaining,
        workflow_progress,
        min(1.0, known_routes / 32.0),
        min(1.0, known_profiles / 32.0),
        min(1.0, known_objects / 32.0),
        *route_roles,
        *profile_roles,
        float(bool(_fact_values(facts, "observed_own_object:"))),
        float(bool(_fact_values(facts, "observed_target_object:"))),
        min(1.0, tried_objects / 32.0),
        action_count_fraction,
        unique_action_count_fraction,
        request_fraction,
        object_request_fraction,
    )
    if len(descriptor) != REL_DESCRIPTOR_SIZE:
        raise AssertionError(
            f"relational state descriptor v2 size drift: {len(descriptor)} != {REL_DESCRIPTOR_SIZE}"
        )
    return descriptor


def relational_state_vector_v2(state: StateSnapshot) -> tuple[float, ...]:
    descriptor = relational_state_descriptor_v2(state)
    if len(descriptor) > AGENT_STATE_SIZE:
        raise AssertionError("relational state descriptor exceeds DQN state size")
    return descriptor + (0.0,) * (AGENT_STATE_SIZE - len(descriptor))


def relational_state_key_v2(state: StateSnapshot) -> tuple[float, ...]:
    return tuple(round(value, 8) for value in relational_state_descriptor_v2(state))


def install_relational_state_contract() -> None:
    from . import current_generation as generation
    from .current_relational_decode_v2 import decode_relational_state_v2

    generation.relational_state_descriptor = relational_state_descriptor_v2
    generation.relational_state_vector = relational_state_vector_v2
    generation.relational_state_key = relational_state_key_v2

    try:
        from . import current_hardware as hardware
        hardware.relational_state_key = relational_state_key_v2
    except ImportError:  # pragma: no cover
        pass
    try:
        from . import current_runtime as runtime
        if hasattr(runtime, "relational_state_vector"):
            runtime.relational_state_vector = relational_state_vector_v2
        if hasattr(runtime, "relational_state_key"):
            runtime.relational_state_key = relational_state_key_v2
    except ImportError:  # pragma: no cover
        pass
    try:
        from . import dreamerv3_baseline as dreamer
        dreamer.relational_state_vector = relational_state_vector_v2
    except ImportError:  # pragma: no cover
        pass
    try:
        from . import current_relational_codec as codec
        codec.decode_relational_state = decode_relational_state_v2
    except ImportError:  # pragma: no cover
        pass
