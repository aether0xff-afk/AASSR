from __future__ import annotations

from collections import Counter
from typing import Sequence

from .pentest_agent_main_test import AGENT_STATE_SIZE
from .pentest_curriculum_env import (
    PROFILE_RELATIONS,
    ROUTE_RELATIONS,
    relational_action_features,
)
from .skills import SKILL_VERB
from .types import Action, StateSnapshot


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
    + 4  # audit pressure, request usage, session remaining, workflow progress
    + 3  # known route/profile/object counts
    + len(ROUTE_RELATIONS)
    + len(PROFILE_RELATIONS)
    + 2  # own/target object observed
    + 1  # tried object fraction
    + 4  # action-surface summary
)


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


def _role_counts(
    facts: frozenset[str],
    prefix: str,
    roles: Sequence[str],
) -> tuple[float, ...]:
    counts = Counter()
    for fact in facts:
        if not fact.startswith(prefix):
            continue
        try:
            _, role = fact.removeprefix(prefix).rsplit(":", 1)
        except ValueError:
            continue
        counts[role] += 1
    normalizer = float(max(1, sum(counts.values())))
    return tuple(counts[role] / normalizer for role in roles)


def workflow_progress_fraction(state: StateSnapshot) -> float:
    """Return public dependency progress without reading hidden scenario state."""
    explicit = state.metadata.get("relational_workflow_progress")
    if explicit is not None:
        return _clamp01(explicit)

    progress = state.metadata.get("workflow_progress")
    depth = state.metadata.get("workflow_depth")
    if progress is not None and depth is not None:
        try:
            denominator = max(1.0, float(depth))
            return _clamp01(float(progress) / denominator)
        except (TypeError, ValueError):
            pass

    for fact in state.facts:
        if not fact.startswith("workflow_progress:"):
            continue
        parts = fact.split(":")
        if len(parts) != 3:
            continue
        try:
            return _clamp01(float(parts[1]) / max(1.0, float(parts[2])))
        except (TypeError, ValueError, ZeroDivisionError):
            continue

    # Generic HTTP worlds do not expose the curriculum dependency counter. Their
    # public goal progress is the best available structural fallback.
    return _clamp01(state.goal_progress)


def relational_state_descriptor_v2(state: StateSnapshot) -> tuple[float, ...]:
    """Rename-invariant public state used by all current transfer learners.

    Unlike v1, this contract keeps the observable resource-pressure axes that
    actually define later curriculum levels: audit/lockout pressure, request/rate
    usage, session lifetime remaining, and dependency workflow progress.
    """
    facts = state.facts
    controls = tuple(
        _clamp01(state.vector[index] if index < len(state.vector) else 0.0)
        for index in range(CONTROL_SIZE)
    )
    audit_pressure = _clamp01(
        state.vector[AUDIT_PRESSURE_INDEX]
        if len(state.vector) > AUDIT_PRESSURE_INDEX
        else 0.0
    )
    request_usage = _clamp01(
        state.vector[REQUEST_USAGE_INDEX]
        if len(state.vector) > REQUEST_USAGE_INDEX
        else 0.0
    )
    session_remaining = _clamp01(
        state.vector[SESSION_REMAINING_INDEX]
        if len(state.vector) > SESSION_REMAINING_INDEX
        else 0.0
    )
    workflow_progress = workflow_progress_fraction(state)

    known_routes = len(_fact_values(facts, "known_route:"))
    known_profiles = len(_fact_values(facts, "known_profile:"))
    known_objects = len(_fact_values(facts, "known_object:"))
    tried_objects = len(_fact_values(facts, "tried_object:"))

    route_roles = _role_counts(
        facts,
        "observed_route_role:",
        ROUTE_RELATIONS,
    )
    profile_roles = _role_counts(
        facts,
        "observed_profile_role:",
        PROFILE_RELATIONS,
    )

    actions = tuple(state.available_actions)
    relational_actions = {
        relational_action_features(state, action)
        for action in actions
        if action.verb_name != SKILL_VERB
    }
    request_fraction = (
        sum(action.verb_name == "request" for action in actions) / len(actions)
        if actions
        else 0.0
    )
    object_request_fraction = (
        sum(action.verb_name == "request_object" for action in actions) / len(actions)
        if actions
        else 0.0
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
        min(1.0, len(actions) / 128.0),
        min(1.0, len(relational_actions) / 32.0),
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
    """Make legacy current classes consume the v2 public-state contract.

    Current-generation classes were originally defined in `current_generation`.
    Their methods resolve these helper names at runtime, so patching the module
    globals upgrades the active classes without editing historical reproduction
    code. Modules that imported helper functions by value are patched explicitly.
    """
    from . import current_generation as generation

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

        codec.relational_state_descriptor = relational_state_descriptor_v2
        codec.relational_state_vector = relational_state_vector_v2
        codec.REL_DESCRIPTOR_SIZE = REL_DESCRIPTOR_SIZE
    except ImportError:  # pragma: no cover
        pass

    try:
        from . import current_relational_model as model

        model.relational_state_vector = relational_state_vector_v2
        model.REL_DESCRIPTOR_SIZE = REL_DESCRIPTOR_SIZE
    except ImportError:  # pragma: no cover
        pass

    try:
        from . import current_repair as repair

        repair.relational_state_key = relational_state_key_v2
    except ImportError:  # pragma: no cover
        pass
