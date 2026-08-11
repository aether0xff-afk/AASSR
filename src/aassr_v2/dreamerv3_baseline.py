from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .pentest_agent_main_test import ACTION_FEATURE_SIZE
from .pentest_curriculum_env import (
    OBJECT_RELATIONS,
    PROFILE_RELATIONS,
    ROUTE_RELATIONS,
    relational_action_features,
)
from .current_relational_state_v3 import relational_state_vector_v3
from .types import Action, StateSnapshot


DREAMERV3_CONDITION = "dreamerv3_relational"
DREAMERV3_UPSTREAM_REPOSITORY = "danijar/dreamerv3"
DREAMERV3_UPSTREAM_COMMIT = "e3f02248693a79dc8b0ebd62c93683888ddaccfe"
DREAMERV3_BASELINE_VERSION = "official-dreamerv3-relational-categorical-v4"

_DREAMER_OBJECT_RELATIONS = tuple(
    item for item in OBJECT_RELATIONS if item != "none"
)
DREAMERV3_ACTION_SLOTS: tuple[tuple[str, str, str, str], ...] = tuple(
    slot
    for route_role in ROUTE_RELATIONS
    for profile_role in PROFILE_RELATIONS
    for slot in (
        ("request", route_role, profile_role, "none"),
        *(
            (
                "request_object",
                route_role,
                profile_role,
                object_role,
            )
            for object_role in _DREAMER_OBJECT_RELATIONS
        ),
    )
)
DREAMERV3_ACTION_SLOT_INDEX = {
    slot: index for index, slot in enumerate(DREAMERV3_ACTION_SLOTS)
}
DREAMERV3_ACTION_SLOT_COUNT = len(DREAMERV3_ACTION_SLOTS)

if DREAMERV3_ACTION_SLOT_COUNT != (
    len(ROUTE_RELATIONS)
    * len(PROFILE_RELATIONS)
    * (1 + len(_DREAMER_OBJECT_RELATIONS))
):
    raise AssertionError("DreamerV3 relational action vocabulary size drift")

_VERBS = ("request", "request_object")
_SLOT_VECTOR_SIZE = (
    len(_VERBS)
    + len(ROUTE_RELATIONS)
    + len(PROFILE_RELATIONS)
    + len(OBJECT_RELATIONS)
)


@dataclass(frozen=True, slots=True)
class DreamerActionProjection:
    action: Action
    relational_features: tuple[float, ...]
    requested_slot_index: int
    slot_index: int
    squared_distance: float
    tied_candidates: int


def _observed_roles(
    state: StateSnapshot,
    action: Action,
) -> tuple[str, str, str]:
    route_id = str(action.parameters.get("route_id", ""))
    profile_id = str(action.parameters.get("profile_id", ""))
    object_id = action.parameters.get("object_id")

    route_role = "unknown"
    profile_role = "browse" if profile_id == "profile-browse" else "unknown"
    own = False
    target = False
    tried: set[str] = set()

    for fact in state.facts:
        if fact.startswith(f"observed_route_role:{route_id}:"):
            route_role = fact.rsplit(":", 1)[1]
        elif fact.startswith(f"observed_profile_role:{profile_id}:"):
            profile_role = fact.rsplit(":", 1)[1]
        elif fact.startswith("observed_own_object:") and object_id is not None:
            own = fact.split(":", 1)[1] == str(object_id)
        elif fact.startswith("observed_target_object:") and object_id is not None:
            target = fact.split(":", 1)[1] == str(object_id)
        elif fact.startswith("tried_object:"):
            tried.add(fact.split(":", 1)[1])

    if route_role not in ROUTE_RELATIONS:
        route_role = "unknown"
    if profile_role not in PROFILE_RELATIONS:
        profile_role = "unknown"

    if object_id is None:
        object_role = "none"
    elif own:
        object_role = "own"
    elif target:
        object_role = "target"
    elif str(object_id) in tried:
        object_role = "tried"
    else:
        object_role = "untried"
    return route_role, profile_role, object_role


def dreamer_action_slot_key(
    state: StateSnapshot,
    action: Action,
) -> tuple[str, str, str, str]:
    if action.verb_name not in _VERBS:
        raise ValueError(
            "DreamerV3 baseline does not expose non-HTTP primitive "
            f"{action.verb_name!r}"
        )
    route_role, profile_role, object_role = _observed_roles(state, action)
    if action.verb_name == "request":
        object_role = "none"
    elif object_role == "none":
        raise ValueError("request_object action is missing an object relation")
    key = (action.verb_name, route_role, profile_role, object_role)
    if key not in DREAMERV3_ACTION_SLOT_INDEX:
        raise AssertionError(
            f"unregistered DreamerV3 relational action slot: {key}"
        )
    return key


def dreamer_action_surface_mask(state: StateSnapshot) -> tuple[float, ...]:
    values = [0.0] * DREAMERV3_ACTION_SLOT_COUNT
    for action in state.available_actions:
        if action.verb_name not in _VERBS:
            continue
        values[
            DREAMERV3_ACTION_SLOT_INDEX[dreamer_action_slot_key(state, action)]
        ] = 1.0
    return tuple(values)


def dreamer_observation_vector(state: StateSnapshot) -> tuple[float, ...]:
    """Use exactly the active public relational v3 Policy representation."""
    return tuple(float(value) for value in relational_state_vector_v3(state))


def dreamer_relational_action_features(
    state: StateSnapshot,
    action: Action,
) -> tuple[float, ...]:
    values = tuple(
        float(value) for value in relational_action_features(state, action)
    )
    if len(values) != ACTION_FEATURE_SIZE:
        raise AssertionError("DreamerV3 relational action feature size drift")
    return values


def dreamer_action_slot_vector(
    slot: tuple[str, str, str, str],
) -> tuple[float, ...]:
    verb, route_role, profile_role, object_role = slot
    values: list[float] = []
    values.extend(float(verb == item) for item in _VERBS)
    values.extend(float(route_role == item) for item in ROUTE_RELATIONS)
    values.extend(float(profile_role == item) for item in PROFILE_RELATIONS)
    values.extend(float(object_role == item) for item in OBJECT_RELATIONS)
    if len(values) != _SLOT_VECTOR_SIZE:
        raise AssertionError("DreamerV3 slot vector size drift")
    return tuple(values)


def _squared_distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("DreamerV3 action projection dimensionality mismatch")
    return float(
        sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))
    )


def project_dreamer_action(
    state: StateSnapshot,
    proposal: int,
) -> DreamerActionProjection:
    """Project one categorical Dreamer slot onto the current legal surface.

    Projection uses only the public current state and available action surface. It
    never reads hidden benchmark scenario, reward, future state, or success path.
    Structurally identical concrete candidates remain deterministic signature ties.
    """
    try:
        requested_slot = int(proposal)
    except (TypeError, ValueError) as exc:
        raise ValueError("DreamerV3 categorical action must be an integer slot") from exc
    if not 0 <= requested_slot < DREAMERV3_ACTION_SLOT_COUNT:
        raise ValueError(
            f"DreamerV3 categorical slot {requested_slot} outside "
            f"[0, {DREAMERV3_ACTION_SLOT_COUNT})"
        )

    requested_vector = dreamer_action_slot_vector(
        DREAMERV3_ACTION_SLOTS[requested_slot]
    )
    candidates: list[
        tuple[float, int, str, Action, tuple[float, ...]]
    ] = []
    for action in state.available_actions:
        if action.verb_name not in _VERBS:
            continue
        slot = DREAMERV3_ACTION_SLOT_INDEX[
            dreamer_action_slot_key(state, action)
        ]
        slot_vector = dreamer_action_slot_vector(DREAMERV3_ACTION_SLOTS[slot])
        candidates.append(
            (
                _squared_distance(requested_vector, slot_vector),
                slot,
                action.signature,
                action,
                dreamer_relational_action_features(state, action),
            )
        )
    if not candidates:
        raise ValueError(
            "cannot project a DreamerV3 action from an empty primitive surface"
        )

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    best = candidates[0]
    tied = sum(abs(item[0] - best[0]) <= 1e-12 for item in candidates)
    return DreamerActionProjection(
        action=best[3],
        relational_features=best[4],
        requested_slot_index=requested_slot,
        slot_index=best[1],
        squared_distance=best[0],
        tied_candidates=tied,
    )


def dreamer_adapter_manifest() -> dict[str, object]:
    return {
        "condition": DREAMERV3_CONDITION,
        "adapter_version": DREAMERV3_BASELINE_VERSION,
        "upstream_repository": DREAMERV3_UPSTREAM_REPOSITORY,
        "upstream_commit": DREAMERV3_UPSTREAM_COMMIT,
        "state_representation": "current-relational-public-state-v3+latest-http-status",
        "available_action_observation": "240-slot-relational-mask",
        "actor_action_space": "categorical-relational-slot-240",
        "legal_action_adapter": "nearest-current-relational-slot",
        "slot_projection_embedding": "verb-route-profile-object-one-hot-v1",
        "slot_count": DREAMERV3_ACTION_SLOT_COUNT,
        "oracle_information": False,
        "upstream_agent_modified": False,
    }
