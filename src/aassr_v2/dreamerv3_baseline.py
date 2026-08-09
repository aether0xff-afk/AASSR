from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .pentest_agent_main_test import ACTION_FEATURE_SIZE
from .pentest_curriculum_env import (
    OBJECT_RELATIONS,
    PROFILE_RELATIONS,
    ROUTE_RELATIONS,
    relational_action_features,
)
from .current_generation import relational_state_vector
from .types import Action, StateSnapshot


DREAMERV3_CONDITION = "dreamerv3_relational"
DREAMERV3_UPSTREAM_REPOSITORY = "danijar/dreamerv3"
# Pin the exact upstream code used by this adapter contract. The experiment
# runner records the actual checkout SHA and refuses a mismatch by default.
DREAMERV3_UPSTREAM_COMMIT = "e3f02248693a79dc8b0ebd62c93683888ddaccfe"
DREAMERV3_BASELINE_VERSION = "official-dreamerv3-relational-projection-v1"

# DreamerV3 expects a fixed action space, whereas the pentest benchmark exposes
# a state-dependent set of legal HTTP actions. We therefore expose the complete
# *relational* discrete surface as observation metadata while leaving the
# official DreamerV3 actor itself untouched. The actor uses its supported fixed
# continuous action head; the environment deterministically projects that vector
# onto the nearest currently legal relational action feature.
#
# The slot vocabulary contains no concrete route/profile/object identifiers and
# therefore stays invariant when a benchmark seed renames those identifiers.
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


@dataclass(frozen=True, slots=True)
class DreamerActionProjection:
    action: Action
    relational_features: tuple[float, ...]
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
    if action.verb_name not in {"request", "request_object"}:
        raise ValueError(
            f"DreamerV3 baseline does not expose non-HTTP primitive {action.verb_name!r}"
        )
    route_role, profile_role, object_role = _observed_roles(state, action)
    if action.verb_name == "request":
        object_role = "none"
    elif object_role == "none":
        raise ValueError("request_object action is missing an object relation")
    key = (action.verb_name, route_role, profile_role, object_role)
    if key not in DREAMERV3_ACTION_SLOT_INDEX:
        raise AssertionError(f"unregistered DreamerV3 relational action slot: {key}")
    return key


def dreamer_action_surface_mask(state: StateSnapshot) -> tuple[float, ...]:
    values = [0.0] * DREAMERV3_ACTION_SLOT_COUNT
    for action in state.available_actions:
        if action.verb_name not in {"request", "request_object"}:
            # Current pentest Dreamer baseline intentionally operates only on
            # real primitive HTTP actions. AASSR Skill macros are not an
            # environment action and are therefore never exposed to baselines.
            continue
        values[
            DREAMERV3_ACTION_SLOT_INDEX[dreamer_action_slot_key(state, action)]
        ] = 1.0
    return tuple(values)


def dreamer_observation_vector(state: StateSnapshot) -> tuple[float, ...]:
    """Use exactly the current relational Policy state representation."""

    return tuple(float(value) for value in relational_state_vector(state))


def dreamer_relational_action_features(
    state: StateSnapshot,
    action: Action,
) -> tuple[float, ...]:
    values = tuple(float(value) for value in relational_action_features(state, action))
    if len(values) != ACTION_FEATURE_SIZE:
        raise AssertionError("DreamerV3 relational action feature size drift")
    return values


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("DreamerV3 action projection dimensionality mismatch")
    return float(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def project_dreamer_action(
    state: StateSnapshot,
    proposal: Sequence[float],
) -> DreamerActionProjection:
    """Project a fixed continuous Dreamer action onto the legal relational set.

    The projection uses only the public current state and its available action
    surface. It never reads the hidden benchmark scenario, reward, target role,
    future state, or success trajectory. Structurally identical candidates have
    identical distances; their concrete signature is used only as a deterministic
    final tie-break, matching the existing relational DQN convention.
    """

    proposal = tuple(float(value) for value in proposal)
    if len(proposal) != ACTION_FEATURE_SIZE:
        raise ValueError(
            f"DreamerV3 action proposal has {len(proposal)} values; "
            f"expected {ACTION_FEATURE_SIZE}"
        )

    candidates: list[tuple[float, int, str, Action, tuple[float, ...]]] = []
    for action in state.available_actions:
        if action.verb_name not in {"request", "request_object"}:
            continue
        features = dreamer_relational_action_features(state, action)
        slot = DREAMERV3_ACTION_SLOT_INDEX[dreamer_action_slot_key(state, action)]
        candidates.append(
            (
                _squared_distance(proposal, features),
                slot,
                action.signature,
                action,
                features,
            )
        )
    if not candidates:
        raise ValueError("cannot project a DreamerV3 action from an empty primitive surface")

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    best = candidates[0]
    tied = sum(abs(item[0] - best[0]) <= 1e-12 for item in candidates)
    return DreamerActionProjection(
        action=best[3],
        relational_features=best[4],
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
        "state_representation": "current-relational-state-vector",
        "available_action_observation": "240-slot-relational-mask",
        "actor_action_space": f"continuous-relational-vector-{ACTION_FEATURE_SIZE}",
        "legal_action_adapter": "nearest-current-relational-action",
        "slot_count": DREAMERV3_ACTION_SLOT_COUNT,
        "oracle_information": False,
        "upstream_agent_modified": False,
    }
