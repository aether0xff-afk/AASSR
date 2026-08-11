from __future__ import annotations

from typing import Sequence

from .current_relational_state import (
    ACTION_COUNT_INDEX,
    AUDIT_PRESSURE_INDEX,
    CONTROL_SIZE,
    KNOWN_OBJECT_INDEX,
    KNOWN_PROFILE_INDEX,
    KNOWN_ROUTE_INDEX,
    OBJECT_REQUEST_FRACTION_INDEX,
    REL_DESCRIPTOR_SIZE,
    REQUEST_FRACTION_INDEX,
    REQUEST_USAGE_INDEX,
    ROLE_START_INDEX,
    SESSION_REMAINING_INDEX,
    UNIQUE_ACTION_COUNT_INDEX,
    WORKFLOW_PROGRESS_INDEX,
)
from .dreamerv3_baseline import DREAMERV3_ACTION_SLOTS
from .pentest_curriculum_env import PROFILE_RELATIONS, ROUTE_RELATIONS
from .types import Action, StateSnapshot


def _canonical_action(slot: int) -> Action:
    """Materialize one anonymous Action per relational legal-action slot."""
    verb, route_role, profile_role, object_role = DREAMERV3_ACTION_SLOTS[slot]
    route_id = f"imag-route-{route_role}"
    profile_id = (
        "profile-browse"
        if profile_role == "browse"
        else f"imag-profile-{profile_role}"
    )
    parameters: dict[str, str] = {
        "route_id": route_id,
        "profile_id": profile_id,
    }
    if verb == "request_object":
        parameters["object_id"] = f"imag-object-{object_role}"
    return Action(verb, parameters=parameters)


def _role_fill(
    facts: set[str],
    *,
    prefix: str,
    id_prefix: str,
    existing: int,
    target: int,
    probabilities: Sequence[float],
    roles: Sequence[str],
) -> None:
    remaining = max(0, target - existing)
    if not remaining:
        return
    weights = [max(0.0, float(value)) for value in probabilities]
    total = sum(weights)
    if total <= 1e-12:
        weights = [float(role == "unknown") for role in roles]
        total = sum(weights) or 1.0
    counts = [int(remaining * value / total) for value in weights]
    for _ in range(remaining - sum(counts)):
        index = max(
            range(len(roles)),
            key=lambda item: (
                remaining * weights[item] / total - counts[item],
                -item,
            ),
        )
        counts[index] += 1
    cursor = 0
    for role, count in zip(roles, counts, strict=True):
        for _ in range(count):
            identifier = f"{id_prefix}{cursor:02d}"
            cursor += 1
            facts.add(f"known_{prefix}:{identifier}")
            if role != "unknown":
                facts.add(f"observed_{prefix}_role:{identifier}:{role}")


def decode_relational_state_v2(
    predicted_descriptor: Sequence[float],
    mask_probabilities: Sequence[float],
    *,
    scaffold: StateSnapshot,
    predicted_terminal: int,
    source: str,
) -> StateSnapshot:
    """Decode a relational future without recreating concrete episode IDs.

    ``available_actions`` contains exactly one anonymous Action per predicted
    structural legal-action slot. Observable concrete multiplicity is kept only
    in descriptor metadata. The four-way terminal head is authoritative over
    noisy auxiliary status bits.
    """
    from .current_relational_codec import (
        ACTION_SLOT_COUNT,
        TERMINAL_ACTIVE,
        TERMINAL_FAILURE,
        TERMINAL_SUCCESS,
        TERMINAL_TRUNCATION,
    )

    values = [max(0.0, min(1.0, float(value))) for value in predicted_descriptor]
    if len(values) != REL_DESCRIPTOR_SIZE:
        raise ValueError(
            f"unexpected relational descriptor size {len(values)} != {REL_DESCRIPTOR_SIZE}"
        )
    if len(mask_probabilities) != ACTION_SLOT_COUNT:
        raise ValueError("unexpected relational action-mask size")
    terminal = int(predicted_terminal)
    if terminal not in {
        TERMINAL_ACTIVE,
        TERMINAL_SUCCESS,
        TERMINAL_FAILURE,
        TERMINAL_TRUNCATION,
    }:
        raise ValueError(f"unknown relational terminal class: {terminal}")

    vector = list(float(value) for value in scaffold.vector)
    if len(vector) < 11:
        vector.extend([0.0] * (11 - len(vector)))
    for index in range(CONTROL_SIZE):
        vector[index] = values[index]
    vector[AUDIT_PRESSURE_INDEX] = 0.0
    vector[REQUEST_USAGE_INDEX] = values[REQUEST_USAGE_INDEX]
    vector[SESSION_REMAINING_INDEX] = 0.0

    workflow_fraction = values[WORKFLOW_PROGRESS_INDEX]
    vector[WORKFLOW_PROGRESS_INDEX] = workflow_fraction

    success = terminal == TERMINAL_SUCCESS
    failed = terminal == TERMINAL_FAILURE
    truncated = terminal == TERMINAL_TRUNCATION
    vector[3] = float(success)
    vector[4] = float(failed)
    vector[5] = float(failed)
    vector[6] = float(truncated)

    if success or failed:
        active_slots: tuple[int, ...] = ()
    else:
        active_slots = tuple(
            index
            for index, probability in enumerate(mask_probabilities)
            if float(probability) >= 0.5
        )
        # An active state must have at least one legal structural action. During
        # early/noisy prediction, fall back to the highest-probability slot rather
        # than manufacturing concrete duplicates. Truncation may legitimately have
        # an empty mask, so it receives no fallback.
        if not active_slots and terminal == TERMINAL_ACTIVE:
            best = max(
                range(ACTION_SLOT_COUNT),
                key=lambda index: (float(mask_probabilities[index]), -index),
            )
            active_slots = (best,)
    actions = tuple(_canonical_action(slot) for slot in active_slots)

    facts: set[str] = {"imagined_relational_world"}
    route_ids: set[str] = set()
    profile_ids: set[str] = set()
    object_ids: set[str] = set()

    for slot, action in zip(active_slots, actions, strict=True):
        _, route_role, profile_role, object_role = DREAMERV3_ACTION_SLOTS[slot]
        route_id = str(action.parameters["route_id"])
        profile_id = str(action.parameters["profile_id"])
        route_ids.add(route_id)
        profile_ids.add(profile_id)
        facts.add(f"known_route:{route_id}")
        facts.add(f"known_profile:{profile_id}")
        if route_role != "unknown":
            facts.add(f"observed_route_role:{route_id}:{route_role}")
        if profile_role not in {"unknown", "browse"}:
            facts.add(f"observed_profile_role:{profile_id}:{profile_role}")

        object_id = action.parameters.get("object_id")
        if object_id is not None:
            object_id = str(object_id)
            object_ids.add(object_id)
            facts.add(f"known_object:{object_id}")
            if object_role == "own":
                facts.add(f"observed_own_object:{object_id}")
            elif object_role == "target":
                facts.add(f"observed_target_object:{object_id}")
            elif object_role == "tried":
                facts.add(f"tried_object:{object_id}")

    route_target = int(round(values[KNOWN_ROUTE_INDEX] * 32.0))
    profile_target = int(round(values[KNOWN_PROFILE_INDEX] * 32.0))
    object_target = int(round(values[KNOWN_OBJECT_INDEX] * 32.0))
    route_start = ROLE_START_INDEX
    profile_start = route_start + len(ROUTE_RELATIONS)

    _role_fill(
        facts,
        prefix="route",
        id_prefix="imag-route-extra-",
        existing=len(route_ids),
        target=route_target,
        probabilities=values[route_start:profile_start],
        roles=ROUTE_RELATIONS,
    )
    _role_fill(
        facts,
        prefix="profile",
        id_prefix="imag-profile-extra-",
        existing=len(profile_ids),
        target=profile_target,
        probabilities=values[
            profile_start : profile_start + len(PROFILE_RELATIONS)
        ],
        roles=PROFILE_RELATIONS,
    )

    for index in range(max(0, object_target - len(object_ids))):
        object_id = f"imag-object-extra-{index:02d}"
        object_ids.add(object_id)
        facts.add(f"known_object:{object_id}")

    own_index = profile_start + len(PROFILE_RELATIONS)
    target_index = own_index + 1
    tried_index = target_index + 1
    if object_ids:
        ordered = sorted(object_ids)
        if values[own_index] >= 0.5 and not any(
            fact.startswith("observed_own_object:") for fact in facts
        ):
            facts.add(f"observed_own_object:{ordered[0]}")
        if values[target_index] >= 0.5 and not any(
            fact.startswith("observed_target_object:") for fact in facts
        ):
            facts.add(f"observed_target_object:{ordered[-1]}")
        tried_count = min(
            len(ordered),
            int(round(values[tried_index] * 32.0)),
        )
        for object_id in ordered[:tried_count]:
            facts.add(f"tried_object:{object_id}")

    if values[0] >= 0.5:
        facts.add("authenticated")
    if values[1] >= 0.5:
        facts.add("workflow_completed")
    if values[2] >= 0.5:
        facts.add("proof_acquired")
    if success:
        facts.add("success")
    elif failed:
        facts.update(("failed", "locked"))
    elif truncated:
        facts.add("rate_limited")

    metadata = dict(scaffold.metadata)
    action_count_fraction = 0.0 if success or failed else values[ACTION_COUNT_INDEX]
    unique_count_fraction = 0.0 if success or failed else values[UNIQUE_ACTION_COUNT_INDEX]
    request_fraction = 0.0 if success or failed else values[REQUEST_FRACTION_INDEX]
    object_request_fraction = (
        0.0 if success or failed else values[OBJECT_REQUEST_FRACTION_INDEX]
    )
    metadata.update(
        {
            "imagined_relational_world": True,
            "imagined_relational_source": source,
            "imagined_action_surface": "one-action-per-relational-legal-slot",
            "imagined_terminal_class": terminal,
            "relational_workflow_progress": workflow_fraction,
            "relational_action_count_fraction": action_count_fraction,
            "relational_unique_action_count_fraction": unique_count_fraction,
            "relational_request_fraction": request_fraction,
            "relational_object_request_fraction": object_request_fraction,
        }
    )
    scale = metadata.get("workflow_progress_scale")
    try:
        scale_value = max(1.0, float(scale)) if scale is not None else None
    except (TypeError, ValueError):
        scale_value = None
    if scale_value is not None:
        progress = max(0, int(round(workflow_fraction * scale_value)))
        metadata["workflow_progress"] = progress
        facts.add(f"workflow_progress:{progress}")

    return StateSnapshot(
        vector=tuple(vector),
        facts=frozenset(facts),
        available_actions=actions,
        goal_progress=1.0 if success else float(scaffold.goal_progress),
        metadata=metadata,
    )
