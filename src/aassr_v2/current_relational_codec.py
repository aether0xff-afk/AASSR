from __future__ import annotations

from statistics import fmean
from typing import Sequence

from .current_generation import relational_action_key
from .current_relational_state import (
    REL_DESCRIPTOR_SIZE,
    relational_state_descriptor_v2,
    relational_state_vector_v2,
)
from .dreamerv3_baseline import (
    DREAMERV3_ACTION_SLOTS,
    DREAMERV3_ACTION_SLOT_INDEX,
    dreamer_action_slot_key,
)
from .pentest_curriculum_env import PROFILE_RELATIONS, ROUTE_RELATIONS
from .skills import SKILL_VERB
from .types import Action, Prediction, StateSnapshot


ACTION_SLOT_COUNT = len(DREAMERV3_ACTION_SLOTS)
TERMINAL_ACTIVE = 0
TERMINAL_SUCCESS = 1
TERMINAL_FAILURE = 2
TERMINAL_TRUNCATION = 3
TERMINAL_CLASSES = 4


def descriptor(state: StateSnapshot) -> tuple[float, ...]:
    values = tuple(float(v) for v in relational_state_descriptor_v2(state))
    if len(values) != REL_DESCRIPTOR_SIZE:
        raise AssertionError(
            f"relational descriptor size drift: {len(values)} != {REL_DESCRIPTOR_SIZE}"
        )
    return values


def _control(state: StateSnapshot, index: int) -> float:
    return float(state.vector[index]) if index < len(state.vector) else 0.0


def terminal_class(state: StateSnapshot) -> int:
    """Classify task terminal semantics, including non-failure truncation.

    The transfer protocol treats rate limiting as truncation with reward 0, not
    true failure -1. Curriculum snapshots deliberately keep legal actions after a
    rate-limit event, so available_actions alone cannot define terminality.
    """
    facts = state.facts
    success = (
        state.goal_progress >= 1.0
        or "success" in facts
        or _control(state, 3) >= 0.5
    )
    if success:
        return TERMINAL_SUCCESS

    rate_limited = "rate_limited" in facts or _control(state, 6) >= 0.5
    if rate_limited:
        return TERMINAL_TRUNCATION

    locked = "locked" in facts or _control(state, 5) >= 0.5
    failed = "failed" in facts or _control(state, 4) >= 0.5
    if locked and failed:
        return TERMINAL_FAILURE

    if not state.available_actions:
        return TERMINAL_FAILURE if failed else TERMINAL_TRUNCATION
    return TERMINAL_ACTIVE


def legal_action_mask(state: StateSnapshot) -> tuple[float, ...]:
    mask = [0.0] * ACTION_SLOT_COUNT
    for action in state.available_actions:
        if action.verb_name == SKILL_VERB:
            continue
        slot = DREAMERV3_ACTION_SLOT_INDEX[dreamer_action_slot_key(state, action)]
        mask[slot] = 1.0
    return tuple(mask)


def transition_target(
    state: StateSnapshot,
    action: Action,
    next_state: StateSnapshot,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], int]:
    """Exact permutation-invariant input/target contract used by Prophecy."""
    return (
        relational_state_vector_v2(state) + relational_action_key(state, action),
        descriptor(next_state),
        legal_action_mask(next_state),
        terminal_class(next_state),
    )


def _canonical_action(slot: int, *, variant: int = 0) -> Action:
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
    if variant:
        parameters["imagined_variant"] = str(variant)
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
    weights = [max(0.0, float(v)) for v in probabilities]
    total = sum(weights)
    if total <= 1e-12:
        weights = [float(role == "unknown") for role in roles]
        total = sum(weights) or 1.0
    counts = [int(remaining * value / total) for value in weights]
    for _ in range(remaining - sum(counts)):
        index = max(
            range(len(roles)),
            key=lambda i: (remaining * weights[i] / total - counts[i], -i),
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


def decode_relational_state(
    predicted_descriptor: Sequence[float],
    mask_probabilities: Sequence[float],
    *,
    scaffold: StateSnapshot,
    predicted_terminal: int,
    source: str,
) -> StateSnapshot:
    """Legacy-compatible relational decoder; current v2 installs its replacement."""
    values = [max(0.0, min(1.0, float(v))) for v in predicted_descriptor]
    if len(values) != REL_DESCRIPTOR_SIZE:
        raise ValueError("unexpected relational descriptor size")
    if len(mask_probabilities) != ACTION_SLOT_COUNT:
        raise ValueError("unexpected relational action-mask size")

    vector = list(float(v) for v in scaffold.vector)
    if len(vector) < 11:
        vector.extend([0.0] * (11 - len(vector)))
    for index in range(7):
        vector[index] = values[index]
    vector[8] = values[7]
    vector[10] = values[8]

    success = predicted_terminal == TERMINAL_SUCCESS
    failed = predicted_terminal == TERMINAL_FAILURE
    truncated = predicted_terminal == TERMINAL_TRUNCATION
    vector[3] = float(success)
    vector[4] = float(failed)
    vector[6] = float(truncated)

    if success or failed:
        action_slots: tuple[int, ...] = ()
        actions: tuple[Action, ...] = ()
    else:
        unique_count = max(1, min(32, int(round(values[-3] * 32.0))))
        total_count = max(
            unique_count,
            min(128, int(round(values[-4] * 128.0))),
        )
        ranked = sorted(
            range(ACTION_SLOT_COUNT),
            key=lambda index: (-float(mask_probabilities[index]), index),
        )
        active_slots = tuple(ranked[:unique_count])
        expanded_slots = list(active_slots)
        while len(expanded_slots) < total_count:
            expanded_slots.append(
                active_slots[(len(expanded_slots) - unique_count) % len(active_slots)]
            )
        action_slots = tuple(expanded_slots)
        per_slot_seen: dict[int, int] = {}
        materialized = []
        for slot in action_slots:
            seen = per_slot_seen.get(slot, 0)
            materialized.append(_canonical_action(slot, variant=seen))
            per_slot_seen[slot] = seen + 1
        actions = tuple(materialized)

    facts: set[str] = {"imagined_relational_world"}
    route_ids: set[str] = set()
    profile_ids: set[str] = set()
    object_ids: set[str] = set()

    for slot, action in zip(action_slots, actions, strict=True):
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

    route_target = int(round(values[9] * 32.0))
    profile_target = int(round(values[10] * 32.0))
    object_target = int(round(values[11] * 32.0))
    route_start = 12
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
        probabilities=values[profile_start : profile_start + len(PROFILE_RELATIONS)],
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
        for object_id in ordered[
            : min(len(ordered), int(round(values[tried_index] * 32.0)))
        ]:
            facts.add(f"tried_object:{object_id}")

    if values[0] >= 0.5:
        facts.add("authenticated")
    if values[1] >= 0.5:
        facts.add("workflow_completed")
    if values[2] >= 0.5:
        facts.add("proof_acquired")
    if values[5] >= 0.5:
        facts.add("locked")
    if truncated or values[6] >= 0.5:
        facts.add("rate_limited")
    if success:
        facts.add("success")
    if failed:
        facts.add("failed")

    metadata = dict(scaffold.metadata)
    metadata.update(
        {
            "imagined_relational_world": True,
            "imagined_relational_source": source,
            "imagined_action_surface": "predicted_relational_legal_mask",
            "imagined_terminal_class": int(predicted_terminal),
        }
    )
    return StateSnapshot(
        vector=tuple(vector),
        facts=frozenset(facts),
        available_actions=actions,
        goal_progress=1.0 if success else float(scaffold.goal_progress),
        metadata=metadata,
    )


def _mask_jaccard(left: Sequence[float], right: Sequence[float]) -> float:
    a = {i for i, value in enumerate(left) if float(value) >= 0.5}
    b = {i for i, value in enumerate(right) if float(value) >= 0.5}
    return 1.0 if not (a or b) else len(a & b) / len(a | b)


def semantic_prediction_score(
    predictions: Sequence[Prediction],
    actual: StateSnapshot,
) -> float:
    if not predictions:
        return 0.0
    target_descriptor = descriptor(actual)
    target_mask = legal_action_mask(actual)
    target_terminal = terminal_class(actual)
    scores = []
    for prediction in predictions:
        predicted = prediction.next_state
        semantic_error = fmean(
            abs(left - right)
            for left, right in zip(
                descriptor(predicted),
                target_descriptor,
                strict=True,
            )
        )
        scores.append(
            0.55 * max(0.0, 1.0 - semantic_error)
            + 0.30 * _mask_jaccard(legal_action_mask(predicted), target_mask)
            + 0.15 * float(terminal_class(predicted) == target_terminal)
        )
    return max(scores)
