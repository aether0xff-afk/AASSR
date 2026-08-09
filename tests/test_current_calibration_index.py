from __future__ import annotations

from statistics import fmean

from aassr_v2.current_runtime import FrozenReplayRelationalCalibratedProphecy
from aassr_v2.replay import ReplayBuffer, ReplayTransition
from aassr_v2.types import Action, Prediction, StateSnapshot


_REQUEST = Action(
    "request",
    parameters={"route_id": "route-a", "profile_id": "profile-browse"},
)
_OBJECT_REQUEST = Action(
    "request_object",
    parameters={
        "route_id": "route-a",
        "profile_id": "profile-read",
        "object_id": "object-a",
    },
)


class _FakeNeuralDelta:
    def __init__(self) -> None:
        self.gradient_updates = 0
        self.batch_calls: list[
            tuple[tuple[StateSnapshot, Action], ...]
        ] = []

    def predict_batch(self, states, actions, *, samples):
        assert samples == 1
        pairs = tuple(zip(states, actions, strict=True))
        self.batch_calls.append(pairs)
        return tuple(
            (Prediction(state, 1.0, source="fake-neural-delta"),)
            for state in states
        )


def _state(index: int) -> StateSnapshot:
    return StateSnapshot(
        vector=(float(index) / 1_000.0, 0.0, 0.0),
        facts=frozenset(
            {
                "known_route:route-a",
                "known_profile:profile-browse",
                "known_profile:profile-read",
                "known_object:object-a",
            }
        ),
        available_actions=(_REQUEST, _OBJECT_REQUEST),
    )


def _transition(index: int, action: Action) -> ReplayTransition:
    state = _state(index)
    return ReplayTransition(state, action, state, f"row-{index}")


def _add_holdout(replay: ReplayBuffer, transition: ReplayTransition) -> None:
    while replay.add(transition):
        pass


def _scalar_items(
    calibrated: FrozenReplayRelationalCalibratedProphecy,
    source: tuple[ReplayTransition, ...],
    state: StateSnapshot,
    action: Action,
) -> tuple[ReplayTransition, ...]:
    key = calibrated._key(state, action)
    return tuple(
        item
        for item in source
        if calibrated._key(item.state, item.action) == key
    )


def _scalar_reference(
    calibrated: FrozenReplayRelationalCalibratedProphecy,
    base: _FakeNeuralDelta,
    source: tuple[ReplayTransition, ...],
    state: StateSnapshot,
    action: Action,
) -> float:
    items = _scalar_items(calibrated, source, state, action)
    if len(items) < calibrated.minimum_count:
        return 0.0
    selected = items[-calibrated.evaluation_limit :]
    rows = base.predict_batch(
        tuple(item.state for item in selected),
        tuple(item.action for item in selected),
        samples=1,
    )
    scores = []
    for item, row in zip(selected, rows, strict=True):
        predicted = row[0].next_state
        error = fmean(
            abs(left - right)
            for left, right in zip(
                predicted.vector,
                item.next_state.vector,
                strict=True,
            )
        )
        terminal_match = (
            calibrated._terminal_class(predicted)
            == calibrated._terminal_class(item.next_state)
        )
        action_ratio = min(
            len(predicted.available_actions),
            len(item.next_state.available_actions),
        ) / float(
            max(
                1,
                len(predicted.available_actions),
                len(item.next_state.available_actions),
            )
        )
        scores.append(
            max(0.0, 1.0 - error)
            * float(terminal_match)
            * action_ratio
        )
    return max(0.0, min(1.0, fmean(scores) if scores else 0.0))


def _calibrator(
    replay: ReplayBuffer,
    base: _FakeNeuralDelta | None = None,
) -> FrozenReplayRelationalCalibratedProphecy:
    return FrozenReplayRelationalCalibratedProphecy(
        base or _FakeNeuralDelta(),
        replay,
        minimum_count=2,
        evaluation_limit=48,
        refresh_stride=32,
    )


def test_calibration_index_matches_ordered_scalar_reference_and_limit() -> None:
    replay = ReplayBuffer(capacity=256, holdout_stride=2)
    for index in range(80):
        action = _OBJECT_REQUEST if index % 4 == 1 else _REQUEST
        _add_holdout(replay, _transition(index, action))

    base = _FakeNeuralDelta()
    calibrated = _calibrator(replay, base)
    source = replay.holdout()
    calibrated.freeze_holdout(source)

    request_items = _scalar_items(calibrated, source, source[0].state, _REQUEST)
    object_items = _scalar_items(
        calibrated,
        source,
        source[0].state,
        _OBJECT_REQUEST,
    )
    assert len(request_items) == 60
    assert len(object_items) == 20

    reference_base = _FakeNeuralDelta()
    expected_request = _scalar_reference(
        calibrated,
        reference_base,
        source,
        source[0].state,
        _REQUEST,
    )
    assert calibrated._calibration(source[0].state, _REQUEST) == expected_request

    request_key = calibrated._key(source[0].state, _REQUEST)
    assert len(calibrated._holdout_index[request_key]) == len(request_items)
    assert all(
        indexed is scalar
        for indexed, scalar in zip(
            calibrated._holdout_index[request_key],
            request_items,
            strict=True,
        )
    )
    assert base.batch_calls[0] == tuple(
        (item.state, item.action) for item in request_items[-48:]
    )

    batch_calls = len(base.batch_calls)
    assert calibrated._calibration(source[0].state, _REQUEST) == expected_request
    assert len(base.batch_calls) == batch_calls

    expected_object = _scalar_reference(
        calibrated,
        reference_base,
        source,
        source[0].state,
        _OBJECT_REQUEST,
    )
    assert calibrated._calibration(
        source[0].state,
        _OBJECT_REQUEST,
    ) == expected_object
    assert base.batch_calls[-1] == tuple(
        (item.state, item.action) for item in object_items
    )

    diagnostics = calibrated.diagnostics()
    assert diagnostics["calibration_index_rebuilds"] == 1
    assert diagnostics["calibration_index_hits"] == 2
    assert diagnostics["calibration_index_rows"] == len(source)
    assert diagnostics["calibration_batch_refreshes"] == 2
    assert diagnostics["calibration_batch_rows"] == 48 + len(object_items)


def test_calibration_index_preserves_freeze_release_and_append_semantics() -> None:
    replay = ReplayBuffer(capacity=128, holdout_stride=2)
    for index in range(8):
        _add_holdout(replay, _transition(index, _REQUEST))

    calibrated = _calibrator(replay)
    frozen = replay.holdout()
    calibrated.freeze_holdout(frozen)
    assert calibrated._calibration(frozen[0].state, _REQUEST) == _scalar_reference(
        calibrated,
        _FakeNeuralDelta(),
        frozen,
        frozen[0].state,
        _REQUEST,
    )
    assert calibrated.calibration_index_rebuilds == 1

    train_only = _transition(100, _OBJECT_REQUEST)
    assert replay.add(train_only) is True
    calibrated._calibration(frozen[0].state, _OBJECT_REQUEST)
    assert calibrated.calibration_index_rebuilds == 1

    calibrated.release_holdout()
    calibrated._calibration(frozen[0].state, _REQUEST)
    assert calibrated.calibration_index_rebuilds == 1

    appended = _transition(101, _OBJECT_REQUEST)
    assert replay.add(appended) is False
    live = replay.holdout()
    calibrated._calibration(live[0].state, _OBJECT_REQUEST)
    assert calibrated.calibration_index_rebuilds == 2
    indexed_object = calibrated._holdout_index[
        calibrated._key(live[0].state, _OBJECT_REQUEST)
    ]
    assert indexed_object == (appended,)
    assert indexed_object[0] is appended

    calibrated.freeze_holdout(live)
    calibrated._calibration(live[0].state, _OBJECT_REQUEST)
    assert calibrated._holdout_index_source is calibrated._frozen_holdout
    replay.add(_transition(102, _REQUEST))
    later = _transition(103, _REQUEST)
    assert replay.add(later) is False
    calibrated._calibration(live[0].state, _REQUEST)
    assert calibrated.calibration_index_rebuilds == 2
    calibrated.release_holdout()
    calibrated._calibration(live[0].state, _REQUEST)
    assert calibrated.calibration_index_rebuilds == 3
    assert calibrated._holdout_index_source is not None
    assert all(
        indexed is current
        for indexed, current in zip(
            calibrated._holdout_index_source,
            replay.holdout(),
            strict=True,
        )
    )


def test_calibration_index_rebuilds_safely_after_holdout_eviction() -> None:
    replay = ReplayBuffer(capacity=3, holdout_stride=2)
    original = tuple(_transition(index, _REQUEST) for index in range(3))
    for transition in original:
        _add_holdout(replay, transition)

    calibrated = _calibrator(replay)
    calibrated._calibration(original[0].state, _REQUEST)
    assert calibrated.calibration_index_rebuilds == 1

    assert replay.add(_transition(200, _OBJECT_REQUEST)) is True
    replacement = _transition(201, _OBJECT_REQUEST)
    assert replay.add(replacement) is False
    current = replay.holdout()
    assert len(current) == 3
    assert original[0] not in current

    actual = calibrated._calibration(current[0].state, _OBJECT_REQUEST)
    expected = _scalar_reference(
        calibrated,
        _FakeNeuralDelta(),
        current,
        current[0].state,
        _OBJECT_REQUEST,
    )
    assert actual == expected
    assert calibrated.calibration_index_rebuilds == 2
    assert calibrated.calibration_index_rows == 6
    assert all(
        indexed is current_item
        for indexed, current_item in zip(
            calibrated._holdout_index_source or (),
            current,
            strict=True,
        )
    )
    assert all(
        item is not original[0]
        for bucket in calibrated._holdout_index.values()
        for item in bucket
    )
