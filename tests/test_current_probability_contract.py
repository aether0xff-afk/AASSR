from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from aassr_v2.current_planner import CurrentFullyBatchedImaginationTree
from aassr_v2.current_relational_codec import ACTION_SLOT_COUNT, TERMINAL_CLASSES
from aassr_v2.current_relational_mixture_model import RelationalMixtureProphecyConfig
from aassr_v2.current_relational_model import RelationalPrediction
from aassr_v2.current_relational_state_v3 import (
    REL_DESCRIPTOR_SIZE_V3,
    STATUS_CODES_V3,
    STATUS_SIZE,
    STATUS_START_INDEX,
    decode_relational_state_v3,
    latest_status_code,
    latest_status_vector,
)
from aassr_v2.current_status_models import StatusAwareConditionalMixtureRelationalProphecy
from aassr_v2.pentest_agent_main_test import AGENT_STATE_SIZE
from aassr_v2.types import Action, StateSnapshot


def _state() -> StateSnapshot:
    action = Action(
        "request",
        parameters={"route_id": "route-01", "profile_id": "profile-browse"},
    )
    return StateSnapshot(
        vector=(0.0,) * AGENT_STATE_SIZE,
        facts=frozenset(),
        available_actions=(action,),
        metadata={},
    )


def _descriptor(status: int) -> list[float]:
    values = [0.0] * REL_DESCRIPTOR_SIZE_V3
    values[STATUS_START_INDEX + STATUS_CODES_V3.index(status)] = 1.0
    return values


def test_v3_decode_realizes_argmax_status_even_below_half_probability() -> None:
    state = _state()
    descriptor = [0.0] * REL_DESCRIPTOR_SIZE_V3
    distribution = [0.10] * STATUS_SIZE
    distribution[STATUS_CODES_V3.index(403)] = 0.30
    descriptor[STATUS_START_INDEX:] = distribution
    decoded = decode_relational_state_v3(
        descriptor,
        (1.0,) * ACTION_SLOT_COUNT,
        scaffold=state,
        predicted_terminal=0,
        source="categorical-test",
    )
    assert latest_status_code(decoded) == 403
    assert latest_status_vector(decoded) == tuple(
        float(code == 403) for code in STATUS_CODES_V3
    )
    assert decoded.metadata["relational_status_distribution"] == pytest.approx(
        tuple(distribution)
    )
    assert decoded.metadata["relational_status_probabilities"] == tuple(
        float(code == 403) for code in STATUS_CODES_V3
    )


def test_mixture_status_decoder_uses_softmax_geometry() -> None:
    model = StatusAwareConditionalMixtureRelationalProphecy(
        seed=7,
        device="cpu",
        config=RelationalMixtureProphecyConfig(
            hidden_units=8,
            ensemble_size=1,
            mixture_components=2,
            replay_capacity=8,
            batch_size=2,
            warmup_steps=2,
        ),
    )
    outputs = model.torch.zeros((1, 1, model.output_size), dtype=model.torch.float32)
    descriptors, _, _, _ = model._decoded_outputs(outputs)
    status = descriptors[
        0,
        0,
        :,
        STATUS_START_INDEX : STATUS_START_INDEX + STATUS_SIZE,
    ]
    assert model.torch.allclose(
        status.sum(dim=1),
        model.torch.ones(model.config.mixture_components),
    )
    assert model.torch.allclose(
        status,
        model.torch.full_like(status, 1.0 / STATUS_SIZE),
    )


def test_mode_merge_never_collapses_status_or_legal_surface() -> None:
    base_mask = [0.0] * ACTION_SLOT_COUNT
    base_mask[0] = 1.0
    terminal = [1.0] + [0.0] * (TERMINAL_CLASSES - 1)
    left = {
        "descriptor": _descriptor(403),
        "mask": list(base_mask),
        "terminal": list(terminal),
    }
    status_changed = {
        "descriptor": _descriptor(404),
        "mask": list(base_mask),
        "terminal": list(terminal),
    }
    mask_changed = {
        "descriptor": _descriptor(403),
        "mask": list(base_mask),
        "terminal": list(terminal),
    }
    mask_changed["mask"][1] = 1.0
    assert StatusAwareConditionalMixtureRelationalProphecy._mode_distance(
        left, status_changed
    ) == pytest.approx(1.0)
    assert StatusAwareConditionalMixtureRelationalProphecy._mode_distance(
        left, mask_changed
    ) == pytest.approx(1.0)


def test_active_mixture_keeps_all_learned_probability_mass_even_when_samples_is_one() -> None:
    model = StatusAwareConditionalMixtureRelationalProphecy(
        seed=11,
        device="cpu",
        config=RelationalMixtureProphecyConfig(
            hidden_units=8,
            ensemble_size=1,
            mixture_components=3,
            replay_capacity=8,
            batch_size=2,
            warmup_steps=2,
        ),
    )
    state = _state()
    descriptors = [[
        [_descriptor(200), _descriptor(403), _descriptor(404)]
    ]]
    mask = [1.0] + [0.0] * (ACTION_SLOT_COUNT - 1)
    masks = [[[list(mask), list(mask), list(mask)]]]
    terminal = [1.0] + [0.0] * (TERMINAL_CLASSES - 1)
    terminals = [[[list(terminal), list(terminal), list(terminal)]]]
    mixtures = [[[0.50, 0.30, 0.20]]]

    predictions = model._mixture_predictions(
        state=state,
        row_index=0,
        descriptors=descriptors,
        masks=masks,
        terminals=terminals,
        mixtures=mixtures,
        confidence=0.9,
        samples=1,
    )
    assert len(predictions) == 3
    assert sum(item.outcome_probability for item in predictions) == pytest.approx(1.0)
    assert sorted(
        (item.outcome_probability for item in predictions), reverse=True
    ) == pytest.approx([0.50, 0.30, 0.20])


def test_planner_does_not_topk_condition_a_complete_distribution() -> None:
    state = _state()
    predictions = tuple(
        RelationalPrediction(
            state,
            0.9,
            source=f"mode-{index}",
            outcome_probability=mass,
        )
        for index, mass in enumerate((0.50, 0.30, 0.20))
    )
    fake_tree = SimpleNamespace(
        prophecy=SimpleNamespace(complete_outcome_distribution=True)
    )
    normalized = CurrentFullyBatchedImaginationTree._normalized_predictions(
        fake_tree,
        predictions,
        limit=1,
    )
    assert len(normalized) == 3
    assert [mass for _, mass in normalized] == pytest.approx([0.50, 0.30, 0.20])
