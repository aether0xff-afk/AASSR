from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("torch")

from aassr_v2.current_relational_mixture_model import RelationalMixtureProphecyConfig
from aassr_v2.current_relational_model import RelationalProphecyConfig
from aassr_v2.current_relational_state_v3 import (
    STATUS_CODES_V3,
    install_status_aware_relational_contract,
)
from aassr_v2.current_status_models import (
    STATUS_CLASS_WEIGHT_CAP,
    STATUS_LOSS_WEIGHT,
    STATUS_OBJECTIVE,
    StatusAwareConditionalMixtureRelationalProphecy,
    StatusAwareRelationalStochasticProphecy,
)
from aassr_v2.pentest_agent_main_test import AGENT_STATE_SIZE
from aassr_v2.pentest_transfer_stages import TRANSFER_STAGES, TransferDiagnosticWorld


install_status_aware_relational_contract()


def _with_status(state, status: int):
    vector = list(state.vector)
    if len(vector) < AGENT_STATE_SIZE:
        vector.extend([0.0] * (AGENT_STATE_SIZE - len(vector)))
    start = AGENT_STATE_SIZE - len(STATUS_CODES_V3)
    for index in range(len(STATUS_CODES_V3)):
        vector[start + index] = 0.0
    vector[start + STATUS_CODES_V3.index(status)] = 1.0
    facts = {fact for fact in state.facts if not fact.startswith("last_status:")}
    facts.add(f"last_status:{status}")
    return replace(state, vector=tuple(vector), facts=frozenset(facts))


def _transition_states():
    world = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[0])
    before = world.snapshot()
    action = before.available_actions[0]
    return before, action, _with_status(before, 200), _with_status(before, 404)


def test_base_world_model_uses_balanced_categorical_status_loss() -> None:
    before, action, ok, missing = _transition_states()
    model = StatusAwareRelationalStochasticProphecy(
        seed=7,
        device="cpu",
        config=RelationalProphecyConfig(
            hidden_units=8,
            ensemble_size=1,
            replay_capacity=8,
            batch_size=2,
            warmup_steps=2,
            gradient_steps_per_observation=1,
        ),
    )
    model.learn(before, action, ok)
    model.learn(before, action, missing)
    diagnostics = model.diagnostics()
    assert diagnostics["gradient_updates"] == 1
    assert diagnostics["status_supervision"] == 1
    assert diagnostics["status_output_channels"] == 8
    assert diagnostics["status_loss_weight"] == pytest.approx(STATUS_LOSS_WEIGHT)
    assert diagnostics["status_training_objective"] == STATUS_OBJECTIVE
    assert diagnostics["last_status_training_loss"] > 0.0
    assert 0.0 <= diagnostics["last_status_training_accuracy"] <= 1.0


def test_base_world_model_preserves_complete_learned_ensemble_mass() -> None:
    before, action, ok, _ = _transition_states()
    model = StatusAwareRelationalStochasticProphecy(
        seed=8,
        device="cpu",
        config=RelationalProphecyConfig(
            hidden_units=8,
            ensemble_size=3,
            replay_capacity=8,
            batch_size=1,
            warmup_steps=1,
            gradient_steps_per_observation=1,
        ),
    )
    model.learn(before, action, ok)
    rows = model.predict(before, action, samples=1)
    assert len(rows) == 3
    assert sum(float(row.outcome_probability) for row in rows) == pytest.approx(1.0)
    assert all(float(row.outcome_probability) == pytest.approx(1.0 / 3.0) for row in rows)
    assert model.diagnostics()["complete_learned_ensemble_mass"] == 1


def test_mixture_world_model_uses_balanced_categorical_status_loss() -> None:
    before, action, ok, missing = _transition_states()
    model = StatusAwareConditionalMixtureRelationalProphecy(
        seed=11,
        device="cpu",
        config=RelationalMixtureProphecyConfig(
            hidden_units=8,
            ensemble_size=1,
            mixture_components=2,
            replay_capacity=8,
            batch_size=2,
            warmup_steps=2,
            gradient_steps_per_observation=1,
        ),
    )
    model.learn(before, action, ok)
    model.learn(before, action, missing)
    diagnostics = model.diagnostics()
    assert diagnostics["gradient_updates"] == 1
    assert diagnostics["status_supervision"] == 1
    assert diagnostics["status_output_channels"] == 8
    assert diagnostics["status_loss_weight"] == pytest.approx(STATUS_LOSS_WEIGHT)
    assert diagnostics["status_training_objective"] == STATUS_OBJECTIVE
    assert diagnostics["last_status_training_loss"] > 0.0
    assert 0.0 <= diagnostics["last_status_training_accuracy"] <= 1.0


def test_status_balancing_uses_frequency_not_status_identity() -> None:
    before, action, ok, missing = _transition_states()
    forbidden = _with_status(before, 403)
    model = StatusAwareRelationalStochasticProphecy(
        seed=17,
        device="cpu",
        config=RelationalProphecyConfig(
            hidden_units=8,
            ensemble_size=1,
            replay_capacity=32,
            batch_size=16,
            warmup_steps=32,
            gradient_steps_per_observation=1,
        ),
    )
    for _ in range(8):
        model.learn(before, action, ok)
    model.learn(before, action, forbidden)
    model.learn(before, action, missing)

    diagnostics = model.diagnostics()
    assert diagnostics["status_count_200"] == 8
    assert diagnostics["status_count_403"] == 1
    assert diagnostics["status_count_404"] == 1
    assert diagnostics["status_class_weight_403"] == pytest.approx(
        diagnostics["status_class_weight_404"]
    )
    assert diagnostics["status_class_weight_403"] > diagnostics["status_class_weight_200"]
    assert diagnostics["status_class_weight_403"] <= STATUS_CLASS_WEIGHT_CAP
    assert diagnostics["status_class_weighting"] == (
        "inverse-sqrt-frequency-capped-normalized"
    )
