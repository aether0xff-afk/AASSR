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
    STATUS_LOSS_WEIGHT,
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


def test_base_world_model_uses_dedicated_status_loss() -> None:
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
    assert diagnostics["last_status_training_loss"] > 0.0


def test_mixture_world_model_uses_dedicated_status_loss() -> None:
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
    assert diagnostics["last_status_training_loss"] > 0.0
