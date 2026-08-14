from __future__ import annotations

import math

import pytest

from aassr_v2.core.plugin_contract import (
    ObservationField,
    PluginObservation,
    PluginSchema,
    PluginStepResult,
    ValueKind,
    validate_step_result,
)


SCHEMA = PluginSchema(
    plugin_id="control-boundary-test",
    version="v1",
    observations=(ObservationField("value", ValueKind.SCALAR),),
    actions=(),
)


def _result(**kwargs) -> PluginStepResult:
    return PluginStepResult(
        observation=PluginObservation({"value": 1.0}),
        **kwargs,
    )


def test_valid_step_result_passes_control_boundary_validation() -> None:
    validate_step_result(
        SCHEMA,
        _result(
            reward=1.0,
            terminated=True,
            diagnostics={"wire_note": "public-only"},
        ),
    )


@pytest.mark.parametrize(
    "reserved",
    ("external_reward", "terminated", "truncated"),
)
def test_plugin_diagnostics_cannot_shadow_core_control_channels(
    reserved: str,
) -> None:
    with pytest.raises(ValueError, match="shadow Core control channels"):
        validate_step_result(
            SCHEMA,
            _result(diagnostics={reserved: 999}),
        )


@pytest.mark.parametrize("reward", (math.nan, math.inf, -math.inf))
def test_non_finite_reward_is_rejected(reward: float) -> None:
    with pytest.raises(ValueError, match="reward must be finite"):
        validate_step_result(SCHEMA, _result(reward=reward))


def test_boolean_reward_is_rejected() -> None:
    with pytest.raises(TypeError, match="reward must be a real number"):
        validate_step_result(SCHEMA, _result(reward=True))


@pytest.mark.parametrize("field", ("terminated", "truncated", "error"))
def test_control_flags_must_be_real_booleans(field: str) -> None:
    result = _result(**{field: "false"})
    with pytest.raises(TypeError, match=rf"plugin {field} must be bool"):
        validate_step_result(SCHEMA, result)


def test_error_code_must_be_text_or_none() -> None:
    with pytest.raises(TypeError, match="error_code must be str or None"):
        validate_step_result(SCHEMA, _result(error_code=404))
