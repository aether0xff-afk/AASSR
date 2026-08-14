from __future__ import annotations

from dataclasses import dataclass

import pytest

from aassr_v2.core import (
    ActionParameter,
    ActionSpec,
    ObservationField,
    PluginObservation,
    PluginSchema,
    PluginStepResult,
    ValueKind,
    build_aassr_core,
)


SCHEMA = PluginSchema(
    plugin_id="two-choice-real-contract",
    version="v1",
    observations=(
        ObservationField("location", ValueKind.ENTITY),
        ObservationField(
            "available",
            ValueKind.SET,
            item_kind=ValueKind.ENTITY,
        ),
    ),
    actions=(
        ActionSpec(
            "choose",
            parameters=(
                ActionParameter("target", ValueKind.ENTITY),
            ),
        ),
    ),
)


@dataclass
class TwoChoicePlugin:
    schema = SCHEMA
    done: bool = False

    def reset(self, *, seed=None):
        del seed
        self.done = False
        return PluginStepResult(
            PluginObservation(
                {
                    "location": "start",
                    "available": ("left", "right"),
                }
            )
        )

    def step(self, command):
        assert not self.done
        self.done = True
        target = str(command.arguments["target"])
        reward = 1.0 if target == "right" else -1.0
        return PluginStepResult(
            PluginObservation(
                {
                    "location": target,
                    "available": (),
                }
            ),
            reward=reward,
            terminated=True,
        )


def test_core_runtime_uses_only_minimal_contract() -> None:
    pytest.importorskip("torch")
    core = build_aassr_core(
        TwoChoicePlugin(),
        seed=7,
        device="cpu",
        use_imagination=True,
        train_transitions=16,
    )
    value = core.run_episode(
        episode=0,
        max_steps=2,
        training=True,
    )
    assert value in {-1.0, 1.0}
    diagnostics = core.diagnostics()
    assert diagnostics["plugin"]["id"] == "two-choice-real-contract"
    assert diagnostics["representation"]["owner"] == "core"
    assert diagnostics["imagination"]["valid_treatment"] is False
