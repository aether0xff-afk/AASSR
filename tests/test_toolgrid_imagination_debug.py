from __future__ import annotations

from aassr_v2 import toolgrid_factorial_masked as env
from aassr_v2.toolgrid_imagination_debug import (
    OracleAssessment,
    _effect_label,
    assess_action,
)


def test_oracle_assessment_distinguishes_correct_and_wrong_tools() -> None:
    world = env.ToolGridWorld(12345, grid_size=3, action_count=8)
    oracle = world.oracle_actions()
    for action in oracle[:-1]:
        outcome = world.step(action)
        assert not outcome.error
        assert not world.failed

    correct = oracle[-1]
    wrong = next(
        action
        for action in world.snapshot().available_actions
        if action.signature != correct.signature
    )
    assert assess_action(world, correct) == OracleAssessment(1, 0)
    assert assess_action(world, wrong) == OracleAssessment(0, -1)


def test_effect_labels_only_changed_actions() -> None:
    viable = OracleAssessment(1, 3)
    shorter = OracleAssessment(1, 2)
    longer = OracleAssessment(1, 4)
    failed = OracleAssessment(0, -1)

    assert _effect_label(viable, failed, True) == "harmful"
    assert _effect_label(failed, viable, True) == "beneficial"
    assert _effect_label(viable, shorter, True) == "shorter_viable"
    assert _effect_label(viable, longer, True) == "longer_viable"
    assert _effect_label(viable, failed, False) == "no_change"
