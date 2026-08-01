from __future__ import annotations

import json
import subprocess
import sys

from aassr_v2.aassr_core import (
    FULL_CORE_EVIDENCE,
    CORE_MODULES,
    TRAINABLE_CORE_MODULES,
    AASSRCore,
    AASSRCoreConfig,
    build_full_aassr_core,
    build_no_imagination_core,
    build_tabular_fixed_goal_core,
)
from aassr_v2.environment_plugin import (
    CoreEnvironmentSession,
    ObservableEnvironmentTransition,
)
from aassr_v2.grid_push_plugin import GridPushEnvironmentPlugin
from aassr_v2.grid_push_world import GridPushSpec
from aassr_v2.imagination_tree import ImaginationConfig
from aassr_v2.types import Action


def _module_probe_spec() -> GridPushSpec:
    width, height = 5, 4
    boundary = frozenset(
        (x, y)
        for x in range(width)
        for y in range(height)
        if x in {0, width - 1} or y in {0, height - 1}
    )
    return GridPushSpec(
        width=width,
        height=height,
        walls=boundary,
        start=(1, 1),
        goal=(2, 1),
        blocks=frozenset({(2, 2)}),
        pits=frozenset({(3, 2)}),
        plates=frozenset({(1, 2)}),
        doors=frozenset({(3, 1)}),
        plate_links={(1, 2): ((3, 1),)},
        generator_seed=88001,
    )


def _plugin() -> GridPushEnvironmentPlugin:
    spec = _module_probe_spec()
    return GridPushEnvironmentPlugin(lambda _seed: spec)


def _core() -> AASSRCore:
    return AASSRCore(
        seed=19,
        config=AASSRCoreConfig(
            epsilon_start=0.0,
            epsilon_end=0.0,
            skill_promotion_successes=1,
            imagination_minimum_coverage=0.0,
            imagination=ImaginationConfig(
                branching_factor=2,
                maximum_depth=2,
                beam_width=8,
                outcome_samples=1,
                minimum_path_confidence=0.0,
                uncertainty_penalty=0.0,
                update_policy=False,
            ),
        ),
    )


def test_grid_push_plugin_exposes_only_primitive_observable_contract() -> None:
    plugin = _plugin()
    initial = plugin.reset(88001)
    assert tuple(schema.action_id for schema in plugin.action_schemas()) == (
        "MOVE_NORTH",
        "MOVE_SOUTH",
        "MOVE_WEST",
        "MOVE_EAST",
    )
    payload = json.dumps(initial.to_dict(), sort_keys=True).lower()
    for forbidden in (
        "plate_links",
        "solver",
        "optimal_action",
        "goal_distance",
        "goal_progress",
        "block_role",
        "solution_family",
    ):
        assert forbidden not in payload
    assert not hasattr(plugin, "solve")
    assert not hasattr(plugin, "optimal_action")


def test_importing_grid_runtime_plugin_does_not_load_solver_module() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import aassr_v2.grid_push_plugin; "
                "print('aassr_v2.grid_push_solver' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False"


def test_environment_transition_rejects_nonterminal_reward() -> None:
    plugin = _plugin()
    before = plugin.reset(88001)
    action = plugin.action_schemas()[0].build({})
    try:
        ObservableEnvironmentTransition(
            before,
            action,
            before,
            False,
            False,
            1.0,
        )
    except ValueError as exc:
        assert "non-terminal" in str(exc)
    else:
        raise AssertionError("non-terminal reward was accepted")


def test_complete_core_modules_are_exercised_before_frozen_evaluation() -> None:
    core = _core()
    records = [
        core.run_episode(
            _plugin(),
            world_seed=88001,
            episode=episode,
            maximum_steps=4,
            training=True,
            phase="development_module_call_probe",
        )
        for episode in range(6)
    ]
    assert all(record.success for record in records)
    assert all(value > 0 for value in core.audit.calls.values())
    assert all(
        core.audit.learning_updates[module] > 0
        for module in TRAINABLE_CORE_MODULES
    )
    assert core.audit.work_units["goal"] > 0
    assert core.audit.work_units["imagination_tree"] > 0
    assert core.audit.work_units["delayed_credit_assigner"] > 0
    assert len(core.replay.holdout()) > 0
    assert core.skills.all()
    assert not core.checkpoint_contains_forbidden_environment_data()
    serialized = records[0].to_dict()
    assert serialized["decisions"][0]["selected_action"].startswith("MOVE_EAST")
    assert serialized["transitions"][0]["raw_observation_before"]


def test_frozen_core_clone_calls_modules_without_learning_or_state_change() -> None:
    trained = _core()
    for episode in range(6):
        trained.run_episode(
            _plugin(),
            world_seed=88001,
            episode=episode,
            maximum_steps=4,
            training=True,
            phase="development_module_call_probe",
        )
    checkpoint = trained.export_checkpoint()
    clone = AASSRCore.from_checkpoint(checkpoint)
    before = clone.checkpoint_fingerprint()
    record = clone.run_episode(
        _plugin(),
        world_seed=88001,
        episode=0,
        maximum_steps=4,
        training=False,
        phase="evaluation_train_world_frozen",
    )
    after = clone.checkpoint_fingerprint()
    assert record.success
    assert before == after
    assert all(value == 0 for value in clone.audit.learning_updates.values())
    for module in CORE_MODULES:
        if module == "delayed_credit_assigner":
            continue
        assert clone.audit.calls[module] > 0


def test_core_snapshot_never_synthesizes_goal_progress() -> None:
    plugin = _plugin()
    plugin.reset(88001)
    core = _core()
    session = CoreEnvironmentSession(plugin, core.observation_encoder)
    assert session.snapshot().goal_progress == 0.0


def test_core_rejects_non_plugin_action() -> None:
    plugin = _plugin()
    plugin.reset(88001)
    try:
        plugin.execute(Action("MOVE_EAST"))
    except ValueError as exc:
        assert "different environment plugin" in str(exc)
    else:
        raise AssertionError("unregistered action was accepted")


def test_factories_name_only_the_gru_dynamic_goal_path_full_aassr() -> None:
    config = AASSRCoreConfig(
        epsilon_start=0.0,
        epsilon_end=0.0,
        imagination_minimum_coverage=0.0,
    )
    tabular = build_tabular_fixed_goal_core(config=config, seed=1)
    no_imagination = build_no_imagination_core(config=config, seed=1)
    full = build_full_aassr_core(config=config, seed=1)
    assert tabular.condition_name == "tabular_fixed_goal_core"
    assert no_imagination.condition_name == "full_aassr_no_imagination"
    assert full.condition_name == "full_aassr"
    assert "TabularProphecy" in tabular.component_class_manifest()
    assert "GoalGenerator" not in tabular.component_class_manifest()
    for required in (
        "GoalGenerator",
        "OnlineGRUProphecy",
        "SkillAwareProphecy",
        "ImaginationTree",
        "AdvancedTransitionEvaluator",
    ):
        assert required in full.component_class_manifest()
    assert no_imagination.use_imagination is False
    assert full.use_imagination is True


def test_full_core_fixture_proves_dynamic_goal_gru_and_value_flow() -> None:
    config = AASSRCoreConfig(
        epsilon_start=0.0,
        epsilon_end=0.0,
        skill_promotion_successes=1,
        imagination_minimum_coverage=0.0,
        imagination=ImaginationConfig(
            branching_factor=2,
            maximum_depth=2,
            beam_width=8,
            outcome_samples=1,
            minimum_path_confidence=0.0,
            uncertainty_penalty=0.0,
            update_policy=False,
        ),
    )
    core = build_full_aassr_core(config=config, seed=19)
    records = [
        core.run_episode(
            _plugin(),
            world_seed=88001,
            episode=episode,
            maximum_steps=4,
            training=True,
            phase="development_full_core_connection_fixture",
        )
        for episode in range(6)
    ]
    evidence = core.audit.evidence
    assert all(evidence[name] > 0 for name in FULL_CORE_EVIDENCE)
    assert core.goal_runtime is not None
    assert core.goal_runtime.records()
    first_goal = core.goal_runtime.records()[0]
    assert first_goal.evidence_observation
    assert first_goal.selected
    assert first_goal.achieved
    assert any(record.information_flows for record in records)
    flow = records[0].information_flows[0]
    assert flow.feature_memory_value == flow.policy_update_value
    assert all(
        decision.real_hidden_unchanged is True
        for record in records
        for decision in record.decisions
    )
    assert any(
        step.prediction_loss_after is not None
        for record in records
        for step in record.transitions
    )


def test_full_core_frozen_clone_preserves_complete_checkpoint() -> None:
    core = build_full_aassr_core(
        config=AASSRCoreConfig(
            epsilon_start=0.0,
            epsilon_end=0.0,
            imagination_minimum_coverage=0.0,
        ),
        seed=23,
    )
    for episode in range(6):
        core.run_episode(
            _plugin(),
            world_seed=88001,
            episode=episode,
            maximum_steps=4,
            training=True,
            phase="development_full_core_connection_fixture",
        )
    clone = AASSRCore.from_checkpoint(core.export_checkpoint())
    before = clone.checkpoint_fingerprint()
    clone.run_episode(
        _plugin(),
        world_seed=88001,
        episode=0,
        maximum_steps=4,
        training=False,
        phase="evaluation_train_world_frozen",
    )
    assert before == clone.checkpoint_fingerprint()
    assert all(value == 0 for value in clone.audit.learning_updates.values())
