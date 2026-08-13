from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import aassr_v2.current_checkpoint as current_checkpoint_module
from aassr_v2.current_checkpoint import (
    CURRENT_FROZEN_CHECKPOINT_VERSION,
    LEGACY_CURRENT_FROZEN_CHECKPOINT_VERSION,
    checkpoint_manifest,
    current_frozen_checkpoint_payload,
    resolve_clean_checkpoint_source_commit,
    restore_current_frozen_checkpoint,
    restore_trusted_legacy_current_frozen_checkpoint,
    save_current_frozen_checkpoint,
)
from aassr_v2.current_core_manifest import CURRENT_CORE_VERSION
from aassr_v2.current_critic_support import _action_key
from aassr_v2.current_entrypoint import build_current_pentest_aassr_core
from aassr_v2.current_generation import relational_action_key
from aassr_v2.current_manifest import (
    CURRENT_GENERATION_VERSION,
    CURRENT_RUNTIME_ASSEMBLY,
    CURRENT_SCIENTIFIC_CONTRACT_VERSION,
)
from aassr_v2.current_relational_state_v3 import (
    latest_status_code,
    relational_state_descriptor_v3,
)
from aassr_v2.pentest_transfer_stages import TRANSFER_STAGES, TransferDiagnosticWorld
from aassr_v2.replay import ReplayTransition
from aassr_v2.skills import Skill


def test_current_frozen_checkpoint_restores_learned_and_gate_state(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        current_checkpoint_module,
        "resolve_clean_checkpoint_source_commit",
        lambda declared_git_commit=None: str(declared_git_commit),
    )
    agent = build_current_pentest_aassr_core(
        seed=7,
        train_transitions=128,
        use_imagination=True,
        device="cpu",
        allow_tf32=False,
    )
    state = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[0]).snapshot()
    action = state.available_actions[0]

    with agent.dqn.torch.no_grad():
        next(agent.dqn.online.parameters()).add_(0.123)
        next(agent.base_neural_prophecy.models[0].parameters()).add_(0.234)
    agent.dqn.target.load_state_dict(agent.dqn.online.state_dict())
    agent.dqn.environment_steps = 17
    agent.dqn.gradient_updates = 3
    agent.base_neural_prophecy.observations = 19
    agent.base_neural_prophecy.gradient_updates = 5
    agent._critic_counts.update(
        {
            "episodes": 32,
            "successes": 4,
            "non_successes": 4,
            "positive_returns": 4,
            "negative_returns": 4,
            "zero_returns": 24,
            # Frozen-evaluation checkpoints intentionally persist compact recent
            # readiness counts instead of the full Critic training replay.
            "recent_return_window": 128,
            "recent_positive_returns": 4,
            "recent_negative_returns": 4,
            "recent_zero_returns": 0,
        }
    )
    agent.critic.episodes = 32
    agent.critic.transitions = 64
    agent.critic.gradient_updates = 2
    key = _action_key(state, action)
    agent._critic_support_rows[key].append(
        (
            tuple(relational_state_descriptor_v3(state)),
            latest_status_code(state),
        )
    )

    # Reproduce the exact object-rich state from a real run. Action.parameters
    # and StateSnapshot.metadata are MappingProxyType instances, so a raw pickle
    # of ReplayTransition would fail even though an empty-checkpoint test passes.
    for index in range(5):
        agent.evaluator.replay.add(
            ReplayTransition(state, action, state, f"portable-{index}")
        )
    assert len(agent.evaluator.replay.train()) == 4
    assert len(agent.evaluator.replay.holdout()) == 1

    skill = Skill(
        skill_id="skill-0001",
        primitive_actions=(action,),
        achieved_goal_ids=("external:success",),
        required_facts=frozenset(),
        added_facts=frozenset(),
        removed_facts=frozenset(),
        successes=2,
        failures=0,
    )
    agent.skills._skills[skill.skill_id] = skill
    agent.skills._templates[skill.skill_id] = (relational_action_key(state, action),)
    agent.skills._next_id = 2

    before_q = agent.dqn.score_actions(state, (action,))[0]
    before_prophecy = tuple(
        parameter.detach().clone()
        for parameter in agent.base_neural_prophecy.models[0].parameters()
    )

    path = tmp_path / "aassr_frozen.pt"
    with pytest.raises(ValueError, match="exact, non-'unknown'"):
        current_frozen_checkpoint_payload(
            agent,
            research_seed=7,
            transition_budget=128,
            git_commit="unknown",
        )
    save_current_frozen_checkpoint(
        agent,
        path,
        research_seed=7,
        transition_budget=128,
        git_commit="checkpoint-test-sha",
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    manifest = checkpoint_manifest(payload)
    assert manifest["checkpoint_version"] == CURRENT_FROZEN_CHECKPOINT_VERSION
    assert manifest["git_commit"] == "checkpoint-test-sha"
    assert manifest["core_version"] == CURRENT_CORE_VERSION
    assert manifest["plugin_id"] == CURRENT_RUNTIME_ASSEMBLY["plugin_id"]
    assert manifest["plugin_version"] == CURRENT_RUNTIME_ASSEMBLY["plugin_version"]
    assert manifest["architecture_version"] == CURRENT_GENERATION_VERSION
    assert (
        manifest["scientific_contract_version"]
        == CURRENT_SCIENTIFIC_CONTRACT_VERSION
    )
    restored = restore_current_frozen_checkpoint(
        path,
        device="cpu",
        allow_tf32=False,
        expected_git_commit="checkpoint-test-sha",
    )

    assert restored.dqn.score_actions(state, (action,))[0] == pytest.approx(
        before_q, abs=1e-7, rel=1e-7
    )
    for left, right in zip(
        before_prophecy,
        restored.base_neural_prophecy.models[0].parameters(),
        strict=True,
    ):
        assert left.equal(right.detach())
    assert restored.dqn.environment_steps == 17
    assert restored.dqn.gradient_updates == 3
    assert restored.base_neural_prophecy.observations == 19
    assert restored.base_neural_prophecy.gradient_updates == 5
    assert restored._critic_counts["non_successes"] == 4
    assert restored._critic_counts["recent_negative_returns"] == 4
    assert restored.critic.support_confidence(state, action) > 0.0
    assert restored.critic_ready is True
    assert restored.critic_reliably_ready() is True
    # Frozen prediction/calibration uses holdout only. The training partition is
    # recorded by count in the manifest, not duplicated into the checkpoint.
    assert len(restored.evaluator.replay.train()) == 0
    assert len(restored.evaluator.replay.holdout()) == 1
    assert restored.skills.get("skill-0001").primitive_actions[0].signature == action.signature
    assert restored.skills.template_length("skill-0001") == 1
    assert restored.config.imagination_intervention_margin == pytest.approx(0.05)

    # Portable v2 lacks core/plugin/scientific provenance. Canonical restore must
    # reject it, while the narrowly named trusted-local compatibility path keeps
    # historical frozen artifacts reproducible with exact commit/architecture.
    legacy_payload = dict(payload)
    legacy_payload["checkpoint_version"] = LEGACY_CURRENT_FROZEN_CHECKPOINT_VERSION
    for field in (
        "core_version",
        "plugin_id",
        "plugin_version",
        "scientific_contract_version",
    ):
        legacy_payload.pop(field)
    legacy_path = tmp_path / "aassr_frozen_legacy_v2.pt"
    torch.save(legacy_payload, legacy_path)
    with pytest.raises(ValueError, match="trusted legacy v2"):
        restore_current_frozen_checkpoint(
            legacy_path,
            expected_git_commit="checkpoint-test-sha",
            device="cpu",
            allow_tf32=False,
        )
    legacy_restored = restore_trusted_legacy_current_frozen_checkpoint(
        legacy_path,
        expected_git_commit="checkpoint-test-sha",
        device="cpu",
        allow_tf32=False,
        allow_noncanonical_source=True,
    )
    assert legacy_restored.dqn.score_actions(state, (action,))[0] == pytest.approx(
        before_q, abs=1e-7, rel=1e-7
    )


def _canonical_provenance_header() -> dict[str, object]:
    return {
        "checkpoint_version": CURRENT_FROZEN_CHECKPOINT_VERSION,
        "checkpoint_scope": "frozen-evaluation-only",
        "git_commit": "checkpoint-test-sha",
        "core_version": CURRENT_CORE_VERSION,
        "plugin_id": CURRENT_RUNTIME_ASSEMBLY["plugin_id"],
        "plugin_version": CURRENT_RUNTIME_ASSEMBLY["plugin_version"],
        "architecture_version": CURRENT_GENERATION_VERSION,
        "scientific_contract_version": CURRENT_SCIENTIFIC_CONTRACT_VERSION,
    }


@pytest.mark.parametrize(
    ("field", "mismatch"),
    (
        ("git_commit", "different-sha"),
        ("core_version", "different-core"),
        ("plugin_id", "different-plugin"),
        ("plugin_version", "different-plugin-version"),
        ("architecture_version", "different-architecture"),
        ("scientific_contract_version", "different-scientific-contract"),
    ),
)
def test_canonical_checkpoint_restore_fails_closed_on_each_provenance_mismatch(
    tmp_path,
    monkeypatch,
    field: str,
    mismatch: str,
) -> None:
    monkeypatch.setattr(
        current_checkpoint_module,
        "resolve_clean_checkpoint_source_commit",
        lambda declared_git_commit=None: str(declared_git_commit),
    )
    payload = _canonical_provenance_header()
    payload[field] = mismatch
    path = tmp_path / f"mismatch-{field}.pt"
    torch.save(payload, path)

    with pytest.raises(ValueError, match=field):
        restore_current_frozen_checkpoint(
            path,
            expected_git_commit="checkpoint-test-sha",
        )


def test_canonical_checkpoint_restore_requires_an_exact_expected_commit(
    tmp_path,
) -> None:
    path = tmp_path / "not-read.pt"
    with pytest.raises(TypeError, match="expected_git_commit"):
        restore_current_frozen_checkpoint(path)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="exact, non-'unknown'"):
        restore_current_frozen_checkpoint(path, expected_git_commit="unknown")


@pytest.mark.parametrize(
    ("agent_field", "mismatch"),
    (
        ("current_generation_version", "runtime-architecture-drift"),
        ("current_core_version", "runtime-core-drift"),
        ("current_plugin_id", "runtime-plugin-drift"),
        ("current_plugin_version", "runtime-plugin-version-drift"),
    ),
)
def test_built_runtime_drift_fails_before_checkpoint_state_injection(
    tmp_path,
    monkeypatch,
    agent_field: str,
    mismatch: str,
) -> None:
    monkeypatch.setattr(
        current_checkpoint_module,
        "resolve_clean_checkpoint_source_commit",
        lambda declared_git_commit=None: str(declared_git_commit),
    )
    payload = {
        **_canonical_provenance_header(),
        "research_seed": 7,
        "transition_budget": 64,
        "requested_imagination": True,
    }
    path = tmp_path / f"runtime-drift-{agent_field}.pt"
    torch.save(payload, path)
    runtime = SimpleNamespace(
        current_generation_version=CURRENT_GENERATION_VERSION,
        current_core_version=CURRENT_CORE_VERSION,
        current_plugin_id=CURRENT_RUNTIME_ASSEMBLY["plugin_id"],
        current_plugin_version=CURRENT_RUNTIME_ASSEMBLY["plugin_version"],
    )
    setattr(runtime, agent_field, mismatch)
    monkeypatch.setattr(
        current_checkpoint_module,
        "build_current_pentest_aassr_core",
        lambda **_: runtime,
    )

    with pytest.raises(ValueError, match="before state restore"):
        restore_current_frozen_checkpoint(
            path,
            expected_git_commit="checkpoint-test-sha",
        )
    assert not hasattr(runtime, "dqn")


def test_checkpoint_source_commit_rejects_dirty_or_mismatched_worktrees(
    monkeypatch,
) -> None:
    head = "c8ae46a0a74eeb697f343516bbfa85c9df4cfd60"
    dirty = True

    def fake_run(command, **_):
        if command[1:3] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=f"{head}\n")
        assert command[1:3] == ["status", "--porcelain"]
        return SimpleNamespace(stdout=" M src/aassr_v2/current_checkpoint.py\n" if dirty else "")

    monkeypatch.setattr(current_checkpoint_module.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        resolve_clean_checkpoint_source_commit(head)

    dirty = False
    with pytest.raises(ValueError, match="does not match Git HEAD"):
        resolve_clean_checkpoint_source_commit("different-commit")
    assert resolve_clean_checkpoint_source_commit(head) == head


def test_canonical_restore_checks_expected_commit_against_clean_runtime_head(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "not-deserialized.pt"
    calls: list[str | None] = []

    def reject_runtime_head(declared_git_commit=None):
        calls.append(declared_git_commit)
        raise ValueError("declared git_commit does not match Git HEAD")

    monkeypatch.setattr(
        current_checkpoint_module,
        "resolve_clean_checkpoint_source_commit",
        reject_runtime_head,
    )
    monkeypatch.setattr(
        current_checkpoint_module,
        "_load_trusted_local_checkpoint",
        lambda *_args, **_kwargs: pytest.fail(
            "checkpoint deserialization must follow runtime commit validation"
        ),
    )

    with pytest.raises(ValueError, match="does not match Git HEAD"):
        restore_current_frozen_checkpoint(
            path,
            expected_git_commit="checkpoint-test-sha",
        )
    assert calls == ["checkpoint-test-sha"]


def test_legacy_noncanonical_override_waives_only_current_source_check(
    tmp_path,
    monkeypatch,
) -> None:
    payload = {
        **_canonical_provenance_header(),
        "checkpoint_version": LEGACY_CURRENT_FROZEN_CHECKPOINT_VERSION,
    }
    for field in (
        "core_version",
        "plugin_id",
        "plugin_version",
        "scientific_contract_version",
    ):
        payload.pop(field)
    path = tmp_path / "legacy-header-only.pt"
    torch.save(payload, path)
    monkeypatch.setattr(
        current_checkpoint_module,
        "resolve_clean_checkpoint_source_commit",
        lambda *_: pytest.fail("explicit legacy override must skip current HEAD check"),
    )
    monkeypatch.setattr(
        current_checkpoint_module,
        "_restore_validated_current_frozen_payload",
        lambda loaded, **_: loaded,
    )

    restored = restore_trusted_legacy_current_frozen_checkpoint(
        path,
        expected_git_commit="checkpoint-test-sha",
        allow_noncanonical_source=True,
    )
    assert restored["git_commit"] == "checkpoint-test-sha"
    with pytest.raises(ValueError, match="git commit"):
        restore_trusted_legacy_current_frozen_checkpoint(
            path,
            expected_git_commit="different-sha",
            allow_noncanonical_source=True,
        )
