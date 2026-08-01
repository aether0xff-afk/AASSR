from __future__ import annotations

from aassr_v2.causal_dependency_world import CausalDependencyWorldV2
from aassr_v2.transfer_diagnostic_v2 import run_transfer_diagnostic


def test_unseen_composition_changes_template_not_causal_law() -> None:
    train = CausalDependencyWorldV2(world_seed=82001, composition_template="base")
    unseen = CausalDependencyWorldV2(
        world_seed=84001, composition_template="novel_composition_v1"
    )
    assert train.causal_law_sha256 == unseen.causal_law_sha256
    assert train.composition_template_sha256 != unseen.composition_template_sha256


def test_transfer_budgets_branch_from_same_checkpoint_and_freeze_evaluation() -> None:
    rows, metrics = run_transfer_diagnostic(
        research_seeds=[2003],
        train_world_seeds=[82001],
        unseen_world_seeds=[84001],
        pretraining_episodes=40,
        evaluation_episodes=3,
        budgets=[0, 1, 4],
    )
    for condition in {row.condition for row in rows}:
        assert len(
            {
                row.branch_start_fingerprint
                for row in rows
                if row.condition == condition
            }
        ) == 1
    assert all(
        row.evaluation_fingerprint_before == row.evaluation_fingerprint_after
        for row in rows
    )
    assert "transfer_minus_scratch_auc" in metrics
