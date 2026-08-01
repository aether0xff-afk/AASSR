from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Sequence

from .causal_dependency_world import CausalDependencyWorldV2
from .causal_representation import RelationalEffectEncoder, RepresentedReturnAgent
from .paper_v2_protocol import checkpoint_fingerprint, clone_agent_from_checkpoint
from .representation_diagnostic import run_episode


@dataclass(frozen=True, slots=True)
class TransferCurveRow:
    condition: str
    research_seed: int
    adaptation_budget: int
    success_rate: float
    branch_start_fingerprint: str
    evaluation_fingerprint_before: str
    evaluation_fingerprint_after: str

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }


def _factory(seed: int):
    return lambda: RepresentedReturnAgent(RelationalEffectEncoder(), seed=seed)


def _pretrain(seed: int, worlds: Sequence[int], episodes: int) -> RepresentedReturnAgent:
    agent = _factory(seed)()
    for episode in range(episodes):
        world_seed = int(worlds[episode % len(worlds)])
        run_episode(
            agent,
            world_seed=world_seed,
            token_seed=92001,
            observation_seed=world_seed,
            episode=episode,
            learn=True,
        )
    return agent


def _novel_episode(
    agent: RepresentedReturnAgent,
    *,
    world_seed: int,
    episode: int,
    learn: bool,
) -> bool:
    world = CausalDependencyWorldV2(
        world_seed=world_seed,
        token_seed=94001,
        observation_seed=world_seed,
        composition_template="novel_composition_v1",
    )
    epsilon = max(0.05, 0.8 * (1.0 - episode / 64.0)) if learn else 0.0
    from .representation_diagnostic import _transition

    for _ in range(8):
        before = world.observe()
        if before.terminal:
            break
        action = agent.select_action(before, epsilon=epsilon)
        outcome = world.step(action)
        if learn:
            agent.observe_transition(_transition(before, action, outcome))
        if world.terminal:
            break
    success = world.analysis_private_state.success
    if learn:
        agent.finish_episode(success)
    return success


def run_transfer_diagnostic(
    *,
    research_seeds: Sequence[int],
    train_world_seeds: Sequence[int],
    unseen_world_seeds: Sequence[int],
    pretraining_episodes: int = 500,
    evaluation_episodes: int = 100,
    budgets: Sequence[int] = (0, 1, 4, 16, 64),
) -> tuple[list[TransferCurveRow], dict[str, float]]:
    rows = []
    for research_seed in research_seeds:
        seed = int(research_seed) * 43
        pretrained = _pretrain(seed, train_world_seeds, pretraining_episodes)
        fresh = _factory(seed + 1)()
        for condition, base, factory_seed in (
            ("relational_transfer", pretrained, seed),
            ("from_scratch", fresh, seed + 1),
        ):
            base_fingerprint = checkpoint_fingerprint(base.export_full_checkpoint())
            for budget in budgets:
                branch, branch_start = clone_agent_from_checkpoint(
                    base, _factory(factory_seed)
                )
                if branch_start != base_fingerprint:
                    raise RuntimeError("transfer branch checkpoint mismatch")
                for episode in range(int(budget)):
                    _novel_episode(
                        branch,
                        world_seed=int(unseen_world_seeds[episode % len(unseen_world_seeds)]),
                        episode=episode,
                        learn=True,
                    )
                frozen, evaluation_before = clone_agent_from_checkpoint(
                    branch, _factory(factory_seed)
                )
                successes = [
                    _novel_episode(
                        frozen,
                        world_seed=int(unseen_world_seeds[index % len(unseen_world_seeds)]),
                        episode=int(budget) + index,
                        learn=False,
                    )
                    for index in range(evaluation_episodes)
                ]
                evaluation_after = checkpoint_fingerprint(
                    frozen.export_full_checkpoint()
                )
                rows.append(
                    TransferCurveRow(
                        condition,
                        int(research_seed),
                        int(budget),
                        fmean(successes),
                        branch_start,
                        evaluation_before,
                        evaluation_after,
                    )
                )
    ordered_budgets = sorted(set(int(value) for value in budgets))
    scale = max(ordered_budgets) or 1
    auc_by_condition: dict[str, float] = {}
    for condition in ("relational_transfer", "from_scratch"):
        points = []
        for budget in ordered_budgets:
            values = [
                row.success_rate
                for row in rows
                if row.condition == condition and row.adaptation_budget == budget
            ]
            points.append((budget, fmean(values)))
        area = sum(
            (right_x - left_x) * (left_y + right_y) / 2.0
            for (left_x, left_y), (right_x, right_y) in zip(points, points[1:])
        ) / scale
        auc_by_condition[condition] = area
    return rows, {
        **auc_by_condition,
        "transfer_minus_scratch_auc": auc_by_condition["relational_transfer"]
        - auc_by_condition["from_scratch"],
    }
