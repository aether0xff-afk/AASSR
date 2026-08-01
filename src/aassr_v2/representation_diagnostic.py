from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Callable, Sequence

from .causal_dependency_world import CausalDependencyWorldV2
from .causal_representation import (
    IdentityEncoder,
    ObservableTransition,
    RelationalEffectEncoder,
    RepresentedReturnAgent,
)
from .paper_v2_protocol import checkpoint_fingerprint, clone_agent_from_checkpoint


@dataclass(frozen=True, slots=True)
class RepresentationDiagnosticRow:
    diagnostic: str
    representation: str
    research_seed: int
    adaptation_budget: int
    success_rate: float
    start_checkpoint_fingerprint: str
    evaluation_checkpoint_fingerprint: str
    final_checkpoint_fingerprint: str
    effect_updates: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy() if hasattr(self, "__dict__") else {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }


def _transition(before, action: str, outcome) -> ObservableTransition:
    return ObservableTransition(
        before=before,
        action=action,
        after=outcome.observation,
        action_succeeded=outcome.action_succeeded,
        inventory_delta=outcome.inventory_delta,
        facts_added=len(outcome.facts_added),
        facts_removed=len(outcome.facts_removed),
        unlocked_actions=len(outcome.unlocked_actions),
        resource_cost=outcome.resource_cost,
        damage=outcome.damage,
        spatial_changed=outcome.spatial_change is not None,
        terminal_reward=outcome.reward,
    )


def run_episode(
    agent: RepresentedReturnAgent,
    *,
    world_seed: int,
    token_seed: int,
    observation_seed: int,
    episode: int,
    learn: bool,
    expose_affordances: bool = False,
) -> bool:
    world = CausalDependencyWorldV2(
        world_seed=world_seed,
        token_seed=token_seed,
        observation_seed=observation_seed,
        expose_affordances=expose_affordances,
    )
    epsilon = max(0.05, 0.8 * (1.0 - episode / 400.0)) if learn else 0.0
    for _ in range(8):
        before = world.observe()
        if before.terminal or not before.available_actions:
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


def _factory(name: str, seed: int) -> Callable[[], RepresentedReturnAgent]:
    encoder_factory = IdentityEncoder if name == "identity_representation" else RelationalEffectEncoder
    return lambda: RepresentedReturnAgent(encoder_factory(), seed=seed)


def _train(
    name: str,
    *,
    research_seed: int,
    train_world_seeds: Sequence[int],
    episodes: int,
) -> RepresentedReturnAgent:
    seed = research_seed * 37 + (0 if name == "identity_representation" else 1)
    agent = _factory(name, seed)()
    shared_action_schema_seed = 92001
    for episode in range(episodes):
        world_seed = int(train_world_seeds[episode % len(train_world_seeds)])
        run_episode(
            agent,
            world_seed=world_seed,
            token_seed=shared_action_schema_seed,
            observation_seed=world_seed,
            episode=episode,
            learn=True,
        )
    return agent


def run_diagnostic_two_a(
    *,
    research_seeds: Sequence[int],
    train_world_seeds: Sequence[int],
    isomorphic_world_seeds: Sequence[int],
    training_episodes: int = 500,
    evaluation_episodes: int = 100,
) -> list[RepresentationDiagnosticRow]:
    rows = []
    for research_seed in research_seeds:
        for name in ("identity_representation", "relational_effect_representation"):
            agent = _train(
                name,
                research_seed=int(research_seed),
                train_world_seeds=train_world_seeds,
                episodes=training_episodes,
            )
            seed = agent.seed
            clone, start = clone_agent_from_checkpoint(agent, _factory(name, seed))
            successes = [
                run_episode(
                    clone,
                    world_seed=int(isomorphic_world_seeds[index % len(isomorphic_world_seeds)]),
                    token_seed=92001,
                    observation_seed=int(isomorphic_world_seeds[index % len(isomorphic_world_seeds)]),
                    episode=training_episodes + index,
                    learn=False,
                )
                for index in range(evaluation_episodes)
            ]
            final = checkpoint_fingerprint(clone.export_full_checkpoint())
            memory = getattr(clone.encoder, "memory", None)
            rows.append(
                RepresentationDiagnosticRow(
                    "diagnostic_2a",
                    name,
                    int(research_seed),
                    0,
                    fmean(successes),
                    start,
                    start,
                    final,
                    int(getattr(memory, "update_count", 0)),
                )
            )
    return rows


def run_diagnostic_two_b(
    *,
    research_seeds: Sequence[int],
    train_world_seeds: Sequence[int],
    adaptation_world_seeds: Sequence[int],
    training_episodes: int = 500,
    evaluation_episodes: int = 100,
    budgets: Sequence[int] = (1, 4, 16),
) -> list[RepresentationDiagnosticRow]:
    rows = []
    for research_seed in research_seeds:
        for name in ("identity_representation", "relational_effect_representation"):
            agent = _train(
                name,
                research_seed=int(research_seed),
                train_world_seeds=train_world_seeds,
                episodes=training_episodes,
            )
            base_fingerprint = checkpoint_fingerprint(agent.export_full_checkpoint())
            for budget in budgets:
                clone, start = clone_agent_from_checkpoint(
                    agent, _factory(name, agent.seed)
                )
                if start != base_fingerprint:
                    raise RuntimeError("adaptation branch did not start from base checkpoint")
                for episode in range(int(budget)):
                    world_seed = int(
                        adaptation_world_seeds[episode % len(adaptation_world_seeds)]
                    )
                    run_episode(
                        clone,
                        world_seed=world_seed,
                        token_seed=93001,
                        observation_seed=world_seed,
                        episode=episode,
                        learn=True,
                    )
                frozen, evaluation_start = clone_agent_from_checkpoint(
                    clone, _factory(name, clone.seed)
                )
                successes = [
                    run_episode(
                        frozen,
                        world_seed=int(adaptation_world_seeds[index % len(adaptation_world_seeds)]),
                        token_seed=93001,
                        observation_seed=int(adaptation_world_seeds[index % len(adaptation_world_seeds)]),
                        episode=int(budget) + index,
                        learn=False,
                    )
                    for index in range(evaluation_episodes)
                ]
                final = checkpoint_fingerprint(frozen.export_full_checkpoint())
                memory = getattr(frozen.encoder, "memory", None)
                rows.append(
                    RepresentationDiagnosticRow(
                        "diagnostic_2b",
                        name,
                        int(research_seed),
                        int(budget),
                        fmean(successes),
                        start,
                        evaluation_start,
                        final,
                        int(getattr(memory, "update_count", 0)),
                    )
                )
    return rows
