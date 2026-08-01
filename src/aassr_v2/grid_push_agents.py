from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence

from .causal_agent_v2 import CausalAASSRAgent
from .causal_imagination import (
    CausalImaginationPlanner,
    ImaginationGateConfig,
    LearnedReturnModel,
)
from .causal_representation import (
    IdentityEncoder,
    ObservableTransition,
    RelationalEffectEncoder,
    RepresentedReturnAgent,
)
from .grid_push_world import GridPushSpec, GridPushStep, GridPushWorld, MOVE_DELTAS
from .paper_v2_protocol import checkpoint_fingerprint, clone_agent_from_checkpoint
from .paper_v2_types import FullAgentCheckpoint, RawCausalObservation


def _quantize(value: float, width: float = 0.25) -> float:
    return round(round(value / width) * width, 6)


class GridRelationalEffectEncoder(RelationalEffectEncoder):
    """Translation-relative grid state plus visible learned action effects."""

    name = "grid_relational_effect_representation"

    @staticmethod
    def _cells(observation: RawCausalObservation) -> dict[tuple[int, int], str]:
        cells: dict[tuple[int, int], str] = {}
        for key, value in observation.spatial_observations.items():
            if not key.startswith("cell:"):
                continue
            x, y = key.removeprefix("cell:").split(",")
            cells[(int(x), int(y))] = str(value)
        return cells

    def state_key(self, observation: RawCausalObservation) -> str:
        cells = self._cells(observation)
        player = next(
            (position for position, value in cells.items() if "player" in value),
            (0, 0),
        )
        relative = tuple(
            sorted(
                (
                    position[0] - player[0],
                    position[1] - player[1],
                    value.replace("+player", ""),
                )
                for position, value in cells.items()
            )
        )
        return repr(
            (
                relative,
                observation.last_action_succeeded,
                observation.terminal,
            )
        )

    def action_key(self, observation: RawCausalObservation, action: str) -> str:
        affordance = tuple(observation.action_affordances.get(action, ("move", action)))
        base = "motion:" + repr(affordance)
        profile = self.memory.profile(action)
        if profile is None:
            return base + "|unknown"
        return base + "|effect:" + repr(
            tuple(_quantize(value) for value in profile)
        )


def grid_observable_transition(
    before: RawCausalObservation,
    action: str,
    outcome: GridPushStep,
) -> ObservableTransition:
    return ObservableTransition(
        before=before,
        action=action,
        after=outcome.observation,
        action_succeeded=outcome.action_succeeded,
        inventory_delta={},
        facts_added=len(outcome.observation.observable_facts - before.observable_facts),
        facts_removed=len(before.observable_facts - outcome.observation.observable_facts),
        unlocked_actions=len(
            set(outcome.observation.available_actions) - set(before.available_actions)
        ),
        resource_cost=0.0,
        damage=0.0,
        spatial_changed=(
            outcome.observation.spatial_observations
            != before.spatial_observations
        ),
        terminal_reward=outcome.reward,
    )


@dataclass(frozen=True, slots=True)
class GridEpisodeRecord:
    condition: str
    research_seed: int
    phase: str
    episode: int
    world_seed: int
    success: bool
    steps: int
    failed_actions: int
    block_moves: int
    imagination_interventions: int
    actions: tuple[str, ...]
    event_steps: tuple[tuple[Mapping[str, Any], ...], ...]
    initial_agent_observation: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["actions"] = list(self.actions)
        payload["event_steps"] = [
            [dict(event) for event in events] for events in self.event_steps
        ]
        return payload


@dataclass(frozen=True, slots=True)
class GridConditionSummary:
    condition: str
    research_seed: int
    training_episodes: int
    training_final_tail_success: float
    frozen_success_rate: float
    mean_steps: float
    failed_action_rate: float
    mean_block_moves: float
    imagination_intervention_rate: float
    checkpoint_before_evaluation: str
    checkpoint_after_evaluation: str
    evaluation_learning_calls: int
    relational_key_migrations: int
    prophecy_return_brier: float | None = None
    prophecy_effect_error: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _episode(
    *,
    condition: str,
    research_seed: int,
    phase: str,
    episode: int,
    spec: GridPushSpec,
    maximum_steps: int,
    choose: Callable[[RawCausalObservation], tuple[str, bool]],
    learn: Callable[[ObservableTransition], None] | None,
    finish: Callable[[bool], None] | None,
) -> GridEpisodeRecord:
    world = GridPushWorld(spec)
    actions: list[str] = []
    event_steps: list[tuple[Mapping[str, Any], ...]] = []
    initial_agent_observation: Mapping[str, Any] = {}
    failures = 0
    block_moves = 0
    interventions = 0
    for _step in range(maximum_steps):
        before = world.observe()
        if not initial_agent_observation:
            initial_agent_observation = before.to_dict()
        if before.terminal:
            break
        action, intervened = choose(before)
        outcome = world.step(action)
        transition = grid_observable_transition(before, action, outcome)
        if learn is not None:
            learn(transition)
        actions.append(action)
        failures += int(not outcome.action_succeeded)
        block_moves += sum(
            event.kind == "block_moved" for event in world.analysis_last_events
        )
        interventions += int(intervened)
        event_steps.append(
            tuple(event.to_dict() for event in world.analysis_last_events)
        )
        if world.terminal:
            break
    success = world.analysis_private_state.success
    if finish is not None:
        finish(success)
    return GridEpisodeRecord(
        condition=condition,
        research_seed=int(research_seed),
        phase=phase,
        episode=int(episode),
        world_seed=int(spec.generator_seed or 0),
        success=success,
        steps=len(actions),
        failed_actions=failures,
        block_moves=block_moves,
        imagination_interventions=interventions,
        actions=tuple(actions),
        event_steps=tuple(event_steps),
        initial_agent_observation=dict(initial_agent_observation),
    )


def _summarize(
    *,
    condition: str,
    research_seed: int,
    training_successes: Sequence[bool],
    evaluation: Sequence[GridEpisodeRecord],
    checkpoint_before: str,
    checkpoint_after: str,
    evaluation_learning_calls: int,
    relational_key_migrations: int = 0,
    calibration: Mapping[str, float] | None = None,
) -> GridConditionSummary:
    tail = training_successes[-min(50, len(training_successes)) :]
    total_steps = sum(record.steps for record in evaluation)
    return GridConditionSummary(
        condition=condition,
        research_seed=int(research_seed),
        training_episodes=len(training_successes),
        training_final_tail_success=fmean(tail) if tail else 0.0,
        frozen_success_rate=fmean(record.success for record in evaluation),
        mean_steps=fmean(record.steps for record in evaluation),
        failed_action_rate=(
            sum(record.failed_actions for record in evaluation) / max(1, total_steps)
        ),
        mean_block_moves=fmean(record.block_moves for record in evaluation),
        imagination_intervention_rate=(
            sum(record.imagination_interventions for record in evaluation)
            / max(1, total_steps)
        ),
        checkpoint_before_evaluation=checkpoint_before,
        checkpoint_after_evaluation=checkpoint_after,
        evaluation_learning_calls=evaluation_learning_calls,
        relational_key_migrations=int(relational_key_migrations),
        prophecy_return_brier=None if calibration is None else calibration["return_brier_score"],
        prophecy_effect_error=None if calibration is None else calibration["observable_effect_error"],
    )


def run_small_grid_diagnostic(
    *,
    specs: Sequence[GridPushSpec],
    research_seeds: Sequence[int],
    training_episodes: int = 300,
    evaluation_episodes: int = 40,
    maximum_steps: int = 40,
    imagination_gate: ImaginationGateConfig | None = None,
) -> tuple[list[GridConditionSummary], list[GridEpisodeRecord], list[dict[str, Any]]]:
    if not specs:
        raise ValueError("at least one grid spec is required")
    gate = imagination_gate or ImaginationGateConfig(maximum_depth=3, branching_factor=4)
    summaries: list[GridConditionSummary] = []
    episodes: list[GridEpisodeRecord] = []
    decisions: list[dict[str, Any]] = []
    for seed in research_seeds:
        # Paired random evaluation on the same world schedule.
        random_agent = __import__("random").Random(int(seed))
        random_rows = []
        for episode in range(evaluation_episodes):
            spec = specs[episode % len(specs)]
            random_rows.append(
                _episode(
                    condition="random",
                    research_seed=int(seed),
                    phase="evaluation_train_world_frozen",
                    episode=episode,
                    spec=spec,
                    maximum_steps=maximum_steps,
                    choose=lambda observation, rng=random_agent: (
                        rng.choice(observation.available_actions), False
                    ),
                    learn=None,
                    finish=None,
                )
            )
        episodes.extend(random_rows)
        summaries.append(
            _summarize(
                condition="random",
                research_seed=int(seed),
                training_successes=(),
                evaluation=random_rows,
                checkpoint_before="",
                checkpoint_after="",
                evaluation_learning_calls=0,
                relational_key_migrations=0,
            )
        )

        contextual = RepresentedReturnAgent(IdentityEncoder(), seed=int(seed))
        contextual_training: list[bool] = []
        for episode in range(training_episodes):
            epsilon = max(0.05, 0.8 * (1.0 - episode / max(1, training_episodes)))
            record = _episode(
                condition="contextual_policy",
                research_seed=int(seed),
                phase="training",
                episode=episode,
                spec=specs[episode % len(specs)],
                maximum_steps=maximum_steps,
                choose=lambda observation, value=epsilon: (
                    contextual.select_action(observation, epsilon=value), False
                ),
                learn=contextual.observe_transition,
                finish=contextual.finish_episode,
            )
            contextual_training.append(record.success)
        contextual_clone, contextual_before = clone_agent_from_checkpoint(
            contextual,
            lambda: RepresentedReturnAgent(IdentityEncoder(), seed=int(seed)),
        )
        updates_before = contextual_clone.update_count
        contextual_rows = [
            _episode(
                condition="contextual_policy",
                research_seed=int(seed),
                phase="evaluation_train_world_frozen",
                episode=episode,
                spec=specs[episode % len(specs)],
                maximum_steps=maximum_steps,
                choose=lambda observation: (
                    contextual_clone.select_action(observation, epsilon=0.0), False
                ),
                learn=None,
                finish=None,
            )
            for episode in range(evaluation_episodes)
        ]
        contextual_after = checkpoint_fingerprint(
            contextual_clone.export_full_checkpoint()
        )
        episodes.extend(contextual_rows)
        summaries.append(
            _summarize(
                condition="contextual_policy",
                research_seed=int(seed),
                training_successes=contextual_training,
                evaluation=contextual_rows,
                checkpoint_before=contextual_before,
                checkpoint_after=contextual_after,
                evaluation_learning_calls=contextual_clone.update_count - updates_before,
                relational_key_migrations=contextual_clone.key_migration_count,
            )
        )

        full = CausalAASSRAgent(GridRelationalEffectEncoder, seed=int(seed))
        planner = CausalImaginationPlanner(
            LearnedReturnModel(full.prophecy), config=gate, gated=True
        )
        full_training: list[bool] = []
        for episode in range(training_episodes):
            epsilon = max(0.05, 0.8 * (1.0 - episode / max(1, training_episodes)))

            def training_choice(observation: RawCausalObservation) -> tuple[str, bool]:
                if full.policy.rng.random() < epsilon:
                    return full.policy.rng.choice(observation.available_actions), False
                decision = planner.decide(observation, full.policy)
                decisions.append({
                    "phase": "training", "research_seed": int(seed),
                    "episode": episode, **decision.to_dict(),
                })
                return decision.final_selected_action, decision.intervened

            record = _episode(
                condition="reduced_causal_agent",
                research_seed=int(seed),
                phase="training",
                episode=episode,
                spec=specs[episode % len(specs)],
                maximum_steps=maximum_steps,
                choose=training_choice,
                learn=full.observe_transition,
                finish=full.finish_episode,
            )
            full_training.append(record.success)
        full_clone, full_before = clone_agent_from_checkpoint(
            full,
            lambda: CausalAASSRAgent(GridRelationalEffectEncoder, seed=int(seed)),
        )
        frozen_planner = CausalImaginationPlanner(
            LearnedReturnModel(full_clone.prophecy), config=gate, gated=True
        )
        updates_before = full_clone.policy.update_count + full_clone.prophecy.total_updates

        def frozen_choice(observation: RawCausalObservation) -> tuple[str, bool]:
            decision = frozen_planner.decide(observation, full_clone.policy)
            decisions.append({
                "phase": "evaluation_train_world_frozen",
                "research_seed": int(seed), **decision.to_dict(),
            })
            return decision.final_selected_action, decision.intervened

        full_rows = [
            _episode(
                condition="reduced_causal_agent",
                research_seed=int(seed),
                phase="evaluation_train_world_frozen",
                episode=episode,
                spec=specs[episode % len(specs)],
                maximum_steps=maximum_steps,
                choose=frozen_choice,
                learn=None,
                finish=None,
            )
            for episode in range(evaluation_episodes)
        ]
        full_after_checkpoint = full_clone.export_full_checkpoint()
        full_after = checkpoint_fingerprint(full_after_checkpoint)
        updates_after = full_clone.policy.update_count + full_clone.prophecy.total_updates
        episodes.extend(full_rows)
        summaries.append(
            _summarize(
                condition="reduced_causal_agent",
                research_seed=int(seed),
                training_successes=full_training,
                evaluation=full_rows,
                checkpoint_before=full_before,
                checkpoint_after=full_after,
                evaluation_learning_calls=updates_after - updates_before,
                relational_key_migrations=full_clone.policy.key_migration_count,
                calibration=full_clone.prophecy.calibration_metrics(),
            )
        )
    return summaries, episodes, decisions


def checkpoint_contains_private_solver_data(checkpoint: FullAgentCheckpoint) -> bool:
    payload = repr(checkpoint.to_dict()).lower()
    return any(
        forbidden in payload
        for forbidden in (
            "plate_links", "solver_reference", "minimum_actions",
            "correct_path", "goal_distance", "goal_progress", "viability",
        )
    )
