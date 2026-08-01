from __future__ import annotations

import hashlib
import json
import pickle
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence

from .aassr_core import (
    FULL_CORE_EVIDENCE,
    AASSRCore,
    AASSRCoreConfig,
    build_full_aassr_core,
    build_no_imagination_core,
    build_tabular_fixed_goal_core,
)
from .autonomous_agent import ContextualPolicy
from .causal_agent_v2 import CausalAASSRAgent
from .causal_imagination import (
    CausalImaginationPlanner,
    ImaginationGateConfig,
    LearnedReturnModel,
)
from .causal_representation import ObservableTransition
from .environment_plugin import CoreEnvironmentSession, CoreObservationEncoder
from .grid_push_agents import GridRelationalEffectEncoder
from .grid_push_creativity import freeze_solver_reference
from .grid_push_plugin import GridPushEnvironmentPlugin
from .grid_push_solver import ProceduralGridPushGenerator
from .grid_push_world import GRID_PUSH_LAW_SHA256, GridPushSpec, GridPushWorld
from .imagination_tree import ImaginationConfig
from .paper_v2_protocol import (
    V2ArtifactWriter,
    build_run_identity,
    implementation_tree_sha256,
    replay_gzip_trace,
    reserve_run,
    v2_run_directory,
)
from .types import Action


CONDITIONS = (
    "random",
    "contextual_policy",
    "reduced_causal_agent",
    "tabular_fixed_goal_core",
    "full_aassr_no_imagination",
    "full_aassr",
)


@dataclass(frozen=True, slots=True)
class GridCoreComparisonArtifacts:
    output_dir: Path
    manifest_path: Path
    report_path: Path
    episode_rows: int


def _plugin_for(specs: Mapping[int, GridPushSpec]) -> GridPushEnvironmentPlugin:
    return GridPushEnvironmentPlugin(lambda seed: specs[int(seed)])


def _fixture_spec() -> GridPushSpec:
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


def _core_config(settings: Mapping[str, Any], *, fixture: bool = False) -> AASSRCoreConfig:
    core = settings["core"]
    return AASSRCoreConfig(
        gamma=float(core.get("gamma", 0.97)),
        epsilon_start=0.0 if fixture else float(core.get("epsilon_start", 0.8)),
        epsilon_end=0.0 if fixture else float(core.get("epsilon_end", 0.05)),
        epsilon_decay_episodes=int(core.get("epsilon_decay_episodes", 40)),
        prophecy_samples=int(core.get("prophecy_samples", 1)),
        replay_capacity=int(core.get("replay_capacity", 2048)),
        holdout_stride=int(core.get("holdout_stride", 5)),
        skill_promotion_successes=(
            1 if fixture else int(core.get("skill_promotion_successes", 2))
        ),
        skill_maximum_length=int(core.get("skill_maximum_length", 12)),
        feature_dimension=int(core.get("feature_dimension", 64)),
        internal_goal_weight=float(core.get("internal_goal_weight", 2.0)),
        maximum_internal_goals=int(core.get("maximum_internal_goals", 256)),
        gru_action_feature_size=int(core.get("gru_action_feature_size", 16)),
        gru_hidden_size=int(core.get("gru_hidden_size", 24)),
        gru_learning_rate=float(core.get("gru_learning_rate", 0.02)),
        imagination_minimum_coverage=(
            0.0
            if fixture
            else float(core.get("imagination_minimum_coverage", 0.25))
        ),
        imagination=ImaginationConfig(
            branching_factor=int(core.get("imagination_branching_factor", 2)),
            maximum_depth=int(core.get("imagination_depth", 2)),
            beam_width=int(core.get("imagination_beam_width", 8)),
            outcome_samples=int(core.get("prophecy_samples", 1)),
            minimum_path_confidence=(
                0.0 if fixture else float(core.get("minimum_path_confidence", 0.1))
            ),
            uncertainty_penalty=float(core.get("uncertainty_penalty", 0.2)),
            aggregation=str(core.get("imagination_aggregation", "max")),
            update_policy=False,
        ),
    )


def _observable_transition(transition: object) -> ObservableTransition:
    before = transition.before
    after = transition.after
    return ObservableTransition(
        before=before,
        action=transition.action.verb_name,
        after=after,
        action_succeeded=bool(transition.action_succeeded),
        inventory_delta={
            key: int(after.inventory.get(key, 0)) - int(before.inventory.get(key, 0))
            for key in set(before.inventory) | set(after.inventory)
        },
        facts_added=len(after.observable_facts - before.observable_facts),
        facts_removed=len(before.observable_facts - after.observable_facts),
        unlocked_actions=len(set(after.available_actions) - set(before.available_actions)),
        resource_cost=float(after.resource_cost - before.resource_cost),
        damage=float(after.damage - before.damage),
        spatial_changed=after.spatial_observations != before.spatial_observations,
        terminal_reward=float(transition.final_sparse_reward),
    )


def _baseline_episode(
    *,
    condition: str,
    plugin: GridPushEnvironmentPlugin,
    world_seed: int,
    research_seed: int,
    episode: int,
    maximum_steps: int,
    training: bool,
    choose: Callable[[CoreEnvironmentSession, float], Action],
    observe: Callable[[object, object, object], None] | None = None,
    finish: Callable[[bool], None] | None = None,
    epsilon: float = 0.0,
) -> dict[str, Any]:
    plugin.reset(world_seed)
    session = CoreEnvironmentSession(plugin, CoreObservationEncoder())
    transitions: list[dict[str, Any]] = []
    while not plugin.terminal and len(transitions) < maximum_steps:
        before_raw = plugin.raw_observation()
        before = session.snapshot()
        action = choose(session, epsilon)
        visible = plugin.execute(action)
        after = session.snapshot()
        if training and observe is not None:
            observe(before, action, visible)
        transitions.append(
            {
                "action": action.signature,
                "action_succeeded": visible.action_succeeded,
                "terminal": visible.terminal,
                "reward": visible.final_sparse_reward,
                "before": before_raw.to_dict(),
                "after": visible.after.to_dict(),
            }
        )
    success = plugin.final_sparse_reward() == 1.0
    if training and finish is not None:
        finish(success)
    return {
        "condition": condition,
        "research_seed": int(research_seed),
        "phase": "training" if training else "evaluation_train_world_frozen",
        "episode": int(episode),
        "world_seed": int(world_seed),
        "training": training,
        "success": success,
        "final_sparse_reward": plugin.final_sparse_reward(),
        "primitive_steps": len(transitions),
        "decisions": [],
        "transitions": transitions,
        "goal_lifecycle": [],
        "information_flows": [],
    }


def _run_random(
    specs: Mapping[int, GridPushSpec],
    worlds: Sequence[int],
    *,
    research_seed: int,
    episodes: int,
    maximum_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(research_seed)
    rows = []
    for episode in range(episodes):
        rows.append(
            _baseline_episode(
                condition="random",
                plugin=_plugin_for(specs),
                world_seed=worlds[episode % len(worlds)],
                research_seed=research_seed,
                episode=episode,
                maximum_steps=maximum_steps,
                training=False,
                choose=lambda session, _epsilon: rng.choice(
                    session.snapshot().available_actions
                ),
            )
        )
    return rows, {
        "checkpoint_before_evaluation": "not_applicable",
        "checkpoint_after_evaluation": "not_applicable",
        "checkpoint_immutable": True,
        "evaluation_learning_calls": 0,
    }


def _run_contextual(
    specs: Mapping[int, GridPushSpec],
    worlds: Sequence[int],
    *,
    research_seed: int,
    training_episodes: int,
    evaluation_episodes: int,
    maximum_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy = ContextualPolicy()
    rng = random.Random(research_seed)
    episode_pairs: list[tuple[object, Action]] = []

    def observe(before: object, action: Action, _visible: object) -> None:
        episode_pairs.append((before, action))

    def finish(success: bool) -> None:
        target = float(success)
        for state, action in reversed(episode_pairs):
            policy.observe_return(state, action, target)
            target *= 0.97
        episode_pairs.clear()

    rows = []
    for episode in range(training_episodes):
        epsilon = max(0.05, 0.8 * (1.0 - episode / max(1, training_episodes)))
        rows.append(
            _baseline_episode(
                condition="contextual_policy",
                plugin=_plugin_for(specs),
                world_seed=worlds[episode % len(worlds)],
                research_seed=research_seed,
                episode=episode,
                maximum_steps=maximum_steps,
                training=True,
                choose=lambda session, value: policy.select(
                    session.snapshot(),
                    randomizer=rng,
                    epsilon=value,
                    exploration_bonus=0.3,
                ),
                observe=observe,
                finish=finish,
                epsilon=epsilon,
            )
        )
    clone = pickle.loads(pickle.dumps(policy, protocol=5))
    before = hashlib.sha256(pickle.dumps(clone, protocol=5)).hexdigest()
    for episode in range(evaluation_episodes):
        rows.append(
            _baseline_episode(
                condition="contextual_policy",
                plugin=_plugin_for(specs),
                world_seed=worlds[episode % len(worlds)],
                research_seed=research_seed,
                episode=episode,
                maximum_steps=maximum_steps,
                training=False,
                choose=lambda session, _epsilon: clone.select(
                    session.snapshot(),
                    randomizer=rng,
                    epsilon=0.0,
                    exploration_bonus=0.0,
                ),
            )
        )
    after = hashlib.sha256(pickle.dumps(clone, protocol=5)).hexdigest()
    return rows, {
        "checkpoint_before_evaluation": before,
        "checkpoint_after_evaluation": after,
        "checkpoint_immutable": before == after,
        "evaluation_learning_calls": 0,
    }


def _run_reduced(
    specs: Mapping[int, GridPushSpec],
    worlds: Sequence[int],
    *,
    research_seed: int,
    training_episodes: int,
    evaluation_episodes: int,
    maximum_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    agent = CausalAASSRAgent(GridRelationalEffectEncoder, seed=research_seed)
    planner = CausalImaginationPlanner(
        LearnedReturnModel(agent.prophecy),
        config=ImaginationGateConfig(maximum_depth=2, branching_factor=2),
        gated=True,
    )

    def run_one(episode: int, training: bool, active: CausalAASSRAgent) -> dict[str, Any]:
        plugin = _plugin_for(specs)

        def choose(session: CoreEnvironmentSession, epsilon: float) -> Action:
            observation = plugin.raw_observation()
            if training and active.policy.rng.random() < epsilon:
                action_id = active.policy.rng.choice(observation.available_actions)
            else:
                action_id = planner.decide(observation, active.policy).final_selected_action
            schemas = {schema.action_id: schema for schema in plugin.action_schemas()}
            return schemas[action_id].build({})

        def observe(_before: object, _action: Action, visible: object) -> None:
            active.observe_transition(_observable_transition(visible))

        return _baseline_episode(
            condition="reduced_causal_agent",
            plugin=plugin,
            world_seed=worlds[episode % len(worlds)],
            research_seed=research_seed,
            episode=episode,
            maximum_steps=maximum_steps,
            training=training,
            choose=choose,
            observe=observe,
            finish=active.finish_episode,
            epsilon=(
                max(0.05, 0.8 * (1.0 - episode / max(1, training_episodes)))
                if training
                else 0.0
            ),
        )

    rows = [run_one(episode, True, agent) for episode in range(training_episodes)]
    checkpoint = agent.export_full_checkpoint()
    clone = CausalAASSRAgent(GridRelationalEffectEncoder, seed=research_seed)
    clone.import_full_checkpoint(checkpoint)
    frozen_planner = CausalImaginationPlanner(
        LearnedReturnModel(clone.prophecy),
        config=ImaginationGateConfig(maximum_depth=2, branching_factor=2),
        gated=True,
    )
    planner = frozen_planner
    before = hashlib.sha256(repr(clone.export_full_checkpoint()).encode()).hexdigest()
    rows.extend(run_one(episode, False, clone) for episode in range(evaluation_episodes))
    after = hashlib.sha256(repr(clone.export_full_checkpoint()).encode()).hexdigest()
    return rows, {
        "checkpoint_before_evaluation": before,
        "checkpoint_after_evaluation": after,
        "checkpoint_immutable": before == after,
        "evaluation_learning_calls": 0,
    }


def _run_core_condition(
    factory: Callable[..., AASSRCore],
    specs: Mapping[int, GridPushSpec],
    worlds: Sequence[int],
    *,
    config: AASSRCoreConfig,
    research_seed: int,
    training_episodes: int,
    evaluation_episodes: int,
    maximum_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], AASSRCore]:
    core = factory(config=config, seed=research_seed)
    rows = []
    for episode in range(training_episodes):
        record = core.run_episode(
            _plugin_for(specs),
            world_seed=worlds[episode % len(worlds)],
            episode=episode,
            maximum_steps=maximum_steps,
            training=True,
            phase="training",
        )
        rows.append(
            {
                "condition": core.condition_name,
                "research_seed": research_seed,
                **record.to_dict(),
            }
        )
    clone = AASSRCore.from_checkpoint(core.export_checkpoint())
    before = clone.checkpoint_fingerprint()
    for episode in range(evaluation_episodes):
        record = clone.run_episode(
            _plugin_for(specs),
            world_seed=worlds[episode % len(worlds)],
            episode=episode,
            maximum_steps=maximum_steps,
            training=False,
            phase="evaluation_train_world_frozen",
        )
        rows.append(
            {
                "condition": core.condition_name,
                "research_seed": research_seed,
                **record.to_dict(),
            }
        )
    after = clone.checkpoint_fingerprint()
    return rows, {
        "checkpoint_before_evaluation": before,
        "checkpoint_after_evaluation": after,
        "checkpoint_immutable": before == after,
        "evaluation_learning_calls": sum(clone.audit.learning_updates.values()),
        "training_audit": core.audit.to_dict(),
        "frozen_audit": clone.audit.to_dict(),
        "checkpoint_private_solver_leak": core.checkpoint_contains_forbidden_environment_data(),
    }, core


def _summary(condition: str, rows: Sequence[Mapping[str, Any]], integrity: Mapping[str, Any]) -> dict[str, Any]:
    training = [row for row in rows if row["phase"] == "training"]
    frozen = [row for row in rows if row["phase"] == "evaluation_train_world_frozen"]
    tail = training[-min(10, len(training)) :]
    return {
        "condition": condition,
        "training_episodes": len(training),
        "evaluation_episodes": len(frozen),
        "training_final_tail_success": (
            fmean(bool(row["success"]) for row in tail) if tail else None
        ),
        "frozen_success_rate": fmean(bool(row["success"]) for row in frozen),
        "frozen_mean_steps": fmean(int(row["primitive_steps"]) for row in frozen),
        **integrity,
    }


def _module_fixture(config: AASSRCoreConfig, episodes: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = _fixture_spec()
    core = build_full_aassr_core(config=config, seed=88101)
    records = [
        core.run_episode(
            GridPushEnvironmentPlugin(lambda _seed: spec),
            world_seed=88001,
            episode=episode,
            maximum_steps=4,
            training=True,
            phase="development_full_core_connection_fixture",
        )
        for episode in range(episodes)
    ]
    checkpoint = core.export_checkpoint()
    clone = AASSRCore.from_checkpoint(checkpoint)
    before = clone.checkpoint_fingerprint()
    frozen = clone.run_episode(
        GridPushEnvironmentPlugin(lambda _seed: spec),
        world_seed=88001,
        episode=0,
        maximum_steps=4,
        training=False,
        phase="evaluation_train_world_frozen",
    )
    after = clone.checkpoint_fingerprint()
    evidence = core.audit.evidence
    goal_records = () if core.goal_runtime is None else core.goal_runtime.records()
    sample_transition = next(
        step.to_dict()
        for record in records
        for step in record.transitions
        if step.prediction_loss_after is not None
    )
    sample_flow = max(
        (
            flow.to_dict()
            for record in records
            for flow in record.information_flows
        ),
        key=lambda item: abs(float(item["predicted_information_value"])),
    )
    return {
        "fixture_role": "engineering_connection_evidence_not_performance_evidence",
        "component_classes": list(core.component_class_manifest()),
        "audit": core.audit.to_dict(),
        "all_required_evidence_positive": all(
            evidence[name] > 0 for name in FULL_CORE_EVIDENCE
        ),
        "goal_generator_trace": [record.to_dict() for record in goal_records[:4]],
        "gru_training_trace": sample_transition,
        "information_flow_trace": sample_flow,
        "checkpoint_before_frozen": before,
        "checkpoint_after_frozen": after,
        "frozen_checkpoint_immutable": before == after,
        "frozen_learning_updates_zero": all(
            value == 0 for value in clone.audit.learning_updates.values()
        ),
        "strict_sparse_environment_reward": all(
            step.reward == 0.0 or step.terminal
            for record in (*records, frozen)
            for step in record.transitions
        ),
    }, [record.to_dict() for record in (*records, frozen)]


def run_grid_push_core_comparison_development(
    config: Mapping[str, Any],
    *,
    run_id: str,
    repository_root: str | Path | None = None,
) -> GridCoreComparisonArtifacts:
    root = Path(repository_root or Path.cwd()).resolve()
    identity = build_run_identity(config, run_id=run_id, repository_root=root)
    output = v2_run_directory(identity, repository_root=root)
    reserve_run(output, identity, resume=False)
    settings = config["grid_push_core_comparison"]
    started = datetime.now(timezone.utc).isoformat()

    generator = ProceduralGridPushGenerator(
        maximum_attempts=int(settings["maximum_generation_attempts"])
    )
    specs: dict[int, GridPushSpec] = {}
    certifications = []
    reference_dir = output / "manifests" / "solver_references"
    for world_seed in settings["certification_world_seeds"]:
        spec, certification, solver_result = generator.generate(
            int(world_seed),
            maximum_actions=int(settings["solver_maximum_actions"]),
            random_rollouts=int(settings["certification_random_rollouts"]),
        )
        specs[int(world_seed)] = spec
        reference = freeze_solver_reference(
            reference_dir / f"world_{int(world_seed)}.json",
            world=GridPushWorld(spec),
            solver_result=solver_result,
            maximum_actions=int(settings["solver_maximum_actions"]),
        )
        row = certification.to_dict()
        row["frozen_reference_sha256"] = reference["reference_sha256"]
        certifications.append(row)
    del generator

    fixture, fixture_traces = _module_fixture(
        _core_config(settings, fixture=True),
        int(settings.get("module_fixture_episodes", 6)),
    )
    worlds = tuple(int(seed) for seed in settings["certification_world_seeds"])[
        : int(settings["training_world_count"])
    ]
    training_episodes = int(settings["training_episodes"])
    evaluation_episodes = int(settings["evaluation_episodes"])
    maximum_steps = int(settings["episode_maximum_steps"])
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    core_manifests: dict[str, list[str]] = {}
    for research_seed in config["research_seeds"]:
        rows, integrity = _run_random(
            specs,
            worlds,
            research_seed=int(research_seed),
            episodes=evaluation_episodes,
            maximum_steps=maximum_steps,
        )
        all_rows.extend(rows)
        summaries.append(_summary("random", rows, integrity))

        rows, integrity = _run_contextual(
            specs,
            worlds,
            research_seed=int(research_seed),
            training_episodes=training_episodes,
            evaluation_episodes=evaluation_episodes,
            maximum_steps=maximum_steps,
        )
        all_rows.extend(rows)
        summaries.append(_summary("contextual_policy", rows, integrity))

        rows, integrity = _run_reduced(
            specs,
            worlds,
            research_seed=int(research_seed),
            training_episodes=training_episodes,
            evaluation_episodes=evaluation_episodes,
            maximum_steps=maximum_steps,
        )
        all_rows.extend(rows)
        summaries.append(_summary("reduced_causal_agent", rows, integrity))

        for factory in (
            build_tabular_fixed_goal_core,
            build_no_imagination_core,
            build_full_aassr_core,
        ):
            rows, integrity, core = _run_core_condition(
                factory,
                specs,
                worlds,
                config=_core_config(settings),
                research_seed=int(research_seed),
                training_episodes=training_episodes,
                evaluation_episodes=evaluation_episodes,
                maximum_steps=maximum_steps,
            )
            core_manifests[core.condition_name] = list(
                core.component_class_manifest()
            )
            all_rows.extend(rows)
            summaries.append(_summary(core.condition_name, rows, integrity))

    condition_components = {
        "random": ["random.Random", "GridPushEnvironmentPlugin"],
        "contextual_policy": [
            "ContextualPolicy",
            "CoreObservationEncoder",
            "GridPushEnvironmentPlugin",
        ],
        "reduced_causal_agent": [
            "CausalAASSRAgent",
            "GridRelationalEffectEncoder",
            "RepresentedReturnAgent",
            "EmpiricalCausalProphecy",
            "CausalImaginationPlanner",
            "GridPushEnvironmentPlugin",
        ],
        **{
            name: [*classes, "GridPushEnvironmentPlugin"]
            for name, classes in core_manifests.items()
        },
    }
    raw_dir = output / "raw"
    columns = (
        "condition",
        "research_seed",
        "phase",
        "episode",
        "world_seed",
        "training",
        "success",
        "final_sparse_reward",
        "primitive_steps",
        "decision_count",
        "imagination_decisions",
        "imagined_nodes",
    )
    with V2ArtifactWriter(raw_dir, columns) as writer:
        for row in all_rows:
            decisions = row.get("decisions", ())
            writer.write_episode(
                {
                    "condition": row["condition"],
                    "research_seed": row["research_seed"],
                    "phase": row["phase"],
                    "episode": row["episode"],
                    "world_seed": row["world_seed"],
                    "training": row["training"],
                    "success": row["success"],
                    "final_sparse_reward": row["final_sparse_reward"],
                    "primitive_steps": row["primitive_steps"],
                    "decision_count": len(decisions),
                    "imagination_decisions": sum(
                        bool(item.get("used_imagination")) for item in decisions
                    ),
                    "imagined_nodes": sum(
                        int(item.get("imagined_nodes", 0)) for item in decisions
                    ),
                }
            )
            writer.write_trace({"record_type": "condition_episode", **row})
        for row in fixture_traces:
            writer.write_trace({"record_type": "module_fixture", **row})
        episode_rows = writer.episode_rows
        trace_rows = writer.trace_rows

    evidence_path = output / "module_connection_evidence.json"
    evidence_path.write_text(
        json.dumps(fixture, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    components_path = output / "condition_component_classes.json"
    components_path.write_text(
        json.dumps(condition_components, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_path = output / "research_seed_summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    engineering_gates = {
        "worlds_certified": all(row["adequate"] for row in certifications),
        "six_condition_names_exact": set(condition_components) == set(CONDITIONS),
        "full_core_connection_evidence_positive": fixture[
            "all_required_evidence_positive"
        ],
        "full_core_frozen_checkpoint_immutable": fixture[
            "frozen_checkpoint_immutable"
        ],
        "full_core_frozen_learning_updates_zero": fixture[
            "frozen_learning_updates_zero"
        ],
        "strict_sparse_environment_reward": fixture[
            "strict_sparse_environment_reward"
        ],
        "all_learned_condition_checkpoints_immutable": all(
            bool(row["checkpoint_immutable"]) for row in summaries
        ),
        "all_evaluation_learning_calls_zero": all(
            int(row["evaluation_learning_calls"]) == 0 for row in summaries
        ),
        "gzip_trace_replays": (
            len(replay_gzip_trace(raw_dir / "trace.jsonl.gz")) == trace_rows
        ),
    }
    completed = datetime.now(timezone.utc).isoformat()
    artifacts = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            raw_dir / "episodes.csv",
            raw_dir / "trace.jsonl.gz",
            evidence_path,
            components_path,
            summary_path,
        )
    }
    manifest = {
        "schema_version": 2,
        **identity.to_dict(),
        "suite": "grid_push_core_comparison_development",
        "development_only_not_research_evidence": True,
        "locked_confirmation_created": False,
        "pilot_executed": False,
        "final_executed": False,
        "started_at_utc": started,
        "completed_at_utc": completed,
        "implementation_tree_sha256": implementation_tree_sha256(root),
        "causal_law_sha256": GRID_PUSH_LAW_SHA256,
        "condition_component_classes": condition_components,
        "world_certifications": certifications,
        "module_connection_evidence": fixture,
        "research_seed_summaries": summaries,
        "engineering_gates": engineering_gates,
        "episode_rows": episode_rows,
        "trace_rows": trace_rows,
        "artifact_sha256": artifacts,
    }
    manifest_path = output / "manifests" / "protocol_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path = output / "report.md"
    report_path.write_text(
        "# GridPush core comparison Development Diagnostic\n\n"
        "Engineering connection evidence only; not performance evidence. "
        "No Confirmation, Pilot, or Final was run.\n\n"
        f"- Conditions: {', '.join(CONDITIONS)}\n"
        f"- Certified worlds: {len(certifications)}\n"
        f"- Connection evidence: {'PASS' if fixture['all_required_evidence_positive'] else 'FAIL'}\n"
        f"- Engineering gates: {'PASS' if all(engineering_gates.values()) else 'FAIL'}\n"
        f"- Episode rows: {episode_rows}\n"
        f"- Trace rows: {trace_rows}\n",
        encoding="utf-8",
    )
    return GridCoreComparisonArtifacts(
        output_dir=output,
        manifest_path=manifest_path,
        report_path=report_path,
        episode_rows=episode_rows,
    )
