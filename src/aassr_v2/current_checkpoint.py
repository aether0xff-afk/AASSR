from __future__ import annotations

from collections import Counter
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping

from .autonomous_agent_core import RunningValue
from .current_entrypoint import build_current_pentest_aassr_core
from .current_manifest import CURRENT_GENERATION_VERSION
from .replay import ReplayTransition
from .skills import Skill
from .types import Action, StateSnapshot


CURRENT_FROZEN_CHECKPOINT_VERSION = "aassr-current-frozen-checkpoint-v2-portable"


def _model_state_dicts(models: Any) -> list[dict[str, Any]]:
    return [model.state_dict() for model in models]


def _restore_model_state_dicts(models: Any, rows: list[dict[str, Any]]) -> None:
    if len(models) != len(rows):
        raise ValueError(
            "checkpoint/model ensemble-size mismatch: "
            f"{len(rows)} != {len(models)}"
        )
    for model, state in zip(models, rows, strict=True):
        model.load_state_dict(state)


def _plain(value: Any) -> Any:
    """Recursively remove immutable mapping proxies from public payload fields."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, MappingABC):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, frozenset):
        return frozenset(_plain(item) for item in value)
    if isinstance(value, set):
        return set(_plain(item) for item in value)
    raise TypeError(f"unsupported portable checkpoint value: {type(value)!r}")


def _action_to_data(action: Action) -> dict[str, Any]:
    return {
        "verb": action.verb_name,
        "target": action.target,
        "tool": action.tool,
        "destination": action.destination,
        "metadata": _plain(action.metadata),
        "parameters": _plain(action.parameters),
    }


def _action_from_data(data: Mapping[str, Any]) -> Action:
    return Action(
        str(data["verb"]),
        target=data.get("target"),
        tool=data.get("tool"),
        destination=data.get("destination"),
        metadata=dict(data.get("metadata", {})),
        parameters=dict(data.get("parameters", {})),
    )


def _state_to_data(state: StateSnapshot) -> dict[str, Any]:
    return {
        "vector": tuple(float(value) for value in state.vector),
        "facts": tuple(sorted(str(fact) for fact in state.facts)),
        "available_actions": [
            _action_to_data(action) for action in state.available_actions
        ],
        "goal_progress": float(state.goal_progress),
        "metadata": _plain(state.metadata),
    }


def _state_from_data(data: Mapping[str, Any]) -> StateSnapshot:
    return StateSnapshot(
        vector=tuple(float(value) for value in data["vector"]),
        facts=frozenset(str(fact) for fact in data.get("facts", ())),
        available_actions=tuple(
            _action_from_data(action)
            for action in data.get("available_actions", ())
        ),
        goal_progress=float(data.get("goal_progress", 0.0)),
        metadata=dict(data.get("metadata", {})),
    )


def _replay_transition_to_data(item: ReplayTransition) -> dict[str, Any]:
    return {
        "state": _state_to_data(item.state),
        "action": _action_to_data(item.action),
        "next_state": _state_to_data(item.next_state),
        "trace_id": str(item.trace_id),
    }


def _replay_transition_from_data(data: Mapping[str, Any]) -> ReplayTransition:
    return ReplayTransition(
        _state_from_data(data["state"]),
        _action_from_data(data["action"]),
        _state_from_data(data["next_state"]),
        str(data.get("trace_id", "")),
    )


def _running_value_to_data(value: RunningValue) -> dict[str, Any]:
    return {"count": int(value.count), "mean": float(value.mean)}


def _running_value_from_data(data: Mapping[str, Any]) -> RunningValue:
    return RunningValue(count=int(data["count"]), mean=float(data["mean"]))


def _skill_to_data(skill: Skill) -> dict[str, Any]:
    return {
        "skill_id": str(skill.skill_id),
        "primitive_actions": [
            _action_to_data(action) for action in skill.primitive_actions
        ],
        "achieved_goal_ids": tuple(str(item) for item in skill.achieved_goal_ids),
        "required_facts": tuple(sorted(str(item) for item in skill.required_facts)),
        "added_facts": tuple(sorted(str(item) for item in skill.added_facts)),
        "removed_facts": tuple(sorted(str(item) for item in skill.removed_facts)),
        "successes": int(skill.successes),
        "failures": int(skill.failures),
    }


def _skill_from_data(data: Mapping[str, Any]) -> Skill:
    return Skill(
        skill_id=str(data["skill_id"]),
        primitive_actions=tuple(
            _action_from_data(action)
            for action in data.get("primitive_actions", ())
        ),
        achieved_goal_ids=tuple(str(item) for item in data.get("achieved_goal_ids", ())),
        required_facts=frozenset(str(item) for item in data.get("required_facts", ())),
        added_facts=frozenset(str(item) for item in data.get("added_facts", ())),
        removed_facts=frozenset(str(item) for item in data.get("removed_facts", ())),
        successes=int(data.get("successes", 0)),
        failures=int(data.get("failures", 0)),
    )


def _skills_to_data(library: object) -> dict[str, Any]:
    """Store only state used by frozen selection/Skill rollout."""
    return {
        "promotion_successes": int(library.promotion_successes),
        "maximum_length": int(library.maximum_length),
        "next_id": int(library._next_id),
        "skills": [_skill_to_data(skill) for skill in library.all()],
        "templates": {
            str(skill_id): tuple(tuple(float(value) for value in row) for row in rows)
            for skill_id, rows in library._templates.items()
        },
    }


def _restore_skills(library: object, data: Mapping[str, Any]) -> None:
    if int(data.get("promotion_successes", library.promotion_successes)) != int(
        library.promotion_successes
    ):
        raise ValueError("checkpoint Skill promotion contract differs from runtime")
    if int(data.get("maximum_length", library.maximum_length)) != int(
        library.maximum_length
    ):
        raise ValueError("checkpoint Skill maximum-length contract differs from runtime")
    library._skills.clear()
    for row in data.get("skills", ()):
        skill = _skill_from_data(row)
        library._skills[skill.skill_id] = skill
    library._templates.clear()
    library._templates.update(
        {
            str(skill_id): tuple(
                tuple(float(value) for value in template)
                for template in templates
            )
            for skill_id, templates in data.get("templates", {}).items()
        }
    )
    library._next_id = int(data.get("next_id", 1))
    library._candidates.clear()
    library._rel_candidates.clear()


def current_frozen_checkpoint_payload(
    agent: object,
    *,
    research_seed: int,
    transition_budget: int,
    git_commit: str,
) -> dict[str, Any]:
    """Materialize learned state required for fresh-process frozen evaluation.

    Live agents contain MappingProxyType fields and runtime-bound methods, so this
    format never pickles the agent or object-rich replay directly. Public holdout
    observations and promoted Skills are encoded into explicit plain structures.
    Training-only optimizer/replay state is deliberately omitted: this artifact's
    contract is exact frozen OFF/ON re-evaluation without retraining.
    """
    base = agent.base_neural_prophecy
    replay = agent.evaluator.replay
    critic = agent.critic
    support_rows = getattr(agent, "_critic_support_rows", {})

    return {
        "checkpoint_version": CURRENT_FROZEN_CHECKPOINT_VERSION,
        "checkpoint_scope": "frozen-evaluation-only",
        "architecture_version": CURRENT_GENERATION_VERSION,
        "git_commit": str(git_commit),
        "research_seed": int(research_seed),
        "transition_budget": int(transition_budget),
        "requested_imagination": bool(agent.requested_imagination),
        "agent_config": {
            "imagination_intervention_margin": float(
                agent.config.imagination_intervention_margin
            ),
            "imagination_minimum_coverage": float(
                agent.config.imagination_minimum_coverage
            ),
            "gamma": float(agent.config.gamma),
        },
        "dqn": {
            "online": agent.dqn.online.state_dict(),
            "target": agent.dqn.target.state_dict(),
            "environment_steps": int(agent.dqn.environment_steps),
            "gradient_updates": int(agent.dqn.gradient_updates),
            "randomizer_state": agent.dqn.randomizer.getstate(),
        },
        "policy": {
            "information": [
                {
                    "state_key": tuple(float(value) for value in state_key),
                    "action_key": tuple(float(value) for value in action_key),
                    "value": _running_value_to_data(value),
                }
                for (state_key, action_key), value in agent.policy._information.items()
            ],
            "skill_values": {
                str(key): _running_value_to_data(value)
                for key, value in agent.policy._skill_values.items()
            },
        },
        "prophecy": {
            "models": _model_state_dicts(base.models),
            "observations": int(base.observations),
            "gradient_updates": int(base.gradient_updates),
            "outcomes": dict(base._outcomes),
            "status_observation_counts": list(
                getattr(base, "_status_observation_counts", ())
            ),
            "status_no_observation_count": int(
                getattr(base, "_status_no_observation_count", 0)
            ),
            "last_status_training_loss": float(
                getattr(base, "_last_status_training_loss", 0.0)
            ),
            "last_status_training_accuracy": float(
                getattr(base, "_last_status_training_accuracy", 0.0)
            ),
        },
        "calibration_replay": {
            # SemanticCalibratedProphecy consults only holdout() during frozen
            # confidence evaluation. Keep the observed train count for provenance
            # but do not duplicate the much larger object-rich training partition.
            "train_count": len(replay._train),
            "holdout": [
                _replay_transition_to_data(item) for item in replay._holdout
            ],
            "seen": int(replay._seen),
        },
        "critic": {
            "gru": critic.gru.state_dict(),
            "output": critic.output.state_dict(),
            "episodes": int(critic.episodes),
            "transitions": int(critic.transitions),
            "gradient_updates": int(critic.gradient_updates),
            "agent_counts": dict(agent._critic_counts),
            "support_rows": {
                key: list(rows) for key, rows in support_rows.items()
            },
            "support_diagnostics": dict(
                getattr(agent, "_critic_support_diagnostics", {})
            ),
        },
        "skills": _skills_to_data(agent.skills),
        "agent_runtime": {
            "randomizer_state": agent.randomizer.getstate(),
            "steps": int(agent._steps),
            "skill_uses": int(agent._skill_uses),
            "promoted_skills": int(agent._promoted_skills),
            "imagination_diagnostics": dict(agent._imagination_diagnostics),
        },
    }


def save_current_frozen_checkpoint(
    agent: object,
    path: str | Path,
    *,
    research_seed: int,
    transition_budget: int,
    git_commit: str,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = current_frozen_checkpoint_payload(
        agent,
        research_seed=research_seed,
        transition_budget=transition_budget,
        git_commit=git_commit,
    )
    agent.dqn.torch.save(payload, target)
    return target


def restore_current_frozen_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    allow_tf32: bool = True,
    expected_git_commit: str | None = None,
) -> object:
    """Rebuild the canonical runtime and restore a frozen research checkpoint."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("restoring current checkpoint requires torch") from exc

    payload = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )
    if payload.get("checkpoint_version") != CURRENT_FROZEN_CHECKPOINT_VERSION:
        raise ValueError("unsupported current frozen checkpoint version")
    if payload.get("checkpoint_scope") != "frozen-evaluation-only":
        raise ValueError("current checkpoint does not declare frozen-evaluation scope")
    if payload.get("architecture_version") != CURRENT_GENERATION_VERSION:
        raise ValueError(
            "checkpoint architecture mismatch: "
            f"{payload.get('architecture_version')!r} != {CURRENT_GENERATION_VERSION!r}"
        )
    if expected_git_commit is not None and str(payload.get("git_commit")) != str(
        expected_git_commit
    ):
        raise ValueError("checkpoint git commit does not match requested runtime")

    agent = build_current_pentest_aassr_core(
        seed=int(payload["research_seed"]),
        train_transitions=int(payload["transition_budget"]),
        use_imagination=bool(payload.get("requested_imagination", True)),
        device=device,
        allow_tf32=bool(allow_tf32),
    )
    configured_margin = float(
        payload.get("agent_config", {}).get(
            "imagination_intervention_margin",
            agent.config.imagination_intervention_margin,
        )
    )
    if abs(configured_margin - agent.config.imagination_intervention_margin) > 1e-12:
        raise ValueError(
            "checkpoint intervention-margin contract differs from canonical runtime"
        )

    dqn = payload["dqn"]
    agent.dqn.online.load_state_dict(dqn["online"])
    agent.dqn.target.load_state_dict(dqn["target"])
    agent.dqn.environment_steps = int(dqn["environment_steps"])
    agent.dqn.gradient_updates = int(dqn["gradient_updates"])
    agent.dqn.randomizer.setstate(dqn["randomizer_state"])

    policy = payload["policy"]
    agent.policy._information.clear()
    for row in policy.get("information", ()):
        key = (
            tuple(float(value) for value in row["state_key"]),
            tuple(float(value) for value in row["action_key"]),
        )
        agent.policy._information[key] = _running_value_from_data(row["value"])
    agent.policy._skill_values.clear()
    agent.policy._skill_values.update(
        {
            str(key): _running_value_from_data(value)
            for key, value in policy.get("skill_values", {}).items()
        }
    )

    prophecy = payload["prophecy"]
    base = agent.base_neural_prophecy
    _restore_model_state_dicts(base.models, prophecy["models"])
    base.observations = int(prophecy["observations"])
    base.gradient_updates = int(prophecy["gradient_updates"])
    base._outcomes.clear()
    base._outcomes.update(prophecy["outcomes"])
    if hasattr(base, "_status_observation_counts"):
        base._status_observation_counts[:] = list(
            prophecy.get("status_observation_counts", ())
        )
        base._status_no_observation_count = int(
            prophecy.get("status_no_observation_count", 0)
        )
        base._last_status_training_loss = float(
            prophecy.get("last_status_training_loss", 0.0)
        )
        base._last_status_training_accuracy = float(
            prophecy.get("last_status_training_accuracy", 0.0)
        )

    calibration = payload["calibration_replay"]
    replay = agent.evaluator.replay
    replay._train.clear()
    replay._holdout[:] = [
        _replay_transition_from_data(item)
        for item in calibration.get("holdout", ())
    ]
    replay._seen = int(calibration["seen"])
    agent.calibrated_prophecy._cache.clear()

    critic_payload = payload["critic"]
    critic = agent.critic
    critic.gru.load_state_dict(critic_payload["gru"])
    critic.output.load_state_dict(critic_payload["output"])
    critic.episodes = int(critic_payload["episodes"])
    critic.transitions = int(critic_payload["transitions"])
    critic.gradient_updates = int(critic_payload["gradient_updates"])
    agent._critic_counts.clear()
    agent._critic_counts.update(critic_payload["agent_counts"])

    support_rows = getattr(agent, "_critic_support_rows", None)
    if support_rows is None:
        raise RuntimeError("canonical runtime did not expose Critic support store")
    support_rows.clear()
    for key, rows in critic_payload.get("support_rows", {}).items():
        support_rows[key].extend(rows)
    support_diagnostics = getattr(agent, "_critic_support_diagnostics", None)
    if support_diagnostics is not None:
        support_diagnostics.clear()
        support_diagnostics.update(critic_payload.get("support_diagnostics", {}))

    _restore_skills(agent.skills, payload.get("skills", {}))

    runtime = payload.get("agent_runtime", {})
    if "randomizer_state" in runtime:
        agent.randomizer.setstate(runtime["randomizer_state"])
    agent._steps = int(runtime.get("steps", 0))
    agent._skill_uses = int(runtime.get("skill_uses", 0))
    agent._promoted_skills = int(runtime.get("promoted_skills", 0))
    agent._imagination_diagnostics = Counter(
        runtime.get("imagination_diagnostics", {})
    )
    return agent


def checkpoint_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Small JSON-safe manifest for artifact provenance without loading tensors."""
    return {
        "checkpoint_version": payload["checkpoint_version"],
        "checkpoint_scope": payload["checkpoint_scope"],
        "architecture_version": payload["architecture_version"],
        "git_commit": payload["git_commit"],
        "research_seed": payload["research_seed"],
        "transition_budget": payload["transition_budget"],
        "imagination_intervention_margin": payload["agent_config"][
            "imagination_intervention_margin"
        ],
        "dqn_environment_steps": payload["dqn"]["environment_steps"],
        "prophecy_observations": payload["prophecy"]["observations"],
        "critic_episodes": payload["critic"]["episodes"],
        "calibration_train_rows_observed": payload["calibration_replay"]["train_count"],
        "calibration_train_rows_stored": 0,
        "calibration_holdout_rows": len(payload["calibration_replay"]["holdout"]),
        "promoted_skills": len(payload["skills"]["skills"]),
    }
