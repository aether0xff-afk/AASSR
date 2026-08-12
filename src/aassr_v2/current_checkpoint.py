from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .current_entrypoint import build_current_pentest_aassr_core
from .current_manifest import CURRENT_GENERATION_VERSION


CURRENT_FROZEN_CHECKPOINT_VERSION = "aassr-current-frozen-checkpoint-v1"


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


def current_frozen_checkpoint_payload(
    agent: object,
    *,
    research_seed: int,
    transition_budget: int,
    git_commit: str,
) -> dict[str, Any]:
    """Materialize all learned state needed for a fresh-process frozen evaluation.

    This intentionally does not attempt to pickle the live agent object because
    the current runtime installs several bound gate/optimization methods. Instead
    a fresh canonical agent is rebuilt and these explicit learned stores are
    restored into it.
    """
    base = agent.base_neural_prophecy
    replay = agent.evaluator.replay
    critic = agent.critic
    support_rows = getattr(agent, "_critic_support_rows", {})

    return {
        "checkpoint_version": CURRENT_FROZEN_CHECKPOINT_VERSION,
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
            "replay": list(agent.dqn.replay),
            "randomizer_state": agent.dqn.randomizer.getstate(),
        },
        "policy": {
            "information": dict(agent.policy._information),
            "skill_values": dict(agent.policy._skill_values),
        },
        "prophecy": {
            "models": _model_state_dicts(base.models),
            "observations": int(base.observations),
            "gradient_updates": int(base.gradient_updates),
            "replay": list(base.replay),
            "outcomes": dict(base._outcomes),
            "losses": list(base._losses),
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
            "train": list(replay._train),
            "holdout": list(replay._holdout),
            "seen": int(replay._seen),
        },
        "critic": {
            "gru": critic.gru.state_dict(),
            "output": critic.output.state_dict(),
            "episodes": int(critic.episodes),
            "transitions": int(critic.transitions),
            "gradient_updates": int(critic.gradient_updates),
            "losses": list(critic._losses),
            "agent_counts": dict(agent._critic_counts),
            "support_rows": {
                key: list(rows) for key, rows in support_rows.items()
            },
            "support_diagnostics": dict(
                getattr(agent, "_critic_support_diagnostics", {})
            ),
        },
        "skills": dict(agent.skills.__dict__),
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


def _restore_deque(target: Any, values: Any) -> None:
    target.clear()
    target.extend(values)


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
    _restore_deque(agent.dqn.replay, dqn["replay"])
    agent.dqn.randomizer.setstate(dqn["randomizer_state"])

    policy = payload["policy"]
    agent.policy._information.clear()
    agent.policy._information.update(policy["information"])
    agent.policy._skill_values.clear()
    agent.policy._skill_values.update(policy["skill_values"])

    prophecy = payload["prophecy"]
    base = agent.base_neural_prophecy
    _restore_model_state_dicts(base.models, prophecy["models"])
    base.observations = int(prophecy["observations"])
    base.gradient_updates = int(prophecy["gradient_updates"])
    _restore_deque(base.replay, prophecy["replay"])
    base._outcomes.clear()
    base._outcomes.update(prophecy["outcomes"])
    _restore_deque(base._losses, prophecy.get("losses", ()))
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
    replay._train[:] = list(calibration["train"])
    replay._holdout[:] = list(calibration["holdout"])
    replay._seen = int(calibration["seen"])
    agent.calibrated_prophecy._cache.clear()

    critic_payload = payload["critic"]
    critic = agent.critic
    critic.gru.load_state_dict(critic_payload["gru"])
    critic.output.load_state_dict(critic_payload["output"])
    critic.episodes = int(critic_payload["episodes"])
    critic.transitions = int(critic_payload["transitions"])
    critic.gradient_updates = int(critic_payload["gradient_updates"])
    _restore_deque(critic._losses, critic_payload.get("losses", ()))
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

    # Mutate the existing library rather than replacing it so installed Skill
    # Prophecy wrappers keep the same object reference.
    agent.skills.__dict__.clear()
    agent.skills.__dict__.update(payload.get("skills", {}))

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
    }
