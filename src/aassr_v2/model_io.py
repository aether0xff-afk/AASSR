from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import threading
from typing import Any, Mapping

from .autonomous_agent import HoldoutTransition, RunningValue
from .escape_reporting import serialize_agent_checkpoint
from .types import Action, StateSnapshot


MODEL_FORMAT = "aassr.escape.model"
MODEL_SCHEMA_VERSION = 1
MODEL_EXTENSION = ".aassr-model.gz"


class ModelFormatError(ValueError):
    pass


class ModelCompatibilityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ModelLoadInfo:
    path: str
    saved_at_utc: str
    completed_episodes: int
    training_config: Mapping[str, Any]
    label: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _hashable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _hashable(item)) for key, item in value.items()))
    return value


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _tuple_tree(item) for key, item in value.items()}
    return value


def _deserialize_action(payload: Mapping[str, Any]) -> Action:
    return Action(
        str(payload.get("verb", "")),
        target=payload.get("target"),
        tool=payload.get("tool"),
        destination=payload.get("destination"),
        metadata=dict(payload.get("metadata", {})),
        parameters=dict(payload.get("parameters", {})),
    )


def _deserialize_snapshot(payload: Mapping[str, Any]) -> StateSnapshot:
    return StateSnapshot(
        vector=tuple(float(value) for value in payload.get("vector", ())),
        facts=frozenset(str(value) for value in payload.get("facts", ())),
        available_actions=tuple(
            _deserialize_action(item) for item in payload.get("available_actions", ())
        ),
        goal_progress=float(payload.get("goal_progress", 0.0)),
        metadata=dict(payload.get("metadata", {})),
    )


def _config_mapping(config: object | Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    safe = _json_safe(config)
    return dict(safe) if isinstance(safe, Mapping) else {}


def _compatibility(training_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "environment": "colored_key_escape_gridworld",
        "color_count": training_config.get("color_count"),
        "distractor_boxes": training_config.get("distractor_boxes"),
        "agent_family": "AutonomousLearningAgent+ContextualPolicy+TabularProphecy",
    }


def save_agent_model(
    agent: object,
    path: str | Path,
    *,
    completed_episodes: int,
    training_config: object | Mapping[str, Any] | None = None,
    label: str = "",
    notes: str = "",
) -> Path:
    destination = Path(path)
    if not str(destination).endswith(MODEL_EXTENSION):
        destination = Path(str(destination) + MODEL_EXTENSION)
    destination.parent.mkdir(parents=True, exist_ok=True)
    config_payload = _config_mapping(training_config)
    payload = {
        "format": MODEL_FORMAT,
        "schema_version": MODEL_SCHEMA_VERSION,
        "saved_at_utc": _utc_now_iso(),
        "label": label,
        "notes": notes,
        "completed_episodes": int(completed_episodes),
        "training_config": config_payload,
        "compatibility": _compatibility(config_payload),
        "checkpoint": serialize_agent_checkpoint(
            agent,
            episode=int(completed_episodes),
        ),
    }
    encoded = json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    temporary = destination.with_name(destination.name + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=6) as handle:
        handle.write(encoded)
    temporary.replace(destination)
    return destination


def read_model(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelFormatError(f"모델 파일을 읽을 수 없습니다: {source}") from exc
    if not isinstance(payload, dict) or payload.get("format") != MODEL_FORMAT:
        raise ModelFormatError("AASSR Escape 모델 파일이 아닙니다.")
    if int(payload.get("schema_version", -1)) != MODEL_SCHEMA_VERSION:
        raise ModelFormatError(
            f"지원하지 않는 모델 스키마입니다: {payload.get('schema_version')}"
        )
    if not isinstance(payload.get("checkpoint"), dict):
        raise ModelFormatError("모델 checkpoint가 없습니다.")
    return payload


def _validate_compatibility(
    payload: Mapping[str, Any],
    expected_training_config: object | Mapping[str, Any] | None,
) -> None:
    expected = _config_mapping(expected_training_config)
    if not expected:
        return
    actual = payload.get("compatibility", {})
    if not isinstance(actual, Mapping):
        raise ModelCompatibilityError("모델 호환성 정보가 손상되었습니다.")
    mismatches = []
    for key in ("color_count", "distractor_boxes"):
        expected_value = expected.get(key)
        actual_value = actual.get(key)
        if expected_value is not None and actual_value is not None and expected_value != actual_value:
            mismatches.append(f"{key}: model={actual_value}, current={expected_value}")
    if mismatches:
        raise ModelCompatibilityError(
            "현재 GridWorld 설정과 모델이 호환되지 않습니다: " + ", ".join(mismatches)
        )


def restore_agent_checkpoint(agent: object, checkpoint: Mapping[str, Any]) -> None:
    policy = getattr(agent, "policy")
    prophecy = getattr(agent, "prophecy")
    holdout = getattr(agent, "holdout")

    policy_payload = checkpoint.get("policy", {})
    policy._local = {}
    for item in policy_payload.get("local", ()):  # type: ignore[attr-defined]
        state = _hashable(item["state"])
        policy._local[(state, str(item["action_signature"]))] = RunningValue(  # type: ignore[attr-defined]
            count=int(item.get("count", 0)),
            mean=float(item.get("mean", 0.0)),
        )
    policy._global = {  # type: ignore[attr-defined]
        str(signature): RunningValue(
            count=int(value.get("count", 0)),
            mean=float(value.get("mean", 0.0)),
        )
        for signature, value in policy_payload.get("global", {}).items()
    }
    policy._state_visits = {  # type: ignore[attr-defined]
        _hashable(item["state"]): int(item.get("visits", 0))
        for item in policy_payload.get("state_visits", ())
    }

    prophecy_payload = checkpoint.get("prophecy", {})
    prophecy._exact = defaultdict(Counter)  # type: ignore[attr-defined]
    for item in prophecy_payload.get("exact", ()):
        state = _hashable(item["state"])
        action_signature = str(item["action_signature"])
        counter = Counter()
        for next_item in item.get("next_states", ()):
            counter[_hashable(next_item["state"])] = int(next_item.get("count", 0))
        prophecy._exact[(state, action_signature)] = counter  # type: ignore[attr-defined]
    prophecy._global = defaultdict(Counter)  # type: ignore[attr-defined]
    for verb, entries in prophecy_payload.get("global", {}).items():
        counter = Counter()
        for item in entries:
            counter[_hashable(item["state"])] = int(item.get("count", 0))
        prophecy._global[str(verb)] = counter  # type: ignore[attr-defined]
    prophecy._states = {  # type: ignore[attr-defined]
        _hashable(item["fingerprint"]): _deserialize_snapshot(item["snapshot"])
        for item in prophecy_payload.get("states", ())
    }

    holdout_payload = checkpoint.get("holdout", {})
    holdout._seen = int(holdout_payload.get("seen", 0))  # type: ignore[attr-defined]
    holdout._items = [  # type: ignore[attr-defined]
        HoldoutTransition(
            _deserialize_snapshot(item["before"]),
            _deserialize_action(item["action"]),
            _deserialize_snapshot(item["after"]),
        )
        for item in holdout_payload.get("items", ())
    ]
    random_state = checkpoint.get("random_state")
    if random_state is not None:
        getattr(agent, "randomizer").setstate(_tuple_tree(random_state))
    holdout_random_state = holdout_payload.get("random_state")
    if holdout_random_state is not None:
        holdout._randomizer.setstate(_tuple_tree(holdout_random_state))  # type: ignore[attr-defined]

    agent._transition_index = int(checkpoint.get("transition_index", 0))  # type: ignore[attr-defined]
    agent._decision_index = int(checkpoint.get("decision_index", 0))  # type: ignore[attr-defined]
    agent._seen_effect_motifs = {  # type: ignore[attr-defined]
        _hashable(item)
        for item in checkpoint.get("effect_novelty_motifs", ())
    }
    agent._episode.clear()  # type: ignore[attr-defined]
    agent._recent_pairs.clear()  # type: ignore[attr-defined]


def load_agent_model(
    agent: object,
    path: str | Path,
    *,
    expected_training_config: object | Mapping[str, Any] | None = None,
) -> ModelLoadInfo:
    payload = read_model(path)
    _validate_compatibility(payload, expected_training_config)
    restore_agent_checkpoint(agent, payload["checkpoint"])
    return ModelLoadInfo(
        path=str(Path(path)),
        saved_at_utc=str(payload.get("saved_at_utc", "")),
        completed_episodes=int(payload.get("completed_episodes", 0)),
        training_config=dict(payload.get("training_config", {})),
        label=str(payload.get("label", "")),
    )


class ModelManagedAgent:
    """Serialize a coherent model while the training worker is active."""

    def __init__(self, agent: object, *, base_episode_offset: int = 0) -> None:
        self._agent = agent
        self._lock = threading.RLock()
        self.base_episode_offset = int(base_episode_offset)
        self.session_completed_episodes = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)

    @property
    def completed_episodes(self) -> int:
        return self.base_episode_offset + self.session_completed_episodes

    def epsilon(self, episode: int) -> float:
        with self._lock:
            return self._agent.epsilon(self.base_episode_offset + int(episode))

    def select_action(self, state: StateSnapshot, *, episode: int, explore: bool = True) -> Any:
        with self._lock:
            return self._agent.select_action(
                state,
                episode=self.base_episode_offset + int(episode),
                explore=explore,
            )

    def observe(self, before: StateSnapshot, action: Action, outcome: object) -> Any:
        with self._lock:
            return self._agent.observe(before, action, outcome)

    def finish_episode(self, *, final_return: float) -> None:
        with self._lock:
            self._agent.finish_episode(final_return=final_return)
            self.session_completed_episodes += 1

    def discard_episode(self) -> None:
        with self._lock:
            self._agent.discard_episode()

    def save_model(
        self,
        path: str | Path,
        *,
        training_config: object | Mapping[str, Any] | None = None,
        label: str = "",
        notes: str = "",
    ) -> Path:
        with self._lock:
            return save_agent_model(
                self._agent,
                path,
                completed_episodes=self.completed_episodes,
                training_config=training_config,
                label=label,
                notes=notes,
            )
