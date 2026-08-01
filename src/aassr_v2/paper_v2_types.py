from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class V2StudyStage(str, Enum):
    DEVELOPMENT_DIAGNOSTIC = "development_diagnostic"
    LOCKED_CONFIRMATION = "locked_confirmation"
    PILOT = "pilot"


@dataclass(frozen=True, slots=True)
class RawCausalObservation:
    """The only environment data available to Protocol v2 agents."""

    inventory: Mapping[str, int] = field(default_factory=dict)
    observable_facts: frozenset[str] = frozenset()
    available_actions: tuple[str, ...] = ()
    action_affordances: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    resource_cost: float = 0.0
    health: float = 1.0
    damage: float = 0.0
    spatial_observations: Mapping[str, str | float | int] = field(
        default_factory=dict
    )
    last_action_succeeded: bool | None = None
    terminal_reward: float = 0.0
    terminal: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["observable_facts"] = sorted(self.observable_facts)
        payload["available_actions"] = list(self.available_actions)
        payload["action_affordances"] = {
            key: list(value) for key, value in self.action_affordances.items()
        }
        return payload


@dataclass(frozen=True, slots=True)
class FullAgentCheckpoint:
    policy: Mapping[str, Any] = field(default_factory=dict)
    prophecy: Mapping[str, Any] = field(default_factory=dict)
    holdout: Mapping[str, Any] = field(default_factory=dict)
    rng: Any = None
    planner_cache: Mapping[str, Any] = field(default_factory=dict)
    counters: Mapping[str, int] = field(default_factory=dict)
    replay_buffer: tuple[Mapping[str, Any], ...] = ()
    normalization_state: Mapping[str, Any] = field(default_factory=dict)
    calibration_buffer: tuple[Mapping[str, Any], ...] = ()
    relational_representation: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class V2RunIdentity:
    protocol_version: str
    stage: V2StudyStage
    run_id: str
    config_sha256: str
    seed_commitment_sha256: str
    causal_law_sha256: str
    implementation_commit: str

    def to_dict(self) -> dict[str, str]:
        payload = asdict(self)
        payload["stage"] = self.stage.value
        return payload


@dataclass(frozen=True, slots=True)
class CausalProphecyPredictionV20:
    next_observable_state: RawCausalObservation | None
    observable_effect_delta: Mapping[str, float]
    action_unlock_probability: float
    terminal_return_probability: float
    visit_count: int


@dataclass(frozen=True, slots=True)
class CausalProphecyPredictionV21:
    base: CausalProphecyPredictionV20
    expected_resource_cost: float
    expected_damage: float
    uncertainty: float
    ood_score: float
    calibration_confidence: float
