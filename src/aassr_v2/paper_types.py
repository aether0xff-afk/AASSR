from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class ExperimentPhase(str, Enum):
    TRAINING = "training"
    EVALUATION_SEEN = "evaluation_seen"
    EVALUATION_UNSEEN_ZERO_SHOT = "evaluation_unseen_zero_shot"
    ADAPTATION = "adaptation"
    EVALUATION_UNSEEN_ADAPTATION = "evaluation_unseen_adaptation"

    @property
    def permits_learning(self) -> bool:
        return self in {self.TRAINING, self.ADAPTATION}


@dataclass(slots=True)
class BudgetLedger:
    """Track real and imagined work without allowing budget overruns."""

    real_transition_limit: int
    action_proposal_limit: int | None = None
    wall_clock_limit_seconds: float | None = None
    real_transitions: int = 0
    imagined_transitions: int = 0
    action_proposals: int = 0

    def __post_init__(self) -> None:
        if self.real_transition_limit < 0:
            raise ValueError("real_transition_limit must be non-negative")
        if self.action_proposal_limit is not None and self.action_proposal_limit < 0:
            raise ValueError("action_proposal_limit must be non-negative")
        if (
            self.wall_clock_limit_seconds is not None
            and self.wall_clock_limit_seconds <= 0.0
        ):
            raise ValueError("wall_clock_limit_seconds must be positive")

    @property
    def real_remaining(self) -> int:
        return self.real_transition_limit - self.real_transitions

    def consume_real(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        if self.real_transitions + count > self.real_transition_limit:
            raise RuntimeError("real environment transition budget exceeded")
        self.real_transitions += count

    def record_imagined(self, count: int) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        self.imagined_transitions += count

    def record_proposals(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        if (
            self.action_proposal_limit is not None
            and self.action_proposals + count > self.action_proposal_limit
        ):
            raise RuntimeError("action proposal budget exceeded")
        self.action_proposals += count

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AgentCheckpointParts:
    """Serializable learning state split into independently retainable parts."""

    policy: Mapping[str, Any] = field(default_factory=dict)
    prophecy: Mapping[str, Any] = field(default_factory=dict)
    holdout: Mapping[str, Any] = field(default_factory=dict)
    effect_representation: Mapping[str, Any] = field(default_factory=dict)
    counters: Mapping[str, Any] = field(default_factory=dict)
    random_state: Any = None

    def selected(
        self,
        *,
        policy: bool,
        prophecy: bool,
        holdout: bool,
        effect_representation: bool,
    ) -> AgentCheckpointParts:
        return AgentCheckpointParts(
            self.policy if policy else {},
            self.prophecy if prophecy else {},
            self.holdout if holdout else {},
            self.effect_representation if effect_representation else {},
            self.counters,
            self.random_state,
        )


@dataclass(frozen=True, slots=True)
class EffectProfile:
    """Name-independent empirical action-effect representation."""

    executions: int = 0
    error_rate: float = 0.0
    mean_state_change: float = 0.0
    mean_facts_added: float = 0.0
    mean_facts_removed: float = 0.0
    unlock_rate: float = 0.0
    mean_risk_change: float = 0.0
    mean_goal_change: float = 0.0
    prediction_uncertainty: float = 0.0
    information_gain: float = 0.0

    @classmethod
    def from_observations(
        cls, observations: Sequence[Mapping[str, float | int | bool]]
    ) -> EffectProfile:
        if not observations:
            return cls()

        def mean(name: str) -> float:
            return sum(float(item.get(name, 0.0)) for item in observations) / len(
                observations
            )

        return cls(
            executions=len(observations),
            error_rate=mean("error"),
            mean_state_change=mean("state_change"),
            mean_facts_added=mean("facts_added"),
            mean_facts_removed=mean("facts_removed"),
            unlock_rate=mean("unlocked"),
            mean_risk_change=mean("risk_change"),
            mean_goal_change=mean("goal_change"),
            prediction_uncertainty=mean("prediction_uncertainty"),
            information_gain=mean("information_gain"),
        )

    def vector(self) -> tuple[float, ...]:
        return (
            self.error_rate,
            self.mean_state_change,
            self.mean_facts_added,
            self.mean_facts_removed,
            self.unlock_rate,
            self.mean_risk_change,
            self.mean_goal_change,
            self.prediction_uncertainty,
            self.information_gain,
        )


@dataclass(frozen=True, slots=True)
class CausalEffectGraph:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str, str], ...]
    effect_sequence: tuple[str, ...]
    solution_family: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(sorted(set(self.nodes))))
        object.__setattr__(self, "edges", tuple(sorted(set(self.edges))))

    @property
    def motifs(self) -> frozenset[str]:
        return frozenset(
            f"{source}>{relation}>{target}"
            for source, target, relation in self.edges
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "edges": [list(edge) for edge in self.edges],
            "effect_sequence": list(self.effect_sequence),
            "solution_family": self.solution_family,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CausalEffectGraph:
        return cls(
            nodes=tuple(str(item) for item in payload.get("nodes", ())),
            edges=tuple(
                (str(item[0]), str(item[1]), str(item[2]))
                for item in payload.get("edges", ())
            ),
            effect_sequence=tuple(
                str(item) for item in payload.get("effect_sequence", ())
            ),
            solution_family=str(payload.get("solution_family", "")),
        )


@dataclass(frozen=True, slots=True)
class StrategyRecord:
    strategy_id: str
    source_kind: str
    research_seed: int
    world_seed: int
    success: bool
    primitive_steps: int
    errors: int
    resources_used: float
    risk_entries: int
    graph: CausalEffectGraph
    trace: tuple[Mapping[str, Any], ...] = ()
    novelty_components: Mapping[str, float] = field(default_factory=dict)
    reusable_success_rate: float = 0.0
    novelty_score: float | None = None
    valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["graph"] = self.graph.to_dict()
        payload["trace"] = [dict(item) for item in self.trace]
        payload["novelty_components"] = dict(self.novelty_components)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StrategyRecord:
        return cls(
            strategy_id=str(payload["strategy_id"]),
            source_kind=str(payload["source_kind"]),
            research_seed=int(payload["research_seed"]),
            world_seed=int(payload["world_seed"]),
            success=bool(payload["success"]),
            primitive_steps=int(payload["primitive_steps"]),
            errors=int(payload["errors"]),
            resources_used=float(payload["resources_used"]),
            risk_entries=int(payload["risk_entries"]),
            graph=CausalEffectGraph.from_dict(payload["graph"]),
            trace=tuple(
                dict(item)
                for item in payload.get("trace", ())
                if isinstance(item, Mapping)
            ),
            novelty_components={
                str(key): float(value)
                for key, value in dict(
                    payload.get("novelty_components", {})
                ).items()
            },
            reusable_success_rate=float(payload.get("reusable_success_rate", 0.0)),
            novelty_score=(
                None
                if payload.get("novelty_score") is None
                else float(payload["novelty_score"])
            ),
            valid=bool(payload.get("valid", True)),
        )


@dataclass(frozen=True, slots=True)
class PaperManifest:
    protocol_version: str
    study_stage: str
    git_commit_sha: str
    config_sha256: str
    research_seeds: tuple[int, ...]
    world_seeds: Mapping[str, tuple[int, ...]]
    phase_definitions: Mapping[str, bool]
    started_at_utc: str
    completed_at_utc: str
    software: Mapping[str, str]
    hardware: Mapping[str, Any]
    execution: Mapping[str, Any]
    failed_runs: tuple[Mapping[str, Any], ...] = ()
    excluded_runs: tuple[Mapping[str, Any], ...] = ()
    human_dataset_version: str = ""
    human_approval_id: str = ""
    protocol_locks: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
