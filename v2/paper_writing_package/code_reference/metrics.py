from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Iterable

from .gridworld import ActionCandidate, StepResult
from .knowledge import KK


@dataclass(frozen=True)
class StepMetric:
    condition: str
    seed: int
    episode: int
    step: int
    action: str
    template: str
    action_signature: str
    total_reward: float
    external_reward: float
    intrinsic_reward: float
    semantic_gain: int
    prophecy_error: float
    error: bool
    flag_found: bool
    done: bool
    imagination_score: float
    imagination_candidate_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeMetric:
    condition: str
    seed: int
    episode: int
    success: bool
    steps_to_flag: int
    total_reward: float
    external_reward: float
    semantic_gain_total: int
    prophecy_error_mean: float
    repeat_count: int
    error_count: int
    knowledge_reuse_count: int
    unique_action_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SummaryMetric:
    condition: str
    seeds: int
    episodes: int
    success_rate: float
    steps_to_flag_mean: float
    total_reward_mean: float
    external_reward_mean: float
    semantic_gain_mean: float
    prophecy_error_mean: float
    repeat_count_mean: float
    error_count_mean: float
    knowledge_reuse_mean: float
    unique_action_count_mean: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def action_signature(candidate: ActionCandidate) -> str:
    bindings = ",".join(
        f"{kk.value}={repr(value)}"
        for kk, value in sorted(candidate.bindings.items(), key=lambda item: item[0].value)
        if kk != KK.CURRENT_POS
    )
    return f"{candidate.template}|{bindings}"


def step_metric(
    *,
    condition: str,
    seed: int,
    episode: int,
    result: StepResult,
) -> StepMetric:
    payload = result.to_dict()
    return StepMetric(
        condition=condition,
        seed=seed,
        episode=episode,
        step=result.step,
        action=result.action.name.value,
        template=result.action.template,
        action_signature=action_signature(result.action),
        total_reward=result.total_reward,
        external_reward=result.external_reward,
        intrinsic_reward=result.intrinsic_reward,
        semantic_gain=result.delta_k.semantic_information_gain(),
        prophecy_error=result.prophecy_error,
        error=result.error,
        flag_found=result.flag_found,
        done=result.done,
        imagination_score=payload["imagination_selected_score"],
        imagination_candidate_count=payload["imagination_candidate_count"],
    )


def episode_metric(
    *,
    condition: str,
    seed: int,
    episode: int,
    steps: list[StepMetric],
    knowledge_reuse_count: int,
    step_limit: int,
) -> EpisodeMetric:
    signatures = [step.action_signature for step in steps]
    unique_actions = set(signatures)
    success = any(step.flag_found for step in steps)
    return EpisodeMetric(
        condition=condition,
        seed=seed,
        episode=episode,
        success=success,
        steps_to_flag=len(steps) if success else step_limit,
        total_reward=sum(step.total_reward for step in steps),
        external_reward=sum(step.external_reward for step in steps),
        semantic_gain_total=sum(step.semantic_gain for step in steps),
        prophecy_error_mean=_safe_mean(step.prophecy_error for step in steps),
        repeat_count=len(signatures) - len(unique_actions),
        error_count=sum(1 for step in steps if step.error),
        knowledge_reuse_count=knowledge_reuse_count,
        unique_action_count=len(unique_actions),
    )


def summary_metric(condition: str, episodes: list[EpisodeMetric]) -> SummaryMetric:
    seeds = {episode.seed for episode in episodes}
    return SummaryMetric(
        condition=condition,
        seeds=len(seeds),
        episodes=len(episodes),
        success_rate=_safe_mean(1.0 if episode.success else 0.0 for episode in episodes),
        steps_to_flag_mean=_safe_mean(episode.steps_to_flag for episode in episodes),
        total_reward_mean=_safe_mean(episode.total_reward for episode in episodes),
        external_reward_mean=_safe_mean(episode.external_reward for episode in episodes),
        semantic_gain_mean=_safe_mean(episode.semantic_gain_total for episode in episodes),
        prophecy_error_mean=_safe_mean(episode.prophecy_error_mean for episode in episodes),
        repeat_count_mean=_safe_mean(episode.repeat_count for episode in episodes),
        error_count_mean=_safe_mean(episode.error_count for episode in episodes),
        knowledge_reuse_mean=_safe_mean(episode.knowledge_reuse_count for episode in episodes),
        unique_action_count_mean=_safe_mean(episode.unique_action_count for episode in episodes),
    )


def _safe_mean(values: Iterable[float | int]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return float(mean(materialized))
