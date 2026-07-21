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
    imagination_rollout_count: int = 0
    imagined_state_transition_count: int = 0
    imagined_step_count: int = 0
    imagined_trajectory_count: int = 0
    imagined_trajectory_depth: int = 0
    imagined_trajectory_depth_mean: float = 0.0
    imagined_trajectory_depth_max: int = 0
    future_candidate_generation_count: int = 0
    future_candidate_count: int = 0
    newly_unlocked_action_count: int = 0
    raw_future_candidate_count: int = 0
    unique_future_candidate_count: int = 0
    duplicate_future_candidate_count: int = 0
    future_candidate_dedup_ratio: float = 0.0
    raw_newly_unlocked_action_count: int = 0
    unique_newly_unlocked_action_count: int = 0
    unique_unlock_ratio: float = 0.0
    selected_action_has_future_dependency: bool = False
    selected_action_immediate_value: float = 0.0
    selected_action_future_value: float = 0.0
    selected_action_future_value_ratio: float = 0.0
    mean_transition_confidence: float = 0.0
    mean_selected_path_confidence: float = 0.0
    mean_placeholder_grounding_factor: float = 0.0
    mean_selected_effective_confidence: float = 0.0
    uncalibrated_selected_future_value: float = 0.0
    calibrated_selected_future_value: float = 0.0
    future_value_discount_ratio: float = 0.0
    placeholder_dependent_transition_count: int = 0
    concrete_transition_count: int = 0
    mixed_grounding_transition_count: int = 0
    setup_action_selected: bool = False
    predicted_placeholder_kv_count: int = 0
    placeholder_generated_candidate_count: int = 0
    placeholder_selected_candidate_count: int = 0
    placeholder_execution_attempt_count: int = 0
    predicted_kk_precision: float = 0.0
    predicted_kk_recall: float = 0.0
    predicted_kk_f1: float = 0.0
    predicted_error_accuracy: float = 0.0
    predicted_flag_accuracy: float = 0.0
    predicted_semantic_gain: float = 0.0
    actual_semantic_gain: float = 0.0
    kk_brier_score: float = 0.0
    error_brier_score: float = 0.0
    flag_brier_score: float = 0.0
    imagined_next_action_match: float = 0.0
    imagined_next_action_what_match: float = 0.0
    imagined_next_action_where_match: float = 0.0
    imagined_next_action_template_match: float = 0.0
    imagined_next_action_match_observed: int = 0

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
    imagined_state_transition_total: int = 0
    newly_unlocked_action_total: int = 0
    unique_newly_unlocked_action_total: int = 0
    duplicate_future_candidate_total: int = 0
    future_candidate_dedup_ratio_mean: float = 0.0
    unique_unlock_ratio_mean: float = 0.0
    setup_action_selected_count: int = 0
    future_dependency_selection_rate: float = 0.0
    imagined_trajectory_depth_mean: float = 0.0
    imagined_trajectory_depth_max: int = 0
    predicted_kk_precision_mean: float = 0.0
    predicted_kk_recall_mean: float = 0.0
    predicted_kk_f1_mean: float = 0.0
    imagined_action_execution_match_rate: float = 0.0
    imagined_action_what_match_rate: float = 0.0
    imagined_action_where_match_rate: float = 0.0
    imagined_action_template_match_rate: float = 0.0
    mean_transition_confidence: float = 0.0
    mean_selected_path_confidence: float = 0.0
    mean_placeholder_grounding_factor: float = 0.0
    mean_selected_effective_confidence: float = 0.0
    uncalibrated_selected_future_value_mean: float = 0.0
    calibrated_selected_future_value_mean: float = 0.0
    future_value_discount_ratio_mean: float = 0.0
    placeholder_dependent_transition_total: int = 0
    concrete_transition_total: int = 0
    mixed_grounding_transition_total: int = 0
    placeholder_generated_candidate_total: int = 0
    placeholder_execution_attempt_total: int = 0

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
    imagined_state_transition_mean: float = 0.0
    newly_unlocked_action_mean: float = 0.0
    unique_newly_unlocked_action_mean: float = 0.0
    duplicate_future_candidate_mean: float = 0.0
    future_candidate_dedup_ratio_mean: float = 0.0
    unique_unlock_ratio_mean: float = 0.0
    setup_action_selected_mean: float = 0.0
    future_dependency_selection_rate_mean: float = 0.0
    imagined_trajectory_depth_mean: float = 0.0
    imagined_trajectory_depth_max_mean: float = 0.0
    predicted_kk_precision_mean: float = 0.0
    predicted_kk_recall_mean: float = 0.0
    predicted_kk_f1_mean: float = 0.0
    imagined_action_execution_match_rate_mean: float = 0.0
    mean_transition_confidence_mean: float = 0.0
    mean_selected_path_confidence_mean: float = 0.0
    mean_placeholder_grounding_factor_mean: float = 0.0
    mean_selected_effective_confidence_mean: float = 0.0
    uncalibrated_selected_future_value_mean: float = 0.0
    calibrated_selected_future_value_mean: float = 0.0
    future_value_discount_ratio_mean: float = 0.0
    placeholder_dependent_transition_mean: float = 0.0
    concrete_transition_mean: float = 0.0
    mixed_grounding_transition_mean: float = 0.0
    placeholder_generated_candidate_mean: float = 0.0

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
        imagination_rollout_count=payload["imagination_rollout_count"],
        imagined_state_transition_count=payload["imagined_state_transition_count"],
        imagined_step_count=payload["imagined_step_count"],
        imagined_trajectory_count=payload["imagined_trajectory_count"],
        imagined_trajectory_depth=payload["imagined_trajectory_depth"],
        imagined_trajectory_depth_mean=payload["imagined_trajectory_depth_mean"],
        imagined_trajectory_depth_max=payload["imagined_trajectory_depth_max"],
        future_candidate_generation_count=payload["future_candidate_generation_count"],
        future_candidate_count=payload["future_candidate_count"],
        newly_unlocked_action_count=payload["newly_unlocked_action_count"],
        raw_future_candidate_count=payload["raw_future_candidate_count"],
        unique_future_candidate_count=payload["unique_future_candidate_count"],
        duplicate_future_candidate_count=payload["duplicate_future_candidate_count"],
        future_candidate_dedup_ratio=payload["future_candidate_dedup_ratio"],
        raw_newly_unlocked_action_count=payload["raw_newly_unlocked_action_count"],
        unique_newly_unlocked_action_count=payload["unique_newly_unlocked_action_count"],
        unique_unlock_ratio=payload["unique_unlock_ratio"],
        selected_action_has_future_dependency=payload["selected_action_has_future_dependency"],
        selected_action_immediate_value=payload["selected_action_immediate_value"],
        selected_action_future_value=payload["selected_action_future_value"],
        selected_action_future_value_ratio=payload["selected_action_future_value_ratio"],
        mean_transition_confidence=payload["mean_transition_confidence"],
        mean_selected_path_confidence=payload["mean_selected_path_confidence"],
        mean_placeholder_grounding_factor=payload["mean_placeholder_grounding_factor"],
        mean_selected_effective_confidence=payload["mean_selected_effective_confidence"],
        uncalibrated_selected_future_value=payload["uncalibrated_selected_future_value"],
        calibrated_selected_future_value=payload["calibrated_selected_future_value"],
        future_value_discount_ratio=payload["future_value_discount_ratio"],
        placeholder_dependent_transition_count=payload["placeholder_dependent_transition_count"],
        concrete_transition_count=payload["concrete_transition_count"],
        mixed_grounding_transition_count=payload["mixed_grounding_transition_count"],
        setup_action_selected=payload["setup_action_selected"],
        predicted_placeholder_kv_count=payload["predicted_placeholder_kv_count"],
        placeholder_generated_candidate_count=payload["placeholder_generated_candidate_count"],
        placeholder_selected_candidate_count=payload["placeholder_selected_candidate_count"],
        placeholder_execution_attempt_count=payload["placeholder_execution_attempt_count"],
        predicted_kk_precision=payload["predicted_kk_precision"],
        predicted_kk_recall=payload["predicted_kk_recall"],
        predicted_kk_f1=payload["predicted_kk_f1"],
        predicted_error_accuracy=payload["predicted_error_accuracy"],
        predicted_flag_accuracy=payload["predicted_flag_accuracy"],
        predicted_semantic_gain=payload["predicted_semantic_gain"],
        actual_semantic_gain=payload["actual_semantic_gain"],
        kk_brier_score=payload["kk_brier_score"],
        error_brier_score=payload["error_brier_score"],
        flag_brier_score=payload["flag_brier_score"],
        imagined_next_action_match=_optional_bool_metric(payload["imagined_next_action_match"]),
        imagined_next_action_what_match=_optional_bool_metric(payload["imagined_next_action_what_match"]),
        imagined_next_action_where_match=_optional_bool_metric(payload["imagined_next_action_where_match"]),
        imagined_next_action_template_match=_optional_bool_metric(payload["imagined_next_action_template_match"]),
        imagined_next_action_match_observed=int(payload["imagined_next_action_match"] is not None),
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
        imagined_state_transition_total=sum(step.imagined_state_transition_count for step in steps),
        newly_unlocked_action_total=sum(step.newly_unlocked_action_count for step in steps),
        unique_newly_unlocked_action_total=sum(step.unique_newly_unlocked_action_count for step in steps),
        duplicate_future_candidate_total=sum(step.duplicate_future_candidate_count for step in steps),
        future_candidate_dedup_ratio_mean=_safe_mean(step.future_candidate_dedup_ratio for step in steps if step.raw_future_candidate_count > 0),
        unique_unlock_ratio_mean=_safe_mean(step.unique_unlock_ratio for step in steps if step.raw_newly_unlocked_action_count > 0),
        setup_action_selected_count=sum(1 for step in steps if step.setup_action_selected),
        future_dependency_selection_rate=_safe_mean(
            1.0 if step.selected_action_has_future_dependency else 0.0
            for step in steps
            if step.imagined_trajectory_count > 0
        ),
        imagined_trajectory_depth_mean=_safe_mean(
            step.imagined_trajectory_depth for step in steps if step.imagined_trajectory_depth > 0
        ),
        imagined_trajectory_depth_max=max((step.imagined_trajectory_depth for step in steps), default=0),
        predicted_kk_precision_mean=_safe_mean(step.predicted_kk_precision for step in steps if step.prophecy_error >= 0.0),
        predicted_kk_recall_mean=_safe_mean(step.predicted_kk_recall for step in steps if step.prophecy_error >= 0.0),
        predicted_kk_f1_mean=_safe_mean(step.predicted_kk_f1 for step in steps if step.prophecy_error >= 0.0),
        imagined_action_execution_match_rate=_safe_mean(
            step.imagined_next_action_match for step in steps if step.imagined_next_action_match_observed
        ),
        imagined_action_what_match_rate=_safe_mean(
            step.imagined_next_action_what_match for step in steps if step.imagined_next_action_match_observed
        ),
        imagined_action_where_match_rate=_safe_mean(
            step.imagined_next_action_where_match for step in steps if step.imagined_next_action_match_observed
        ),
        imagined_action_template_match_rate=_safe_mean(
            step.imagined_next_action_template_match for step in steps if step.imagined_next_action_match_observed
        ),
        mean_transition_confidence=_safe_mean(step.mean_transition_confidence for step in steps if step.imagined_trajectory_count > 0),
        mean_selected_path_confidence=_safe_mean(step.mean_selected_path_confidence for step in steps if step.imagined_trajectory_count > 0),
        mean_placeholder_grounding_factor=_safe_mean(step.mean_placeholder_grounding_factor for step in steps if step.mean_placeholder_grounding_factor > 0),
        mean_selected_effective_confidence=_safe_mean(step.mean_selected_effective_confidence for step in steps if step.imagined_trajectory_count > 0),
        uncalibrated_selected_future_value_mean=_safe_mean(step.uncalibrated_selected_future_value for step in steps if step.imagined_trajectory_count > 0),
        calibrated_selected_future_value_mean=_safe_mean(step.calibrated_selected_future_value for step in steps if step.imagined_trajectory_count > 0),
        future_value_discount_ratio_mean=_safe_mean(step.future_value_discount_ratio for step in steps if step.uncalibrated_selected_future_value != 0),
        placeholder_dependent_transition_total=sum(step.placeholder_dependent_transition_count for step in steps),
        concrete_transition_total=sum(step.concrete_transition_count for step in steps),
        mixed_grounding_transition_total=sum(step.mixed_grounding_transition_count for step in steps),
        placeholder_generated_candidate_total=sum(step.placeholder_generated_candidate_count for step in steps),
        placeholder_execution_attempt_total=sum(step.placeholder_execution_attempt_count for step in steps),
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
        imagined_state_transition_mean=_safe_mean(episode.imagined_state_transition_total for episode in episodes),
        newly_unlocked_action_mean=_safe_mean(episode.newly_unlocked_action_total for episode in episodes),
        unique_newly_unlocked_action_mean=_safe_mean(episode.unique_newly_unlocked_action_total for episode in episodes),
        duplicate_future_candidate_mean=_safe_mean(episode.duplicate_future_candidate_total for episode in episodes),
        future_candidate_dedup_ratio_mean=_safe_mean(episode.future_candidate_dedup_ratio_mean for episode in episodes),
        unique_unlock_ratio_mean=_safe_mean(episode.unique_unlock_ratio_mean for episode in episodes),
        setup_action_selected_mean=_safe_mean(episode.setup_action_selected_count for episode in episodes),
        future_dependency_selection_rate_mean=_safe_mean(episode.future_dependency_selection_rate for episode in episodes),
        imagined_trajectory_depth_mean=_safe_mean(episode.imagined_trajectory_depth_mean for episode in episodes),
        imagined_trajectory_depth_max_mean=_safe_mean(episode.imagined_trajectory_depth_max for episode in episodes),
        predicted_kk_precision_mean=_safe_mean(episode.predicted_kk_precision_mean for episode in episodes),
        predicted_kk_recall_mean=_safe_mean(episode.predicted_kk_recall_mean for episode in episodes),
        predicted_kk_f1_mean=_safe_mean(episode.predicted_kk_f1_mean for episode in episodes),
        imagined_action_execution_match_rate_mean=_safe_mean(episode.imagined_action_execution_match_rate for episode in episodes),
        mean_transition_confidence_mean=_safe_mean(episode.mean_transition_confidence for episode in episodes),
        mean_selected_path_confidence_mean=_safe_mean(episode.mean_selected_path_confidence for episode in episodes),
        mean_placeholder_grounding_factor_mean=_safe_mean(episode.mean_placeholder_grounding_factor for episode in episodes),
        mean_selected_effective_confidence_mean=_safe_mean(episode.mean_selected_effective_confidence for episode in episodes),
        uncalibrated_selected_future_value_mean=_safe_mean(episode.uncalibrated_selected_future_value_mean for episode in episodes),
        calibrated_selected_future_value_mean=_safe_mean(episode.calibrated_selected_future_value_mean for episode in episodes),
        future_value_discount_ratio_mean=_safe_mean(episode.future_value_discount_ratio_mean for episode in episodes),
        placeholder_dependent_transition_mean=_safe_mean(episode.placeholder_dependent_transition_total for episode in episodes),
        concrete_transition_mean=_safe_mean(episode.concrete_transition_total for episode in episodes),
        mixed_grounding_transition_mean=_safe_mean(episode.mixed_grounding_transition_total for episode in episodes),
        placeholder_generated_candidate_mean=_safe_mean(episode.placeholder_generated_candidate_total for episode in episodes),
    )


def _safe_mean(values: Iterable[float | int]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return float(mean(materialized))


def _optional_bool_metric(value: Any) -> float:
    if value is None:
        return 0.0
    return 1.0 if value else 0.0
