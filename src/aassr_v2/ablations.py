from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Literal

Aggregation = Literal[
    "max",
    "mean",
    "top_mean",
    "risk_adjusted",
]
FeatureSource = Literal[
    "none",
    "experience",
    "embedding",
    "hybrid",
]


@dataclass(frozen=True, slots=True)
class AblationConfig:
    name: str
    branching_factor: int = 2
    maximum_depth: int = 2
    adaptive_depth: bool = True
    aggregation: Aggregation = "max"
    use_goal_value: bool = True
    use_skills: bool = True
    use_information_value: bool = True
    use_clustering: bool = True
    use_hierarchical_slot_selection: bool = True
    use_online_reclustering: bool = True
    use_action_slot_context: bool = True
    feature_source: FeatureSource = "experience"


def imagination_ablation_matrix() -> tuple[AblationConfig, ...]:
    configs = [
        AblationConfig(
            "I0_no_imagination",
            branching_factor=1,
            maximum_depth=1,
            adaptive_depth=False,
            use_goal_value=False,
            use_skills=False,
        )
    ]
    for branch, depth, aggregation in product(
        (1, 2, 3),
        (1, 2, 3),
        ("max", "mean", "risk_adjusted"),
    ):
        configs.append(
            AblationConfig(
                f"I_b{branch}_d{depth}_{aggregation}",
                branching_factor=branch,
                maximum_depth=depth,
                adaptive_depth=False,
                aggregation=aggregation,
            )
        )
    configs.append(
        AblationConfig(
            "I_adaptive",
            branching_factor=2,
            maximum_depth=5,
            adaptive_depth=True,
        )
    )
    return tuple(configs)


def representation_ablation_matrix() -> tuple[AblationConfig, ...]:
    return (
        AblationConfig(
            "S0_direct",
            use_clustering=False,
            use_hierarchical_slot_selection=False,
            use_online_reclustering=False,
            use_action_slot_context=False,
            feature_source="none",
        ),
        AblationConfig(
            "S1_features",
            use_clustering=False,
            use_hierarchical_slot_selection=False,
            feature_source="experience",
        ),
        AblationConfig(
            "S2_clusters",
            use_hierarchical_slot_selection=False,
            feature_source="experience",
        ),
        AblationConfig(
            "S3_two_stage_static",
            use_online_reclustering=False,
            use_action_slot_context=False,
            feature_source="experience",
        ),
        AblationConfig(
            "S4_two_stage_online",
            use_action_slot_context=False,
            feature_source="experience",
        ),
        AblationConfig(
            "S5_contextual",
            feature_source="experience",
        ),
        AblationConfig(
            "E2_embedding",
            feature_source="embedding",
        ),
        AblationConfig(
            "E3_hybrid",
            feature_source="hybrid",
        ),
    )


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    success: bool
    steps: int
    imagined_nodes: int = 0
    imagination_depth: int = 0
    errors: int = 0
    repeats: int = 0
    prediction_score: float = 0.0


@dataclass(frozen=True, slots=True)
class AblationSummary:
    name: str
    episodes: int
    success_rate: float
    mean_steps: float
    mean_imagined_nodes: float
    mean_depth: float
    error_rate: float
    repeat_rate: float
    mean_prediction_score: float


def summarize(
    name: str,
    episodes: Iterable[EpisodeMetrics],
) -> AblationSummary:
    items = tuple(episodes)
    if not items:
        raise ValueError("at least one episode is required")
    count = len(items)
    total_steps = max(
        1,
        sum(item.steps for item in items),
    )
    return AblationSummary(
        name,
        count,
        sum(item.success for item in items) / count,
        sum(item.steps for item in items) / count,
        sum(item.imagined_nodes for item in items) / count,
        sum(item.imagination_depth for item in items) / count,
        sum(item.errors for item in items) / total_steps,
        sum(item.repeats for item in items) / total_steps,
        sum(item.prediction_score for item in items) / count,
    )
