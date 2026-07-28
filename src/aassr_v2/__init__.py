"""AASSR v2 research package."""

from .confidence import AdaptiveDepthController
from .evaluation import (
    EvaluatedTransition,
    ImaginationTransitionEvaluator,
    PlannedTransition,
    TransitionEvaluator,
)
from .gridworld import GridWorldEnv, GridWorldSpec
from .imagination_tree import (
    ImaginationConfig,
    ImaginationNode,
    ImaginationResult,
    ImaginationTree,
    RootActionEvaluation,
    StateDeltaScorer,
)
from .information_value import InformationValueBreakdown, InformationValueWeights
from .knowledge import KnowledgeDelta, KnowledgeEntry, KnowledgeStore
from .policy import PolicyMemory, ScoredAction, WeightedPolicy
from .prophecy import ProphecyStep
from .tabular_prophecy import TabularProphecy
from .types import Action, ActionVerb, Prediction, StateSnapshot, TransitionTrace

__all__ = [
    "Action",
    "ActionVerb",
    "AdaptiveDepthController",
    "EvaluatedTransition",
    "GridWorldEnv",
    "GridWorldSpec",
    "ImaginationConfig",
    "ImaginationNode",
    "ImaginationResult",
    "ImaginationTransitionEvaluator",
    "ImaginationTree",
    "InformationValueBreakdown",
    "InformationValueWeights",
    "KnowledgeDelta",
    "KnowledgeEntry",
    "KnowledgeStore",
    "PlannedTransition",
    "PolicyMemory",
    "Prediction",
    "ProphecyStep",
    "RootActionEvaluation",
    "ScoredAction",
    "StateDeltaScorer",
    "StateSnapshot",
    "TabularProphecy",
    "TransitionEvaluator",
    "TransitionTrace",
    "WeightedPolicy",
]

__version__ = "0.1.0"
