"""AASSR v2 research package."""

from .ablations import (
    AblationConfig,
    AblationSummary,
    EpisodeMetrics,
    imagination_ablation_matrix,
    representation_ablation_matrix,
    summarize,
)
from .action_plugins import (
    ActionPlugin,
    ActionRegistry,
    ActionSchema,
    ParameterLesson,
    ParameterSpec,
    PluginOutcome,
    SlotCandidateResolver,
    parameter_lessons,
)
from .adapters import (
    AuthorizedAssessmentPlugin,
    DryRunTransport,
    MinecraftControlPlugin,
)
from .confidence import AdaptiveDepthController
from .counterexamples import (
    LearnableVsRandomWorld,
    LongDependencyWorld,
    NoisyInformationWrapper,
    opaque_name_map,
    permuted_positions,
)
from .curriculum_engine import (
    AcademyLesson,
    AcademyStage,
    CurriculumTeacher,
    DEFAULT_ACADEMY,
)
from .evaluation import (
    EvaluatedTransition,
    ImaginationTransitionEvaluator,
    PlannedTransition,
    TransitionEvaluator,
)
from .feature_memory import (
    FeatureRecord,
    HashEmbeddingProvider,
    OnlineFeatureMemory,
    SelectiveEmbeddingRouter,
)
from .goals import (
    Goal,
    GoalEvaluation,
    GoalGenerator,
    GoalKind,
    GoalSet,
    GoalStateScorer,
    choose_goal,
)
from .gridworld import GridWorldEnv, GridWorldSpec
from .gru_prophecy import (
    GRUMemory,
    GRUTrainingStats,
    OnlineGRUProphecy,
)
from .imagination_tree import (
    ImaginationConfig,
    ImaginationNode,
    ImaginationResult,
    ImaginationTree,
    RootActionEvaluation,
    StateDeltaScorer,
)
from .information_value import (
    InformationValueBreakdown,
    InformationValueWeights,
)
from .knowledge import (
    KnowledgeDelta,
    KnowledgeEntry,
    KnowledgeStore,
)
from .learning import (
    ActionUnlockValueEstimator,
    AdvancedEvaluation,
    AdvancedTransitionEvaluator,
    CreditedTrace,
    DelayedCreditAssigner,
    InformationValuePredictor,
    LearningEffectReport,
)
from .policy import PolicyMemory, ScoredAction, WeightedPolicy
from .prophecy import ProphecyStep
from .replay import (
    PredictionValidator,
    ReplayBuffer,
    ReplayTransition,
    ValidationScore,
)
from .sandbox import SandboxActionPlugin, SandboxEnv, SandboxSpec
from .serialization import JsonlLedgerWriter
from .skills import (
    SKILL_VERB,
    Skill,
    SkillAwareProphecy,
    SkillExecutionResult,
    SkillExecutor,
    SkillLibrary,
)
from .tabular_prophecy import TabularProphecy
from .types import (
    Action,
    ActionVerb,
    Prediction,
    StateSnapshot,
    TransitionTrace,
    action_verb_name,
)

__all__ = [name for name in globals() if not name.startswith("_")]

__version__ = "0.2.0"
