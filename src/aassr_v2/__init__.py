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
from .aassr_core import (
    CORE_MODULES,
    TRAINABLE_CORE_MODULES,
    AASSRCore,
    AASSRCoreConfig,
    AASSREpisodeRecord,
    CoreCallAudit,
    CoreDecision,
    CorePrimitiveStep,
)
from .adapters import (
    AuthorizedAssessmentPlugin,
    DryRunTransport,
    MinecraftControlPlugin,
)
from .agent import AgentStep, LearningAgent
from .autonomous_agent import (
    ActionDecision,
    AutonomousAgentConfig,
    AutonomousLearningAgent,
    ContextualPolicy,
    FrozenHoldout,
    ObservationMetrics,
)
from .autonomous_benchmarks import AutonomousStep, OpaqueDependencyWorld
from .autonomous_experiment import (
    load_autonomous_config,
    planned_autonomous_run_count,
    run_autonomous_experiment,
    validate_autonomous_config,
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
from .environment_plugin import (
    CoreEnvironmentSession,
    CoreObservationEncoder,
    EnvironmentPlugin,
    ObservableEnvironmentTransition,
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
from .paper_types import (
    AgentCheckpointParts,
    BudgetLedger,
    CausalEffectGraph,
    EffectProfile,
    ExperimentPhase,
    PaperManifest,
    StrategyRecord,
)
from .paper_protocol import (
    load_paper_config,
    planned_paper_run_count,
    validate_paper_config,
)
from .paper_runner import PaperArtifacts, run_paper_suite
from .grid_push_world import (
    GridPushSpec,
    GridPushWorld,
)
from .grid_push_plugin import GridPushEnvironmentPlugin

_LAZY_GRID_SOLVER_EXPORTS = {
    "ProceduralGridPushGenerator",
    "certify_grid_world",
    "solve_grid_world",
}


def __getattr__(name: str):
    if name in _LAZY_GRID_SOLVER_EXPORTS:
        from . import grid_push_solver

        return getattr(grid_push_solver, name)
    raise AttributeError(name)

__all__ = [name for name in globals() if not name.startswith("_")] + sorted(
    _LAZY_GRID_SOLVER_EXPORTS
)

__version__ = "0.3.0"
