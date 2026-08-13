"""AASSR v2 research package."""

import sys
import types
import warnings as _warnings


# ``resource`` is POSIX-only, but several benchmark modules are importable on
# Windows and use it solely for optional peak-RSS reporting. Install the same
# zero-valued compatibility module used by baseline_efficiency_portable before
# any package submodule can import the POSIX module directly.
if sys.platform == "win32" and "resource" not in sys.modules:
    resource = types.ModuleType("resource")
    resource.RUSAGE_SELF = 0

    class _Usage:
        ru_maxrss = 0.0

    def _getrusage(_: int) -> _Usage:
        return _Usage()

    resource.getrusage = _getrusage
    sys.modules["resource"] = resource


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
from .effect_prophecy import (
    EffectComposedProphecy,
    StateEffect,
    effect_context_key,
)
from .persistent_effect_prophecy import PersistentEffectComposedProphecy
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
from .integrated_agent import (
    ContextualSkillAwareProphecy,
    IntegratedAASSRConfig,
    IntegratedAASSRAgent,
    IntegratedActionDecision,
    IntegratedAgentStep,
    build_full_aassr_core as build_legacy_v040_full_aassr_core,
    build_pentest_aassr_core as build_legacy_v040_pentest_aassr_core,
)
from .current_manifest import (
    CURRENT_COMPONENTS,
    CURRENT_GENERATION_VERSION,
    LEGACY_COMPONENTS_ACTIVE,
)
from .current_agent import (
    CurrentStandalonePentestAASSRAgent,
    CurrentAgentConfig,
)
from .current_entrypoint import build_current_pentest_aassr_core

# Package-level current names point only at the standalone current runtime.
CurrentPentestAASSRAgent = CurrentStandalonePentestAASSRAgent
build_pentest_aassr_core = build_current_pentest_aassr_core

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
from .semantic_control import (
    SemanticContextualPolicy,
    SemanticSelfLoopASEQ,
    SemanticStateKey,
    SemanticStateKeyFn,
    raw_semantic_state_key,
)
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


# Keep the historical unqualified constructor source-compatible without eagerly
# advertising it as a current API. New/reproduction code should select either the
# canonical ``build_pentest_aassr_core`` or an explicit ``build_legacy_v040_*``
# name. Remove this shim only in a declared compatibility-breaking release.
_LEGACY_COMPAT_ALIASES = {
    "build_full_aassr_core": build_legacy_v040_full_aassr_core,
}


def __getattr__(name: str) -> object:
    if name in _LEGACY_COMPAT_ALIASES:
        _warnings.warn(
            f"aassr_v2.{name} is a legacy v0.4 compatibility alias; use "
            "build_legacy_v040_full_aassr_core explicitly",
            DeprecationWarning,
            stacklevel=2,
        )
        return _LEGACY_COMPAT_ALIASES[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [name for name in globals() if not name.startswith("_")]

__version__ = "0.4.0"
