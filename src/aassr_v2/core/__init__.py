"""Domain-independent AASSR Core.

The Core learns task meaning. Environment plugins are intentionally restricted
to command syntax, public data types, and real I/O.
"""

from .manifest import (
    CORE_COMPONENTS,
    PLUGIN_ALLOWED_AUTHORITIES,
    PLUGIN_FORBIDDEN_AUTHORITIES,
)
from .plugin_contract import (
    ActionCommand,
    ActionParameter,
    ActionSpec,
    MinimalRuntimePlugin,
    ObservationField,
    PluginObservation,
    PluginSchema,
    PluginStepResult,
    TemporalKind,
    ValueKind,
    validate_minimal_plugin,
)
from .public_memory import CorePublicKnowledge, MemoryBackedRepresentation
from .representation import (
    CoreExperienceMemory,
    CoreRepresentationConfig,
    PluginEnvironmentAdapter,
    SchemaDrivenStateCodec,
)
from .schema_representation import SchemaDrivenRepresentation
from .runtime import (
    AASSRCoreRuntime,
    CORE_VERSION,
    PLUGIN_CONTRACT_VERSION,
    CoreRuntimeConfig,
    build_aassr_core,
)

__all__ = (
    "ActionCommand",
    "ActionParameter",
    "ActionSpec",
    "MinimalRuntimePlugin",
    "ObservationField",
    "PluginObservation",
    "PluginSchema",
    "PluginStepResult",
    "TemporalKind",
    "ValueKind",
    "validate_minimal_plugin",
    "CoreExperienceMemory",
    "CorePublicKnowledge",
    "CoreRepresentationConfig",
    "MemoryBackedRepresentation",
    "PluginEnvironmentAdapter",
    "SchemaDrivenRepresentation",
    "SchemaDrivenStateCodec",
    "AASSRCoreRuntime",
    "CORE_VERSION",
    "PLUGIN_CONTRACT_VERSION",
    "CoreRuntimeConfig",
    "build_aassr_core",
    "CORE_COMPONENTS",
    "PLUGIN_ALLOWED_AUTHORITIES",
    "PLUGIN_FORBIDDEN_AUTHORITIES",
)
