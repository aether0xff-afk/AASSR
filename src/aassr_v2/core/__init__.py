"""Domain-independent AASSR Core.

The Core learns task meaning. Environment plugins are intentionally restricted
to command syntax, public data types, and real I/O.
"""

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
from .representation import (
    CoreExperienceMemory,
    CoreRepresentationConfig,
    PluginEnvironmentAdapter,
    SchemaDrivenRepresentation,
    SchemaDrivenStateCodec,
)
from .manifest import (
    CORE_COMPONENTS,
    PLUGIN_ALLOWED_AUTHORITIES,
    PLUGIN_FORBIDDEN_AUTHORITIES,
)
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
    "CoreRepresentationConfig",
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
