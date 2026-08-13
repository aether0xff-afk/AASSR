from __future__ import annotations

from dataclasses import dataclass
from types import MethodType, SimpleNamespace
from typing import Any, Callable, Hashable, Mapping, Protocol, Sequence

from .current_core_manifest import CURRENT_CORE_COMPONENTS, CURRENT_CORE_VERSION
from .types import Action, Prediction, StateSnapshot


class StateCodecProtocol(Protocol):
    """Domain codec consumed by a concrete world-model implementation."""

    def encode(self, state: StateSnapshot) -> Sequence[float]: ...


@dataclass(frozen=True, slots=True)
class CurrentRepresentationBinding:
    """Immutable, runtime-scoped environment representation contract.

    The binding contains behavior rather than changing imported module globals.
    Two runtimes can therefore select different representations in one process
    without either installation changing the other runtime.
    """

    binding_id: str
    observation_contract: str
    state_size: int
    action_feature_size: int
    state_codec_factory: Callable[[], StateCodecProtocol]
    validate_observation: Callable[[StateSnapshot], None]
    state_vector: Callable[[StateSnapshot], tuple[float, ...]]
    state_key: Callable[[StateSnapshot], Hashable]
    semantic_state_identity: Callable[[StateSnapshot], Hashable]
    action_structure: Callable[[StateSnapshot, Action], tuple[float, ...]]
    decode_state: Callable[..., StateSnapshot]
    prediction_score: Callable[[Sequence[Prediction], StateSnapshot], float]
    state_descriptor: Callable[[StateSnapshot], tuple[float, ...]]
    descriptor_size: int
    diagnostics: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class CurrentRuntimePlugin:
    """Environment/domain binding for the current AASSR core.

    Plugins own observation/action/state encoding and environment-specific outcome
    semantics. They may install a concrete Prophecy head/codec, but the AASSR core
    still owns Policy, Prophecy learning/planning roles, Imagination, Critic,
    Knowledge, Skills and ASEQ responsibilities.
    """

    plugin_id: str
    version: str
    components: Mapping[str, str]
    representation: CurrentRepresentationBinding
    install_world_model: Callable[..., object]
    environment_factory: Callable[..., object]


class CurrentAASSRCoreView:
    """Live, domain-independent view of the algorithmic core.

    The assembled agent replaces some runtime objects (notably episode Knowledge)
    during normal operation. Properties therefore resolve through the agent every
    time instead of capturing stale construction-time references.
    """

    __slots__ = ("_agent",)

    def __init__(self, agent: object) -> None:
        self._agent = agent

    @property
    def policy(self) -> object:
        return self._agent.policy

    @property
    def prophecy(self) -> object:
        return self._agent.prophecy

    @property
    def planner(self) -> object:
        return self._agent.planner

    @property
    def critic(self) -> object:
        return self._agent.critic

    @property
    def knowledge(self) -> object:
        return self._agent.knowledge

    @property
    def skills(self) -> object:
        return self._agent.skills

    @property
    def aseq(self) -> object:
        return self._agent.aseq

    @property
    def goals(self) -> object:
        return self._agent.goals


def bind_current_core_plugin_boundary(
    agent: object,
    plugin: CurrentRuntimePlugin,
) -> object:
    """Expose and diagnose the core/plugin split without changing behavior."""

    if getattr(agent, "current_core_plugin_boundary", False):
        return agent

    agent.aassr_core = CurrentAASSRCoreView(agent)
    agent.runtime_plugin = SimpleNamespace(
        plugin_id=str(plugin.plugin_id),
        version=str(plugin.version),
        components=dict(plugin.components),
        representation=plugin.representation,
        environment_factory=plugin.environment_factory,
    )
    agent.current_core_version = CURRENT_CORE_VERSION
    agent.current_core_components = dict(CURRENT_CORE_COMPONENTS)
    agent.current_plugin_id = str(plugin.plugin_id)
    agent.current_plugin_version = str(plugin.version)
    agent.current_plugin_components = dict(plugin.components)
    agent.current_core_plugin_boundary = True

    original_diagnostics = agent.diagnostics

    def diagnostics(self: object) -> dict[str, Any]:
        result = dict(original_diagnostics())
        result["aassr_core"] = {
            "version": self.current_core_version,
            "components": dict(self.current_core_components),
        }
        result["runtime_plugin"] = {
            "id": self.current_plugin_id,
            "version": self.current_plugin_version,
            "components": dict(self.current_plugin_components),
            "representation": {
                "id": plugin.representation.binding_id,
                **dict(plugin.representation.diagnostics),
            },
            "environment_factory": (
                "plugin-provided" if plugin.environment_factory is not None else "unbound"
            ),
        }
        result["core_plugin_boundary"] = True
        return result

    agent.diagnostics = MethodType(diagnostics, agent)
    return agent
