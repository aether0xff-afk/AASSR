from __future__ import annotations

from dataclasses import dataclass
from types import MethodType, SimpleNamespace
from typing import Any, Callable, Mapping

from .current_core_manifest import CURRENT_CORE_COMPONENTS, CURRENT_CORE_VERSION


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
    install_contract: Callable[[], None]
    install_world_model: Callable[..., object]


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
        }
        result["core_plugin_boundary"] = True
        return result

    agent.diagnostics = MethodType(diagnostics, agent)
    return agent
