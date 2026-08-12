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


@dataclass(frozen=True, slots=True)
class CurrentAASSRCoreView:
    """Explicit view of the algorithmic AASSR core inside an assembled runtime."""

    policy: object
    prophecy: object
    planner: object
    critic: object
    knowledge: object
    skills: object
    aseq: object
    goals: object


def bind_current_core_plugin_boundary(
    agent: object,
    plugin: CurrentRuntimePlugin,
) -> object:
    """Expose and diagnose the core/plugin split without changing behavior."""

    if getattr(agent, "current_core_plugin_boundary", False):
        return agent

    agent.aassr_core = CurrentAASSRCoreView(
        policy=agent.policy,
        prophecy=agent.prophecy,
        planner=agent.planner,
        critic=agent.critic,
        knowledge=agent.knowledge,
        skills=agent.skills,
        aseq=agent.aseq,
        goals=agent.goals,
    )
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
