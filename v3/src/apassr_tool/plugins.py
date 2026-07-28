from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Protocol

from .actions import (
    ActionCandidate,
    generate_candidates,
    generate_candidates_for_policy,
    generate_file_surface_candidates,
    generate_file_surface_candidates_for_policy,
    generate_input_mutation_candidates,
    generate_input_mutation_candidates_for_policy,
    generate_json_api_candidates,
    generate_json_api_candidates_for_policy,
)
from .knowledge import KK, KnowledgeStore, seed_knowledge
from .parser import parse_tool_result
from .policy import PolicyView
from .reward import JuiceShopChallengeObserver, RewardObserver
from .tools import ToolResult


@dataclass(frozen=True)
class PluginMetadata:
    name: str
    description: str
    domain: str
    reward_observers: tuple[str, ...] = ("none",)
    capabilities: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    adapter_modules: tuple[str, ...] = ()
    safety_notes: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


class TargetPlugin(Protocol):
    name: str
    metadata: PluginMetadata

    def seed(self, base_url: str) -> KnowledgeStore:
        ...

    def candidates(self, store: KnowledgeStore) -> list[ActionCandidate]:
        ...

    def candidates_for_policy(self, store: KnowledgeStore, policy: PolicyView) -> list[ActionCandidate]:
        ...

    def parse(self, result: ToolResult) -> list[tuple[KK, str]]:
        ...

    def reward_observer(self, name: str, base_url: str) -> RewardObserver | None:
        ...


@dataclass(frozen=True)
class WebPentestPlugin:
    name: str = "web"
    metadata: PluginMetadata = field(
        default_factory=lambda: PluginMetadata(
            name="web",
            description="Local web pentesting adapter with primitive HTTP/nmap/whatweb actions.",
            domain="local_web",
            reward_observers=("none", "juice-shop"),
            capabilities=(
                "http_get",
                "http_head",
                "http_options",
                "form_post",
                "query_probe",
                "combo_post",
                "cookie_session",
                "shallow_nmap",
                "passive_fingerprint",
            ),
            dependencies=("actions", "tools", "parser", "reward"),
            adapter_modules=("actions.py", "tools.py", "parser.py", "reward.py"),
            safety_notes=(
                "loopback targets only by default",
                "fixed tool templates only",
                "no human-authored challenge stage list",
            ),
            limitations=(
                "does not model SPA storage",
                "does not execute browser DOM actions",
                "does not generate JSON API bodies beyond observed key/value pools",
            ),
        )
    )

    def seed(self, base_url: str) -> KnowledgeStore:
        return seed_knowledge(base_url)

    def candidates(self, store: KnowledgeStore) -> list[ActionCandidate]:
        return generate_candidates(store)

    def candidates_for_policy(self, store: KnowledgeStore, policy: PolicyView) -> list[ActionCandidate]:
        return generate_candidates_for_policy(store, policy)

    def parse(self, result: ToolResult) -> list[tuple[KK, str]]:
        return parse_tool_result(result)

    def reward_observer(self, name: str, base_url: str) -> RewardObserver | None:
        if name not in self.metadata.reward_observers:
            raise ValueError(f"unsupported reward observer for web plugin: {name}")
        if name == "none":
            return None
        if name == "juice-shop":
            return JuiceShopChallengeObserver(base_url)
        raise ValueError(f"unsupported reward observer for web plugin: {name}")


@dataclass(frozen=True)
class JuiceShopPlugin(WebPentestPlugin):
    name: str = "juice-shop"
    metadata: PluginMetadata = field(
        default_factory=lambda: PluginMetadata(
            name="juice-shop",
            description="Juice Shop target adapter built on the generic web plugin surface.",
            domain="local_web_spa_lab",
            reward_observers=("none", "juice-shop"),
            capabilities=(
                "spa_asset_discovery",
                "rest_api_discovery",
                "json_api_actions",
                "http_get",
                "http_head",
                "http_options",
                "form_post",
                "query_probe",
                "combo_post",
                "cookie_session",
                "scoreboard_reward_observer",
            ),
            dependencies=(
                "web",
                "json-api-actions",
                "future:browser_dom_actions",
                "future:storage_state_observer",
                "file-surface-actions",
            ),
            adapter_modules=("actions.py", "parser.py", "reward.py"),
            safety_notes=(
                "local Juice Shop only",
                "challenge API may be used only as post-action reward/progress feedback",
                "no challenge order, known credentials, flags, or writeup-derived action ranking",
            ),
            limitations=(
                "currently reuses generic web actions",
                "does not yet add browser click/fill primitives",
                "does not yet inspect localStorage/sessionStorage",
            ),
        )
    )

    def candidates(self, store: KnowledgeStore) -> list[ActionCandidate]:
        return [*generate_candidates(store), *generate_json_api_candidates(store)]

    def candidates_for_policy(self, store: KnowledgeStore, policy: PolicyView) -> list[ActionCandidate]:
        return [*generate_candidates_for_policy(store, policy), *generate_json_api_candidates_for_policy(store, policy)]


@dataclass(frozen=True)
class JsonApiActionsPlugin(WebPentestPlugin):
    name: str = "json-api-actions"
    metadata: PluginMetadata = field(
        default_factory=lambda: PluginMetadata(
            name="json-api-actions",
            description="Generic JSON API action adapter for observed endpoints, fields, and values.",
            domain="local_web_json_api",
            reward_observers=("none", "juice-shop"),
            capabilities=(
                "json_post",
                "json_put",
                "json_patch",
                "json_delete",
                "observed_field_binding",
                "observed_value_binding",
            ),
            dependencies=("web",),
            adapter_modules=("actions.py", "tools.py", "parser.py"),
            safety_notes=(
                "uses only observed endpoint, field, and value pools",
                "does not inject endpoint-specific payloads",
                "does not rank actions from writeups",
            ),
            limitations=(
                "JSON values are currently string-like observed scalar values",
                "does not infer nested schemas",
                "does not yet use browser storage tokens",
            ),
        )
    )

    def candidates(self, store: KnowledgeStore) -> list[ActionCandidate]:
        return [*generate_candidates(store), *generate_json_api_candidates(store)]

    def candidates_for_policy(self, store: KnowledgeStore, policy: PolicyView) -> list[ActionCandidate]:
        return [*generate_candidates_for_policy(store, policy), *generate_json_api_candidates_for_policy(store, policy)]


@dataclass(frozen=True)
class InputMutationPlugin(WebPentestPlugin):
    name: str = "input-mutation-actions"
    metadata: PluginMetadata = field(
        default_factory=lambda: PluginMetadata(
            name="input-mutation-actions",
            description="Generic input mutation adapter for query, form, and JSON probes.",
            domain="local_web_fuzzing",
            reward_observers=("none", "juice-shop"),
            capabilities=(
                "numeric_boundary_values",
                "boolean_null_values",
                "string_edge_values",
                "query_mutation",
                "form_mutation",
                "json_mutation",
            ),
            dependencies=("web", "json-api-actions"),
            adapter_modules=("actions.py", "tools.py"),
            safety_notes=(
                "uses generic mutation values only",
                "uses observed endpoint and parameter pools",
                "does not include challenge-specific payload recipes",
            ),
            limitations=(
                "mutation set is intentionally small",
                "does not adapt payload grammar from natural language challenge text",
            ),
        )
    )

    def candidates(self, store: KnowledgeStore) -> list[ActionCandidate]:
        return [
            *generate_candidates(store),
            *generate_json_api_candidates(store),
            *generate_input_mutation_candidates(store),
        ]

    def candidates_for_policy(self, store: KnowledgeStore, policy: PolicyView) -> list[ActionCandidate]:
        return [
            *generate_candidates_for_policy(store, policy),
            *generate_json_api_candidates_for_policy(store, policy),
            *generate_input_mutation_candidates_for_policy(store, policy),
        ]


@dataclass(frozen=True)
class FileSurfacePlugin(WebPentestPlugin):
    name: str = "file-surface-actions"
    metadata: PluginMetadata = field(
        default_factory=lambda: PluginMetadata(
            name="file-surface-actions",
            description="File and static asset surface adapter based on observed paths.",
            domain="local_web_files",
            reward_observers=("none", "juice-shop"),
            capabilities=(
                "observed_file_get",
                "backup_suffix_probe",
                "directory_file_probe",
                "static_asset_probe",
            ),
            dependencies=("web",),
            adapter_modules=("actions.py", "parser.py"),
            safety_notes=(
                "uses only observed file-like paths as anchors",
                "does not seed hidden Juice Shop file names",
                "loopback targets only through ToolExecutor",
            ),
            limitations=(
                "does not yet perform multipart upload",
                "does not inspect downloaded binary metadata",
            ),
        )
    )

    def candidates(self, store: KnowledgeStore) -> list[ActionCandidate]:
        return [*generate_candidates(store), *generate_file_surface_candidates(store)]

    def candidates_for_policy(self, store: KnowledgeStore, policy: PolicyView) -> list[ActionCandidate]:
        return [*generate_candidates_for_policy(store, policy), *generate_file_surface_candidates_for_policy(store, policy)]


@dataclass(frozen=True)
class BrowserDomPlugin(WebPentestPlugin):
    name: str = "browser-dom-actions"
    metadata: PluginMetadata = field(
        default_factory=lambda: PluginMetadata(
            name="browser-dom-actions",
            description="Browser DOM action plugin placeholder for Playwright-backed click/fill/navigation primitives.",
            domain="local_web_browser",
            reward_observers=("none", "juice-shop"),
            capabilities=(
                "future:browser_navigation",
                "future:form_fill",
                "future:button_click",
                "future:dom_text_observation",
            ),
            dependencies=("future:playwright", "web"),
            adapter_modules=("future:browser_tools.py", "future:browser_parser.py"),
            safety_notes=(
                "must stay loopback-only",
                "must expose primitive DOM actions, not challenge steps",
                "must not read writeups or seed known solutions",
            ),
            limitations=("registered for dependency planning; action execution not implemented yet",),
        )
    )


@dataclass(frozen=True)
class StorageStatePlugin(WebPentestPlugin):
    name: str = "storage-state-observer"
    metadata: PluginMetadata = field(
        default_factory=lambda: PluginMetadata(
            name="storage-state-observer",
            description="Browser storage observer placeholder for cookies, localStorage, sessionStorage, and JWT-like values.",
            domain="local_web_browser_state",
            reward_observers=("none", "juice-shop"),
            capabilities=(
                "cookie_observation",
                "future:local_storage_observation",
                "future:session_storage_observation",
                "future:jwt_candidate_extraction",
            ),
            dependencies=("web", "future:browser-dom-actions"),
            adapter_modules=("parser.py", "future:browser_storage.py"),
            safety_notes=(
                "observes only state produced by the local target",
                "does not seed auth tokens",
            ),
            limitations=("cookie observation is available; browser storage observation is not implemented yet",),
        )
    )


@dataclass(frozen=True)
class WebReconPlugin(WebPentestPlugin):
    name: str = "web-recon-actions"
    metadata: PluginMetadata = field(
        default_factory=lambda: PluginMetadata(
            name="web-recon-actions",
            description="Reconnaissance adapter for shallow service and passive web fingerprinting.",
            domain="local_web_recon",
            reward_observers=("none", "juice-shop"),
            capabilities=("shallow_nmap", "passive_fingerprint", "http_options", "header_observation"),
            dependencies=("web",),
            adapter_modules=("actions.py", "tools.py", "parser.py"),
            safety_notes=("shallow scans only", "loopback targets only", "no exploit modules"),
            limitations=("does not run directory brute forcing or vulnerability scanners",),
        )
    )


@dataclass(frozen=True)
class JuiceShopFullPlugin(WebPentestPlugin):
    name: str = "juice-shop-full"
    metadata: PluginMetadata = field(
        default_factory=lambda: PluginMetadata(
            name="juice-shop-full",
            description="Composite Juice Shop training adapter with web, JSON API, mutation, file-surface, recon, and reward plugins.",
            domain="local_web_spa_lab",
            reward_observers=("none", "juice-shop"),
            capabilities=(
                "web_baseline",
                "json_api_actions",
                "input_mutation_actions",
                "file_surface_actions",
                "web_recon_actions",
                "scoreboard_reward_observer",
                "planned_browser_dom_actions",
                "planned_storage_state_observer",
            ),
            dependencies=(
                "web",
                "json-api-actions",
                "input-mutation-actions",
                "file-surface-actions",
                "web-recon-actions",
                "browser-dom-actions",
                "storage-state-observer",
            ),
            adapter_modules=("actions.py", "tools.py", "parser.py", "reward.py"),
            safety_notes=(
                "local Juice Shop only",
                "scoreboard is reward/progress only",
                "no known challenge solution steps are seeded",
            ),
            limitations=(
                "browser DOM and storage plugins are registered but not executable yet",
                "does not guarantee full Juice Shop completion without further learned experience",
            ),
        )
    )

    def candidates(self, store: KnowledgeStore) -> list[ActionCandidate]:
        return _dedupe_plugin_candidates(
            [
                *generate_candidates(store),
                *generate_json_api_candidates(store),
                *generate_input_mutation_candidates(store),
                *generate_file_surface_candidates(store),
            ]
        )

    def candidates_for_policy(self, store: KnowledgeStore, policy: PolicyView) -> list[ActionCandidate]:
        return _dedupe_plugin_candidates(
            [
                *generate_candidates_for_policy(store, policy),
                *generate_json_api_candidates_for_policy(store, policy),
                *generate_input_mutation_candidates_for_policy(store, policy),
                *generate_file_surface_candidates_for_policy(store, policy),
            ]
        )


class PluginRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, type[TargetPlugin] | TargetPlugin] = {}

    def register(self, plugin: type[TargetPlugin] | TargetPlugin) -> None:
        instance = plugin() if isinstance(plugin, type) else plugin
        if instance.name in self._factories:
            raise ValueError(f"plugin already registered: {instance.name}")
        self._factories[instance.name] = plugin

    def create(self, name: str) -> TargetPlugin:
        plugin = self._factories.get(name)
        if plugin is None:
            known = ", ".join(self.names()) or "<none>"
            raise ValueError(f"unknown APASSR target plugin: {name}; available: {known}")
        return plugin() if isinstance(plugin, type) else plugin

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def metadata(self) -> tuple[PluginMetadata, ...]:
        rows: list[PluginMetadata] = []
        for name in self.names():
            rows.append(self.create(name).metadata)
        return tuple(rows)

    def manifest_rows(self) -> list[dict[str, object]]:
        return [
            {
                "name": row.name,
                "description": row.description,
                "domain": row.domain,
                "reward_observers": list(row.reward_observers),
                "capabilities": list(row.capabilities),
                "dependencies": list(row.dependencies),
                "adapter_modules": list(row.adapter_modules),
                "safety_notes": list(row.safety_notes),
                "limitations": list(row.limitations),
            }
            for row in self.metadata()
        ]


REGISTRY = PluginRegistry()
REGISTRY.register(WebPentestPlugin)
REGISTRY.register(JuiceShopPlugin)
REGISTRY.register(JsonApiActionsPlugin)
REGISTRY.register(InputMutationPlugin)
REGISTRY.register(FileSurfacePlugin)
REGISTRY.register(BrowserDomPlugin)
REGISTRY.register(StorageStatePlugin)
REGISTRY.register(WebReconPlugin)
REGISTRY.register(JuiceShopFullPlugin)


def _dedupe_plugin_candidates(candidates: list[ActionCandidate]) -> list[ActionCandidate]:
    seen: set[str] = set()
    output: list[ActionCandidate] = []
    for candidate in candidates:
        key = candidate.tried_key or candidate.label
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return output


def register_plugin(plugin: type[TargetPlugin] | TargetPlugin) -> None:
    REGISTRY.register(plugin)


def available_plugins() -> tuple[str, ...]:
    return REGISTRY.names()


def plugin_metadata() -> tuple[PluginMetadata, ...]:
    return REGISTRY.metadata()


def plugin_manifest_rows() -> list[dict[str, object]]:
    return REGISTRY.manifest_rows()


def get_plugin(name: str) -> TargetPlugin:
    return REGISTRY.create(name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect APASSR target plugins.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    rows = plugin_manifest_rows()
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    for row in rows:
        observers = ", ".join(row["reward_observers"])  # type: ignore[arg-type]
        print(f"{row['name']}: {row['description']} reward_observers=[{observers}]")


if __name__ == "__main__":
    main()
