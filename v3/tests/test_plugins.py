from __future__ import annotations

import unittest

from apassr_tool.dmp import APASSRToolDMP
from apassr_tool.knowledge import KK, KnowledgeStore, seed_knowledge
from apassr_tool.plugins import (
    BrowserDomPlugin,
    FileSurfacePlugin,
    InputMutationPlugin,
    JuiceShopPlugin,
    JsonApiActionsPlugin,
    PluginMetadata,
    StorageStatePlugin,
    WebPentestPlugin,
    WebReconPlugin,
    available_plugins,
    get_plugin,
    plugin_manifest_rows,
    register_plugin,
)
from apassr_tool.tools import ToolResult


class PluginTests(unittest.TestCase):
    def test_web_plugin_wraps_seed_candidates_parse_and_reward_observer(self) -> None:
        plugin = WebPentestPlugin()
        store = plugin.seed("http://127.0.0.1:3000")

        self.assertEqual(store.first(KK.BASE_URL), "http://127.0.0.1:3000")
        self.assertTrue(plugin.candidates(store))

        result = ToolResult(
            tool="CURL_GET",
            command=[],
            status=200,
            stdout='<a href="/debug">debug</a>',
        )
        parsed = plugin.parse(result)
        self.assertIn((KK.PATH, "/debug"), parsed)
        self.assertIsNone(plugin.reward_observer("none", "http://127.0.0.1:3000"))

    def test_dmp_can_use_named_plugin_without_changing_core_loop(self) -> None:
        dmp = APASSRToolDMP(base_url="http://127.0.0.1:3000", plugin="web", step_limit=1)

        self.assertEqual(dmp.plugin.name, "web")
        self.assertTrue(dmp.choose_candidate())

    def test_unknown_plugin_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_plugin("unknown")

    def test_plugin_registry_exposes_metadata_and_manifest(self) -> None:
        self.assertIn("web", available_plugins())
        self.assertIn("juice-shop", available_plugins())
        self.assertIn("juice-shop-full", available_plugins())
        self.assertIn("json-api-actions", available_plugins())
        self.assertIn("input-mutation-actions", available_plugins())
        self.assertIn("file-surface-actions", available_plugins())
        self.assertIn("browser-dom-actions", available_plugins())
        self.assertIn("storage-state-observer", available_plugins())
        self.assertIn("web-recon-actions", available_plugins())
        rows = plugin_manifest_rows()
        web = next(row for row in rows if row["name"] == "web")
        juice = next(row for row in rows if row["name"] == "juice-shop")
        json_api = next(row for row in rows if row["name"] == "json-api-actions")
        juice_full = next(row for row in rows if row["name"] == "juice-shop-full")

        self.assertEqual(web["domain"], "local_web")
        self.assertIn("juice-shop", web["reward_observers"])
        self.assertIn("actions.py", web["adapter_modules"])
        self.assertEqual(juice["domain"], "local_web_spa_lab")
        self.assertIn("json-api-actions", juice["dependencies"])
        self.assertIn("future:browser_dom_actions", juice["dependencies"])
        self.assertIn("scoreboard_reward_observer", juice["capabilities"])
        self.assertIn("no challenge order, known credentials, flags, or writeup-derived action ranking", juice["safety_notes"])
        self.assertIn("json_post", json_api["capabilities"])
        self.assertIn("observed_field_binding", json_api["capabilities"])
        self.assertIn("input-mutation-actions", juice_full["dependencies"])
        self.assertIn("file-surface-actions", juice_full["dependencies"])
        self.assertIn("planned_browser_dom_actions", juice_full["capabilities"])

    def test_juice_shop_plugin_reuses_web_surface_without_answer_seed(self) -> None:
        plugin = JuiceShopPlugin()
        store = plugin.seed("http://127.0.0.1:3000")

        self.assertEqual(store.first(KK.BASE_URL), "http://127.0.0.1:3000")
        self.assertTrue(plugin.candidates(store))
        self.assertFalse(store.has(KK.FLAG))
        self.assertFalse(store.has(KK.PASSWORD_CANDIDATE))

    def test_json_api_plugin_adds_json_candidates_from_observed_knowledge(self) -> None:
        plugin = JsonApiActionsPlugin()
        store = plugin.seed("http://127.0.0.1:3000")
        store.add(KK.ENDPOINT, "/api/Users", source="test")
        store.add(KK.PARAM_NAME, "email", source="test")
        store.add(KK.PROBE_VALUE, "a@example.test", source="test")

        candidates = plugin.candidates(store)
        labels = {candidate.label for candidate in candidates}

        self.assertTrue(any(label.startswith("JSON_POST /api/Users") for label in labels))
        self.assertTrue(any(label == "JSON_DELETE /api/Users" for label in labels))

    def test_juice_shop_full_composes_executable_web_capabilities(self) -> None:
        plugin = get_plugin("juice-shop-full")
        store = plugin.seed("http://127.0.0.1:3000")
        store.add(KK.ENDPOINT, "/api/Users", source="test")
        store.add(KK.PARAM_NAME, "email", source="test")
        store.add(KK.PROBE_VALUE, "a@example.test", source="test")
        store.add(KK.PATH, "/ftp", source="test")

        labels = {candidate.label for candidate in plugin.candidates(store)}

        self.assertTrue(any(label.startswith("JSON_POST /api/Users") for label in labels))
        self.assertTrue(any(label.startswith("MUTATE_QUERY /api/Users") for label in labels))
        self.assertIn("FILE_GET /ftp/package.json", labels)

    def test_planned_plugins_are_registered_as_non_solution_capabilities(self) -> None:
        for plugin_cls in [BrowserDomPlugin, StorageStatePlugin, WebReconPlugin, InputMutationPlugin, FileSurfacePlugin]:
            plugin = plugin_cls()
            self.assertTrue(plugin.metadata.safety_notes)
            self.assertNotIn("solution", " ".join(plugin.metadata.capabilities).lower())

    def test_can_register_new_plugin_without_changing_core(self) -> None:
        class TinyPlugin:
            name = "tiny-test"
            metadata = PluginMetadata(
                name="tiny-test",
                description="test-only plugin",
                domain="unit_test",
            )

            def seed(self, base_url: str) -> KnowledgeStore:
                return seed_knowledge(base_url)

            def candidates(self, store: KnowledgeStore):
                return []

            def parse(self, result: ToolResult):
                return []

            def reward_observer(self, name: str, base_url: str):
                return None

        if "tiny-test" not in available_plugins():
            register_plugin(TinyPlugin)

        plugin = get_plugin("tiny-test")
        self.assertEqual(plugin.name, "tiny-test")
        self.assertEqual(plugin.metadata.domain, "unit_test")


if __name__ == "__main__":
    unittest.main()
