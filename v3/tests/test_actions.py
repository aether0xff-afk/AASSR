from __future__ import annotations

import unittest

from apassr_tool.actions import (
    ActionTemplate,
    generate_candidates,
    generate_file_surface_candidates,
    generate_input_mutation_candidates,
    generate_json_api_candidates,
)
from apassr_tool.knowledge import KK, seed_knowledge
from apassr_tool.policy import What
from apassr_tool.tools import ToolName


class ActionGenerationTests(unittest.TestCase):
    def test_seed_url_creates_host_port_tool_candidates(self) -> None:
        store = seed_knowledge("http://127.0.0.1:8088")
        self.assertIn("127.0.0.1", store.values(KK.HOST))
        self.assertIn("8088", store.values(KK.PORT))

        candidates = generate_candidates(store)
        nmap = [candidate for candidate in candidates if candidate.template == ActionTemplate.NMAP_SCAN_HOST]
        self.assertTrue(nmap)
        self.assertEqual(nmap[0].policy.what, What.PORT_SCAN)
        self.assertEqual(nmap[0].tool_call.target_host, "127.0.0.1")
        self.assertEqual(nmap[0].tool_call.port_range, "8088")

    def test_basic_http_metadata_candidates_are_generated(self) -> None:
        store = seed_knowledge("http://127.0.0.1:8088")
        candidates = generate_candidates(store)
        templates = {candidate.template for candidate in candidates}
        self.assertIn(ActionTemplate.HTTP_HEAD_PATH, templates)
        self.assertIn(ActionTemplate.HTTP_OPTIONS_PATH, templates)

    def test_query_probe_candidates_are_generated_from_parameter_knowledge(self) -> None:
        store = seed_knowledge("http://127.0.0.1:8088")
        store.add(KK.ENDPOINT, "/api/users", source="test")
        store.add(KK.PARAM_NAME, "id", source="test")
        store.add(KK.PROBE_VALUE, "7", source="observed:test")
        candidates = generate_candidates(store)
        probes = [candidate for candidate in candidates if candidate.template == ActionTemplate.HTTP_QUERY_PROBE]
        self.assertTrue(probes)
        self.assertTrue(any(candidate.label == "PROBE /api/users?id=7" for candidate in probes))
        self.assertTrue(any(candidate.policy.what == What.QUERY_PROBE for candidate in probes))

    def test_post_probe_candidates_are_generated_from_parameter_knowledge(self) -> None:
        store = seed_knowledge("http://127.0.0.1:8088")
        store.add(KK.ENDPOINT, "/rest/user/login", source="test")
        store.add(KK.PARAM_NAME, "email", source="test")
        store.add(KK.PARAM_NAME, "password", source="test")
        store.add(KK.PROBE_VALUE, "observed@example.test", source="observed:test")
        candidates = generate_candidates(store)
        probes = [candidate for candidate in candidates if candidate.template == ActionTemplate.HTTP_POST_PROBE]
        combos = [candidate for candidate in candidates if candidate.template == ActionTemplate.HTTP_POST_COMBO]
        self.assertTrue(probes)
        self.assertTrue(any(candidate.tool_call.data == {"email": "observed@example.test"} for candidate in probes))
        self.assertTrue(
            any(
                candidate.tool_call.data
                == {"email": "observed@example.test", "password": "observed@example.test"}
                for candidate in combos
            )
        )

    def test_wild_binding_allows_cross_context_parameter_bindings(self) -> None:
        store = seed_knowledge("http://127.0.0.1:8088")
        store.add(KK.ENDPOINT, "/api/feedback", source="observed:test")
        store.add(KK.PARAM_NAME, "background", source="GET /")
        store.add(KK.PROBE_VALUE, "blue", source="GET /")
        store.add(KK.PARAM_NAME, "rating", source="GET /api/feedback")
        store.add(KK.PROBE_VALUE, "5", source="GET /api/feedback")

        probes = [
            candidate
            for candidate in generate_candidates(store)
            if candidate.template == ActionTemplate.HTTP_QUERY_PROBE
        ]

        labels = {candidate.label for candidate in probes}
        self.assertIn("PROBE /api/feedback?background=blue", labels)
        self.assertIn("PROBE /api/feedback?rating=5", labels)

    def test_combo_post_candidates_use_observed_parameter_pool(self) -> None:
        store = seed_knowledge("http://127.0.0.1:8088")
        store.add(KK.ENDPOINT, "/api/Users", source="GET /api/Users")
        store.add(KK.PARAM_NAME, "email", source="GET /api/Users")
        store.add(KK.PARAM_NAME, "password", source="GET /api/Users")
        store.add(KK.PARAM_NAME, "role", source="GET /api/Users")
        store.add(KK.PROBE_VALUE, "a@example.test", source="GET /api/Users")
        store.add(KK.PROBE_VALUE, "true", source="GET /api/Users")

        candidates = [
            candidate
            for candidate in generate_candidates(store)
            if candidate.template == ActionTemplate.HTTP_POST_COMBO
            and candidate.label.startswith("POST_COMBO /api/Users")
        ]

        self.assertTrue(candidates)
        self.assertEqual(set(candidates[0].tool_call.data), {"email", "password", "role"})
        self.assertTrue(set(candidates[0].tool_call.data.values()).issubset({"a@example.test", "true"}))

    def test_json_api_candidates_use_observed_endpoint_fields_and_values(self) -> None:
        store = seed_knowledge("http://127.0.0.1:8088")
        store.add(KK.ENDPOINT, "/api/Users", source="GET /api/Users")
        store.add(KK.PARAM_NAME, "email", source="GET /api/Users")
        store.add(KK.PARAM_NAME, "password", source="GET /api/Users")
        store.add(KK.PROBE_VALUE, "a@example.test", source="GET /api/Users")

        candidates = generate_json_api_candidates(store)
        templates = {candidate.template for candidate in candidates}

        self.assertIn(ActionTemplate.HTTP_JSON_POST, templates)
        self.assertIn(ActionTemplate.HTTP_JSON_PUT, templates)
        self.assertIn(ActionTemplate.HTTP_JSON_PATCH, templates)
        self.assertIn(ActionTemplate.HTTP_JSON_DELETE, templates)
        post = next(candidate for candidate in candidates if candidate.template == ActionTemplate.HTTP_JSON_POST)
        self.assertEqual(post.tool_call.tool, ToolName.CURL_JSON)
        self.assertEqual(post.tool_call.method, "POST")
        self.assertTrue(set(post.tool_call.data).issubset({"email", "password"}))

    def test_input_mutation_candidates_add_generic_values_without_answer_seed(self) -> None:
        store = seed_knowledge("http://127.0.0.1:8088")
        store.add(KK.ENDPOINT, "/api/search", source="test")
        store.add(KK.PARAM_NAME, "q", source="test")
        store.add(KK.PROBE_VALUE, "7", source="observed:test")

        candidates = generate_input_mutation_candidates(store)
        labels = {candidate.label for candidate in candidates}

        self.assertTrue(any("MUTATE_QUERY /api/search?q=0" in label for label in labels))
        self.assertTrue(any("MUTATE_POST /api/search q=true" in label for label in labels))
        self.assertTrue(any("MUTATE_JSON /api/search q=" in label for label in labels))

    def test_file_surface_candidates_are_anchored_to_observed_paths(self) -> None:
        store = seed_knowledge("http://127.0.0.1:8088")
        store.add(KK.PATH, "/ftp", source="observed:test")
        store.add(KK.PATH, "/assets/app.js", source="observed:test")

        candidates = generate_file_surface_candidates(store)
        labels = {candidate.label for candidate in candidates}

        self.assertIn("FILE_GET /ftp/package.json", labels)
        self.assertIn("FILE_GET /assets/app.js.bak", labels)

    def test_seed_does_not_include_probe_values(self) -> None:
        store = seed_knowledge("http://127.0.0.1:8088")
        self.assertFalse(store.values(KK.PROBE_VALUE))

    def test_auth_get_requires_observed_auth_path(self) -> None:
        store = seed_knowledge("http://127.0.0.1:8088")
        store.add(KK.SESSION_COOKIE, "session=abc", source="observed:test")

        self.assertFalse(
            [candidate for candidate in generate_candidates(store) if candidate.template == ActionTemplate.HTTP_AUTH_GET]
        )

        store.add(KK.AUTH_PATH, "/observed-admin", source="observed:test")
        self.assertTrue(
            [
                candidate
                for candidate in generate_candidates(store)
                if candidate.template == ActionTemplate.HTTP_AUTH_GET
                and candidate.bindings.get(KK.AUTH_PATH) == "/observed-admin"
            ]
        )


if __name__ == "__main__":
    unittest.main()
