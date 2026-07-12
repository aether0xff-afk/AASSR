from __future__ import annotations

import unittest

from apassr_tool.tools import SafetyError, ToolCall, ToolExecutor, ToolName
from apassr_tool.sandbox_server import make_server

import threading


class ToolSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = make_server("127.0.0.1", 0, quiet=True)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_blocks_external_url(self) -> None:
        executor = ToolExecutor(prefer_curl=False)
        with self.assertRaises(SafetyError):
            executor.execute(ToolCall(ToolName.CURL_GET, url="https://example.com/"))

    def test_nmap_unavailable_or_safe(self) -> None:
        executor = ToolExecutor(prefer_curl=False)
        result = executor.execute(ToolCall(ToolName.NMAP_SCAN, target_host="127.0.0.1"))
        self.assertIn(result.status, {0, 1, 127})
        if result.unavailable:
            self.assertIn("nmap", result.stderr)

    def test_wsl_backend_still_blocks_external_url(self) -> None:
        executor = ToolExecutor(backend="wsl", prefer_curl=True)
        with self.assertRaises(SafetyError):
            executor.execute(ToolCall(ToolName.CURL_GET, url="https://example.com/"))

    def test_wsl_command_uses_fixed_distro_wrapper(self) -> None:
        executor = ToolExecutor(backend="wsl", wsl_distro="kali-linux")
        command = executor._wrap_wsl(["curl", "-i", "http://127.0.0.1:8088/"])
        self.assertEqual(command[:4], [executor.wsl_path or "wsl.exe", "-d", "kali-linux", "--"])
        self.assertEqual(command[4:], ["curl", "-i", "http://127.0.0.1:8088/"])

    def test_post_body_is_split_for_wsl_safe_arguments(self) -> None:
        executor = ToolExecutor(backend="wsl")
        args = executor._curl_args(
            ToolCall(
                ToolName.CURL_POST,
                url="http://127.0.0.1:8088/login",
                data={"username": "admin", "password": "admin7"},
            )
        )
        self.assertIn("-d", args)
        self.assertNotIn("username=admin&password=admin7", args)
        self.assertIn("username=admin", args)
        self.assertIn("password=admin7", args)

    def test_json_body_is_sent_as_single_json_payload_for_curl(self) -> None:
        executor = ToolExecutor(backend="wsl")
        args = executor._curl_args(
            ToolCall(
                ToolName.CURL_JSON,
                url="http://127.0.0.1:8088/api/users",
                data={"email": "a@example.test"},
                method="PATCH",
            )
        )

        self.assertIn("PATCH", args)
        self.assertIn("Content-Type: application/json", args)
        self.assertIn('{"email":"a@example.test"}', args)

    def test_python_requests_json_tool_returns_blocked_on_unreachable_target(self) -> None:
        executor = ToolExecutor(prefer_curl=False)
        result = executor.execute(
            ToolCall(
                ToolName.CURL_JSON,
                url="http://127.0.0.1:1/api/test",
                data={"email": "a@example.test"},
                method="PATCH",
            )
        )

        self.assertEqual(result.status, 0)
        self.assertTrue(result.blocked)

    def test_python_requests_fallback_preserves_session_cookies(self) -> None:
        executor = ToolExecutor(prefer_curl=False)
        base_url = f"http://127.0.0.1:{self.port}"

        login = executor.execute(
            ToolCall(
                ToolName.CURL_POST,
                url=f"{base_url}/login",
                data={"username": "admin", "password": "admin7"},
            )
        )
        admin = executor.execute(ToolCall(ToolName.CURL_GET, url=f"{base_url}/admin"))

        self.assertEqual(login.status, 200)
        self.assertEqual(admin.status, 200)
        self.assertIn("FLAG{LOCAL_TOOL_APASSR_CHAIN}", admin.stdout)


if __name__ == "__main__":
    unittest.main()
