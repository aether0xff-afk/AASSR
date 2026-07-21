from __future__ import annotations

import subprocess
import unittest
from unittest.mock import MagicMock, Mock, patch

from apassr_tool.targets import _find_local_juice_shop_runtime, ensure_local_target


class TargetManagerTests(unittest.TestCase):
    @patch("apassr_tool.targets._is_reachable", return_value=True)
    def test_existing_target_is_reused(self, reachable: Mock) -> None:
        status = ensure_local_target("http://127.0.0.1:3000", plugin="juice-shop-full")

        self.assertTrue(status.ok)
        self.assertEqual(status.method, "existing")

    def test_refuses_non_local_target(self) -> None:
        status = ensure_local_target("https://example.com", plugin="juice-shop-full")

        self.assertFalse(status.ok)
        self.assertIn("non-local", status.message)

    @patch("apassr_tool.targets.Path.exists", return_value=True)
    @patch("apassr_tool.targets.Path.glob")
    def test_local_juice_runtime_prefers_prebuilt_build(self, glob: Mock, exists: Mock) -> None:
        candidate = MagicMock()
        candidate.__truediv__.return_value = candidate
        candidate.exists.return_value = True
        glob.return_value = [candidate]

        runtime = _find_local_juice_shop_runtime()

        self.assertIs(runtime, candidate)

    @patch("apassr_tool.targets._start_juice_shop")
    @patch("apassr_tool.targets._is_reachable", return_value=False)
    def test_juice_plugin_routes_to_juice_shop_starter(self, reachable: Mock, start_juice: Mock) -> None:
        start_juice.return_value.ok = True
        start_juice.return_value.message = "started"

        ensure_local_target("http://127.0.0.1:3000", plugin="juice-shop-full")

        start_juice.assert_called_once()

    @patch("apassr_tool.targets._start_sandbag")
    @patch("apassr_tool.targets._is_reachable", return_value=False)
    def test_8088_routes_to_sandbag_starter(self, reachable: Mock, start_sandbag: Mock) -> None:
        start_sandbag.return_value.ok = True
        start_sandbag.return_value.message = "started"

        ensure_local_target("http://127.0.0.1:8088", plugin="web")

        start_sandbag.assert_called_once()

    @patch("apassr_tool.targets._start_local_juice_shop")
    @patch("apassr_tool.targets._wait_reachable", return_value=True)
    @patch("apassr_tool.targets.shutil.which", return_value="docker")
    @patch("apassr_tool.targets.Path.exists", return_value=True)
    @patch("apassr_tool.targets.subprocess.run")
    def test_juice_shop_uses_docker_compose_when_available(
        self,
        run: Mock,
        exists: Mock,
        which: Mock,
        wait: Mock,
        local_fallback: Mock,
    ) -> None:
        from apassr_tool.targets import _start_juice_shop

        run.return_value = subprocess.CompletedProcess(["docker"], 0, "", "")
        status = _start_juice_shop("http://127.0.0.1:3000", timeout_s=0.1)

        self.assertTrue(status.ok)
        self.assertEqual(status.method, "docker-compose")
        local_fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
