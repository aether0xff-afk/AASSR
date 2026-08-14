from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from aassr_v2.core.plugin_contract import ActionCommand
from aassr_v2.plugins.local_http import LocalHttpConfig, LocalHttpPlugin


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/empty":
            body = b"<p>empty page</p>"
        else:
            body = b'<a href="/next">next</a>'
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        del format, args


def _serve(handler=Handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_local_plugin_is_loopback_only_and_uses_real_socket() -> None:
    server, thread = _serve()
    try:
        port = server.server_address[1]
        plugin = LocalHttpPlugin(
            LocalHttpConfig(f"http://127.0.0.1:{port}")
        )
        initial = plugin.reset()
        assert initial.observation.values["status"] is None

        result = plugin.step(
            ActionCommand(
                "request",
                {
                    "method": "GET",
                    "url": f"http://127.0.0.1:{port}/",
                },
            )
        )
        assert result.observation.values["status"] == 200
        assert f"http://127.0.0.1:{port}/next" in set(
            result.observation.values["links"]
        )
        assert result.reward == 0.0
        assert result.terminated is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_plugin_does_not_keep_discovery_history_between_responses() -> None:
    server, thread = _serve()
    try:
        port = server.server_address[1]
        base = f"http://127.0.0.1:{port}"
        plugin = LocalHttpPlugin(base)
        plugin.reset()
        first = plugin.step(
            ActionCommand("request", {"method": "GET", "url": f"{base}/"})
        )
        assert f"{base}/next" in set(first.observation.values["links"])

        second = plugin.step(
            ActionCommand("request", {"method": "GET", "url": f"{base}/empty"})
        )
        # The adapter reports only what the current response exposes. Remembering
        # /next for later decisions is a Core responsibility.
        assert tuple(second.observation.values["links"]) == ()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_local_plugin_refuses_non_loopback_target() -> None:
    with pytest.raises(ValueError, match="loopback-only"):
        LocalHttpPlugin("https://example.com/")


class ExternalRedirectHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(302)
        self.send_header("Location", "https://example.com/outside")
        self.end_headers()

    def log_message(self, format, *args):
        del format, args


def test_local_plugin_blocks_external_redirect_before_following_it() -> None:
    server, thread = _serve(ExternalRedirectHandler)
    try:
        port = server.server_address[1]
        plugin = LocalHttpPlugin(f"http://127.0.0.1:{port}")
        plugin.reset()
        with pytest.raises(ValueError, match="loopback-only"):
            plugin.step(
                ActionCommand(
                    "request",
                    {"method": "GET", "url": f"http://127.0.0.1:{port}/"},
                )
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


class ControlHeaderHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = b"<p>terminal result</p>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-AASSR-Reward", "1")
        self.send_header("X-AASSR-Terminated", "1")
        self.send_header("X-Public-Note", "visible")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        del format, args


def test_control_headers_are_not_learner_visible_observations() -> None:
    server, thread = _serve(ControlHeaderHandler)
    try:
        port = server.server_address[1]
        base = f"http://127.0.0.1:{port}"
        plugin = LocalHttpPlugin(base)
        plugin.reset()
        result = plugin.step(
            ActionCommand("request", {"method": "GET", "url": f"{base}/"})
        )

        assert result.reward == 1.0
        assert result.terminated is True
        headers = {
            str(key).lower(): str(value)
            for key, value in result.observation.values["headers"].items()
        }
        assert headers["x-public-note"] == "visible"
        assert "x-aassr-reward" not in headers
        assert "x-aassr-terminated" not in headers
        assert "x-aassr-truncated" not in headers
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
