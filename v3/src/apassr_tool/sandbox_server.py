from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from urllib.parse import parse_qs, urlparse


FLAG = "FLAG{LOCAL_TOOL_APASSR_CHAIN}"
SESSION_VALUE = "local-session-admin-7"


class SandbagHandler(BaseHTTPRequestHandler):
    server_version = "AASSRLocalSandbag/0.1"

    def log_message(self, format: str, *args: object) -> None:
        if getattr(self.server, "quiet", False):
            return
        super().log_message(format, *args)

    def _write(
        self,
        status: HTTPStatus,
        body: str | bytes,
        *,
        content_type: str = "text/plain; charset=utf-8",
        headers: dict[str, str] | None = None,
    ) -> None:
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path == "/":
            self._write(
                HTTPStatus.OK,
                """
                <html>
                  <head><title>AASSR Sandbag</title></head>
                  <body>
                    <h1>AASSR Local Sandbag</h1>
                    <a href="/robots.txt">robots</a>
                    <script src="/static/app.js"></script>
                    <form method="POST" action="/login">
                      <input name="username">
                      <input name="password" type="password">
                    </form>
                  </body>
                </html>
                """,
                content_type="text/html; charset=utf-8",
            )
            return
        if path == "/robots.txt":
            self._write(HTTPStatus.OK, "User-agent: *\nDisallow: /debug\n")
            return
        if path == "/debug":
            self._write(
                HTTPStatus.OK,
                "debug=true\nadmin user id is 7\ntry /api/users?id=7\nadmin area: /admin\n",
            )
            return
        if path == "/api/users":
            user_id = (query.get("id") or [""])[0]
            if user_id == "7":
                self._write(
                    HTTPStatus.OK,
                    json.dumps({"id": 7, "username": "admin", "role": "admin"}),
                    content_type="application/json; charset=utf-8",
                )
                return
            self._write(
                HTTPStatus.NOT_FOUND,
                json.dumps({"error": "unknown user"}),
                content_type="application/json; charset=utf-8",
            )
            return
        if path == "/static/app.js":
            self._write(
                HTTPStatus.OK,
                """
                const passwordRule = "password = role + id";
                const loginEndpoint = "/login";
                const flagPath = "/admin";
                """,
                content_type="application/javascript; charset=utf-8",
            )
            return
        if path == "/admin":
            cookie = self.headers.get("Cookie", "")
            if f"session={SESSION_VALUE}" in cookie:
                self._write(HTTPStatus.OK, f"welcome admin\n{FLAG}\n")
                return
            self._write(HTTPStatus.UNAUTHORIZED, "auth required: login first\n")
            return
        self._write(HTTPStatus.NOT_FOUND, "not found\n")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/login":
            self._write(HTTPStatus.NOT_FOUND, "not found\n")
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8")
        form = parse_qs(raw_body)
        username = (form.get("username") or [""])[0]
        password = (form.get("password") or [""])[0]
        if username == "admin" and password == "admin7":
            self._write(
                HTTPStatus.OK,
                "logged in\nnext: /admin\n",
                headers={"Set-Cookie": f"session={SESSION_VALUE}; Path=/; HttpOnly"},
            )
            return
        self._write(HTTPStatus.UNAUTHORIZED, "bad credentials\n")


def make_server(host: str, port: int, *, quiet: bool = False) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), SandbagHandler)
    server.quiet = quiet  # type: ignore[attr-defined]
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the AASSR local sandbag web server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    server = make_server(args.host, args.port, quiet=args.quiet)
    print(f"sandbag listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
