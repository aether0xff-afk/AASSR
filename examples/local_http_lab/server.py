from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


class LabHandler(BaseHTTPRequestHandler):
    server_version = "AASSRLocalLab/1.0"

    def _send(
        self,
        status: int,
        body: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        payload = body.encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        path = urlparse(self.path).path
        cookie = self.headers.get("Cookie", "")

        if path == "/":
            self._send(
                HTTPStatus.OK,
                """
                <h1>Local research service</h1>
                <a href="/catalog">catalog</a>
                <a href="/notes">notes</a>
                <a href="/trap">mirror</a>
                """,
            )
            return

        if path == "/catalog":
            self._send(
                HTTPStatus.OK,
                """
                <a href="/item/a">item a</a>
                <a href="/item/b">item b</a>
                <a href="/item/c">item c</a>
                """,
            )
            return

        if path == "/notes":
            self._send(
                HTTPStatus.OK,
                """
                <p>Archived service notes.</p>
                <a href="/catalog">back</a>
                """,
            )
            return

        if path == "/item/a":
            self._send(
                HTTPStatus.OK,
                """
                <p>ordinary item</p>
                <a href="/catalog">back</a>
                """,
            )
            return

        if path == "/item/b":
            self._send(
                HTTPStatus.OK,
                """
                <p>session capability issued</p>
                <a href="/finish">continue</a>
                <a href="/catalog">back</a>
                """,
                headers={"Set-Cookie": "permit=1; Path=/"},
            )
            return

        if path == "/item/c":
            self._send(
                HTTPStatus.OK,
                """
                <p>ordinary item</p>
                <a href="/catalog">back</a>
                """,
            )
            return

        if path == "/finish":
            if "permit=1" not in cookie:
                self._send(
                    HTTPStatus.FORBIDDEN,
                    """
                    <p>missing capability</p>
                    <a href="/catalog">catalog</a>
                    """,
                )
                return
            self._send(
                HTTPStatus.OK,
                "<p>done</p>",
                headers={
                    "X-AASSR-Reward": "1",
                    "X-AASSR-Terminated": "1",
                },
            )
            return

        if path == "/trap":
            self._send(
                HTTPStatus.CONFLICT,
                "<p>terminal dead end</p>",
                headers={
                    "X-AASSR-Reward": "-1",
                    "X-AASSR-Terminated": "1",
                },
            )
            return

        self._send(
            HTTPStatus.NOT_FOUND,
            """
            <p>not found</p>
            <a href="/">home</a>
            """,
        )

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the loopback-only AASSR real HTTP research service."
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", int(args.port)), LabHandler)
    print(f"listening on http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
