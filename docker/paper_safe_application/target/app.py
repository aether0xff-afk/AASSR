from __future__ import annotations

import hashlib
import json
import os
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


SEED = int(os.environ.get("WORLD_SEED", "41001"))
FLAG = os.environ.get("LOCAL_FLAG", "FLAG{LOCAL_ONLY}")
TOKEN = hashlib.sha256(f"token:{SEED}".encode()).hexdigest()[:12]
CONFIG = hashlib.sha256(f"config:{SEED}".encode()).hexdigest()[:8]
PORT = 8080 + (SEED % 4)
RANDOMIZER = random.Random(SEED)
SERVICE = f"local-svc-{RANDOMIZER.getrandbits(16):04x}"
DISTRACTORS = [
    {
        "host": f"decoy-{RANDOMIZER.randrange(100, 999)}",
        "port": RANDOMIZER.choice((7001, 7002, 7003, 7004)),
    }
    for _ in range(2)
]
ROUTE_SALT = hashlib.sha256(f"routes:{SEED}".encode()).hexdigest()
ROUTES = {
    "observed": f"/flag-{ROUTE_SALT[:6]}",
    "configuration": f"/configured-{ROUTE_SALT[6:12]}",
    "combined": f"/combine-{ROUTE_SALT[12:18]}",
}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/health":
            self._send(200, {"status": "ok"})
        elif parsed.path == "/observe":
            self._send(
                200,
                {
                    "service": SERVICE,
                    "target": {"host": "paper-target", "port": PORT},
                    "distractors": DISTRACTORS,
                    "token_hint": TOKEN,
                    "config_hint": CONFIG,
                    "routes": ROUTES,
                },
            )
        elif (
            parsed.path == ROUTES["observed"]
            and query.get("token") == [TOKEN]
        ):
            self._send(200, {"flag": FLAG, "route": "observed-token"})
        elif (
            parsed.path == ROUTES["combined"]
            and query.get("note") == [TOKEN]
            and query.get("config") == [CONFIG]
        ):
            self._send(200, {"flag": FLAG, "route": "combined-evidence"})
        elif (
            parsed.path == ROUTES["configuration"]
            and query.get("config") == [CONFIG]
        ):
            self._send(200, {"flag": FLAG, "route": "configuration"})
        else:
            self._send(404, {"error": "local route not satisfied"})

    def log_message(self, format: str, *args: object) -> None:
        del format, args


ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
