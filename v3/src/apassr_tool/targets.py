from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import urlparse

import requests


@dataclass(frozen=True)
class TargetStatus:
    ok: bool
    url: str
    message: str
    method: str = "none"
    pid: int | None = None


def ensure_local_target(url: str, *, plugin: str = "", timeout_s: float = 45.0) -> TargetStatus:
    parsed = urlparse(url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return TargetStatus(False, url, f"refusing to auto-start non-local target: {parsed.hostname}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if _is_reachable(url, timeout_s=2.0):
        return TargetStatus(True, url, "target already reachable", method="existing")
    if port == 8088:
        return _start_sandbag(url, timeout_s=timeout_s)
    if port == 3000 or "juice" in plugin:
        return _start_juice_shop(url, timeout_s=timeout_s)
    return TargetStatus(False, url, f"no auto-start target is registered for port {port}")


def _start_sandbag(url: str, *, timeout_s: float) -> TargetStatus:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8088
    runtime = Path(".runtime")
    runtime.mkdir(exist_ok=True)
    log = runtime / "sandbag_auto.log"
    command = [
        sys.executable,
        "-m",
        "apassr_tool.sandbox_server",
        "--host",
        host,
        "--port",
        str(port),
        "--quiet",
    ]
    handle = log.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=_creationflags(),
    )
    (runtime / f"sandbag_{port}.pid").write_text(str(proc.pid), encoding="ascii")
    if _wait_reachable(url, timeout_s=timeout_s):
        return TargetStatus(True, url, f"sandbag started on {url}", method="python", pid=proc.pid)
    return TargetStatus(False, url, f"sandbag start timed out; see {log}", method="python", pid=proc.pid)


def _start_juice_shop(url: str, *, timeout_s: float) -> TargetStatus:
    compose = Path("docker-compose.juice-shop.yml")
    docker = shutil.which("docker")
    if docker and compose.exists():
        command = [docker, "compose", "-f", str(compose), "up", "-d"]
        try:
            proc = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=60, check=False)
        except subprocess.TimeoutExpired:
            proc = subprocess.CompletedProcess(command, 124, "", "docker compose timed out")
        if proc.returncode == 0 and _wait_reachable(url, timeout_s=timeout_s):
            return TargetStatus(True, url, "Juice Shop started with docker compose", method="docker-compose")
    local = _start_local_juice_shop(url, timeout_s=timeout_s)
    if local.ok:
        return local
    if docker and compose.exists():
        return TargetStatus(False, url, "docker compose did not make Juice Shop reachable; local fallback also failed")
    return local


def _start_local_juice_shop(url: str, *, timeout_s: float) -> TargetStatus:
    app_dir = _find_local_juice_shop_runtime()
    package = app_dir / "package.json"
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if not package.exists() or not npm:
        return TargetStatus(False, url, "local Juice Shop runtime or npm was not found", method="local-node")
    runtime = Path(".runtime")
    runtime.mkdir(exist_ok=True)
    stdout = Path("juice-auto.out.log").open("a", encoding="utf-8")
    stderr = Path("juice-auto.err.log").open("a", encoding="utf-8")
    proc = subprocess.Popen(
        [npm, "start"],
        cwd=app_dir,
        stdout=stdout,
        stderr=stderr,
        text=True,
        creationflags=_creationflags(),
    )
    Path("juice-auto.pid").write_text(str(proc.pid), encoding="ascii")
    if _wait_reachable(url, timeout_s=timeout_s):
        return TargetStatus(True, url, "Juice Shop started with local npm runtime", method="local-node", pid=proc.pid)
    return TargetStatus(False, url, "local Juice Shop start timed out; see juice-auto.*.log", method="local-node", pid=proc.pid)


def _find_local_juice_shop_runtime() -> Path:
    prebuilt_root = Path(".runtime") / "juice-shop-prebuilt"
    if prebuilt_root.exists():
        for candidate in sorted(prebuilt_root.glob("juice-shop_*")):
            if (candidate / "build" / "app.js").exists() and (candidate / "package.json").exists():
                return candidate
    return Path(".runtime") / "juice-shop"


def _is_reachable(url: str, *, timeout_s: float) -> bool:
    try:
        response = requests.get(url, timeout=timeout_s)
    except requests.RequestException:
        return False
    return 200 <= response.status_code < 500


def _wait_reachable(url: str, *, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _is_reachable(url, timeout_s=2.0):
            return True
        time.sleep(1.0)
    return False


def _creationflags() -> int:
    if sys.platform.startswith("win"):
        return subprocess.CREATE_NO_WINDOW
    return 0
