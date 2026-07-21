from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import shutil
import subprocess
import time
from typing import Literal
from urllib.parse import urlparse

import requests


ExecutionBackend = Literal["local", "wsl"]


class ToolName(str, Enum):
    CURL_GET = "CURL_GET"
    CURL_HEAD = "CURL_HEAD"
    CURL_OPTIONS = "CURL_OPTIONS"
    CURL_POST = "CURL_POST"
    CURL_JSON = "CURL_JSON"
    NMAP_SCAN = "NMAP_SCAN"
    WHATWEB_SCAN = "WHATWEB_SCAN"


@dataclass(frozen=True)
class ToolCall:
    tool: ToolName
    url: str | None = None
    data: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    target_host: str | None = None
    port_range: str = "1-1024"
    method: str = "GET"


@dataclass
class ToolResult:
    tool: str
    command: list[str]
    status: int
    stdout: str
    stderr: str = ""
    duration_s: float = 0.0
    blocked: bool = False
    unavailable: bool = False

    @property
    def ok(self) -> bool:
        return not self.blocked and not self.unavailable and 200 <= self.status < 400


class SafetyError(ValueError):
    pass


class ToolExecutor:
    def __init__(
        self,
        *,
        allowed_hosts: set[str] | None = None,
        timeout_s: float = 5.0,
        prefer_curl: bool = True,
        backend: ExecutionBackend = "local",
        wsl_distro: str = "kali-linux",
    ) -> None:
        self.allowed_hosts = allowed_hosts or {"127.0.0.1", "localhost", "::1"}
        self.timeout_s = timeout_s
        self.prefer_curl = prefer_curl
        self.backend = backend
        self.wsl_distro = wsl_distro
        self.curl_path = shutil.which("curl.exe") or shutil.which("curl")
        self.nmap_path = shutil.which("nmap.exe") or shutil.which("nmap")
        self.whatweb_path = shutil.which("whatweb.exe") or shutil.which("whatweb")
        self.wsl_path = shutil.which("wsl.exe") or shutil.which("wsl")
        self.session = requests.Session()
        if backend not in {"local", "wsl"}:
            raise SafetyError(f"unsupported execution backend: {backend}")

    def execute(self, call: ToolCall) -> ToolResult:
        if call.tool in {ToolName.CURL_GET, ToolName.CURL_HEAD, ToolName.CURL_OPTIONS, ToolName.CURL_POST, ToolName.CURL_JSON}:
            if not call.url:
                raise SafetyError("HTTP tool requires url")
            self._assert_allowed_url(call.url)
            if self.backend == "wsl":
                return self._execute_wsl_curl(call)
            if self.prefer_curl and self.curl_path:
                return self._execute_curl(call)
            return self._execute_requests(call)
        if call.tool == ToolName.NMAP_SCAN:
            if not call.target_host:
                raise SafetyError("nmap tool requires target_host")
            self._assert_allowed_host(call.target_host)
            if self.backend == "wsl":
                return self._execute_wsl_nmap(call)
            return self._execute_nmap(call)
        if call.tool == ToolName.WHATWEB_SCAN:
            if not call.url:
                raise SafetyError("whatweb tool requires url")
            self._assert_allowed_url(call.url)
            if self.backend == "wsl":
                return self._execute_wsl_whatweb(call)
            return self._execute_whatweb(call)
        raise SafetyError(f"unsupported tool: {call.tool}")

    def _assert_allowed_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise SafetyError(f"blocked non-http URL: {url}")
        host = parsed.hostname
        if not host:
            raise SafetyError(f"blocked URL without host: {url}")
        self._assert_allowed_host(host)

    def _assert_allowed_host(self, host: str) -> None:
        if host not in self.allowed_hosts:
            raise SafetyError(f"blocked target outside allowlist: {host}")

    def _execute_curl(self, call: ToolCall) -> ToolResult:
        command = [self.curl_path or "curl", *self._curl_args(call)]
        return self._run_curl_command(call, command)

    def _execute_wsl_curl(self, call: ToolCall) -> ToolResult:
        command = self._wrap_wsl(["curl", *self._curl_args(call)])
        if not self.wsl_path:
            return ToolResult(
                tool=call.tool.value,
                command=command,
                status=127,
                stdout="",
                stderr="wsl is not installed",
                unavailable=True,
            )
        return self._run_curl_command(call, command)

    def _curl_args(self, call: ToolCall) -> list[str]:
        assert call.url is not None
        args = ["-i", "-sS", "--max-time", str(self.timeout_s)]
        for key, value in sorted(call.headers.items()):
            args.extend(["-H", f"{key}: {value}"])
        if call.tool == ToolName.CURL_HEAD:
            args.extend(["-I"])
        if call.tool == ToolName.CURL_OPTIONS:
            args.extend(["-X", "OPTIONS"])
        if call.tool == ToolName.CURL_POST:
            args.extend(["-X", "POST"])
            for key, value in sorted(call.data.items()):
                args.extend(["-d", f"{key}={value}"])
        if call.tool == ToolName.CURL_JSON:
            method = call.method.upper()
            args.extend(["-X", method])
            args.extend(["-H", "Content-Type: application/json"])
            if method != "DELETE" or call.data:
                args.extend(["-d", json.dumps(call.data, separators=(",", ":"))])
        args.append(call.url)
        return args

    def _run_curl_command(self, call: ToolCall, command: list[str]) -> ToolResult:
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self.timeout_s + 1.0,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start
            return ToolResult(
                tool=call.tool.value,
                command=command,
                status=0,
                stdout=exc.stdout or "",
                stderr=f"command timed out after {self.timeout_s + 1.0:.1f}s",
                duration_s=duration,
                blocked=True,
            )
        duration = time.perf_counter() - start
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        status = _status_from_curl_output(stdout) or proc.returncode
        return ToolResult(
            tool=call.tool.value,
            command=command,
            status=status,
            stdout=stdout,
            stderr=stderr,
            duration_s=duration,
        )

    def _execute_requests(self, call: ToolCall) -> ToolResult:
        assert call.url is not None
        start = time.perf_counter()
        headers = {"Connection": "close", **call.headers}
        try:
            if call.tool == ToolName.CURL_GET:
                response = self.session.get(call.url, headers=headers, timeout=self.timeout_s)
                command = ["python-requests", "GET", call.url]
            elif call.tool == ToolName.CURL_HEAD:
                response = self.session.head(call.url, headers=headers, timeout=self.timeout_s)
                command = ["python-requests", "HEAD", call.url]
            elif call.tool == ToolName.CURL_OPTIONS:
                response = self.session.options(call.url, headers=headers, timeout=self.timeout_s)
                command = ["python-requests", "OPTIONS", call.url]
            elif call.tool == ToolName.CURL_JSON:
                method = call.method.upper()
                headers = {"Content-Type": "application/json", **headers}
                response = self.session.request(
                    method,
                    call.url,
                    json=call.data if method != "DELETE" or call.data else None,
                    headers=headers,
                    timeout=self.timeout_s,
                )
                command = ["python-requests", method, "JSON", call.url]
            else:
                response = self.session.post(
                    call.url,
                    data=call.data,
                    headers=headers,
                    timeout=self.timeout_s,
                )
                command = ["python-requests", "POST", call.url]
        except requests.RequestException as exc:
            duration = time.perf_counter() - start
            return ToolResult(
                tool=call.tool.value,
                command=["python-requests", call.tool.value, call.url],
                status=0,
                stdout="",
                stderr=str(exc),
                duration_s=duration,
                blocked=True,
            )
        duration = time.perf_counter() - start
        header_lines = [f"HTTP/1.1 {response.status_code}"]
        header_lines.extend(f"{key}: {value}" for key, value in response.headers.items())
        stdout = "\n".join(header_lines) + "\n\n" + response.text
        return ToolResult(
            tool=call.tool.value,
            command=command,
            status=response.status_code,
            stdout=stdout,
            duration_s=duration,
        )

    def _execute_nmap(self, call: ToolCall) -> ToolResult:
        command = [
            self.nmap_path or "nmap",
            "-Pn",
            "-sV",
            "-p",
            call.port_range,
            call.target_host or "",
        ]
        if not self.nmap_path:
            return ToolResult(
                tool=call.tool.value,
                command=command,
                status=127,
                stdout="",
                stderr="nmap is not installed",
                unavailable=True,
        )
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=max(self.timeout_s, 10.0),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start
            return ToolResult(
                tool=call.tool.value,
                command=command,
                status=0,
                stdout=exc.stdout or "",
                stderr=f"command timed out after {max(self.timeout_s, 10.0):.1f}s",
                duration_s=duration,
                blocked=True,
            )
        duration = time.perf_counter() - start
        return ToolResult(
            tool=call.tool.value,
            command=command,
            status=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            duration_s=duration,
        )

    def _execute_wsl_nmap(self, call: ToolCall) -> ToolResult:
        command = self._wrap_wsl(
            [
                "nmap",
                "-Pn",
                "-sV",
                "-p",
                call.port_range,
                call.target_host or "",
            ]
        )
        if not self.wsl_path:
            return ToolResult(
                tool=call.tool.value,
                command=command,
                status=127,
                stdout="",
                stderr="wsl is not installed",
                unavailable=True,
        )
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=max(self.timeout_s, 10.0),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start
            return ToolResult(
                tool=call.tool.value,
                command=command,
                status=0,
                stdout=exc.stdout or "",
                stderr=f"command timed out after {max(self.timeout_s, 10.0):.1f}s",
                duration_s=duration,
                blocked=True,
            )
        duration = time.perf_counter() - start
        return ToolResult(
            tool=call.tool.value,
            command=command,
            status=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            duration_s=duration,
        )

    def _execute_whatweb(self, call: ToolCall) -> ToolResult:
        assert call.url is not None
        command = [self.whatweb_path or "whatweb", "--no-errors", call.url]
        if not self.whatweb_path:
            return ToolResult(
                tool=call.tool.value,
                command=command,
                status=127,
                stdout="",
                stderr="whatweb is not installed",
                unavailable=True,
            )
        return self._run_external(call, command, timeout=max(self.timeout_s, 10.0))

    def _execute_wsl_whatweb(self, call: ToolCall) -> ToolResult:
        assert call.url is not None
        command = self._wrap_wsl(["whatweb", "--no-errors", call.url])
        if not self.wsl_path:
            return ToolResult(
                tool=call.tool.value,
                command=command,
                status=127,
                stdout="",
                stderr="wsl is not installed",
                unavailable=True,
            )
        return self._run_external(call, command, timeout=max(self.timeout_s, 10.0))

    def _run_external(self, call: ToolCall, command: list[str], *, timeout: float) -> ToolResult:
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start
            return ToolResult(
                tool=call.tool.value,
                command=command,
                status=0,
                stdout=exc.stdout or "",
                stderr=f"command timed out after {timeout:.1f}s",
                duration_s=duration,
                blocked=True,
            )
        duration = time.perf_counter() - start
        return ToolResult(
            tool=call.tool.value,
            command=command,
            status=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            duration_s=duration,
            unavailable=proc.returncode == 127,
        )

    def _wrap_wsl(self, command: list[str]) -> list[str]:
        return [self.wsl_path or "wsl.exe", "-d", self.wsl_distro, "--", *command]


def _status_from_curl_output(output: str) -> int | None:
    for line in output.splitlines():
        if line.startswith("HTTP/"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    return None
