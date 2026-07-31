from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .types import Action, StateSnapshot


@dataclass(frozen=True, slots=True)
class SafeApplicationStep:
    snapshot: StateSnapshot
    added_facts: frozenset[str] = frozenset()
    removed_facts: frozenset[str] = frozenset()
    unlocked_actions: tuple[Action, ...] = ()
    error: bool = False
    error_code: str = ""
    reward: float = 0.0
    effect_events: tuple[Mapping[str, Any], ...] = ()


class SafeLocalApplicationWorld:
    """Deterministic, non-exploitative local assessment simulation."""

    def __init__(
        self,
        *,
        seed: int,
        allowed_hosts: Sequence[str] = ("paper-target",),
    ) -> None:
        if not allowed_hosts:
            raise ValueError("allowed_hosts must be non-empty")
        if any(
            host in {"0.0.0.0", "::", "*"} or "." in str(host).strip(".")
            for host in allowed_hosts
        ):
            raise ValueError("only Compose-internal host names are allowed")
        self.seed = int(seed)
        self.allowed_hosts = tuple(str(host) for host in allowed_hosts)
        randomizer = random.Random(seed)
        self.target = randomizer.choice(self.allowed_hosts)
        self.target_port = 8080 + (self.seed % 4)
        self.service_name = (
            f"local-svc-{randomizer.getrandbits(16):04x}"
        )
        self.distractors = tuple(
            f"decoy-{randomizer.randrange(100, 999)}" for _ in range(2)
        )
        self.distractor_ports = tuple(
            randomizer.choice((7001, 7002, 7003, 7004))
            for _ in self.distractors
        )
        self.token = hashlib.sha256(f"token:{seed}".encode()).hexdigest()[:12]
        self.config_key = hashlib.sha256(
            f"config:{seed}".encode()
        ).hexdigest()[:8]
        route_salt = hashlib.sha256(
            f"routes:{seed}".encode()
        ).hexdigest()
        self.route_paths = {
            "observed": f"/flag-{route_salt[:6]}",
            "configuration": f"/configured-{route_salt[6:12]}",
            "combined": f"/combine-{route_salt[12:18]}",
        }
        self._observed = False
        self._note = False
        self._config = False
        self._terminal_family = ""
        self.terminal = False

    def _actions(self) -> tuple[Action, ...]:
        if self.terminal:
            return ()
        result = [
            Action("observe_local_services"),
            Action("read_local_note"),
            Action("inspect_local_config"),
        ]
        if self._observed:
            result.append(
                Action(
                    "request_local_flag",
                    parameters={
                        "target": self.target,
                        "port": self.target_port,
                        "path": self.route_paths["observed"],
                        "token": self.token,
                    },
                )
            )
        if self._note and self._config:
            result.append(
                Action(
                    "combine_local_evidence",
                    parameters={
                        "target": self.target,
                        "port": self.target_port,
                        "path": self.route_paths["combined"],
                        "note": self.token,
                        "config": self.config_key,
                    },
                )
            )
        if self._config:
            result.append(
                Action(
                    "use_local_configuration",
                    parameters={
                        "target": self.target,
                        "port": self.target_port,
                        "path": self.route_paths["configuration"],
                        "config": self.config_key,
                    },
                )
            )
        return tuple(result)

    def snapshot(self) -> StateSnapshot:
        facts = set()
        if self._observed:
            facts.add(f"local_service:{self.target}")
            facts.add(f"local_port:{self.target_port}")
            facts.add(f"local_service_kind:{self.service_name}")
            facts.add(f"observed_token:{self.token}")
            facts.update(
                f"local_service:{host}:{port}"
                for host, port in zip(
                    self.distractors,
                    self.distractor_ports,
                    strict=True,
                )
            )
        if self._note:
            facts.add(f"local_note:{self.token}")
        if self._config:
            facts.add(f"local_config:{self.config_key}")
        return StateSnapshot(
            (
                float(self._observed),
                float(self._note),
                float(self._config),
                float(self.terminal),
            ),
            frozenset(facts),
            self._actions(),
            1.0 if self.terminal else 0.0,
        )

    def _reject(self, code: str) -> SafeApplicationStep:
        return SafeApplicationStep(
            self.snapshot(), error=True, error_code=code
        )

    def step(self, action: Action) -> SafeApplicationStep:
        if self.terminal:
            raise RuntimeError("cannot step a terminal world")
        before = self.snapshot()
        target = action.parameters.get("target")
        if target is not None and str(target) not in self.allowed_hosts:
            return self._reject("target_not_allowlisted")
        available = {item.signature for item in before.available_actions}
        if action.signature not in available:
            return self._reject("action_not_available")
        events: list[Mapping[str, Any]] = []
        if action.verb_name == "observe_local_services":
            self._observed = True
            events.append({"effect": "information_acquisition"})
        elif action.verb_name == "read_local_note":
            self._note = True
            events.append({"effect": "information_acquisition"})
        elif action.verb_name == "inspect_local_config":
            self._config = True
            events.append({"effect": "configuration_acquisition"})
        elif action.verb_name == "request_local_flag":
            self.terminal = True
            self._terminal_family = "observed_service_parameter"
            events.extend(
                (
                    {
                        "effect": "parameter_application",
                        "prerequisites": ["information_acquisition"],
                        "relation": "parameter_dependency",
                    },
                    {
                        "effect": "goal_achievement",
                        "prerequisites": ["parameter_application"],
                        "relation": "enablement",
                    },
                )
            )
        elif action.verb_name == "combine_local_evidence":
            self.terminal = True
            self._terminal_family = "combined_evidence"
            events.extend(
                (
                    {
                        "effect": "evidence_combination",
                        "prerequisites": [
                            "information_acquisition",
                            "configuration_acquisition",
                        ],
                        "relation": "enablement",
                    },
                    {
                        "effect": "goal_achievement",
                        "prerequisites": ["evidence_combination"],
                        "relation": "enablement",
                    },
                )
            )
        elif action.verb_name == "use_local_configuration":
            self.terminal = True
            self._terminal_family = "configuration_route"
            events.extend(
                (
                    {
                        "effect": "configuration_application",
                        "prerequisites": ["configuration_acquisition"],
                        "relation": "parameter_dependency",
                    },
                    {
                        "effect": "goal_achievement",
                        "prerequisites": ["configuration_application"],
                        "relation": "enablement",
                    },
                )
            )
        after = self.snapshot()
        before_actions = {
            candidate.signature for candidate in before.available_actions
        }
        return SafeApplicationStep(
            after,
            added_facts=after.facts - before.facts,
            removed_facts=before.facts - after.facts,
            unlocked_actions=tuple(
                candidate
                for candidate in after.available_actions
                if candidate.signature not in before_actions
            ),
            reward=1.0 if self.terminal else 0.0,
            effect_events=tuple(events),
        )

    @property
    def analysis_solution_family(self) -> str:
        return self._terminal_family


def validate_compose_safety(path: str | Path) -> None:
    """Fail closed on Compose features outside the local study threat model."""
    text = Path(path).read_text(encoding="utf-8")
    normalized = text.lower()
    if "internal: true" not in normalized:
        raise ValueError("Compose network must set internal: true")
    prohibited = (
        "privileged: true",
        "network_mode: host",
        "pid: host",
        "/var/run/docker.sock",
        "cap_add:",
        "ports:",
        "external: true",
        "extra_hosts:",
        "dns:",
    )
    found = [item for item in prohibited if item in normalized]
    if found:
        raise ValueError(f"unsafe Compose directive(s): {found}")
    if "read_only: true" not in normalized:
        raise ValueError("Compose services must use read_only: true")
    if "cap_drop:" not in normalized or "- all" not in normalized:
        raise ValueError("Compose services must drop all capabilities")
    if "no-new-privileges:true" not in normalized.replace(" ", ""):
        raise ValueError("Compose services must set no-new-privileges")
    if (
        'user: "65534:65534"' not in normalized
        and "user: '65534:65534'" not in normalized
        and "user: 65534:65534" not in normalized
    ):
        raise ValueError("Compose services must declare a non-root user")


def _docker_cli() -> str:
    discovered = shutil.which("docker")
    if discovered:
        return discovered
    candidates = (
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Programs"
        / "DockerDesktop"
        / "resources"
        / "bin"
        / "docker.exe",
        Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
        / "Docker"
        / "Docker"
        / "resources"
        / "bin"
        / "docker.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "Docker CLI was not found; restart the shell after installation"
    )


def _docker_environment(docker_cli: str) -> dict[str, str]:
    environment = dict(os.environ)
    directory = str(Path(docker_cli).parent)
    current = environment.get("PATH", "")
    if directory.lower() not in {
        item.strip().lower()
        for item in current.split(os.pathsep)
        if item.strip()
    }:
        environment["PATH"] = directory + os.pathsep + current
    return environment


def run_safe_compose(
    compose_file: str | Path,
    *,
    action: str,
    project_name: str = "aassr-paper-safe",
    timeout_seconds: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    if action not in {"config", "up", "down"}:
        raise ValueError("action must be config, up, or down")
    compose = Path(compose_file).resolve()
    validate_compose_safety(compose)
    docker_cli = _docker_cli()
    command = [
        docker_cli,
        "compose",
        "--project-name",
        project_name,
        "--file",
        str(compose),
    ]
    if action == "config":
        command.append("config")
    elif action == "up":
        command.extend(("up", "--detach", "--wait", "--build"))
    else:
        command.extend(("down", "--volumes", "--remove-orphans"))
    return subprocess.run(
        command,
        cwd=compose.parent,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=_docker_environment(docker_cli),
    )


def verify_safe_compose_runtime(
    compose_file: str | Path,
    *,
    project_name: str = "aassr-paper-safe",
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Verify the running container, three routes, and isolation controls."""
    compose = Path(compose_file).resolve()
    validate_compose_safety(compose)
    docker_cli = _docker_cli()
    docker_environment = _docker_environment(docker_cli)
    base = [
        docker_cli,
        "compose",
        "--project-name",
        project_name,
        "--file",
        str(compose),
    ]
    container_id = subprocess.run(
        [*base, "ps", "-q", "paper-target"],
        cwd=compose.parent,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=docker_environment,
    ).stdout.strip()
    if not container_id:
        raise RuntimeError("paper-target is not running")
    deadline = time.monotonic() + timeout_seconds
    while True:
        inspection = json.loads(
            subprocess.run(
                [docker_cli, "inspect", container_id],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=docker_environment,
            ).stdout
        )[0]
        health = inspection["State"].get("Health", {}).get("Status")
        if health != "starting" or time.monotonic() >= deadline:
            break
        time.sleep(0.25)
    host = inspection["HostConfig"]
    config = inspection["Config"]
    if host.get("PortBindings"):
        raise RuntimeError("safe application publishes host ports")
    if not host.get("ReadonlyRootfs"):
        raise RuntimeError("safe application root filesystem is writable")
    if "ALL" not in (host.get("CapDrop") or ()):
        raise RuntimeError("safe application did not drop all capabilities")
    if "no-new-privileges:true" not in (
        host.get("SecurityOpt") or ()
    ):
        raise RuntimeError("safe application permits privilege escalation")
    if str(config.get("User", "")).split(":", 1)[0] in {"", "0", "root"}:
        raise RuntimeError("safe application runs as root")
    network_ids = {
        value["NetworkID"]
        for value in inspection["NetworkSettings"]["Networks"].values()
    }
    if not network_ids:
        raise RuntimeError("safe application has no isolated network")
    for network_id in network_ids:
        network = json.loads(
            subprocess.run(
                [docker_cli, "network", "inspect", network_id],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=docker_environment,
            ).stdout
        )[0]
        if not network.get("Internal", False):
            raise RuntimeError("safe application network is not internal")
    probe = r"""
import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

seed = int(os.environ["WORLD_SEED"])
base = f"http://127.0.0.1:{8080 + seed % 4}"
with urllib.request.urlopen(base + "/observe", timeout=3) as response:
    observation = json.load(response)
routes = [
    (
        observation["routes"]["observed"],
        {"token": observation["token_hint"]},
        "observed-token",
    ),
    (
        observation["routes"]["configuration"],
        {"config": observation["config_hint"]},
        "configuration",
    ),
    (
        observation["routes"]["combined"],
        {
            "note": observation["token_hint"],
            "config": observation["config_hint"],
        },
        "combined-evidence",
    ),
]
verified = []
for path, parameters, expected in routes:
    url = base + path + "?" + urllib.parse.urlencode(parameters)
    with urllib.request.urlopen(url, timeout=3) as response:
        payload = json.load(response)
    assert payload["route"] == expected
    assert payload["flag"] == "FLAG{AASSR_LOCAL_SIMULATION_ONLY}"
    verified.append(expected)
try:
    Path("/app/write-probe").write_text("blocked")
except OSError:
    root_write_blocked = True
else:
    root_write_blocked = False
try:
    with socket.create_connection(("1.1.1.1", 80), timeout=2):
        pass
except OSError:
    external_egress_blocked = True
else:
    external_egress_blocked = False
assert root_write_blocked and external_egress_blocked
print(json.dumps({
    "seed": seed,
    "service": observation["service"],
    "port": observation["target"]["port"],
    "route_paths": observation["routes"],
    "distractor_count": len(observation["distractors"]),
    "verified_routes": verified,
    "root_write_blocked": root_write_blocked,
    "external_egress_blocked": external_egress_blocked,
}, sort_keys=True))
"""
    probe_result = subprocess.run(
        [*base, "exec", "-T", "paper-target", "python", "-c", probe],
        cwd=compose.parent,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=docker_environment,
    )
    result = json.loads(probe_result.stdout.strip())
    result.update(
        {
            "healthy": inspection["State"].get("Health", {}).get(
                "Status"
            )
            == "healthy",
            "internal_network": True,
            "host_ports_published": False,
            "capabilities_dropped": True,
            "non_root_user": str(config["User"]),
        }
    )
    if not result["healthy"]:
        raise RuntimeError("safe application is not healthy")
    return result


def safe_world_manifest(world: SafeLocalApplicationWorld) -> dict[str, Any]:
    return {
        "seed": world.seed,
        "allowed_hosts": list(world.allowed_hosts),
        "target_is_internal": world.target in world.allowed_hosts,
        "target_port": world.target_port,
        "service_name": world.service_name,
        "route_paths": dict(world.route_paths),
        "distractors": [
            {"host": host, "port": port}
            for host, port in zip(
                world.distractors,
                world.distractor_ports,
                strict=True,
            )
        ],
        "distractor_count": len(world.distractors),
        "external_network_required": False,
        "contains_real_vulnerability": False,
    }
