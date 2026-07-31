from __future__ import annotations

import subprocess

import pytest

from aassr_v2.safe_application import (
    _docker_cli,
    _docker_environment,
    run_safe_compose,
    verify_safe_compose_runtime,
)


@pytest.mark.docker
def test_docker_safe_application_runtime_is_isolated() -> None:
    try:
        docker = _docker_cli()
        subprocess.run(
            [docker, "info"],
            check=True,
            capture_output=True,
            timeout=20,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        pytest.skip("Docker engine is unavailable")
    compose = "docker/paper_safe_application/docker-compose.yml"
    project = "aassr-paper-safe-pytest"
    run_safe_compose(compose, action="up", project_name=project)
    try:
        result = verify_safe_compose_runtime(
            compose, project_name=project
        )
        assert result["verified_routes"] == [
            "observed-token",
            "configuration",
            "combined-evidence",
        ]
        assert result["distractor_count"] == 2
        assert result["external_egress_blocked"]
        assert result["root_write_blocked"]
        assert result["internal_network"]
        assert not result["host_ports_published"]
        subprocess.run(
            [
                docker,
                "compose",
                "--project-name",
                project,
                "--file",
                compose,
                "restart",
                "paper-target",
            ],
            check=True,
            capture_output=True,
            timeout=30,
            env=_docker_environment(docker),
        )
        repeated = verify_safe_compose_runtime(
            compose, project_name=project
        )
        for field in (
            "seed",
            "service",
            "port",
            "route_paths",
            "distractor_count",
            "verified_routes",
        ):
            assert repeated[field] == result[field]
    finally:
        run_safe_compose(compose, action="down", project_name=project)
    remaining = subprocess.run(
        [
            docker,
            "ps",
            "-a",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--format",
            "{{.ID}}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
        env=_docker_environment(docker),
    ).stdout.strip()
    assert not remaining
