from __future__ import annotations

import ast
from pathlib import Path

from aassr_v2.core.manifest import (
    PLUGIN_ALLOWED_AUTHORITIES,
    PLUGIN_FORBIDDEN_AUTHORITIES,
)


FORBIDDEN_IMPORT_PARTS = (
    "pentest",
    "plugins.",
    ".plugins",
    "current_pentest",
)
FORBIDDEN_DOMAIN_TOKENS = (
    "http",
    "route",
    "profile",
    "csrf",
    "pentest",
    "transferdiagnosticworld",
)


def test_core_source_has_no_environment_dependency() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "aassr_v2" / "core"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for token in FORBIDDEN_DOMAIN_TOKENS:
            assert token not in lowered, f"{path} contains domain token {token!r}"

        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules.extend(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
            for module in modules:
                lowered_module = module.lower()
                assert not any(
                    part in lowered_module
                    for part in FORBIDDEN_IMPORT_PARTS
                ), f"{path} imports environment module {module!r}"


def test_manifest_makes_plugin_authority_one_way() -> None:
    assert "real-io" in PLUGIN_ALLOWED_AUTHORITIES
    assert "world-model-installation" in PLUGIN_FORBIDDEN_AUTHORITIES
    assert "strategic-action-filtering" in PLUGIN_FORBIDDEN_AUTHORITIES
