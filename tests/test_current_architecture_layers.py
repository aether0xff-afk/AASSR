from __future__ import annotations

import ast
from pathlib import Path

from aassr_v2.current_architecture_layers import (
    CURRENT_MODULE_OWNERSHIP,
    CurrentArchitectureLayer,
    modules_for_layer,
)


ROOT = Path(__file__).resolve().parents[1] / "src" / "aassr_v2"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = "." * node.level + module
            found.add(module)
    return found


def test_current_architecture_registry_has_all_four_layers() -> None:
    assert set(CURRENT_MODULE_OWNERSHIP.values()) == set(CurrentArchitectureLayer)
    for layer in CurrentArchitectureLayer:
        assert modules_for_layer(layer)


def test_boundary_core_modules_do_not_import_pentest_plugin_implementation() -> None:
    forbidden_fragments = (
        "pentest",
        "plugins.current_pentest",
        "current_status_models",
        "current_relational_state",
    )
    for filename in ("current_core_manifest.py", "current_plugin_api.py"):
        imports = _imports(ROOT / filename)
        rendered = " ".join(imports).lower()
        for fragment in forbidden_fragments:
            assert fragment not in rendered, (filename, fragment, imports)


def test_current_entrypoint_assembles_domain_through_plugin() -> None:
    imports = _imports(ROOT / "current_entrypoint.py")
    rendered = " ".join(imports)
    assert ".plugins.current_pentest" in rendered
    assert ".current_status_models" not in rendered
    assert ".current_relational_state_v3" not in rendered


def test_performance_modules_are_not_declared_as_core_or_plugin() -> None:
    for module, layer in CURRENT_MODULE_OWNERSHIP.items():
        if "performance" in module or module.endswith("current_hot_path_profile"):
            assert layer is CurrentArchitectureLayer.PERFORMANCE
