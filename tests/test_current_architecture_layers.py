from __future__ import annotations

import ast
from pathlib import Path

from aassr_v2.current_architecture_layers import (
    CURRENT_MODULE_OWNERSHIP,
    CurrentArchitectureLayer,
    modules_for_layer,
)


ROOT = Path(__file__).resolve().parents[1] / "src" / "aassr_v2"
LOCAL_MODULES = {
    "aassr_v2." + ".".join(path.relative_to(ROOT).with_suffix("").parts)
    for path in ROOT.rglob("*.py")
    if path.name != "__init__.py"
}


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


def _module_path(module: str) -> Path:
    prefix = "aassr_v2."
    if not module.startswith(prefix):
        return ROOT / "__init__.py"
    return ROOT / Path(*module.removeprefix(prefix).split(".")).with_suffix(".py")


def _absolute_imports(module: str) -> set[str]:
    path = _module_path(module)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = module.split(".")[:-1]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(
                alias.name
                for alias in node.names
                if alias.name.startswith("aassr_v2")
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level + 1]
                suffix = (node.module or "").split(".") if node.module else []
                imported = ".".join((*base, *suffix))
            else:
                imported = node.module or ""
            if imported.startswith("aassr_v2"):
                if imported in LOCAL_MODULES:
                    found.add(imported)
                for alias in node.names:
                    candidate = f"{imported}.{alias.name}"
                    if candidate in LOCAL_MODULES:
                        found.add(candidate)
    return found


def _transitive_runtime_modules() -> set[str]:
    pending = ["aassr_v2.current_entrypoint"]
    found: set[str] = set()
    while pending:
        module = pending.pop()
        if module in found or not _module_path(module).exists():
            continue
        found.add(module)
        pending.extend(_absolute_imports(module) - found)
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
        "http",
        "route",
        "profile",
        "csrf",
    )
    for module in modules_for_layer(CurrentArchitectureLayer.CORE):
        path = _module_path(module)
        imports = _absolute_imports(module)
        rendered = " ".join(imports).lower()
        for fragment in forbidden_fragments:
            assert fragment not in rendered, (path.name, fragment, imports)


def test_active_current_modules_have_exactly_one_registered_owner() -> None:
    active = _transitive_runtime_modules()
    assert active <= set(CURRENT_MODULE_OWNERSHIP), sorted(
        active - set(CURRENT_MODULE_OWNERSHIP)
    )
    for module in active:
        assert isinstance(CURRENT_MODULE_OWNERSHIP[module], CurrentArchitectureLayer)


def test_architecture_registry_literal_has_no_duplicate_module_keys() -> None:
    path = ROOT / "current_architecture_layers.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys = [
        key.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant)
        and isinstance(key.value, str)
        and key.value.startswith("aassr_v2.")
    ]
    assert len(keys) == len(set(keys))


def test_current_entrypoint_assembles_domain_through_plugin() -> None:
    imports = _imports(ROOT / "current_entrypoint.py")
    rendered = " ".join(imports)
    assert ".plugins.current_pentest" in rendered
    assert ".current_status_models" not in rendered
    assert ".current_relational_state_v3" not in rendered


def test_current_agent_has_no_top_level_pentest_implementation_imports() -> None:
    tree = ast.parse((ROOT / "current_agent.py").read_text(encoding="utf-8"))
    imports = {
        (node.module or "")
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }
    rendered = " ".join(imports).lower()
    assert "pentest" not in rendered
    assert "http" not in rendered


def test_performance_modules_are_not_declared_as_core_or_plugin() -> None:
    for module, layer in CURRENT_MODULE_OWNERSHIP.items():
        if "performance" in module or module.endswith("current_hot_path_profile"):
            assert layer is CurrentArchitectureLayer.PERFORMANCE
