from __future__ import annotations

import argparse
import ast
from collections import deque
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys


FORBIDDEN_IMPORT_PARTS = (
    "pentest",
    "plugins.",
    ".plugins",
    "current_pentest",
    "action_plugins",
)
FORBIDDEN_CORE_TOKENS = (
    "http",
    "route",
    "profile",
    "csrf",
    "pentest",
    "transferdiagnosticworld",
)
FORBIDDEN_TRANSITIVE_MODULE_PARTS = (
    "pentest",
    "gridworld",
    "minecraft",
    "sandbox",
    "benchmark",
    "curriculum",
    ".plugins",
    "action_plugins",
)


@dataclass(frozen=True, slots=True)
class Violation:
    path: Path
    line: int
    kind: str
    detail: str


def _imports(tree: ast.AST) -> list[tuple[int, ast.AST]]:
    rows: list[tuple[int, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            rows.append((node.lineno, node))
    return rows


def _module_name(package_root: Path, path: Path) -> str:
    relative = path.relative_to(package_root.parent).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _module_path(package_root: Path, module: str) -> Path | None:
    if module == package_root.name:
        candidate = package_root / "__init__.py"
        return candidate if candidate.is_file() else None
    prefix = package_root.name + "."
    if not module.startswith(prefix):
        return None
    parts = module[len(prefix) :].split(".")
    file_candidate = package_root.joinpath(*parts).with_suffix(".py")
    if file_candidate.is_file():
        return file_candidate
    package_candidate = package_root.joinpath(*parts, "__init__.py")
    if package_candidate.is_file():
        return package_candidate
    return None


def _resolved_modules(current_module: str, node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(item.name for item in node.names)
    if not isinstance(node, ast.ImportFrom):
        return ()
    module = node.module or ""
    if node.level:
        package = current_module.rsplit(".", 1)[0]
        try:
            resolved = importlib.util.resolve_name(
                "." * node.level + module,
                package,
            )
        except (ImportError, ValueError):
            return ()
        return (resolved,)
    return (module,)


def audit_core(root: Path) -> tuple[Violation, ...]:
    violations: list[Violation] = []
    package_root = root.parent

    core_files = tuple(sorted(root.rglob("*.py")))
    for path in core_files:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))

        for line, node in _imports(tree):
            for module in _resolved_modules(_module_name(package_root, path), node):
                lowered = module.lower()
                if any(part in lowered for part in FORBIDDEN_IMPORT_PARTS):
                    violations.append(Violation(path, line, "import", module))

        lowered = text.lower()
        for token in FORBIDDEN_CORE_TOKENS:
            if token in lowered:
                line = lowered[: lowered.index(token)].count("\n") + 1
                violations.append(Violation(path, line, "domain-token", token))

    queue: deque[Path] = deque(core_files)
    visited: set[Path] = set()
    while queue:
        path = queue.popleft()
        if path in visited:
            continue
        visited.add(path)
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        current_module = _module_name(package_root, path)
        for line, node in _imports(tree):
            for module in _resolved_modules(current_module, node):
                if not module.startswith(package_root.name + "."):
                    continue
                lowered = module.lower()
                tail = lowered.split(".")[-1]
                if any(part in lowered for part in FORBIDDEN_TRANSITIVE_MODULE_PARTS):
                    violations.append(
                        Violation(path, line, "transitive-domain-import", module)
                    )
                    continue
                if tail.startswith("current_") and not module.startswith(
                    f"{package_root.name}.core."
                ):
                    violations.append(
                        Violation(path, line, "legacy-current-import", module)
                    )
                    continue
                dependency = _module_path(package_root, module)
                if dependency is not None and dependency not in visited:
                    queue.append(dependency)

    return tuple(violations)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail when the AASSR Core depends on an environment plugin.",
    )
    parser.add_argument("--core-dir", default="src/aassr_v2/core")
    args = parser.parse_args()
    root = Path(args.core_dir)
    if not root.is_dir():
        print(f"core directory not found: {root}", file=sys.stderr)
        return 2

    violations = audit_core(root)
    if violations:
        for item in violations:
            print(
                f"{item.path}:{item.line}: {item.kind}: {item.detail}",
                file=sys.stderr,
            )
        return 1

    print(
        "core boundary audit passed: "
        f"{len(tuple(root.rglob('*.py')))} core files, transitive graph clean"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
