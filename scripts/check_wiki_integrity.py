from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
WIKI = ROOT / "wiki"

LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")

# These strings may legitimately appear in historical pages, but should never
# re-enter the pages that claim to describe the current runtime.
CURRENT_PAGES = {
    "Home.md",
    "AASSR-in-5-Minutes.md",
    "Current-Status.md",
    "Research-Questions.md",
    "Evidence-Matrix.md",
    "Experiments.md",
    "Reproduction.md",
}

STALE_CURRENT_PATTERNS = {
    "agent/imagination-gate-ablation": "current docs must not pin the old research branch",
    "Stochastic Prophecy v3": "current Prophecy is the manifest v5 conditional-mixture ensemble",
    "relational-stochastic-world-model-v3-status-supervised": "superseded current Prophecy contract",
}

REQUIRED_CURRENT_STRINGS = {
    "Current-Status.md": [
        "aassr-current-generation-v2",
        "relational-conditional-mixture-ensemble-v5-status-balanced",
        "semantic-probability-holdout-calibration-v3-status-aware",
        "local-real-training-support-fail-closed-v1",
    ],
    "Evidence-Matrix.md": [
        "relational-conditional-mixture-ensemble-v5-status-balanced",
        "same checkpoint",
    ],
}


def resolve_internal_target(source: Path, href: str) -> Path | None:
    href = href.strip().strip("<>")
    if not href or href.startswith("#"):
        return None
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", href):
        return None
    if href.startswith("mailto:"):
        return None

    path_part = unquote(href.split("#", 1)[0].split("?", 1)[0])
    if not path_part:
        return None

    # GitHub Wiki links commonly omit .md: (Prophecy), (Concept-Index), etc.
    candidate = (source.parent / path_part).resolve()
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".md")
    return candidate


def main() -> int:
    errors: list[str] = []
    pages = sorted(WIKI.glob("*.md"))
    known = {page.resolve() for page in pages}

    if not pages:
        errors.append("wiki/: no Markdown pages found")

    for page in pages:
        text = page.read_text(encoding="utf-8")

        for href in LINK_RE.findall(text):
            target = resolve_internal_target(page, href)
            if target is None:
                continue
            if target not in known and not target.exists():
                errors.append(f"{page.name}: broken internal link -> {href}")

        if page.name in CURRENT_PAGES:
            for pattern, reason in STALE_CURRENT_PATTERNS.items():
                if pattern in text:
                    errors.append(f"{page.name}: stale current marker {pattern!r} ({reason})")

    for name, required in REQUIRED_CURRENT_STRINGS.items():
        path = WIKI / name
        if not path.exists():
            errors.append(f"missing required current page: {name}")
            continue
        text = path.read_text(encoding="utf-8")
        for marker in required:
            if marker not in text:
                errors.append(f"{name}: missing current contract marker {marker!r}")

    workflow = ROOT / ".github" / "workflows" / "sync-github-wiki.yml"
    if workflow.exists():
        workflow_text = workflow.read_text(encoding="utf-8")
        if '"README.md"' not in workflow_text and "README.md" not in workflow_text:
            errors.append("sync-github-wiki.yml: wiki/README.md is not explicitly excluded")
    else:
        errors.append("missing .github/workflows/sync-github-wiki.yml")

    if errors:
        print("Wiki integrity check FAILED:\n")
        for error in errors:
            print(f"- {error}")
        return 1

    published_count = sum(1 for page in pages if page.name != "README.md")
    print(f"Wiki integrity check passed: {len(pages)} source pages, {published_count} published pages.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
