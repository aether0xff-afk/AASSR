from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from koreanize_wiki_prose import TERMS  # noqa: E402
from koreanize_wiki_common_jargon import COMMON_TERMS  # noqa: E402
from koreanize_wiki_remaining_jargon import EXTRA_TERMS  # noqa: E402
from koreanize_wiki_high_frequency_jargon import TERMS as HIGH_FREQUENCY_TERMS  # noqa: E402

WIKI = Path("wiki")
PROTECTED = re.compile(r"(`[^`]*`|!?\[[^\]]*\]\([^\n)]*\))")
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
ALL_TERMS = {**HIGH_FREQUENCY_TERMS, **EXTRA_TERMS, **COMMON_TERMS, **TERMS}
TERMS_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    + "|".join(
        re.escape(term)
        for term in sorted(ALL_TERMS, key=lambda x: (-len(x), x))
    )
    + r")(?![A-Za-z0-9_-])"
)

DEFINED_BY_PAGE_LINK = {
    "Policy": "Policy",
    "Knowledge": "Knowledge",
    "Prophecy": "Prophecy",
    "Calibration": "Calibration",
    "Critic": "Critic",
    "Imagination": "Imagination",
    "Skills": "Skills",
    "Skill": "Skills",
    "ASEQ": "ASEQ",
    "DQN": "Q-Learning-DQN-and-TD",
    "GRU": "GRU-and-Sequence-Models",
    "OOD": "Critic-Support-and-OOD",
    "DreamerV3": "Experiments",
}


def normalized_link_targets(text: str) -> set[str]:
    targets: set[str] = set()
    for href in LINK.findall(text):
        target = href.split("#", 1)[0].split("?", 1)[0].strip()
        if target.endswith(".md"):
            target = target[:-3]
        if target:
            targets.add(target)
    return targets


def visible_plain_text(path: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence or stripped.startswith("#"):
            continue
        parts = PROTECTED.split(line)
        plain = "".join(part for i, part in enumerate(parts) if i % 2 == 0)
        rows.append((lineno, plain))
    return rows


def main() -> int:
    failures: list[str] = []
    checked = 0
    for path in sorted(WIKI.glob("*.md")):
        if path.name == "README.md":
            continue
        checked += 1
        full_text = path.read_text(encoding="utf-8")
        link_targets = normalized_link_targets(full_text)
        page_defined_terms = {
            term
            for term, target in DEFINED_BY_PAGE_LINK.items()
            if target in link_targets
        }

        for lineno, text in visible_plain_text(path):
            matches = sorted(set(m.group(0) for m in TERMS_RE.finditer(text)))
            matches = [term for term in matches if term not in page_defined_terms]
            if matches:
                failures.append(
                    f"{path}:{lineno}: unexplained/unlinked English jargon: "
                    + ", ".join(matches)
                )

    if failures:
        print("Beginner-language wiki lint failed.")
        print("Bare technical English remains in visible prose. Use Korean-first prose,")
        print("or explain/link the term at least once on the page.")
        for item in failures[:500]:
            print(item)
        if len(failures) > 500:
            print(f"... and {len(failures) - 500} more")
        return 1

    print(
        f"Beginner-language wiki lint passed for {checked} published Markdown pages "
        f"across {len(ALL_TERMS)} mapped technical terms."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
