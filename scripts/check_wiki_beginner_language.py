from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from koreanize_wiki_prose import TERMS  # noqa: E402
from koreanize_wiki_common_jargon import COMMON_TERMS  # noqa: E402
from koreanize_wiki_remaining_jargon import EXTRA_TERMS  # noqa: E402

WIKI = Path("wiki")
PROTECTED = re.compile(r"(`[^`]*`|!?\[[^\]]*\]\([^\n)]*\))")
ALL_TERMS = {**EXTRA_TERMS, **COMMON_TERMS, **TERMS}
TERMS_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    + "|".join(
        re.escape(term)
        for term in sorted(ALL_TERMS, key=lambda x: (-len(x), x))
    )
    + r")(?![A-Za-z0-9_-])"
)


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
        for lineno, text in visible_plain_text(path):
            matches = sorted(set(m.group(0) for m in TERMS_RE.finditer(text)))
            if matches:
                failures.append(
                    f"{path}:{lineno}: unexplained/unlinked English jargon: "
                    + ", ".join(matches)
                )

    if failures:
        print("Beginner-language wiki lint failed.")
        print("Bare technical English remains in visible prose. Use Korean-first prose,")
        print("or make the English term an explicit Markdown link/inline-code identifier.")
        for item in failures[:400]:
            print(item)
        if len(failures) > 400:
            print(f"... and {len(failures) - 400} more")
        return 1

    print(
        f"Beginner-language wiki lint passed for {checked} published Markdown pages "
        f"across {len(ALL_TERMS)} mapped technical terms."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
