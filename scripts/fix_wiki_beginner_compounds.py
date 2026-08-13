from __future__ import annotations

from pathlib import Path

WIKI = Path("wiki")

REPLACEMENTS = {
    "same [체크포인트(checkpoint)](Reproduction)": "[같은 체크포인트(same checkpoint)](Experiments)",
    "same [체크포인트](Reproduction)": "[같은 체크포인트(same checkpoint)](Experiments)",
    "same 체크포인트": "[같은 체크포인트(same checkpoint)](Experiments)",
    "Same [체크포인트(checkpoint)](Reproduction)": "[같은 체크포인트(same checkpoint)](Experiments)",
    "Same 체크포인트": "[같은 체크포인트(same checkpoint)](Experiments)",
}


def main() -> None:
    changed = []
    for path in sorted(WIKI.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        updated = text
        for old, new in REPLACEMENTS.items():
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            changed.append(path.as_posix())
    print(f"Compound terminology fixes changed {len(changed)} files")
    for path in changed:
        print(path)


if __name__ == "__main__":
    main()
