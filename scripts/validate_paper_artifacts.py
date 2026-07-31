from __future__ import annotations

import argparse

from aassr_v2.paper_artifacts import validate_paper_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a complete AASSR paper artifact tree."
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--require-human-merge", action="store_true")
    args = parser.parse_args()
    issues = validate_paper_artifacts(
        args.results, require_human_merge=args.require_human_merge
    )
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}")
        return 1
    print("Paper artifacts are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
