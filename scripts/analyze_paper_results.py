from __future__ import annotations

import argparse

from aassr_v2.paper_statistics import analyze_paper_results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate seed-first AASSR paper statistics."
    )
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    outputs = analyze_paper_results(args.results)
    for name, value in outputs.items():
        print(f"{name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
