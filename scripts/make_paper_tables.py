from __future__ import annotations

import argparse

from aassr_v2.paper_artifacts import make_paper_tables


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper tables.")
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    for path in make_paper_tables(args.results):
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
