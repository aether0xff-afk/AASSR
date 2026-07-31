from __future__ import annotations

import argparse
from pathlib import Path

from aassr_v2.human_study import serve_human_study
from aassr_v2.paper_protocol import load_paper_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the localhost-only anonymous AASSR human study UI."
    )
    parser.add_argument("--config")
    parser.add_argument("--database")
    parser.add_argument("--dataset-version")
    parser.add_argument("--approval-id", default="")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    settings: dict[str, object] = {}
    if args.config:
        config = load_paper_config(args.config)
        raw = config.get("human_study", {})
        if not isinstance(raw, dict):
            parser.error("config does not contain human_study settings")
        settings = dict(raw)
    dataset_version = str(
        args.dataset_version
        or settings.get("dataset_version", "")
    ).strip()
    if not dataset_version:
        parser.error(
            "--dataset-version or human_study.dataset_version is required"
        )
    database = args.database or settings.get("database")
    if not database:
        database = (
            Path("runs")
            / "human_study"
            / f"{dataset_version}.sqlite3"
        )
    host = str(args.host or settings.get("host", "127.0.0.1"))
    port = int(args.port or settings.get("port", 8765))
    approval_id = str(
        args.approval_id or settings.get("approval_id", "")
    )
    server = serve_human_study(
        database=database,
        dataset_version=dataset_version,
        approval_id=approval_id,
        host=host,
        port=port,
    )
    print(f"Human study UI: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
