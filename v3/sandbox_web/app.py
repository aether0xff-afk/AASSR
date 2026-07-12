from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apassr_tool.sandbox_server import main


if __name__ == "__main__":
    host = os.environ.get("SANDBAG_HOST", "0.0.0.0")
    port = int(os.environ.get("SANDBAG_PORT", "8088"))
    main(["--host", host, "--port", str(port)])

