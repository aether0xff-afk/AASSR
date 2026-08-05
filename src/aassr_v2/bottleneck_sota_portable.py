from __future__ import annotations

"""Portable import surface for the bottleneck/SOTA diagnostic.

Importing the baseline portability shim first installs the Windows fallback for
``resource``. The actual experiments still run on Linux so peak RSS remains a
real measurement there.
"""

from . import baseline_efficiency_portable as _baseline_portable  # noqa: F401
from .bottleneck_sota_diagnostic import *  # noqa: F401,F403,E402
from .bottleneck_sota_factory import (  # noqa: E402,F401
    make_bottleneck_agent,
    run_bottleneck_condition,
)
