from __future__ import annotations

"""Portable import surface for the baseline efficiency benchmark.

The benchmark records POSIX peak RSS through the standard ``resource`` module.
Windows does not provide that module, so this shim exposes a zero-valued RSS
fallback before importing the shared benchmark implementation. All performance
experiments run on Linux; the fallback exists so the package and tests remain
portable on Windows.
"""

import sys
import types


if sys.platform == "win32" and "resource" not in sys.modules:
    resource = types.ModuleType("resource")
    resource.RUSAGE_SELF = 0

    class _Usage:
        ru_maxrss = 0.0

    def _getrusage(_: int) -> _Usage:
        return _Usage()

    resource.getrusage = _getrusage
    sys.modules["resource"] = resource


from .baseline_efficiency_benchmark import *  # noqa: E402,F401,F403
