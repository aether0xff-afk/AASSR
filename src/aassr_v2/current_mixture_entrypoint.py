from __future__ import annotations

from .current_entrypoint import build_current_pentest_aassr_core
from .current_manifest import CURRENT_COMPONENTS


# Compatibility surface for repaired/diagnostic runners created before the
# mixture runtime became the canonical current builder. Keep the old names so
# historical scripts remain reproducible without maintaining a second active
# implementation.
MIXTURE_CURRENT_COMPONENTS = dict(CURRENT_COMPONENTS)
build_current_mixture_pentest_aassr_core = build_current_pentest_aassr_core


__all__ = [
    "MIXTURE_CURRENT_COMPONENTS",
    "build_current_mixture_pentest_aassr_core",
]
