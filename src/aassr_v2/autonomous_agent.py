"""Compatibility module for the unified autonomous AASSR core.

The implementation lives in :mod:`aassr_v2.autonomous_agent_core` so all
runners import one Prophecy/Imagination/Policy path while existing imports keep
working.
"""

from .autonomous_agent_core import *  # noqa: F401,F403
