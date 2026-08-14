from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from ..types import Action, StateSnapshot


@dataclass(frozen=True, slots=True)
class CoreTransitionOutcome:
    """Environment-neutral result of one real Core action.

    This replaces the legacy ``action_plugins.PluginOutcome`` dependency inside
    the new Core.  It carries only transition mechanics required by the learner;
    no environment-specific meaning or plugin strategy is represented here.
    """

    snapshot: StateSnapshot
    added_facts: frozenset[str] = frozenset()
    removed_facts: frozenset[str] = frozenset()
    unlocked_actions: tuple[Action, ...] = ()
    error: bool = False
    error_code: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw", MappingProxyType(dict(self.raw)))
