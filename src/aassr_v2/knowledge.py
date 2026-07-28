from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class KnowledgeEntry:
    key: str
    value: Any
    source_trace_id: str
    confidence: float = 1.0
    enabled_action_signatures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class KnowledgeDelta:
    added: tuple[KnowledgeEntry, ...] = ()
    changed: tuple[KnowledgeEntry, ...] = ()
    removed_keys: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.changed or self.removed_keys)


@dataclass(slots=True)
class KnowledgeStore:
    """Explicit knowledge storage with provenance.

    This store intentionally avoids assigning reward to knowledge by itself.
    Reward is computed later from the measurable effect of a KnowledgeDelta.
    """

    _entries: dict[str, KnowledgeEntry] = field(default_factory=dict)

    def get(self, key: str) -> KnowledgeEntry | None:
        return self._entries.get(key)

    def values(self) -> tuple[KnowledgeEntry, ...]:
        return tuple(self._entries.values())

    def apply(
        self,
        entries: Iterable[KnowledgeEntry] = (),
        remove_keys: Iterable[str] = (),
    ) -> KnowledgeDelta:
        added: list[KnowledgeEntry] = []
        changed: list[KnowledgeEntry] = []
        removed: list[str] = []

        for key in remove_keys:
            if key in self._entries:
                del self._entries[key]
                removed.append(key)

        for entry in entries:
            previous = self._entries.get(entry.key)
            self._entries[entry.key] = entry
            if previous is None:
                added.append(entry)
            elif previous != entry:
                changed.append(entry)

        return KnowledgeDelta(tuple(added), tuple(changed), tuple(removed))
