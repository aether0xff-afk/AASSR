from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any


class KK(StrEnum):
    CURRENT_POS = "KK_CURRENT_POS"
    DIRECTION = "KK_DIRECTION"
    SELF = "KK_SELF"
    KNOWN_CELL = "KK_KNOWN_CELL"
    VISITED_CELL = "KK_VISITED_CELL"
    UNKNOWN_NEIGHBOR = "KK_UNKNOWN_NEIGHBOR"
    FRONTIER_CELL = "KK_FRONTIER_CELL"
    WALL_CELL = "KK_WALL_CELL"
    HINT_CELL = "KK_HINT_CELL"
    HINT_VALUE = "KK_HINT_VALUE"
    KEY_CELL = "KK_KEY_CELL"
    KEY_OBJECT = "KK_KEY_OBJECT"
    DOOR_CELL = "KK_DOOR_CELL"
    FLAG_CELL = "KK_FLAG_CELL"


class ValueType(StrEnum):
    CELL_COORD = "CellCoord"
    DIRECTION = "Direction"
    SELF = "Self"
    OBJECT_INSTANCE = "ObjectInstance"
    HINT_TEXT = "HintText"
    HINT_TARGET = "HintTarget"


class KnowledgeSource(StrEnum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    IMAGINED = "imagined"
    PROPHETIC = "prophetic"


class KnowledgeStatus(StrEnum):
    ACTIVE = "active"
    VISITED = "visited"
    BLOCKED = "blocked"
    CONSUMED = "consumed"
    STALE = "stale"


@dataclass(frozen=True)
class KV:
    value: Any
    type: ValueType
    source: KnowledgeSource = KnowledgeSource.OBSERVED
    confidence: float = 1.0
    status: KnowledgeStatus = KnowledgeStatus.ACTIVE
    used_count: int = 0
    success_count: int = 0
    last_updated: int = 0

    def used(self, *, success: bool, step: int) -> KV:
        return replace(
            self,
            used_count=self.used_count + 1,
            success_count=self.success_count + int(success),
            last_updated=step,
        )


@dataclass(frozen=True)
class KnowledgeDelta:
    added: tuple[tuple[KK, KV], ...] = ()
    updated: tuple[tuple[KK, KV], ...] = ()
    status_changed: tuple[tuple[KK, KV], ...] = ()
    removed: tuple[tuple[KK, Any], ...] = ()
    usage_updated: tuple[tuple[KK, KV], ...] = ()

    def changed_kk(self) -> set[KK]:
        return self.semantic_changed_kk()

    def semantic_changed_kk(self) -> set[KK]:
        return {
            kk
            for kk, _ in self.added + self.updated + self.status_changed
        } | {kk for kk, _ in self.removed}

    def information_gain(self) -> int:
        return self.semantic_information_gain()

    def semantic_information_gain(self) -> int:
        ignored = {KK.CURRENT_POS, KK.DIRECTION, KK.SELF}
        semantic_changes = self.added + self.updated + self.status_changed
        return sum(1 for kk, _ in semantic_changes if kk not in ignored)

    def usage_change_count(self) -> int:
        return len(self.usage_updated)

    def has_semantic_changes(self) -> bool:
        return bool(self.added or self.updated or self.status_changed or self.removed)

    def raw_change_count(self) -> int:
        return len(self.added) + len(self.updated) + len(self.status_changed)

    def is_empty(self) -> bool:
        return not (
            self.added
            or self.updated
            or self.status_changed
            or self.removed
            or self.usage_updated
        )


class KnowledgeStore:
    """Stores KV candidates and supplies them for KK slot binding."""

    def __init__(self) -> None:
        self._values: dict[KK, list[KV]] = {kk: [] for kk in KK}

    def add(
        self,
        kk: KK,
        value: Any,
        value_type: ValueType,
        *,
        source: KnowledgeSource = KnowledgeSource.OBSERVED,
        confidence: float = 1.0,
        status: KnowledgeStatus = KnowledgeStatus.ACTIVE,
        step: int = 0,
    ) -> KV:
        kv = KV(
            value=value,
            type=value_type,
            source=source,
            confidence=confidence,
            status=status,
            last_updated=step,
        )
        return self.upsert(kk, kv)

    def set_singleton(
        self,
        kk: KK,
        value: Any,
        value_type: ValueType,
        *,
        source: KnowledgeSource = KnowledgeSource.OBSERVED,
        confidence: float = 1.0,
        status: KnowledgeStatus = KnowledgeStatus.ACTIVE,
        step: int = 0,
    ) -> KV:
        self._values[kk] = []
        return self.add(
            kk,
            value,
            value_type,
            source=source,
            confidence=confidence,
            status=status,
            step=step,
        )

    def upsert(self, kk: KK, kv: KV) -> KV:
        entries = self._values[kk]
        for index, existing in enumerate(entries):
            if existing.value == kv.value:
                merged = replace(
                    kv,
                    used_count=max(existing.used_count, kv.used_count),
                    success_count=max(existing.success_count, kv.success_count),
                )
                entries[index] = merged
                return merged
        entries.append(kv)
        return kv

    def values(
        self,
        kk: KK,
        *,
        include_inactive: bool = False,
        top_k: int | None = None,
    ) -> list[KV]:
        values = self._values[kk]
        if not include_inactive:
            values = [kv for kv in values if kv.status == KnowledgeStatus.ACTIVE]
        values = sorted(
            values,
            key=lambda kv: (kv.used_count, -kv.confidence, -kv.last_updated),
        )
        if top_k is not None:
            values = values[:top_k]
        return list(values)

    def has_active(self, kk: KK) -> bool:
        return bool(self.values(kk))

    def clone(self) -> KnowledgeStore:
        copied = KnowledgeStore()
        copied._values = {kk: list(values) for kk, values in self._values.items()}
        return copied

    def mark(
        self,
        kk: KK,
        value: Any,
        status: KnowledgeStatus,
        *,
        step: int,
    ) -> bool:
        entries = self._values[kk]
        for index, existing in enumerate(entries):
            if existing.value == value:
                entries[index] = replace(existing, status=status, last_updated=step)
                return True
        return False

    def mark_used(self, kk: KK, value: Any, *, success: bool, step: int) -> bool:
        entries = self._values[kk]
        for index, existing in enumerate(entries):
            if existing.value == value:
                entries[index] = existing.used(success=success, step=step)
                return True
        return False

    def remove(self, kk: KK, value: Any) -> bool:
        entries = self._values[kk]
        kept = [kv for kv in entries if kv.value != value]
        self._values[kk] = kept
        return len(kept) != len(entries)

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {
            kk.value: [
                {
                    "value": kv.value,
                    "type": kv.type.value,
                    "source": kv.source.value,
                    "confidence": kv.confidence,
                    "status": kv.status.value,
                    "used_count": kv.used_count,
                    "success_count": kv.success_count,
                    "last_updated": kv.last_updated,
                }
                for kv in values
            ]
            for kk, values in self._values.items()
            if values
        }

    def snapshot_items(self) -> dict[tuple[KK, Any], KV]:
        return {
            (kk, kv.value): kv
            for kk, values in self._values.items()
            for kv in values
        }

    def delta_since(self, before: dict[tuple[KK, Any], KV]) -> KnowledgeDelta:
        after = self.snapshot_items()
        added = []
        updated = []
        status_changed = []
        removed = []
        usage_updated = []

        for key, kv in after.items():
            previous = before.get(key)
            if previous is None:
                added.append((key[0], kv))
            elif previous.status != kv.status:
                status_changed.append((key[0], kv))
            elif not _same_semantic_kv(previous, kv):
                updated.append((key[0], kv))
            elif previous != kv:
                usage_updated.append((key[0], kv))

        for key in before:
            if key not in after:
                removed.append((key[0], key[1]))

        return KnowledgeDelta(
            added=tuple(added),
            updated=tuple(updated),
            status_changed=tuple(status_changed),
            removed=tuple(removed),
            usage_updated=tuple(usage_updated),
        )


def seed_gridworld_knowledge(start: tuple[int, int]) -> KnowledgeStore:
    store = KnowledgeStore()
    store.set_singleton(KK.CURRENT_POS, start, ValueType.CELL_COORD)
    store.add(KK.SELF, "agent", ValueType.SELF)
    for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
        store.add(KK.DIRECTION, direction, ValueType.DIRECTION)
    return store


def _same_semantic_kv(a: KV, b: KV) -> bool:
    return (
        a.value == b.value
        and a.type == b.type
        and a.source == b.source
        and a.confidence == b.confidence
        and a.status == b.status
    )
