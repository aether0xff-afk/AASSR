from __future__ import annotations

from dataclasses import dataclass, field

from .types import TransitionTrace


@dataclass(slots=True)
class TraceLedger:
    """Append-only ASeq transition history used for causal attribution."""

    _records: list[TransitionTrace] = field(default_factory=list)

    def append(self, trace: TransitionTrace) -> None:
        if any(record.trace_id == trace.trace_id for record in self._records):
            raise ValueError(f"duplicate trace_id: {trace.trace_id}")
        self._records.append(trace)

    def all(self) -> tuple[TransitionTrace, ...]:
        return tuple(self._records)

    def get(self, trace_id: str) -> TransitionTrace | None:
        return next(
            (record for record in self._records if record.trace_id == trace_id),
            None,
        )

    def causes_of_fact(self, fact: str) -> tuple[TransitionTrace, ...]:
        return tuple(record for record in self._records if fact in record.added_facts)
