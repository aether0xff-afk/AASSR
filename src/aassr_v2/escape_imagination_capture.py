from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from functools import wraps
import json
from pathlib import Path
import threading
from typing import Any, Callable, Iterator, Mapping
from uuid import uuid4

from .escape_reporting import serialize_action, serialize_snapshot
from .imagination_tree import ImaginationResult, ImaginationTree


@dataclass(frozen=True, slots=True)
class EscapeImaginationEvent:
    sequence: int
    timestamp_utc: str
    root_step: int
    root_position: tuple[int, int] | None
    chosen_action: str
    node_count: int
    expanded_nodes: int
    maximum_depth: int
    payload: Mapping[str, Any]


ImaginationCallback = Callable[[EscapeImaginationEvent], None]
_TLS = threading.local()
_INSTALL_LOCK = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def serialize_imagination_result(result: ImaginationResult) -> dict[str, Any]:
    chosen_signature = result.chosen_action.signature
    root = result.nodes[0]
    return {
        "chosen_action": serialize_action(result.chosen_action),
        "expanded_nodes": result.expanded_nodes,
        "maximum_depth_reached": result.maximum_depth_reached,
        "root_state": serialize_snapshot(root.state),
        "root_evaluations": [
            {
                "action": serialize_action(item.action),
                "chosen": item.action.signature == chosen_signature,
                "leaf_values": list(item.leaf_values),
                "aggregate_value": item.aggregate_value,
                "best_path": list(item.best_path),
                "best_leaf_id": item.best_leaf_id,
            }
            for item in result.root_evaluations
        ],
        "nodes": [
            {
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "depth": node.depth,
                "state": serialize_snapshot(node.state),
                "root_action": (
                    serialize_action(node.root_action) if node.root_action is not None else None
                ),
                "action_from_parent": (
                    serialize_action(node.action_from_parent)
                    if node.action_from_parent is not None
                    else None
                ),
                "state_path": list(node.state_path),
                "action_path": list(node.action_path),
                "cumulative_value": node.cumulative_value,
                "step_confidence": node.step_confidence,
                "cumulative_confidence": node.cumulative_confidence,
                "policy_memory": _json_safe(node.policy_memory),
                "prophecy_memory": _json_safe(node.prophecy_memory),
                "terminal_reason": node.terminal_reason,
            }
            for node in result.nodes
        ],
    }


def _emit_result(result: ImaginationResult) -> None:
    sink = getattr(_TLS, "sink", None)
    if sink is not None:
        sink(result)


def install_imagination_instrumentation() -> None:
    """Instrument ImaginationTree once while keeping capture thread-local."""

    with _INSTALL_LOCK:
        current = ImaginationTree.plan
        if getattr(current, "_aassr_imagination_instrumented", False):
            return
        original = current

        @wraps(original)
        def instrumented(self: ImaginationTree, *args: Any, **kwargs: Any) -> ImaginationResult:
            result = original(self, *args, **kwargs)
            _emit_result(result)
            return result

        setattr(instrumented, "_aassr_imagination_instrumented", True)
        ImaginationTree.plan = instrumented  # type: ignore[method-assign]


@contextmanager
def capture_imaginations(
    sink: Callable[[ImaginationResult], None] | None,
) -> Iterator[None]:
    install_imagination_instrumentation()
    previous = getattr(_TLS, "sink", None)
    _TLS.sink = sink
    try:
        yield
    finally:
        _TLS.sink = previous


def default_visualized_output_dir(
    base_dir: str | Path = "runs/escape_gridworld",
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(base_dir) / f"{timestamp}_imagination_{uuid4().hex[:8]}"


class ImaginationEventStream:
    """Flush every complete imagination tree to JSONL before GUI delivery."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        callback: ImaginationCallback | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "imaginations.jsonl"
        self.summary_path = self.output_dir / "imagination_summary.json"
        self._handle = self.path.open("w", encoding="utf-8", newline="\n")
        self._callback = callback
        self._sequence = 0
        self._total_nodes = 0
        self._maximum_depth = 0
        self._closed = False

    @property
    def count(self) -> int:
        return self._sequence

    def record(self, result: ImaginationResult) -> None:
        if self._closed:
            raise RuntimeError("imagination stream is closed")
        self._sequence += 1
        serialized = serialize_imagination_result(result)
        root_metadata = result.nodes[0].state.metadata
        raw_position = root_metadata.get("position")
        position = (
            (int(raw_position[0]), int(raw_position[1]))
            if isinstance(raw_position, (tuple, list)) and len(raw_position) == 2
            else None
        )
        root_step = int(root_metadata.get("steps", 0))
        event_payload = {
            "sequence": self._sequence,
            "timestamp_utc": _utc_now_iso(),
            "root_step": root_step,
            "root_position": list(position) if position is not None else None,
            **serialized,
        }
        self._handle.write(
            json.dumps(event_payload, ensure_ascii=False, sort_keys=True) + "\n"
        )
        self._handle.flush()
        self._total_nodes += len(result.nodes)
        self._maximum_depth = max(self._maximum_depth, result.maximum_depth_reached)
        event = EscapeImaginationEvent(
            sequence=self._sequence,
            timestamp_utc=str(event_payload["timestamp_utc"]),
            root_step=root_step,
            root_position=position,
            chosen_action=result.chosen_action.signature,
            node_count=len(result.nodes),
            expanded_nodes=result.expanded_nodes,
            maximum_depth=result.maximum_depth_reached,
            payload=event_payload,
        )
        if self._callback is not None:
            try:
                self._callback(event)
            except Exception:
                # GUI delivery must never destroy a research run after durable capture.
                pass

    def close(self) -> None:
        if self._closed:
            return
        self._handle.flush()
        self._handle.close()
        summary = {
            "events": self._sequence,
            "total_nodes": self._total_nodes,
            "mean_nodes": (
                self._total_nodes / self._sequence if self._sequence else 0.0
            ),
            "maximum_depth": self._maximum_depth,
            "imaginations_jsonl": str(self.path),
        }
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self._closed = True

    def __enter__(self) -> ImaginationEventStream:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
