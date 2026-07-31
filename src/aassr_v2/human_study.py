from __future__ import annotations

import csv
import hashlib
import json
import random
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from .creativity import (
    MultiSolutionDependencyWorld,
    strategy_record_from_trace,
)
from .paper_types import StrategyRecord
from .types import Action


SCORE_FIELDS = ("novelty", "utility", "coherence", "surprise")
PROHIBITED_IDENTITY_FIELDS = {
    "name",
    "email",
    "phone",
    "address",
    "payment",
    "ip",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _blind_label(strategy_id: str, dataset_version: str) -> str:
    return hashlib.sha256(
        f"{dataset_version}:{strategy_id}".encode("utf-8")
    ).hexdigest()[:12]


class HumanStudyStore:
    """SQLite-backed anonymous path and blind-rating store."""

    def __init__(
        self,
        path: str | Path,
        *,
        dataset_version: str,
        approval_id: str = "",
    ) -> None:
        if not dataset_version.strip():
            raise ValueError("dataset_version is required")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset_version = dataset_version
        self.approval_id = approval_id.strip()
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS participants (
                    participant_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS strategies (
                    strategy_id TEXT PRIMARY KEY,
                    participant_id TEXT,
                    source_kind TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    FOREIGN KEY(participant_id) REFERENCES participants(participant_id)
                );
                CREATE TABLE IF NOT EXISTS ratings (
                    evaluator_id TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    novelty INTEGER NOT NULL,
                    utility INTEGER NOT NULL,
                    coherence INTEGER NOT NULL,
                    surprise INTEGER NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    PRIMARY KEY(evaluator_id, strategy_id),
                    FOREIGN KEY(evaluator_id) REFERENCES participants(participant_id),
                    FOREIGN KEY(strategy_id) REFERENCES strategies(strategy_id)
                );
                CREATE TABLE IF NOT EXISTS study_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS world_sessions (
                    session_id TEXT PRIMARY KEY,
                    participant_id TEXT NOT NULL,
                    world_seed INTEGER NOT NULL,
                    variant INTEGER NOT NULL,
                    trace_json TEXT NOT NULL,
                    completed INTEGER NOT NULL DEFAULT 0,
                    strategy_id TEXT,
                    updated_at_utc TEXT NOT NULL,
                    FOREIGN KEY(participant_id) REFERENCES participants(participant_id),
                    FOREIGN KEY(strategy_id) REFERENCES strategies(strategy_id)
                );
                """
            )
            existing = {
                str(row["key"]): str(row["value"])
                for row in connection.execute(
                    "SELECT key, value FROM study_metadata"
                )
            }
            expected = {
                "dataset_version": self.dataset_version,
                "approval_id": self.approval_id,
            }
            for key, value in expected.items():
                if key in existing and existing[key] != value:
                    raise ValueError(
                        f"study database {key} does not match requested value"
                    )
                connection.execute(
                    "INSERT OR IGNORE INTO study_metadata VALUES (?, ?)",
                    (key, value),
                )

    def create_participant(self) -> str:
        participant_id = f"p_{uuid4().hex}"
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO participants VALUES (?, ?)",
                (participant_id, _utc_now()),
            )
        return participant_id

    def participant_exists(self, participant_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM participants WHERE participant_id=?",
                (participant_id,),
            ).fetchone()
        return row is not None

    def add_strategy(
        self,
        record: StrategyRecord,
        *,
        participant_id: str | None = None,
    ) -> None:
        if participant_id is not None and not self.participant_exists(participant_id):
            raise ValueError("unknown participant_id")
        encoded = json.dumps(
            record.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO strategies
                    (strategy_id, participant_id, source_kind, record_json, created_at_utc)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.strategy_id,
                    participant_id,
                    record.source_kind,
                    encoded,
                    _utc_now(),
                ),
            )

    def next_assignment(
        self, evaluator_id: str
    ) -> dict[str, Any] | None:
        if not self.participant_exists(evaluator_id):
            raise ValueError("unknown evaluator_id")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT strategy_id, record_json
                FROM strategies
                WHERE (participant_id IS NULL OR participant_id != ?)
                  AND strategy_id NOT IN (
                    SELECT strategy_id FROM ratings WHERE evaluator_id=?
                  )
                ORDER BY strategy_id
                """,
                (evaluator_id, evaluator_id),
            ).fetchall()
        if not rows:
            return None
        randomizer = random.Random(
            hashlib.sha256(
                f"{self.dataset_version}:{evaluator_id}".encode("utf-8")
            ).digest()
        )
        row = rows[randomizer.randrange(len(rows))]
        record = StrategyRecord.from_dict(json.loads(row["record_json"]))
        graph = record.graph.to_dict()
        graph.pop("solution_family", None)
        return {
            "blind_id": _blind_label(record.strategy_id, self.dataset_version),
            "success": record.success,
            "primitive_steps": record.primitive_steps,
            "errors": record.errors,
            "resources_used": record.resources_used,
            "risk_entries": record.risk_entries,
            "graph": graph,
        }

    def _strategy_for_blind_id(self, blind_id: str) -> str:
        with self._connect() as connection:
            identifiers = [
                str(row["strategy_id"])
                for row in connection.execute(
                    "SELECT strategy_id FROM strategies"
                )
            ]
        matches = [
            strategy_id
            for strategy_id in identifiers
            if _blind_label(strategy_id, self.dataset_version) == blind_id
        ]
        if len(matches) != 1:
            raise ValueError("unknown blind strategy id")
        return matches[0]

    def add_rating(
        self,
        evaluator_id: str,
        blind_id: str,
        scores: Mapping[str, Any],
    ) -> None:
        if not self.participant_exists(evaluator_id):
            raise ValueError("unknown evaluator_id")
        values = {field: int(scores[field]) for field in SCORE_FIELDS}
        if any(value < 1 or value > 5 for value in values.values()):
            raise ValueError("all ratings must be integers from 1 through 5")
        strategy_id = self._strategy_for_blind_id(blind_id)
        with self._lock, self._connect() as connection:
            owner = connection.execute(
                "SELECT participant_id FROM strategies WHERE strategy_id=?",
                (strategy_id,),
            ).fetchone()
            if owner is None:
                raise ValueError("unknown strategy_id")
            if owner["participant_id"] == evaluator_id:
                raise ValueError("participants cannot rate their own strategy")
            try:
                connection.execute(
                    """
                    INSERT INTO ratings VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        evaluator_id,
                        strategy_id,
                        values["novelty"],
                        values["utility"],
                        values["coherence"],
                        values["surprise"],
                        _utc_now(),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("duplicate rating") from error

    def export(self, directory: str | Path) -> tuple[Path, Path, Path]:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        strategies_path = target / "human_paths.jsonl"
        ratings_path = target / "human_ratings.csv"
        metadata_path = target / "human_dataset.json"
        with self._connect() as connection:
            strategies = connection.execute(
                "SELECT strategy_id, participant_id, record_json FROM strategies"
            ).fetchall()
            ratings = connection.execute(
                "SELECT * FROM ratings ORDER BY strategy_id, evaluator_id"
            ).fetchall()
        with strategies_path.open("w", encoding="utf-8") as handle:
            for row in strategies:
                payload = json.loads(row["record_json"])
                payload["participant_id"] = row["participant_id"]
                handle.write(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
                )
        with ratings_path.open("w", newline="", encoding="utf-8-sig") as handle:
            fields = (
                "evaluator_id",
                "strategy_id",
                *SCORE_FIELDS,
                "created_at_utc",
            )
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(dict(row) for row in ratings)
        metadata = {
            "dataset_version": self.dataset_version,
            "approval_id": self.approval_id,
            "participant_count": len(
                {row["participant_id"] for row in strategies if row["participant_id"]}
            ),
            "strategy_count": len(strategies),
            "rating_count": len(ratings),
            "contains_direct_identifiers": False,
            "exported_at_utc": _utc_now(),
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return strategies_path, ratings_path, metadata_path

    def ratings(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM ratings ORDER BY strategy_id, evaluator_id"
                )
            ]

    def create_world_session(
        self,
        participant_id: str,
        *,
        seed: int,
        variant: int,
    ) -> str:
        if not self.participant_exists(participant_id):
            raise ValueError("unknown participant_id")
        session_id = f"w_{uuid4().hex}"
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO world_sessions
                    (session_id, participant_id, world_seed, variant,
                     trace_json, completed, strategy_id, updated_at_utc)
                VALUES (?, ?, ?, ?, ?, 0, NULL, ?)
                """,
                (
                    session_id,
                    participant_id,
                    int(seed),
                    int(variant),
                    "[]",
                    _utc_now(),
                ),
            )
        return session_id

    def world_session(
        self, session_id: str, *, participant_id: str | None = None
    ) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM world_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise ValueError("unknown world session")
        if (
            participant_id is not None
            and str(row["participant_id"]) != participant_id
        ):
            raise ValueError("world session does not belong to participant")
        result = dict(row)
        result["trace"] = json.loads(str(result.pop("trace_json")))
        result["completed"] = bool(result["completed"])
        return result

    def update_world_session(
        self,
        session_id: str,
        *,
        trace: Sequence[Mapping[str, Any]],
        completed: bool,
        strategy_id: str | None = None,
    ) -> None:
        with self._lock, self._connect() as connection:
            changed = connection.execute(
                """
                UPDATE world_sessions
                SET trace_json=?, completed=?, strategy_id=?,
                    updated_at_utc=?
                WHERE session_id=?
                """,
                (
                    json.dumps(
                        [dict(item) for item in trace],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    int(completed),
                    strategy_id,
                    _utc_now(),
                    session_id,
                ),
            ).rowcount
        if changed != 1:
            raise ValueError("unknown world session")


def validate_human_merge(settings: Mapping[str, Any]) -> None:
    if not bool(settings.get("merge_enabled", False)):
        return
    if not str(settings.get("approval_id", "")).strip():
        raise ValueError("human result merge requires an approval_id")
    if not str(settings.get("dataset_version", "")).strip():
        raise ValueError("human result merge requires a dataset_version")
    if int(settings.get("minimum_raters", 2)) < 2:
        raise ValueError("human result merge requires at least two raters")


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left, right, strict=True)
    )
    left_scale = sum((item - left_mean) ** 2 for item in left) ** 0.5
    right_scale = sum((item - right_mean) ** 2 for item in right) ** 0.5
    return numerator / (left_scale * right_scale) if left_scale and right_scale else 0.0


def inter_rater_agreement(
    ratings: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    evaluators = sorted({str(row["evaluator_id"]) for row in ratings})
    by_evaluator = {
        evaluator: {
            str(row["strategy_id"]): row
            for row in ratings
            if str(row["evaluator_id"]) == evaluator
        }
        for evaluator in evaluators
    }
    correlations: list[float] = []
    absolute_differences: list[float] = []
    for index, left_id in enumerate(evaluators):
        for right_id in evaluators[index + 1 :]:
            common = sorted(
                set(by_evaluator[left_id]) & set(by_evaluator[right_id])
            )
            for field in SCORE_FIELDS:
                left = [
                    float(by_evaluator[left_id][key][field]) for key in common
                ]
                right = [
                    float(by_evaluator[right_id][key][field]) for key in common
                ]
                if len(common) >= 2:
                    correlations.append(_pearson(left, right))
                absolute_differences.extend(
                    abs(a - b) for a, b in zip(left, right, strict=True)
                )
    return {
        "evaluator_count": len(evaluators),
        "pairwise_correlation": (
            fmean(correlations) if correlations else 0.0
        ),
        "mean_absolute_difference": (
            fmean(absolute_differences) if absolute_differences else 0.0
        ),
    }


def human_automatic_concordance(
    records: Sequence[StrategyRecord],
    ratings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    automatic = {
        item.strategy_id: float(item.novelty_score or 0.0) for item in records
    }
    human: dict[str, list[float]] = {}
    for row in ratings:
        human.setdefault(str(row["strategy_id"]), []).append(
            float(row["novelty"]) / 5.0
        )
    common = sorted(set(automatic) & set(human))
    automatic_values = [automatic[key] for key in common]
    human_values = [fmean(human[key]) for key in common]
    conflicts = [
        key
        for key, automatic_value, human_value in zip(
            common, automatic_values, human_values, strict=True
        )
        if abs(automatic_value - human_value) >= 0.4
    ]
    return {
        "strategy_count": len(common),
        "correlation": _pearson(automatic_values, human_values),
        "conflict_strategy_ids": conflicts,
    }


@dataclass(slots=True)
class _LiveWorld:
    participant_id: str
    world: MultiSolutionDependencyWorld
    events: list[Mapping[str, Any]]
    world_seed: int
    trace: list[Mapping[str, Any]]
    steps: int = 0
    errors: int = 0
    risk_entries: int = 0


class HumanStudyService:
    def __init__(self, store: HumanStudyStore) -> None:
        self.store = store
        self.live_worlds: dict[str, _LiveWorld] = {}
        self._lock = threading.RLock()

    def create_world(self, participant_id: str, *, seed: int) -> dict[str, Any]:
        if not self.store.participant_exists(participant_id):
            raise ValueError("unknown participant_id")
        variant = seed % 7
        session_id = self.store.create_world_session(
            participant_id, seed=seed, variant=variant
        )
        world = MultiSolutionDependencyWorld(seed=seed, variant=variant)
        with self._lock:
            self.live_worlds[session_id] = _LiveWorld(
                participant_id, world, [], seed, []
            )
        return self._public_world(session_id, world)

    def _load_world(
        self,
        session_id: str,
        *,
        participant_id: str | None = None,
    ) -> _LiveWorld:
        session = self.live_worlds.get(session_id)
        if session is not None:
            if (
                participant_id is not None
                and session.participant_id != participant_id
            ):
                raise ValueError(
                    "world session does not belong to participant"
                )
            return session
        stored = self.store.world_session(
            session_id, participant_id=participant_id
        )
        world = MultiSolutionDependencyWorld(
            seed=int(stored["world_seed"]),
            variant=int(stored["variant"]),
        )
        events: list[Mapping[str, Any]] = []
        errors = 0
        risk_entries = 0
        trace = [
            dict(item)
            for item in stored["trace"]
            if isinstance(item, Mapping)
        ]
        for item in trace:
            if world.terminal:
                break
            snapshot = world.snapshot()
            action = next(
                (
                    candidate
                    for candidate in snapshot.available_actions
                    if candidate.signature == str(item.get("action", ""))
                ),
                Action(str(item.get("action", "")).split("|", 1)[0]),
            )
            outcome = world.step(action)
            events.extend(outcome.effect_events)
            errors += int(outcome.error)
            risk_entries += int(outcome.risk_delta > 0.0)
        if bool(stored["completed"]) != world.terminal:
            raise RuntimeError("stored world completion state is inconsistent")
        session = _LiveWorld(
            str(stored["participant_id"]),
            world,
            events,
            int(stored["world_seed"]),
            trace,
            len(trace),
            errors,
            risk_entries,
        )
        self.live_worlds[session_id] = session
        return session

    def resume_world(
        self, session_id: str, *, participant_id: str
    ) -> dict[str, Any]:
        with self._lock:
            session = self._load_world(
                session_id, participant_id=participant_id
            )
            payload = self._public_world(session_id, session.world)
            stored = self.store.world_session(
                session_id, participant_id=participant_id
            )
            payload["completed"] = bool(stored["completed"])
            if stored.get("strategy_id"):
                payload["strategy_id"] = stored["strategy_id"]
            return payload

    def _public_world(
        self, session_id: str, world: MultiSolutionDependencyWorld
    ) -> dict[str, Any]:
        snapshot = world.snapshot()
        descriptions = world.primitive_action_descriptions()
        return {
            "session_id": session_id,
            "terminal": world.terminal,
            "observations": sorted(snapshot.facts),
            "actions": [
                {
                    "action": action.signature,
                    "description": descriptions[action.signature],
                }
                for action in snapshot.available_actions
            ],
            "goal_progress": snapshot.goal_progress,
        }

    def step_world(
        self,
        session_id: str,
        action_signature: str,
        *,
        participant_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            session = self._load_world(
                session_id, participant_id=participant_id
            )
            if session.world.terminal:
                raise ValueError("world session is already complete")
            snapshot = session.world.snapshot()
            action = next(
                (
                    item
                    for item in snapshot.available_actions
                    if item.signature == action_signature
                ),
                Action("invalid"),
            )
            outcome = session.world.step(action)
            session.steps += 1
            session.errors += int(outcome.error)
            session.risk_entries += int(outcome.risk_delta > 0.0)
            session.events.extend(outcome.effect_events)
            trace_item = {
                "step": session.steps,
                "action": action.signature,
                "before": {
                    "vector": list(snapshot.vector),
                    "facts": sorted(snapshot.facts),
                    "available_actions": [
                        item.signature
                        for item in snapshot.available_actions
                    ],
                    "goal_progress": snapshot.goal_progress,
                },
                "after": {
                    "vector": list(outcome.snapshot.vector),
                    "facts": sorted(outcome.snapshot.facts),
                    "available_actions": [
                        item.signature
                        for item in outcome.snapshot.available_actions
                    ],
                    "goal_progress": outcome.snapshot.goal_progress,
                },
                "error": outcome.error,
                "reward": outcome.reward,
                "resource_cost": outcome.resource_cost,
                "risk_delta": outcome.risk_delta,
                "effect_events": [
                    dict(item) for item in outcome.effect_events
                ],
            }
            session.trace.append(trace_item)
            payload = self._public_world(session_id, session.world)
            payload["error"] = outcome.error
            if session.world.terminal:
                payload["completed"] = True
                strategy_id = f"human_{uuid4().hex}"
                record = strategy_record_from_trace(
                    strategy_id=strategy_id,
                    source_kind="human",
                    research_seed=0,
                    world_seed=session.world_seed,
                    success=True,
                    primitive_steps=session.steps,
                    errors=session.errors,
                    resources_used=session.world.analysis_resource_total,
                    risk_entries=session.risk_entries,
                    events=session.events,
                    solution_family=session.world.analysis_solution_family,
                    trace=session.trace,
                )
                self.store.add_strategy(
                    record, participant_id=session.participant_id
                )
                payload["strategy_id"] = strategy_id
                self.store.update_world_session(
                    session_id,
                    trace=session.trace,
                    completed=True,
                    strategy_id=strategy_id,
                )
            else:
                self.store.update_world_session(
                    session_id,
                    trace=session.trace,
                    completed=False,
                )
            return payload

    def export_dataset(self) -> dict[str, Any]:
        paths = self.store.export(self.store.path.parent / "export")
        return {
            "files": [path.name for path in paths],
            "directory": str(paths[0].parent),
        }


_INDEX_HTML = """<!doctype html>
<html lang="en"><meta charset="utf-8">
<title>AASSR blind strategy study</title>
<style>body{font:16px system-ui;max-width:800px;margin:40px auto;padding:0 16px}
button{margin:6px;padding:10px}fieldset{margin:18px 0}pre{white-space:pre-wrap}</style>
<h1>AASSR blind strategy study</h1>
<p>This local tool records an anonymous ID, action trace, and blind ratings only.
Do not enter your name, email, address, payment, or contact information.</p>
<button id="start">Create anonymous session</button>
<button id="world" disabled>Start strategy task</button>
<button id="rate" disabled>Load blind rating</button>
<button id="export">Export local dataset</button>
<div id="actions"></div><pre id="status"></pre>
<fieldset id="rating" hidden><legend>Blind strategy rating (1–5)</legend>
<label>Novelty <input id="novelty" type="number" min="1" max="5" value="3"></label>
<label>Utility <input id="utility" type="number" min="1" max="5" value="3"></label>
<label>Coherence <input id="coherence" type="number" min="1" max="5" value="3"></label>
<label>Surprise <input id="surprise" type="number" min="1" max="5" value="3"></label>
<button id="submit-rating">Submit blind rating</button></fieldset>
<script>
let participant=localStorage.getItem("aassr_participant")||"";
let world=localStorage.getItem("aassr_world")||"", blind="";
const status=x=>document.querySelector("#status").textContent=JSON.stringify(x,null,2);
function enable(){document.querySelector("#world").disabled=!participant;document.querySelector("#rate").disabled=!participant}
document.querySelector("#start").onclick=async()=>{let r=await fetch("/api/participant",{method:"POST",headers:{"content-type":"application/json"},body:"{}"});let x=await r.json();participant=x.participant_id;localStorage.setItem("aassr_participant",participant);enable();status(x)};
document.querySelector("#world").onclick=async()=>{let r=await fetch("/api/world",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({participant_id:participant,seed:Date.now()%100000})});render(await r.json())};
document.querySelector("#rate").onclick=async()=>{let r=await fetch("/api/assignment?participant_id="+encodeURIComponent(participant));let x=await r.json();if(x.assignment){blind=x.assignment.blind_id;document.querySelector("#rating").hidden=false}status(x)};
document.querySelector("#submit-rating").onclick=async()=>{let body={evaluator_id:participant,blind_id:blind};for(const k of ["novelty","utility","coherence","surprise"])body[k]=Number(document.querySelector("#"+k).value);let r=await fetch("/api/rating",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(body)});let x=await r.json();document.querySelector("#rating").hidden=true;status(x)};
document.querySelector("#export").onclick=async()=>{let r=await fetch("/api/export");status(await r.json())};
async function step(action){let r=await fetch("/api/step",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({participant_id:participant,session_id:world,action})});render(await r.json())}
function render(x){world=x.session_id||world;if(world)localStorage.setItem("aassr_world",world);if(x.completed)localStorage.removeItem("aassr_world");status(x);let d=document.querySelector("#actions");d.textContent="";for(const a of x.actions||[]){let b=document.createElement("button");b.textContent=a.description;b.onclick=()=>step(a.action);d.appendChild(b)}}
enable();
if(participant&&world)fetch("/api/world?participant_id="+encodeURIComponent(participant)+"&session_id="+encodeURIComponent(world)).then(r=>r.json()).then(render);
</script></html>"""


def make_human_study_handler(service: HumanStudyService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AASSRHumanStudy/1"

        def _json(self, status: int, payload: Mapping[str, Any]) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_000_000:
                raise ValueError("request body too large")
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            prohibited = PROHIBITED_IDENTITY_FIELDS & {
                str(key).lower() for key in payload
            }
            if prohibited:
                raise ValueError(
                    f"direct identity fields are prohibited: {sorted(prohibited)}"
                )
            return payload

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                encoded = _INDEX_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return
            if parsed.path == "/api/assignment":
                evaluator = parse_qs(parsed.query).get(
                    "participant_id", [""]
                )[0]
                try:
                    assignment = service.store.next_assignment(evaluator)
                    self._json(
                        HTTPStatus.OK,
                        {"assignment": assignment},
                    )
                except ValueError as error:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            if parsed.path == "/api/world":
                query = parse_qs(parsed.query)
                try:
                    result = service.resume_world(
                        query.get("session_id", [""])[0],
                        participant_id=query.get(
                            "participant_id", [""]
                        )[0],
                    )
                    self._json(HTTPStatus.OK, result)
                except (RuntimeError, ValueError) as error:
                    self._json(
                        HTTPStatus.BAD_REQUEST, {"error": str(error)}
                    )
                return
            if parsed.path == "/api/export":
                self._json(HTTPStatus.OK, service.export_dataset())
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._body()
                if self.path == "/api/participant":
                    result = {
                        "participant_id": service.store.create_participant()
                    }
                elif self.path == "/api/world":
                    result = service.create_world(
                        str(payload["participant_id"]),
                        seed=int(payload.get("seed", 0)),
                    )
                elif self.path == "/api/step":
                    result = service.step_world(
                        str(payload["session_id"]),
                        str(payload["action"]),
                        participant_id=str(
                            payload.get("participant_id", "")
                        )
                        or None,
                    )
                elif self.path == "/api/rating":
                    service.store.add_rating(
                        str(payload["evaluator_id"]),
                        str(payload["blind_id"]),
                        payload,
                    )
                    result = {"stored": True}
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                self._json(HTTPStatus.OK, result)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

    return Handler


def serve_human_study(
    *,
    database: str | Path,
    dataset_version: str,
    approval_id: str = "",
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("human study UI must bind to localhost")
    store = HumanStudyStore(
        database,
        dataset_version=dataset_version,
        approval_id=approval_id,
    )
    service = HumanStudyService(store)
    return ThreadingHTTPServer((host, port), make_human_study_handler(service))
