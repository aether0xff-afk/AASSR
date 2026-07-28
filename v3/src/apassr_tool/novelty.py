from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlparse

from .actions import ActionCandidate
from .knowledge import KK


VOLATILE_KEYS = {
    "timestamp", "time", "datetime", "createdat", "updatedat", "requestid",
    "request_id", "traceid", "trace_id", "nonce", "correlationid", "correlation_id",
}
MEANINGFUL_VALUE_KEYS = {
    "authenticated", "authorized", "solved", "success", "status", "state",
    "role", "permission", "permissions", "challenge", "progress", "flag",
}
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.I)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
TOKEN_RE = re.compile(r"^(?:[A-Za-z0-9_-]{20,}|eyJ[A-Za-z0-9_.-]+)$")
ISO_TIME_RE = re.compile(r"^\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:?\d\d)?$")


@dataclass(frozen=True)
class NoveltyPrediction:
    score: float
    signature_count: int
    chain_count: int


@dataclass(frozen=True)
class NoveltyUpdate:
    bonus: float
    signature: str
    chain: str
    response_signature: str
    raw_response_hash: str = ""
    normalized_response_hash: str = ""
    semantic_novelty: int = 0
    repeated_action: bool = False
    repeated_response: bool = False
    meaningful_transition: bool = False


@dataclass
class NoveltyMemory:
    """Run-scoped novelty ledger, optionally persisted by experiment/run id."""

    signature_counts: dict[str, int] = field(default_factory=dict)
    chain_counts: dict[str, int] = field(default_factory=dict)
    response_counts: dict[str, int] = field(default_factory=dict)
    raw_response_counts: dict[str, int] = field(default_factory=dict)
    semantic_facts: set[str] = field(default_factory=set)
    observation_variants: set[str] = field(default_factory=set)
    challenge_attempts: dict[str, list[str]] = field(default_factory=dict)
    solved_challenges: set[str] = field(default_factory=set)
    last_signature: str | None = None
    candidate_weight: float = 0.35
    chain_weight: float = 0.25
    response_weight: float = 0.4
    max_bonus: float = 3.0
    semantic_bonus_cap: int = 5
    persistence_path: str | Path | None = None
    run_id: str = "default"

    def __post_init__(self) -> None:
        self._load_safely()

    def predict(self, candidate: ActionCandidate) -> NoveltyPrediction:
        signature = self.signature(candidate)
        chain = self.chain_signature(signature)
        signature_count = self.signature_counts.get(signature, 0)
        chain_count = self.chain_counts.get(chain, 0)
        score = self.candidate_weight / (1.0 + signature_count)
        score += self.chain_weight / (1.0 + chain_count)
        return NoveltyPrediction(score=score, signature_count=signature_count, chain_count=chain_count)

    def update(
        self,
        candidate: ActionCandidate,
        *,
        status: int,
        new_kv: int = 0,
        solved_delta: int = 0,
        response_body: str = "",
        semantic_items: Iterable[tuple[KK, str]] = (),
        challenge_keys: Iterable[str] = (),
    ) -> NoveltyUpdate:
        signature = self.signature(candidate)
        chain = self.chain_signature(signature)
        prediction = self.predict(candidate)
        raw_hash = _hash_text(response_body)
        normalized_hash = _hash_text(normalize_response(response_body))
        response_signature = f"{signature}:status={_bucket_status(status)}:body={normalized_hash}"
        response_count = self.response_counts.get(response_signature, 0)
        repeated_action = self.signature_counts.get(signature, 0) > 0
        repeated_response = response_count > 0

        new_facts = 0
        for fact in semantic_facts(candidate, status=status, items=semantic_items):
            if fact not in self.semantic_facts:
                self.semantic_facts.add(fact)
                new_facts += 1
        variant = f"{signature}:status={status}:response={normalized_hash}"
        meaningful_transition = bool(response_body) and variant not in self.observation_variants and repeated_action
        self.observation_variants.add(variant)

        for key in challenge_keys:
            attempts = self.challenge_attempts.setdefault(str(key), [])
            attempts.append(f"{signature}:status={status}:solved={solved_delta > 0}")
            if solved_delta > 0:
                self.solved_challenges.add(str(key))

        response_novelty = 0.0 if repeated_response else self.response_weight
        bonus = prediction.score + response_novelty + min(new_facts, self.semantic_bonus_cap)
        if repeated_action and repeated_response and new_facts == 0:
            bonus = 0.0
        bonus = min(self.max_bonus, bonus)

        self.signature_counts[signature] = self.signature_counts.get(signature, 0) + 1
        self.chain_counts[chain] = self.chain_counts.get(chain, 0) + 1
        self.response_counts[response_signature] = response_count + 1
        if raw_hash:
            self.raw_response_counts[raw_hash] = self.raw_response_counts.get(raw_hash, 0) + 1
        self.last_signature = signature
        self._save_safely()
        return NoveltyUpdate(
            bonus=bonus,
            signature=signature,
            chain=chain,
            response_signature=response_signature,
            raw_response_hash=raw_hash,
            normalized_response_hash=normalized_hash,
            semantic_novelty=new_facts,
            repeated_action=repeated_action,
            repeated_response=repeated_response,
            meaningful_transition=meaningful_transition,
        )

    def signature(self, candidate: ActionCandidate) -> str:
        call = candidate.tool_call
        method = (call.method or _method_for_tool(call.tool.value)).upper()
        target = _canonical_url(call.url or candidate.bindings.get(KK.ENDPOINT, "") or candidate.bindings.get(KK.PATH, ""))
        data = ",".join(f"{key}={_value_type(value)}" for key, value in sorted(call.data.items()))
        headers = ",".join(f"{key.lower()}={_value_type(value)}" for key, value in sorted(call.headers.items()))
        if call.target_host:
            target = f"host={call.target_host}:port={_value_type(call.port_range)}"
        return f"{method} {target}|body={data}|headers={headers}"

    def chain_signature(self, signature: str) -> str:
        return f"{self.last_signature or '<start>'} -> {signature}"

    def reset(self, *, delete_persisted: bool = False) -> None:
        self.signature_counts.clear()
        self.chain_counts.clear()
        self.response_counts.clear()
        self.raw_response_counts.clear()
        self.semantic_facts.clear()
        self.observation_variants.clear()
        self.challenge_attempts.clear()
        self.solved_challenges.clear()
        self.last_signature = None
        path = self._run_path()
        if delete_persisted and path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _run_path(self) -> Path | None:
        if self.persistence_path is None:
            return None
        root = Path(self.persistence_path)
        return root / f"{_safe_id(self.run_id)}.json" if root.suffix == "" else root

    def _load_safely(self) -> None:
        path = self._run_path()
        if path is None or not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.signature_counts.update(data.get("signature_counts", {}))
            self.chain_counts.update(data.get("chain_counts", {}))
            self.response_counts.update(data.get("response_counts", {}))
            self.raw_response_counts.update(data.get("raw_response_counts", {}))
            self.semantic_facts.update(data.get("semantic_facts", []))
            self.observation_variants.update(data.get("observation_variants", []))
            self.challenge_attempts.update(data.get("challenge_attempts", {}))
            self.solved_challenges.update(data.get("solved_challenges", []))
            self.last_signature = data.get("last_signature")
        except (OSError, ValueError, TypeError):
            return

    def _save_safely(self) -> None:
        path = self._run_path()
        if path is None:
            return
        payload = {
            "signature_counts": self.signature_counts, "chain_counts": self.chain_counts,
            "response_counts": self.response_counts, "raw_response_counts": self.raw_response_counts,
            "semantic_facts": sorted(self.semantic_facts), "observation_variants": sorted(self.observation_variants),
            "challenge_attempts": self.challenge_attempts, "solved_challenges": sorted(self.solved_challenges),
            "last_signature": self.last_signature,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            temporary.replace(path)
        except OSError:
            return


def semantic_facts(candidate: ActionCandidate, *, status: int, items: Iterable[tuple[KK, str]]) -> set[str]:
    method = (candidate.tool_call.method or _method_for_tool(candidate.tool_call.tool.value)).upper()
    target = _canonical_url(candidate.tool_call.url or "")
    facts = {f"status:{method}:{target}:{_bucket_status(status)}"}
    if target:
        facts.add(f"endpoint:{method}:{target.split('?', 1)[0]}")
    for kk, value in items:
        if kk in {KK.ENDPOINT, KK.PATH, KK.AUTH_PATH, KK.HTTP_METHOD, KK.PARAM_NAME, KK.FLAG, KK.ROLE, KK.SESSION_COOKIE}:
            if kk in {KK.FLAG, KK.SESSION_COOKIE}:
                normalized = "sha256:" + _hash_text(value)
            else:
                normalized = _canonical_url(value) if kk in {KK.ENDPOINT, KK.PATH, KK.AUTH_PATH} else value.lower()
            facts.add(f"{kk.value}:{normalized}")
    return facts


def normalize_response(text: str) -> str:
    body = text.replace("\r\n", "\n").strip()
    if "\n\n" in body and body.lstrip().upper().startswith("HTTP/"):
        body = body.split("\n\n", 1)[1]
    try:
        value = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        body = ISO_TIME_RE.sub("{timestamp}", body)
        body = re.sub(r'(?i)(request[-_ ]?id|nonce|timestamp)(["\s:=]+)[A-Za-z0-9_.:+-]+', r'\1\2{volatile}', body)
        return re.sub(r"\s+", " ", body).strip()
    return json.dumps(_normalize_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_json(value: object, *, key: str = "") -> object:
    if isinstance(value, dict):
        return {str(k): _normalize_json(v, key=str(k)) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize_json(item, key=key) for item in value]
    if key.lower().replace("-", "_") in VOLATILE_KEYS:
        return "{volatile}"
    if isinstance(value, str) and ISO_TIME_RE.match(value):
        return "{timestamp}"
    if key.lower().replace("-", "_") in MEANINGFUL_VALUE_KEYS:
        return value
    if value is None:
        return "{null}"
    if isinstance(value, bool):
        return "{boolean}"
    if isinstance(value, int):
        return "{integer}"
    if isinstance(value, float):
        return "{float}"
    return _value_type(value)


def _canonical_url(value: str) -> str:
    if not value:
        return ""
    parsed = urlparse(value)
    path = parsed.path or (value.split("?", 1)[0] if not parsed.scheme else "/")
    query = parsed.query if parsed.scheme or "?" in value else ""
    pairs = [(key, _value_type(val)) for key, val in parse_qsl(query, keep_blank_values=True)]
    return path + ("?" + urlencode(sorted(pairs), safe="{}") if pairs else "")


def _value_type(value: object) -> str:
    text = str(value).strip()
    lower = text.lower()
    if lower in {"true", "false"}: return "{boolean}"
    if re.fullmatch(r"[-+]?\d+", text): return "{integer}"
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)", text): return "{float}"
    if UUID_RE.match(text): return "{uuid}"
    if EMAIL_RE.match(text): return "{email}"
    if TOKEN_RE.match(text): return "{token}"
    if ISO_TIME_RE.match(text): return "{timestamp}"
    return "{string}"


def _method_for_tool(tool: str) -> str:
    if "HEAD" in tool: return "HEAD"
    if "OPTIONS" in tool: return "OPTIONS"
    if "POST" in tool: return "POST"
    return "GET"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest() if text else ""


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value) or "default"


def _bucket_status(status: int) -> str:
    if status == 0: return "blocked"
    if status < 200: return "1xx"
    if status < 300: return "2xx"
    if status < 400: return "3xx"
    if status < 500: return "4xx"
    return "5xx"
