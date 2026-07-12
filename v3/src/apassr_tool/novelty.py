from __future__ import annotations

from dataclasses import dataclass, field

from .actions import ActionCandidate
from .knowledge import KK


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


@dataclass
class NoveltyMemory:
    signature_counts: dict[str, int] = field(default_factory=dict)
    chain_counts: dict[str, int] = field(default_factory=dict)
    response_counts: dict[str, int] = field(default_factory=dict)
    last_signature: str | None = None
    candidate_weight: float = 0.35
    chain_weight: float = 0.25
    response_weight: float = 0.4
    max_bonus: float = 3.0

    def predict(self, candidate: ActionCandidate) -> NoveltyPrediction:
        signature = self.signature(candidate)
        chain = self.chain_signature(signature)
        signature_count = self.signature_counts.get(signature, 0)
        chain_count = self.chain_counts.get(chain, 0)
        score = self.candidate_weight / (1.0 + signature_count)
        score += self.chain_weight / (1.0 + chain_count)
        return NoveltyPrediction(
            score=score,
            signature_count=signature_count,
            chain_count=chain_count,
        )

    def update(
        self,
        candidate: ActionCandidate,
        *,
        status: int,
        new_kv: int,
        solved_delta: int,
    ) -> NoveltyUpdate:
        signature = self.signature(candidate)
        chain = self.chain_signature(signature)
        response_signature = f"{candidate.template.value}:status={_bucket_status(status)}:kv={_bucket_kv(new_kv)}:solved={solved_delta > 0}"

        prediction = self.predict(candidate)
        response_count = self.response_counts.get(response_signature, 0)
        bonus = prediction.score + self.response_weight / (1.0 + response_count)
        bonus = min(self.max_bonus, bonus)

        self.signature_counts[signature] = self.signature_counts.get(signature, 0) + 1
        self.chain_counts[chain] = self.chain_counts.get(chain, 0) + 1
        self.response_counts[response_signature] = response_count + 1
        self.last_signature = signature

        return NoveltyUpdate(
            bonus=bonus,
            signature=signature,
            chain=chain,
            response_signature=response_signature,
        )

    def signature(self, candidate: ActionCandidate) -> str:
        endpoint = candidate.bindings.get(KK.ENDPOINT, "")
        path = candidate.bindings.get(KK.PATH, "")
        param = candidate.bindings.get(KK.PARAM_NAME, "")
        tool = candidate.tool_call.tool.value
        return "|".join(
            [
                candidate.template.value,
                candidate.policy.what.value,
                candidate.policy.how.value,
                candidate.policy.where.value,
                f"tool={tool}",
                f"endpoint={endpoint}",
                f"path={path}",
                f"param={param}",
            ]
        )

    def chain_signature(self, signature: str) -> str:
        previous = self.last_signature or "<start>"
        return f"{previous} -> {signature}"


def _bucket_status(status: int) -> str:
    if status == 0:
        return "blocked"
    if status < 200:
        return "1xx"
    if status < 300:
        return "2xx"
    if status < 400:
        return "3xx"
    if status < 500:
        return "4xx"
    return "5xx"


def _bucket_kv(new_kv: int) -> str:
    if new_kv <= 0:
        return "none"
    if new_kv <= 2:
        return "low"
    if new_kv <= 8:
        return "medium"
    return "high"
