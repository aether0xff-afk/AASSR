from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Mapping, Protocol


def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    information_id: str
    vector: tuple[float, ...]
    cluster_id: int
    observations: int


@dataclass(slots=True)
class _Cluster:
    cluster_id: int
    centroid: list[float]
    count: int = 1

    def update(self, vector: tuple[float, ...]) -> None:
        self.count += 1
        rate = 1.0 / self.count
        for index, value in enumerate(vector):
            self.centroid[index] += rate * (value - self.centroid[index])


@dataclass(slots=True)
class _RunningValue:
    count: int = 0
    mean: float = 0.0

    def observe(self, value: float) -> None:
        self.count += 1
        self.mean += (value - self.mean) / self.count


class EmbeddingProvider(Protocol):
    def embed(self, tokens: Iterable[str]) -> tuple[float, ...]: ...


class HashEmbeddingProvider:
    """Deterministic dependency-free feature baseline, not an LLM."""

    def __init__(self, dimension: int = 32) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    def embed(self, tokens: Iterable[str]) -> tuple[float, ...]:
        vector = [0.0] * self.dimension
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return tuple(vector)


class SelectiveEmbeddingRouter:
    """Use embedding only when observations or candidate sets become large."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        token_threshold: int = 12,
        candidate_threshold: int = 32,
    ) -> None:
        self.provider = provider
        self.token_threshold = token_threshold
        self.candidate_threshold = candidate_threshold

    def should_embed(
        self,
        tokens: Iterable[str],
        *,
        candidate_count: int,
    ) -> bool:
        token_count = sum(1 for _ in tokens)
        return (
            token_count >= self.token_threshold
            or candidate_count >= self.candidate_threshold
        )

    def encode(
        self,
        tokens: Iterable[str],
        *,
        candidate_count: int = 0,
    ) -> tuple[float, ...]:
        materialized = tuple(tokens)
        if self.should_embed(materialized, candidate_count=candidate_count):
            return self.provider.embed(materialized)
        return self.provider.embed(
            f"symbol:{token}" for token in materialized
        )


class OnlineFeatureMemory:
    """Experience-derived meaning through similarity and action-slot outcomes."""

    def __init__(
        self,
        encoder: EmbeddingProvider | None = None,
        *,
        cluster_threshold: float = 0.72,
    ) -> None:
        if not -1.0 <= cluster_threshold <= 1.0:
            raise ValueError("cluster_threshold must be in [-1, 1]")
        self.encoder = encoder or HashEmbeddingProvider()
        self.cluster_threshold = cluster_threshold
        self._records: dict[str, FeatureRecord] = {}
        self._clusters: list[_Cluster] = []
        self._role_values: dict[tuple[str, str, int], _RunningValue] = {}
        self._item_values: dict[tuple[str, str, str], _RunningValue] = {}

    def _assign_cluster(self, vector: tuple[float, ...]) -> int:
        if not self._clusters:
            self._clusters.append(_Cluster(0, list(vector)))
            return 0
        best = max(
            self._clusters,
            key=lambda cluster: cosine(vector, tuple(cluster.centroid)),
        )
        if cosine(vector, tuple(best.centroid)) < self.cluster_threshold:
            cluster_id = len(self._clusters)
            self._clusters.append(_Cluster(cluster_id, list(vector)))
            return cluster_id
        best.update(vector)
        return best.cluster_id

    def observe_information(
        self,
        information_id: str,
        tokens: Iterable[str],
    ) -> FeatureRecord:
        vector = self.encoder.embed(tokens)
        previous = self._records.get(information_id)
        cluster_id = self._assign_cluster(vector)
        observations = 1 if previous is None else previous.observations + 1
        record = FeatureRecord(
            information_id,
            vector,
            cluster_id,
            observations,
        )
        self._records[information_id] = record
        return record

    def observe_use(
        self,
        information_id: str,
        *,
        action_id: str,
        slot: str,
        value: float,
    ) -> None:
        record = self._records[information_id]
        self._role_values.setdefault(
            (action_id, slot, record.cluster_id),
            _RunningValue(),
        ).observe(value)
        self._item_values.setdefault(
            (action_id, slot, information_id),
            _RunningValue(),
        ).observe(value)

    def record(self, information_id: str) -> FeatureRecord | None:
        return self._records.get(information_id)

    def rank_for_slot(
        self,
        action_id: str,
        slot: str,
        candidate_ids: Iterable[str],
        *,
        limit: int,
    ) -> tuple[str, ...]:
        if limit <= 0:
            return ()
        scored = []
        for information_id in candidate_ids:
            score = self.estimated_value(action_id, slot, information_id)
            scored.append((score, information_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return tuple(
            information_id for _, information_id in scored[:limit]
        )

    def estimated_value(
        self,
        action_id: str,
        slot: str,
        information_id: str,
    ) -> float:
        """Return the learned value of an observed item for a generic slot."""

        record = self._records.get(information_id)
        if record is None:
            return 0.0
        cluster = self._role_values.get((action_id, slot, record.cluster_id))
        item = self._item_values.get((action_id, slot, information_id))
        cluster_score = 0.0 if cluster is None else cluster.mean
        item_score = 0.0 if item is None else item.mean
        novelty = 1.0 / (1.0 + record.observations)
        return 0.6 * cluster_score + 0.3 * item_score + 0.1 * novelty

    def cluster_count(self) -> int:
        return len(self._clusters)

    def snapshot(self) -> Mapping[str, FeatureRecord]:
        return dict(self._records)
