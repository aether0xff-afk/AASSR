from __future__ import annotations

from dataclasses import dataclass, field

from .actions import ActionCandidate
from .knowledge import KK


@dataclass
class ProphecyStat:
    count: int = 0
    reward_total: float = 0.0
    knowledge_total: int = 0
    solved_total: int = 0
    error_total: int = 0

    def update(self, *, reward: float, new_kv: int, solved_delta: int, error: bool) -> None:
        self.count += 1
        self.reward_total += reward
        self.knowledge_total += new_kv
        self.solved_total += solved_delta
        self.error_total += 1 if error else 0

    @property
    def reward_mean(self) -> float:
        return self.reward_total / self.count if self.count else 0.0

    @property
    def knowledge_mean(self) -> float:
        return self.knowledge_total / self.count if self.count else 0.0

    @property
    def solved_rate(self) -> float:
        return self.solved_total / self.count if self.count else 0.0

    @property
    def error_rate(self) -> float:
        return self.error_total / self.count if self.count else 0.0


@dataclass
class ProphecyPrediction:
    expected_reward: float = 0.0
    expected_knowledge: float = 0.0
    solved_rate: float = 0.0
    error_rate: float = 0.0
    support: int = 0


@dataclass
class TableProphecyModel:
    stats: dict[str, ProphecyStat] = field(default_factory=dict)

    def update(self, candidate: ActionCandidate, *, reward: float, new_kv: int, solved_delta: int, status: int) -> None:
        error = status >= 400 or status == 0
        for key in self._keys(candidate):
            self.stats.setdefault(key, ProphecyStat()).update(
                reward=reward,
                new_kv=new_kv,
                solved_delta=solved_delta,
                error=error,
            )

    def predict(self, candidate: ActionCandidate) -> ProphecyPrediction:
        rows = [(key, self.stats[key]) for key in self._keys(candidate) if key in self.stats]
        if not rows:
            return ProphecyPrediction()
        weighted_support = sum(self._key_weight(key) * row.count for key, row in rows)
        support = sum(row.count for _, row in rows)
        if support <= 0 or weighted_support <= 0:
            return ProphecyPrediction()
        return ProphecyPrediction(
            expected_reward=sum(self._key_weight(key) * row.reward_mean * row.count for key, row in rows)
            / weighted_support,
            expected_knowledge=sum(self._key_weight(key) * row.knowledge_mean * row.count for key, row in rows)
            / weighted_support,
            solved_rate=sum(self._key_weight(key) * row.solved_rate * row.count for key, row in rows)
            / weighted_support,
            error_rate=sum(self._key_weight(key) * row.error_rate * row.count for key, row in rows)
            / weighted_support,
            support=support,
        )

    def _keys(self, candidate: ActionCandidate) -> tuple[str, ...]:
        keys = [
            f"exact:{candidate.tried_key}",
            f"template:{candidate.template.value}",
            f"what:{candidate.policy.what.value}",
            f"how:{candidate.policy.how.value}",
            f"where:{candidate.policy.where.value}",
        ]
        endpoint = candidate.bindings.get(KK.ENDPOINT)
        if endpoint:
            keys.append(f"endpoint:{endpoint}")
            keys.append(f"template_endpoint:{candidate.template.value}:{endpoint}")
        path = candidate.bindings.get(KK.PATH)
        if path:
            keys.append(f"path:{path}")
        param_name = candidate.bindings.get(KK.PARAM_NAME)
        if param_name:
            keys.append(f"param:{param_name}")
        return tuple(keys)

    def _key_weight(self, key: str) -> float:
        if key.startswith("exact:"):
            return 5.0
        if key.startswith("template_endpoint:"):
            return 4.0
        if key.startswith(("endpoint:", "path:", "param:")):
            return 3.0
        if key.startswith("template:"):
            return 2.0
        return 1.0
