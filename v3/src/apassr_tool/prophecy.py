from __future__ import annotations

from dataclasses import dataclass, field
import json
import random
from pathlib import Path
from typing import Sequence

from .actions import ActionCandidate
from .knowledge import KK


@dataclass
class ProphecyStat:
    count: int = 0
    reward_total: float = 0.0
    knowledge_total: int = 0
    solved_total: float = 0.0
    progress_total: float = 0.0
    eventual_solve_total: float = 0.0
    expected_progress_total: float = 0.0
    error_total: int = 0

    def update(self, *, reward: float, new_kv: int, solved_delta: int, progress: float, error: bool) -> None:
        self.count += 1
        self.reward_total += reward
        self.knowledge_total += new_kv
        self.solved_total += 1.0 if solved_delta > 0 else 0.0
        self.progress_total += 1.0 if progress > 0 else 0.0
        self.expected_progress_total += max(0.0, min(1.0, progress))
        self.error_total += 1 if error else 0

    @property
    def reward_mean(self) -> float: return self.reward_total / self.count if self.count else 0.0

    @property
    def knowledge_mean(self) -> float: return self.knowledge_total / self.count if self.count else 0.0

    @property
    def solved_rate(self) -> float: return self.solved_total / self.count if self.count else 0.0

    @property
    def progress_rate(self) -> float: return self.progress_total / self.count if self.count else 0.0

    @property
    def eventual_solve_rate(self) -> float: return self.eventual_solve_total / self.count if self.count else 0.0

    @property
    def expected_progress(self) -> float: return self.expected_progress_total / self.count if self.count else 0.0

    @property
    def error_rate(self) -> float: return self.error_total / self.count if self.count else 0.0


@dataclass(frozen=True)
class ProphecyPrediction:
    expected_reward: float = 0.0
    expected_knowledge: float = 0.0
    solved_rate: float = 0.01  # compatibility: immediate solve probability
    progress_probability: float = 0.1
    eventual_solve_probability: float = 0.02
    expected_progress: float = 0.0
    error_rate: float = 0.0
    support: int = 0

    @property
    def immediate_solve_probability(self) -> float:
        return self.solved_rate


@dataclass
class ReplaySample:
    candidate: ActionCandidate
    reward: float
    status: int
    progress: float = 0.0
    immediate_solved: float = 0.0
    eventual_solve_credit: float = 0.0
    relative_position: float = 0.0
    distance_to_solve: int | None = None

    @property
    def positive(self) -> bool:
        return self.immediate_solved > 0 or self.eventual_solve_credit > 0 or self.progress > 0


@dataclass
class TableProphecyModel:
    stats: dict[str, ProphecyStat] = field(default_factory=dict)
    replay: list[ReplaySample] = field(default_factory=list)
    hindsight_decay: float = 0.8
    hindsight_window: int = 4
    minimum_positive_ratio: float = 0.25
    bootstrap_immediate_prior: float = 0.01
    bootstrap_progress_prior: float = 0.1
    bootstrap_eventual_prior: float = 0.02
    prior_strength: float = 2.0
    seed: int = 0

    def update(
        self, candidate: ActionCandidate, *, reward: float, new_kv: int,
        solved_delta: int, status: int, progress: float = 0.0,
    ) -> None:
        error = status >= 400 or status == 0
        for key in self._keys(candidate):
            self.stats.setdefault(key, ProphecyStat()).update(
                reward=reward, new_kv=new_kv, solved_delta=solved_delta,
                progress=progress, error=error,
            )
        self.replay.append(ReplaySample(
            candidate=candidate, reward=reward, status=status,
            progress=max(0.0, min(1.0, progress)), immediate_solved=1.0 if solved_delta > 0 else 0.0,
        ))

    def finalize_episode(self, start_index: int) -> None:
        trajectory = self.replay[start_index:]
        if not trajectory:
            return
        denominator = max(1, len(trajectory) - 1)
        for index, sample in enumerate(trajectory):
            sample.relative_position = index / denominator
        solve_indices = [index for index, sample in enumerate(trajectory) if sample.immediate_solved > 0]
        for solve_index in solve_indices:
            for distance in range(self.hindsight_window):
                index = solve_index - distance
                if index < 0:
                    break
                sample = trajectory[index]
                credit = self.hindsight_decay ** distance
                if credit <= sample.eventual_solve_credit:
                    continue
                delta = credit - sample.eventual_solve_credit
                sample.eventual_solve_credit = credit
                sample.distance_to_solve = distance
                for key in self._keys(sample.candidate):
                    self.stats[key].eventual_solve_total += delta

    def predict(self, candidate: ActionCandidate) -> ProphecyPrediction:
        rows = [(key, self.stats[key]) for key in self._keys(candidate) if key in self.stats]
        if not rows:
            return ProphecyPrediction(
                solved_rate=self.bootstrap_immediate_prior,
                progress_probability=self.bootstrap_progress_prior,
                eventual_solve_probability=self.bootstrap_eventual_prior,
            )
        weighted_support = sum(self._key_weight(key) * row.count for key, row in rows)
        support = sum(row.count for _, row in rows)
        if support <= 0 or weighted_support <= 0:
            return ProphecyPrediction()
        mean = lambda attr: sum(self._key_weight(key) * getattr(row, attr) for key, row in rows) / weighted_support
        smooth = lambda total, prior: (total + self.prior_strength * prior) / (weighted_support + self.prior_strength)
        return ProphecyPrediction(
            expected_reward=mean("reward_total"),
            expected_knowledge=mean("knowledge_total"),
            solved_rate=smooth(mean("solved_total") * weighted_support, self.bootstrap_immediate_prior),
            progress_probability=smooth(mean("progress_total") * weighted_support, self.bootstrap_progress_prior),
            eventual_solve_probability=smooth(mean("eventual_solve_total") * weighted_support, self.bootstrap_eventual_prior),
            expected_progress=mean("expected_progress_total"),
            error_rate=mean("error_total"), support=support,
        )

    def sample_batch(self, size: int, *, minimum_positive_ratio: float | None = None) -> list[ReplaySample]:
        if size <= 0 or not self.replay:
            return []
        rng = random.Random(self.seed + len(self.replay) + size)
        positives = [sample for sample in self.replay if sample.positive]
        negatives = [sample for sample in self.replay if not sample.positive]
        ratio = self.minimum_positive_ratio if minimum_positive_ratio is None else minimum_positive_ratio
        positive_count = min(len(positives), max(1 if positives else 0, int(round(size * ratio))))
        selected = rng.sample(positives, min(positive_count, len(positives)))
        remaining = size - len(selected)
        pool = negatives or positives
        if pool:
            selected.extend(rng.choices(pool, k=remaining))
        rng.shuffle(selected)
        return selected

    def load_success_trajectories(self, path: str | Path) -> int:
        """Load sanitized JSON trajectories; malformed/missing history is ignored safely."""
        try:
            rows = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return 0
        # Import is intentionally conservative: only trajectories produced with live candidates
        # can train this table. JSON history still raises bootstrap priors without leaking actions.
        successes = sum(1 for row in rows if isinstance(row, dict) and row.get("solved")) if isinstance(rows, list) else 0
        if successes:
            self.bootstrap_eventual_prior = max(self.bootstrap_eventual_prior, min(0.2, successes / max(len(rows), 1)))
        return successes

    def _keys(self, candidate: ActionCandidate) -> tuple[str, ...]:
        keys = [
            f"exact:{candidate.tried_key}", f"template:{candidate.template.value}",
            f"what:{candidate.policy.what.value}", f"how:{candidate.policy.how.value}",
            f"where:{candidate.policy.where.value}",
        ]
        endpoint = candidate.bindings.get(KK.ENDPOINT)
        if endpoint:
            keys.extend((f"endpoint:{endpoint}", f"template_endpoint:{candidate.template.value}:{endpoint}"))
        path = candidate.bindings.get(KK.PATH)
        if path: keys.append(f"path:{path}")
        param_name = candidate.bindings.get(KK.PARAM_NAME)
        if param_name: keys.append(f"param:{param_name}")
        return tuple(keys)

    def _key_weight(self, key: str) -> float:
        if key.startswith("exact:"): return 5.0
        if key.startswith("template_endpoint:"): return 4.0
        if key.startswith(("endpoint:", "path:", "param:")): return 3.0
        if key.startswith("template:"): return 2.0
        return 1.0
