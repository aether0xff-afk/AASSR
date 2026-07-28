from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urljoin, urlparse

import requests


@dataclass(frozen=True)
class RewardConfig:
    """Bounded reward terms; solving dominates any single observation."""

    challenge_solved: float = 50.0
    challenge_progress: float = 4.0
    semantic_novelty_unit: float = 0.35
    semantic_novelty_cap: float = 2.0
    useful_observation: float = 0.1
    repeated_action: float = 0.25
    repeated_response: float = 0.5
    invalid_action: float = 1.0
    no_progress: float = 0.1
    repeated_failure_growth: float = 0.15
    repeated_failure_cap: float = 1.5


@dataclass(frozen=True)
class RewardBreakdown:
    reward_challenge_solved: float = 0.0
    reward_challenge_progress: float = 0.0
    reward_semantic_novelty: float = 0.0
    reward_useful_observation: float = 0.0
    penalty_repeated_action: float = 0.0
    penalty_repeated_response: float = 0.0
    penalty_invalid_action: float = 0.0
    penalty_no_progress: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.reward_challenge_solved + self.reward_challenge_progress
            + self.reward_semantic_novelty + self.reward_useful_observation
            - self.penalty_repeated_action - self.penalty_repeated_response
            - self.penalty_invalid_action - self.penalty_no_progress
        )


@dataclass(frozen=True)
class RewardSignal:
    new_solved: tuple[str, ...] = ()
    solved_total: int = 0
    challenge_total: int = 0


class RewardObserver(Protocol):
    def reset(self) -> None:
        ...

    def observe(self) -> RewardSignal:
        ...


@dataclass
class JuiceShopChallengeObserver:
    base_url: str
    allowed_hosts: set[str] = field(default_factory=lambda: {"127.0.0.1", "localhost", "::1"})
    timeout_s: float = 5.0
    _known_solved: set[str] = field(default_factory=set, init=False)
    _last_challenge_total: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("challenge observer requires an HTTP base URL")
        if parsed.hostname not in self.allowed_hosts:
            raise ValueError(f"blocked challenge observer host: {parsed.hostname}")

    def reset(self) -> None:
        solved, total = self._fetch_progress()
        self._known_solved = solved
        self._last_challenge_total = total

    def observe(self) -> RewardSignal:
        solved, total = self._fetch_progress()
        new = tuple(sorted(solved - self._known_solved))
        self._known_solved = solved
        self._last_challenge_total = total
        return RewardSignal(new_solved=new, solved_total=len(solved), challenge_total=total)

    def _fetch_solved(self) -> set[str]:
        solved, _ = self._fetch_progress()
        return solved

    def _fetch_progress(self) -> tuple[set[str], int]:
        url = urljoin(self.base_url.rstrip("/") + "/", "api/Challenges")
        try:
            response = requests.get(url, timeout=self.timeout_s)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException:
            return set(self._known_solved), self._last_challenge_total
        rows = payload.get("data", [])
        solved: set[str] = set()
        for row in rows:
            if isinstance(row, dict) and row.get("solved"):
                key = str(row.get("key") or row.get("name") or row.get("id"))
                solved.add(key)
        return solved, len(rows) if isinstance(rows, list) else 0

    @property
    def solved_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._known_solved))

    @property
    def challenge_total(self) -> int:
        return self._last_challenge_total
