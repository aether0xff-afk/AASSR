from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

from .knowledge import KK


class What(str, Enum):
    HTTP_GET = "HTTP_GET"
    FORM_POST = "FORM_POST"
    AUTHENTICATED_GET = "AUTHENTICATED_GET"
    PORT_SCAN = "PORT_SCAN"
    WEB_FINGERPRINT = "WEB_FINGERPRINT"
    HTTP_METADATA = "HTTP_METADATA"
    QUERY_PROBE = "QUERY_PROBE"


class How(str, Enum):
    NORMAL = "NORMAL"
    PARAMETERIZED = "PARAMETERIZED"
    AUTH_ATTEMPT = "AUTH_ATTEMPT"
    AUTHENTICATED = "AUTHENTICATED"
    SHALLOW_SCAN = "SHALLOW_SCAN"
    PASSIVE_FINGERPRINT = "PASSIVE_FINGERPRINT"
    HEADER_ONLY = "HEADER_ONLY"
    METHOD_DISCOVERY = "METHOD_DISCOVERY"
    PROBE_VALUE = "PROBE_VALUE"


class Where(str, Enum):
    KK_PATH = "KK_PATH"
    KK_ENDPOINT = "KK_ENDPOINT"
    KK_USERNAME = "KK_USERNAME"
    KK_AUTH_PATH = "KK_AUTH_PATH"
    KK_BASE_URL = "KK_BASE_URL"
    KK_HOST = "KK_HOST"
    KK_PARAM_NAME = "KK_PARAM_NAME"


@dataclass(frozen=True)
class PolicyView:
    what: What
    how: How
    where: Where


@dataclass
class PolicyABC:
    lr: float = 0.05
    min_prob: float = 0.02
    what_probs: dict[What, float] = field(default_factory=dict)
    how_probs: dict[How, float] = field(default_factory=dict)
    where_probs: dict[Where, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.what_probs:
            self.what_probs = _uniform(What)
        if not self.how_probs:
            self.how_probs = _uniform(How)
        if not self.where_probs:
            self.where_probs = _uniform(Where)

    def score(self, policy: PolicyView, *, tried_count: int = 0) -> float:
        base = (
            self.what_probs[policy.what]
            + self.how_probs[policy.how]
            + self.where_probs[policy.where]
        ) / 3.0
        return base / (1.0 + tried_count)

    def update(self, policy: PolicyView, reward: float) -> None:
        self._boost(self.what_probs, policy.what, reward)
        self._boost(self.how_probs, policy.how, reward)
        self._boost(self.where_probs, policy.where, reward)

    def _boost(self, table: dict, key: object, reward: float) -> None:
        table[key] *= math.exp(self.lr * reward)
        _normalize_with_floor(table, self.min_prob)


def where_from_kk(kk: KK) -> Where:
    mapping = {
        KK.PATH: Where.KK_PATH,
        KK.ENDPOINT: Where.KK_ENDPOINT,
        KK.USERNAME: Where.KK_USERNAME,
        KK.AUTH_PATH: Where.KK_AUTH_PATH,
        KK.BASE_URL: Where.KK_BASE_URL,
        KK.HOST: Where.KK_HOST,
        KK.PARAM_NAME: Where.KK_PARAM_NAME,
    }
    return mapping.get(kk, Where.KK_BASE_URL)


def _uniform(enum_cls: type[Enum]) -> dict:
    values = list(enum_cls)
    probability = 1.0 / len(values)
    return {value: probability for value in values}


def _normalize_with_floor(table: dict, floor: float) -> None:
    for key in list(table):
        table[key] = max(table[key], floor)
    total = sum(table.values())
    for key in list(table):
        table[key] = table[key] / total
