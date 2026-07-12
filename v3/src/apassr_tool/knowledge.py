from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Iterable
from urllib.parse import urlparse


class KK(str, Enum):
    BASE_URL = "KK_BASE_URL"
    HOST = "KK_HOST"
    PORT = "KK_PORT"
    PORT_STATE = "KK_PORT_STATE"
    SERVICE = "KK_SERVICE"
    WEB_TECH = "KK_WEB_TECH"
    HTTP_METHOD = "KK_HTTP_METHOD"
    PATH = "KK_PATH"
    ROBOTS_PATH = "KK_ROBOTS_PATH"
    ENDPOINT = "KK_ENDPOINT"
    QUERY_PARAM = "KK_QUERY_PARAM"
    PARAM_NAME = "KK_PARAM_NAME"
    PROBE_VALUE = "KK_PROBE_VALUE"
    USER_ID = "KK_USER_ID"
    USERNAME = "KK_USERNAME"
    ROLE = "KK_ROLE"
    PASSWORD_HINT = "KK_PASSWORD_HINT"
    PASSWORD_CANDIDATE = "KK_PASSWORD_CANDIDATE"
    SESSION_COOKIE = "KK_SESSION_COOKIE"
    AUTH_PATH = "KK_AUTH_PATH"
    FLAG = "KK_FLAG"


@dataclass
class KV:
    value: str
    type: str = "str"
    source: str = "unknown"
    confidence: float = 1.0
    status: str = "active"
    used_count: int = 0
    success_count: int = 0
    last_updated: float = field(default_factory=time.time)


class KnowledgeStore:
    def __init__(self) -> None:
        self._data: dict[KK, list[KV]] = {kk: [] for kk in KK}

    def add(self, kk: KK, value: str, *, source: str = "observation", type: str = "str") -> bool:
        normalized = str(value).strip()
        if not normalized:
            return False
        if any(item.value == normalized and item.status == "active" for item in self._data[kk]):
            return False
        self._data[kk].append(KV(value=normalized, type=type, source=source))
        return True

    def add_many(self, items: Iterable[tuple[KK, str]], *, source: str = "observation") -> int:
        count = 0
        for kk, value in items:
            count += 1 if self.add(kk, value, source=source) else 0
        return count

    def values(self, kk: KK, *, active_only: bool = True) -> list[str]:
        rows = self._data[kk]
        if active_only:
            rows = [row for row in rows if row.status == "active"]
        return [row.value for row in rows]

    def has(self, kk: KK, value: str | None = None) -> bool:
        values = self.values(kk)
        if value is None:
            return bool(values)
        return value in values

    def first(self, kk: KK) -> str | None:
        values = self.values(kk)
        return values[0] if values else None

    def mark_used(self, kk: KK, value: str, *, success: bool = False) -> None:
        for item in self._data[kk]:
            if item.value == value:
                item.used_count += 1
                if success:
                    item.success_count += 1
                item.last_updated = time.time()
                return

    def rows(self) -> list[dict[str, str | int | float]]:
        output: list[dict[str, str | int | float]] = []
        for kk, values in self._data.items():
            for kv in values:
                output.append(
                    {
                        "kk": kk.value,
                        "value": kv.value,
                        "source": kv.source,
                        "status": kv.status,
                        "used_count": kv.used_count,
                        "success_count": kv.success_count,
                    }
                )
        return output

    def derive(self) -> int:
        count = 0
        if self.has(KK.PASSWORD_HINT, "role+id"):
            for role in self.values(KK.ROLE):
                for user_id in self.values(KK.USER_ID):
                    count += 1 if self.add(
                        KK.PASSWORD_CANDIDATE,
                        f"{role}{user_id}",
                        source="derived:role+id",
                    ) else 0
        return count


def seed_knowledge(base_url: str) -> KnowledgeStore:
    store = KnowledgeStore()
    store.add(KK.BASE_URL, base_url, source="seed")
    parsed = urlparse(base_url)
    if parsed.hostname:
        store.add(KK.HOST, parsed.hostname, source="seed")
    if parsed.port:
        store.add(KK.PORT, str(parsed.port), source="seed")
    elif parsed.scheme == "https":
        store.add(KK.PORT, "443", source="seed")
    elif parsed.scheme == "http":
        store.add(KK.PORT, "80", source="seed")
    store.add(KK.PATH, "/", source="seed")
    store.add(KK.PATH, "/robots.txt", source="seed")
    return store
