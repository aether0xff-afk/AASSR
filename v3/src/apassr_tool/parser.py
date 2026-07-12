from __future__ import annotations

import json
import re
from urllib.parse import parse_qsl, urlparse

from .knowledge import KK
from .tools import ToolResult


FLAG_RE = re.compile(r"FLAG\{[^}]+\}")
HREF_RE = re.compile(r"""href=["']([^"']+)["']""", re.IGNORECASE)
SCRIPT_RE = re.compile(r"""src=["']([^"']+\.js[^"']*)["']""", re.IGNORECASE)
FORM_RE = re.compile(r"""action=["']([^"']+)["']""", re.IGNORECASE)
INPUT_NAME_RE = re.compile(r"""<input[^>]+name=["']([^"']+)["']""", re.IGNORECASE)
OBJECT_KEY_RE = re.compile(r"""["']([A-Za-z_][A-Za-z0-9_-]{0,40})["']\s*:""")
MAX_OBJECT_PARAM_NAMES = 24
DISALLOW_RE = re.compile(r"Disallow:\s*(/\S+)", re.IGNORECASE)
USER_ID_RE = re.compile(r"(?:user id is|id=)\s*(\d+)", re.IGNORECASE)
API_PATH_RE = re.compile(r"(/(?:api|rest)/[A-Za-z0-9_/\-?=&:]+)")
COOKIE_RE = re.compile(r"Set-Cookie:\s*([^;\r\n]+)", re.IGNORECASE)
ALLOW_RE = re.compile(r"^Allow:\s*([A-Z,\s]+)$", re.IGNORECASE | re.MULTILINE)
PASSWORD_RULE_RE = re.compile(r"password\s*=\s*role\s*\+\s*id", re.IGNORECASE)
LOGIN_ENDPOINT_RE = re.compile(r"""loginEndpoint\s*=\s*["']([^"']+)["']""")
FLAG_PATH_RE = re.compile(r"""flagPath\s*=\s*["']([^"']+)["']""")
ADMIN_PATH_RE = re.compile(r"admin area:\s*(/\S+)", re.IGNORECASE)
NMAP_PORT_RE = re.compile(r"^(\d+)/tcp\s+([A-Za-z0-9_.-]+)\s+([A-Za-z0-9_.-]+)", re.MULTILINE)
WHATWEB_TECH_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9_.:+ -]{1,80})\]")


def parse_tool_result(result: ToolResult) -> list[tuple[KK, str]]:
    text = result.stdout
    items: list[tuple[KK, str]] = []
    for match in FLAG_RE.findall(text):
        items.append((KK.FLAG, match))
    for pattern in [HREF_RE, SCRIPT_RE, FORM_RE]:
        for path in pattern.findall(text):
            items.append(_path_item(path))
    for name in INPUT_NAME_RE.findall(text):
        items.append((KK.PARAM_NAME, name))
    for name in _bounded_object_keys(text):
        items.append((KK.PARAM_NAME, name))
    for path in DISALLOW_RE.findall(text):
        items.append((KK.ROBOTS_PATH, path))
        items.append((KK.PATH, path))
    for user_id in USER_ID_RE.findall(text):
        items.append((KK.USER_ID, user_id))
        items.append((KK.QUERY_PARAM, f"id={user_id}"))
        items.append((KK.PROBE_VALUE, user_id))
    for endpoint in API_PATH_RE.findall(text):
        path, _, query = endpoint.partition("?")
        items.append((KK.ENDPOINT, path))
        if query:
            items.append((KK.QUERY_PARAM, query))
            for name, value in parse_qsl(query, keep_blank_values=True):
                items.append((KK.PARAM_NAME, name))
                if value:
                    items.append((KK.PROBE_VALUE, value))
    for cookie in COOKIE_RE.findall(text):
        items.append((KK.SESSION_COOKIE, cookie.strip()))
    for methods in ALLOW_RE.findall(text):
        for method in methods.split(","):
            cleaned = method.strip().upper()
            if cleaned:
                items.append((KK.HTTP_METHOD, cleaned))
    for endpoint in LOGIN_ENDPOINT_RE.findall(text):
        items.append((KK.PATH, endpoint))
    for path in FLAG_PATH_RE.findall(text):
        items.append((KK.AUTH_PATH, path))
        items.append((KK.PATH, path))
    for path in ADMIN_PATH_RE.findall(text):
        items.append((KK.AUTH_PATH, path))
        items.append((KK.PATH, path))
    if PASSWORD_RULE_RE.search(text):
        items.append((KK.PASSWORD_HINT, "role+id"))
    for port, state, service in NMAP_PORT_RE.findall(text):
        items.append((KK.PORT, port))
        items.append((KK.PORT_STATE, f"{port}/tcp:{state}"))
        items.append((KK.SERVICE, service))
    if result.tool == "WHATWEB_SCAN":
        for tech in WHATWEB_TECH_RE.findall(text):
            cleaned = tech.strip()
            if cleaned and not cleaned[0].isdigit():
                items.append((KK.WEB_TECH, cleaned))

    items.extend(_json_items(text))
    items.extend(_derived_items(items))
    return _dedupe(items)


def _path_item(path: str) -> tuple[KK, str]:
    path = path.strip()
    if (
        not path
        or path in {".", "..", "none"}
        or "{{" in path
        or path.startswith(("#", "mailto:", "javascript:", "data:"))
    ):
        return (KK.PATH, "")
    if path.startswith("http://") or path.startswith("https://"):
        parsed = urlparse(path)
        return (KK.PATH, parsed.path or "/")
    return (KK.PATH, path)


def _json_items(text: str) -> list[tuple[KK, str]]:
    stripped = text.strip()
    # curl -i includes headers; use text after last blank line when possible.
    if "\n\n" in stripped:
        stripped = stripped.split("\n\n")[-1].strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return []
    items: list[tuple[KK, str]] = []
    _walk_json(data, items)
    return items


def _bounded_object_keys(text: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for name in OBJECT_KEY_RE.findall(text):
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
        if len(names) >= MAX_OBJECT_PARAM_NAMES:
            break
    return names


def _walk_json(value: object, items: list[tuple[KK, str]], *, key: str = "") -> None:
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            _walk_json(child_value, items, key=str(child_key))
        return
    if isinstance(value, list):
        for child_value in value:
            _walk_json(child_value, items, key=key)
        return
    if value is None:
        return
    if isinstance(value, bool):
        text = "true" if value else "false"
        items.append((KK.PROBE_VALUE, text))
        return
    text = str(value).strip()
    if not _is_probe_like(text):
        return
    normalized_key = key.lower()
    if normalized_key == "id":
        items.append((KK.USER_ID, text))
    if normalized_key == "username":
        items.append((KK.USERNAME, text))
    if normalized_key == "role":
        items.append((KK.ROLE, text))
    if normalized_key in {"id", "email", "username", "role", "name", "key", "q", "search"}:
        items.append((KK.PROBE_VALUE, text))


def _is_probe_like(value: str) -> bool:
    if not value or len(value) > 80:
        return False
    if any(char in value for char in "\r\n\t{}[]<>"):
        return False
    return True


def _derived_items(items: list[tuple[KK, str]]) -> list[tuple[KK, str]]:
    values: dict[KK, set[str]] = {}
    for kk, value in items:
        values.setdefault(kk, set()).add(value)
    derived: list[tuple[KK, str]] = []
    if "role+id" in values.get(KK.PASSWORD_HINT, set()):
        for role in values.get(KK.ROLE, set()):
            for user_id in values.get(KK.USER_ID, set()):
                derived.append((KK.PASSWORD_CANDIDATE, f"{role}{user_id}"))
    return derived


def _dedupe(items: list[tuple[KK, str]]) -> list[tuple[KK, str]]:
    seen: set[tuple[KK, str]] = set()
    output: list[tuple[KK, str]] = []
    for item in items:
        if not item[1]:
            continue
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
