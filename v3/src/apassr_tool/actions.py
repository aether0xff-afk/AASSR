from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urljoin

from .knowledge import KK, KnowledgeStore
from .policy import How, PolicyView, What, Where
from .tools import ToolCall, ToolName


MAX_QUERY_PARAMS_PER_ENDPOINT = 64
MAX_PARAM_NAMES_PER_ENDPOINT = 64
MAX_PROBE_VALUES_PER_ENDPOINT = 64
MAX_LOGIN_PATHS = 64
MAX_COMBO_POST_FIELDS = 12
MAX_COMBO_POSTS_PER_ENDPOINT = 32
MAX_JSON_FIELDS = 12
MAX_JSON_BODIES_PER_ENDPOINT = 32
MAX_MUTATED_VALUES = 24
MAX_MUTATION_CANDIDATES_PER_ENDPOINT = 96
MAX_FILE_SURFACE_CANDIDATES = 64


class ActionTemplate(str, Enum):
    HTTP_GET_PATH = "HTTP_GET_PATH"
    HTTP_HEAD_PATH = "HTTP_HEAD_PATH"
    HTTP_OPTIONS_PATH = "HTTP_OPTIONS_PATH"
    HTTP_GET_API = "HTTP_GET_API"
    HTTP_QUERY_PROBE = "HTTP_QUERY_PROBE"
    HTTP_POST_PROBE = "HTTP_POST_PROBE"
    HTTP_POST_COMBO = "HTTP_POST_COMBO"
    HTTP_JSON_POST = "HTTP_JSON_POST"
    HTTP_JSON_PUT = "HTTP_JSON_PUT"
    HTTP_JSON_PATCH = "HTTP_JSON_PATCH"
    HTTP_JSON_DELETE = "HTTP_JSON_DELETE"
    HTTP_MUTATION_QUERY = "HTTP_MUTATION_QUERY"
    HTTP_MUTATION_POST = "HTTP_MUTATION_POST"
    HTTP_MUTATION_JSON = "HTTP_MUTATION_JSON"
    HTTP_FILE_SURFACE_GET = "HTTP_FILE_SURFACE_GET"
    HTTP_POST_LOGIN = "HTTP_POST_LOGIN"
    HTTP_AUTH_GET = "HTTP_AUTH_GET"
    NMAP_SCAN_HOST = "NMAP_SCAN_HOST"
    WEB_FINGERPRINT = "WEB_FINGERPRINT"


@dataclass(frozen=True)
class ActionCandidate:
    template: ActionTemplate
    bindings: dict[KK, str]
    policy: PolicyView
    label: str
    tool_call: ToolCall
    required_slots: tuple[KK, ...]
    tried_key: str = ""


def generate_candidates(store: KnowledgeStore) -> list[ActionCandidate]:
    base_url = store.first(KK.BASE_URL)
    if not base_url:
        return []
    candidates: list[ActionCandidate] = []

    for host in store.values(KK.HOST):
        ports = store.values(KK.PORT) or ["1-1024"]
        for port in ports:
            candidates.append(
                ActionCandidate(
                    template=ActionTemplate.NMAP_SCAN_HOST,
                    bindings={KK.HOST: host, KK.PORT: port},
                    policy=PolicyView(What.PORT_SCAN, How.SHALLOW_SCAN, Where.KK_HOST),
                    label=f"NMAP {host}:{port}",
                    tool_call=ToolCall(ToolName.NMAP_SCAN, target_host=host, port_range=port),
                    required_slots=(KK.HOST,),
                    tried_key=f"NMAP:{host}:{port}",
                )
            )

    candidates.append(
        ActionCandidate(
            template=ActionTemplate.WEB_FINGERPRINT,
            bindings={KK.BASE_URL: base_url},
            policy=PolicyView(What.WEB_FINGERPRINT, How.PASSIVE_FINGERPRINT, Where.KK_BASE_URL),
            label=f"WHATWEB {base_url}",
            tool_call=ToolCall(ToolName.WHATWEB_SCAN, url=base_url),
            required_slots=(KK.BASE_URL,),
            tried_key=f"WHATWEB:{base_url}",
        )
    )

    for path in store.values(KK.PATH):
        url = urljoin(base_url, path)
        candidates.append(
            ActionCandidate(
                template=ActionTemplate.HTTP_GET_PATH,
                bindings={KK.PATH: path},
                policy=PolicyView(What.HTTP_GET, How.NORMAL, Where.KK_PATH),
                label=f"GET {path}",
                tool_call=ToolCall(ToolName.CURL_GET, url=url),
                required_slots=(KK.BASE_URL, KK.PATH),
                tried_key=f"GET:{path}",
            )
        )
        candidates.append(
            ActionCandidate(
                template=ActionTemplate.HTTP_HEAD_PATH,
                bindings={KK.PATH: path},
                policy=PolicyView(What.HTTP_METADATA, How.HEADER_ONLY, Where.KK_PATH),
                label=f"HEAD {path}",
                tool_call=ToolCall(ToolName.CURL_HEAD, url=url),
                required_slots=(KK.BASE_URL, KK.PATH),
                tried_key=f"HEAD:{path}",
            )
        )
        candidates.append(
            ActionCandidate(
                template=ActionTemplate.HTTP_OPTIONS_PATH,
                bindings={KK.PATH: path},
                policy=PolicyView(What.HTTP_METADATA, How.METHOD_DISCOVERY, Where.KK_PATH),
                label=f"OPTIONS {path}",
                tool_call=ToolCall(ToolName.CURL_OPTIONS, url=url),
                required_slots=(KK.BASE_URL, KK.PATH),
                tried_key=f"OPTIONS:{path}",
            )
        )

    for endpoint in store.values(KK.ENDPOINT):
        params = _limited(store.values(KK.QUERY_PARAM), MAX_QUERY_PARAMS_PER_ENDPOINT) or [""]
        for query in params:
            suffix = f"{endpoint}?{query}" if query else endpoint
            url = urljoin(base_url, suffix)
            candidates.append(
                ActionCandidate(
                    template=ActionTemplate.HTTP_GET_API,
                    bindings={KK.ENDPOINT: endpoint, KK.QUERY_PARAM: query},
                    policy=PolicyView(What.HTTP_GET, How.PARAMETERIZED, Where.KK_ENDPOINT),
                    label=f"GET {suffix}",
                    tool_call=ToolCall(ToolName.CURL_GET, url=url),
                    required_slots=(KK.BASE_URL, KK.ENDPOINT),
                    tried_key=f"GET:{suffix}",
                )
            )
        param_candidates = _limited(store.values(KK.PARAM_NAME), MAX_PARAM_NAMES_PER_ENDPOINT)
        probe_candidates = _limited(store.values(KK.PROBE_VALUE), MAX_PROBE_VALUES_PER_ENDPOINT)
        for param_name in param_candidates:
            for probe_value in probe_candidates:
                suffix = f"{endpoint}?{param_name}={probe_value}"
                url = urljoin(base_url, suffix)
                candidates.append(
                    ActionCandidate(
                        template=ActionTemplate.HTTP_QUERY_PROBE,
                        bindings={
                            KK.ENDPOINT: endpoint,
                            KK.PARAM_NAME: param_name,
                            KK.PROBE_VALUE: probe_value,
                        },
                        policy=PolicyView(What.QUERY_PROBE, How.PROBE_VALUE, Where.KK_PARAM_NAME),
                        label=f"PROBE {endpoint}?{param_name}={probe_value}",
                        tool_call=ToolCall(ToolName.CURL_GET, url=url),
                        required_slots=(KK.BASE_URL, KK.ENDPOINT, KK.PARAM_NAME, KK.PROBE_VALUE),
                        tried_key=f"PROBE:{endpoint}:{param_name}:{probe_value}",
                    )
                )
                candidates.append(
                    ActionCandidate(
                        template=ActionTemplate.HTTP_POST_PROBE,
                        bindings={
                            KK.ENDPOINT: endpoint,
                            KK.PARAM_NAME: param_name,
                            KK.PROBE_VALUE: probe_value,
                        },
                        policy=PolicyView(What.FORM_POST, How.PROBE_VALUE, Where.KK_PARAM_NAME),
                        label=f"POST_PROBE {endpoint} {param_name}={probe_value}",
                        tool_call=ToolCall(
                            ToolName.CURL_POST,
                            url=urljoin(base_url, endpoint),
                            data={param_name: probe_value},
                        ),
                        required_slots=(KK.BASE_URL, KK.ENDPOINT, KK.PARAM_NAME, KK.PROBE_VALUE),
                        tried_key=f"POST_PROBE:{endpoint}:{param_name}:{probe_value}",
                    )
                )

        for index, body in enumerate(_combo_bodies(param_candidates, probe_candidates)):
            candidates.append(
                ActionCandidate(
                    template=ActionTemplate.HTTP_POST_COMBO,
                    bindings={
                        KK.ENDPOINT: endpoint,
                        KK.PARAM_NAME: ",".join(body),
                        KK.PROBE_VALUE: ",".join(body.values()),
                    },
                    policy=PolicyView(What.FORM_POST, How.PROBE_VALUE, Where.KK_PARAM_NAME),
                    label=f"POST_COMBO {endpoint} fields={','.join(body)}",
                    tool_call=ToolCall(
                        ToolName.CURL_POST,
                        url=urljoin(base_url, endpoint),
                        data=body,
                    ),
                    required_slots=(KK.BASE_URL, KK.ENDPOINT, KK.PARAM_NAME, KK.PROBE_VALUE),
                    tried_key=f"POST_COMBO:{endpoint}:{index}:{tuple(body.items())}",
                )
            )

    for username in store.values(KK.USERNAME):
        for password in store.values(KK.PASSWORD_CANDIDATE):
            for login_path in _limited(store.values(KK.PATH), MAX_LOGIN_PATHS):
                url = urljoin(base_url, login_path)
                candidates.append(
                    ActionCandidate(
                        template=ActionTemplate.HTTP_POST_LOGIN,
                        bindings={KK.USERNAME: username, KK.PASSWORD_CANDIDATE: password, KK.PATH: login_path},
                        policy=PolicyView(What.FORM_POST, How.AUTH_ATTEMPT, Where.KK_USERNAME),
                        label=f"POST {login_path} as {username}",
                        tool_call=ToolCall(
                            ToolName.CURL_POST,
                            url=url,
                            data={"username": username, "password": password},
                        ),
                        required_slots=(KK.BASE_URL, KK.USERNAME, KK.PASSWORD_CANDIDATE, KK.PATH),
                        tried_key=f"POST:{login_path}:{username}:{password}",
                    )
                )

    for auth_path in store.values(KK.AUTH_PATH):
        for cookie in store.values(KK.SESSION_COOKIE):
            url = urljoin(base_url, auth_path)
            candidates.append(
                ActionCandidate(
                    template=ActionTemplate.HTTP_AUTH_GET,
                    bindings={KK.AUTH_PATH: auth_path, KK.SESSION_COOKIE: cookie},
                    policy=PolicyView(What.AUTHENTICATED_GET, How.AUTHENTICATED, Where.KK_AUTH_PATH),
                    label=f"AUTH GET {auth_path}",
                    tool_call=ToolCall(ToolName.CURL_GET, url=url, headers={"Cookie": cookie}),
                    required_slots=(KK.BASE_URL, KK.AUTH_PATH, KK.SESSION_COOKIE),
                    tried_key=f"AUTH_GET:{auth_path}:{cookie}",
                )
            )

    return _dedupe_candidates(candidates)


def generate_json_api_candidates(store: KnowledgeStore) -> list[ActionCandidate]:
    base_url = store.first(KK.BASE_URL)
    if not base_url:
        return []
    candidates: list[ActionCandidate] = []
    param_candidates = _limited(store.values(KK.PARAM_NAME), MAX_PARAM_NAMES_PER_ENDPOINT)
    probe_candidates = _limited(store.values(KK.PROBE_VALUE), MAX_PROBE_VALUES_PER_ENDPOINT)

    for endpoint in store.values(KK.ENDPOINT):
        url = urljoin(base_url, endpoint)
        for method, template in [
            ("POST", ActionTemplate.HTTP_JSON_POST),
            ("PUT", ActionTemplate.HTTP_JSON_PUT),
            ("PATCH", ActionTemplate.HTTP_JSON_PATCH),
        ]:
            for index, body in enumerate(_json_bodies(param_candidates, probe_candidates)):
                candidates.append(
                    ActionCandidate(
                        template=template,
                        bindings={
                            KK.ENDPOINT: endpoint,
                            KK.PARAM_NAME: ",".join(body),
                            KK.PROBE_VALUE: ",".join(body.values()),
                        },
                        policy=PolicyView(What.FORM_POST, How.PROBE_VALUE, Where.KK_PARAM_NAME),
                        label=f"JSON_{method} {endpoint} fields={','.join(body)}",
                        tool_call=ToolCall(
                            ToolName.CURL_JSON,
                            url=url,
                            data=body,
                            method=method,
                        ),
                        required_slots=(KK.BASE_URL, KK.ENDPOINT, KK.PARAM_NAME, KK.PROBE_VALUE),
                        tried_key=f"JSON_{method}:{endpoint}:{index}:{tuple(body.items())}",
                    )
                )
        candidates.append(
            ActionCandidate(
                template=ActionTemplate.HTTP_JSON_DELETE,
                bindings={KK.ENDPOINT: endpoint},
                policy=PolicyView(What.FORM_POST, How.PROBE_VALUE, Where.KK_ENDPOINT),
                label=f"JSON_DELETE {endpoint}",
                tool_call=ToolCall(ToolName.CURL_JSON, url=url, method="DELETE"),
                required_slots=(KK.BASE_URL, KK.ENDPOINT),
                tried_key=f"JSON_DELETE:{endpoint}",
            )
        )

    return _dedupe_candidates(candidates)


def generate_input_mutation_candidates(store: KnowledgeStore) -> list[ActionCandidate]:
    base_url = store.first(KK.BASE_URL)
    if not base_url:
        return []
    candidates: list[ActionCandidate] = []
    param_candidates = _limited(store.values(KK.PARAM_NAME), MAX_PARAM_NAMES_PER_ENDPOINT)
    mutation_values = _mutated_values(store.values(KK.PROBE_VALUE))[:MAX_MUTATED_VALUES]

    for endpoint in store.values(KK.ENDPOINT):
        endpoint_count = 0
        for param_name in param_candidates:
            for value in mutation_values:
                if endpoint_count >= MAX_MUTATION_CANDIDATES_PER_ENDPOINT:
                    break
                suffix = f"{endpoint}?{param_name}={value}"
                candidates.append(
                    ActionCandidate(
                        template=ActionTemplate.HTTP_MUTATION_QUERY,
                        bindings={KK.ENDPOINT: endpoint, KK.PARAM_NAME: param_name, KK.PROBE_VALUE: value},
                        policy=PolicyView(What.QUERY_PROBE, How.PROBE_VALUE, Where.KK_PARAM_NAME),
                        label=f"MUTATE_QUERY {endpoint}?{param_name}={value}",
                        tool_call=ToolCall(ToolName.CURL_GET, url=urljoin(base_url, suffix)),
                        required_slots=(KK.BASE_URL, KK.ENDPOINT, KK.PARAM_NAME),
                        tried_key=f"MUTATE_QUERY:{endpoint}:{param_name}:{value}",
                    )
                )
                candidates.append(
                    ActionCandidate(
                        template=ActionTemplate.HTTP_MUTATION_POST,
                        bindings={KK.ENDPOINT: endpoint, KK.PARAM_NAME: param_name, KK.PROBE_VALUE: value},
                        policy=PolicyView(What.FORM_POST, How.PROBE_VALUE, Where.KK_PARAM_NAME),
                        label=f"MUTATE_POST {endpoint} {param_name}={value}",
                        tool_call=ToolCall(ToolName.CURL_POST, url=urljoin(base_url, endpoint), data={param_name: value}),
                        required_slots=(KK.BASE_URL, KK.ENDPOINT, KK.PARAM_NAME),
                        tried_key=f"MUTATE_POST:{endpoint}:{param_name}:{value}",
                    )
                )
                candidates.append(
                    ActionCandidate(
                        template=ActionTemplate.HTTP_MUTATION_JSON,
                        bindings={KK.ENDPOINT: endpoint, KK.PARAM_NAME: param_name, KK.PROBE_VALUE: value},
                        policy=PolicyView(What.FORM_POST, How.PROBE_VALUE, Where.KK_PARAM_NAME),
                        label=f"MUTATE_JSON {endpoint} {param_name}={value}",
                        tool_call=ToolCall(
                            ToolName.CURL_JSON,
                            url=urljoin(base_url, endpoint),
                            data={param_name: value},
                            method="POST",
                        ),
                        required_slots=(KK.BASE_URL, KK.ENDPOINT, KK.PARAM_NAME),
                        tried_key=f"MUTATE_JSON:{endpoint}:{param_name}:{value}",
                    )
                )
                endpoint_count += 3

    return _dedupe_candidates(candidates)


def generate_file_surface_candidates(store: KnowledgeStore) -> list[ActionCandidate]:
    base_url = store.first(KK.BASE_URL)
    if not base_url:
        return []
    candidates: list[ActionCandidate] = []
    paths = [path for path in store.values(KK.PATH) if _looks_file_surface_path(path)]
    for path in paths:
        for candidate_path in _file_surface_paths(path):
            candidates.append(
                ActionCandidate(
                    template=ActionTemplate.HTTP_FILE_SURFACE_GET,
                    bindings={KK.PATH: candidate_path},
                    policy=PolicyView(What.HTTP_GET, How.NORMAL, Where.KK_PATH),
                    label=f"FILE_GET {candidate_path}",
                    tool_call=ToolCall(ToolName.CURL_GET, url=urljoin(base_url, candidate_path)),
                    required_slots=(KK.BASE_URL, KK.PATH),
                    tried_key=f"FILE_GET:{candidate_path}",
                )
            )
            if len(candidates) >= MAX_FILE_SURFACE_CANDIDATES:
                return _dedupe_candidates(candidates)
    return _dedupe_candidates(candidates)


def _dedupe_candidates(candidates: list[ActionCandidate]) -> list[ActionCandidate]:
    seen: set[str] = set()
    output: list[ActionCandidate] = []
    for candidate in candidates:
        key = candidate.tried_key or candidate.label
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return output


def _limited(values: list[str], limit: int) -> list[str]:
    return values[:limit]


def _combo_bodies(param_names: list[str], probe_values: list[str]) -> list[dict[str, str]]:
    if len(param_names) < 2 or not probe_values:
        return []
    names = param_names[:MAX_COMBO_POST_FIELDS]
    bodies: list[dict[str, str]] = []
    for offset in range(min(MAX_COMBO_POSTS_PER_ENDPOINT, len(probe_values))):
        body = {
            name: probe_values[(index + offset) % len(probe_values)]
            for index, name in enumerate(names)
        }
        if body not in bodies:
            bodies.append(body)
    return bodies


def _json_bodies(param_names: list[str], probe_values: list[str]) -> list[dict[str, str]]:
    if not param_names or not probe_values:
        return []
    names = param_names[:MAX_JSON_FIELDS]
    bodies: list[dict[str, str]] = []
    for name in names:
        bodies.append({name: probe_values[0]})
    for body in _combo_bodies(names, probe_values):
        bodies.append(body)
    output: list[dict[str, str]] = []
    for body in bodies:
        if body not in output:
            output.append(body)
        if len(output) >= MAX_JSON_BODIES_PER_ENDPOINT:
            break
    return output


def _mutated_values(observed: list[str]) -> list[str]:
    seeds = [
        "",
        "0",
        "1",
        "-1",
        "9999",
        "true",
        "false",
        "null",
        "undefined",
        "test",
        "a@example.test",
        "'",
        '"',
        "<script>alert(1)</script>",
        "../",
        "../../",
    ]
    values: list[str] = []
    for value in [*observed, *seeds]:
        if value not in values:
            values.append(value)
        if value and value.isdigit():
            for number in [str(int(value) - 1), str(int(value) + 1)]:
                if number not in values:
                    values.append(number)
    return values


def _looks_file_surface_path(path: str) -> bool:
    lowered = path.lower()
    return (
        "." in lowered.rsplit("/", 1)[-1]
        or lowered.endswith("/")
        or lowered in {"/ftp", "/assets", "/public", "/files", "/uploads"}
    )


def _file_surface_paths(path: str) -> list[str]:
    output = [path]
    clean = path.rstrip("/")
    if clean and "." in clean.rsplit("/", 1)[-1]:
        for suffix in [".bak", ".old", "~", "%00"]:
            output.append(f"{clean}{suffix}")
    if clean:
        for name in ["package.json", "robots.txt", "security.txt", ".well-known/security.txt"]:
            output.append(f"{clean}/{name}")
    deduped: list[str] = []
    for item in output:
        if item not in deduped:
            deduped.append(item)
    return deduped
