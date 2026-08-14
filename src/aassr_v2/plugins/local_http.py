from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from http.cookiejar import CookieJar
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import (
    HTTPCookieProcessor,
    HTTPRedirectHandler,
    Request,
    build_opener,
)

from ..core.plugin_contract import (
    ActionCommand,
    ActionParameter,
    ActionSpec,
    ObservationField,
    PluginObservation,
    PluginSchema,
    PluginStepResult,
    TemporalKind,
    ValueKind,
)


LOCAL_HTTP_PLUGIN_ID = "local-http-minimal-v1"
LOCAL_HTTP_PLUGIN_VERSION = "minimal-public-io-v1"


class _PublicHtmlParser(HTMLParser):
    """Extract browser-visible mechanical affordances from one response."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: set[str] = set()
        self.forms: list[dict[str, Any]] = []
        self._active_form: dict[str, Any] | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        values = {key.lower(): value for key, value in attrs}
        lowered = tag.lower()
        if lowered == "a" and values.get("href"):
            self.links.add(urljoin(self.base_url, str(values["href"])))
            return
        if lowered == "form":
            self._active_form = {
                "action": urljoin(
                    self.base_url,
                    str(values.get("action") or self.base_url),
                ),
                "method": str(values.get("method") or "GET").upper(),
                "inputs": [],
            }
            return
        if lowered == "input" and self._active_form is not None:
            name = values.get("name")
            if name:
                self._active_form["inputs"].append(str(name))

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "form" and self._active_form is not None:
            self.forms.append(self._active_form)
            self._active_form = None


class _SameOriginRedirectHandler(HTTPRedirectHandler):
    """Reject a redirect before urllib can leave the configured loopback origin."""

    def __init__(self, validator) -> None:
        super().__init__()
        self._validator = validator

    def redirect_request(
        self,
        req,
        fp,
        code,
        msg,
        headers,
        newurl,
    ):
        self._validator(str(newurl))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@dataclass(frozen=True, slots=True)
class LocalHttpConfig:
    base_url: str
    timeout_seconds: float = 5.0
    reward_header: str = "X-AASSR-Reward"
    termination_header: str = "X-AASSR-Terminated"
    truncation_header: str = "X-AASSR-Truncated"


class LocalHttpPlugin:
    """Minimal loopback-only real HTTP I/O plugin.

    The plugin owns only HTTP mechanics. It does not accumulate discovered links,
    infer page roles, rank requests, install models, or retain problem-solving
    knowledge. CookieJar state is retained because cookies are part of HTTP
    transport/session mechanics required to perform later requests.

    Reward/termination transport headers are consumed as standardized environment
    control signals and are deliberately removed from the learner-visible header
    observation. This keeps benchmark instrumentation separate from world data.

    Mechanical ``value_space`` labels only express protocol compatibility: URL
    observations can fill URL parameters and form-payload templates can fill a
    request body. They do not mark any URL/payload as good, bad, target, or decoy.
    """

    def __init__(self, config: LocalHttpConfig | str) -> None:
        self.config = (
            config if isinstance(config, LocalHttpConfig) else LocalHttpConfig(config)
        )
        self._base = self._validate_loopback_url(self.config.base_url)
        self._origin = (
            self._base.scheme,
            self._base.hostname,
            self._base.port,
        )
        self._jar = CookieJar()
        self._opener = build_opener(
            HTTPCookieProcessor(self._jar),
            _SameOriginRedirectHandler(self._validate_same_origin),
        )
        self._last_result: PluginStepResult | None = None
        self._schema = PluginSchema(
            plugin_id=LOCAL_HTTP_PLUGIN_ID,
            version=LOCAL_HTTP_PLUGIN_VERSION,
            observations=(
                ObservationField(
                    "current_url",
                    ValueKind.ENTITY,
                    TemporalKind.STATE,
                    value_space="url",
                    description="현재 공개 URL",
                ),
                ObservationField(
                    "status",
                    ValueKind.CATEGORICAL,
                    TemporalKind.EVENT,
                    value_space="response-status",
                    description="공개 응답 상태 값",
                ),
                ObservationField(
                    "headers",
                    ValueKind.MAPPING,
                    TemporalKind.EVENT,
                    value_space="response-headers",
                    description="공개 응답 헤더(환경 제어용 예약 헤더 제외)",
                ),
                ObservationField(
                    "body",
                    ValueKind.TEXT,
                    TemporalKind.EVENT,
                    value_space="response-body",
                    description="공개 응답 본문",
                ),
                ObservationField(
                    "cookies",
                    ValueKind.MAPPING,
                    TemporalKind.STATE,
                    value_space="cookie-jar",
                    description="클라이언트가 보유한 공개 쿠키",
                ),
                ObservationField(
                    "links",
                    ValueKind.SET,
                    TemporalKind.STATE,
                    item_kind=ValueKind.ENTITY,
                    value_space="url",
                    description="현재 응답에서 기계적으로 발견한 URL 집합",
                ),
                ObservationField(
                    "form_payload_templates",
                    ValueKind.SET,
                    TemporalKind.STATE,
                    item_kind=ValueKind.TEXT,
                    value_space="form-payload",
                    description="현재 응답 form input 이름으로 만든 빈 payload 형식",
                ),
                ObservationField(
                    "latency_ms",
                    ValueKind.SCALAR,
                    TemporalKind.MEASUREMENT,
                    value_space="round-trip-ms",
                    description="클라이언트에서 측정한 왕복 시간",
                ),
            ),
            actions=(
                ActionSpec(
                    "request",
                    parameters=(
                        ActionParameter(
                            "method",
                            ValueKind.CATEGORICAL,
                            enum_values=("GET", "POST"),
                            value_space="http-method",
                            description="요청 method",
                        ),
                        ActionParameter(
                            "url",
                            ValueKind.ENTITY,
                            value_space="url",
                            description="요청 대상 URL",
                        ),
                        ActionParameter(
                            "body",
                            ValueKind.TEXT,
                            required=False,
                            value_space="form-payload",
                            description="선택적 요청 body",
                        ),
                    ),
                    description="실제 loopback HTTP 요청 보내기",
                ),
            ),
        )

    @property
    def schema(self) -> PluginSchema:
        return self._schema

    @staticmethod
    def _validate_loopback_url(value: str):
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("local HTTP plugin requires http or https")
        host = (parsed.hostname or "").lower()
        if host not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("local HTTP plugin is intentionally loopback-only")
        return parsed

    def _validate_same_origin(self, value: str) -> str:
        parsed = self._validate_loopback_url(value)
        origin = (parsed.scheme, parsed.hostname, parsed.port)
        if origin != self._origin:
            raise ValueError("request URL must stay on the configured local origin")
        return value

    def _cookies(self) -> Mapping[str, str]:
        return {
            str(cookie.name): str(cookie.value)
            for cookie in self._jar
        }

    def _control_header_names(self) -> frozenset[str]:
        return frozenset(
            {
                self.config.reward_header.lower(),
                self.config.termination_header.lower(),
                self.config.truncation_header.lower(),
            }
        )

    def _public_headers(self, headers: Mapping[str, str]) -> Mapping[str, str]:
        reserved = self._control_header_names()
        return {
            str(key): str(value)
            for key, value in headers.items()
            if str(key).lower() not in reserved
        }

    def reset(self, *, seed: int | None = None) -> PluginStepResult:
        # The plugin does not own hidden server state; seed is intentionally not
        # translated into a task-specific reset rule.
        del seed
        self._jar.clear()
        result = PluginStepResult(
            observation=PluginObservation(
                values={
                    "current_url": self.config.base_url,
                    "status": None,
                    "headers": {},
                    "body": "",
                    "cookies": {},
                    "links": (),
                    "form_payload_templates": (),
                    "latency_ms": 0.0,
                }
            )
        )
        self._last_result = result
        return result

    def _public_affordances(
        self,
        *,
        current_url: str,
        body: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        parser = _PublicHtmlParser(current_url)
        try:
            parser.feed(body)
        except Exception:
            return (), ()

        links: set[str] = set()
        for link in parser.links:
            try:
                links.add(self._validate_same_origin(link))
            except ValueError:
                continue

        payloads: set[str] = set()
        for form in parser.forms:
            action = str(form["action"])
            try:
                links.add(self._validate_same_origin(action))
            except ValueError:
                continue
            names = tuple(
                sorted(
                    str(name)
                    for name in form.get("inputs", ())
                    if name
                )
            )
            if names:
                payloads.add(urlencode({name: "" for name in names}))
        return tuple(sorted(links)), tuple(sorted(payloads))

    @staticmethod
    def _header_truth(value: str | None) -> bool:
        return str(value or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "done",
        }

    def step(self, command: ActionCommand) -> PluginStepResult:
        command.to_action(self.schema)
        if command.action_id != "request":
            raise ValueError(f"unsupported local command {command.action_id!r}")

        method = str(command.arguments["method"]).upper()
        url = self._validate_same_origin(str(command.arguments["url"]))
        body_arg = command.arguments.get("body")
        data = None if body_arg is None else str(body_arg).encode("utf-8")
        request = Request(url, data=data, method=method)
        if data is not None:
            request.add_header(
                "Content-Type",
                "application/x-www-form-urlencoded",
            )

        started = time.perf_counter()
        response_status: int | None = None
        response_headers: dict[str, str] = {}
        response_body = ""
        response_url = url
        error = False
        error_code: str | None = None

        try:
            response = self._opener.open(
                request,
                timeout=float(self.config.timeout_seconds),
            )
            response_status = int(response.status)
            response_headers = {
                str(key).lower(): str(value)
                for key, value in response.headers.items()
            }
            response_url = str(response.geturl())
            response_body = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            # 4xx/5xx is a public environment result, not an internal Core error.
            response_status = int(exc.code)
            response_headers = {
                str(key).lower(): str(value)
                for key, value in exc.headers.items()
            }
            response_url = str(exc.geturl())
            response_body = exc.read().decode("utf-8", errors="replace")
        except (URLError, TimeoutError, OSError) as exc:
            error = True
            error_code = exc.__class__.__name__
            response_body = str(exc)

        latency_ms = (time.perf_counter() - started) * 1000.0
        links, payloads = self._public_affordances(
            current_url=response_url,
            body=response_body,
        )

        reward = 0.0
        terminated = False
        truncated = False
        if response_headers:
            raw_reward = response_headers.get(self.config.reward_header.lower())
            if raw_reward is not None:
                try:
                    reward = float(raw_reward)
                except ValueError:
                    reward = 0.0
            terminated = self._header_truth(
                response_headers.get(self.config.termination_header.lower())
            )
            truncated = self._header_truth(
                response_headers.get(self.config.truncation_header.lower())
            )
        if terminated:
            truncated = False

        result = PluginStepResult(
            observation=PluginObservation(
                values={
                    "current_url": response_url,
                    "status": response_status,
                    "headers": self._public_headers(response_headers),
                    "body": response_body,
                    "cookies": self._cookies(),
                    "links": links,
                    "form_payload_templates": payloads,
                    "latency_ms": latency_ms,
                }
            ),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            error=error,
            error_code=error_code,
            diagnostics={"wire_plugin": "loopback-only"},
        )
        self._last_result = result
        return result
