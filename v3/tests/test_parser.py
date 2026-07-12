from __future__ import annotations

import unittest

from apassr_tool.knowledge import KK
from apassr_tool.parser import parse_tool_result
from apassr_tool.tools import ToolResult


class ParserTests(unittest.TestCase):
    def parse(self, text: str):
        return parse_tool_result(ToolResult(tool="test", command=[], status=200, stdout=text))

    def test_extracts_paths_from_html(self) -> None:
        items = self.parse('<a href="/robots.txt">r</a><script src="/static/app.js"></script>')
        self.assertIn((KK.PATH, "/robots.txt"), items)
        self.assertIn((KK.PATH, "/static/app.js"), items)

    def test_ignores_template_placeholder_paths(self) -> None:
        items = self.parse('<a href="{{href}}">bad</a><a href="javascript:void(0)">bad</a>')
        self.assertNotIn((KK.PATH, "{{href}}"), items)
        self.assertNotIn((KK.PATH, "javascript:void(0)"), items)

    def test_does_not_extract_web_tech_from_plain_javascript(self) -> None:
        items = self.parse("const chunks = [warning-icon, data-icon, object Promise];")
        self.assertFalse([item for item in items if item[0] == KK.WEB_TECH])
    
    def test_extracts_form_input_names_as_parameter_candidates(self) -> None:
        items = self.parse('<form action="/login"><input name="username"><input name="password"></form>')
        self.assertIn((KK.PARAM_NAME, "username"), items)
        self.assertIn((KK.PARAM_NAME, "password"), items)

    def test_extracts_common_javascript_parameter_names(self) -> None:
        items = self.parse('const body = {"email": value, "password": value, "q": value};')
        self.assertIn((KK.PARAM_NAME, "email"), items)
        self.assertIn((KK.PARAM_NAME, "password"), items)
        self.assertIn((KK.PARAM_NAME, "q"), items)

    def test_extracts_robot_and_debug_hints(self) -> None:
        items = self.parse("Disallow: /debug\nadmin user id is 7\ntry /api/users?id=7\n")
        self.assertIn((KK.PATH, "/debug"), items)
        self.assertIn((KK.USER_ID, "7"), items)
        self.assertIn((KK.PROBE_VALUE, "7"), items)
        self.assertIn((KK.ENDPOINT, "/api/users"), items)
        self.assertIn((KK.QUERY_PARAM, "id=7"), items)
        self.assertIn((KK.PARAM_NAME, "id"), items)

    def test_extracts_rest_endpoints(self) -> None:
        items = self.parse('fetch("/rest/products/search?q=test"); fetch("/rest/user/login");')
        self.assertIn((KK.ENDPOINT, "/rest/products/search"), items)
        self.assertIn((KK.QUERY_PARAM, "q=test"), items)
        self.assertIn((KK.PARAM_NAME, "q"), items)
        self.assertIn((KK.PROBE_VALUE, "test"), items)
        self.assertIn((KK.ENDPOINT, "/rest/user/login"), items)

    def test_extracts_json_values_as_observed_probe_values(self) -> None:
        items = self.parse('HTTP/1.1 200\n\n{"id": 3, "email": "a@example.test", "username": "alice"}')
        self.assertIn((KK.USER_ID, "3"), items)
        self.assertIn((KK.USERNAME, "alice"), items)
        self.assertIn((KK.PROBE_VALUE, "3"), items)
        self.assertIn((KK.PROBE_VALUE, "a@example.test"), items)
        self.assertIn((KK.PROBE_VALUE, "alice"), items)

    def test_extracts_json_boolean_values_as_observed_probe_values(self) -> None:
        items = self.parse('HTTP/1.1 200\n\n{"isAdmin": true, "active": false}')
        self.assertIn((KK.PROBE_VALUE, "true"), items)
        self.assertIn((KK.PROBE_VALUE, "false"), items)

    def test_extracts_cookie_and_flag(self) -> None:
        items = self.parse("HTTP/1.1 200\nSet-Cookie: session=abc; Path=/\n\nFLAG{x}")
        self.assertIn((KK.SESSION_COOKIE, "session=abc"), items)
        self.assertIn((KK.FLAG, "FLAG{x}"), items)

    def test_extracts_allowed_http_methods(self) -> None:
        items = self.parse("HTTP/1.1 204\nAllow: GET, POST, OPTIONS\n\n")
        self.assertIn((KK.HTTP_METHOD, "GET"), items)
        self.assertIn((KK.HTTP_METHOD, "POST"), items)
        self.assertIn((KK.HTTP_METHOD, "OPTIONS"), items)

    def test_extracts_nmap_open_port_and_service(self) -> None:
        items = self.parse("8088/tcp open  http\n")
        self.assertIn((KK.PORT, "8088"), items)
        self.assertIn((KK.PORT_STATE, "8088/tcp:open"), items)
        self.assertIn((KK.SERVICE, "http"), items)

    def test_extracts_nmap_filtered_port_as_knowledge(self) -> None:
        items = self.parse("8088/tcp filtered radan-http\n")
        self.assertIn((KK.PORT_STATE, "8088/tcp:filtered"), items)
        self.assertIn((KK.SERVICE, "radan-http"), items)


if __name__ == "__main__":
    unittest.main()
