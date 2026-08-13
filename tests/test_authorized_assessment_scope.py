from __future__ import annotations

import pytest

from aassr_v2.adapters import AuthorizedAssessmentPlugin, DryRunTransport
from aassr_v2.types import Action


def _assessment(
    transport: DryRunTransport,
) -> AuthorizedAssessmentPlugin:
    return AuthorizedAssessmentPlugin(
        transport,
        allowlisted_targets=("lab.local", "1"),
        allowlisted_resources={
            "lab.local": ("report/current", "1"),
            "1": ("report/current",),
        },
    )


def _allowed_read(**overrides: object) -> Action:
    parameters: dict[str, object] = {
        "target": "lab.local",
        "resource": "report/current",
    }
    parameters.update(overrides)
    return Action("read", parameters=parameters)


def test_read_defaults_to_fail_closed_without_a_resource_allowlist() -> None:
    transport = DryRunTransport()
    assessment = AuthorizedAssessmentPlugin(
        transport,
        allowlisted_targets=("lab.local",),
    )

    outcome = assessment.execute(_allowed_read())

    assert outcome.error
    assert outcome.error_code == "resource_not_allowlisted"
    assert transport.calls == []


def test_scope_aliases_and_legacy_fields_never_reach_transport() -> None:
    transport = DryRunTransport()
    assessment = _assessment(transport)
    rejected = (
        Action(
            "read",
            target="outside.local",
            parameters={
                "target": "lab.local",
                "resource": "report/current",
            },
        ),
        _allowed_read(endpoint="outside.local"),
        Action(
            "connect",
            parameters={
                "endpoint": "lab.local",
                "target": "outside.local",
            },
        ),
        Action(
            "scan",
            parameters={
                "target": "lab.local",
                "resource": "report/current",
            },
        ),
    )

    outcomes = tuple(assessment.execute(action) for action in rejected)

    assert all(outcome.error for outcome in outcomes)
    assert all(
        outcome.error_code == "parameter_not_allowlisted"
        for outcome in outcomes
    )
    assert transport.calls == []


def test_mismatched_schema_metadata_and_non_string_verbs_are_rejected() -> None:
    class ReadLike:
        def __str__(self) -> str:
            return "read"

    transport = DryRunTransport()
    assessment = _assessment(transport)
    rejected = (
        Action(
            "read",
            parameters=_allowed_read().parameters,
            metadata={"plugin_id": "other-plugin"},
        ),
        Action(
            "read",
            parameters=_allowed_read().parameters,
            metadata={
                "plugin_id": assessment.plugin_id,
                "schema_id": f"{assessment.plugin_id}:scan",
            },
        ),
        Action(ReadLike(), parameters=_allowed_read().parameters),
        Action("READ", parameters=_allowed_read().parameters),
    )

    outcomes = tuple(assessment.execute(action) for action in rejected)

    assert all(outcome.error for outcome in outcomes)
    assert all(
        outcome.error_code == "action_not_allowlisted"
        for outcome in outcomes
    )
    assert transport.calls == []


def test_non_string_target_and_resource_cannot_match_by_string_coercion() -> None:
    transport = DryRunTransport()
    assessment = _assessment(transport)
    rejected = (
        Action(
            "read",
            parameters={"target": 1, "resource": "report/current"},
        ),
        Action(
            "read",
            parameters={"target": "lab.local", "resource": 1},
        ),
        Action("scan", parameters={"target": 1}),
        Action("connect", parameters={"endpoint": ["lab.local"]}),
    )

    outcomes = tuple(assessment.execute(action) for action in rejected)

    assert all(outcome.error for outcome in outcomes)
    assert transport.calls == []


def test_non_string_parameter_names_cannot_alias_canonical_scope_names() -> None:
    class TargetAlias:
        def __hash__(self) -> int:
            return hash("target")

        def __eq__(self, other: object) -> bool:
            return other == "target"

    transport = DryRunTransport()
    assessment = _assessment(transport)
    action = Action(
        "read",
        parameters={
            TargetAlias(): "lab.local",
            "resource": "report/current",
        },
    )

    outcome = assessment.execute(action)

    assert outcome.error
    assert outcome.error_code == "parameter_not_allowlisted"
    assert transport.calls == []


@pytest.mark.parametrize(
    "targets,resources",
    (
        ((1,), None),
        ("lab.local", None),
        (("lab.local",), {1: ("report/current",)}),
        (("lab.local",), {"lab.local": (1,)}),
        (("lab.local",), {"lab.local": "report/current"}),
    ),
)
def test_non_string_or_ambiguous_allowlist_configuration_is_rejected(
    targets: object,
    resources: object,
) -> None:
    with pytest.raises(TypeError):
        AuthorizedAssessmentPlugin(
            DryRunTransport(),
            allowlisted_targets=targets,  # type: ignore[arg-type]
            allowlisted_resources=resources,  # type: ignore[arg-type]
        )


def test_exact_allowlisted_read_is_the_only_read_sent_to_transport() -> None:
    transport = DryRunTransport()
    assessment = _assessment(transport)

    outcome = assessment.execute(_allowed_read())

    assert not outcome.error
    assert transport.calls == [
        (
            "read",
            {
                "target": "lab.local",
                "resource": "report/current",
            },
        )
    ]
