from __future__ import annotations

from types import SimpleNamespace

from aassr_v2.current_planner import CurrentFullyBatchedImaginationTree
from aassr_v2.pentest_agent_main_test import AGENT_STATE_SIZE
from aassr_v2.policy import ScoredAction
from aassr_v2.types import Action, StateSnapshot


def test_deeper_branching_deduplicates_relational_aliases_before_top_k() -> None:
    aliases = tuple(
        Action(
            "request",
            parameters={
                "route_id": f"route-{index:02d}",
                "profile_id": "profile-browse",
            },
        )
        for index in range(3)
    )
    catalog = Action(
        "request",
        parameters={"route_id": "route-09", "profile_id": "profile-browse"},
    )
    facts = {
        "known_profile:profile-browse",
        "known_route:route-09",
        "observed_route_role:route-09:catalog",
    }
    facts.update(
        f"known_route:route-{index:02d}" for index in range(3)
    )
    state = StateSnapshot(
        (0.0,) * AGENT_STATE_SIZE,
        facts=frozenset(facts),
        available_actions=(*aliases, catalog),
    )
    ranked = (
        ScoredAction(aliases[0], 1.0),
        ScoredAction(aliases[1], 1.0),
        ScoredAction(aliases[2], 1.0),
        ScoredAction(catalog, 0.9),
    )

    planner = object.__new__(CurrentFullyBatchedImaginationTree)
    planner.config = SimpleNamespace(branching_factor=2, expand_all_root_actions=True)
    planner.structural_alias_rows_removed = 0

    root = planner._structural_limit(state, ranked, depth=1)
    deeper = planner._structural_limit(state, ranked, depth=2)

    assert root == ranked
    assert tuple(item.action for item in deeper) == (aliases[0], catalog)
    assert planner.structural_alias_rows_removed == 2
