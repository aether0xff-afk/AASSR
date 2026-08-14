from __future__ import annotations

from aassr_v2.core.skills_core import CoreRelationalSkillLibrary
from aassr_v2.types import Action, StateSnapshot


class StubRepresentation:
    def action_structure(self, state, action):
        del state, action
        return (1.0,)

    def semantic_state_identity(self, state):
        del state
        return ("same-structural-state",)


def _state() -> StateSnapshot:
    return StateSnapshot(
        vector=(0.0,),
        available_actions=(
            Action("use", parameters={"candidate": "aaa"}),
            Action("use", parameters={"candidate": "zzz"}),
        ),
    )


def _install_one_step_skill(library: CoreRelationalSkillLibrary) -> None:
    library._templates["skill-0001"] = ((1.0,),)


def test_skill_grounding_prefers_learned_value_not_lexicographic_signature() -> None:
    def value(state, action):
        del state
        return 1.0 if action.parameters["candidate"] == "zzz" else 0.0

    library = CoreRelationalSkillLibrary(
        StubRepresentation(),
        primitive_value=value,
        seed=7,
    )
    _install_one_step_skill(library)

    selected = library.resolve_primitive("skill-0001", 0, _state())
    assert selected is not None
    assert selected.parameters["candidate"] == "zzz"
    assert library.ambiguous_groundings == 1
    assert library.value_groundings == 1
    assert library.symmetric_groundings == 0


def test_equal_value_grounding_is_seed_symmetric_not_always_first_signature() -> None:
    selected = set()
    for seed in range(32):
        library = CoreRelationalSkillLibrary(
            StubRepresentation(),
            primitive_value=lambda state, action: 0.0,
            seed=seed,
        )
        _install_one_step_skill(library)
        action = library.resolve_primitive("skill-0001", 0, _state())
        assert action is not None
        selected.add(str(action.parameters["candidate"]))
        assert library.symmetric_groundings == 1

    # The old implementation always returned lexicographic "aaa". The Core-owned
    # tie mechanism must leave initially symmetric concrete candidates symmetric
    # across Core seeds instead of baking that name preference into Skills.
    assert selected == {"aaa", "zzz"}
