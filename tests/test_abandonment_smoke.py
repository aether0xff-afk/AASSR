from __future__ import annotations

from types import SimpleNamespace

from aassr_v2.abandonment_smoke import CriticAbandonmentProbe
from aassr_v2.branch_critic import BranchCriticStep
from aassr_v2.types import Action, StateSnapshot


STATE = StateSnapshot((0.0,), frozenset(), (Action("move"),))
ACTION = Action("move")


class _FakeCritic:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def initial_memory(self) -> tuple[int, ...]:
        return ()

    def score_step(self, *args, memory=None, **kwargs) -> BranchCriticStep:
        del args, kwargs
        history = tuple(memory or ()) + (1,)
        return BranchCriticStep(next(self.values), history)


class _FakeProphecy:
    def confidence(self, state: StateSnapshot, action: Action) -> float:
        del state, action
        return 1.0


class _FakeAgent:
    def __init__(self, values: list[float], *, ready: bool = True) -> None:
        self.critic = _FakeCritic(values)
        self.critic_ready = ready
        self.agent = SimpleNamespace(prophecy=_FakeProphecy())


def test_abandonment_requires_minimum_steps_and_patience() -> None:
    probe = CriticAbandonmentProbe(
        _FakeAgent([0.10, 0.10]),
        threshold=0.20,
        minimum_steps=2,
        patience=2,
    )
    probe.observe(STATE, ACTION, STATE)
    first = probe.signal()
    assert not first.should_abandon
    assert first.reason == "minimum_steps"

    probe.observe(STATE, ACTION, STATE)
    second = probe.signal()
    assert second.should_abandon
    assert second.reason == "critic_low_success_probability"


def test_abandonment_streak_resets_when_probability_recovers() -> None:
    probe = CriticAbandonmentProbe(
        _FakeAgent([0.10, 0.80, 0.10, 0.10]),
        threshold=0.20,
        minimum_steps=0,
        patience=2,
    )
    probe.observe(STATE, ACTION, STATE)
    assert not probe.signal().should_abandon
    probe.observe(STATE, ACTION, STATE)
    assert probe.signal().low_probability_streak == 0
    probe.observe(STATE, ACTION, STATE)
    assert not probe.signal().should_abandon
    probe.observe(STATE, ACTION, STATE)
    assert probe.signal().should_abandon


def test_unready_critic_never_declares_abandonment() -> None:
    probe = CriticAbandonmentProbe(
        _FakeAgent([0.0, 0.0], ready=False),
        threshold=0.50,
        minimum_steps=0,
        patience=1,
    )
    probe.observe(STATE, ACTION, STATE)
    signal = probe.signal()
    assert not signal.should_abandon
    assert signal.reason == "critic_not_ready"
