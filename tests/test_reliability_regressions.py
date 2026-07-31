from __future__ import annotations

import math

import pytest

from aassr_v2.autonomous_benchmarks import OpaqueDependencyWorld
from aassr_v2.autonomous_experiment import (
    _world_seed,
    planned_autonomous_run_count,
)
from aassr_v2.counterexamples import (
    LearnableVsRandomWorld,
    NoisyInformationWrapper,
)
from aassr_v2.gru_prophecy import OnlineGRUProphecy
from aassr_v2.types import Action


def test_noise_snapshot_is_pure_and_rng_stable() -> None:
    environment = NoisyInformationWrapper(
        LearnableVsRandomWorld(seed=1),
        facts_per_step=5,
        seed=9,
    )
    rng_before = environment.random.getstate()
    first = environment.snapshot()
    second = environment.snapshot()
    rng_after = environment.random.getstate()

    assert first == second
    assert rng_before == rng_after


def test_noise_transition_records_removed_and_added_noise() -> None:
    environment = NoisyInformationWrapper(
        LearnableVsRandomWorld(seed=1),
        facts_per_step=4,
        seed=3,
    )
    before_noise = {
        fact for fact in environment.snapshot().facts if fact.startswith("noise:")
    }
    outcome = environment.step(Action("probe_stable"))
    after_noise = {
        fact for fact in outcome.snapshot.facts if fact.startswith("noise:")
    }

    assert len(before_noise) == 4
    assert len(after_noise) == 4
    assert after_noise <= outcome.added_facts
    assert before_noise <= outcome.removed_facts


def test_gru_confidence_remains_probability_mass() -> None:
    world = OpaqueDependencyWorld(2, seed=101)
    before = world.snapshot()
    action = before.available_actions[0]
    after = world.step(action).snapshot
    model = OnlineGRUProphecy(
        len(before.vector), hidden_size=8, action_feature_size=8, seed=4
    )

    for _ in range(12):
        model.reset_sequence()
        model.learn(before, action, after)

    confidence = model.confidence(before, action)
    predictions = model.predict(before, action, samples=1)
    uncertain = [
        item for item in predictions if item.source.endswith(":uncertain")
    ]

    assert 0.0 < confidence < 1.0
    assert math.isclose(
        sum(item.probability for item in predictions), 1.0, abs_tol=1e-9
    )
    assert uncertain
    assert math.isclose(
        uncertain[0].probability, 1.0 - confidence, abs_tol=1e-9
    )
    assert model.coverage(before, (action,)) == confidence


def test_train_and_unseen_evaluation_worlds_are_disjoint() -> None:
    environment = {
        "length": 4,
        "seed_offset": 4000,
        "eval_seed_offset": 4004000,
        "train_worlds_per_seed": 8,
        "eval_worlds_per_seed": 32,
    }
    train = {
        _world_seed(7, environment, mode="training", episode=index)
        for index in range(8)
    }
    seen = {
        _world_seed(7, environment, mode="seen", episode=index)
        for index in range(8)
    }
    unseen = {
        _world_seed(7, environment, mode="unseen", episode=index)
        for index in range(32)
    }

    assert train == seen
    assert train.isdisjoint(unseen)


def test_planned_rows_include_each_evaluation_mode() -> None:
    config = {
        "runner": "autonomous_main",
        "name": "split",
        "seeds": [1, 2],
        "train_episodes": 10,
        "eval_episodes": 3,
        "evaluation_modes": ["seen", "unseen"],
        "environments": [{"length": 4}],
        "conditions": [{"name": "full"}],
    }
    assert planned_autonomous_run_count(config) == 2 * (10 + 3 + 3)


def test_torch_gru_cpu_optional() -> None:
    pytest.importorskip("torch")
    from aassr_v2.torch_gru_prophecy import TorchGRUProphecy

    world = OpaqueDependencyWorld(2, seed=5)
    before = world.snapshot()
    action = before.available_actions[0]
    after = world.step(action).snapshot
    model = TorchGRUProphecy(
        len(before.vector), hidden_size=8, action_feature_size=8, device="cpu"
    )
    model.learn(before, action, after)
    predictions = model.predict(before, action, samples=1)
    assert math.isclose(
        sum(item.probability for item in predictions), 1.0, abs_tol=1e-6
    )


@pytest.mark.cuda
def test_torch_gru_uses_cuda_when_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from aassr_v2.torch_gru_prophecy import TorchGRUProphecy

    model = TorchGRUProphecy(5, hidden_size=8, action_feature_size=8, device="cuda")
    assert str(model.device).startswith("cuda")
