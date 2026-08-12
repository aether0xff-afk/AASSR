from __future__ import annotations

from collections import deque
import random

from aassr_v2.current_replay_performance import IndexableReplayRing


def test_indexable_replay_ring_matches_deque_fifo_and_index_semantics() -> None:
    expected = deque(maxlen=7)
    actual = IndexableReplayRing[int](7)
    for value in range(31):
        expected.append(value)
        actual.append(value)
        assert tuple(actual) == tuple(expected)
        assert len(actual) == len(expected)
        if expected:
            assert actual[0] == expected[0]
            assert actual[-1] == expected[-1]
    # The list mirror is deliberately disabled after rollover so a future run
    # beyond replay capacity keeps native deque O(1) eviction semantics.
    assert actual.performance_index_active is False


def test_indexable_replay_preserves_random_sample_rng_selection() -> None:
    for population_size in (64, 128, 256, 300, 1_000, 10_000):
        values = tuple(range(population_size))
        replay = IndexableReplayRing[int](population_size, values)
        for seed in (1, 7, 42, 100):
            old_rng = random.Random(seed)
            new_rng = random.Random(seed)
            expected = old_rng.sample(list(deque(values)), min(64, population_size))
            actual = new_rng.sample(replay.sampling_population, min(64, population_size))
            assert actual == expected
            # The next RNG draw must also agree; identical selected rows alone is
            # not enough if one sampling path consumed a different number of draws.
            assert new_rng.random() == old_rng.random()


def test_ring_overflow_keeps_same_logical_sampling_population_as_deque() -> None:
    expected = deque(maxlen=4_000)
    actual = IndexableReplayRing[int](4_000)
    for value in range(7_000):
        expected.append(value)
        actual.append(value)
    assert tuple(actual) == tuple(expected)
    assert actual.performance_index_active is False

    old_rng = random.Random(20260812)
    new_rng = random.Random(20260812)
    assert new_rng.sample(actual.sampling_population, 16) == old_rng.sample(list(expected), 16)
    assert new_rng.random() == old_rng.random()


def test_critic_tuple_snapshot_sampling_matches_old_list_population_and_rng() -> None:
    # The active Critic keeps its 4k deque and snapshots it once per changed
    # suffix set. random.sample(tuple(snapshot), k) must be exactly equivalent to
    # the historical random.sample(list(replay), k), including RNG consumption.
    values = deque(range(4_000), maxlen=4_000)
    for seed in (7, 42, 100, 20260812):
        old_rng = random.Random(seed)
        new_rng = random.Random(seed)
        expected = old_rng.sample(list(values), 16)
        actual = new_rng.sample(tuple(values), 16)
        assert actual == expected
        assert new_rng.random() == old_rng.random()
