from __future__ import annotations

from collections import deque
from types import MethodType
from typing import Any, Generic, Iterable, TypeVar, overload


T = TypeVar("T")


class IndexableReplayRing(deque[T], Generic[T]):
    """Deque-compatible FIFO with a fast logical-index mirror before capacity.

    Current pre-10k DQN/Prophecy replay capacity is 50k, so the active scaling
    run never evicts a row. While below capacity, integer indexing and sampling
    use a normal CPython list, avoiding deque middle traversal. If a future run
    exceeds capacity, the list mirror is disabled and the object falls back to
    native deque semantics instead of paying O(capacity) list-front deletion on
    every append.
    """

    def __init__(self, maxlen: int, values: Iterable[T] = ()) -> None:
        capacity = int(maxlen)
        if capacity <= 0:
            raise ValueError("replay maxlen must be positive")
        super().__init__(maxlen=capacity)
        self._performance_index: list[T] | None = []
        self.extend(values)

    @property
    def performance_index_active(self) -> bool:
        return self._performance_index is not None

    @property
    def sampling_population(self) -> list[T]:
        index = self._performance_index
        if index is not None:
            return index
        return list(super().__iter__())

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[T, ...]: ...

    def __getitem__(self, index: int | slice) -> T | tuple[T, ...]:
        mirror = self._performance_index
        if isinstance(index, slice):
            if mirror is not None:
                return tuple(mirror[index])
            return tuple(super().__iter__())[index]
        if mirror is not None:
            return mirror[index]
        return super().__getitem__(index)

    def append(self, value: T) -> None:
        mirror = self._performance_index
        # Crossing maxlen would require deleting mirror[0] every transition.
        # Disable the pre-capacity optimization instead and preserve deque's O(1)
        # eviction behavior for future >50k experiments.
        if mirror is not None and len(self) >= int(self.maxlen or 0):
            self._performance_index = None
            mirror = None
        super().append(value)
        if mirror is not None:
            mirror.append(value)

    def extend(self, values: Iterable[T]) -> None:
        for value in values:
            self.append(value)

    def clear(self) -> None:
        super().clear()
        self._performance_index = []


def _as_fast_replay(replay: object, *, fallback_capacity: int) -> IndexableReplayRing[Any]:
    capacity = getattr(replay, "maxlen", None)
    if capacity is None:
        capacity = int(fallback_capacity)
    return IndexableReplayRing(int(capacity), replay)


def _install_dqn_indexed_sampling(dqn: object) -> None:
    dqn.replay = _as_fast_replay(dqn.replay, fallback_capacity=50_000)
    dqn.performance_indexed_replay = True
    dqn.performance_full_replay_copies = 0

    def train_step(self: object) -> None:
        # random.sample receives the exact same ordered row population as the old
        # list(deque), with identical RNG consumption. Below capacity that list is
        # maintained incrementally rather than rebuilt for every gradient update.
        population = self.replay.sampling_population
        batch = self.randomizer.sample(population, self.batch_size)
        inputs = self._tensor([item[0] + item[1] for item in batch])
        predicted = self.online(inputs).squeeze(1)

        flat: list[tuple[float, ...]] = []
        owners: list[int] = []
        for index, (_, _, _, next_state, next_actions, terminal) in enumerate(batch):
            if terminal or not next_actions:
                continue
            flat.extend(next_state + features for features in next_actions)
            owners.extend([index] * len(next_actions))
            self.device_target_reductions += 1

        next_values = self.torch.full(
            (len(batch),),
            float("-inf"),
            dtype=self.torch.float32,
            device=self.device,
        )
        if flat:
            with self.torch.no_grad():
                scored = self.target(self._tensor(flat)).squeeze(1)
                owner_tensor = self._tensor(owners, dtype=self.torch.int64)
                next_values.scatter_reduce_(
                    0,
                    owner_tensor,
                    scored,
                    reduce="amax",
                    include_self=True,
                )
                self.fused_target_reduce_calls += 1
        next_values = self.torch.where(
            self.torch.isfinite(next_values),
            next_values,
            self.torch.zeros_like(next_values),
        )

        rewards = self._tensor([item[2] for item in batch])
        terminals = self._tensor([float(item[5]) for item in batch])
        targets = rewards + self.gamma * (1.0 - terminals) * next_values
        loss = self.loss(predicted, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.gradient_updates += 1
        if self.gradient_updates % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

    dqn._train_step = MethodType(train_step, dqn)


def _install_prophecy_indexed_sampling(prophecy: object) -> None:
    prophecy.replay = _as_fast_replay(
        prophecy.replay,
        fallback_capacity=int(prophecy.config.replay_capacity),
    )
    # Prophecy already samples by integer index. IndexableReplayRing redirects
    # those exact logical indices to its C-level list mirror while below maxlen.
    prophecy.performance_indexed_replay = True


def _install_critic_snapshot_sampling(critic: object) -> None:
    # Critic capacity is only 4k and naturally rolls over. Keep its native deque
    # FIFO but build the random-access population once per changed suffix set,
    # instead of once for each gradient step (two by default).
    critic.performance_indexed_replay = True
    critic.performance_full_replay_copies = 0
    critic._performance_replay_snapshot_revision = -1
    critic._performance_replay_snapshot = ()

    def train_step(self: object) -> None:
        revision = int(self.suffix_sequences)
        if self._performance_replay_snapshot_revision != revision:
            self._performance_replay_snapshot = tuple(self.replay)
            self._performance_replay_snapshot_revision = revision
            self.performance_full_replay_copies += 1
        batch = self.randomizer.sample(
            self._performance_replay_snapshot,
            self.batch_size,
        )
        lengths = [len(encoded) for encoded, _ in batch]
        max_length = max(lengths)
        hidden = self.torch.zeros(
            (len(batch), self.hidden_units),
            dtype=self.torch.float32,
            device=self.device,
        )
        loss_sums = self.torch.zeros(
            len(batch),
            dtype=self.torch.float32,
            device=self.device,
        )
        zero = (0.0,) * self.encoder.feature_size

        for step_index in range(max_length):
            rows = [
                encoded[step_index] if step_index < len(encoded) else zero
                for encoded, _ in batch
            ]
            hidden = self.gru(self._tensor(rows), hidden)
            predicted = self.torch.tanh(self.output(hidden).squeeze(1))
            target = self._tensor(
                [
                    targets[step_index] if step_index < len(targets) else 0.0
                    for _, targets in batch
                ]
            )
            per_row = self.nn.functional.smooth_l1_loss(
                predicted,
                target,
                reduction="none",
            )
            mask = self._tensor(
                [float(step_index < length) for length in lengths]
            )
            loss_sums = loss_sums + per_row * mask

        loss = (loss_sums / self._tensor(lengths)).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.nn.utils.clip_grad_norm_(
            tuple(self.gru.parameters()) + tuple(self.output.parameters()),
            5.0,
        )
        self.optimizer.step()
        self.gradient_updates += 1
        self._losses.append(float(loss.detach().cpu().item()))
        self.train_batch_calls += 1
        self.train_batch_time_steps += max_length
        self.train_batch_transition_rows += sum(lengths)

    critic._train_step = MethodType(train_step, critic)


def install_indexable_current_replays(agent: object) -> object:
    """Remove replay-size-dependent Python sampling work without changing rows."""

    if getattr(agent, "current_indexable_replays", False):
        return agent
    _install_dqn_indexed_sampling(agent.dqn)
    _install_prophecy_indexed_sampling(agent.base_neural_prophecy)
    _install_critic_snapshot_sampling(agent.critic)
    agent.current_indexable_replays = True
    return agent
