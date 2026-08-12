from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from types import MethodType
from typing import Any, Generic, TypeVar, overload


T = TypeVar("T")


class IndexableReplayRing(Sequence[T], Generic[T]):
    """Fixed-capacity FIFO replay with O(1) random indexing.

    Logical iteration order is exactly deque(maxlen=N): oldest to newest. The
    active learners use Python's existing ``random.sample`` / ``randrange`` with
    this sequence, so RNG calls and selected logical rows stay unchanged while
    eliminating repeated deque traversal and full ``list(replay)`` copies.
    """

    __slots__ = ("_storage", "_start", "_size", "maxlen")

    def __init__(self, maxlen: int, values: Iterable[T] = ()) -> None:
        capacity = int(maxlen)
        if capacity <= 0:
            raise ValueError("replay maxlen must be positive")
        self.maxlen = capacity
        self._storage: list[T | None] = [None] * capacity
        self._start = 0
        self._size = 0
        self.extend(values)

    def __len__(self) -> int:
        return self._size

    def _physical(self, logical: int) -> int:
        index = int(logical)
        if index < 0:
            index += self._size
        if not 0 <= index < self._size:
            raise IndexError("replay index out of range")
        return (self._start + index) % self.maxlen

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[T, ...]: ...

    def __getitem__(self, index: int | slice) -> T | tuple[T, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._size)
            return tuple(self[item] for item in range(start, stop, step))
        value = self._storage[self._physical(index)]
        if value is None:  # pragma: no cover - internal invariant
            raise RuntimeError("replay ring contains an empty live slot")
        return value

    def __iter__(self) -> Iterator[T]:
        for index in range(self._size):
            yield self[index]

    def append(self, value: T) -> None:
        if self._size < self.maxlen:
            position = (self._start + self._size) % self.maxlen
            self._storage[position] = value
            self._size += 1
            return
        self._storage[self._start] = value
        self._start = (self._start + 1) % self.maxlen

    def extend(self, values: Iterable[T]) -> None:
        for value in values:
            self.append(value)

    def clear(self) -> None:
        self._storage = [None] * self.maxlen
        self._start = 0
        self._size = 0


def _as_indexable_replay(replay: object, *, fallback_capacity: int) -> IndexableReplayRing[Any]:
    capacity = getattr(replay, "maxlen", None)
    if capacity is None:
        capacity = int(fallback_capacity)
    return IndexableReplayRing(int(capacity), replay)


def _install_dqn_indexed_sampling(dqn: object) -> None:
    dqn.replay = _as_indexable_replay(dqn.replay, fallback_capacity=50_000)
    dqn.performance_indexed_replay = True
    dqn.performance_full_replay_copies = 0

    def train_step(self: object) -> None:
        # random.sample sees the same logical sequence and therefore consumes the
        # same RNG draws as random.sample(list(old_deque), batch_size), without
        # materializing all replay rows on every gradient update.
        batch = self.randomizer.sample(self.replay, self.batch_size)
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
    prophecy.replay = _as_indexable_replay(
        prophecy.replay,
        fallback_capacity=int(prophecy.config.replay_capacity),
    )
    # Prophecy already samples by integer index. Replacing deque with the ring is
    # sufficient to change each lookup from O(N) traversal to O(1).
    prophecy.performance_indexed_replay = True


def _install_critic_indexed_sampling(critic: object) -> None:
    critic.replay = _as_indexable_replay(critic.replay, fallback_capacity=4_000)
    critic.performance_indexed_replay = True
    critic.performance_full_replay_copies = 0

    def train_step(self: object) -> None:
        batch = self.randomizer.sample(self.replay, self.batch_size)
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
    """Remove replay-size-dependent Python sampling overhead from active learners."""

    if getattr(agent, "current_indexable_replays", False):
        return agent
    _install_dqn_indexed_sampling(agent.dqn)
    _install_prophecy_indexed_sampling(agent.base_neural_prophecy)
    _install_critic_indexed_sampling(agent.critic)
    agent.current_indexable_replays = True
    return agent
