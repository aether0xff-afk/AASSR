from __future__ import annotations

from collections import deque
from typing import Any, Sequence
import io
import random

from ..branch_critic import BranchCriticStats, BranchCriticStep, CriticTransition
from ..types import Action, StateSnapshot
from .representation import SchemaDrivenRepresentation


class _RepresentationCriticEncoder:
    def __init__(self, representation: SchemaDrivenRepresentation) -> None:
        self.representation = representation
        self.feature_size = (
            representation.state_size * 2
            + representation.action_feature_size
            + 1
        )

    def encode(self, transition: CriticTransition) -> tuple[float, ...]:
        return (
            self.representation.state_vector(transition.before)
            + self.representation.action_features(
                transition.before,
                transition.action,
            )
            + self.representation.state_vector(transition.after)
            + (
                max(
                    0.0,
                    min(1.0, float(transition.prophecy_confidence)),
                ),
            )
        )


class SignedCoreGRUCritic:
    """Sequence critic trained only on the real sparse episode return."""

    name = "core-signed-gru-critic-v1"

    def __init__(
        self,
        representation: SchemaDrivenRepresentation,
        *,
        hidden_units: int = 64,
        replay_capacity: int = 4_000,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        gradient_steps_per_episode: int = 2,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("SignedCoreGRUCritic requires torch") from exc
        self.torch = torch
        self.nn = nn
        self.representation = representation
        self.encoder = _RepresentationCriticEncoder(representation)
        self.hidden_units = int(hidden_units)
        self.batch_size = int(batch_size)
        self.gradient_steps_per_episode = int(gradient_steps_per_episode)
        self.randomizer = random.Random(int(seed))
        torch.manual_seed(int(seed))
        self.device = torch.device(device)
        self.gru = nn.GRUCell(
            self.encoder.feature_size,
            self.hidden_units,
        ).to(self.device)
        self.output = nn.Linear(self.hidden_units, 1).to(self.device)
        self.optimizer = torch.optim.Adam(
            tuple(self.gru.parameters()) + tuple(self.output.parameters()),
            lr=float(learning_rate),
        )
        self.replay: deque[
            tuple[tuple[tuple[float, ...], ...], float]
        ] = deque(maxlen=int(replay_capacity))
        self.episodes = 0
        self.transitions = 0
        self.gradient_updates = 0
        self._losses: deque[float] = deque(maxlen=512)

    def initial_memory(self) -> Any:
        return self.torch.zeros(
            (1, self.hidden_units),
            dtype=self.torch.float32,
            device=self.device,
        )

    def observe_episode(
        self,
        trajectory: Sequence[CriticTransition],
        *,
        final_return: float,
    ) -> None:
        encoded = tuple(self.encoder.encode(item) for item in trajectory)
        if encoded:
            self.replay.append(
                (
                    encoded,
                    max(-1.0, min(1.0, float(final_return))),
                )
            )
        self.episodes += 1
        self.transitions += len(encoded)
        if len(self.replay) < self.batch_size:
            return
        for _ in range(self.gradient_steps_per_episode):
            self._train_step()

    def _episode_loss(
        self,
        encoded: Sequence[tuple[float, ...]],
        target: float,
    ) -> Any:
        hidden = self.initial_memory()
        values = []
        for item in encoded:
            tensor = self.torch.as_tensor(
                item,
                dtype=self.torch.float32,
                device=self.device,
            ).unsqueeze(0)
            hidden = self.gru(tensor, hidden)
            values.append(self.torch.tanh(self.output(hidden)[0, 0]))
        stacked = self.torch.stack(values)
        targets = self.torch.full_like(stacked, float(target))
        return self.nn.functional.smooth_l1_loss(stacked, targets)

    def _train_step(self) -> None:
        batch = self.randomizer.sample(list(self.replay), self.batch_size)
        loss = self.torch.stack(
            [self._episode_loss(encoded, target) for encoded, target in batch]
        ).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(
            tuple(self.gru.parameters()) + tuple(self.output.parameters()),
            5.0,
        )
        self.optimizer.step()
        self.gradient_updates += 1
        self._losses.append(float(loss.detach().cpu().item()))

    def score_step(
        self,
        before: StateSnapshot,
        action: Action,
        after: StateSnapshot,
        *,
        memory: Any = None,
        prophecy_confidence: float = 1.0,
    ) -> BranchCriticStep:
        encoded = self.encoder.encode(
            CriticTransition(before, action, after, prophecy_confidence)
        )
        hidden = self.initial_memory() if memory is None else memory
        with self.torch.no_grad():
            tensor = self.torch.as_tensor(
                encoded,
                dtype=self.torch.float32,
                device=self.device,
            ).unsqueeze(0)
            next_hidden = self.gru(tensor, hidden)
            value = float(
                self.torch.tanh(self.output(next_hidden)[0, 0]).item()
            )
        return BranchCriticStep(value, next_hidden.detach().clone())

    def stats(self) -> BranchCriticStats:
        handle = io.BytesIO()
        self.torch.save(
            {
                "gru": self.gru.state_dict(),
                "output": self.output.state_dict(),
            },
            handle,
        )
        parameters = tuple(self.gru.parameters()) + tuple(self.output.parameters())
        mean_loss = (
            sum(self._losses) / len(self._losses)
            if self._losses
            else 0.0
        )
        return BranchCriticStats(
            episodes=self.episodes,
            transitions=self.transitions,
            gradient_updates=self.gradient_updates,
            mean_loss=mean_loss,
            parameter_count=sum(parameter.numel() for parameter in parameters),
            model_bytes=len(handle.getvalue()),
        )


def build_signed_core_critic(
    representation: SchemaDrivenRepresentation,
    *,
    seed: int,
    device: str = "cpu",
) -> SignedCoreGRUCritic:
    return SignedCoreGRUCritic(
        representation,
        seed=int(seed),
        device=device,
    )
