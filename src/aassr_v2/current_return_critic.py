from __future__ import annotations

from collections import deque
from typing import Any, Sequence

from .branch_critic import BranchCriticStep, CriticTransition
from .current_hardware import HardwareRelationalGRUBranchCritic
from .types import Action, StateSnapshot


RECENT_CRITIC_RETURN_WINDOW = 128


class ReturnAwareHardwareRelationalGRUBranchCritic(
    HardwareRelationalGRUBranchCritic
):
    """GRU branch critic aligned with the actual sparse {-1,0,+1} objective.

    The environment reward stays sparse. Planning can begin from any current real
    decision with a zero GRU memory, so training must expose that same contract.
    Every real episode therefore contributes every trajectory suffix. A suffix
    beginning at decision ``s`` receives the root-scale sparse return

        final_return * gamma ** (remaining_transitions - 1)

    at every prefix inside that suffix. Thus depth-1 and deeper imagined branches
    remain directly comparable, while the same zero-memory scorer is trained from
    every real decision state rather than only from episode start.

    Readiness is also tied to a bounded recent episode-return window. Cumulative
    historical success/failure counts are useful provenance, but cannot by
    themselves prove that a long-run FIFO learner still has both positive and
    negative sparse-return evidence in its current training regime.
    """

    name = "hardware-relational-gru-root-discounted-sparse-return-v4-recent-support"
    value_center = 0.0

    def __init__(self, seed: int, *, device: str = "cpu") -> None:
        super().__init__(seed, device=device)
        self._next_final_return = 0.0
        self._next_gamma = 1.0
        self.replay = deque(maxlen=4_000)
        self.recent_episode_returns: deque[float] = deque(
            maxlen=RECENT_CRITIC_RETURN_WINDOW
        )
        self.suffix_sequences = 0

    def set_episode_return(self, final_return: float, gamma: float) -> None:
        self._next_final_return = max(-1.0, min(1.0, float(final_return)))
        self._next_gamma = max(0.0, min(1.0, float(gamma)))

    def recent_signed_support(self) -> dict[str, int]:
        positive = sum(value > 0.0 for value in self.recent_episode_returns)
        negative = sum(value < 0.0 for value in self.recent_episode_returns)
        zero = len(self.recent_episode_returns) - positive - negative
        return {
            "window_capacity": RECENT_CRITIC_RETURN_WINDOW,
            "observed": len(self.recent_episode_returns),
            "positive": int(positive),
            "zero": int(zero),
            "negative": int(negative),
        }

    def observe_episode(
        self,
        trajectory: Sequence[CriticTransition],
        *,
        success: bool,
    ) -> None:
        del success
        encoded = tuple(self.encoder.encode(item) for item in trajectory)
        if encoded:
            final_return = float(self._next_final_return)
            self.recent_episode_returns.append(final_return)
            for start in range(len(encoded)):
                suffix = encoded[start:]
                root_return = final_return * self._next_gamma ** (
                    len(suffix) - 1
                )
                targets = (float(root_return),) * len(suffix)
                self.replay.append((suffix, targets))
                self.suffix_sequences += 1
        self.episodes += 1
        self.transitions += len(encoded)
        if len(self.replay) < self.batch_size:
            return
        for _ in range(self.gradient_steps_per_episode):
            self._train_step()

    def _train_step(self) -> None:
        batch = self.randomizer.sample(list(self.replay), self.batch_size)
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

    def score_step(
        self,
        before: StateSnapshot,
        action: Action,
        after: StateSnapshot,
        *,
        memory: Any = None,
        prophecy_confidence: float = 1.0,
    ) -> BranchCriticStep:
        self.scalar_score_calls += 1
        encoded = self.encoder.encode(
            CriticTransition(before, action, after, prophecy_confidence)
        )
        hidden = self.initial_memory() if memory is None else memory.to(self.device)
        with self.torch.no_grad():
            next_hidden = self.gru(self._tensor(encoded).unsqueeze(0), hidden)
            value = float(
                self.torch.tanh(self.output(next_hidden)[0, 0])
                .detach()
                .cpu()
                .item()
            )
        return BranchCriticStep(value, next_hidden.detach().clone())

    def score_step_batch(
        self,
        befores: Sequence[StateSnapshot],
        actions: Sequence[Action],
        afters: Sequence[StateSnapshot],
        memories: Sequence[Any],
        prophecy_confidences: Sequence[float],
    ) -> tuple[BranchCriticStep, ...]:
        length = len(befores)
        if not (
            length
            == len(actions)
            == len(afters)
            == len(memories)
            == len(prophecy_confidences)
        ):
            raise ValueError("critic batch inputs have different lengths")
        if not length:
            return ()

        encoded = tuple(
            self.encoder.encode(
                CriticTransition(before, action, after, confidence)
            )
            for before, action, after, confidence in zip(
                befores,
                actions,
                afters,
                prophecy_confidences,
                strict=True,
            )
        )
        hidden = self.torch.cat(
            [
                self.initial_memory()
                if memory is None
                else memory.to(self.device)
                for memory in memories
            ],
            dim=0,
        )
        with self.torch.no_grad():
            next_hidden = self.gru(self._tensor(encoded), hidden)
            host_values = (
                self.torch.tanh(self.output(next_hidden).squeeze(1))
                .detach()
                .cpu()
                .tolist()
            )
        self.batch_score_calls += 1
        self.batch_score_rows += length
        return tuple(
            BranchCriticStep(
                float(value),
                next_hidden[index : index + 1].detach().clone(),
            )
            for index, value in enumerate(host_values)
        )

    def hardware_stats(self) -> dict[str, Any]:
        return {
            **super().hardware_stats(),
            "value_center": self.value_center,
            "signed_sparse_return": 1,
            "zero_memory_suffix_training": 1,
            "suffix_sequences": self.suffix_sequences,
            "recent_signed_return_support": 1,
            **{
                f"recent_return_{key}": value
                for key, value in self.recent_signed_support().items()
            },
        }
