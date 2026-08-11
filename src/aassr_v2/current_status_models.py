from __future__ import annotations

import random
from statistics import fmean
from typing import Any

from .current_relational_codec import ACTION_SLOT_COUNT, TERMINAL_CLASSES
from .current_relational_mixture_model import ConditionalMixtureRelationalProphecy
from .current_relational_model import RelationalStochasticProphecy
from .current_relational_state_v3 import STATUS_SIZE, STATUS_START_INDEX


STATUS_LOSS_WEIGHT = 0.5


class StatusAwareRelationalStochasticProphecy(RelationalStochasticProphecy):
    """Relational world model with explicit supervision for public HTTP status.

    Latest response status is decision-critical public information. Treating its
    eight channels as only 8/43 of a mean descriptor loss recreates the dilution
    that let 403/404 mistakes coexist with high aggregate semantic quality in the
    first repaired 2k run. The status slice therefore receives a dedicated BCE
    term in addition to the ordinary descriptor loss.
    """

    name = "current-relational-stochastic-world-model-v3-status-aware"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        descriptor_size = self.output_size - ACTION_SLOT_COUNT - TERMINAL_CLASSES
        if descriptor_size < STATUS_START_INDEX + STATUS_SIZE:
            raise ValueError(
                "status-aware Prophecy requires relational public state v3"
            )
        self._last_status_training_loss = 0.0

    def _train_step(self) -> None:
        status_losses: list[float] = []
        descriptor_size = self.output_size - ACTION_SLOT_COUNT - TERMINAL_CLASSES
        for model_index, (model, optimizer) in enumerate(
            zip(self.models, self.optimizers, strict=True)
        ):
            local = random.Random(
                (self.observations + 1) * 1_000_003
                + (self.gradient_updates + 1) * 97
                + model_index
            )
            batch = [
                self.replay[local.randrange(len(self.replay))]
                for _ in range(self.config.batch_size)
            ]
            output = model(self._tensor([row[0] for row in batch]))
            descriptor_logits = output[:, :descriptor_size]
            mask_start = descriptor_size
            mask_end = mask_start + ACTION_SLOT_COUNT
            mask_logits = output[:, mask_start:mask_end]
            terminal_logits = output[:, mask_end:]

            descriptor_target = self._tensor([row[1] for row in batch])
            descriptor_loss = self.nn.functional.smooth_l1_loss(
                self.torch.sigmoid(descriptor_logits),
                descriptor_target,
            )
            status_loss = self.nn.functional.binary_cross_entropy_with_logits(
                descriptor_logits[
                    :, STATUS_START_INDEX : STATUS_START_INDEX + STATUS_SIZE
                ],
                descriptor_target[
                    :, STATUS_START_INDEX : STATUS_START_INDEX + STATUS_SIZE
                ],
            )
            mask_loss = self.nn.functional.binary_cross_entropy_with_logits(
                mask_logits,
                self._tensor([row[2] for row in batch]),
            )
            terminal_loss = self.nn.functional.cross_entropy(
                terminal_logits,
                self._tensor(
                    [row[3] for row in batch],
                    dtype=self.torch.int64,
                ),
            )
            loss = (
                descriptor_loss
                + STATUS_LOSS_WEIGHT * status_loss
                + 0.35 * mask_loss
                + 0.25 * terminal_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
            optimizer.step()
            self._losses.append(float(loss.detach().cpu().item()))
            status_losses.append(float(status_loss.detach().cpu().item()))
        self._last_status_training_loss = fmean(status_losses) if status_losses else 0.0
        self.gradient_updates += 1

    def diagnostics(self) -> dict[str, int | float | str]:
        return {
            **super().diagnostics(),
            "status_supervision": 1,
            "status_output_channels": STATUS_SIZE,
            "status_loss_weight": STATUS_LOSS_WEIGHT,
            "last_status_training_loss": self._last_status_training_loss,
        }


class StatusAwareConditionalMixtureRelationalProphecy(
    ConditionalMixtureRelationalProphecy
):
    """Conditional mixture model with an explicit status reconstruction term."""

    name = "current-relational-conditional-mixture-world-model-v4-status-aware"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        descriptor_size = (
            self.component_size - ACTION_SLOT_COUNT - TERMINAL_CLASSES
        )
        if descriptor_size < STATUS_START_INDEX + STATUS_SIZE:
            raise ValueError(
                "status-aware mixture Prophecy requires relational public state v3"
            )
        self._last_status_training_loss = 0.0

    def _train_step(self) -> None:
        torch = self.torch
        nn = self.nn
        k = self.config.mixture_components
        status_losses: list[float] = []
        for model_index, (model, optimizer) in enumerate(
            zip(self.models, self.optimizers, strict=True)
        ):
            local = random.Random(
                (self.observations + 1) * 1_000_003
                + (self.gradient_updates + 1) * 97
                + model_index
            )
            batch = [
                self.replay[local.randrange(len(self.replay))]
                for _ in range(self.config.batch_size)
            ]
            output = model(self._tensor([row[0] for row in batch]))
            descriptor_logits, mask_logits, terminal_logits, mixture_logits = (
                self._split_output(output)
            )

            descriptor_target = self._tensor([row[1] for row in batch]).unsqueeze(1)
            descriptor_target = descriptor_target.expand(-1, k, -1)
            descriptor_loss = nn.functional.smooth_l1_loss(
                torch.sigmoid(descriptor_logits),
                descriptor_target,
                reduction="none",
            ).mean(dim=2)
            status_loss = nn.functional.binary_cross_entropy_with_logits(
                descriptor_logits[
                    :, :, STATUS_START_INDEX : STATUS_START_INDEX + STATUS_SIZE
                ],
                descriptor_target[
                    :, :, STATUS_START_INDEX : STATUS_START_INDEX + STATUS_SIZE
                ],
                reduction="none",
            ).mean(dim=2)

            mask_target = self._tensor([row[2] for row in batch]).unsqueeze(1)
            mask_target = mask_target.expand(-1, k, -1)
            mask_loss = nn.functional.binary_cross_entropy_with_logits(
                mask_logits,
                mask_target,
                reduction="none",
            ).mean(dim=2)

            terminal_target = self._tensor(
                [row[3] for row in batch],
                dtype=torch.int64,
            ).unsqueeze(1).expand(-1, k)
            terminal_loss = nn.functional.cross_entropy(
                terminal_logits.reshape(-1, TERMINAL_CLASSES),
                terminal_target.reshape(-1),
                reduction="none",
            ).reshape(-1, k)

            reconstruction = (
                descriptor_loss
                + STATUS_LOSS_WEIGHT * status_loss
                + 0.35 * mask_loss
                + 0.25 * terminal_loss
            )
            log_mixture = torch.log_softmax(mixture_logits, dim=1)
            nll = -torch.logsumexp(log_mixture - reconstruction, dim=1).mean()
            mixture = torch.softmax(mixture_logits, dim=1)
            entropy = -(mixture * log_mixture).sum(dim=1).mean()
            loss = nll - self.config.mixture_entropy_weight * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
            optimizer.step()
            self._losses.append(float(loss.detach().cpu().item()))
            status_losses.append(float(status_loss.mean().detach().cpu().item()))
        self._last_status_training_loss = fmean(status_losses) if status_losses else 0.0
        self.gradient_updates += 1

    def diagnostics(self) -> dict[str, int | float | str]:
        return {
            **super().diagnostics(),
            "status_supervision": 1,
            "status_output_channels": STATUS_SIZE,
            "status_loss_weight": STATUS_LOSS_WEIGHT,
            "last_status_training_loss": self._last_status_training_loss,
        }
