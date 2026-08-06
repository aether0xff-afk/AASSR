from __future__ import annotations

import csv
import io
import json
import random
import time
from collections import Counter, deque
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

from . import toolgrid_factorial as base
from .autonomous_agent_core import ActionDecision
from .branch_critic import GRUBranchCritic
from .types import Action, Prediction, StateSnapshot


# This pilot isolates spatial horizon and semantic tool branching. A single
# station keeps the final-success signal observable often enough for the learned
# branch critic to see both successful and failed real episodes. Dependency
# depth is deliberately held constant here and can be restored as a third factor
# after these two manipulations are validated.
STAGE_COUNT = 1
TOOLGRID_STATE_SIZE = 2 + 1 + 1 + STAGE_COUNT * 3 + base.MAX_GRID_SIZE**2


def _terminal_class(state: StateSnapshot) -> int:
    if state.available_actions:
        return 0
    return 1 if state.goal_progress >= 1.0 or "success" in state.facts else 2


class ToolGridWorld(base.ToolGridWorld):
    """ToolGrid with context-valid action masking.

    Navigation states expose only unvisited in-bounds moves. At a station the
    movement actions disappear and the tool choices become available. This keeps
    the semantic branching manipulation while avoiding a floor effect caused by
    asking a random explorer to choose among movement and tools at every tick.
    Episodes still have no artificial environment step limit: a self-avoiding
    walk either reaches the station or exhausts all valid moves and fails.
    """

    def _valid_move_actions(self) -> tuple[Action, ...]:
        actions: list[Action] = []
        for index, name in enumerate(base.MOVE_NAMES):
            dx, dy = base.MOVE_DELTAS[name]
            candidate = self.agent[0] + dx, self.agent[1] + dy
            if (
                0 <= candidate[0] < self.grid_size
                and 0 <= candidate[1] < self.grid_size
                and candidate not in self.used_cells
            ):
                actions.append(self.actions[index])
        return tuple(actions)

    def _available_actions(self) -> tuple[Action, ...]:
        if self.success or self.failed:
            return ()
        if self.agent == self.current_station:
            return self.actions[4:]
        return self._valid_move_actions()

    def snapshot(self) -> StateSnapshot:
        return StateSnapshot(
            vector=self._vector(),
            facts=self._facts(),
            available_actions=self._available_actions(),
            goal_progress=1.0 if self.success else 0.0,
            metadata={
                "map_seed": self.seed,
                "grid_size": self.grid_size,
                "action_count": self.action_count,
                "tool_count": self.tool_count,
                "stage_count": STAGE_COUNT,
                "stations": self.stations,
                "required_tools": self.required_tools,
                "optimal_steps": self.optimal_steps,
                "termination": "toolgrid_masked_irreversible",
            },
        )

    def step(self, action: Action) -> base.GridPushStep:
        before = self.snapshot()
        error = False
        reward = 0.0
        self.steps += 1
        allowed = {item.signature for item in before.available_actions}
        if action.signature not in allowed:
            self.failed = True
            error = True
        else:
            index = base.action_index(action, self.actions)
            if index < 4:
                if not self._move(base.MOVE_NAMES[index]):
                    self.failed = True
                    error = True
                elif self.agent != self.current_station and not self._valid_move_actions():
                    self.failed = True
            else:
                tool_index = index - 4
                if self.agent != self.current_station or tool_index != self.current_tool:
                    self.failed = True
                    error = True
                else:
                    self.phase += 1
                    if self.phase >= STAGE_COUNT:
                        self.success = True
                        reward = 1.0
                    else:
                        self.used_cells = {self.agent}
        after = self.snapshot()
        before_actions = {item.signature for item in before.available_actions}
        unlocked = tuple(
            item for item in after.available_actions if item.signature not in before_actions
        )
        return base.GridPushStep(
            snapshot=after,
            added_facts=after.facts - before.facts,
            removed_facts=before.facts - after.facts,
            unlocked_actions=unlocked,
            error=error,
            reward=reward,
        )


def encode_toolgrid_state(state: StateSnapshot) -> tuple[float, ...]:
    """Return the frozen raw observation used by DQN and the branch critic."""

    values = tuple(float(value) for value in state.vector)
    if len(values) != TOOLGRID_STATE_SIZE:
        raise ValueError(
            f"ToolGrid state must contain {TOOLGRID_STATE_SIZE} values, got {len(values)}"
        )
    return values


@dataclass(frozen=True, slots=True)
class ToolGridCodec(base.StateCodec):
    """Categorical world-model codec.

    The frozen environment observation retains its original scalar tool field so
    the DQN and critic protocol do not change. Prophecy alone receives a one-hot
    tool identity. This removes the unintended ordinal geometry between tool IDs
    without exposing transition rules or the correct action.
    """

    action_count: int

    def __post_init__(self) -> None:
        base.build_actions(self.action_count)

    @property
    def tool_count(self) -> int:
        return self.action_count - 4

    @property
    def dimension(self) -> int:
        return TOOLGRID_STATE_SIZE - 1 + self.tool_count

    def encode(self, state: StateSnapshot) -> tuple[float, ...]:
        raw = list(encode_toolgrid_state(state))
        tool = min(
            self.tool_count - 1,
            max(
                0,
                int(round(raw[6] * float(base.MAX_TOOL_COUNT - 1))),
            ),
        )
        category = [0.0] * self.tool_count
        category[tool] = 1.0
        return tuple(raw[:6] + category + raw[7:])

    @staticmethod
    def _bounded(value: float) -> float:
        return min(1.0, max(0.0, float(value)))

    def decode(
        self,
        encoded: Sequence[float],
        *,
        scaffold: StateSnapshot,
        terminal_class: int,
        source: str,
    ) -> StateSnapshot:
        if len(encoded) != self.dimension:
            raise ValueError("categorical ToolGrid neural state has an unexpected size")

        bounded = [self._bounded(value) for value in encoded]
        category = bounded[6 : 6 + self.tool_count]
        tool = max(range(self.tool_count), key=lambda index: category[index])
        raw = (
            bounded[:6]
            + [tool / float(base.MAX_TOOL_COUNT - 1)]
            + bounded[6 + self.tool_count :]
        )

        grid_size = int(scaffold.metadata.get("grid_size", base.MAX_GRID_SIZE))
        scale = float(grid_size - 1)
        raw[0] = round(raw[0] * scale) / scale
        raw[1] = round(raw[1] * scale) / scale
        phase = min(STAGE_COUNT, max(0, int(round(raw[2] * STAGE_COUNT))))
        raw[2] = phase / float(STAGE_COUNT)
        raw[3] = self.tool_count / float(base.MAX_TOOL_COUNT)
        raw[4] = round(raw[4] * scale) / scale
        raw[5] = round(raw[5] * scale) / scale
        raw[6] = tool / float(base.MAX_TOOL_COUNT - 1)
        for index in range(7, len(raw)):
            raw[index] = float(raw[index] >= 0.5)

        agent = int(round(raw[0] * scale)), int(round(raw[1] * scale))
        station = int(round(raw[4] * scale)), int(round(raw[5] * scale))
        used: set[tuple[int, int]] = set()
        facts = {
            f"phase:{phase}",
            f"grid_size:{grid_size}",
            f"action_count:{self.action_count}",
        }
        for index, occupied in enumerate(raw[7:]):
            if occupied < 0.5:
                continue
            x = index % base.MAX_GRID_SIZE
            y = index // base.MAX_GRID_SIZE
            if x < grid_size and y < grid_size:
                used.add((x, y))
                facts.add(f"used:{x}:{y}")
        if phase < STAGE_COUNT:
            facts.add(f"required_tool:{tool}")
        if terminal_class == 1:
            facts.add("success")
        elif terminal_class == 2:
            facts.add("failed")

        actions = base.build_actions(self.action_count)
        available: tuple[Action, ...] = ()
        if terminal_class == 0 and phase < STAGE_COUNT:
            if agent == station:
                available = actions[4:]
            else:
                candidates: list[Action] = []
                for index, name in enumerate(base.MOVE_NAMES):
                    dx, dy = base.MOVE_DELTAS[name]
                    point = agent[0] + dx, agent[1] + dy
                    if (
                        0 <= point[0] < grid_size
                        and 0 <= point[1] < grid_size
                        and point not in used
                    ):
                        candidates.append(actions[index])
                available = tuple(candidates)

        metadata = dict(scaffold.metadata)
        metadata.update(
            {
                "imagined_neural_delta": True,
                "imagined_neural_delta_source": source,
                "action_count": self.action_count,
                "stage_count": STAGE_COUNT,
            }
        )
        return StateSnapshot(
            vector=tuple(raw),
            facts=frozenset(facts),
            available_actions=available,
            goal_progress=1.0 if terminal_class == 1 else 0.0,
            metadata=metadata,
        )


class EnumeratedBalancedNeuralDeltaProphecy(base.NeuralDeltaProphecy):
    """Neural Prophecy with collision-free action identity and balanced replay.

    A fixed action vocabulary is represented one-hot instead of by signed hash
    buckets. Replay batches are sampled uniformly over (action identity,
    qualitative outcome) strata, preventing abundant navigation transitions from
    drowning out rare irreversible choices. Every real transition is stored once;
    only minibatch sampling is balanced.
    """

    def __init__(self, codec: ToolGridCodec, *args: Any, **kwargs: Any) -> None:
        actions = base.build_actions(codec.action_count)
        self._action_index = {
            action.signature: index for index, action in enumerate(actions)
        }
        super().__init__(codec, *args, **kwargs)
        if len(self._action_index) > self.config.action_feature_size:
            raise ValueError("action feature size is smaller than ToolGrid vocabulary")
        self.replay = deque(maxlen=self.config.replay_capacity)

    def _action_features(self, action: Action) -> tuple[float, ...]:
        values = [0.0] * self.config.action_feature_size
        try:
            values[self._action_index[action.signature]] = 1.0
        except KeyError as exc:
            raise ValueError(
                f"unknown ToolGrid action for Prophecy: {action.signature}"
            ) from exc
        return tuple(values)

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        before = self.codec.encode(state)
        after = self.codec.encode(actual_next_state)
        delta = tuple(right - left for left, right in zip(before, after, strict=True))
        self.replay.append(
            (
                self._input(state, action),
                before,
                delta,
                self._terminal_class(actual_next_state),
                action.signature,
            )
        )
        self.observations += 1
        if len(self.replay) < max(self.config.batch_size, self.config.warmup_steps):
            return
        for _ in range(self.config.gradient_steps_per_observation):
            self._train_step()

    def _train_step(self) -> None:
        torch = self.torch
        nn = self.nn
        strata: dict[tuple[str, int], list[Any]] = {}
        for item in self.replay:
            strata.setdefault((item[4], item[3]), []).append(item)
        groups = tuple(strata.values())
        for model_index, (model, optimizer) in enumerate(
            zip(self.models, self.optimizers, strict=True)
        ):
            local = random.Random(
                (self.observations + 1) * 1_000_003
                + (self.gradient_updates + 1) * 97
                + model_index
            )
            batch = [
                local.choice(local.choice(groups))
                for _ in range(self.config.batch_size)
            ]
            inputs = self._tensor([item[0] for item in batch])
            deltas = self._tensor([item[2] for item in batch])
            terminal = self._tensor(
                [item[3] for item in batch],
                dtype=torch.int64,
            )
            output = model(inputs)
            predicted_delta = output[:, : self.codec.dimension]
            terminal_logits = output[:, self.codec.dimension :]
            delta_loss = nn.functional.smooth_l1_loss(predicted_delta, deltas)
            terminal_loss = nn.functional.cross_entropy(terminal_logits, terminal)
            loss = delta_loss + 0.25 * terminal_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
            optimizer.step()
            self._losses.append(float(loss.detach().item()))
        self.gradient_updates += 1


class OutcomeAwareCalibratedProphecy(base.ToolGridCalibratedProphecy):
    """Per-action calibration that tracks model revisions and outcome semantics."""

    def _calibration(self, action: Action) -> float:
        items = [
            item
            for item in getattr(self.holdout, "_items", ())
            if item.action.signature == action.signature
        ]
        if len(items) < self.minimum_count:
            return 0.0

        holdout_bucket = (len(items) - self.minimum_count) // self.refresh_stride
        model_bucket = (
            int(getattr(self.base, "gradient_updates", 0)) // self.refresh_stride
        )
        key = (holdout_bucket, model_bucket, action.signature)
        if key in self._cache:
            return self._cache[key]

        selected = items[-self.evaluation_limit :]
        scores: list[float] = []
        for item in selected:
            prediction = self.base.predict(item.before, item.action, samples=1)[0]
            predicted = prediction.next_state
            vector_error = fmean(
                abs(left - right)
                for left, right in zip(
                    predicted.vector,
                    item.after.vector,
                    strict=True,
                )
            )
            terminal_match = _terminal_class(predicted) == _terminal_class(item.after)
            available_match = {
                candidate.signature for candidate in predicted.available_actions
            } == {
                candidate.signature for candidate in item.after.available_actions
            }
            structural = 1.0 if available_match else 0.75
            scores.append(
                max(0.0, 1.0 - vector_error)
                * (1.0 if terminal_match else 0.0)
                * structural
            )

        value = (fmean(scores) if scores else 0.0) ** self.calibration_power
        value = max(0.0, min(1.0, value))
        self._cache[key] = value
        return value


class MaskAwareToolGridDQNAgent(base.ToolGridDQNAgent):
    """DQN whose Bellman target respects contextual action availability."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.replay = deque(maxlen=self.replay.maxlen)

    def _masked_next_values(
        self,
        next_observations: Any,
        next_action_masks: Any,
        terminals: Any,
    ) -> Any:
        with self.torch.no_grad():
            values = self.target(next_observations)
            floor = self.torch.finfo(values.dtype).min
            masked = values.masked_fill(~next_action_masks, floor)
            maxima = masked.max(dim=1).values
            return self.torch.where(
                terminals > 0.5,
                self.torch.zeros_like(maxima),
                maxima,
            )

    def observe(
        self,
        before: StateSnapshot,
        action: Action,
        outcome: base.GridPushStep,
    ) -> None:
        terminal = not outcome.snapshot.available_actions
        available = {
            item.signature for item in outcome.snapshot.available_actions
        }
        mask = tuple(
            candidate.signature in available for candidate in self.actions
        )
        self.replay.append(
            (
                encode_toolgrid_state(before),
                self.action_by_signature[action.signature],
                float(outcome.reward),
                encode_toolgrid_state(outcome.snapshot),
                terminal,
                mask,
            )
        )
        self.environment_steps += 1
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return

        batch = self.randomizer.sample(list(self.replay), self.batch_size)
        observations = self._tensor([item[0] for item in batch])
        actions = self.torch.as_tensor(
            [item[1] for item in batch],
            dtype=self.torch.int64,
        )
        rewards = self._tensor([item[2] for item in batch])
        next_observations = self._tensor([item[3] for item in batch])
        terminals = self._tensor([float(item[4]) for item in batch])
        next_masks = self.torch.as_tensor(
            [item[5] for item in batch],
            dtype=self.torch.bool,
        )

        predicted = self.online(observations).gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)
        next_values = self._masked_next_values(
            next_observations,
            next_masks,
            terminals,
        )
        targets = rewards + self.gamma * (1.0 - terminals) * next_values
        loss = self.loss(predicted, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.gradient_updates += 1
        if self.gradient_updates % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

    def abort_episode(self, *, training: bool) -> None:
        del training


class ProductionToolGridHybridAgent(base.ToolGridHybridAgent):
    """Matched hybrid with evaluation-only, consequence-gated Imagination."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._production_gate_counts: Counter[str] = Counter()
        if self.critic is None:
            seed = int(args[0] if args else kwargs.get("seed", 0))
            self.critic = GRUBranchCritic(
                encode_toolgrid_state,
                TOOLGRID_STATE_SIZE,
                hidden_units=64,
                batch_size=16,
                replay_capacity=4_000,
                gradient_steps_per_episode=2,
                seed=seed ^ 0x43524954,
            )
            self.agent.planner.scorer = self.critic

    def _predicted_terminal_choice(self, state: StateSnapshot) -> bool:
        if len(state.available_actions) < 2:
            return False
        return all(
            not self.agent.prophecy.predict(
                state,
                action,
                samples=1,
            )[0].next_state.available_actions
            for action in state.available_actions
        )

    def select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        training: bool,
    ) -> base.AgentDecision:
        original = self.agent.config
        terminal_choice = False
        if not self.use_imagination:
            reason = "policy_only"
        elif training:
            reason = "training_suppressed"
        elif not self.critic_ready:
            reason = "critic_not_ready"
        else:
            terminal_choice = self._predicted_terminal_choice(state)
            reason = "terminal_choice" if terminal_choice else "nonterminal_choice"

        allow_imagination = (
            self.use_imagination
            and not training
            and self.critic_ready
            and terminal_choice
        )
        self._production_gate_counts[reason] += 1
        self.agent.config = replace(
            original,
            use_imagination=allow_imagination,
        )
        try:
            decision: ActionDecision = self.agent.select_action(
                state,
                episode=episode,
                explore=training,
            )
        finally:
            self.agent.config = original
        return base.AgentDecision(
            action=decision.action,
            imagined_nodes=decision.imagined_nodes,
            used_imagination=decision.used_imagination,
        )

    def abort_episode(self, *, training: bool) -> None:
        self.dqn.end_episode(success=False, training=training)
        self.agent.discard_episode()
        self._critic_trajectory.clear()

    def model_stats(self) -> dict[str, int | float]:
        stats = super().model_stats()
        handle = io.BytesIO()
        self.base_prophecy.torch.save(
            [model.state_dict() for model in self.base_prophecy.models],
            handle,
        )
        stats["model_bytes"] = int(stats.get("model_bytes", 0)) + len(
            handle.getvalue()
        )
        stats.update(
            {
                f"imagination_gate_{key}": value
                for key, value in self._production_gate_counts.items()
            }
        )
        stats["training_imagination_interventions"] = 0
        return stats


_TRAINING_SEGMENTS: list[tuple[int, float, str]] = []


def _run_episode(
    agent: Any,
    *,
    condition: str,
    research_seed: int,
    phase: str,
    grid_size: int,
    action_count: int,
    checkpoint_transition_target: int,
    episode_index: int,
    map_seed: int,
    training: bool,
    environment_steps_total: int,
    schedule_horizon: int,
) -> tuple[base.EpisodeRow, int]:
    """Run an episode or an exact-budget training segment.

    Reaching a transition checkpoint pauses learning without reporting a false
    environment failure. Partial episodic critic/return buffers are discarded;
    every transition-level learner still receives exactly the configured number
    of real transitions.
    """

    world = ToolGridWorld(map_seed, grid_size=grid_size, action_count=action_count)
    agent.begin_episode(training=training)
    steps = 0
    select_seconds = 0.0
    update_seconds = 0.0
    imagined_nodes = 0
    imagination_runs = 0
    segment_started = time.perf_counter()
    budget_paused = False

    while world.snapshot().available_actions:
        if training and environment_steps_total >= checkpoint_transition_target:
            budget_paused = True
            break
        before = world.snapshot()
        schedule_position = min(schedule_horizon, environment_steps_total)
        started = time.perf_counter()
        decision = agent.select_action(
            before,
            episode=schedule_position,
            training=training,
        )
        select_seconds += time.perf_counter() - started
        outcome = world.step(decision.action)
        steps += 1
        imagined_nodes += int(decision.imagined_nodes)
        imagination_runs += int(decision.used_imagination)
        if training:
            environment_steps_total += 1
            started = time.perf_counter()
            agent.observe(before, decision.action, outcome)
            update_seconds += time.perf_counter() - started
        if world.success or world.failed:
            break
        if training and environment_steps_total >= checkpoint_transition_target:
            budget_paused = True
            break

    if budget_paused and not (world.success or world.failed):
        abort = getattr(agent, "abort_episode", None)
        if callable(abort):
            abort(training=training)
        else:
            agent.end_episode(success=False, training=training)
        termination = "budget_checkpoint"
    else:
        agent.end_episode(success=world.success, training=training)
        termination = "success" if world.success else "environment_failure"

    if training:
        _TRAINING_SEGMENTS.append(
            (
                environment_steps_total,
                time.perf_counter() - segment_started,
                termination,
            )
        )

    return (
        base.EpisodeRow(
            condition=condition,
            seed=research_seed,
            phase=phase,
            grid_size=grid_size,
            action_count=action_count,
            checkpoint_transition_target=checkpoint_transition_target,
            episode=episode_index,
            map_seed=map_seed,
            success=int(world.success),
            reward=1.0 if world.success else 0.0,
            steps=steps,
            optimal_steps=world.optimal_steps,
            path_efficiency=(
                world.optimal_steps / steps if world.success and steps else 0.0
            ),
            environment_steps_total=environment_steps_total,
            select_seconds=select_seconds,
            update_seconds=update_seconds,
            imagined_nodes=imagined_nodes,
            imagination_runs=imagination_runs,
            termination=termination,
        ),
        environment_steps_total,
    )


def _rewrite_experiment_metrics(output: Path, payload: dict[str, Any]) -> None:
    checkpoint_path = output / "checkpoints.csv"
    with checkpoint_path.open(newline="", encoding="utf-8") as handle:
        checkpoint_rows = list(csv.DictReader(handle))

    for row in checkpoint_rows:
        target = int(row["checkpoint_transition_target"])
        relevant = [item for item in _TRAINING_SEGMENTS if item[0] <= target]
        row["training_wall_seconds"] = repr(sum(item[1] for item in relevant))
        row["training_episode_count"] = str(
            sum(item[2] != "budget_checkpoint" for item in relevant)
        )

    with checkpoint_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checkpoint_rows[0]))
        writer.writeheader()
        writer.writerows(checkpoint_rows)

    final = checkpoint_rows[-1]
    for key, value in final.items():
        if key in {
            "condition",
        }:
            payload["final"][key] = value
        elif key in {
            "seed",
            "grid_size",
            "action_count",
            "checkpoint_transition_target",
            "actual_training_transitions",
            "training_episode_count",
            "model_units",
            "model_bytes",
            "gradient_updates",
        }:
            payload["final"][key] = int(value)
        else:
            payload["final"][key] = float(value)

    manifest_path = output / "map_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    for row in manifest:
        row["effective_branching_factor"] = row["tool_count"]
        row["global_action_vocabulary"] = row["action_count"]
        row["semantic_branching_factor"] = row["tool_count"]
    fields = list(manifest[0])
    for field in ("global_action_vocabulary", "semantic_branching_factor"):
        if field not in fields:
            fields.append(field)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)

    payload["config"].update(
        {
            "training_imagination_interventions": False,
            "paired_training_trajectory": True,
            "prophecy_action_encoding": "one_hot_fixed_vocabulary",
            "prophecy_tool_encoding": "categorical_one_hot",
            "prophecy_replay_sampling": "balanced_action_outcome",
            "imagination_gate": "all_available_actions_predicted_terminal",
            "dqn_target_action_masking": True,
            "exact_transition_budget": True,
        }
    )
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=repr),
        encoding="utf-8",
    )


def run_toolgrid_factorial(
    output_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    _TRAINING_SEGMENTS.clear()
    payload = base.run_toolgrid_factorial(output_dir, **kwargs)
    _rewrite_experiment_metrics(Path(output_dir), payload)
    return payload


# Install the corrected production components into the base experiment module.
# The base runner resolves these globals at execution time.
base.STAGE_COUNT = STAGE_COUNT
base.TOOLGRID_STATE_SIZE = TOOLGRID_STATE_SIZE
base.ToolGridWorld = ToolGridWorld
base.ToolGridCodec = ToolGridCodec
base.encode_toolgrid_state = encode_toolgrid_state
base.NeuralDeltaProphecy = EnumeratedBalancedNeuralDeltaProphecy
base.ToolGridCalibratedProphecy = OutcomeAwareCalibratedProphecy
base.ToolGridDQNAgent = MaskAwareToolGridDQNAgent
base.ToolGridHybridAgent = ProductionToolGridHybridAgent
base._run_episode = _run_episode

GRID_SIZES = base.GRID_SIZES
ACTION_COUNTS = base.ACTION_COUNTS
TOOLGRID_CONDITIONS = base.TOOLGRID_CONDITIONS
build_actions = base.build_actions
