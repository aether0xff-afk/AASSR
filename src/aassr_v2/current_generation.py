from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from statistics import fmean
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Hashable, Iterable, Mapping, Sequence
import random

from .action_plugins import PluginOutcome
from .autonomous_agent import AutonomousAgentConfig
from .autonomous_agent_core import RunningValue
from .branch_critic import CriticTransition, GRUBranchCritic
from .integrated_agent import (
    ContextualSkillAwareProphecy,
    IntegratedAASSRConfig,
    IntegratedAASSRAgent,
    IntegratedActionDecision,
    IntegratedAgentStep,
    IntegratedProphecyView,
)
from .knowledge import KnowledgeStore
from .neural_delta_prophecy import NeuralDeltaConfig, NeuralDeltaProphecy
from .pentest_agent_main_test import (
    ACTION_FEATURE_SIZE,
    AGENT_STATE_SIZE,
    HttpAgentCodec,
)
from .pentest_curriculum_causal import OBSERVATION_CONTRACT
from .pentest_curriculum_dedup import DeduplicatedRelationalDQN
from .pentest_curriculum_env import (
    OBJECT_RELATIONS,
    PROFILE_RELATIONS,
    ROUTE_RELATIONS,
    relational_action_features,
)
from .pentest_curriculum_schedule import semantic_fingerprint
from .policy import PolicyMemory, ScoredAction
from .prophecy import ProphecyStep
from .replay import ReplayBuffer
from .skills import SKILL_VERB, Skill, SkillLibrary
from .types import Action, Prediction, StateSnapshot, TransitionTrace

if TYPE_CHECKING:
    from .current_plugin_api import CurrentRepresentationBinding


CURRENT_GENERATION_VERSION = "aassr-current-generation-v1"

# This is deliberately executable metadata. Tests assert that the canonical
# current-generation builder actually instantiates these components, so an old
# implementation cannot silently become active merely because its file still
# exists in the repository.
CURRENT_COMPONENTS: Mapping[str, str] = {
    "observation": "response_causal_observation_v3",
    "aseq": "semantic-self-loop-empirical-v3",
    "policy": "relational-invariant-dqn+information-residual-v1",
    "prophecy": "neural-delta-ensemble+relational-action-v1",
    "calibration": "frozen-replay-relational-holdout-v1",
    "knowledge": "episode-local-response-knowledge-context-v1",
    "imagination": "parallel-universe-tree-v2",
    "critic": "relational-gru-branch-critic-final-outcome-v1",
    "skills": "relational-aseq-template-v1",
    "goals": "external-final-goal+relational-skill-promotion-v1",
    "effect_composition": "superseded-by-neural-delta-disabled",
    "training_imagination": "disabled-same-checkpoint",
}

LEGACY_COMPONENTS_ACTIVE: tuple[str, ...] = ()


def _fact_values(facts: frozenset[str], prefix: str) -> tuple[str, ...]:
    return tuple(
        fact.removeprefix(prefix)
        for fact in facts
        if fact.startswith(prefix)
    )


def _role_counts(
    facts: frozenset[str],
    prefix: str,
    roles: Sequence[str],
) -> tuple[float, ...]:
    counts = Counter()
    for fact in facts:
        if not fact.startswith(prefix):
            continue
        try:
            _, role = fact.removeprefix(prefix).rsplit(":", 1)
        except ValueError:
            continue
        counts[role] += 1
    normalizer = float(max(1, sum(counts.values())))
    return tuple(counts[role] / normalizer for role in roles)


def relational_state_descriptor(state: StateSnapshot) -> tuple[float, ...]:
    """Permutation-invariant public-state description for transfer learners.

    ASEQ intentionally does *not* use this representation. ASEQ must distinguish
    concrete episode-local entities, whereas Policy/Critic/Skill transfer should
    treat seed-renamed identifiers as the same structural situation.
    """

    facts = state.facts
    controls = tuple(
        float(state.vector[index]) if index < len(state.vector) else 0.0
        for index in range(7)
    )
    request_count = float(state.vector[8]) if len(state.vector) > 8 else 0.0
    workflow_progress = float(state.vector[10]) if len(state.vector) > 10 else 0.0

    known_routes = len(_fact_values(facts, "known_route:"))
    known_profiles = len(_fact_values(facts, "known_profile:"))
    known_objects = len(_fact_values(facts, "known_object:"))
    tried_objects = len(_fact_values(facts, "tried_object:"))

    route_roles = _role_counts(
        facts,
        "observed_route_role:",
        ROUTE_RELATIONS,
    )
    profile_roles = _role_counts(
        facts,
        "observed_profile_role:",
        PROFILE_RELATIONS,
    )

    actions = tuple(state.available_actions)
    relational_actions = {
        relational_action_features(state, action)
        for action in actions
        if action.verb_name != SKILL_VERB
    }
    request_fraction = (
        sum(action.verb_name == "request" for action in actions) / len(actions)
        if actions
        else 0.0
    )
    object_request_fraction = (
        sum(action.verb_name == "request_object" for action in actions) / len(actions)
        if actions
        else 0.0
    )

    return (
        *controls,
        request_count,
        workflow_progress,
        min(1.0, known_routes / 32.0),
        min(1.0, known_profiles / 32.0),
        min(1.0, known_objects / 32.0),
        *route_roles,
        *profile_roles,
        float(bool(_fact_values(facts, "observed_own_object:"))),
        float(bool(_fact_values(facts, "observed_target_object:"))),
        min(1.0, tried_objects / 32.0),
        min(1.0, len(actions) / 128.0),
        min(1.0, len(relational_actions) / 32.0),
        request_fraction,
        object_request_fraction,
    )


def relational_state_vector(state: StateSnapshot) -> tuple[float, ...]:
    descriptor = relational_state_descriptor(state)
    if len(descriptor) > AGENT_STATE_SIZE:
        raise AssertionError("relational state descriptor exceeds DQN state size")
    return descriptor + (0.0,) * (AGENT_STATE_SIZE - len(descriptor))


def relational_state_key(state: StateSnapshot) -> tuple[float, ...]:
    return tuple(round(value, 8) for value in relational_state_descriptor(state))


def relational_action_key(state: StateSnapshot, action: Action) -> tuple[float, ...]:
    if action.verb_name == SKILL_VERB:
        # Skill identity is learned separately and is never mixed with primitive
        # HTTP action identity.
        return ()
    return tuple(float(value) for value in relational_action_features(state, action))


class RelationalInvariantDQN(DeduplicatedRelationalDQN):
    """Audited DQN with seed-renaming-invariant state input."""

    @staticmethod
    def encode_state(state: StateSnapshot) -> tuple[float, ...]:
        return relational_state_vector(state)

    def action_features(
        self,
        state: StateSnapshot,
        action: Action,
    ) -> tuple[float, ...]:
        return relational_action_key(state, action)

    def observe(
        self,
        before: StateSnapshot,
        action: Action,
        outcome: PluginOutcome,
        *,
        reward: float,
    ) -> None:
        terminal = self._consume_episode_boundary() or not outcome.snapshot.available_actions
        raw_next = tuple(outcome.snapshot.available_actions)
        next_features, _ = self._deduplicate(outcome.snapshot, raw_next)
        self.raw_next_actions_stored += len(raw_next)
        self.unique_next_features_stored += len(next_features)
        self.replay.append(
            (
                self.encode_state(before),
                self.action_features(before, action),
                float(reward),
                self.encode_state(outcome.snapshot),
                next_features,
                terminal,
            )
        )
        self.environment_steps += 1
        if len(self.replay) >= max(self.batch_size, self.warmup_steps):
            self._train_step()


def bind_relational_dqn_representation(
    dqn: RelationalInvariantDQN,
    representation: CurrentRepresentationBinding,
) -> RelationalInvariantDQN:
    """Bind state/action encoders to one DQN instance, never to a module."""

    dqn.encode_state = representation.state_vector
    dqn.action_features = representation.action_structure

    def deduplicate(
        state: StateSnapshot,
        actions: Sequence[Action],
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[int, ...]]:
        unique: dict[tuple[float, ...], int] = {}
        indices: list[int] = []
        for action in actions:
            features = representation.action_structure(state, action)
            indices.append(unique.setdefault(features, len(unique)))
        return tuple(unique), tuple(indices)

    dqn._deduplicate = deduplicate
    return dqn


class CurrentRelationalPolicy:
    """Sparse-reward DQN Policy plus a separate learned information residual.

    The DQN head receives only the environment's sparse external reward. The
    auxiliary residual receives only delayed *internal information value* from
    the integrated evaluator. Keeping the two targets separate avoids turning an
    internal learning signal into an environment reward or double-counting the
    terminal return.
    """

    name = "current-relational-policy"

    def __init__(
        self,
        dqn: RelationalInvariantDQN,
        *,
        information_learning_rate: float = 0.2,
        representation: CurrentRepresentationBinding | None = None,
    ) -> None:
        self.dqn = dqn
        self.representation = representation
        self.information_learning_rate = float(information_learning_rate)
        self._information: dict[
            tuple[Hashable, tuple[float, ...]], RunningValue
        ] = {}
        self._skill_values: dict[str, RunningValue] = {}

    @staticmethod
    def empty_memory() -> PolicyMemory:
        return PolicyMemory.empty()

    def _information_entry(
        self,
        state: StateSnapshot,
        action: Action,
    ) -> RunningValue:
        if action.verb_name == SKILL_VERB:
            return RunningValue()
        return self._information_entry_for_state_key(
            self._state_key(state),
            state,
            action,
        )

    def _information_entry_for_state_key(
        self,
        state_key: Hashable,
        state: StateSnapshot,
        action: Action,
    ) -> RunningValue:
        """Look up one primitive residual using a precomputed state key."""

        return self._information.get(
            (state_key, self._action_key(state, action)),
            RunningValue(),
        )

    def value(self, state: StateSnapshot, action: Action) -> float:
        if action.verb_name == SKILL_VERB:
            return self._skill_values.get(str(action.target), RunningValue()).mean
        external = self.dqn.score_actions(state, (action,))[0]
        return external + self._information_entry(state, action).mean

    def rank(
        self,
        state: StateSnapshot,
        *,
        limit: int,
        memory: PolicyMemory | None = None,
    ) -> tuple[ScoredAction, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        actions = tuple(state.available_actions)
        if not actions:
            return ()
        deltas = {} if memory is None else memory.deltas
        primitive = tuple(action for action in actions if action.verb_name != SKILL_VERB)
        primitive_values = (
            self.dqn.score_actions(state, primitive) if primitive else ()
        )
        information_state_key = (
            self._state_key(state) if primitive else ()
        )
        by_signature = {
            action.signature: value
            for action, value in zip(primitive, primitive_values, strict=True)
        }
        rows = []
        for action in actions:
            if action.verb_name == SKILL_VERB:
                base = self._skill_values.get(str(action.target), RunningValue()).mean
            else:
                base = by_signature[action.signature] + (
                    self._information_entry_for_state_key(
                        information_state_key,
                        state,
                        action,
                    ).mean
                )
            rows.append(
                ScoredAction(
                    action,
                    float(base) + float(deltas.get(action.signature, 0.0)),
                )
            )
        rows.sort(key=lambda item: (-item.score, item.action.signature))
        return tuple(rows[:limit])

    def select(
        self,
        state: StateSnapshot,
        *,
        randomizer: random.Random,
        epsilon: float,
        exploration_bonus: float,
    ) -> Action:
        del exploration_bonus
        if not state.available_actions:
            raise ValueError("cannot select from an empty action set")
        if epsilon > 0.0 and randomizer.random() < epsilon:
            return randomizer.choice(state.available_actions)
        return self.rank(state, limit=1)[0].action

    def imagine_update(
        self,
        memory: PolicyMemory,
        action: Action,
        value: float,
    ) -> PolicyMemory:
        deltas = dict(memory.deltas)
        deltas[action.signature] = deltas.get(action.signature, 0.0) + 0.1 * float(value)
        return PolicyMemory(deltas)

    def reinforce(self, action: Action, advantage: float) -> None:
        # Current Imagination uses update_policy=False. Keep this method for the
        # WeightedPolicy protocol without allowing imagined experience to train
        # the real Policy.
        del action, advantage

    def observe_information_return(
        self,
        state: StateSnapshot,
        action: Action,
        value: float,
    ) -> None:
        if action.verb_name == SKILL_VERB:
            return
        key = (self._state_key(state), self._action_key(state, action))
        entry = self._information.setdefault(key, RunningValue())
        entry.observe(float(value), learning_rate=self.information_learning_rate)

    def observe_return(
        self,
        state: StateSnapshot,
        action: Action,
        target: float,
    ) -> None:
        del state
        if action.verb_name != SKILL_VERB:
            # Primitive external return is owned by DQN TD learning.
            return
        entry = self._skill_values.setdefault(str(action.target), RunningValue())
        entry.observe(float(target), learning_rate=self.information_learning_rate)

    def _state_key(self, state: StateSnapshot) -> Hashable:
        if self.representation is not None:
            return self.representation.state_key(state)
        return relational_state_key(state)

    def _action_key(
        self,
        state: StateSnapshot,
        action: Action,
    ) -> tuple[float, ...]:
        if self.representation is not None:
            return self.representation.action_structure(state, action)
        return relational_action_key(state, action)

    def diagnostics(self) -> dict[str, int | float]:
        return {
            "information_entries": len(self._information),
            "skill_value_entries": len(self._skill_values),
            **{
                f"dqn:{key}": value
                for key, value in self.dqn.model_stats().items()
            },
        }


class CurrentNeuralDeltaProphecy(NeuralDeltaProphecy):
    """Latest Neural Delta model with relational action input and CUDA batching."""

    name = "current-relational-neural-delta"

    def __init__(
        self,
        codec: HttpAgentCodec,
        *,
        config: NeuralDeltaConfig,
        seed: int,
        device: str = "cpu",
    ) -> None:
        super().__init__(codec, config=config, seed=seed)
        self.device = self.torch.device(device)
        for model in self.models:
            model.to(self.device)
        self.batch_prediction_calls = 0
        self.batch_prediction_rows = 0
        self.training_loss_bulk_host_transfer_calls = 0
        self.training_loss_bulk_host_transfer_rows = 0

    def _train_step(self) -> None:
        """Train the ensemble with one bulk loss transfer to the host.

        Keep the reference implementation's model-by-model bootstrap and
        optimizer ordering.  Only the loss bookkeeping is deferred: retaining
        detached scalar tensors lets CUDA copy all ensemble losses to the host
        together instead of synchronizing once per model.
        """

        torch = self.torch
        nn = self.nn
        detached_losses = []
        for model_index, (model, optimizer) in enumerate(
            zip(self.models, self.optimizers, strict=True)
        ):
            # Match NeuralDeltaProphecy._train_step's independent bootstrap
            # batches exactly, including its seed and replay access order.
            local = random.Random(
                (self.observations + 1) * 1_000_003
                + (self.gradient_updates + 1) * 97
                + model_index
            )
            batch = [
                self.replay[local.randrange(len(self.replay))]
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
            delta_loss = nn.functional.smooth_l1_loss(
                predicted_delta,
                deltas,
            )
            terminal_loss = nn.functional.cross_entropy(
                terminal_logits,
                terminal,
            )
            loss = delta_loss + 0.25 * terminal_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
            optimizer.step()
            detached_losses.append(loss.detach())

        host_losses = torch.stack(detached_losses).cpu().tolist()
        self.training_loss_bulk_host_transfer_calls += 1
        self.training_loss_bulk_host_transfer_rows += len(detached_losses)
        self._losses.extend(float(value) for value in host_losses)
        self.gradient_updates += 1

    def _tensor(self, values: Any, *, dtype: Any | None = None) -> Any:
        return self.torch.as_tensor(
            values,
            dtype=dtype or self.torch.float32,
            device=self.device,
        )

    def _action_features(self, action: Action) -> tuple[float, ...]:
        # The state-dependent form is supplied by _input below.
        raise RuntimeError("CurrentNeuralDeltaProphecy uses state-relative action features")

    def _input(self, state: StateSnapshot, action: Action) -> tuple[float, ...]:
        return self.codec.encode(state) + relational_action_key(state, action)

    @property
    def training_stats(self) -> Any:
        # The generic replay validator fingerprints ``training_stats.updates``.
        # Expose Neural Delta's real optimizer revision through that contract so
        # the validator cache can never survive a model update.
        return SimpleNamespace(updates=int(self.gradient_updates))

    def _batch_outputs(
        self,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
    ) -> tuple[Any, Any, Any]:
        inputs = self._tensor(
            [self._input(state, action) for state, action in zip(states, actions, strict=True)]
        )
        encoded = self._tensor([self.codec.encode(state) for state in states])
        with self.torch.no_grad():
            outputs = self.torch.stack(
                [model(inputs) for model in self.models],
                dim=0,
            )
            deltas = outputs[:, :, : self.codec.dimension]
            terminal = self.torch.softmax(
                outputs[:, :, self.codec.dimension :],
                dim=-1,
            )
            next_states = encoded.unsqueeze(0) + deltas
        return next_states, terminal, encoded

    def _batch_confidence(self, next_states: Any, terminal: Any) -> Any:
        means = next_states.mean(dim=0)
        variance = ((next_states - means.unsqueeze(0)) ** 2).mean(dim=(0, 2))
        terminal_means = terminal.mean(dim=0)
        terminal_variance = (
            (terminal - terminal_means.unsqueeze(0)) ** 2
        ).mean(dim=(0, 2))
        total_variance = variance + terminal_variance
        self._last_ensemble_variance = float(total_variance.mean().detach().cpu().item())
        sample_confidence = self.observations / (
            self.observations + self.config.confidence_prior
        )
        return self.torch.clamp(
            sample_confidence
            * self.torch.exp(-self.config.variance_scale * total_variance),
            min=0.05,
            max=0.995,
        )

    def predict_batch(
        self,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
        *,
        samples: int,
    ) -> tuple[tuple[Prediction, ...], ...]:
        if len(states) != len(actions):
            raise ValueError("states/actions batch length mismatch")
        if samples <= 0:
            raise ValueError("samples must be positive")
        if not states:
            return ()
        self.batch_prediction_calls += 1
        self.batch_prediction_rows += len(states)
        if self.observations < self.config.warmup_steps:
            return tuple(
                (Prediction(state, 0.0, source=f"{self.name}:unseen"),)
                for state in states
            )

        next_states, terminal, _ = self._batch_outputs(states, actions)
        confidence = self._batch_confidence(next_states, terminal)
        mean_states = next_states.mean(dim=0)
        terminal_classes = terminal.mean(dim=0).argmax(dim=1)
        rows = []
        for index, state in enumerate(states):
            decoded = self.codec.decode(
                mean_states[index].detach().cpu().tolist(),
                scaffold=state,
                terminal_class=int(terminal_classes[index].detach().cpu().item()),
                source=f"{self.name}:ensemble",
            )
            rows.append(
                (
                    Prediction(
                        decoded,
                        float(confidence[index].detach().cpu().item()),
                        source=f"{self.name}:ensemble",
                    ),
                )
            )
        return tuple(rows)

    def confidence_batch(
        self,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
    ) -> tuple[float, ...]:
        if len(states) != len(actions):
            raise ValueError("states/actions batch length mismatch")
        if not states:
            return ()
        if self.observations < self.config.warmup_steps:
            return (0.0,) * len(states)
        next_states, terminal, _ = self._batch_outputs(states, actions)
        confidence = self._batch_confidence(next_states, terminal)
        return tuple(float(value) for value in confidence.detach().cpu().tolist())

    def diagnostics(self) -> dict[str, int | float | str]:
        stats = self.stats()
        return {
            "name": self.name,
            "device": str(self.device),
            "observations": stats.observations,
            "gradient_updates": stats.gradient_updates,
            "replay_size": stats.replay_size,
            "mean_training_loss": stats.mean_training_loss,
            "last_ensemble_variance": stats.last_ensemble_variance,
            "parameter_count": stats.parameter_count,
            "batch_prediction_calls": self.batch_prediction_calls,
            "batch_prediction_rows": self.batch_prediction_rows,
            "training_loss_bulk_host_transfer_calls": (
                self.training_loss_bulk_host_transfer_calls
            ),
            "training_loss_bulk_host_transfer_rows": (
                self.training_loss_bulk_host_transfer_rows
            ),
            "per_model_training_loss_item_syncs": 0,
        }


class ReplayRelationalCalibratedProphecy:
    """Frozen-holdout calibration keyed by structural action, never raw ID."""

    name = "current-relational-holdout-calibrated-prophecy"

    def __init__(
        self,
        base: CurrentNeuralDeltaProphecy,
        replay: ReplayBuffer,
        *,
        minimum_count: int = 8,
        evaluation_limit: int = 48,
        refresh_stride: int = 32,
    ) -> None:
        self.base = base
        self.replay = replay
        self.minimum_count = int(minimum_count)
        self.evaluation_limit = int(evaluation_limit)
        self.refresh_stride = int(refresh_stride)
        self._cache: dict[tuple[Hashable, int, int], float] = {}
        self.refreshes = 0

    @property
    def training_stats(self) -> Any:
        return self.base.training_stats

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def _key(self, state: StateSnapshot, action: Action) -> tuple[float, ...]:
        representation = getattr(self.base, "representation", None)
        if representation is not None:
            return representation.action_structure(state, action)
        return relational_action_key(state, action)

    @staticmethod
    def _terminal_class(state: StateSnapshot) -> int:
        if state.available_actions:
            return 0
        return 1 if state.goal_progress >= 1.0 or "success" in state.facts else 2

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        self.base.learn(state, action, actual_next_state)

    def _calibration(self, state: StateSnapshot, action: Action) -> float:
        key = self._key(state, action)
        items = [
            item
            for item in self.replay.holdout()
            if self._key(item.state, item.action) == key
        ]
        revision = int(self.base.gradient_updates)
        cache_key = (key, len(items) // self.refresh_stride, revision // self.refresh_stride)
        if cache_key in self._cache:
            return self._cache[cache_key]
        self.refreshes += 1
        if len(items) < self.minimum_count:
            value = 0.0
        else:
            scores = []
            for item in items[-self.evaluation_limit :]:
                prediction = self.base.predict(item.state, item.action, samples=1)[0]
                predicted = prediction.next_state
                error = fmean(
                    abs(left - right)
                    for left, right in zip(
                        predicted.vector,
                        item.next_state.vector,
                        strict=True,
                    )
                )
                terminal_match = (
                    self._terminal_class(predicted)
                    == self._terminal_class(item.next_state)
                )
                action_ratio = min(
                    len(predicted.available_actions),
                    len(item.next_state.available_actions),
                ) / float(
                    max(
                        1,
                        len(predicted.available_actions),
                        len(item.next_state.available_actions),
                    )
                )
                scores.append(
                    max(0.0, 1.0 - error)
                    * float(terminal_match)
                    * action_ratio
                )
            value = max(0.0, min(1.0, fmean(scores) if scores else 0.0))
        self._cache[cache_key] = value
        return value

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        calibration = self._calibration(state, action)
        return tuple(
            Prediction(
                prediction.next_state,
                prediction.probability * calibration,
                source=f"{prediction.source}:relational-calibrated",
            )
            for prediction in self.base.predict(state, action, samples=samples)
        )

    def predict_batch(
        self,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
        *,
        samples: int,
    ) -> tuple[tuple[Prediction, ...], ...]:
        rows = self.base.predict_batch(states, actions, samples=samples)
        return tuple(
            tuple(
                Prediction(
                    prediction.next_state,
                    prediction.probability * self._calibration(state, action),
                    source=f"{prediction.source}:relational-calibrated",
                )
                for prediction in row
            )
            for state, action, row in zip(states, actions, rows, strict=True)
        )

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        return min(
            float(self.base.confidence(state, action)),
            self._calibration(state, action),
        )

    def coverage(
        self,
        state: StateSnapshot,
        actions: Iterable[Action],
    ) -> float:
        materialized = tuple(actions)
        if not materialized:
            return 1.0
        if len(materialized) > 32:
            stride = max(1, len(materialized) // 32)
            materialized = materialized[::stride][:32]
        return fmean(self.confidence(state, action) for action in materialized)

    def diagnostics(self) -> dict[str, int | float]:
        return {
            "calibration_refreshes": self.refreshes,
            "calibration_cache_entries": len(self._cache),
        }


class KnowledgeBoundProphecy:
    """Add only pre-existing response Knowledge to a neural prediction.

    Context-free ``predict`` remains untouched for frozen holdout evaluation.
    ``predict_with_context`` may only use the KnowledgeStore explicitly supplied
    by the caller, preserving the same-transition anti-hindsight boundary.
    """

    name = "current-knowledge-bound-prophecy"

    def __init__(self, base: ReplayRelationalCalibratedProphecy) -> None:
        self.base = base

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    @property
    def training_stats(self) -> Any:
        return self.base.training_stats

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        self.base.learn(state, action, actual_next_state)

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        return self.base.predict(state, action, samples=samples)

    def predict_batch(
        self,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
        *,
        samples: int,
    ) -> tuple[tuple[Prediction, ...], ...]:
        return self.base.predict_batch(states, actions, samples=samples)

    def predict_with_context(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        knowledge: KnowledgeStore,
        samples: int,
    ) -> tuple[Prediction, ...]:
        entries = tuple(knowledge.values())
        known_facts = frozenset(
            entry.key
            for entry in entries
            if bool(entry.value) and entry.confidence > 0.0
        )
        enabled = {
            signature
            for entry in entries
            for signature in entry.enabled_action_signatures
        }
        current_actions = {
            item.signature: item for item in state.available_actions
        }
        enabled_actions = tuple(
            current_actions[signature]
            for signature in sorted(enabled)
            if signature in current_actions
        )
        output = []
        for prediction in self.base.predict(state, action, samples=samples):
            next_state = prediction.next_state
            action_map = {
                item.signature: item for item in next_state.available_actions
            }
            for item in enabled_actions:
                action_map[item.signature] = item
            output.append(
                replace(
                    prediction,
                    next_state=replace(
                        next_state,
                        facts=next_state.facts | known_facts,
                        available_actions=tuple(
                            action_map[key] for key in sorted(action_map)
                        ),
                    ),
                )
            )
        return tuple(output)

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        return self.base.confidence(state, action)

    def coverage(self, state: StateSnapshot, actions: Iterable[Action]) -> float:
        return self.base.coverage(state, actions)

    def diagnostics(self) -> dict[str, int | float | str]:
        return {
            **self.base.base.diagnostics(),
            **self.base.diagnostics(),
            "knowledge_context": 1,
            "effect_observations": 0,
            "effect_bucket_count": 0,
        }


@dataclass(slots=True)
class _RelationalSkillCandidate:
    actions: tuple[Action, ...]
    templates: tuple[tuple[float, ...], ...]
    goals: tuple[str, ...]
    added: frozenset[str]
    removed: frozenset[str]
    successes: int = 0
    failures: int = 0


class RelationalSkillLibrary(SkillLibrary):
    """Promote successful ASeq by structural templates, not raw HTTP IDs."""

    def __init__(
        self,
        *,
        promotion_successes: int = 2,
        maximum_length: int = 12,
        representation: CurrentRepresentationBinding | None = None,
    ) -> None:
        super().__init__(
            promotion_successes=promotion_successes,
            maximum_length=maximum_length,
        )
        self._rel_candidates: dict[
            tuple[tuple[float, ...], ...], _RelationalSkillCandidate
        ] = {}
        self._templates: dict[str, tuple[tuple[float, ...], ...]] = {}
        self.representation = representation

    def _trace_templates(
        self,
        traces: Sequence[TransitionTrace],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            (
                self.representation.action_structure(trace.before, trace.action)
                if self.representation is not None
                else relational_action_key(trace.before, trace.action)
            )
            for trace in traces
        )

    def observe_goal_completion(
        self,
        traces: Iterable[TransitionTrace],
        *,
        achieved_goal_ids: Iterable[str],
    ) -> Skill | None:
        materialized = tuple(traces)[-self.maximum_length :]
        goals = tuple(sorted(set(achieved_goal_ids)))
        if not materialized or not goals:
            return None
        templates = self._trace_templates(materialized)
        if any(not template for template in templates):
            return None
        candidate = self._rel_candidates.get(templates)
        if candidate is None:
            candidate = _RelationalSkillCandidate(
                actions=tuple(trace.action for trace in materialized),
                templates=templates,
                goals=goals,
                added=frozenset().union(
                    *(trace.added_facts for trace in materialized)
                ),
                removed=frozenset().union(
                    *(trace.removed_facts for trace in materialized)
                ),
            )
            self._rel_candidates[templates] = candidate
        candidate.successes += 1
        if candidate.successes < self.promotion_successes:
            return None

        existing_id = next(
            (
                skill_id
                for skill_id, stored in self._templates.items()
                if stored == templates
            ),
            None,
        )
        if existing_id is not None:
            old = self._skills[existing_id]
            updated = replace(
                old,
                successes=candidate.successes,
                failures=candidate.failures,
            )
            self._skills[existing_id] = updated
            return updated

        skill_id = f"skill-{self._next_id:04d}"
        self._next_id += 1
        skill = Skill(
            skill_id=skill_id,
            primitive_actions=candidate.actions,
            achieved_goal_ids=goals,
            required_facts=frozenset(),
            added_facts=candidate.added,
            removed_facts=candidate.removed,
            successes=candidate.successes,
            failures=candidate.failures,
        )
        self._skills[skill_id] = skill
        self._templates[skill_id] = templates
        return skill

    def template_length(self, skill_id: str) -> int:
        return len(self._templates[skill_id])

    def resolve_primitive(
        self,
        skill_id: str,
        index: int,
        state: StateSnapshot,
    ) -> Action | None:
        templates = self._templates.get(skill_id)
        if templates is None or not 0 <= index < len(templates):
            return None
        target = templates[index]
        candidates = [
            action
            for action in state.available_actions
            if action.verb_name != SKILL_VERB
            and (
                self.representation.action_structure(state, action)
                if self.representation is not None
                else relational_action_key(state, action)
            ) == target
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda action: action.signature)

    def actions_for(self, state: StateSnapshot) -> tuple[Action, ...]:
        output = []
        for skill in self.all():
            if skill.reliability < 0.5:
                continue
            if self.resolve_primitive(skill.skill_id, 0, state) is not None:
                output.append(skill.as_action())
        return tuple(output)

    def diagnostics(self) -> dict[str, int]:
        return {
            "relational_candidates": len(self._rel_candidates),
            "relational_templates": len(self._templates),
            "promoted_skills": len(self._skills),
        }


class RelationalContextualSkillAwareProphecy(ContextualSkillAwareProphecy):
    def __init__(
        self,
        base: object,
        library: RelationalSkillLibrary,
        knowledge: KnowledgeStore | None = None,
    ) -> None:
        super().__init__(base, library, knowledge)
        self.library: RelationalSkillLibrary

    def _skill_predictions(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        knowledge: KnowledgeStore | None,
        samples: int,
    ) -> tuple[Prediction, ...]:
        skill_id = str(action.target)
        current = state
        probability = 1.0
        for index in range(self.library.template_length(skill_id)):
            primitive = self.library.resolve_primitive(skill_id, index, current)
            if primitive is None:
                return (
                    Prediction(
                        current,
                        0.0,
                        source=f"{self.name}:relational-skill-unavailable",
                    ),
                )
            if knowledge is not None:
                predictions = self._base_context_predictions(
                    current,
                    primitive,
                    knowledge=knowledge,
                    samples=max(1, samples if index == 0 else 1),
                )
            else:
                predictions = tuple(
                    self.base.predict(current, primitive, samples=max(1, samples if index == 0 else 1))
                )
            best = max(predictions, key=lambda item: item.probability)
            current = best.next_state
            probability *= best.probability
        return (
            Prediction(
                self.library.augment_state(current),
                max(0.0, min(1.0, probability)),
                source=f"{self.name}:relational-skill",
            ),
        )

    def predict_with_context(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        knowledge: KnowledgeStore,
        samples: int,
    ) -> tuple[Prediction, ...]:
        if action.verb_name != SKILL_VERB:
            return super().predict_with_context(
                state,
                action,
                knowledge=knowledge,
                samples=samples,
            )
        return self._skill_predictions(
            state,
            action,
            knowledge=knowledge,
            samples=samples,
        )

    def predict_step(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        memory: Any,
        samples: int,
    ) -> ProphecyStep:
        if action.verb_name != SKILL_VERB:
            return super().predict_step(
                state,
                action,
                memory=memory,
                samples=samples,
            )
        return ProphecyStep(
            self._skill_predictions(
                state,
                action,
                knowledge=self.knowledge,
                samples=samples,
            ),
            memory,
        )


class _RelationalCriticEncoder:
    def __init__(self, representation: CurrentRepresentationBinding | None = None) -> None:
        self.representation = representation
        self.feature_size = (
            representation.state_size * 2 + representation.action_feature_size + 1
            if representation is not None
            else AGENT_STATE_SIZE * 2 + ACTION_FEATURE_SIZE + 1
        )

    def encode(self, transition: CriticTransition) -> tuple[float, ...]:
        state_vector = (
            self.representation.state_vector
            if self.representation is not None
            else relational_state_vector
        )
        action_structure = (
            self.representation.action_structure
            if self.representation is not None
            else relational_action_key
        )
        return (
            state_vector(transition.before)
            + action_structure(transition.before, transition.action)
            + state_vector(transition.after)
            + (max(0.0, min(1.0, float(transition.prophecy_confidence))),)
        )


class RelationalGRUBranchCritic(GRUBranchCritic):
    name = "current-relational-gru-branch-critic"

    def __init__(
        self,
        seed: int,
        *,
        representation: CurrentRepresentationBinding | None = None,
    ) -> None:
        state_vector = (
            representation.state_vector
            if representation is not None
            else relational_state_vector
        )
        state_size = representation.state_size if representation is not None else AGENT_STATE_SIZE
        action_size = (
            representation.action_feature_size
            if representation is not None
            else ACTION_FEATURE_SIZE
        )
        super().__init__(
            state_vector,
            state_size,
            hidden_units=64,
            action_feature_size=action_size,
            batch_size=16,
            replay_capacity=4_000,
            gradient_steps_per_episode=2,
            seed=seed,
        )
        # Same dimensionality as the base encoder, but action identity is now
        # structural instead of raw-signature hashing.
        self.encoder = _RelationalCriticEncoder(representation)


@dataclass(frozen=True, slots=True)
class _PendingPolicyTransition:
    before: StateSnapshot
    action: Action
    after: StateSnapshot


class CurrentPentestAASSRAgent(IntegratedAASSRAgent):
    """Current-generation pentest AASSR.

    Legacy files remain importable for reproduction, but this class rewires the
    active runtime to the latest compatible Policy/Prophecy/Critic/ASEQ stack.
    """

    def __init__(
        self,
        *,
        seed: int,
        train_transitions: int,
        use_imagination: bool = True,
        device: str = "cpu",
    ) -> None:
        self.current_generation_version = CURRENT_GENERATION_VERSION
        self.current_components = dict(CURRENT_COMPONENTS)
        self.legacy_components_active = LEGACY_COMPONENTS_ACTIVE
        self.requested_imagination = bool(use_imagination)
        self.training_imagination = False

        self.dqn = RelationalInvariantDQN(
            int(seed) ^ 0xD1A6,
            train_transitions=int(train_transitions),
        )
        current_policy = CurrentRelationalPolicy(self.dqn)
        current_skills = RelationalSkillLibrary()
        neural = CurrentNeuralDeltaProphecy(
            HttpAgentCodec(),
            config=NeuralDeltaConfig(
                action_feature_size=ACTION_FEATURE_SIZE,
                hidden_units=128,
                ensemble_size=3,
                replay_capacity=50_000,
                batch_size=64,
                warmup_steps=128,
                learning_rate=1e-3,
                gradient_steps_per_observation=1,
                confidence_prior=256.0,
            ),
            seed=int(seed) ^ 0x4E455552,
            device=device,
        )

        core_config = AutonomousAgentConfig(
            learn_policy=False,
            learn_prophecy=False,
            use_imagination=bool(use_imagination),
            # Neural Delta superseded snapshot/effect composition in the latest
            # Imagination-v2 path. Do not silently reactivate the older model.
            use_effect_composition=False,
            imagination_depth=4,
            imagination_branching_factor=6,
            imagination_beam_width=24,
            imagination_outcome_samples=1,
            imagination_interval=1,
            imagination_minimum_coverage=0.55,
            imagination_intervention_margin=0.10,
            imagination_uncertainty_margin=1.0,
            imagination_aggregation="risk-adjusted",
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_episodes=800,
            exploration_bonus=0.0,
            effect_novelty_weight=0.0,
        )
        super().__init__(
            neural,
            core_config=core_config,
            integration_config=IntegratedAASSRConfig(
                expected_observation_contract=OBSERVATION_CONTRACT,
                preserve_knowledge_across_episodes=False,
            ),
            seed=int(seed) ^ 0xA441,
            semantic_state_key=semantic_fingerprint,
            skills=current_skills,
        )

        # Replace the constructor's legacy tabular contextual Policy immediately.
        self.policy = current_policy
        self.core.policy = current_policy
        self.core.planner.policy = current_policy

        self.base_neural_prophecy = neural
        self.calibrated_prophecy = ReplayRelationalCalibratedProphecy(
            neural,
            self.evaluator.replay,
        )
        self.knowledge_prophecy = KnowledgeBoundProphecy(self.calibrated_prophecy)
        self.skill_prophecy = RelationalContextualSkillAwareProphecy(
            self.knowledge_prophecy,
            current_skills,
            self.knowledge,
        )
        self.core.base_prophecy = self.skill_prophecy
        self.core.prophecy = self.skill_prophecy
        self.effect_prophecy = self.skill_prophecy
        self.prophecy = IntegratedProphecyView(
            self.skill_prophecy,
            self.skill_prophecy,
        )
        self.core.planner.prophecy = self.prophecy
        self.evaluator.prophecy = self.prophecy

        self.critic = RelationalGRUBranchCritic(int(seed) ^ 0x43524954)
        self.core.planner.scorer = self.critic
        self._critic_trajectory: list[CriticTransition] = []
        self._critic_counts: Counter[str] = Counter()
        self._pending_policy_transition: _PendingPolicyTransition | None = None

    @property
    def critic_ready(self) -> bool:
        stats = self.critic.stats()
        return (
            self._critic_counts["episodes"] >= 32
            and self._critic_counts["successes"] >= 4
            and self._critic_counts["non_successes"] >= 4
            and stats.gradient_updates > 0
        )

    def begin_episode(self, *, clear_knowledge: bool | None = None) -> None:
        # A pending transition would mean the previous episode was never closed.
        # Drop it rather than accidentally bootstrapping across scenario seeds.
        self._pending_policy_transition = None
        self._critic_trajectory.clear()
        super().begin_episode(clear_knowledge=clear_knowledge)

    def select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        explore: bool = True,
    ) -> IntegratedActionDecision:
        allow_imagination = (
            self.requested_imagination
            and self.critic_ready
            and (self.training_imagination or not explore)
        )
        original = self.core.config
        self.core.config = replace(original, use_imagination=allow_imagination)
        try:
            return super().select_action(
                state,
                episode=episode,
                explore=explore,
            )
        finally:
            self.core.config = original

    def _observe_policy_transition(
        self,
        transition: _PendingPolicyTransition,
        *,
        reward: float,
        terminal: bool,
    ) -> None:
        if terminal:
            self.dqn.mark_episode_boundary()
        self.dqn.observe(
            transition.before,
            transition.action,
            PluginOutcome(snapshot=transition.after),
            reward=float(reward),
        )

    def _flush_pending_policy(
        self,
        *,
        reward: float,
        terminal: bool,
    ) -> None:
        if self._pending_policy_transition is None:
            return
        self._observe_policy_transition(
            self._pending_policy_transition,
            reward=reward,
            terminal=terminal,
        )
        self._pending_policy_transition = None

    def _queue_policy_trace(self, trace: TransitionTrace) -> None:
        # Seeing another real transition proves the previous transition was not
        # the episode boundary. Learn it now with zero sparse reward.
        self._flush_pending_policy(reward=0.0, terminal=False)
        self._pending_policy_transition = _PendingPolicyTransition(
            trace.before,
            trace.action,
            trace.after,
        )

    @staticmethod
    def _trace_confidence(trace: TransitionTrace) -> float:
        values = []
        for prediction in trace.predictions:
            value = float(prediction.probability)
            source = prediction.source.lower()
            if source.endswith(":unseen"):
                value = 0.0
            elif "action-family" in source:
                value *= 0.5
            values.append(value)
        return max(values, default=0.0)

    def step(
        self,
        environment: object,
        *,
        episode: int,
        training: bool = True,
        primitive_budget: int | None = None,
    ) -> IntegratedAgentStep:
        if primitive_budget is not None and primitive_budget <= 0:
            raise ValueError("primitive_budget must be positive when supplied")
        if training:
            self._flush_pending_policy(reward=0.0, terminal=False)

        raw_before = environment.snapshot()
        decision = self.select_action(
            raw_before,
            episode=episode,
            explore=training,
        )
        selected = decision.action
        executed: list[Action] = []
        traces: list[TransitionTrace] = []
        evaluations = []
        skill_failed = False

        if selected.verb_name == SKILL_VERB:
            self._skill_uses += 1
            skill_id = str(selected.target)
            if training:
                self._selected_skill_steps.append(
                    (
                        self.skills.augment_state(raw_before),
                        selected,
                        len(self._episode_traces),
                    )
                )
            primitive_count = self.skills.template_length(skill_id)
            primitive_indices: Iterable[int] = range(primitive_count)
        else:
            skill_id = ""
            primitive_indices = (0,)

        for index in primitive_indices:
            if primitive_budget is not None and len(executed) >= primitive_budget:
                break
            if self._terminal(environment):
                break
            if selected.verb_name == SKILL_VERB:
                current_state = environment.snapshot()
                primitive = self.skills.resolve_primitive(
                    skill_id,
                    index,
                    current_state,
                )
                if primitive is None:
                    skill_failed = True
                    break
            else:
                primitive = selected

            trace, evaluation = self._execute_primitive(
                environment,
                primitive,
                training=training,
            )
            executed.append(primitive)
            traces.append(trace)
            if evaluation is not None:
                evaluations.append(evaluation)
            if training:
                self._queue_policy_trace(trace)
                self._critic_trajectory.append(
                    CriticTransition(
                        trace.before,
                        trace.action,
                        trace.after,
                        self._trace_confidence(trace),
                    )
                )
            if trace.error:
                skill_failed = selected.verb_name == SKILL_VERB
                break

        if skill_failed and training and selected.verb_name == SKILL_VERB:
            self.skills.record_failure(skill_id)

        raw_after = environment.snapshot()
        semantic_after = self.semantic_state_key(raw_after)
        if self.integration_config.use_aseq:
            self.aseq.observe(
                decision.semantic_state,
                selected,
                semantic_after,
            )

        newly_achieved, promoted = self._goal_update(
            raw_after,
            training=training,
        )
        self._steps += 1
        return IntegratedAgentStep(
            decision,
            tuple(executed),
            tuple(traces),
            tuple(evaluations),
            newly_achieved,
            promoted,
            self._terminal(environment),
        )

    def finish_episode(
        self,
        *,
        final_return: float,
        training: bool = True,
    ) -> None:
        if training:
            self._flush_pending_policy(
                reward=float(final_return),
                terminal=True,
            )
            if self._critic_trajectory:
                success = float(final_return) > 0.0
                self.critic.observe_episode(
                    tuple(self._critic_trajectory),
                    success=success,
                )
                self._critic_counts["episodes"] += 1
                self._critic_counts[
                    "successes" if success else "non_successes"
                ] += 1

        if training and self._episode_evaluations:
            credits = self.evaluator.finish_episode(
                self._episode_evaluations,
                final_return=float(final_return),
                policy=None,
            )
            by_id = {
                evaluation.trace.trace_id: evaluation
                for evaluation in self._episode_evaluations
            }
            for credited in credits:
                evaluation = by_id[credited.trace_id]
                learned_information_value = self.evaluator.predictor.predict(
                    evaluation.features
                )
                self.policy.observe_information_return(
                    evaluation.trace.before,
                    evaluation.trace.action,
                    self.integration_config.information_value_weight
                    * learned_information_value,
                )
                self._learn_feature_use(
                    evaluation.trace.action,
                    credited.credit,
                )

            for state, action, trace_index in self._selected_skill_steps:
                remaining = max(0, len(self._episode_traces) - trace_index - 1)
                macro_credit = float(final_return) * (
                    self.core.config.gamma ** remaining
                )
                self.policy.observe_return(state, action, macro_credit)

        self._episode_evaluations.clear()
        self._episode_traces.clear()
        self._previous_goal_ids.clear()
        self._selected_skill_steps.clear()
        self._critic_trajectory.clear()
        self.aseq.reset_episode()
        recent = getattr(self.evaluator, "_recent_pairs", None)
        if isinstance(recent, list):
            recent.clear()

    def diagnostics(self) -> dict[str, Any]:
        base = super().diagnostics()
        critic = self.critic.stats()
        base.update(
            {
                "version": CURRENT_GENERATION_VERSION,
                "current_generation": True,
                "current_components": dict(self.current_components),
                "legacy_components_active": list(self.legacy_components_active),
                "semantic_state_contract_shared": False,
                "identity_contracts": {
                    "aseq_cycle_detection": "concrete-response-semantic-v3",
                    "policy_transfer": "relational-structural-v1",
                    "critic_transfer": "relational-structural-v1",
                    "skill_transfer": "relational-action-template-v1",
                },
                "effect_composition_active": False,
                "training_imagination": self.training_imagination,
                "critic_ready": self.critic_ready,
                "critic": {
                    "episodes": critic.episodes,
                    "transitions": critic.transitions,
                    "gradient_updates": critic.gradient_updates,
                    "mean_loss": critic.mean_loss,
                    "parameter_count": critic.parameter_count,
                    **dict(self._critic_counts),
                },
                "policy": self.policy.diagnostics(),
                "prophecy_current": self.base_neural_prophecy.diagnostics(),
                "prophecy_calibration": self.calibrated_prophecy.diagnostics(),
                "relational_skills": self.skills.diagnostics(),
            }
        )
        return base


def build_current_pentest_aassr_core(
    *,
    seed: int = 0,
    train_transitions: int = 10_000,
    use_imagination: bool = True,
    device: str = "cpu",
) -> CurrentPentestAASSRAgent:
    """Canonical current-generation pentest AASSR entrypoint."""

    return CurrentPentestAASSRAgent(
        seed=int(seed),
        train_transitions=int(train_transitions),
        use_imagination=bool(use_imagination),
        device=device,
    )
