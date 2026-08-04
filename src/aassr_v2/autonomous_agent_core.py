from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
from math import log, sqrt
import random
from statistics import fmean
from typing import Mapping

from .effect_prophecy import EffectComposedProphecy
from .imagination_tree import ImaginationConfig, ImaginationTree, StateDeltaScorer
from .metrics import expected_prediction_vector, prediction_similarity
from .policy import PolicyMemory, ScoredAction
from .types import Action, StateSnapshot


StateKey = tuple[tuple[float, ...], tuple[str, ...]]


def state_key(state: StateSnapshot) -> StateKey:
    """Return a semantic-name-agnostic state key."""

    return (
        tuple(round(value, 8) for value in state.vector),
        tuple(sorted(state.facts)),
    )


@dataclass(slots=True)
class RunningValue:
    count: int = 0
    mean: float = 0.0

    def observe(self, value: float, *, learning_rate: float | None = None) -> None:
        self.count += 1
        rate = learning_rate if learning_rate is not None else 1.0 / self.count
        self.mean += rate * (value - self.mean)


class ContextualPolicy:
    """State-conditioned policy over opaque actions."""

    def __init__(self, *, learning_rate: float = 0.2) -> None:
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        self.learning_rate = learning_rate
        self._local: dict[tuple[StateKey, str], RunningValue] = {}
        self._global: dict[str, RunningValue] = {}
        self._state_visits: dict[StateKey, int] = {}

    def _entry(self, state: StateSnapshot, action: Action) -> RunningValue:
        return self._local.get((state_key(state), action.signature), RunningValue())

    def value(self, state: StateSnapshot, action: Action) -> float:
        local = self._entry(state, action)
        if local.count:
            return local.mean
        return self._global.get(action.signature, RunningValue()).mean

    def rank(
        self,
        state: StateSnapshot,
        *,
        limit: int,
        memory: PolicyMemory | None = None,
    ) -> tuple[ScoredAction, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        deltas: Mapping[str, float] = {} if memory is None else memory.deltas
        ranked = sorted(
            (
                ScoredAction(
                    action,
                    self.value(state, action)
                    + deltas.get(action.signature, 0.0),
                )
                for action in state.available_actions
            ),
            key=lambda item: (-item.score, item.action.signature),
        )
        return tuple(ranked[:limit])

    def select(
        self,
        state: StateSnapshot,
        *,
        randomizer: random.Random,
        epsilon: float,
        exploration_bonus: float,
    ) -> Action:
        if not state.available_actions:
            raise ValueError("state has no available actions")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if epsilon > 0.0 and randomizer.random() < epsilon:
            return randomizer.choice(state.available_actions)
        key = state_key(state)
        total = self._state_visits.get(key, 0)
        scored = []
        for action in state.available_actions:
            entry = self._entry(state, action)
            bonus = exploration_bonus * sqrt(
                log(total + 2.0) / (entry.count + 1.0)
            )
            scored.append(
                ScoredAction(action, self.value(state, action) + bonus)
            )
        scored.sort(key=lambda item: (-item.score, item.action.signature))
        return scored[0].action

    def observe_return(
        self,
        state: StateSnapshot,
        action: Action,
        target: float,
    ) -> None:
        key = state_key(state)
        self._state_visits[key] = self._state_visits.get(key, 0) + 1
        local = self._local.setdefault((key, action.signature), RunningValue())
        local.observe(target, learning_rate=self.learning_rate)
        global_entry = self._global.setdefault(
            action.signature,
            RunningValue(),
        )
        global_entry.observe(target, learning_rate=self.learning_rate)

    def imagine_update(
        self,
        memory: PolicyMemory,
        action: Action,
        value: float,
    ) -> PolicyMemory:
        deltas = dict(memory.deltas)
        deltas[action.signature] = (
            deltas.get(action.signature, 0.0) + 0.1 * value
        )
        return PolicyMemory(deltas)

    def reinforce(self, action: Action, advantage: float) -> None:
        entry = self._global.setdefault(action.signature, RunningValue())
        entry.observe(advantage, learning_rate=self.learning_rate)


@dataclass(frozen=True, slots=True)
class HoldoutTransition:
    before: StateSnapshot
    action: Action
    after: StateSnapshot


@dataclass(slots=True)
class FrozenHoldout:
    stride: int = 5
    minimum_count: int = 4
    capacity: int = 512
    seed: int = 0
    _seen: int = 0
    _items: list[HoldoutTransition] = field(default_factory=list)
    _randomizer: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.stride <= 1:
            raise ValueError("stride must exceed one")
        if self.minimum_count <= 0 or self.capacity <= 0:
            raise ValueError("minimum_count and capacity must be positive")
        self._randomizer = random.Random(self.seed)

    def next_is_train(self) -> bool:
        return self._randomizer.randrange(self.stride) != 0

    def commit(self, transition: HoldoutTransition, *, train: bool) -> None:
        self._seen += 1
        if train:
            return
        self._items.append(transition)
        if len(self._items) > self.capacity:
            self._items.pop(0)

    @property
    def ready(self) -> bool:
        return len(self._items) >= self.minimum_count

    def score(
        self,
        prophecy: object,
        *,
        samples: int = 1,
        limit: int = 32,
    ) -> float:
        if not self.ready:
            return 0.0
        if limit <= 0:
            raise ValueError("limit must be positive")
        values = []
        for item in self._items[-limit:]:
            predictions = prophecy.predict(
                item.before,
                item.action,
                samples=samples,
            )
            values.append(
                prediction_similarity(
                    expected_prediction_vector(predictions),
                    item.after.vector,
                )
            )
        return fmean(values) if values else 0.0


@dataclass(frozen=True, slots=True)
class AutonomousAgentConfig:
    gamma: float = 0.97
    epsilon_start: float = 0.8
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 1000
    exploration_bonus: float = 0.3
    policy_learning_rate: float = 0.2
    learn_policy: bool = True
    learn_prophecy: bool = True
    random_policy: bool = False
    use_imagination: bool = True
    imagination_depth: int = 8
    imagination_branching_factor: int = 2
    imagination_beam_width: int = 32
    imagination_minimum_coverage: float = 0.35
    imagination_intervention_margin: float = 0.05
    imagination_uncertainty_margin: float = 0.20
    validated_gain_weight: float = 0.2
    repeat_penalty: float = 0.05
    error_penalty: float = 0.2
    holdout_stride: int = 5
    minimum_holdout_count: int = 4
    holdout_evaluation_limit: int = 32
    validation_interval: int = 8
    imagination_interval: int = 1
    imagination_aggregation: str = "max"
    effect_novelty_weight: float = 0.0
    extrinsic_reward_weight: float = 1.0
    use_effect_composition: bool = True
    effect_minimum_samples: int = 2

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon bounds are invalid")
        if self.epsilon_decay_episodes <= 0:
            raise ValueError("epsilon_decay_episodes must be positive")
        if self.imagination_depth <= 0:
            raise ValueError("imagination_depth must be positive")
        if self.holdout_evaluation_limit <= 0:
            raise ValueError("holdout_evaluation_limit must be positive")
        if self.validation_interval <= 0 or self.imagination_interval <= 0:
            raise ValueError("intervals must be positive")
        if not 0.0 <= self.imagination_minimum_coverage <= 1.0:
            raise ValueError(
                "imagination_minimum_coverage must be in [0, 1]"
            )
        if self.imagination_intervention_margin < 0.0:
            raise ValueError(
                "imagination_intervention_margin must be non-negative"
            )
        if self.imagination_uncertainty_margin < 0.0:
            raise ValueError(
                "imagination_uncertainty_margin must be non-negative"
            )
        if self.imagination_aggregation not in {
            "max",
            "mean",
            "risk-adjusted",
        }:
            raise ValueError(
                "imagination_aggregation must be max, mean, or risk-adjusted"
            )
        if self.effect_novelty_weight < 0.0:
            raise ValueError("effect_novelty_weight must be non-negative")
        if self.extrinsic_reward_weight < 0.0:
            raise ValueError("extrinsic_reward_weight must be non-negative")
        if self.effect_minimum_samples <= 0:
            raise ValueError("effect_minimum_samples must be positive")


@dataclass(frozen=True, slots=True)
class ActionDecision:
    action: Action
    used_imagination: bool
    imagined_nodes: int = 0
    imagination_depth: int = 0
    root_imagined_value: float = 0.0
    policy_action_signature: str = ""
    imagination_opportunity: bool = False
    imagination_eligible: bool = False
    imagination_gate_reason: str = "disabled"
    imagination_changed_action: bool = False
    model_coverage: float = 0.0
    imagination_preferred_action_signature: str = ""
    imagination_policy_value: float = 0.0
    imagination_preferred_value: float = 0.0
    imagination_advantage: float = 0.0
    imagination_required_advantage: float = 0.0
    imagination_switch_candidate: bool = False
    imagination_intervention_allowed: bool = False


@dataclass(frozen=True, slots=True)
class ObservationMetrics:
    prediction_score: float
    holdout_before: float
    holdout_after: float
    holdout_gain: float
    intrinsic_value: float
    repeated: bool
    error: bool


@dataclass(frozen=True, slots=True)
class _EpisodeTransition:
    state: StateSnapshot
    action: Action
    intrinsic_value: float


class AutonomousLearningAgent:
    """Online agent that discovers and composes transition effects."""

    def __init__(
        self,
        prophecy: object,
        *,
        config: AutonomousAgentConfig | None = None,
        seed: int = 0,
        policy: ContextualPolicy | None = None,
    ) -> None:
        self.config = config or AutonomousAgentConfig()
        self.base_prophecy = (
            prophecy.base
            if isinstance(prophecy, EffectComposedProphecy)
            else prophecy
        )
        self.prophecy = (
            prophecy
            if isinstance(prophecy, EffectComposedProphecy)
            else EffectComposedProphecy(
                prophecy,
                minimum_samples=self.config.effect_minimum_samples,
            )
            if self.config.use_effect_composition
            else prophecy
        )
        self._seed = int(seed)
        self.randomizer = random.Random(seed)
        self.policy = policy or ContextualPolicy(
            learning_rate=self.config.policy_learning_rate
        )
        self.holdout = FrozenHoldout(
            stride=self.config.holdout_stride,
            minimum_count=self.config.minimum_holdout_count,
            seed=seed ^ 0x5A17,
        )
        self._episode: list[_EpisodeTransition] = []
        self._transition_index = 0
        self._decision_index = 0
        self._recent_pairs: list[tuple[StateKey, str]] = []
        self._seen_effect_motifs: set[
            tuple[float | int | bool, ...]
        ] = set()
        self._imagination_diagnostics: Counter[str] = Counter()
        self.planner = ImaginationTree(
            self.policy,
            self.prophecy,
            config=ImaginationConfig(
                branching_factor=self.config.imagination_branching_factor,
                maximum_depth=self.config.imagination_depth,
                beam_width=self.config.imagination_beam_width,
                outcome_samples=1,
                minimum_path_confidence=0.1,
                uncertainty_penalty=0.2,
                aggregation=(
                    "risk_adjusted"
                    if self.config.imagination_aggregation == "risk-adjusted"
                    else self.config.imagination_aggregation
                ),
                update_policy=False,
            ),
            scorer=StateDeltaScorer(
                goal_progress_weight=10.0,
                new_fact_weight=0.0,
                unlocked_action_weight=0.0,
                step_cost=0.01,
            ),
        )

    def epsilon(self, episode: int) -> float:
        fraction = min(
            1.0,
            max(0.0, episode / self.config.epsilon_decay_episodes),
        )
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def model_coverage(self, state: StateSnapshot) -> float:
        if not state.available_actions:
            return 1.0
        coverage = getattr(self.prophecy, "coverage", None)
        if callable(coverage):
            return max(
                0.0,
                min(
                    1.0,
                    float(coverage(state, state.available_actions)),
                ),
            )
        known = 0.0
        for action in state.available_actions:
            prediction = self.prophecy.predict(state, action, samples=1)[0]
            source = prediction.source.lower()
            if source.endswith(":exact"):
                known += 1.0
            elif source.endswith(":action-family"):
                known += 0.5
        return known / len(state.available_actions)

    def _record_decision(self, decision: ActionDecision) -> ActionDecision:
        if decision.imagination_opportunity:
            self._imagination_diagnostics["opportunities"] += 1
        if decision.imagination_eligible:
            self._imagination_diagnostics["eligible"] += 1
        if decision.used_imagination:
            self._imagination_diagnostics["runs"] += 1
        if decision.imagination_switch_candidate:
            self._imagination_diagnostics["switch_candidates"] += 1
        if decision.imagination_intervention_allowed:
            self._imagination_diagnostics["interventions"] += 1
        if (
            decision.imagination_switch_candidate
            and not decision.imagination_intervention_allowed
        ):
            self._imagination_diagnostics["suppressed_switches"] += 1
        if decision.imagination_changed_action:
            self._imagination_diagnostics["changed_actions"] += 1
        self._imagination_diagnostics[
            f"gate:{decision.imagination_gate_reason}"
        ] += 1
        return decision

    def imagination_diagnostics(self) -> dict[str, int | float]:
        opportunities = self._imagination_diagnostics["opportunities"]
        changed = self._imagination_diagnostics["changed_actions"]
        runs = self._imagination_diagnostics["runs"]
        return {
            **dict(self._imagination_diagnostics),
            "change_rate_per_run": changed / runs if runs else 0.0,
            "intervention_rate_per_candidate": (
                self._imagination_diagnostics["interventions"]
                / self._imagination_diagnostics["switch_candidates"]
                if self._imagination_diagnostics["switch_candidates"]
                else 0.0
            ),
            "eligibility_rate": (
                self._imagination_diagnostics["eligible"] / opportunities
                if opportunities
                else 0.0
            ),
        }

    def prophecy_diagnostics(self) -> dict[str, int]:
        diagnostics = getattr(self.prophecy, "diagnostics", None)
        return dict(diagnostics()) if callable(diagnostics) else {}

    def select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        explore: bool = True,
    ) -> ActionDecision:
        if not state.available_actions:
            raise ValueError("state has no available actions")
        epsilon = self.epsilon(episode) if explore else 0.0
        if self.config.random_policy:
            if explore:
                action = self.randomizer.choice(state.available_actions)
            else:
                digest = hashlib.sha256(
                    (
                        f"{self._seed}:{episode}:"
                        f"{state_key(state)!r}:"
                        f"{tuple(item.signature for item in state.available_actions)!r}"
                    ).encode("utf-8")
                ).digest()
                action = state.available_actions[
                    int.from_bytes(digest[:8], "big")
                    % len(state.available_actions)
                ]
            return self._record_decision(
                ActionDecision(
                    action,
                    False,
                    policy_action_signature=action.signature,
                    imagination_gate_reason="random_policy",
                )
            )

        if explore and self.randomizer.random() < epsilon:
            action = self.randomizer.choice(state.available_actions)
            return self._record_decision(
                ActionDecision(
                    action,
                    False,
                    policy_action_signature=action.signature,
                    imagination_gate_reason="epsilon_random",
                )
            )

        policy_action = self.policy.select(
            state,
            randomizer=self.randomizer,
            epsilon=0.0,
            exploration_bonus=(
                self.config.exploration_bonus if explore else 0.0
            ),
        )
        decision_index = self._decision_index + 1
        if explore:
            self._decision_index = decision_index
        coverage = self.model_coverage(state)
        opportunity = self.config.use_imagination
        interval_passed = (
            decision_index % self.config.imagination_interval == 0
        )
        eligible = (
            opportunity
            and interval_passed
            and coverage >= self.config.imagination_minimum_coverage
        )

        if not opportunity:
            reason = "disabled"
        elif not interval_passed:
            reason = "interval"
        elif coverage < self.config.imagination_minimum_coverage:
            reason = "coverage"
        else:
            reason = "eligible"

        if eligible:
            plan = self.planner.plan(state)
            best_imagined = max(
                item.aggregate_value for item in plan.root_evaluations
            )
            candidates = [
                item
                for item in plan.root_evaluations
                if abs(item.aggregate_value - best_imagined) <= 1e-12
            ]
            preferred = min(
                candidates,
                key=lambda item: (
                    -self.policy.value(state, item.action),
                    item.action.signature,
                ),
            )
            policy_evaluation = next(
                (
                    item
                    for item in plan.root_evaluations
                    if item.action.signature == policy_action.signature
                ),
                None,
            )
            switch_candidate = (
                preferred.action.signature != policy_action.signature
            )
            policy_value = (
                policy_evaluation.aggregate_value
                if policy_evaluation is not None
                else preferred.aggregate_value
            )
            advantage = (
                preferred.aggregate_value - policy_value
                if policy_evaluation is not None
                else 0.0
            )
            required_advantage = (
                self.config.imagination_intervention_margin
                + self.config.imagination_uncertainty_margin
                * (1.0 - coverage)
            )
            intervention_allowed = (
                switch_candidate
                and policy_evaluation is not None
                and advantage >= required_advantage
            )
            if not switch_candidate:
                intervention_reason = "policy_agreement"
            elif policy_evaluation is None:
                intervention_reason = "policy_not_evaluated"
            elif intervention_allowed:
                intervention_reason = "intervention"
            else:
                intervention_reason = "insufficient_advantage"
            executed_action = (
                preferred.action if intervention_allowed else policy_action
            )
            executed_value = (
                preferred.aggregate_value
                if intervention_allowed
                else policy_value
            )
            return self._record_decision(
                ActionDecision(
                    executed_action,
                    True,
                    imagined_nodes=len(plan.nodes),
                    imagination_depth=plan.maximum_depth_reached,
                    root_imagined_value=executed_value,
                    policy_action_signature=policy_action.signature,
                    imagination_opportunity=True,
                    imagination_eligible=True,
                    imagination_gate_reason=intervention_reason,
                    imagination_changed_action=intervention_allowed,
                    model_coverage=coverage,
                    imagination_preferred_action_signature=(
                        preferred.action.signature
                    ),
                    imagination_policy_value=policy_value,
                    imagination_preferred_value=(
                        preferred.aggregate_value
                    ),
                    imagination_advantage=advantage,
                    imagination_required_advantage=required_advantage,
                    imagination_switch_candidate=switch_candidate,
                    imagination_intervention_allowed=intervention_allowed,
                )
            )

        return self._record_decision(
            ActionDecision(
                policy_action,
                False,
                policy_action_signature=policy_action.signature,
                imagination_opportunity=opportunity,
                imagination_eligible=False,
                imagination_gate_reason=reason,
                model_coverage=coverage,
            )
        )

    def observe(
        self,
        before: StateSnapshot,
        action: Action,
        outcome: object,
    ) -> ObservationMetrics:
        after: StateSnapshot = outcome.snapshot
        predictions_before = self.prophecy.predict(
            before,
            action,
            samples=1,
        )
        prediction_score = prediction_similarity(
            expected_prediction_vector(predictions_before),
            after.vector,
        )
        self._transition_index += 1
        validate_now = (
            self.config.validated_gain_weight > 0.0
            and self._transition_index % self.config.validation_interval == 0
        )
        holdout_before = (
            self.holdout.score(
                self.prophecy,
                limit=self.config.holdout_evaluation_limit,
            )
            if validate_now
            else 0.0
        )
        train = self.holdout.next_is_train()
        if self.config.learn_prophecy and train:
            self.prophecy.learn(before, action, after)
        holdout_after = (
            self.holdout.score(
                self.prophecy,
                limit=self.config.holdout_evaluation_limit,
            )
            if validate_now
            else 0.0
        )
        gain = (
            holdout_after - holdout_before
            if validate_now and self.holdout.ready
            else 0.0
        )
        self.holdout.commit(
            HoldoutTransition(before, action, after),
            train=train,
        )
        pair = (state_key(before), action.signature)
        repeated = pair in self._recent_pairs[-16:]
        self._recent_pairs.append(pair)
        error = bool(getattr(outcome, "error", False))
        intrinsic = self.config.validated_gain_weight * max(0.0, gain)
        effect_motif: tuple[float | int | bool, ...] = (
            *(
                round(right - left, 6)
                for left, right in zip(
                    before.vector,
                    after.vector,
                    strict=False,
                )
            ),
            len(after.facts - before.facts),
            len(before.facts - after.facts),
            len(getattr(outcome, "unlocked_actions", ())),
            error,
            round(after.goal_progress - before.goal_progress, 6),
        )
        if effect_motif not in self._seen_effect_motifs:
            intrinsic += self.config.effect_novelty_weight
            self._seen_effect_motifs.add(effect_motif)
        if repeated:
            intrinsic -= self.config.repeat_penalty
        if error:
            intrinsic -= self.config.error_penalty
        self._episode.append(_EpisodeTransition(before, action, intrinsic))
        return ObservationMetrics(
            prediction_score,
            holdout_before,
            holdout_after,
            gain,
            intrinsic,
            repeated,
            error,
        )

    def finish_episode(self, *, final_return: float) -> None:
        if self.config.learn_policy and not self.config.random_policy:
            future = (
                float(final_return)
                * self.config.extrinsic_reward_weight
            )
            for transition in reversed(self._episode):
                target = future + transition.intrinsic_value
                self.policy.observe_return(
                    transition.state,
                    transition.action,
                    target,
                )
                future *= self.config.gamma
        self._episode.clear()
        self._recent_pairs.clear()

    def discard_episode(self) -> None:
        self._episode.clear()
        self._recent_pairs.clear()
