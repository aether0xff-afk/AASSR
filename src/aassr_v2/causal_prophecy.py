from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Mapping

from .causal_representation import CausalEncoder, ObservableTransition
from .paper_v2_types import (
    CausalProphecyPredictionV20,
    CausalProphecyPredictionV21,
    RawCausalObservation,
)


@dataclass(slots=True)
class _OutcomeAccumulator:
    visits: int = 0
    effect_totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    unlock_total: float = 0.0
    resource_total: float = 0.0
    damage_total: float = 0.0
    return_probability: float = 0.0
    return_updates: int = 0
    last_next_state: RawCausalObservation | None = None
    prediction_errors: deque[float] = field(default_factory=lambda: deque(maxlen=64))
    calibration_errors: deque[float] = field(default_factory=lambda: deque(maxlen=64))


def _effect_vector(transition: ObservableTransition) -> dict[str, float]:
    return {
        "inventory_change": float(
            sum(abs(value) for value in transition.inventory_delta.values())
        ),
        "facts_added": float(transition.facts_added),
        "facts_removed": float(transition.facts_removed),
        "unlocked_actions": float(transition.unlocked_actions),
        "spatial_changed": float(transition.spatial_changed),
        "action_succeeded": float(transition.action_succeeded),
        "terminal_reward": float(transition.terminal_reward),
    }


class EmpiricalCausalProphecy:
    """Observable multi-head model with outcome-derived success targets.

    No method accepts a private world state, viability flag, solution family,
    or oracle transition.  Return targets enter only through ``finish_episode``.
    """

    def __init__(
        self,
        encoder: CausalEncoder,
        *,
        gamma: float = 0.97,
        return_target_mode: str = "monte_carlo",
        reliability_bins: int = 10,
    ) -> None:
        if return_target_mode not in {"monte_carlo", "td"}:
            raise ValueError("return_target_mode must be monte_carlo or td")
        if reliability_bins < 2:
            raise ValueError("reliability_bins must be at least two")
        self.encoder = encoder
        self.gamma = float(gamma)
        self.return_target_mode = return_target_mode
        self.reliability_bins = int(reliability_bins)
        self._outcomes: dict[tuple[str, str], _OutcomeAccumulator] = {}
        self._episode_keys: list[tuple[str, str]] = []
        self.total_updates = 0

    def _key(self, observation: RawCausalObservation, action: str) -> tuple[str, str]:
        return (
            self.encoder.state_key(observation),
            self.encoder.action_key(observation, action),
        )

    def predict_v20(
        self, observation: RawCausalObservation, action: str
    ) -> CausalProphecyPredictionV20:
        outcome = self._outcomes.get(self._key(observation, action))
        if outcome is None or outcome.visits == 0:
            return CausalProphecyPredictionV20(None, {}, 0.0, 0.0, 0)
        return CausalProphecyPredictionV20(
            next_observable_state=outcome.last_next_state,
            observable_effect_delta={
                key: value / outcome.visits
                for key, value in outcome.effect_totals.items()
            },
            action_unlock_probability=outcome.unlock_total / outcome.visits,
            terminal_return_probability=max(
                0.0, min(1.0, outcome.return_probability)
            ),
            visit_count=outcome.visits,
        )

    def predict_v21(
        self, observation: RawCausalObservation, action: str
    ) -> CausalProphecyPredictionV21:
        base = self.predict_v20(observation, action)
        outcome = self._outcomes.get(self._key(observation, action))
        if outcome is None or outcome.visits == 0:
            return CausalProphecyPredictionV21(base, 0.0, 0.0, 1.0, 1.0, 0.0)
        count_uncertainty = 1.0 / math.sqrt(outcome.visits + 1.0)
        holdout_error = fmean(outcome.prediction_errors) if outcome.prediction_errors else 1.0
        uncertainty = max(0.0, min(1.0, 0.5 * count_uncertainty + 0.5 * holdout_error))
        ood = 1.0 / (outcome.visits + 1.0)
        brier = fmean(outcome.calibration_errors) if outcome.calibration_errors else 1.0
        confidence = max(0.0, min(1.0, 1.0 - 0.5 * (uncertainty + brier)))
        return CausalProphecyPredictionV21(
            base=base,
            expected_resource_cost=outcome.resource_total / outcome.visits,
            expected_damage=outcome.damage_total / outcome.visits,
            uncertainty=uncertainty,
            ood_score=ood,
            calibration_confidence=confidence,
        )

    def observe_transition(self, transition: ObservableTransition) -> None:
        key = self._key(transition.before, transition.action)
        prediction = self.predict_v20(transition.before, transition.action)
        actual = _effect_vector(transition)
        if prediction.visit_count:
            predicted = prediction.observable_effect_delta
            names = set(actual) | set(predicted)
            error = fmean(
                min(1.0, abs(actual.get(name, 0.0) - predicted.get(name, 0.0)))
                for name in names
            )
        else:
            error = 1.0
        outcome = self._outcomes.setdefault(key, _OutcomeAccumulator())
        outcome.prediction_errors.append(error)
        outcome.visits += 1
        for name, value in actual.items():
            outcome.effect_totals[name] += value
        outcome.unlock_total += float(transition.unlocked_actions > 0)
        outcome.resource_total += transition.resource_cost
        outcome.damage_total += transition.damage
        outcome.last_next_state = transition.after
        self.encoder.observe(transition)
        learned_key = self._key(transition.before, transition.action)
        if learned_key != key:
            self._outcomes[learned_key] = outcome
            key = learned_key
        self._episode_keys.append(key)
        self.total_updates += 1

    def finish_episode(self, terminal_success: bool) -> None:
        target = float(terminal_success)
        next_value = target
        for distance, key in enumerate(reversed(self._episode_keys)):
            outcome = self._outcomes[key]
            prediction_before = outcome.return_probability
            if self.return_target_mode == "monte_carlo":
                update_target = (self.gamma ** distance) * target
            else:
                update_target = target if distance == 0 else self.gamma * next_value
            outcome.return_updates += 1
            rate = 1.0 / outcome.return_updates
            outcome.return_probability += rate * (
                update_target - outcome.return_probability
            )
            outcome.calibration_errors.append(
                (prediction_before - update_target) ** 2
            )
            next_value = outcome.return_probability
        self._episode_keys.clear()

    def calibration_metrics(self) -> dict[str, float]:
        errors = [
            value
            for outcome in self._outcomes.values()
            for value in outcome.calibration_errors
        ]
        prediction_errors = [
            value
            for outcome in self._outcomes.values()
            for value in outcome.prediction_errors
        ]
        return {
            "return_brier_score": fmean(errors) if errors else 1.0,
            "observable_effect_error": fmean(prediction_errors)
            if prediction_errors
            else 1.0,
            "state_action_coverage": float(len(self._outcomes)),
        }

    def export(self) -> dict[str, Any]:
        return {
            "gamma": self.gamma,
            "return_target_mode": self.return_target_mode,
            "reliability_bins": self.reliability_bins,
            "total_updates": self.total_updates,
            "encoder": dict(self.encoder.export()),
            "outcomes": {
                repr(key): {
                    "visits": value.visits,
                    "effect_totals": dict(value.effect_totals),
                    "unlock_total": value.unlock_total,
                    "resource_total": value.resource_total,
                    "damage_total": value.damage_total,
                    "return_probability": value.return_probability,
                    "return_updates": value.return_updates,
                    "last_next_state": None
                    if value.last_next_state is None
                    else value.last_next_state.to_dict(),
                    "prediction_errors": list(value.prediction_errors),
                    "calibration_errors": list(value.calibration_errors),
                }
                for key, value in self._outcomes.items()
            },
        }

    def restore(self, payload: Mapping[str, Any]) -> None:
        self.gamma = float(payload["gamma"])
        self.return_target_mode = str(payload["return_target_mode"])
        self.reliability_bins = int(payload["reliability_bins"])
        self.total_updates = int(payload.get("total_updates", 0))
        self.encoder.restore(dict(payload.get("encoder", {})))
        self._outcomes = {}
        for raw_key, raw_value in dict(payload.get("outcomes", {})).items():
            key = tuple(__import__("ast").literal_eval(raw_key))
            value = dict(raw_value)
            raw_state = value.get("last_next_state")
            state = None
            if isinstance(raw_state, Mapping):
                state = RawCausalObservation(
                    inventory={
                        str(name): int(amount)
                        for name, amount in dict(raw_state.get("inventory", {})).items()
                    },
                    observable_facts=frozenset(raw_state.get("observable_facts", ())),
                    available_actions=tuple(raw_state.get("available_actions", ())),
                    action_affordances={
                        str(name): tuple(items)
                        for name, items in dict(raw_state.get("action_affordances", {})).items()
                    },
                    resource_cost=float(raw_state.get("resource_cost", 0.0)),
                    health=float(raw_state.get("health", 1.0)),
                    damage=float(raw_state.get("damage", 0.0)),
                    spatial_observations=dict(raw_state.get("spatial_observations", {})),
                    last_action_succeeded=raw_state.get("last_action_succeeded"),
                    terminal_reward=float(raw_state.get("terminal_reward", 0.0)),
                    terminal=bool(raw_state.get("terminal", False)),
                )
            accumulator = _OutcomeAccumulator(
                visits=int(value.get("visits", 0)),
                effect_totals=defaultdict(
                    float,
                    {
                        str(name): float(number)
                        for name, number in dict(value.get("effect_totals", {})).items()
                    },
                ),
                unlock_total=float(value.get("unlock_total", 0.0)),
                resource_total=float(value.get("resource_total", 0.0)),
                damage_total=float(value.get("damage_total", 0.0)),
                return_probability=float(value.get("return_probability", 0.0)),
                return_updates=int(value.get("return_updates", 0)),
                last_next_state=state,
                prediction_errors=deque(value.get("prediction_errors", ()), maxlen=64),
                calibration_errors=deque(value.get("calibration_errors", ()), maxlen=64),
            )
            self._outcomes[(str(key[0]), str(key[1]))] = accumulator
        self._episode_keys.clear()
