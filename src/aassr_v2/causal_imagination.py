from __future__ import annotations

import copy
import hashlib
import random
from collections import deque
from dataclasses import dataclass, replace
from statistics import fmean
from typing import Any, Mapping, Protocol, Sequence

from .causal_dependency_world import CausalDependencyWorldV2
from .causal_prophecy import EmpiricalCausalProphecy
from .causal_representation import RepresentedReturnAgent
from .paper_v2_types import ImaginationDecisionRecord, RawCausalObservation


_DISTANCE_CACHE: dict[tuple[Any, ...], int | None] = {}


def _private_cache_key(world: CausalDependencyWorldV2) -> tuple[Any, ...]:
    state = world.analysis_private_state
    return (
        world.action_token_sha256,
        state.completed,
        state.inventory,
        state.location,
        state.health,
        state.latent_risk,
        state.terminal,
        state.success,
        state.dead_end,
    )


@dataclass(frozen=True, slots=True)
class ModelEstimate:
    expected_discounted_return: float
    next_observation: RawCausalObservation | None
    uncertainty: float
    ood_score: float
    calibration_confidence: float
    transition_exact: bool = False
    terminal_dead_end: bool = False


class ReturnModel(Protocol):
    name: str

    def estimate(
        self,
        observation: RawCausalObservation,
        action: str,
        *,
        world: CausalDependencyWorldV2 | None = None,
    ) -> ModelEstimate: ...


class LearnedReturnModel:
    name = "learned_prophecy"

    def __init__(self, prophecy: EmpiricalCausalProphecy) -> None:
        self.prophecy = prophecy

    def estimate(
        self,
        observation: RawCausalObservation,
        action: str,
        *,
        world: CausalDependencyWorldV2 | None = None,
    ) -> ModelEstimate:
        del world
        prediction = self.prophecy.predict_v21(observation, action)
        return ModelEstimate(
            prediction.base.terminal_return_probability,
            prediction.base.next_observable_state,
            prediction.uncertainty,
            prediction.ood_score,
            prediction.calibration_confidence,
        )


class RandomReturnModel:
    name = "random_prophecy"

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    def estimate(
        self,
        observation: RawCausalObservation,
        action: str,
        *,
        world: CausalDependencyWorldV2 | None = None,
    ) -> ModelEstimate:
        del world
        digest = hashlib.sha256(
            f"{self.seed}:{action}:{observation.available_actions}".encode()
        ).digest()
        value = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        return ModelEstimate(value, observation, 1.0, 1.0, 0.0)


def _shortest_success_distance(
    world: CausalDependencyWorldV2, *, maximum_depth: int = 12
) -> int | None:
    cache_key = (*_private_cache_key(world), maximum_depth)
    if cache_key in _DISTANCE_CACHE:
        return _DISTANCE_CACHE[cache_key]
    if world.analysis_private_state.success:
        return 0
    queue = deque([(world.clone(), 0)])
    seen = set()
    while queue:
        current, depth = queue.popleft()
        state = current.analysis_private_state
        key = (
            state.completed,
            state.inventory,
            state.location,
            state.health,
            state.terminal,
        )
        if key in seen:
            continue
        seen.add(key)
        if state.success:
            _DISTANCE_CACHE[cache_key] = depth
            return depth
        if state.terminal or depth >= maximum_depth:
            continue
        for action in current.observe().available_actions:
            child = current.clone()
            child.step(action)
            queue.append((child, depth + 1))
    _DISTANCE_CACHE[cache_key] = None
    return None


def exact_root_action_values(
    world: CausalDependencyWorldV2, *, gamma: float = 0.97
) -> dict[str, float]:
    values = {}
    for action in world.observe().available_actions:
        child = world.clone()
        child.step(action)
        distance = _shortest_success_distance(child)
        values[action] = 0.0 if distance is None else gamma ** distance
    return values


class OracleReturnModel:
    name = "oracle_transition_upper_bound"

    def __init__(self, *, gamma: float = 0.97) -> None:
        self.gamma = float(gamma)

    def estimate(
        self,
        observation: RawCausalObservation,
        action: str,
        *,
        world: CausalDependencyWorldV2 | None = None,
    ) -> ModelEstimate:
        if world is None:
            raise ValueError("oracle model requires private world handle")
        if observation != world.observe():
            raise ValueError("oracle world and observation are out of sync")
        child = world.clone()
        outcome = child.step(action)
        distance = _shortest_success_distance(child)
        value = 0.0 if distance is None else self.gamma ** distance
        return ModelEstimate(
            value,
            outcome.observation,
            0.0,
            0.0,
            1.0,
            transition_exact=True,
            terminal_dead_end=child.analysis_private_state.dead_end,
        )


@dataclass(frozen=True, slots=True)
class ImaginationGateConfig:
    calibration_confidence_minimum: float = 0.8
    uncertainty_maximum: float = 0.25
    ood_maximum: float = 0.25
    minimum_return_margin: float = 0.05
    maximum_depth: int = 4
    branching_factor: int = 4
    gamma: float = 0.97


class CausalImaginationPlanner:
    def __init__(
        self,
        model: ReturnModel,
        *,
        config: ImaginationGateConfig | None = None,
        gated: bool = True,
    ) -> None:
        self.model = model
        self.config = config or ImaginationGateConfig()
        self.gated = bool(gated)

    def _rollout(
        self,
        observation: RawCausalObservation,
        action: str,
        policy: RepresentedReturnAgent,
        *,
        world: CausalDependencyWorldV2 | None,
        depth: int,
    ) -> tuple[float, int, int, ModelEstimate]:
        estimate = self.model.estimate(observation, action, world=world)
        nodes = 1
        maximum_depth = depth
        if (
            depth >= self.config.maximum_depth
            or estimate.next_observation is None
            or estimate.next_observation.terminal
            or not estimate.next_observation.available_actions
        ):
            return estimate.expected_discounted_return, nodes, maximum_depth, estimate
        next_world = None
        if world is not None and isinstance(self.model, OracleReturnModel):
            next_world = world.clone()
            next_world.step(action)
        ranked = sorted(
            estimate.next_observation.available_actions,
            key=lambda item: (-policy.q_value(estimate.next_observation, item), item),
        )[: self.config.branching_factor]
        children = []
        for child_action in ranked:
            value, child_nodes, child_depth, _ = self._rollout(
                estimate.next_observation,
                child_action,
                policy,
                world=next_world,
                depth=depth + 1,
            )
            children.append(value)
            nodes += child_nodes
            maximum_depth = max(maximum_depth, child_depth)
        rollout = max(children, default=0.0)
        value = max(
            estimate.expected_discounted_return,
            self.config.gamma * rollout,
        )
        return value, nodes, maximum_depth, estimate

    def decide(
        self,
        observation: RawCausalObservation,
        policy: RepresentedReturnAgent,
        *,
        world: CausalDependencyWorldV2 | None = None,
    ) -> ImaginationDecisionRecord:
        actions = observation.available_actions
        policy_q = {action: policy.q_value(observation, action) for action in actions}
        policy_action = min(actions, key=lambda item: (-policy_q[item], item))
        model_q: dict[str, float] = {}
        estimates: dict[str, ModelEstimate] = {}
        nodes = 0
        depth = 0
        for action in actions:
            value, action_nodes, action_depth, estimate = self._rollout(
                observation, action, policy, world=world, depth=1
            )
            model_q[action] = max(0.0, min(1.0, value))
            estimates[action] = estimate
            nodes += action_nodes
            depth = max(depth, action_depth)
        best = min(actions, key=lambda item: (-model_q[item], item))
        estimate = estimates[best]
        advantage = model_q[best] - policy_q[policy_action]
        predicates = {
            "calibration": estimate.calibration_confidence
            >= self.config.calibration_confidence_minimum,
            "uncertainty": estimate.uncertainty <= self.config.uncertainty_maximum,
            "ood": estimate.ood_score <= self.config.ood_maximum,
            "margin": advantage >= self.config.minimum_return_margin,
        }
        allowed = all(predicates.values())
        intervened = best != policy_action and (allowed or not self.gated)
        final = best if intervened else policy_action
        if intervened:
            reason = "ungated" if not self.gated else "all_gate_predicates_passed"
        elif best == policy_action:
            reason = "model_agrees_with_policy"
        else:
            reason = "blocked:" + ",".join(
                key for key, passed in predicates.items() if not passed
            )
        return ImaginationDecisionRecord(
            policy_only_action=policy_action,
            final_selected_action=final,
            root_policy_q=policy_q,
            root_model_q=model_q,
            uncertainty=estimate.uncertainty,
            ood_score=estimate.ood_score,
            calibration_confidence=estimate.calibration_confidence,
            imagined_advantage=advantage,
            intervened=intervened,
            intervention_reason=reason,
            imagined_nodes=nodes,
            maximum_depth_reached=depth,
        )
