from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import fmean, pstdev
from typing import Any, Literal, Protocol

from .policy import PolicyMemory, WeightedPolicy
from .prophecy import ProphecyStep
from .types import Action, Prediction, StateSnapshot

Aggregation = Literal["max", "mean", "top_mean", "risk_adjusted"]


class ImaginationScorer(Protocol):
    def score(
        self,
        before: StateSnapshot,
        action: Action,
        after: StateSnapshot,
    ) -> float: ...


@dataclass(frozen=True, slots=True)
class StateDeltaScorer:
    """Default model-agnostic imagined transition value.

    Domains with an explicit reward model can replace this scorer without
    changing the tree search.
    """

    goal_progress_weight: float = 10.0
    new_fact_weight: float = 0.1
    unlocked_action_weight: float = 0.1
    step_cost: float = 0.01

    def score(
        self,
        before: StateSnapshot,
        action: Action,
        after: StateSnapshot,
    ) -> float:
        del action
        before_actions = {item.signature for item in before.available_actions}
        after_actions = {item.signature for item in after.available_actions}
        return (
            self.goal_progress_weight
            * (after.goal_progress - before.goal_progress)
            + self.new_fact_weight * len(after.facts - before.facts)
            + self.unlocked_action_weight * len(after_actions - before_actions)
            - self.step_cost
        )


@dataclass(frozen=True, slots=True)
class ImaginationConfig:
    branching_factor: int = 2
    maximum_depth: int = 2
    beam_width: int = 8
    outcome_samples: int = 1
    discount: float = 0.95
    minimum_path_confidence: float = 0.15
    uncertainty_penalty: float = 1.0
    goal_threshold: float = 1.0
    aggregation: Aggregation = "max"
    top_mean_count: int = 2
    update_policy: bool = True
    expand_all_root_actions: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.branching_factor,
            self.maximum_depth,
            self.beam_width,
            self.outcome_samples,
            self.top_mean_count,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("tree sizes and sample counts must be positive")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if not 0.0 <= self.minimum_path_confidence <= 1.0:
            raise ValueError("minimum_path_confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ImaginationNode:
    node_id: int
    parent_id: int | None
    depth: int
    state: StateSnapshot
    root_action: Action | None
    action_from_parent: Action | None
    state_path: tuple[str, ...]
    action_path: tuple[str, ...]
    cumulative_value: float
    step_confidence: float
    cumulative_confidence: float
    policy_memory: PolicyMemory
    prophecy_memory: Any = None
    terminal_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RootActionEvaluation:
    action: Action
    leaf_values: tuple[float, ...]
    aggregate_value: float
    best_path: tuple[str, ...]
    best_leaf_id: int


@dataclass(frozen=True, slots=True)
class ImaginationResult:
    chosen_action: Action
    root_evaluations: tuple[RootActionEvaluation, ...]
    nodes: tuple[ImaginationNode, ...]
    expanded_nodes: int
    maximum_depth_reached: int


class ImaginationTree:
    """Branching parallel-universe rollout driven entirely by Prophecy."""

    def __init__(
        self,
        policy: WeightedPolicy,
        prophecy: object,
        *,
        config: ImaginationConfig | None = None,
        scorer: ImaginationScorer | None = None,
    ) -> None:
        self.policy = policy
        self.prophecy = prophecy
        self.config = config or ImaginationConfig()
        self.scorer = scorer or StateDeltaScorer()

    def _initial_prophecy_memory(self) -> Any:
        factory = getattr(self.prophecy, "initial_memory", None)
        return factory() if callable(factory) else None

    def _predict(self, node: ImaginationNode, action: Action) -> ProphecyStep:
        predict_step = getattr(self.prophecy, "predict_step", None)
        if callable(predict_step):
            return predict_step(
                node.state,
                action,
                memory=node.prophecy_memory,
                samples=self.config.outcome_samples,
            )

        predictions = self.prophecy.predict(
            node.state,
            action,
            samples=self.config.outcome_samples,
        )
        return ProphecyStep(predictions, node.prophecy_memory)

    @staticmethod
    def _prediction_confidence(prediction: Prediction) -> float:
        confidence = prediction.probability
        source = prediction.source.lower()
        if source.endswith(":unseen"):
            return 0.0
        if source.endswith(":action-family"):
            confidence *= 0.5
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _state_key(state: StateSnapshot) -> str:
        return repr(
            (
                tuple(round(value, 8) for value in state.vector),
                tuple(sorted(state.facts)),
                tuple(action.signature for action in state.available_actions),
            )
        )

    def _adjusted_value(self, node: ImaginationNode) -> float:
        uncertainty = 1.0 - node.cumulative_confidence
        return node.cumulative_value - self.config.uncertainty_penalty * uncertainty

    def _aggregate(self, values: tuple[float, ...]) -> float:
        if self.config.aggregation == "max":
            return max(values)
        if self.config.aggregation == "mean":
            return fmean(values)
        if self.config.aggregation == "top_mean":
            selected = sorted(values, reverse=True)[: self.config.top_mean_count]
            return fmean(selected)
        if self.config.aggregation == "risk_adjusted":
            spread = pstdev(values) if len(values) > 1 else 0.0
            return fmean(values) - spread
        raise ValueError(f"unknown aggregation: {self.config.aggregation}")

    def plan(
        self,
        state: StateSnapshot,
        *,
        maximum_depth: int | None = None,
    ) -> ImaginationResult:
        depth_limit = maximum_depth or self.config.maximum_depth
        if depth_limit <= 0:
            raise ValueError("maximum_depth must be positive")
        if not state.available_actions:
            raise ValueError("state has no available actions")

        root = ImaginationNode(
            node_id=0,
            parent_id=None,
            depth=0,
            state=state,
            root_action=None,
            action_from_parent=None,
            state_path=(self._state_key(state),),
            action_path=(),
            cumulative_value=0.0,
            step_confidence=1.0,
            cumulative_confidence=1.0,
            policy_memory=PolicyMemory.empty(),
            prophecy_memory=self._initial_prophecy_memory(),
        )

        nodes = [root]
        frontier = [root]
        terminal: list[ImaginationNode] = []
        next_id = 1
        expanded_nodes = 0

        for depth in range(1, depth_limit + 1):
            children: list[ImaginationNode] = []
            for node in frontier:
                rank_limit = self.config.branching_factor
                if depth == 1 and self.config.expand_all_root_actions:
                    rank_limit = len(node.state.available_actions)
                ranked = self.policy.rank(
                    node.state,
                    limit=rank_limit,
                    memory=node.policy_memory,
                )
                if not ranked:
                    terminal.append(replace(node, terminal_reason="no_actions"))
                    continue

                expanded_nodes += 1
                for scored_action in ranked:
                    step = self._predict(node, scored_action.action)
                    predictions = sorted(
                        step.predictions,
                        key=lambda item: item.probability,
                        reverse=True,
                    )[: self.config.outcome_samples]

                    for prediction in predictions:
                        immediate_value = self.scorer.score(
                            node.state,
                            scored_action.action,
                            prediction.next_state,
                        )
                        cumulative_value = node.cumulative_value + (
                            self.config.discount ** (depth - 1)
                        ) * immediate_value
                        step_confidence = self._prediction_confidence(prediction)
                        cumulative_confidence = (
                            node.cumulative_confidence * step_confidence
                        )
                        root_action = node.root_action or scored_action.action
                        state_path = node.state_path + (
                            self._state_key(prediction.next_state),
                        )
                        action_path = node.action_path + (
                            scored_action.action.signature,
                        )
                        repeated_state = state_path[-1] in state_path[:-1]

                        terminal_reason = None
                        if (
                            prediction.next_state.goal_progress
                            >= self.config.goal_threshold
                        ):
                            terminal_reason = "goal"
                        elif (
                            cumulative_confidence
                            < self.config.minimum_path_confidence
                        ):
                            terminal_reason = "low_confidence"
                        elif not prediction.next_state.available_actions:
                            terminal_reason = "no_actions"
                        elif repeated_state:
                            terminal_reason = "repeated_state"
                        elif depth >= depth_limit:
                            terminal_reason = "depth_limit"

                        branch_policy_memory = self.policy.imagine_update(
                            node.policy_memory,
                            scored_action.action,
                            immediate_value,
                        )
                        child = ImaginationNode(
                            node_id=next_id,
                            parent_id=node.node_id,
                            depth=depth,
                            state=prediction.next_state,
                            root_action=root_action,
                            action_from_parent=scored_action.action,
                            state_path=state_path,
                            action_path=action_path,
                            cumulative_value=cumulative_value,
                            step_confidence=step_confidence,
                            cumulative_confidence=cumulative_confidence,
                            policy_memory=branch_policy_memory,
                            prophecy_memory=step.memory,
                            terminal_reason=terminal_reason,
                        )
                        next_id += 1
                        nodes.append(child)
                        if terminal_reason:
                            terminal.append(child)
                        else:
                            children.append(child)

            if not children:
                frontier = []
                break

            children.sort(
                key=lambda child: (
                    -self._adjusted_value(child),
                    child.root_action.signature if child.root_action else "",
                    child.node_id,
                )
            )
            frontier = children[: self.config.beam_width]

        if frontier:
            terminal.extend(
                replace(node, terminal_reason="depth_limit") for node in frontier
            )
        if not terminal:
            terminal = [node for node in nodes if node.depth > 0]

        grouped: dict[str, list[ImaginationNode]] = {}
        root_actions: dict[str, Action] = {}
        for leaf in terminal:
            if leaf.root_action is None:
                continue
            signature = leaf.root_action.signature
            grouped.setdefault(signature, []).append(leaf)
            root_actions[signature] = leaf.root_action

        evaluations: list[RootActionEvaluation] = []
        for signature, leaves in grouped.items():
            adjusted_values = tuple(self._adjusted_value(leaf) for leaf in leaves)
            best_leaf = max(leaves, key=self._adjusted_value)
            evaluations.append(
                RootActionEvaluation(
                    action=root_actions[signature],
                    leaf_values=adjusted_values,
                    aggregate_value=self._aggregate(adjusted_values),
                    best_path=best_leaf.action_path,
                    best_leaf_id=best_leaf.node_id,
                )
            )

        if not evaluations:
            raise RuntimeError("imagination produced no root action evaluation")

        evaluations.sort(
            key=lambda item: (-item.aggregate_value, item.action.signature)
        )
        chosen = evaluations[0]

        if self.config.update_policy:
            baseline = fmean(item.aggregate_value for item in evaluations)
            self.policy.reinforce(
                chosen.action,
                chosen.aggregate_value - baseline,
            )

        return ImaginationResult(
            chosen_action=chosen.action,
            root_evaluations=tuple(evaluations),
            nodes=tuple(nodes),
            expanded_nodes=expanded_nodes,
            maximum_depth_reached=max(node.depth for node in nodes),
        )
