from __future__ import annotations

from dataclasses import replace
from math import sqrt
from statistics import fmean
from typing import Any, Sequence

from .current_generation import relational_action_key
from .imagination_tree import (
    ImaginationNode,
    ImaginationResult,
    RootActionEvaluation,
)
from .native_batching import DepthBatchedImaginationTree
from .policy import PolicyMemory
from .skills import SKILL_VERB
from .types import Action, Prediction, StateSnapshot


def _structural_action_key(state: StateSnapshot, action: Action) -> tuple[Any, ...]:
    if action.verb_name == SKILL_VERB:
        return ("skill", str(action.target))
    return ("primitive", *relational_action_key(state, action))


class CurrentFullyBatchedImaginationTree(DepthBatchedImaginationTree):
    """Current tree with explicit chance and decision nodes.

    Environment uncertainty and agent choice are different operations. Prophecy
    outcomes for one action are chance branches and are backed up with their
    stochastic ``outcome_probability``. Different actions available at the same
    imagined state are decision branches and are backed up with ``max`` because
    the future agent can choose among them.

    Predicted task terminals use the exact external sparse return instead of a
    learned Critic estimate: success +1, true lockout failure -1, truncation 0,
    discounted only by the number of imagined transitions from the current root.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.policy_batch_calls = 0
        self.policy_batch_rows = 0
        self.policy_scalar_fallback_rows = 0
        self.critic_batch_calls = 0
        self.critic_batch_rows = 0
        self.critic_scalar_fallback_rows = 0
        self.structural_alias_rows_removed = 0
        self.outcome_probability_rows = 0
        self.chance_backup_groups = 0
        self.decision_backup_nodes = 0
        self.task_success_leaves = 0
        self.task_truncation_leaves = 0
        self.task_failure_leaves = 0
        self.exact_terminal_value_leaves = 0
        self._last_outcome_probability_by_node: dict[int, float] = {}

    def _structural_limit(
        self,
        state: StateSnapshot,
        ranked: tuple[Any, ...],
        *,
        depth: int,
    ) -> tuple[Any, ...]:
        if depth == 1 and self.config.expand_all_root_actions:
            return ranked
        selected = []
        seen: set[tuple[Any, ...]] = set()
        duplicate_count = 0
        for scored in ranked:
            key = _structural_action_key(state, scored.action)
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            selected.append(scored)
            if len(selected) >= self.config.branching_factor:
                break
        self.structural_alias_rows_removed += duplicate_count
        return tuple(selected)

    def _rank_frontier(
        self,
        frontier: list[ImaginationNode],
        *,
        depth: int,
    ) -> tuple[tuple[Any, ...], ...]:
        request_limits = tuple(len(node.state.available_actions) for node in frontier)
        batch_rank = getattr(self.policy, "rank_batch", None)
        if callable(batch_rank):
            raw_rows = batch_rank(
                tuple(node.state for node in frontier),
                request_limits,
                tuple(node.policy_memory for node in frontier),
            )
            if len(raw_rows) != len(frontier):
                raise RuntimeError("current Policy batch returned wrong row count")
            self.policy_batch_calls += 1
            self.policy_batch_rows += len(frontier)
        else:
            self.policy_scalar_fallback_rows += len(frontier)
            raw_rows = tuple(
                self.policy.rank(
                    node.state,
                    limit=limit,
                    memory=node.policy_memory,
                )
                for node, limit in zip(frontier, request_limits, strict=True)
            )

        return tuple(
            self._structural_limit(
                node.state,
                tuple(row),
                depth=depth,
            )
            for node, row in zip(frontier, raw_rows, strict=True)
        )

    def _score_candidates(
        self,
        candidates: tuple[tuple[ImaginationNode, Any, StateSnapshot, float], ...],
    ) -> tuple[tuple[float, Any, str], ...]:
        if not candidates:
            return ()
        batch_score = getattr(self.scorer, "score_step_batch", None)
        if not callable(batch_score):
            self.critic_scalar_fallback_rows += len(candidates)
            return tuple(
                self._score(
                    node,
                    scored_action.action,
                    after,
                    prophecy_confidence=confidence,
                )
                for node, scored_action, after, confidence in candidates
            )

        results = batch_score(
            tuple(node.state for node, _, _, _ in candidates),
            tuple(scored_action.action for _, scored_action, _, _ in candidates),
            tuple(after for _, _, after, _ in candidates),
            tuple(node.scorer_memory for node, _, _, _ in candidates),
            tuple(confidence for _, _, _, confidence in candidates),
        )
        if len(results) != len(candidates):
            raise RuntimeError("current Critic batch returned wrong row count")
        self.critic_batch_calls += 1
        self.critic_batch_rows += len(candidates)

        rows = []
        for (node, _, _, _), result in zip(candidates, results, strict=True):
            mode = getattr(
                result,
                "value_mode",
                getattr(self.scorer, "value_mode", "absolute"),
            )
            if mode not in {"incremental", "absolute"}:
                raise ValueError(f"unknown scorer value mode: {mode}")
            rows.append(
                (
                    float(result.value),
                    getattr(result, "memory", node.scorer_memory),
                    mode,
                )
            )
        return tuple(rows)

    def _normalized_predictions(
        self,
        predictions: Sequence[Prediction],
        *,
        limit: int,
    ) -> tuple[tuple[Prediction, float], ...]:
        if limit <= 0:
            raise ValueError("outcome limit must be positive")
        selected = sorted(
            predictions,
            key=lambda item: (
                float(getattr(item, "outcome_probability", 1.0)),
                float(item.probability),
                item.source,
            ),
            reverse=True,
        )
        complete = bool(
            getattr(self.prophecy, "complete_outcome_distribution", False)
        )
        # A complete distribution must never be converted into a conditional
        # top-k distribution. The active status-mixture model explicitly marks
        # its batched view complete, so every emitted chance mass is retained.
        # Historical/incomplete predictors keep the configured limit.
        if not complete:
            selected = selected[:limit]
        if not selected:
            return ()
        raw = [
            max(0.0, float(getattr(item, "outcome_probability", 1.0)))
            for item in selected
        ]
        total = sum(raw)
        if complete:
            # Do not let the planner silently conceal a future regression where a
            # predictor advertises a complete distribution but drops tail mass.
            # Only tiny floating-point drift is corrected by normalization.
            if total <= 1e-12 or abs(total - 1.0) > 1e-6:
                raise RuntimeError(
                    "complete Prophecy outcome mass must sum to 1.0 before "
                    f"expected-return backup; observed {total:.12f}"
                )
            probabilities = [value / total for value in raw]
        elif total <= 1e-12:
            probabilities = [1.0 / len(selected)] * len(selected)
        else:
            probabilities = [value / total for value in raw]
        return tuple(zip(selected, probabilities, strict=True))

    def _aggregate_outcomes(
        self,
        values: Sequence[float],
        probabilities: Sequence[float],
    ) -> float:
        if len(values) != len(probabilities) or not values:
            raise ValueError("chance backup requires aligned non-empty rows")
        weights = [max(0.0, float(value)) for value in probabilities]
        total = sum(weights)
        if total <= 1e-12:
            weights = [1.0 / len(values)] * len(values)
        else:
            weights = [value / total for value in weights]
        materialized = [float(value) for value in values]

        if self.config.aggregation == "max":
            return max(materialized)
        if self.config.aggregation == "mean":
            return sum(
                weight * value
                for weight, value in zip(weights, materialized, strict=True)
            )
        if self.config.aggregation == "top_mean":
            ranked = sorted(
                zip(materialized, weights, strict=True),
                key=lambda item: item[0],
                reverse=True,
            )[: self.config.top_mean_count]
            selected_weight = sum(weight for _, weight in ranked)
            if selected_weight <= 1e-12:
                return fmean(value for value, _ in ranked)
            return sum(value * weight for value, weight in ranked) / selected_weight
        if self.config.aggregation == "risk_adjusted":
            mean = sum(
                weight * value
                for weight, value in zip(weights, materialized, strict=True)
            )
            variance = sum(
                weight * (value - mean) ** 2
                for weight, value in zip(weights, materialized, strict=True)
            )
            return mean - sqrt(max(0.0, variance))
        raise ValueError(f"unknown aggregation: {self.config.aggregation}")

    @staticmethod
    def _control(state: StateSnapshot, index: int) -> float:
        return float(state.vector[index]) if index < len(state.vector) else 0.0

    def _task_terminal_reason(self, state: StateSnapshot) -> str | None:
        facts = state.facts
        rate_limited = "rate_limited" in facts or self._control(state, 6) >= 0.5
        if rate_limited:
            return "truncation"
        failed = "failed" in facts or self._control(state, 4) >= 0.5
        locked = "locked" in facts or self._control(state, 5) >= 0.5
        if failed and locked:
            return "failure"
        return None

    def _exact_terminal_value(self, reason: str, depth: int) -> float:
        if reason == "goal":
            reward = 1.0
        elif reason == "failure":
            reward = -1.0
        elif reason == "truncation":
            reward = 0.0
        else:
            raise ValueError(f"not an external task terminal: {reason}")
        return (self.config.discount ** (depth - 1)) * reward

    def _bellman_backups(
        self,
        nodes: Sequence[ImaginationNode],
        outcome_probability: dict[int, float],
    ) -> tuple[
        dict[int, float],
        dict[int, int],
        dict[int, tuple[str, ...]],
        dict[tuple[int, str], list[ImaginationNode]],
    ]:
        by_parent_action: dict[tuple[int, str], list[ImaginationNode]] = {}
        actions_by_parent: dict[int, set[str]] = {}
        for child in nodes:
            if child.depth <= 0 or child.parent_id is None or child.action_from_parent is None:
                continue
            signature = child.action_from_parent.signature
            by_parent_action.setdefault((child.parent_id, signature), []).append(child)
            actions_by_parent.setdefault(child.parent_id, set()).add(signature)

        backed: dict[int, float] = {}
        best_leaf: dict[int, int] = {}
        best_path: dict[int, tuple[str, ...]] = {}
        for node in nodes:
            if node.depth <= 0:
                continue
            backed[node.node_id] = self._adjusted_value(node)
            best_leaf[node.node_id] = node.node_id
            best_path[node.node_id] = node.action_path

        for depth in range(max((node.depth for node in nodes), default=0), 0, -1):
            for node in nodes:
                if node.depth != depth:
                    continue
                signatures = actions_by_parent.get(node.node_id)
                if not signatures or node.terminal_reason is not None:
                    continue

                action_values: list[tuple[float, str, int, tuple[str, ...]]] = []
                for signature in sorted(signatures):
                    outcomes = by_parent_action[(node.node_id, signature)]
                    values = [backed[child.node_id] for child in outcomes]
                    masses = [outcome_probability.get(child.node_id, 1.0) for child in outcomes]
                    value = self._aggregate_outcomes(values, masses)
                    representative = max(
                        outcomes,
                        key=lambda child: (backed[child.node_id], -child.node_id),
                    )
                    action_values.append(
                        (
                            value,
                            signature,
                            best_leaf[representative.node_id],
                            best_path[representative.node_id],
                        )
                    )
                    self.chance_backup_groups += 1

                chosen = max(action_values, key=lambda item: (item[0], item[1]))
                backed[node.node_id] = chosen[0]
                best_leaf[node.node_id] = chosen[2]
                best_path[node.node_id] = chosen[3]
                self.decision_backup_nodes += 1

        return backed, best_leaf, best_path, by_parent_action

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
            policy_memory=(
                self.policy.empty_memory()
                if hasattr(self.policy, "empty_memory")
                else PolicyMemory.empty()
            ),
            prophecy_memory=self._initial_prophecy_memory(),
            scorer_memory=self._initial_scorer_memory(),
        )

        nodes = [root]
        frontier = [root]
        next_id = 1
        expanded_nodes = 0
        outcome_probability_by_node: dict[int, float] = {}

        for depth in range(1, depth_limit + 1):
            ranked_rows = self._rank_frontier(frontier, depth=depth)
            work: list[tuple[ImaginationNode, Any]] = []
            for node, ranked in zip(frontier, ranked_rows, strict=True):
                if not ranked:
                    continue
                expanded_nodes += 1
                work.extend((node, scored_action) for scored_action in ranked)

            if not work:
                frontier = []
                break

            batch_predict = getattr(self.prophecy, "predict_step_batch", None)
            if callable(batch_predict):
                steps = batch_predict(
                    tuple(node.state for node, _ in work),
                    tuple(scored.action for _, scored in work),
                    tuple(node.prophecy_memory for node, _ in work),
                    samples=self.config.outcome_samples,
                )
            else:
                steps = tuple(
                    self._predict(node, scored.action)
                    for node, scored in work
                )

            candidate_metadata: list[
                tuple[ImaginationNode, Any, Prediction, float, float, Any]
            ] = []
            for (node, scored_action), step in zip(work, steps, strict=True):
                normalized = self._normalized_predictions(
                    step.predictions,
                    limit=self.config.outcome_samples,
                )
                if not normalized:
                    raise RuntimeError(
                        "current Prophecy returned no outcome for an evaluated action"
                    )
                for prediction, outcome_mass in normalized:
                    candidate_metadata.append(
                        (
                            node,
                            scored_action,
                            prediction,
                            self._prediction_confidence(prediction),
                            outcome_mass,
                            step.memory,
                        )
                    )
                    self.outcome_probability_rows += 1

            score_input = tuple(
                (node, scored_action, prediction.next_state, confidence)
                for (
                    node,
                    scored_action,
                    prediction,
                    confidence,
                    _,
                    _,
                ) in candidate_metadata
            )
            scored_rows = self._score_candidates(score_input)

            children: list[ImaginationNode] = []
            for (
                node,
                scored_action,
                prediction,
                step_confidence,
                outcome_mass,
                prophecy_memory,
            ), (
                immediate_value,
                scorer_memory,
                value_mode,
            ) in zip(candidate_metadata, scored_rows, strict=True):
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
                if prediction.next_state.goal_progress >= self.config.goal_threshold:
                    terminal_reason = "goal"
                else:
                    terminal_reason = self._task_terminal_reason(prediction.next_state)

                value_center = float(getattr(self.scorer, "value_center", 0.5))
                if terminal_reason in {"goal", "failure", "truncation"}:
                    cumulative_value = self._exact_terminal_value(terminal_reason, depth)
                    policy_feedback = cumulative_value - value_center
                    self.exact_terminal_value_leaves += 1
                    if terminal_reason == "goal":
                        self.task_success_leaves += 1
                    elif terminal_reason == "failure":
                        self.task_failure_leaves += 1
                    else:
                        self.task_truncation_leaves += 1
                elif value_mode == "absolute":
                    cumulative_value = immediate_value
                    policy_feedback = immediate_value - value_center
                else:
                    cumulative_value = node.cumulative_value + (
                        self.config.discount ** (depth - 1)
                    ) * immediate_value
                    policy_feedback = immediate_value

                if terminal_reason is None:
                    if cumulative_confidence < self.config.minimum_path_confidence:
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
                    policy_feedback,
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
                    prophecy_memory=prophecy_memory,
                    terminal_reason=terminal_reason,
                    scorer_memory=scorer_memory,
                )
                outcome_probability_by_node[next_id] = float(outcome_mass)
                next_id += 1
                nodes.append(child)
                if terminal_reason is None:
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

        self._last_outcome_probability_by_node = dict(outcome_probability_by_node)
        backed, best_leaf, best_path, by_parent_action = self._bellman_backups(
            nodes,
            outcome_probability_by_node,
        )

        evaluations: list[RootActionEvaluation] = []
        for action in state.available_actions:
            outcomes = by_parent_action.get((0, action.signature), ())
            if not outcomes:
                raise RuntimeError(
                    "expand_all_root_actions lost a root before depth-1 evaluation: "
                    f"{action.signature}"
                )
            values = tuple(backed[child.node_id] for child in outcomes)
            masses = tuple(
                outcome_probability_by_node.get(child.node_id, 1.0)
                for child in outcomes
            )
            aggregate = self._aggregate_outcomes(values, masses)
            representative = max(
                outcomes,
                key=lambda child: (backed[child.node_id], -child.node_id),
            )
            evaluations.append(
                RootActionEvaluation(
                    action=action,
                    leaf_values=values,
                    aggregate_value=aggregate,
                    best_path=best_path[representative.node_id],
                    best_leaf_id=best_leaf[representative.node_id],
                )
            )
            self.chance_backup_groups += 1

        evaluations.sort(
            key=lambda item: (-item.aggregate_value, item.action.signature)
        )
        if len(evaluations) != len(state.available_actions):
            raise RuntimeError(
                "root evaluation coverage mismatch: "
                f"{len(evaluations)} != {len(state.available_actions)}"
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

    def runtime_diagnostics(self) -> dict[str, int]:
        return {
            "policy_batch_calls": self.policy_batch_calls,
            "policy_batch_rows": self.policy_batch_rows,
            "policy_scalar_fallback_rows": self.policy_scalar_fallback_rows,
            "critic_batch_calls": self.critic_batch_calls,
            "critic_batch_rows": self.critic_batch_rows,
            "critic_scalar_fallback_rows": self.critic_scalar_fallback_rows,
            "structural_alias_rows_removed": self.structural_alias_rows_removed,
            "outcome_probability_rows": self.outcome_probability_rows,
            "chance_backup_groups": self.chance_backup_groups,
            "decision_backup_nodes": self.decision_backup_nodes,
            "task_success_leaves": self.task_success_leaves,
            "task_truncation_leaves": self.task_truncation_leaves,
            "task_failure_leaves": self.task_failure_leaves,
            "exact_terminal_value_leaves": self.exact_terminal_value_leaves,
        }
