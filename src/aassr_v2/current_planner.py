from __future__ import annotations

from dataclasses import replace
from statistics import fmean
from typing import Any

from .current_generation import relational_action_key
from .imagination_tree import (
    ImaginationNode,
    ImaginationResult,
    RootActionEvaluation,
)
from .native_batching import DepthBatchedImaginationTree
from .policy import PolicyMemory
from .skills import SKILL_VERB
from .types import Action, StateSnapshot


def _structural_action_key(state: StateSnapshot, action: Action) -> tuple[Any, ...]:
    if action.verb_name == SKILL_VERB:
        return ("skill", str(action.target))
    return ("primitive", *relational_action_key(state, action))


class CurrentFullyBatchedImaginationTree(DepthBatchedImaginationTree):
    """Current tree with structural branching and depth-batched learned models."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.policy_batch_calls = 0
        self.policy_batch_rows = 0
        self.policy_scalar_fallback_rows = 0
        self.critic_batch_calls = 0
        self.critic_batch_rows = 0
        self.critic_scalar_fallback_rows = 0
        self.structural_alias_rows_removed = 0

    def _structural_limit(
        self,
        state: StateSnapshot,
        ranked: tuple[Any, ...],
        *,
        depth: int,
    ) -> tuple[Any, ...]:
        # Root actions remain concrete because the eventual intervention must name
        # one real executable action. Deeper planning is in relational space: six
        # seed-renamed aliases must not consume all six branching slots.
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
        # Ask Policy for the complete ordered surface before structural
        # deduplication. Requesting only top-K concrete actions first can return K
        # aliases of one relational action and hide the next-best distinct action.
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
        terminal: list[ImaginationNode] = []
        next_id = 1
        expanded_nodes = 0

        for depth in range(1, depth_limit + 1):
            ranked_rows = self._rank_frontier(frontier, depth=depth)
            work: list[tuple[ImaginationNode, Any]] = []
            for node, ranked in zip(frontier, ranked_rows, strict=True):
                if not ranked:
                    terminal.append(replace(node, terminal_reason="no_actions"))
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
                tuple[ImaginationNode, Any, Any, float, Any]
            ] = []
            for (node, scored_action), step in zip(work, steps, strict=True):
                # ``probability`` is historical model reliability. Explicit
                # stochastic outcome mass is separate, and is used only to retain
                # the most common modes if there are more modes than sample slots.
                predictions = sorted(
                    step.predictions,
                    key=lambda item: (
                        float(getattr(item, "outcome_probability", 1.0)),
                        float(item.probability),
                    ),
                    reverse=True,
                )[: self.config.outcome_samples]
                for prediction in predictions:
                    candidate_metadata.append(
                        (
                            node,
                            scored_action,
                            prediction,
                            self._prediction_confidence(prediction),
                            step.memory,
                        )
                    )

            score_input = tuple(
                (node, scored_action, prediction.next_state, confidence)
                for node, scored_action, prediction, confidence, _ in candidate_metadata
            )
            scored_rows = self._score_candidates(score_input)

            children: list[ImaginationNode] = []
            for (
                node,
                scored_action,
                prediction,
                step_confidence,
                prophecy_memory,
            ), (
                immediate_value,
                scorer_memory,
                value_mode,
            ) in zip(candidate_metadata, scored_rows, strict=True):
                if value_mode == "absolute":
                    cumulative_value = immediate_value
                    policy_feedback = immediate_value - 0.5
                else:
                    cumulative_value = node.cumulative_value + (
                        self.config.discount ** (depth - 1)
                    ) * immediate_value
                    policy_feedback = immediate_value
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
                elif cumulative_confidence < self.config.minimum_path_confidence:
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
        root_actions = {}
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

    def runtime_diagnostics(self) -> dict[str, int]:
        return {
            "policy_batch_calls": self.policy_batch_calls,
            "policy_batch_rows": self.policy_batch_rows,
            "policy_scalar_fallback_rows": self.policy_scalar_fallback_rows,
            "critic_batch_calls": self.critic_batch_calls,
            "critic_batch_rows": self.critic_batch_rows,
            "critic_scalar_fallback_rows": self.critic_scalar_fallback_rows,
            "structural_alias_rows_removed": self.structural_alias_rows_removed,
        }
