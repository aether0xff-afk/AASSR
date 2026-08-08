from __future__ import annotations

from dataclasses import replace
from statistics import fmean
from typing import Any, Sequence

from .empirical_confidence import empirical_confidence
from .imagination_tree import (
    ImaginationNode,
    ImaginationResult,
    ImaginationTree,
    RootActionEvaluation,
)
from .prophecy import ProphecyStep
from .skills import SKILL_VERB
from .torch_gru_prophecy import TorchGRUMemory, TorchGRUProphecy
from .types import Action, Prediction, StateSnapshot


class DepthBatchedProphecyView:
    """Batch primitive recurrent Prophecy calls while preserving symbolic layers.

    The dense GRU forward pass for all primitive branches at one imagination
    depth is executed as one tensor operation. Skill macro-actions remain on the
    canonical scalar path because they have variable primitive lengths. Effect
    composition and Skill state augmentation are still applied row-by-row in the
    exact canonical order after the neural batch returns.
    """

    def __init__(self, integrated_view: object) -> None:
        self.view = integrated_view
        self._batch_calls = 0
        self._batch_rows = 0
        self._scalar_fallback_rows = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.view, name)

    @property
    def name(self) -> str:
        return f"depth-batched:{getattr(self.view, 'name', 'prophecy')}"

    def _compose_effect_row(
        self,
        state: StateSnapshot,
        action: Action,
        base_step: ProphecyStep,
        *,
        samples: int,
    ) -> ProphecyStep:
        outer = self.view._prophecy
        ensure = getattr(outer, "_ensure_reconstructed", None)
        if callable(ensure):
            ensure()
        bucket, tier, source, include_symbolic = outer._select_bucket(state, action)
        if not bucket:
            return base_step

        ranked = sorted(
            bucket.values(),
            key=lambda entry: (-entry.count, repr(entry.effect.fingerprint)),
        )[:samples]
        effect_mass = min(
            0.95,
            empirical_confidence(
                (entry.count for entry in bucket.values()),
                prior_strength=1.0,
                tier=tier,
            ),
        )
        base_mass = 1.0 - effect_mass
        scaffold = max(
            base_step.predictions,
            key=lambda prediction: prediction.probability,
        ).next_state
        merged: dict[tuple[Any, ...], list[Any]] = {}

        def add_prediction(prediction: Prediction, probability: float) -> None:
            if probability <= 0.0:
                return
            key = outer._prediction_key(prediction.next_state)
            current = merged.get(key)
            if current is None:
                merged[key] = [prediction.next_state, probability, prediction.source]
            else:
                previous = float(current[1])
                current[1] = previous + probability
                if probability > previous:
                    current[2] = prediction.source

        ranked_total = sum(entry.count for entry in ranked)
        for entry in ranked:
            probability = effect_mass * entry.count / max(1, ranked_total)
            composed = entry.effect.apply(
                state,
                symbolic_scaffold=scaffold,
                include_symbolic_delta=include_symbolic,
                source=source,
            )
            add_prediction(Prediction(composed, probability, source), probability)

        normalized = outer._normalized_probabilities(tuple(base_step.predictions))
        for prediction, probability in zip(
            base_step.predictions, normalized, strict=True
        ):
            add_prediction(prediction, base_mass * probability)

        total = sum(float(item[1]) for item in merged.values())
        predictions = tuple(
            Prediction(
                next_state=item[0],
                probability=float(item[1]) / total,
                source=str(item[2]),
            )
            for _, item in sorted(
                merged.items(),
                key=lambda pair: (-float(pair[1][1]), repr(pair[0])),
            )
        )
        outer._composed_predictions += 1
        return ProphecyStep(predictions, base_step.memory)

    @staticmethod
    def _torch_primitive_steps(
        model: TorchGRUProphecy,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
        memories: Sequence[Any],
        *,
        samples: int,
    ) -> tuple[ProphecyStep, ...]:
        if not states:
            return ()
        torch = model.torch
        with torch.inference_mode():
            x = model._inputs(states, actions)
            hidden_rows = []
            for memory in memories:
                resolved = memory if memory is not None else model.initial_memory()
                hidden_rows.append(resolved.hidden)
            hidden = torch.stack(hidden_rows, dim=0)
            outputs, next_hidden = model._forward_batch(x, hidden)
            prediction_rows = model._decode_outputs(
                states,
                actions,
                outputs,
                samples=samples,
            )
            return tuple(
                ProphecyStep(
                    predictions,
                    TorchGRUMemory(next_hidden[index].detach().clone()),
                )
                for index, predictions in enumerate(prediction_rows)
            )

    def predict_step_batch(
        self,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
        memories: Sequence[Any],
        *,
        samples: int,
    ) -> tuple[ProphecyStep, ...]:
        if not (len(states) == len(actions) == len(memories)):
            raise ValueError("states/actions/memories batch length mismatch")
        if not states:
            return ()

        skill_wrapper = self.view._contextual_skill_prophecy
        base = getattr(skill_wrapper, "base", None)
        if not isinstance(base, TorchGRUProphecy):
            self._scalar_fallback_rows += len(states)
            return tuple(
                self.view._prophecy.predict_step(
                    state,
                    action,
                    memory=memory,
                    samples=samples,
                )
                for state, action, memory in zip(
                    states, actions, memories, strict=True
                )
            )

        results: list[ProphecyStep | None] = [None] * len(states)
        primitive_indices = [
            index
            for index, action in enumerate(actions)
            if action.verb_name != SKILL_VERB
        ]
        if primitive_indices:
            primitive_states = tuple(states[index] for index in primitive_indices)
            primitive_actions = tuple(actions[index] for index in primitive_indices)
            primitive_memories = tuple(memories[index] for index in primitive_indices)
            base_steps = self._torch_primitive_steps(
                base,
                primitive_states,
                primitive_actions,
                primitive_memories,
                samples=samples,
            )
            self._batch_calls += 1
            self._batch_rows += len(primitive_indices)
            for index, state, action, base_step in zip(
                primitive_indices,
                primitive_states,
                primitive_actions,
                base_steps,
                strict=True,
            ):
                skill_predictions = tuple(
                    replace(
                        prediction,
                        next_state=skill_wrapper.library.augment_state(
                            prediction.next_state
                        ),
                    )
                    for prediction in base_step.predictions
                )
                results[index] = self._compose_effect_row(
                    state,
                    action,
                    ProphecyStep(skill_predictions, base_step.memory),
                    samples=samples,
                )

        for index, (state, action, memory) in enumerate(
            zip(states, actions, memories, strict=True)
        ):
            if results[index] is not None:
                continue
            self._scalar_fallback_rows += 1
            results[index] = self.view._prophecy.predict_step(
                state,
                action,
                memory=memory,
                samples=samples,
            )

        return tuple(item for item in results if item is not None)

    def runtime_diagnostics(self) -> dict[str, int]:
        return {
            "imagination_batch_calls": self._batch_calls,
            "imagination_batch_rows": self._batch_rows,
            "imagination_scalar_fallback_rows": self._scalar_fallback_rows,
        }


class DepthBatchedImaginationTree(ImaginationTree):
    """ImaginationTree with one Prophecy batch per search depth when possible."""

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
            policy_memory=self.policy.empty_memory() if hasattr(self.policy, "empty_memory") else __import__("aassr_v2.policy", fromlist=["PolicyMemory"]).PolicyMemory.empty(),
            prophecy_memory=self._initial_prophecy_memory(),
            scorer_memory=self._initial_scorer_memory(),
        )

        nodes = [root]
        frontier = [root]
        terminal: list[ImaginationNode] = []
        next_id = 1
        expanded_nodes = 0

        for depth in range(1, depth_limit + 1):
            work: list[tuple[ImaginationNode, Any]] = []
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

            children: list[ImaginationNode] = []
            for (node, scored_action), step in zip(work, steps, strict=True):
                predictions = sorted(
                    step.predictions,
                    key=lambda item: item.probability,
                    reverse=True,
                )[: self.config.outcome_samples]
                for prediction in predictions:
                    step_confidence = self._prediction_confidence(prediction)
                    immediate_value, scorer_memory, value_mode = self._score(
                        node,
                        scored_action.action,
                        prediction.next_state,
                        prophecy_confidence=step_confidence,
                    )
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
                        prophecy_memory=step.memory,
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


def enable_depth_batched_imagination(agent: object) -> object:
    """Install depth batching on an already locally accelerated AASSR agent."""

    wrapped = DepthBatchedProphecyView(agent.prophecy)
    old = agent.core.planner
    planner = DepthBatchedImaginationTree(
        old.policy,
        wrapped,
        config=old.config,
        scorer=old.scorer,
    )
    planner._state_key = old._state_key
    agent.prophecy = wrapped
    agent.core.planner = planner
    return agent
