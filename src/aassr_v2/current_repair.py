from __future__ import annotations

from collections import Counter
from dataclasses import replace
from types import MethodType
from typing import Any

from .autonomous_agent_core import ActionDecision
from .current_agent import CurrentProphecyView, CurrentSkillProphecy
from .current_generation import relational_action_key, relational_state_key
from .current_manifest import CURRENT_COMPONENTS
from .current_relational_model import RelationalStochasticProphecy
from .current_semantic_calibration import (
    RelationalDepthBatchedProphecyView,
    SemanticCalibratedProphecy,
    SemanticPredictionValidator,
)
from .current_semantic_evaluator import RelationalAdvancedTransitionEvaluator
from .current_return_critic import ReturnAwareHardwareRelationalGRUBranchCritic
from .imagination_tree import ImaginationResult, RootActionEvaluation
from .skills import SKILL_VERB
from .types import Action, StateSnapshot


def _structural_action_key(state: StateSnapshot, action: Action) -> tuple[Any, ...]:
    if action.verb_name == SKILL_VERB:
        return ("skill", str(action.target))
    return ("primitive", *relational_action_key(state, action))


def preserve_root_evaluations(
    planner: object,
    state: StateSnapshot,
    result: ImaginationResult,
) -> ImaginationResult:
    """Guarantee every observable root keeps at least its depth-1 evaluation."""
    by_signature = {
        item.action.signature: item for item in result.root_evaluations
    }
    depth_one: dict[str, list[Any]] = {}
    for node in result.nodes:
        if node.depth == 1 and node.root_action is not None:
            depth_one.setdefault(node.root_action.signature, []).append(node)

    for action in state.available_actions:
        if action.signature in by_signature:
            continue
        leaves = depth_one.get(action.signature, ())
        if not leaves:
            raise RuntimeError(
                "expand_all_root_actions lost a root before depth-1 evaluation: "
                f"{action.signature}"
            )
        values = tuple(planner._adjusted_value(node) for node in leaves)
        best = max(leaves, key=planner._adjusted_value)
        by_signature[action.signature] = RootActionEvaluation(
            action=action,
            leaf_values=values,
            aggregate_value=planner._aggregate(values),
            best_path=best.action_path,
            best_leaf_id=best.node_id,
        )

    evaluations = tuple(
        sorted(
            by_signature.values(),
            key=lambda item: (-item.aggregate_value, item.action.signature),
        )
    )
    if len(evaluations) != len(state.available_actions):
        raise RuntimeError(
            "root-preserving planner contract mismatch: "
            f"{len(evaluations)} != {len(state.available_actions)}"
        )
    return replace(
        result,
        chosen_action=evaluations[0].action,
        root_evaluations=evaluations,
    )


def _install_planner_audit(agent: object) -> None:
    original_plan = agent.planner.plan
    counters: Counter[str] = Counter()

    def audited_plan(
        planner_self: object,
        state: StateSnapshot,
        *,
        maximum_depth: int | None = None,
    ) -> ImaginationResult:
        raw = original_plan(state, maximum_depth=maximum_depth)
        result = preserve_root_evaluations(planner_self, state, raw)
        ordered = sorted(
            result.root_evaluations,
            key=lambda item: (-float(item.aggregate_value), item.action.signature),
        )
        values = [float(item.aggregate_value) for item in ordered]
        gap = values[0] - values[1] if len(values) > 1 else float("inf")
        top = (
            tuple(
                item
                for item in ordered
                if abs(float(item.aggregate_value) - values[0]) <= 1e-12
            )
            if values
            else ()
        )
        structural_groups = {
            _structural_action_key(state, item.action) for item in top
        }
        ties = len(top)
        group_count = len(structural_groups)

        agent._repair_last_root_top_gap = float(gap)
        agent._repair_last_root_top_ties = int(ties)
        agent._repair_last_root_top_structural_groups = int(group_count)
        counters["plans"] += 1
        counters["expected_roots"] += len(state.available_actions)
        counters["evaluated_roots"] += len(result.root_evaluations)
        counters["unique_structural_roots"] += len(
            {_structural_action_key(state, item.action) for item in result.root_evaluations}
        )
        counters["exact_top_ties"] += int(ties > 1)
        counters["exact_top_ties_equivalent_aliases"] += int(
            ties > 1 and group_count == 1
        )
        counters["exact_top_ties_distinct_structures"] += int(
            ties > 1 and group_count > 1
        )
        counters["near_top_ties"] += int(ties == 1 and gap < 0.01)
        return result

    agent.planner.plan = MethodType(audited_plan, agent.planner)
    agent._repair_planner_diagnostics = counters

    original_record = agent._record_decision

    def audited_record(
        self_agent: object,
        decision: ActionDecision,
    ) -> ActionDecision:
        if decision.imagination_gate_reason == "policy_agreement":
            ties = int(getattr(self_agent, "_repair_last_root_top_ties", 0))
            groups = int(
                getattr(self_agent, "_repair_last_root_top_structural_groups", 0)
            )
            gap = float(
                getattr(self_agent, "_repair_last_root_top_gap", float("inf"))
            )
            if ties > 1 and groups == 1:
                reason = "critic_exact_tie_equivalent_aliases"
            elif ties > 1:
                reason = "critic_exact_tie_distinct_structures_policy_tiebreak"
            elif gap < 0.01:
                reason = "critic_near_tie_policy_best"
            else:
                reason = "critic_strict_policy_best"
            decision = replace(decision, imagination_gate_reason=reason)
        return original_record(decision)

    agent._record_decision = MethodType(audited_record, agent)


def install_current_repairs(
    agent: object,
    *,
    seed: int,
    device: str,
) -> object:
    """Install all representation/world-model/Critic/planner contract repairs."""
    if getattr(agent, "current_relational_repairs", False):
        return agent

    old_evaluator = agent.evaluator
    replay = old_evaluator.replay
    base = RelationalStochasticProphecy(
        seed=int(seed) ^ 0x52454C41,
        device=device,
    )
    calibrated = SemanticCalibratedProphecy(base, replay)
    skill = CurrentSkillProphecy(
        calibrated,
        agent.skills,
        agent.knowledge,
    )
    prophecy = CurrentProphecyView(skill)
    validator = SemanticPredictionValidator(samples=3)
    evaluator = RelationalAdvancedTransitionEvaluator(
        prophecy,
        replay=replay,
        validator=validator,
        predictor=old_evaluator.predictor,
        logger=old_evaluator.logger,
        samples=3,
        intrinsic_cap=float(old_evaluator.intrinsic_cap),
    )

    agent.base_neural_prophecy = base
    agent.calibrated_prophecy = calibrated
    agent.knowledge_prophecy = calibrated
    agent.skill_prophecy = skill
    agent.prophecy = prophecy
    agent.evaluator = evaluator
    agent.current_fast_validation = False
    agent.current_semantic_validation = True
    agent.current_semantic_evaluator = True

    batched = RelationalDepthBatchedProphecyView(agent)
    agent.current_batched_prophecy = batched
    agent.planner.prophecy = batched
    # Imagination now lives in canonical relational states. Repeated-state
    # detection must therefore use the same rename-invariant state identity rather
    # than the concrete ASEQ fingerprint used by the real-world loop guard.
    agent.planner._state_key = lambda state: repr(relational_state_key(state))
    agent.core.prophecy = prophecy

    critic = ReturnAwareHardwareRelationalGRUBranchCritic(
        int(seed) ^ 0x43524954,
        device=device,
    )
    agent.critic = critic
    agent.planner.scorer = critic

    agent.config = replace(agent.config, imagination_outcome_samples=3)
    agent.planner.config = replace(agent.planner.config, outcome_samples=3)

    original_finish = agent.finish_episode

    def finish_with_real_return(
        self_agent: object,
        *,
        final_return: float,
        training: bool = True,
    ) -> Any:
        if training:
            self_agent.critic.set_episode_return(
                float(final_return),
                float(self_agent.config.gamma),
            )
        return original_finish(
            final_return=float(final_return),
            training=training,
        )

    agent.finish_episode = MethodType(finish_with_real_return, agent)
    _install_planner_audit(agent)

    agent.current_relational_repairs = True
    agent.current_repairs_version = "relational-world-model-repair-v1"
    agent.current_components = dict(CURRENT_COMPONENTS)

    original_diagnostics = agent.diagnostics

    def repaired_diagnostics(self_agent: object) -> dict[str, Any]:
        output = dict(original_diagnostics())
        output["current_components"] = dict(self_agent.current_components)
        output["current_repairs"] = {
            "version": self_agent.current_repairs_version,
            "relational_input_output_contract": True,
            "relational_imagination_state_key": True,
            "semantic_decode": True,
            "predicted_legal_action_surface": True,
            "multi_outcome_ensemble": True,
            "reliability_outcome_probability_separated": True,
            "semantic_information_evaluator": True,
            "relational_repeat_unlock_value": True,
            "critic_sparse_return_aligned": True,
            "critic_root_scale_values": True,
            "root_preserving": True,
            "deeper_structural_branching": True,
            **dict(getattr(self_agent, "_repair_planner_diagnostics", {})),
        }
        return output

    agent.diagnostics = MethodType(repaired_diagnostics, agent)
    return agent
