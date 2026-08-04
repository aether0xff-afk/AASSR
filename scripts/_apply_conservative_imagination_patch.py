from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def patch_agent_core() -> None:
    path = ROOT / "src/aassr_v2/autonomous_agent_core.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''    imagination_minimum_coverage: float = 0.35\n    validated_gain_weight: float = 0.2\n''',
        '''    imagination_minimum_coverage: float = 0.35\n    imagination_intervention_margin: float = 0.05\n    imagination_uncertainty_margin: float = 0.20\n    validated_gain_weight: float = 0.2\n''',
        label="intervention config fields",
    )
    text = replace_once(
        text,
        '''        if not 0.0 <= self.imagination_minimum_coverage <= 1.0:\n            raise ValueError(\n                "imagination_minimum_coverage must be in [0, 1]"\n            )\n''',
        '''        if not 0.0 <= self.imagination_minimum_coverage <= 1.0:\n            raise ValueError(\n                "imagination_minimum_coverage must be in [0, 1]"\n            )\n        if self.imagination_intervention_margin < 0.0:\n            raise ValueError(\n                "imagination_intervention_margin must be non-negative"\n            )\n        if self.imagination_uncertainty_margin < 0.0:\n            raise ValueError(\n                "imagination_uncertainty_margin must be non-negative"\n            )\n''',
        label="intervention config validation",
    )
    text = replace_once(
        text,
        '''    imagination_changed_action: bool = False\n    model_coverage: float = 0.0\n''',
        '''    imagination_changed_action: bool = False\n    model_coverage: float = 0.0\n    imagination_preferred_action_signature: str = ""\n    imagination_policy_value: float = 0.0\n    imagination_preferred_value: float = 0.0\n    imagination_advantage: float = 0.0\n    imagination_required_advantage: float = 0.0\n    imagination_switch_candidate: bool = False\n    imagination_intervention_allowed: bool = False\n''',
        label="decision intervention diagnostics",
    )
    text = replace_once(
        text,
        '''        if decision.imagination_changed_action:\n            self._imagination_diagnostics["changed_actions"] += 1\n''',
        '''        if decision.imagination_switch_candidate:\n            self._imagination_diagnostics["switch_candidates"] += 1\n        if decision.imagination_intervention_allowed:\n            self._imagination_diagnostics["interventions"] += 1\n        if (\n            decision.imagination_switch_candidate\n            and not decision.imagination_intervention_allowed\n        ):\n            self._imagination_diagnostics["suppressed_switches"] += 1\n        if decision.imagination_changed_action:\n            self._imagination_diagnostics["changed_actions"] += 1\n''',
        label="intervention counters",
    )
    text = replace_once(
        text,
        '''            "change_rate_per_run": changed / runs if runs else 0.0,\n            "eligibility_rate": (\n''',
        '''            "change_rate_per_run": changed / runs if runs else 0.0,\n            "intervention_rate_per_candidate": (\n                self._imagination_diagnostics["interventions"]\n                / self._imagination_diagnostics["switch_candidates"]\n                if self._imagination_diagnostics["switch_candidates"]\n                else 0.0\n            ),\n            "eligibility_rate": (\n''',
        label="intervention diagnostic rate",
    )
    old_block = '''        if eligible:\n            plan = self.planner.plan(state)\n            best_imagined = max(\n                item.aggregate_value for item in plan.root_evaluations\n            )\n            candidates = [\n                item\n                for item in plan.root_evaluations\n                if abs(item.aggregate_value - best_imagined) <= 1e-12\n            ]\n            selected = min(\n                candidates,\n                key=lambda item: (\n                    -self.policy.value(state, item.action),\n                    item.action.signature,\n                ),\n            )\n            return self._record_decision(\n                ActionDecision(\n                    selected.action,\n                    True,\n                    imagined_nodes=len(plan.nodes),\n                    imagination_depth=plan.maximum_depth_reached,\n                    root_imagined_value=selected.aggregate_value,\n                    policy_action_signature=policy_action.signature,\n                    imagination_opportunity=True,\n                    imagination_eligible=True,\n                    imagination_gate_reason="eligible",\n                    imagination_changed_action=(\n                        selected.action.signature\n                        != policy_action.signature\n                    ),\n                    model_coverage=coverage,\n                )\n            )\n'''
    new_block = '''        if eligible:\n            plan = self.planner.plan(state)\n            best_imagined = max(\n                item.aggregate_value for item in plan.root_evaluations\n            )\n            candidates = [\n                item\n                for item in plan.root_evaluations\n                if abs(item.aggregate_value - best_imagined) <= 1e-12\n            ]\n            preferred = min(\n                candidates,\n                key=lambda item: (\n                    -self.policy.value(state, item.action),\n                    item.action.signature,\n                ),\n            )\n            policy_evaluation = next(\n                (\n                    item\n                    for item in plan.root_evaluations\n                    if item.action.signature == policy_action.signature\n                ),\n                None,\n            )\n            switch_candidate = (\n                preferred.action.signature != policy_action.signature\n            )\n            policy_value = (\n                policy_evaluation.aggregate_value\n                if policy_evaluation is not None\n                else preferred.aggregate_value\n            )\n            advantage = (\n                preferred.aggregate_value - policy_value\n                if policy_evaluation is not None\n                else 0.0\n            )\n            required_advantage = (\n                self.config.imagination_intervention_margin\n                + self.config.imagination_uncertainty_margin\n                * (1.0 - coverage)\n            )\n            intervention_allowed = (\n                switch_candidate\n                and policy_evaluation is not None\n                and advantage >= required_advantage\n            )\n            if not switch_candidate:\n                intervention_reason = "policy_agreement"\n            elif policy_evaluation is None:\n                intervention_reason = "policy_not_evaluated"\n            elif intervention_allowed:\n                intervention_reason = "intervention"\n            else:\n                intervention_reason = "insufficient_advantage"\n            executed_action = (\n                preferred.action if intervention_allowed else policy_action\n            )\n            executed_value = (\n                preferred.aggregate_value\n                if intervention_allowed\n                else policy_value\n            )\n            return self._record_decision(\n                ActionDecision(\n                    executed_action,\n                    True,\n                    imagined_nodes=len(plan.nodes),\n                    imagination_depth=plan.maximum_depth_reached,\n                    root_imagined_value=executed_value,\n                    policy_action_signature=policy_action.signature,\n                    imagination_opportunity=True,\n                    imagination_eligible=True,\n                    imagination_gate_reason=intervention_reason,\n                    imagination_changed_action=intervention_allowed,\n                    model_coverage=coverage,\n                    imagination_preferred_action_signature=(\n                        preferred.action.signature\n                    ),\n                    imagination_policy_value=policy_value,\n                    imagination_preferred_value=(\n                        preferred.aggregate_value\n                    ),\n                    imagination_advantage=advantage,\n                    imagination_required_advantage=required_advantage,\n                    imagination_switch_candidate=switch_candidate,\n                    imagination_intervention_allowed=intervention_allowed,\n                )\n            )\n'''
    text = replace_once(
        text,
        old_block,
        new_block,
        label="conservative intervention block",
    )
    path.write_text(text, encoding="utf-8")


def patch_autonomous_experiment() -> None:
    path = ROOT / "src/aassr_v2/autonomous_experiment.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''        imagination_minimum_coverage=float(\n            condition.get("imagination_minimum_coverage", 0.35)\n        ),\n        validated_gain_weight=float(\n''',
        '''        imagination_minimum_coverage=float(\n            condition.get("imagination_minimum_coverage", 0.35)\n        ),\n        imagination_intervention_margin=float(\n            condition.get("imagination_intervention_margin", 0.05)\n        ),\n        imagination_uncertainty_margin=float(\n            condition.get("imagination_uncertainty_margin", 0.20)\n        ),\n        validated_gain_weight=float(\n''',
        label="experiment intervention config",
    )
    text = replace_once(
        text,
        '''    imagination_coverages: list[float] = []\n    imagination_gate_reasons: Counter[str] = Counter()\n''',
        '''    imagination_coverages: list[float] = []\n    imagination_advantages: list[float] = []\n    imagination_required_advantages: list[float] = []\n    imagination_gate_reasons: Counter[str] = Counter()\n    imagination_switch_candidates = 0\n    imagination_interventions = 0\n    imagination_suppressed_switches = 0\n''',
        label="episode intervention counters",
    )
    text = replace_once(
        text,
        '''        coverage = float(getattr(decision, "model_coverage", 0.0))\n        policy_action_signature = str(\n''',
        '''        coverage = float(getattr(decision, "model_coverage", 0.0))\n        switch_candidate = bool(\n            getattr(decision, "imagination_switch_candidate", False)\n        )\n        intervention_allowed = bool(\n            getattr(decision, "imagination_intervention_allowed", False)\n        )\n        advantage = float(\n            getattr(decision, "imagination_advantage", 0.0)\n        )\n        required_advantage = float(\n            getattr(decision, "imagination_required_advantage", 0.0)\n        )\n        imagination_switch_candidates += int(switch_candidate)\n        imagination_interventions += int(intervention_allowed)\n        imagination_suppressed_switches += int(\n            switch_candidate and not intervention_allowed\n        )\n        if decision.used_imagination:\n            imagination_advantages.append(advantage)\n            imagination_required_advantages.append(required_advantage)\n        policy_action_signature = str(\n''',
        label="decision intervention capture",
    )
    text = replace_once(
        text,
        '''                "model_coverage": coverage,\n                "privileged_analysis": {\n''',
        '''                "model_coverage": coverage,\n                "imagination_preferred_action_signature": getattr(\n                    decision,\n                    "imagination_preferred_action_signature",\n                    decision.action.signature,\n                ),\n                "imagination_policy_value": float(\n                    getattr(decision, "imagination_policy_value", 0.0)\n                ),\n                "imagination_preferred_value": float(\n                    getattr(decision, "imagination_preferred_value", 0.0)\n                ),\n                "imagination_advantage": advantage,\n                "imagination_required_advantage": required_advantage,\n                "imagination_switch_candidate": switch_candidate,\n                "imagination_intervention_allowed": intervention_allowed,\n                "privileged_analysis": {\n''',
        label="transition intervention fields",
    )
    text = replace_once(
        text,
        '''        "imagination_gate_reasons": json.dumps(\n            dict(sorted(imagination_gate_reasons.items())),\n            sort_keys=True,\n        ),\n        "policy_oracle_agreements": policy_oracle_agreements,\n''',
        '''        "imagination_gate_reasons": json.dumps(\n            dict(sorted(imagination_gate_reasons.items())),\n            sort_keys=True,\n        ),\n        "imagination_switch_candidates": imagination_switch_candidates,\n        "imagination_interventions": imagination_interventions,\n        "imagination_suppressed_switches": (\n            imagination_suppressed_switches\n        ),\n        "imagination_intervention_rate": (\n            imagination_interventions / imagination_switch_candidates\n            if imagination_switch_candidates\n            else 0.0\n        ),\n        "imagination_advantage_mean": (\n            fmean(imagination_advantages)\n            if imagination_advantages\n            else 0.0\n        ),\n        "imagination_required_advantage_mean": (\n            fmean(imagination_required_advantages)\n            if imagination_required_advantages\n            else 0.0\n        ),\n        "policy_oracle_agreements": policy_oracle_agreements,\n''',
        label="episode intervention output",
    )
    path.write_text(text, encoding="utf-8")


def patch_experiment_runner() -> None:
    path = ROOT / "src/aassr_v2/experiment_runner.py"
    text = path.read_text(encoding="utf-8")
    result_anchor = '''    "imagination_gate_reasons",\n    "policy_oracle_agreements",\n'''
    result_fields = '''    "imagination_gate_reasons",\n    "imagination_switch_candidates",\n    "imagination_interventions",\n    "imagination_suppressed_switches",\n    "imagination_intervention_rate",\n    "imagination_advantage_mean",\n    "imagination_required_advantage_mean",\n    "policy_oracle_agreements",\n'''
    text = replace_once(
        text,
        result_anchor,
        result_fields,
        label="result intervention fields",
    )
    summary_anchor = '''    "imagination_coverage_mean",\n    "policy_oracle_agreements",\n'''
    summary_fields = '''    "imagination_coverage_mean",\n    "imagination_switch_candidates",\n    "imagination_interventions",\n    "imagination_suppressed_switches",\n    "imagination_intervention_rate",\n    "imagination_advantage_mean",\n    "imagination_required_advantage_mean",\n    "policy_oracle_agreements",\n'''
    text = replace_once(
        text,
        summary_anchor,
        summary_fields,
        label="summary intervention fields",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    patch_agent_core()
    patch_autonomous_experiment()
    patch_experiment_runner()


if __name__ == "__main__":
    main()
