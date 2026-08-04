from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def patch_autonomous_experiment() -> None:
    path = ROOT / "src/aassr_v2/autonomous_experiment.py"
    text = path.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "from collections import deque\n",
        "from collections import Counter, deque\n",
        label="Counter import",
    )
    text = replace_once(
        text,
        '''        imagination_minimum_coverage=float(\n            condition.get("imagination_minimum_coverage", 0.75)\n        ),\n''',
        '''        imagination_minimum_coverage=float(\n            condition.get("imagination_minimum_coverage", 0.35)\n        ),\n''',
        label="coverage default",
    )
    text = replace_once(
        text,
        '''        extrinsic_reward_weight=float(\n            condition.get("extrinsic_reward_weight", 1.0)\n        ),\n    )\n''',
        '''        extrinsic_reward_weight=float(\n            condition.get("extrinsic_reward_weight", 1.0)\n        ),\n        use_effect_composition=bool(\n            condition.get("use_effect_composition", True)\n        ),\n        effect_minimum_samples=int(\n            condition.get("effect_minimum_samples", 2)\n        ),\n    )\n''',
        label="effect config plumbing",
    )
    text = replace_once(
        text,
        '''    root_values: list[float] = []\n    errors = 0\n''',
        '''    root_values: list[float] = []\n    imagination_opportunities = 0\n    imagination_eligible = 0\n    imagination_runs = 0\n    imagination_changed_actions = 0\n    imagination_coverages: list[float] = []\n    imagination_gate_reasons: Counter[str] = Counter()\n    errors = 0\n''',
        label="episode diagnostic counters",
    )
    text = replace_once(
        text,
        '''        imagined_nodes += decision.imagined_nodes\n        imagination_depth = max(\n''',
        '''        opportunity = bool(\n            getattr(decision, "imagination_opportunity", False)\n        )\n        eligible = bool(\n            getattr(decision, "imagination_eligible", False)\n        )\n        changed_action = bool(\n            getattr(decision, "imagination_changed_action", False)\n        )\n        gate_reason = str(\n            getattr(decision, "imagination_gate_reason", "not_supported")\n        )\n        coverage = float(getattr(decision, "model_coverage", 0.0))\n        imagination_opportunities += int(opportunity)\n        imagination_eligible += int(eligible)\n        imagination_runs += int(decision.used_imagination)\n        imagination_changed_actions += int(changed_action)\n        imagination_gate_reasons[gate_reason] += 1\n        if opportunity:\n            imagination_coverages.append(coverage)\n        imagined_nodes += decision.imagined_nodes\n        imagination_depth = max(\n''',
        label="per-decision diagnostic capture",
    )
    text = replace_once(
        text,
        '''                "learning_enabled": learn,\n                "imagined_nodes": decision.imagined_nodes,\n            }\n''',
        '''                "learning_enabled": learn,\n                "imagined_nodes": decision.imagined_nodes,\n                "policy_action_signature": getattr(\n                    decision,\n                    "policy_action_signature",\n                    decision.action.signature,\n                ),\n                "imagination_opportunity": opportunity,\n                "imagination_eligible": eligible,\n                "imagination_gate_reason": gate_reason,\n                "imagination_changed_action": changed_action,\n                "model_coverage": coverage,\n            }\n''',
        label="transition diagnostic fields",
    )
    text = replace_once(
        text,
        '''        "imagined_nodes": imagined_nodes,\n        "imagination_depth": imagination_depth,\n''',
        '''        "imagined_nodes": imagined_nodes,\n        "imagination_opportunities": imagination_opportunities,\n        "imagination_eligible": imagination_eligible,\n        "imagination_runs": imagination_runs,\n        "imagination_changed_actions": imagination_changed_actions,\n        "imagination_change_rate": (\n            imagination_changed_actions / imagination_runs\n            if imagination_runs\n            else 0.0\n        ),\n        "imagination_eligibility_rate": (\n            imagination_eligible / imagination_opportunities\n            if imagination_opportunities\n            else 0.0\n        ),\n        "imagination_coverage_mean": (\n            fmean(imagination_coverages)\n            if imagination_coverages\n            else 0.0\n        ),\n        "imagination_gate_reasons": json.dumps(\n            dict(sorted(imagination_gate_reasons.items())),\n            sort_keys=True,\n        ),\n        "imagination_depth": imagination_depth,\n''',
        label="episode diagnostic output",
    )
    path.write_text(text, encoding="utf-8")


def patch_experiment_runner() -> None:
    path = ROOT / "src/aassr_v2/experiment_runner.py"
    text = path.read_text(encoding="utf-8")
    fields = '''    "imagined_nodes",\n    "imagination_depth",\n'''
    expanded_fields = '''    "imagined_nodes",\n    "imagination_opportunities",\n    "imagination_eligible",\n    "imagination_runs",\n    "imagination_changed_actions",\n    "imagination_change_rate",\n    "imagination_eligibility_rate",\n    "imagination_coverage_mean",\n    "imagination_gate_reasons",\n    "imagination_depth",\n'''
    # RESULT_FIELDS and SUMMARY_METRICS each contain this pair.  The first
    # replacement includes the JSON gate field; the second remains numeric.
    if text.count(fields) != 2:
        raise RuntimeError(
            f"result/summary field anchors: expected 2, found {text.count(fields)}"
        )
    text = text.replace(fields, expanded_fields, 1)
    numeric_fields = expanded_fields.replace(
        '    "imagination_gate_reasons",\n',
        "",
    )
    text = text.replace(fields, numeric_fields, 1)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    patch_autonomous_experiment()
    patch_experiment_runner()


if __name__ == "__main__":
    main()
