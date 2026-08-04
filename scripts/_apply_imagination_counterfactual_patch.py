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
        '''    imagination_coverages: list[float] = []\n    imagination_gate_reasons: Counter[str] = Counter()\n    errors = 0\n''',
        '''    imagination_coverages: list[float] = []\n    imagination_gate_reasons: Counter[str] = Counter()\n    policy_oracle_agreements = 0\n    executed_oracle_agreements = 0\n    imagination_corrections = 0\n    imagination_harms = 0\n    imagination_neutral_changes = 0\n    errors = 0\n''',
        label="oracle audit counters",
    )
    text = replace_once(
        text,
        '''        coverage = float(getattr(decision, "model_coverage", 0.0))\n        imagination_opportunities += int(opportunity)\n''',
        '''        coverage = float(getattr(decision, "model_coverage", 0.0))\n        policy_action_signature = str(\n            getattr(\n                decision,\n                "policy_action_signature",\n                decision.action.signature,\n            )\n        )\n        oracle_action_signature = environment.oracle_action().signature\n        policy_oracle_agreement = (\n            policy_action_signature == oracle_action_signature\n        )\n        executed_oracle_agreement = (\n            decision.action.signature == oracle_action_signature\n        )\n        imagination_oracle_delta = (\n            int(executed_oracle_agreement)\n            - int(policy_oracle_agreement)\n        )\n        policy_oracle_agreements += int(policy_oracle_agreement)\n        executed_oracle_agreements += int(executed_oracle_agreement)\n        if changed_action:\n            if imagination_oracle_delta > 0:\n                imagination_corrections += 1\n            elif imagination_oracle_delta < 0:\n                imagination_harms += 1\n            else:\n                imagination_neutral_changes += 1\n        imagination_opportunities += int(opportunity)\n''',
        label="oracle audit decision comparison",
    )
    text = replace_once(
        text,
        '''                "policy_action_signature": getattr(\n                    decision,\n                    "policy_action_signature",\n                    decision.action.signature,\n                ),\n''',
        '''                "policy_action_signature": policy_action_signature,\n''',
        label="reuse normalized policy signature",
    )
    text = replace_once(
        text,
        '''                "imagination_changed_action": changed_action,\n                "model_coverage": coverage,\n            }\n''',
        '''                "imagination_changed_action": changed_action,\n                "model_coverage": coverage,\n                "privileged_analysis": {\n                    "oracle_action_signature": oracle_action_signature,\n                    "policy_oracle_agreement": policy_oracle_agreement,\n                    "executed_oracle_agreement": executed_oracle_agreement,\n                    "imagination_oracle_delta": imagination_oracle_delta,\n                },\n            }\n''',
        label="privileged transition analysis",
    )
    text = replace_once(
        text,
        '''        "imagination_gate_reasons": json.dumps(\n            dict(sorted(imagination_gate_reasons.items())),\n            sort_keys=True,\n        ),\n        "imagination_depth": imagination_depth,\n''',
        '''        "imagination_gate_reasons": json.dumps(\n            dict(sorted(imagination_gate_reasons.items())),\n            sort_keys=True,\n        ),\n        "policy_oracle_agreements": policy_oracle_agreements,\n        "executed_oracle_agreements": executed_oracle_agreements,\n        "policy_oracle_agreement_rate": (\n            policy_oracle_agreements / steps if steps else 0.0\n        ),\n        "executed_oracle_agreement_rate": (\n            executed_oracle_agreements / steps if steps else 0.0\n        ),\n        "imagination_corrections": imagination_corrections,\n        "imagination_harms": imagination_harms,\n        "imagination_neutral_changes": imagination_neutral_changes,\n        "imagination_net_corrections": (\n            imagination_corrections - imagination_harms\n        ),\n        "imagination_oracle_gain": (\n            executed_oracle_agreements - policy_oracle_agreements\n        ),\n        "imagination_correction_rate": (\n            imagination_corrections / imagination_changed_actions\n            if imagination_changed_actions\n            else 0.0\n        ),\n        "imagination_harm_rate": (\n            imagination_harms / imagination_changed_actions\n            if imagination_changed_actions\n            else 0.0\n        ),\n        "imagination_depth": imagination_depth,\n''',
        label="episode oracle audit output",
    )
    path.write_text(text, encoding="utf-8")


def patch_experiment_runner() -> None:
    path = ROOT / "src/aassr_v2/experiment_runner.py"
    text = path.read_text(encoding="utf-8")
    anchor = '''    "imagination_gate_reasons",\n    "imagination_depth",\n'''
    expanded_result = '''    "imagination_gate_reasons",\n    "policy_oracle_agreements",\n    "executed_oracle_agreements",\n    "policy_oracle_agreement_rate",\n    "executed_oracle_agreement_rate",\n    "imagination_corrections",\n    "imagination_harms",\n    "imagination_neutral_changes",\n    "imagination_net_corrections",\n    "imagination_oracle_gain",\n    "imagination_correction_rate",\n    "imagination_harm_rate",\n    "imagination_depth",\n'''
    text = replace_once(
        text,
        anchor,
        expanded_result,
        label="result oracle fields",
    )
    summary_anchor = '''    "imagination_coverage_mean",\n    "imagination_depth",\n'''
    expanded_summary = '''    "imagination_coverage_mean",\n    "policy_oracle_agreements",\n    "executed_oracle_agreements",\n    "policy_oracle_agreement_rate",\n    "executed_oracle_agreement_rate",\n    "imagination_corrections",\n    "imagination_harms",\n    "imagination_neutral_changes",\n    "imagination_net_corrections",\n    "imagination_oracle_gain",\n    "imagination_correction_rate",\n    "imagination_harm_rate",\n    "imagination_depth",\n'''
    text = replace_once(
        text,
        summary_anchor,
        expanded_summary,
        label="summary oracle fields",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    patch_autonomous_experiment()
    patch_experiment_runner()


if __name__ == "__main__":
    main()
