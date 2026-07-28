import unittest
from types import SimpleNamespace

from aassr.gpt2_experiment import (
    CreativityGuardrailConfig,
    _guardrail_summary,
    _structural_signature,
)


class GPT2ExperimentTests(unittest.TestCase):
    def test_structural_signature_collapses_consecutive_same_template(self) -> None:
        rows = [
            SimpleNamespace(template="INSPECT {A}"),
            SimpleNamespace(template="INSPECT {A}"),
            SimpleNamespace(template="MOVE {B}"),
        ]
        self.assertEqual(
            _structural_signature(rows),
            ("INSPECT {A}", "MOVE {B}"),
        )

    def test_guardrail_requires_success_and_creativity_preservation(self) -> None:
        result = _guardrail_summary(
            evaluation_summary=SimpleNamespace(success_rate=0.7),
            baseline_summary=SimpleNamespace(success_rate=0.6),
            evaluation_creativity={
                "successful_trajectory_count": 5.0,
                "successful_trajectory_diversity": 0.6,
                "trajectory_entropy": 0.5,
                "novel_strategy_rate": 0.4,
            },
            baseline_creativity={
                "successful_trajectory_diversity": 0.65,
                "trajectory_entropy": 0.55,
            },
            config=CreativityGuardrailConfig(),
        )
        self.assertTrue(result["passed"])

    def test_guardrail_rejects_strategy_collapse(self) -> None:
        result = _guardrail_summary(
            evaluation_summary=SimpleNamespace(success_rate=0.8),
            baseline_summary=SimpleNamespace(success_rate=0.6),
            evaluation_creativity={
                "successful_trajectory_count": 5.0,
                "successful_trajectory_diversity": 0.1,
                "trajectory_entropy": 0.1,
                "novel_strategy_rate": 0.0,
            },
            baseline_creativity={
                "successful_trajectory_diversity": 0.7,
                "trajectory_entropy": 0.6,
            },
            config=CreativityGuardrailConfig(),
        )
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
