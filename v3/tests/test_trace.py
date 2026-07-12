from __future__ import annotations

import unittest

from apassr_tool.trace import build_trace_report


class TraceTests(unittest.TestCase):
    def test_report_extracts_solved_window(self) -> None:
        data = {
            "episodes": [
                {
                    "episode": 0,
                    "success": True,
                    "records": [
                        {"step": 0, "template": "A", "status": 200, "new_kv": 1, "solved_delta": 0, "reward": 1},
                        {
                            "step": 1,
                            "template": "B",
                            "status": 200,
                            "new_kv": 2,
                            "solved_delta": 1,
                            "solved_total": 1,
                            "reward": 20,
                            "action": "SOLVE",
                        },
                    ],
                }
            ],
            "novelty": {"signature_count": 2, "chain_count": 2, "response_count": 2},
        }

        report = build_trace_report(data, window=1)

        self.assertIn("Solved Windows", report)
        self.assertIn("SOLVE", report)
        self.assertIn("solved events: 1", report)

    def test_report_falls_back_to_high_novelty_when_unsolved(self) -> None:
        data = {
            "episodes": [
                {
                    "episode": 0,
                    "success": False,
                    "records": [
                        {
                            "step": 0,
                            "template": "WEIRD",
                            "status": 500,
                            "new_kv": 0,
                            "solved_delta": 0,
                            "novelty_score": 0.9,
                            "novelty_bonus": 2.0,
                            "reward": 1.5,
                            "action": "WEIRD ACTION",
                        }
                    ],
                }
            ]
        }

        report = build_trace_report(data)

        self.assertIn("No Solved Event Yet", report)
        self.assertIn("High-Novelty Steps", report)
        self.assertIn("WEIRD ACTION", report)


if __name__ == "__main__":
    unittest.main()
