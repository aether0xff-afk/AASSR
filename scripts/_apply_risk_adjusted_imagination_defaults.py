from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"{label}: expected one match, found {text.count(old)}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


core = ROOT / "src/aassr_v2/autonomous_agent_core.py"
replace_once(
    core,
    '''    imagination_branching_factor: int = 2\n    imagination_beam_width: int = 32\n    imagination_minimum_coverage: float = 0.35\n''',
    '''    imagination_branching_factor: int = 2\n    imagination_beam_width: int = 32\n    imagination_outcome_samples: int = 2\n    imagination_minimum_coverage: float = 0.35\n''',
    "outcome sample config",
)
replace_once(
    core,
    '    imagination_aggregation: str = "max"\n',
    '    imagination_aggregation: str = "risk-adjusted"\n',
    "risk adjusted default",
)
replace_once(
    core,
    '''        if self.imagination_depth <= 0:\n            raise ValueError("imagination_depth must be positive")\n''',
    '''        if self.imagination_depth <= 0:\n            raise ValueError("imagination_depth must be positive")\n        if self.imagination_outcome_samples <= 0:\n            raise ValueError("imagination_outcome_samples must be positive")\n''',
    "outcome sample validation",
)
replace_once(
    core,
    '''                outcome_samples=1,\n                minimum_path_confidence=0.1,\n''',
    '''                outcome_samples=self.config.imagination_outcome_samples,\n                minimum_path_confidence=0.1,\n''',
    "planner outcome samples",
)

experiment = ROOT / "src/aassr_v2/autonomous_experiment.py"
replace_once(
    experiment,
    '''        imagination_beam_width=int(\n            condition.get("imagination_beam_width", 32)\n        ),\n        imagination_minimum_coverage=float(\n''',
    '''        imagination_beam_width=int(\n            condition.get("imagination_beam_width", 32)\n        ),\n        imagination_outcome_samples=int(\n            condition.get("imagination_outcome_samples", 2)\n        ),\n        imagination_minimum_coverage=float(\n''',
    "experiment outcome samples",
)
replace_once(
    experiment,
    '''        imagination_aggregation=str(\n            condition.get("imagination_aggregation", "max")\n        ),\n''',
    '''        imagination_aggregation=str(\n            condition.get("imagination_aggregation", "risk-adjusted")\n        ),\n''',
    "experiment aggregation default",
)
