from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
replacements = {
    ROOT / "src/aassr_v2/autonomous_agent_core.py": (
        "    imagination_uncertainty_margin: float = 0.20\n",
        "    imagination_uncertainty_margin: float = 0.40\n",
    ),
    ROOT / "src/aassr_v2/autonomous_experiment.py": (
        '            condition.get("imagination_uncertainty_margin", 0.20)\n',
        '            condition.get("imagination_uncertainty_margin", 0.40)\n',
    ),
}
for path, (old, new) in replacements.items():
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"{path}: expected one default uncertainty margin")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
