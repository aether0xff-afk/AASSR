from __future__ import annotations

from pathlib import Path


path = Path(__file__).resolve().parents[1] / "src/aassr_v2/autonomous_experiment.py"
text = path.read_text(encoding="utf-8")
old = '''        "terminal_reward_only": True,\n        "train_test_world_separation": any(\n'''
new = '''        "terminal_reward_only": True,\n        "privileged_oracle_analysis_only": True,\n        "oracle_labels_agent_visible": False,\n        "oracle_labels_used_for_learning": False,\n        "train_test_world_separation": any(\n'''
if text.count(old) != 1:
    raise RuntimeError(f"expected one manifest anchor, found {text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
