from __future__ import annotations

from pathlib import Path


path = Path(__file__).resolve().parents[1] / "src/aassr_v2/effect_prophecy.py"
text = path.read_text(encoding="utf-8")

replacements = (
    (
        "from dataclasses import dataclass\nfrom typing import Any, Iterable, Mapping\n\nfrom .prophecy import ProphecyStep\n",
        "from dataclasses import dataclass\nfrom math import sqrt\nfrom typing import Any, Iterable, Mapping\n\nfrom .empirical_confidence import empirical_confidence\nfrom .prophecy import ProphecyStep\n",
        "imports",
    ),
    (
        '''        observation_count = sum(entry.count for entry in bucket.values())\n        experience = observation_count / (observation_count + 1.0)\n        effect_mass = min(0.95, max(0.0, tier * experience))\n''',
        '''        effect_mass = min(\n            0.95,\n            empirical_confidence(\n                (entry.count for entry in bucket.values()),\n                prior_strength=1.0,\n                tier=tier,\n            ),\n        )\n''',
        "effect mass",
    ),
    (
        '''        observations = sum(entry.count for entry in bucket.values())\n        return min(1.0, tier * observations / (observations + 1.0))\n''',
        '''        return empirical_confidence(\n            (entry.count for entry in bucket.values()),\n            prior_strength=1.0,\n            tier=tier,\n        )\n''',
        "effect confidence",
    ),
    (
        '''        return max(base_confidence, self._effect_confidence(state, action))\n''',
        '''        bucket, _, _, _ = self._select_bucket(state, action)\n        if not bucket:\n            return base_confidence\n        effect_confidence = self._effect_confidence(state, action)\n        if base_confidence <= 0.0:\n            return effect_confidence\n        if effect_confidence <= 0.0:\n            return 0.0\n        return sqrt(base_confidence * effect_confidence)\n''',
        "combined confidence",
    ),
)

for old, new, label in replacements:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
