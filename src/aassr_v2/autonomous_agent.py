"""Public entry point for the unified autonomous AASSR core.

All official runners import this module.  The implementation stays in
:mod:`aassr_v2.autonomous_agent_core`, while this entry point installs the
checkpoint-persistent transition-effect Prophecy wrapper.
"""

from .autonomous_agent_core import *  # noqa: F401,F403
from .autonomous_agent_core import (
    AutonomousLearningAgent as _CoreAutonomousLearningAgent,
)
from .effect_prophecy import EffectComposedProphecy
from .persistent_effect_prophecy import PersistentEffectComposedProphecy


class AutonomousLearningAgent(_CoreAutonomousLearningAgent):
    """Unified agent with persistent compositional transition effects."""

    def __init__(
        self,
        prophecy: object,
        *,
        config: AutonomousAgentConfig | None = None,
        seed: int = 0,
        policy: ContextualPolicy | None = None,
    ) -> None:
        resolved = config or AutonomousAgentConfig()
        if resolved.use_effect_composition and not isinstance(
            prophecy,
            PersistentEffectComposedProphecy,
        ):
            base = (
                prophecy.base
                if isinstance(prophecy, EffectComposedProphecy)
                else prophecy
            )
            prophecy = PersistentEffectComposedProphecy(
                base,
                minimum_samples=resolved.effect_minimum_samples,
            )
        super().__init__(
            prophecy,
            config=resolved,
            seed=seed,
            policy=policy,
        )
