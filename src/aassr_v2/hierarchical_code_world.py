from __future__ import annotations

import random

from .goal_gridpush_experiment import GridPushStep
from .types import Action, StateSnapshot


class HierarchicalCodeWorld:
    """Twenty reusable four-choice checkpoints with one final reward.

    Every stage exposes a four-bit code. At every reality step the agent must
    choose bit 0 or bit 1. A wrong choice irreversibly ends the episode. Four
    correct choices open the next checkpoint. Only the twentieth checkpoint
    emits external reward.

    Stage identity is kept in facts for Policy and logs, while the numerical
    state contains only the reusable local code and current position. Prophecy
    can therefore reuse a learned local transition in a new stage order.
    """

    def __init__(
        self,
        seed: int,
        *,
        stage_count: int = 20,
        room_length: int = 4,
    ) -> None:
        if stage_count <= 1:
            raise ValueError("stage_count must exceed one")
        if room_length != 4:
            raise ValueError("HierarchicalCodeWorld uses four-choice codes")
        self.seed = int(seed)
        self.stage_count = int(stage_count)
        self.room_length = 4
        randomizer = random.Random(seed)
        self.codes = tuple(
            tuple(randomizer.randrange(2) for _ in range(self.room_length))
            for _ in range(self.stage_count)
        )
        self.stage = 0
        self.position = 0
        self.just_opened_checkpoint = False
        self.success = False
        self.failed = False
        self.optimal_steps = self.stage_count * self.room_length

    @property
    def code(self) -> tuple[int, ...]:
        return self.codes[min(self.stage, self.stage_count - 1)]

    @staticmethod
    def _action(bit: int) -> Action:
        return Action("enter_bit", parameters={"bit": int(bit)})

    def _available_actions(self) -> tuple[Action, ...]:
        if self.success or self.failed:
            return ()
        return (self._action(0), self._action(1))

    def _facts(self) -> frozenset[str]:
        facts = {f"stage:{self.stage}"}
        if self.just_opened_checkpoint:
            facts.add("checkpoint_transition")
        if self.success:
            facts.add("success")
        if self.failed:
            facts.add("failed")
        return frozenset(facts)

    def snapshot(self) -> StateSnapshot:
        # The full local code is observable. Stage identity is intentionally
        # excluded from the vector so Prophecy can reuse the same code effect
        # at any position in the 80-step dependency chain.
        vector = (
            self.position / float(self.room_length),
            *(1.0 if bit else -1.0 for bit in self.code),
        )
        return StateSnapshot(
            vector,
            self._facts(),
            self._available_actions(),
            1.0 if self.success else 0.0,
            metadata={
                "map_seed": self.seed,
                "stage": self.stage,
                "stage_count": self.stage_count,
                "code_length": self.room_length,
                "optimal_steps": self.optimal_steps,
                "termination": "wrong_bit_or_final_checkpoint",
                "prophecy_scope": "reusable_local_code",
            },
        )

    def step(self, action: Action) -> GridPushStep:
        before = self.snapshot()
        reward = 0.0
        error = False
        available = {item.signature for item in before.available_actions}
        if action.signature not in available:
            self.failed = True
            error = True
        else:
            bit = int(action.parameters.get("bit", -1))
            expected = self.code[self.position]
            self.just_opened_checkpoint = False
            if bit != expected:
                self.failed = True
                error = True
            else:
                self.position += 1
                if self.position >= self.room_length:
                    self.stage += 1
                    if self.stage >= self.stage_count:
                        self.success = True
                        reward = 1.0
                    else:
                        self.position = 0
                        self.just_opened_checkpoint = True

        after = self.snapshot()
        before_actions = {item.signature for item in before.available_actions}
        unlocked = tuple(
            item
            for item in after.available_actions
            if item.signature not in before_actions
        )
        return GridPushStep(
            after,
            after.facts - before.facts,
            before.facts - after.facts,
            unlocked,
            error,
            reward,
        )


def install_hierarchical_code_world() -> None:
    """Install the 80-step code task in the long-horizon experiment runner."""

    from . import long_horizon_goal_experiment

    long_horizon_goal_experiment.LongHorizonDependencyWorld = (
        HierarchicalCodeWorld
    )
