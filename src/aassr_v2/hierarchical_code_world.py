from __future__ import annotations

import random

from .goal_gridpush_experiment import GridPushStep
from .types import Action, StateSnapshot


class HierarchicalCodeWorld:
    """Twenty reusable four-choice checkpoints with one final reward.

    Every stage exposes a four-bit code. At every reality step the agent enters
    bit 0 or bit 1, but the world does not reveal whether that choice was right
    until all four bits have been entered. A wrong four-bit sequence then ends
    the episode; a correct sequence opens the next checkpoint. Only the
    twentieth checkpoint emits external reward.

    Stage identity is kept in facts for Policy and logs. The numerical state
    contains the reusable local code, current position and entered prefix, so
    Prophecy can reuse a learned local transition in a new stage order.
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
        self.entered: list[int] = []
        self.just_opened_checkpoint = False
        self.success = False
        self.failed = False
        self.optimal_steps = self.stage_count * self.room_length

    @property
    def position(self) -> int:
        return len(self.entered)

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
        # -1 means that position has not been entered yet. Entered 0 and 1 are
        # retained explicitly, allowing Prophecy and Imagination to carry the
        # complete branch-local prefix without receiving any correctness label.
        entered_slots = tuple(
            float(self.entered[index]) if index < len(self.entered) else -1.0
            for index in range(self.room_length)
        )
        vector = (
            self.position / float(self.room_length),
            *(float(bit) for bit in self.code),
            *entered_slots,
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
                "termination": "wrong_code_or_final_checkpoint",
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
            if bit not in {0, 1}:
                self.failed = True
                error = True
            else:
                self.just_opened_checkpoint = False
                self.entered.append(bit)
                if len(self.entered) >= self.room_length:
                    if tuple(self.entered) != self.code:
                        self.failed = True
                        error = True
                    else:
                        self.stage += 1
                        if self.stage >= self.stage_count:
                            self.success = True
                            reward = 1.0
                        else:
                            self.entered.clear()
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
