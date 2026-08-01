from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .paper_v2_protocol import sha256_json
from .paper_v2_types import RawCausalObservation


MINECRAFT_CAUSAL_LAW_VERSION = "minecraft-causal-skills-v2.0"
MINECRAFT_CAUSAL_LAW = {
    "collect": "visible resources can be collected",
    "craft": "observed inventory can be transformed by recipes",
    "mine": "a crafted tool can remove a blocking wall",
    "place": "crafted material can span a gap",
    "use": "keys, light, and tools enable matching obstacles",
    "trade": "a collected resource can be exchanged for a key",
    "goal": "terminal reward is emitted only by claiming the goal region",
}
MINECRAFT_CAUSAL_LAW_SHA256 = sha256_json(MINECRAFT_CAUSAL_LAW)


class MinecraftSkillTrack(str, Enum):
    SEMANTIC = "semantic_skill"
    OPAQUE = "opaque_skill"


@dataclass(frozen=True, slots=True)
class MinecraftPrivateState:
    completed: frozenset[str] = frozenset()
    inventory: tuple[tuple[str, int], ...] = ()
    region: str = "spawn"
    health: int = 2
    terminal: bool = False
    success: bool = False
    dead_end: bool = False
    effect_history: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MinecraftStep:
    observation: RawCausalObservation
    reward: float
    action_succeeded: bool
    inventory_delta: Mapping[str, int]
    damage: float


@dataclass(frozen=True, slots=True)
class _SkillLaw:
    key: str
    semantic_action: str
    effect: str
    prerequisites: tuple[str, ...] = ()
    consumes: tuple[tuple[str, int], ...] = ()
    produces: tuple[tuple[str, int], ...] = ()
    region_from: str = ""
    region_to: str = ""
    damage: int = 0
    irreversible: bool = False
    repeatable: bool = False
    affordance: tuple[str, ...] = ()


SKILL_LAWS: tuple[_SkillLaw, ...] = (
    _SkillLaw("collect_log", "COLLECT(log)", "resource:wood", produces=(("log", 1),), affordance=("collectible", "wood")),
    _SkillLaw("craft_planks", "CRAFT(planks)", "craft:planks", ("collect_log",), consumes=(("log", 1),), produces=(("planks", 2),), affordance=("recipe", "wood")),
    _SkillLaw("place_bridge", "PLACE(bridge)", "path:bridge", ("craft_planks",), consumes=(("planks", 1),), region_from="spawn", region_to="goal", affordance=("placeable", "gap")),
    _SkillLaw("collect_stone", "COLLECT(stone)", "resource:stone", produces=(("stone", 1),), affordance=("collectible", "stone")),
    _SkillLaw("craft_table", "CRAFT(table)", "craft:station", ("craft_planks",), consumes=(("planks", 1),), produces=(("table", 1),), affordance=("recipe", "station")),
    _SkillLaw("craft_pickaxe", "CRAFT(pickaxe)", "craft:tool", ("craft_table", "collect_stone"), consumes=(("stone", 1),), produces=(("pickaxe", 1),), affordance=("recipe", "tool")),
    _SkillLaw("mine_tunnel", "MINE(tunnel)", "path:tunnel", ("craft_pickaxe",), consumes=(("pickaxe", 1),), region_from="spawn", region_to="goal", affordance=("mineable", "tool_required")),
    _SkillLaw("collect_coal", "COLLECT(coal)", "resource:coal", produces=(("coal", 1),), affordance=("collectible", "fuel")),
    _SkillLaw("use_light", "USE(coal,passage)", "risk:light", ("collect_coal",), consumes=(("coal", 1),), affordance=("usable", "dark_passage")),
    _SkillLaw("enter_passage", "MOVE_TO(passage)", "path:passage", ("use_light",), region_from="spawn", region_to="goal", affordance=("traversable", "lit")),
    _SkillLaw("collect_emerald", "COLLECT(emerald)", "resource:emerald", produces=(("emerald", 1),), affordance=("collectible", "tradeable")),
    _SkillLaw("trade_key", "TRADE(emerald,key)", "information:key", ("collect_emerald",), consumes=(("emerald", 1),), produces=(("key", 1),), affordance=("trade", "key")),
    _SkillLaw("use_gate", "USE(key,gate)", "path:gate", ("trade_key",), consumes=(("key", 1),), region_from="spawn", region_to="goal", affordance=("usable", "locked_gate")),
    _SkillLaw("claim_goal", "USE(goal)", "goal:claim", region_from="goal", affordance=("terminal",)),
    _SkillLaw("enter_lava", "MOVE_TO(lava)", "failure:damage", damage=2, irreversible=True, affordance=("hazard",)),
    _SkillLaw("discard_items", "USE(destroy_inventory)", "failure:resource_loss", irreversible=True, affordance=("destructive",)),
    _SkillLaw("inspect_noise", "USE(decorative_block)", "none", repeatable=True, affordance=("inspectable",)),
)


class MinecraftCausalWorld:
    """Deterministic skill-level mock with no Minecraft runtime dependency."""

    def __init__(
        self,
        *,
        world_seed: int,
        track: MinecraftSkillTrack | str,
        expose_opaque_affordances: bool = False,
    ) -> None:
        self.world_seed = int(world_seed)
        self.track = MinecraftSkillTrack(track)
        self.expose_opaque_affordances = bool(expose_opaque_affordances)
        rng = random.Random(self.world_seed ^ 0x4D494E45)
        self._opaque_tokens = {
            law.key: f"skill_{rng.getrandbits(64):016x}" for law in SKILL_LAWS
        }
        self._tokens = {
            law.key: (
                law.semantic_action
                if self.track is MinecraftSkillTrack.SEMANTIC
                else self._opaque_tokens[law.key]
            )
            for law in SKILL_LAWS
        }
        self._keys = {token: key for key, token in self._tokens.items()}
        self._observation_salt = rng.getrandbits(64)
        self._state = MinecraftPrivateState()
        self._last_succeeded: bool | None = None
        self._last_cost = 0.0
        self._last_damage = 0.0

    @property
    def causal_law_sha256(self) -> str:
        return MINECRAFT_CAUSAL_LAW_SHA256

    @property
    def action_token_sha256(self) -> str:
        return sha256_json(self._tokens)

    @property
    def analysis_private_state(self) -> MinecraftPrivateState:
        return self._state

    @property
    def terminal(self) -> bool:
        return self._state.terminal

    def _fact(self, value: str) -> str:
        return "mcobs_" + hashlib.sha256(
            f"{self._observation_salt}:{value}".encode("utf-8")
        ).hexdigest()[:16]

    def _available_keys(self) -> tuple[str, ...]:
        if self.terminal:
            return ()
        completed = self._state.completed
        inventory = dict(self._state.inventory)
        available: list[str] = []
        for law in SKILL_LAWS:
            if law.key in completed and not law.repeatable:
                continue
            if law.prerequisites and not set(law.prerequisites) <= completed:
                continue
            if law.region_from and law.region_from != self._state.region:
                continue
            if any(inventory.get(name, 0) < amount for name, amount in law.consumes):
                continue
            available.append(law.key)
        return tuple(available)

    def reset(self) -> RawCausalObservation:
        self._state = MinecraftPrivateState()
        self._last_succeeded = None
        self._last_cost = 0.0
        self._last_damage = 0.0
        return self.observe()

    def observe(self) -> RawCausalObservation:
        available = self._available_keys()
        show_affordances = (
            self.track is MinecraftSkillTrack.SEMANTIC
            or self.expose_opaque_affordances
        )
        affordances = {
            self._tokens[key]: next(law.affordance for law in SKILL_LAWS if law.key == key)
            for key in available
        } if show_affordances else {}
        facts = {self._fact(f"region:{self._state.region}")}
        facts.update(self._fact(f"done:{key}") for key in self._state.completed)
        return RawCausalObservation(
            inventory=dict(self._state.inventory),
            observable_facts=frozenset(facts),
            available_actions=tuple(self._tokens[key] for key in available),
            action_affordances=affordances,
            resource_cost=self._last_cost,
            health=self._state.health / 2.0,
            damage=self._last_damage,
            spatial_observations={"region": self._fact(self._state.region)},
            last_action_succeeded=self._last_succeeded,
            terminal_reward=1.0 if self._state.success else 0.0,
            terminal=self._state.terminal,
        )

    def step(self, action_token: str) -> MinecraftStep:
        if self.terminal:
            raise RuntimeError("cannot step a terminal world")
        key = self._keys.get(action_token)
        available = set(self._available_keys())
        if key is None or key not in available:
            self._last_succeeded = False
            self._last_cost = 0.0
            self._last_damage = 0.0
            return MinecraftStep(self.observe(), 0.0, False, {}, 0.0)
        law = next(item for item in SKILL_LAWS if item.key == key)
        before_inventory = dict(self._state.inventory)
        inventory = dict(before_inventory)
        for name, amount in law.consumes:
            inventory[name] = inventory.get(name, 0) - amount
        for name, amount in law.produces:
            inventory[name] = inventory.get(name, 0) + amount
        completed = set(self._state.completed)
        if not law.repeatable:
            completed.add(law.key)
        region = law.region_to or self._state.region
        health = max(0, self._state.health - law.damage)
        dead_end = health <= 0
        if law.key == "discard_items":
            inventory = {}
            dead_end = True
        success = law.key == "claim_goal" and region == "goal"
        self._state = MinecraftPrivateState(
            completed=frozenset(completed),
            inventory=tuple(sorted((name, amount) for name, amount in inventory.items() if amount)),
            region=region,
            health=health,
            terminal=success or dead_end,
            success=success,
            dead_end=dead_end,
            effect_history=self._state.effect_history + (law.effect,),
        )
        self._last_succeeded = True
        self._last_cost = float(sum(amount for _, amount in law.consumes))
        self._last_damage = float(law.damage)
        after_inventory = dict(self._state.inventory)
        delta = {
            name: after_inventory.get(name, 0) - before_inventory.get(name, 0)
            for name in set(before_inventory) | set(after_inventory)
            if after_inventory.get(name, 0) != before_inventory.get(name, 0)
        }
        return MinecraftStep(
            self.observe(), 1.0 if success else 0.0, True, delta,
            float(law.damage),
        )

    def clone(self) -> "MinecraftCausalWorld":
        return copy.deepcopy(self)

    def private_action_key(self, token: str) -> str:
        return self._keys[token]


@runtime_checkable
class MinecraftAdapter(Protocol):
    """Backend-neutral high-level contract; pixels and input control are out of scope."""

    track: MinecraftSkillTrack

    def reset(self, *, seed: int) -> RawCausalObservation: ...
    def observe(self) -> RawCausalObservation: ...
    def step(self, action: str) -> MinecraftStep: ...
    def close(self) -> None: ...


class MockMinecraftAdapter:
    def __init__(
        self,
        *,
        track: MinecraftSkillTrack | str,
        expose_opaque_affordances: bool = False,
    ) -> None:
        self.track = MinecraftSkillTrack(track)
        self.expose_opaque_affordances = bool(expose_opaque_affordances)
        self._world: MinecraftCausalWorld | None = None

    @property
    def world(self) -> MinecraftCausalWorld:
        if self._world is None:
            raise RuntimeError("adapter must be reset before use")
        return self._world

    def reset(self, *, seed: int) -> RawCausalObservation:
        self._world = MinecraftCausalWorld(
            world_seed=seed,
            track=self.track,
            expose_opaque_affordances=self.expose_opaque_affordances,
        )
        return self._world.observe()

    def observe(self) -> RawCausalObservation:
        return self.world.observe()

    def step(self, action: str) -> MinecraftStep:
        return self.world.step(action)

    def close(self) -> None:
        self._world = None


@dataclass(frozen=True, slots=True)
class MinecraftWorldCertification:
    solvable: bool
    minimum_plan_length: int | None
    valid_solution_count: int
    causal_family_count: int
    dead_end_count: int
    irreversible_decision_count: int
    random_policy_success_estimate: float
    causal_law_sha256: str
    action_token_sha256: str
    private_state_leak_count: int
    adequate: bool
    witness_plans: tuple[tuple[str, ...], ...]
    issues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _state_key(world: MinecraftCausalWorld) -> tuple[Any, ...]:
    state = world.analysis_private_state
    return (
        state.completed, state.inventory, state.region, state.health,
        state.terminal, state.success, state.dead_end,
    )


def _family(effects: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(effect for effect in set(effects) if not effect.startswith("goal:")))


def certify_minecraft_world(
    world: MinecraftCausalWorld,
    *,
    maximum_depth: int = 10,
    random_rollouts: int = 1000,
    random_budget: int = 8,
) -> MinecraftWorldCertification:
    queue = deque([(world.clone(), tuple())])
    seen_depth = {_state_key(world): 0}
    solutions: set[tuple[str, ...]] = set()
    families: set[tuple[str, ...]] = set()
    dead_ends: set[tuple[Any, ...]] = set()
    while queue:
        current, path = queue.popleft()
        if current.analysis_private_state.success:
            solutions.add(path)
            families.add(_family(current.analysis_private_state.effect_history))
            continue
        if current.analysis_private_state.dead_end:
            dead_ends.add(_state_key(current))
            continue
        if len(path) >= maximum_depth:
            continue
        for token in current.observe().available_actions:
            child = current.clone()
            key = child.private_action_key(token)
            child.step(token)
            next_path = path + (key,)
            child_key = _state_key(child)
            prior = seen_depth.get(child_key)
            if prior is None or len(next_path) < prior:
                seen_depth[child_key] = len(next_path)
                queue.append((child, next_path))
    minimum = min(map(len, solutions)) if solutions else None
    rng = random.Random(world.world_seed ^ 0x52414E44)
    successes = 0
    for _ in range(random_rollouts):
        trial = MinecraftCausalWorld(
            world_seed=world.world_seed,
            track=world.track,
            expose_opaque_affordances=world.expose_opaque_affordances,
        )
        for _step in range(random_budget):
            actions = trial.observe().available_actions
            if trial.terminal or not actions:
                break
            trial.step(rng.choice(actions))
        successes += int(trial.analysis_private_state.success)
    payload = json.dumps(world.observe().to_dict(), sort_keys=True).lower()
    forbidden = ("true_graph", "solution_family", "viability", "latent_risk", "optimal_plan", "oracle")
    leaks = sum(token in payload for token in forbidden)
    issues: list[str] = []
    if not solutions:
        issues.append("unsolvable")
    if minimum is None or not 3 <= minimum <= 12:
        issues.append("minimum plan length outside [3, 12]")
    if len(solutions) < 2:
        issues.append("fewer than two valid solutions")
    if len(families) < 3:
        issues.append("fewer than three causal families")
    if not dead_ends:
        issues.append("no reachable dead end")
    irreversible = sum(law.irreversible for law in SKILL_LAWS)
    if irreversible < 1:
        issues.append("no irreversible decision")
    random_success = successes / random_rollouts if random_rollouts else 0.0
    if random_success > 0.10:
        issues.append("random success exceeds 0.10")
    if leaks:
        issues.append("private state leaked into raw observation")
    ordered_plans = tuple(sorted(solutions, key=lambda plan: (len(plan), plan)))
    return MinecraftWorldCertification(
        solvable=bool(solutions),
        minimum_plan_length=minimum,
        valid_solution_count=len(solutions),
        causal_family_count=len(families),
        dead_end_count=len(dead_ends),
        irreversible_decision_count=irreversible,
        random_policy_success_estimate=random_success,
        causal_law_sha256=world.causal_law_sha256,
        action_token_sha256=world.action_token_sha256,
        private_state_leak_count=leaks,
        adequate=not issues,
        witness_plans=ordered_plans[:8],
        issues=tuple(issues),
    )
