from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from statistics import fmean
from typing import Any, Mapping, Sequence

from .paper_v2_protocol import sha256_json
from .paper_v2_types import RawCausalObservation


CAUSAL_LAW_VERSION = "causal-dependency-law-v2.0"
CAUSAL_LAW_DEFINITION = {
    "information": "scan enables parameter binding",
    "resource": "wood and metal are conserved during tool formation",
    "tool": "formed tool removes a blocking obstacle",
    "risk": "stabilization enables a safe traversal",
    "path": "wood can be converted into a bridge",
    "goal": "terminal reward is emitted only after occupying the goal region",
    "failure": "resource destruction and lethal hazard are irreversible",
}
CAUSAL_LAW_SHA256 = sha256_json(CAUSAL_LAW_DEFINITION)


@dataclass(frozen=True, slots=True)
class PrivateWorldState:
    """Analysis/solver state.  It must never be returned by ``observe``."""

    completed: frozenset[str] = frozenset()
    inventory: tuple[tuple[str, int], ...] = ()
    location: str = "start"
    health: int = 2
    latent_risk: int = 1
    terminal: bool = False
    success: bool = False
    dead_end: bool = False
    effect_history: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CausalStep:
    observation: RawCausalObservation
    reward: float
    action_succeeded: bool
    error: bool
    inventory_delta: Mapping[str, int]
    facts_added: frozenset[str]
    facts_removed: frozenset[str]
    unlocked_actions: tuple[str, ...]
    resource_cost: float
    damage: float
    spatial_change: tuple[str, str] | None


@dataclass(frozen=True, slots=True)
class _ActionLaw:
    key: str
    effect: str
    prerequisites: tuple[str, ...] = ()
    consumes: tuple[tuple[str, int], ...] = ()
    produces: tuple[tuple[str, int], ...] = ()
    location_from: str = ""
    location_to: str = ""
    damage: int = 0
    irreversible: bool = False
    distractor: bool = False
    affordance: tuple[str, ...] = ()


ACTION_LAWS: tuple[_ActionLaw, ...] = (
    _ActionLaw("scan", "information_acquisition", affordance=("inspect",)),
    _ActionLaw(
        "bind", "parameter_binding", ("scan",), affordance=("information",)
    ),
    _ActionLaw(
        "open_gate",
        "path_creation",
        ("bind",),
        location_from="start",
        location_to="goal",
        affordance=("gate",),
    ),
    _ActionLaw(
        "collect_wood",
        "resource_acquisition",
        produces=(("wood", 1),),
        affordance=("resource", "wood"),
    ),
    _ActionLaw(
        "collect_metal",
        "resource_acquisition",
        produces=(("metal", 1),),
        affordance=("resource", "metal"),
    ),
    _ActionLaw(
        "craft_tool",
        "tool_formation",
        ("collect_wood", "collect_metal"),
        consumes=(("wood", 1), ("metal", 1)),
        produces=(("tool", 1),),
        affordance=("craft",),
    ),
    _ActionLaw(
        "remove_obstacle",
        "obstacle_removal",
        ("craft_tool",),
        consumes=(("tool", 1),),
        location_from="start",
        location_to="goal",
        affordance=("tool", "obstacle"),
    ),
    _ActionLaw(
        "stabilize",
        "risk_reduction",
        affordance=("safety",),
    ),
    _ActionLaw(
        "safe_traverse",
        "path_creation",
        ("stabilize",),
        location_from="start",
        location_to="goal",
        affordance=("passage",),
    ),
    _ActionLaw(
        "place_bridge",
        "path_creation",
        ("collect_wood",),
        consumes=(("wood", 1),),
        location_from="start",
        location_to="goal",
        affordance=("wood", "gap"),
    ),
    _ActionLaw(
        "claim_goal",
        "goal_achievement",
        location_from="goal",
        affordance=("goal",),
    ),
    _ActionLaw(
        "burn_resources",
        "resource_destruction",
        irreversible=True,
        distractor=True,
        affordance=("destructive",),
    ),
    _ActionLaw(
        "enter_hazard",
        "damage",
        damage=2,
        irreversible=True,
        distractor=True,
        affordance=("hazard",),
    ),
    _ActionLaw("inspect_noise_a", "no_effect", distractor=True),
    _ActionLaw("inspect_noise_b", "no_effect", distractor=True),
)


class CausalDependencyWorldV2:
    """Sparse-reward causal world with token-independent physical laws."""

    def __init__(
        self,
        *,
        world_seed: int,
        token_seed: int | None = None,
        observation_seed: int | None = None,
        composition_template: str = "base",
        reward_mode: str = "strict_sparse",
        expose_affordances: bool = False,
    ) -> None:
        if reward_mode not in {"strict_sparse", "observable_progress"}:
            raise ValueError("unknown reward mode")
        self.world_seed = int(world_seed)
        self.token_seed = int(token_seed if token_seed is not None else world_seed)
        self.observation_seed = int(
            observation_seed if observation_seed is not None else world_seed
        )
        self.composition_template = str(composition_template)
        self.reward_mode = reward_mode
        self.expose_affordances = bool(expose_affordances)
        token_random = random.Random(self.token_seed ^ 0xA551)
        self._tokens = {
            law.key: f"act_{token_random.getrandbits(64):016x}"
            for law in ACTION_LAWS
        }
        self._keys = {token: key for key, token in self._tokens.items()}
        self._observation_salt = random.Random(
            self.observation_seed ^ 0xC415A1
        ).getrandbits(64)
        self._state = PrivateWorldState()
        self._last_succeeded: bool | None = None
        self._last_cost = 0.0
        self._last_damage = 0.0

    @property
    def causal_law_sha256(self) -> str:
        return CAUSAL_LAW_SHA256

    @property
    def composition_template_sha256(self) -> str:
        return hashlib.sha256(self.composition_template.encode("utf-8")).hexdigest()

    @property
    def observation_token_sha256(self) -> str:
        return hashlib.sha256(str(self._observation_salt).encode()).hexdigest()

    @property
    def action_token_sha256(self) -> str:
        return sha256_json(self._tokens)

    @property
    def terminal(self) -> bool:
        return self._state.terminal

    @property
    def analysis_private_state(self) -> PrivateWorldState:
        return self._state

    def _fact_token(self, fact: str) -> str:
        return "obs_" + hashlib.sha256(
            f"{self._observation_salt}:{fact}".encode()
        ).hexdigest()[:16]

    def _available_keys(self) -> tuple[str, ...]:
        if self._state.terminal:
            return ()
        completed = self._state.completed
        inventory = dict(self._state.inventory)
        available: list[str] = []
        for law in ACTION_LAWS:
            if law.key in completed and not law.distractor:
                continue
            if law.prerequisites and not set(law.prerequisites) <= completed:
                continue
            if law.location_from and self._state.location != law.location_from:
                continue
            if any(inventory.get(name, 0) < amount for name, amount in law.consumes):
                continue
            available.append(law.key)
        return tuple(available)

    def observe(self) -> RawCausalObservation:
        facts = {self._fact_token(f"location:{self._state.location}")}
        for key in self._state.completed:
            facts.add(self._fact_token(f"completed:{key}"))
        available = self._available_keys()
        affordances = (
            {
                self._tokens[key]: next(
                    law.affordance for law in ACTION_LAWS if law.key == key
                )
                for key in available
            }
            if self.expose_affordances
            else {}
        )
        spatial: dict[str, str | float | int] = {
            "region": self._fact_token(f"region:{self._state.location}")
        }
        if self.reward_mode == "observable_progress":
            spatial["natural_goal_visible"] = int(self._state.location == "goal")
        return RawCausalObservation(
            inventory=dict(self._state.inventory),
            observable_facts=frozenset(facts),
            available_actions=tuple(self._tokens[key] for key in available),
            action_affordances=affordances,
            resource_cost=self._last_cost,
            health=self._state.health / 2.0,
            damage=self._last_damage,
            spatial_observations=spatial,
            last_action_succeeded=self._last_succeeded,
            terminal_reward=1.0 if self._state.success else 0.0,
            terminal=self._state.terminal,
        )

    def _law_for_token(self, token: str) -> _ActionLaw | None:
        key = self._keys.get(token)
        if key is None:
            return None
        return next(law for law in ACTION_LAWS if law.key == key)

    def step(self, action_token: str) -> CausalStep:
        if self._state.terminal:
            raise RuntimeError("cannot step a terminal world")
        before = self.observe()
        before_inventory = dict(self._state.inventory)
        before_keys = set(self._available_keys())
        law = self._law_for_token(action_token)
        succeeded = law is not None and law.key in before_keys
        if not succeeded:
            self._last_succeeded = False
            self._last_cost = 0.0
            self._last_damage = 0.0
            after = self.observe()
            return CausalStep(
                after, 0.0, False, True, {}, frozenset(), frozenset(), (), 0.0, 0.0, None
            )
        assert law is not None
        completed = set(self._state.completed)
        inventory = dict(before_inventory)
        location = self._state.location
        health = self._state.health
        latent_risk = self._state.latent_risk
        dead_end = self._state.dead_end
        for name, amount in law.consumes:
            inventory[name] = inventory.get(name, 0) - amount
        for name, amount in law.produces:
            inventory[name] = inventory.get(name, 0) + amount
        if law.key == "burn_resources":
            inventory = {}
            dead_end = True
        if law.key == "stabilize":
            latent_risk = 0
        if law.damage:
            health = max(0, health - law.damage)
            dead_end = health <= 0
        if law.location_to:
            location = law.location_to
        completed.add(law.key)
        success = law.key == "claim_goal" and location == "goal"
        terminal = success or dead_end
        effect_history = self._state.effect_history + (law.effect,)
        self._state = PrivateWorldState(
            completed=frozenset(completed),
            inventory=tuple(sorted((key, value) for key, value in inventory.items() if value)),
            location=location,
            health=health,
            latent_risk=latent_risk,
            terminal=terminal,
            success=success,
            dead_end=dead_end,
            effect_history=effect_history,
        )
        resource_cost = float(sum(amount for _, amount in law.consumes))
        self._last_succeeded = True
        self._last_cost = resource_cost
        self._last_damage = float(law.damage)
        after = self.observe()
        after_inventory = dict(self._state.inventory)
        delta = {
            key: after_inventory.get(key, 0) - before_inventory.get(key, 0)
            for key in set(before_inventory) | set(after_inventory)
            if after_inventory.get(key, 0) != before_inventory.get(key, 0)
        }
        unlocked = tuple(
            self._tokens[key] for key in set(self._available_keys()) - before_keys
        )
        return CausalStep(
            observation=after,
            reward=1.0 if success else 0.0,
            action_succeeded=True,
            error=False,
            inventory_delta=delta,
            facts_added=after.observable_facts - before.observable_facts,
            facts_removed=before.observable_facts - after.observable_facts,
            unlocked_actions=unlocked,
            resource_cost=resource_cost,
            damage=float(law.damage),
            spatial_change=(before.spatial_observations["region"], after.spatial_observations["region"])
            if before.spatial_observations["region"] != after.spatial_observations["region"]
            else None,
        )

    def clone(self) -> "CausalDependencyWorldV2":
        return copy.deepcopy(self)

    def oracle_transition(self, action_token: str) -> RawCausalObservation:
        clone = self.clone()
        return clone.step(action_token).observation

    def private_action_key(self, action_token: str) -> str:
        return self._keys[action_token]


@dataclass(frozen=True, slots=True)
class WorldCertification:
    solvable: bool
    minimum_plan_length: int | None
    valid_solution_count: int
    causal_family_count: int
    dead_end_count: int
    irreversible_decision_count: int
    random_policy_success_estimate: float
    shortcut_exists: bool
    causal_law_sha256: str
    composition_template_sha256: str
    observation_token_sha256: str
    action_token_sha256: str
    private_state_leak_count: int
    adequate: bool
    issues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _state_key(world: CausalDependencyWorldV2) -> tuple[Any, ...]:
    state = world.analysis_private_state
    return (
        state.completed,
        state.inventory,
        state.location,
        state.health,
        state.latent_risk,
        state.terminal,
        state.success,
        state.dead_end,
    )


def _family(effects: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(set(effects) - {"goal_achievement", "no_effect"}))


def certify_world(
    world: CausalDependencyWorldV2,
    *,
    maximum_depth: int = 12,
    random_rollouts: int = 2000,
    random_budget: int = 8,
) -> WorldCertification:
    queue = deque([(world.clone(), tuple())])
    seen_depth: dict[tuple[Any, ...], int] = {_state_key(world): 0}
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
            state_key = _state_key(child)
            previous = seen_depth.get(state_key)
            if previous is None or len(next_path) < previous:
                seen_depth[state_key] = len(next_path)
                queue.append((child, next_path))
    minimum = min(map(len, solutions)) if solutions else None
    rng = random.Random(world.world_seed ^ 0x51A7)
    successes = 0
    for _ in range(random_rollouts):
        trial = CausalDependencyWorldV2(
            world_seed=world.world_seed,
            token_seed=world.token_seed,
            observation_seed=world.observation_seed,
            composition_template=world.composition_template,
            reward_mode=world.reward_mode,
            expose_affordances=world.expose_affordances,
        )
        for _step in range(random_budget):
            actions = trial.observe().available_actions
            if not actions or trial.terminal:
                break
            trial.step(rng.choice(actions))
        successes += int(trial.analysis_private_state.success)
    random_success = successes / random_rollouts if random_rollouts else 0.0
    serialized = json.dumps(world.observe().to_dict(), sort_keys=True).lower()
    private_tokens = (
        "viable",
        "solution_family",
        "optimal_plan",
        "oracle_transition",
        "latent_risk",
        "true_causal",
    )
    leaks = sum(token in serialized for token in private_tokens)
    irreversible = sum(law.irreversible for law in ACTION_LAWS)
    issues = []
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
    if irreversible < 1:
        issues.append("no irreversible decision")
    if random_success > 0.10:
        issues.append("random policy success exceeds 0.10")
    if leaks:
        issues.append("private state leaked into raw observation")
    return WorldCertification(
        solvable=bool(solutions),
        minimum_plan_length=minimum,
        valid_solution_count=len(solutions),
        causal_family_count=len(families),
        dead_end_count=len(dead_ends),
        irreversible_decision_count=irreversible,
        random_policy_success_estimate=random_success,
        shortcut_exists=bool(minimum is not None and minimum < 3),
        causal_law_sha256=world.causal_law_sha256,
        composition_template_sha256=world.composition_template_sha256,
        observation_token_sha256=world.observation_token_sha256,
        action_token_sha256=world.action_token_sha256,
        private_state_leak_count=leaks,
        adequate=not issues,
        issues=tuple(issues),
    )
