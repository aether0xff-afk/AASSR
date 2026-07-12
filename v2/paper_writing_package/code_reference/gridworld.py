from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable

from .knowledge import (
    KK,
    KnowledgeDelta,
    KnowledgeSource,
    KnowledgeStatus,
    KnowledgeStore,
    ValueType,
    seed_gridworld_knowledge,
)
from .imagination import ImaginationCycle, ImaginationTrace
from .policy import CandidateScorer, RandomScorer
from .prophecy import (
    ProphecyModule,
    ProphecyPrediction,
    ProphecyUpdate,
    gridworld_state_signature,
)
from .reward import RewardBreakdown, RewardModule

Cell = tuple[int, int]


class CellKind(StrEnum):
    EMPTY = "empty"
    WALL = "wall"
    HINT = "hint"
    KEY = "key"
    DOOR = "door"
    FLAG = "flag"


class ActionName(StrEnum):
    MOVE_TOWARD = "MOVE_TOWARD"
    INSPECT_CELL = "INSPECT_CELL"
    USE_OBJECT = "USE_OBJECT"
    FOLLOW_HINT = "FOLLOW_HINT"


@dataclass(frozen=True)
class ActionCandidate:
    name: ActionName
    template: str
    required_kk_slots: tuple[KK, ...]
    bindings: dict[KK, Any]
    strategy: str = "nearest"


@dataclass(frozen=True)
class DMPConfig:
    use_prophecy: bool = False
    use_imagination: bool = False
    prophecy_beta: float = 0.3


@dataclass(frozen=True)
class StepResult:
    step: int
    action: ActionCandidate
    observation: dict[str, Any]
    delta_k: KnowledgeDelta
    external_reward: float
    intrinsic_reward: float
    total_reward: float
    error: bool
    flag_found: bool
    done: bool
    prophecy_prediction: ProphecyPrediction | None = None
    prophecy_error: float = 0.0
    prophecy_loss: float = 0.0
    imagination_trace: ImaginationTrace | None = None

    def to_dict(self) -> dict[str, Any]:
        predicted_kk_count = 0.0
        predicted_error_prob = 0.0
        predicted_flag_prob = 0.0
        if self.prophecy_prediction is not None:
            predicted_kk_count = self.prophecy_prediction.expected_knowledge_gain()
            predicted_error_prob = self.prophecy_prediction.error_prob
            predicted_flag_prob = self.prophecy_prediction.flag_prob
        imagination_selected_score = 0.0
        imagination_candidate_count = 0
        imagination_best_flag_prob = 0.0
        imagination_best_error_prob = 0.0
        imagination_rollout_value = 0.0
        imagination_rollout_depth = 0
        if self.imagination_trace is not None:
            selected_score = self.imagination_trace.selected_score
            imagination_selected_score = selected_score.score
            imagination_candidate_count = len(self.imagination_trace.scores)
            imagination_best_flag_prob = selected_score.predicted_flag_prob
            imagination_best_error_prob = selected_score.predicted_error_prob
            imagination_rollout_value = selected_score.rollout_value
            imagination_rollout_depth = selected_score.rollout_depth
        return {
            "step": self.step,
            "action": self.action.name.value,
            "template": self.action.template,
            "observation": self.observation,
            "delta_k": {
                "added": len(self.delta_k.added),
                "updated": len(self.delta_k.updated),
                "status_changed": len(self.delta_k.status_changed),
                "removed": len(self.delta_k.removed),
                "usage_updated": len(self.delta_k.usage_updated),
                "semantic_gain": self.delta_k.semantic_information_gain(),
                "changed_kk": sorted(kk.value for kk in self.delta_k.semantic_changed_kk()),
            },
            "external_reward": self.external_reward,
            "intrinsic_reward": self.intrinsic_reward,
            "total_reward": self.total_reward,
            "error": self.error,
            "flag_found": self.flag_found,
            "done": self.done,
            "prophecy_error": self.prophecy_error,
            "prophecy_loss": self.prophecy_loss,
            "predicted_kk_count": predicted_kk_count,
            "predicted_error_prob": predicted_error_prob,
            "predicted_flag_prob": predicted_flag_prob,
            "imagination_selected_score": imagination_selected_score,
            "imagination_candidate_count": imagination_candidate_count,
            "imagination_best_flag_prob": imagination_best_flag_prob,
            "imagination_best_error_prob": imagination_best_error_prob,
            "imagination_rollout_value": imagination_rollout_value,
            "imagination_rollout_depth": imagination_rollout_depth,
        }


class GridWorld:
    """Small deterministic GridWorld used to validate APASSR knowledge flow."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        start: Cell,
        cells: dict[Cell, CellKind] | None = None,
        hints: dict[Cell, str | Cell] | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.start = start
        self.cells = dict(cells or {})
        self.hints = dict(hints or {})
        self.opened_doors: set[Cell] = set()

    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def kind_at(self, cell: Cell) -> CellKind:
        return self.cells.get(cell, CellKind.EMPTY)

    def inspect(self, cell: Cell) -> dict[str, Any]:
        if not self.in_bounds(cell):
            return {"cell": cell, "kind": CellKind.WALL, "blocked": True}
        kind = self.kind_at(cell)
        observation: dict[str, Any] = {
            "cell": cell,
            "kind": kind,
            "blocked": kind == CellKind.WALL,
        }
        if kind == CellKind.HINT:
            observation["hint"] = self.hints[cell]
        return observation

    def can_enter(self, cell: Cell, *, has_key: bool) -> bool:
        if not self.in_bounds(cell):
            return False
        kind = self.kind_at(cell)
        if kind == CellKind.WALL:
            return False
        if kind == CellKind.DOOR and cell not in self.opened_doors:
            return False
        return True

    def open_door(self, cell: Cell, *, has_key: bool) -> bool:
        if self.kind_at(cell) != CellKind.DOOR or not has_key:
            return False
        self.opened_doors.add(cell)
        return True


class GridWorldDMP:
    """Decision-making process for knowledge-driven GridWorld actions."""

    def __init__(
        self,
        world: GridWorld,
        *,
        top_k: int = 5,
        scorer: CandidateScorer[ActionCandidate, GridWorldDMP] | None = None,
        reward_module: RewardModule | None = None,
        prophecy: ProphecyModule | None = None,
        imagination: ImaginationCycle | None = None,
        config: DMPConfig | None = None,
        step_limit: int = 200,
    ) -> None:
        self.world = world
        self.store = seed_gridworld_knowledge(world.start)
        self.position = world.start
        self.step_index = 0
        self.top_k = top_k
        self.scorer = scorer or RandomScorer()
        self.reward_module = reward_module or RewardModule()
        self.prophecy = prophecy
        self.imagination = imagination
        self.config = config or DMPConfig()
        self.last_imagination_trace: ImaginationTrace | None = None
        self.step_limit = step_limit
        self.done = False
        self._executed_signatures: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        self._record_known_position(world.start)
        self._refresh_unknown_neighbors()

    def generate_candidates(self) -> list[ActionCandidate]:
        candidates: list[ActionCandidate] = []
        candidates.extend(self._inspect_candidates(KK.UNKNOWN_NEIGHBOR))
        candidates.extend(self._inspect_candidates(KK.FRONTIER_CELL))
        candidates.extend(self._move_candidates(KK.FLAG_CELL))
        candidates.extend(self._move_candidates(KK.KEY_CELL))
        candidates.extend(self._move_candidates(KK.DOOR_CELL))
        candidates.extend(self._move_candidates(KK.HINT_CELL))
        candidates.extend(self._move_candidates(KK.FRONTIER_CELL))
        candidates.extend(self._use_object_candidates())
        candidates.extend(self._hint_candidates())
        return candidates

    def choose_candidate(self, strategy: str = "nearest") -> ActionCandidate | None:
        candidates = self.generate_candidates()
        if not candidates:
            return None
        if self.config.use_imagination and self.imagination is not None:
            state_signature = gridworld_state_signature(self)
            trace = self.imagination.choose(
                state_signature,
                candidates,
                policy=self.scorer if hasattr(self.scorer, "candidate_probability") else None,
                dmp=self,
            )
            self.last_imagination_trace = trace
            return trace.selected
        if strategy == "random":
            return RandomScorer().choose(candidates, self)
        if strategy in {"nearest", "least_tried", "high_uncertainty"}:
            return self._choose_by_strategy(candidates, strategy)
        return self.scorer.choose(candidates, self)

    def execute(self, candidate: ActionCandidate) -> StepResult:
        state_signature = gridworld_state_signature(self)
        prophecy_prediction = self._predict_prophecy(state_signature, candidate)
        before = self.store.snapshot_items()
        repeated = self._signature(candidate) in self._executed_signatures
        self._executed_signatures.add(self._signature(candidate))
        self.step_index += 1
        observation: dict[str, Any] = {}
        error = False
        flag_found = False

        if candidate.name == ActionName.INSPECT_CELL:
            cell = self._target_cell(candidate)
            observation = self.world.inspect(cell)
            self.apply_observation(observation)
            self._mark_bound_slots_used(candidate, success=True)
            flag_found = observation["kind"] == CellKind.FLAG
        elif candidate.name == ActionName.MOVE_TOWARD:
            target = self._target_cell(candidate)
            moved = self._move_one_step_toward(target)
            self._mark_bound_slots_used(candidate, success=moved)
            error = not moved
            observation = {"target": target, "moved": moved, "position": self.position}
            if self.position == target and self.world.kind_at(self.position) == CellKind.FLAG:
                self.store.add(KK.FLAG_CELL, self.position, ValueType.CELL_COORD, step=self.step_index)
                flag_found = True
        elif candidate.name == ActionName.USE_OBJECT:
            door = candidate.bindings[KK.DOOR_CELL]
            opened = self.world.open_door(door, has_key=self._has_key())
            if opened:
                self.store.mark(KK.DOOR_CELL, door, KnowledgeStatus.CONSUMED, step=self.step_index)
            self._mark_bound_slots_used(candidate, success=opened)
            error = not opened
            observation = {"door": door, "opened": opened}
        elif candidate.name == ActionName.FOLLOW_HINT:
            hint = candidate.bindings[KK.HINT_VALUE]
            self._apply_hint(hint)
            self.store.mark(KK.HINT_VALUE, hint, KnowledgeStatus.CONSUMED, step=self.step_index)
            self._mark_bound_slots_used(candidate, success=True)
            observation = {"hint": hint}
        else:
            raise ValueError(f"Unsupported action: {candidate.name}")

        delta_k = self.store.delta_since(before)
        self.done = flag_found or self.step_index >= self.step_limit
        reward = self.reward_module.compute(
            delta_k=delta_k,
            error=error,
            repeated=repeated,
            flag_found=flag_found,
        )
        prophecy_update = self._update_prophecy(
            state_signature,
            candidate,
            delta_k,
            error=error,
            flag_found=flag_found,
        )
        result = self._step_result(
            candidate,
            observation,
            delta_k,
            reward,
            error,
            flag_found,
            prophecy_prediction=prophecy_prediction,
            prophecy_update=prophecy_update,
            imagination_trace=self.last_imagination_trace
            if self.last_imagination_trace is not None
            and self.last_imagination_trace.selected == candidate
            else None,
        )
        self.last_imagination_trace = None
        self._update_selector(candidate, result.total_reward)
        return result

    def apply_observation(self, observation: dict[str, Any]) -> None:
        cell: Cell = observation["cell"]
        kind: CellKind = observation["kind"]

        if observation.get("blocked"):
            self._remove_from_candidate_pools(cell)
            self.store.add(
                KK.WALL_CELL,
                cell,
                ValueType.CELL_COORD,
                status=KnowledgeStatus.BLOCKED,
                step=self.step_index,
            )
            return

        self._promote_to_known(cell)
        if kind == CellKind.HINT:
            self.store.add(KK.HINT_CELL, cell, ValueType.CELL_COORD, step=self.step_index)
            hint = observation["hint"]
            hint_type = ValueType.HINT_TARGET if isinstance(hint, tuple) else ValueType.HINT_TEXT
            self.store.add(KK.HINT_VALUE, hint, hint_type, step=self.step_index)
        elif kind == CellKind.KEY:
            self.store.add(KK.KEY_CELL, cell, ValueType.CELL_COORD, step=self.step_index)
        elif kind == CellKind.DOOR:
            self.store.add(KK.DOOR_CELL, cell, ValueType.CELL_COORD, step=self.step_index)
        elif kind == CellKind.FLAG:
            self.store.add(KK.FLAG_CELL, cell, ValueType.CELL_COORD, step=self.step_index)
        self._refresh_unknown_neighbors(anchor=cell)

    def metrics(self) -> dict[str, float]:
        all_templates = 10
        candidates = self.generate_candidates()
        bound_templates = {candidate.template for candidate in candidates}
        reused = 0
        for kk in KK:
            reused += sum(kv.used_count > 0 for kv in self.store.values(kk, include_inactive=True))
        return {
            "slot_binding_success_rate": len(bound_templates) / all_templates,
            "valid_action_candidate_ratio": 1.0 if candidates else 0.0,
            "knowledge_reuse_count": float(reused),
        }

    def _inspect_candidates(self, kk: KK) -> list[ActionCandidate]:
        candidates: list[ActionCandidate] = []
        for kv in self.store.values(kk, top_k=self.top_k):
            cell = kv.value
            if self._is_adjacent(cell):
                candidates.append(
                    ActionCandidate(
                        name=ActionName.INSPECT_CELL,
                        template=f"INSPECT_CELL {{{kk.value}}}",
                        required_kk_slots=(KK.CURRENT_POS, kk),
                        bindings={KK.CURRENT_POS: self.position, kk: cell},
                        strategy="least_tried",
                    )
                )
        return candidates

    def _move_candidates(self, kk: KK) -> list[ActionCandidate]:
        candidates: list[ActionCandidate] = []
        for kv in self.store.values(kk, top_k=self.top_k):
            target = kv.value
            if kk == KK.DOOR_CELL and self._is_adjacent(target) and target not in self.world.opened_doors:
                continue
            if target != self.position and self._reachable(target):
                candidates.append(
                    ActionCandidate(
                        name=ActionName.MOVE_TOWARD,
                        template=f"MOVE_TOWARD {{{kk.value}}}",
                        required_kk_slots=(KK.CURRENT_POS, kk),
                        bindings={KK.CURRENT_POS: self.position, kk: target},
                        strategy="nearest",
                    )
                )
        return candidates

    def _use_object_candidates(self) -> list[ActionCandidate]:
        if not self._has_key():
            return []
        candidates: list[ActionCandidate] = []
        keys = self.store.values(KK.KEY_OBJECT, top_k=1)
        for door in self.store.values(KK.DOOR_CELL, top_k=self.top_k):
            if not self._is_adjacent(door.value):
                continue
            candidates.append(
                ActionCandidate(
                    name=ActionName.USE_OBJECT,
                    template="USE_OBJECT {KK_KEY_OBJECT} ON {KK_DOOR_CELL}",
                    required_kk_slots=(KK.KEY_OBJECT, KK.DOOR_CELL),
                    bindings={KK.KEY_OBJECT: keys[0].value, KK.DOOR_CELL: door.value},
                    strategy="normal",
                )
            )
        return candidates

    def _hint_candidates(self) -> list[ActionCandidate]:
        return [
            ActionCandidate(
                name=ActionName.FOLLOW_HINT,
                template="FOLLOW_HINT {KK_HINT_VALUE}",
                required_kk_slots=(KK.HINT_VALUE,),
                bindings={KK.HINT_VALUE: kv.value},
                strategy="prophecy_best",
            )
            for kv in self.store.values(KK.HINT_VALUE, top_k=self.top_k)
        ]

    def _move_one_step_toward(self, target: Cell) -> bool:
        path = self._path_to(target)
        if len(path) < 2:
            return False
        next_cell = path[1]
        if not self.world.can_enter(next_cell, has_key=self._has_key()):
            if self.world.kind_at(next_cell) == CellKind.DOOR:
                self.store.add(
                    KK.DOOR_CELL,
                    next_cell,
                    ValueType.CELL_COORD,
                    step=self.step_index,
                )
                return False
            self.store.add(
                KK.WALL_CELL,
                next_cell,
                ValueType.CELL_COORD,
                status=KnowledgeStatus.BLOCKED,
                step=self.step_index,
            )
            return False
        self.position = next_cell
        self._record_known_position(next_cell)
        self._handle_entered_cell(next_cell)
        self._refresh_unknown_neighbors()
        return True

    def _handle_entered_cell(self, cell: Cell) -> None:
        kind = self.world.kind_at(cell)
        if kind == CellKind.KEY:
            self.store.mark(KK.KEY_CELL, cell, KnowledgeStatus.CONSUMED, step=self.step_index)
            self.store.add(KK.KEY_OBJECT, "key#1", ValueType.OBJECT_INSTANCE, step=self.step_index)
        elif kind == CellKind.DOOR and cell in self.world.opened_doors:
            self.store.mark(KK.DOOR_CELL, cell, KnowledgeStatus.CONSUMED, step=self.step_index)
        elif kind == CellKind.FLAG:
            self.store.add(KK.FLAG_CELL, cell, ValueType.CELL_COORD, step=self.step_index)

    def _apply_hint(self, hint: str | Cell) -> None:
        if isinstance(hint, tuple):
            self.store.add(
                KK.FLAG_CELL,
                hint,
                ValueType.CELL_COORD,
                source=KnowledgeSource.INFERRED,
                confidence=0.7,
                step=self.step_index,
            )

    def _record_known_position(self, cell: Cell) -> None:
        self.store.set_singleton(KK.CURRENT_POS, cell, ValueType.CELL_COORD, step=self.step_index)
        self.store.add(KK.VISITED_CELL, cell, ValueType.CELL_COORD, step=self.step_index)
        self._promote_to_known(cell)

    def _promote_to_known(self, cell: Cell) -> None:
        self._remove_from_candidate_pools(cell)
        self.store.add(KK.KNOWN_CELL, cell, ValueType.CELL_COORD, step=self.step_index)

    def _remove_from_candidate_pools(self, cell: Cell) -> None:
        self.store.remove(KK.UNKNOWN_NEIGHBOR, cell)
        self.store.remove(KK.FRONTIER_CELL, cell)

    def _refresh_unknown_neighbors(self, anchor: Cell | None = None) -> None:
        for cell in self._neighbors(anchor or self.position):
            if not self.world.in_bounds(cell):
                continue
            if self._known_or_blocked(cell):
                continue
            self.store.add(KK.UNKNOWN_NEIGHBOR, cell, ValueType.CELL_COORD, step=self.step_index)
            self.store.add(KK.FRONTIER_CELL, cell, ValueType.CELL_COORD, step=self.step_index)

    def _neighbors(self, cell: Cell) -> Iterable[Cell]:
        x, y = cell
        yield (x, y - 1)
        yield (x, y + 1)
        yield (x - 1, y)
        yield (x + 1, y)

    def _path_to(self, target: Cell) -> list[Cell]:
        queue: deque[Cell] = deque([self.position])
        parents: dict[Cell, Cell | None] = {self.position: None}
        while queue:
            cell = queue.popleft()
            if cell == target:
                break
            for neighbor in self._neighbors(cell):
                if neighbor in parents:
                    continue
                if not self.world.in_bounds(neighbor):
                    continue
                if self._blocked(neighbor):
                    continue
                if neighbor != target and not self._known_or_frontier(neighbor):
                    continue
                parents[neighbor] = cell
                queue.append(neighbor)
        if target not in parents:
            return []
        path = [target]
        while parents[path[-1]] is not None:
            path.append(parents[path[-1]])
        return list(reversed(path))

    def _reachable(self, target: Cell) -> bool:
        return bool(self._path_to(target))

    def _known_or_blocked(self, cell: Cell) -> bool:
        return self._known(cell) or self._blocked(cell)

    def _known_or_frontier(self, cell: Cell) -> bool:
        return self._known(cell) or self._active_value(KK.FRONTIER_CELL, cell)

    def _known(self, cell: Cell) -> bool:
        return self._active_value(KK.KNOWN_CELL, cell)

    def _blocked(self, cell: Cell) -> bool:
        return self._active_value(KK.WALL_CELL, cell, include_inactive=True)

    def _active_value(self, kk: KK, value: Any, *, include_inactive: bool = False) -> bool:
        return any(kv.value == value for kv in self.store.values(kk, include_inactive=include_inactive))

    def _has_key(self) -> bool:
        return self.store.has_active(KK.KEY_OBJECT)

    def _is_adjacent(self, cell: Cell) -> bool:
        return cell in set(self._neighbors(self.position))

    def _target_cell(self, candidate: ActionCandidate) -> Cell:
        for kk, value in candidate.bindings.items():
            if kk != KK.CURRENT_POS and isinstance(value, tuple):
                return value
        raise ValueError(f"Candidate has no cell target: {candidate}")

    def _mark_bound_slots_used(self, candidate: ActionCandidate, *, success: bool) -> None:
        for kk, value in candidate.bindings.items():
            self.store.mark_used(kk, value, success=success, step=self.step_index)

    def _distance(self, candidate: ActionCandidate) -> int:
        try:
            target = self._target_cell(candidate)
        except ValueError:
            return 0
        return abs(target[0] - self.position[0]) + abs(target[1] - self.position[1])

    def distance_for(self, candidate: ActionCandidate) -> int:
        return self._distance(candidate)

    def _used_count(self, candidate: ActionCandidate) -> int:
        counts = []
        for kk, value in candidate.bindings.items():
            for kv in self.store.values(kk, include_inactive=True):
                if kv.value == value:
                    counts.append(kv.used_count)
        return min(counts or [0])

    def _confidence(self, candidate: ActionCandidate) -> float:
        confidences = []
        for kk, value in candidate.bindings.items():
            for kv in self.store.values(kk, include_inactive=True):
                if kv.value == value:
                    confidences.append(kv.confidence)
        return min(confidences or [1.0])

    def _choose_by_strategy(self, candidates: list[ActionCandidate], strategy: str) -> ActionCandidate:
        if strategy == "least_tried":
            return min(candidates, key=self._used_count)
        if strategy == "high_uncertainty":
            return min(candidates, key=self._confidence)
        return min(candidates, key=self._distance)

    def _signature(self, candidate: ActionCandidate) -> tuple[str, tuple[tuple[str, str], ...]]:
        return (
            candidate.template,
            tuple(
                sorted(
                    (kk.value, repr(value))
                    for kk, value in candidate.bindings.items()
                    if kk != KK.CURRENT_POS
                )
            ),
        )

    def _step_result(
        self,
        candidate: ActionCandidate,
        observation: dict[str, Any],
        delta_k: KnowledgeDelta,
        reward: RewardBreakdown,
        error: bool,
        flag_found: bool,
        prophecy_prediction: ProphecyPrediction | None = None,
        prophecy_update: ProphecyUpdate | None = None,
        imagination_trace: ImaginationTrace | None = None,
    ) -> StepResult:
        prophecy_error = prophecy_update.prediction_error if prophecy_update is not None else 0.0
        prophecy_loss = prophecy_update.loss if prophecy_update is not None else 0.0
        total_reward = reward.total_reward + self.config.prophecy_beta * prophecy_error
        return StepResult(
            step=self.step_index,
            action=candidate,
            observation=observation,
            delta_k=delta_k,
            external_reward=reward.external_reward,
            intrinsic_reward=reward.intrinsic_reward,
            total_reward=total_reward,
            error=error,
            flag_found=flag_found,
            done=self.done,
            prophecy_prediction=prophecy_prediction,
            prophecy_error=prophecy_error,
            prophecy_loss=prophecy_loss,
            imagination_trace=imagination_trace,
        )

    def _update_selector(self, candidate: ActionCandidate, reward: float) -> None:
        update = getattr(self.scorer, "update", None)
        if callable(update):
            update(candidate, reward)

    def _predict_prophecy(
        self,
        state_signature: Any,
        candidate: ActionCandidate,
    ) -> ProphecyPrediction | None:
        if not self.config.use_prophecy or self.prophecy is None:
            return None
        return self.prophecy.predict(state_signature, candidate)

    def _update_prophecy(
        self,
        state_signature: Any,
        candidate: ActionCandidate,
        delta_k: KnowledgeDelta,
        *,
        error: bool,
        flag_found: bool,
    ) -> ProphecyUpdate | None:
        if not self.config.use_prophecy or self.prophecy is None:
            return None
        return self.prophecy.update(
            state_signature,
            candidate,
            delta_k,
            error,
            flag_found,
        )
