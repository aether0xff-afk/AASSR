from __future__ import annotations

import csv
import importlib
import json
import random
import subprocess
import sys
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence

from .current_protocol import current_frontier
from .dreamerv3_baseline import (
    DREAMERV3_ACTION_SLOT_COUNT,
    DREAMERV3_BASELINE_VERSION,
    DREAMERV3_CONDITION,
    DREAMERV3_UPSTREAM_COMMIT,
    dreamer_action_surface_mask,
    dreamer_adapter_manifest,
    dreamer_observation_vector,
    project_dreamer_action,
)
from .pentest_agent_main_test import ACTION_FEATURE_SIZE, AGENT_STATE_SIZE
from .pentest_curriculum_env import STALL_PATIENCE
from .pentest_curriculum_schedule import semantic_fingerprint
from .pentest_transfer_stages import (
    TRANSFER_DIAGNOSTIC_SEEDS,
    TRANSFER_STAGES,
    TRANSFER_TRAIN_SEEDS,
    TransferAdaptiveCurriculum,
    TransferDiagnosticWorld,
    stage_manifest,
)
from . import pentest_curriculum_schedule as schedule


DREAMERV3_CURRENT_PROTOCOL_VERSION = "dreamerv3-current-protocol-v1"
DREAMERV3_VALIDATION_SEEDS: tuple[int, ...] = tuple(range(93_001, 93_009))
DREAMERV3_OFFICIAL_CONFIG = "dmc_proprio+size1m"


@dataclass(frozen=True, slots=True)
class DreamerEpisodeResult:
    phase: str
    research_seed: int
    stage_index: int
    stage: str
    scenario_seed: int
    status: str
    success: int
    failure: int
    stalled: int
    truncation: int
    primitive_transitions: int
    reward: float
    projection_mean_squared_distance: float
    projection_max_squared_distance: float
    projection_tie_events: int


@dataclass(slots=True)
class _DreamerTrainState:
    real_transitions: int = 0
    gradient_updates: int = 0


class _DreamerPentestEnv:
    """Embodied-compatible adapter over one fixed current-protocol episode.

    The wrapped MDP has a fixed continuous relational-action vector. Each vector
    is deterministically projected onto the nearest action in the *publicly
    available* action surface of the current state. No hidden scenario fields are
    used for projection. The 240-slot relational availability mask is included in
    the observation so Dreamer receives the legality information that current DQN
    and AASSR receive through StateSnapshot.available_actions.
    """

    def __init__(
        self,
        *,
        elements: Any,
        np: Any,
        research_seed: int,
        stage_index: int,
        scenario_seed: int,
        transition_cap: int,
        phase: str,
    ) -> None:
        self.elements = elements
        self.np = np
        self.research_seed = int(research_seed)
        self.stage_index = int(stage_index)
        self.stage = TRANSFER_STAGES[self.stage_index]
        self.scenario_seed = int(scenario_seed)
        self.transition_cap = int(transition_cap)
        self.phase = str(phase)
        if self.transition_cap <= 0:
            raise ValueError("Dreamer episode transition_cap must be positive")

        self.world: TransferDiagnosticWorld | None = None
        self.done = True
        self.transitions = 0
        self.unchanged = 0
        self.recent_pairs: deque[tuple[tuple[Any, ...], str]] = deque(
            maxlen=STALL_PATIENCE
        )
        self.projection_distances: list[float] = []
        self.projection_tie_events = 0
        self.result: DreamerEpisodeResult | None = None

    @property
    def obs_space(self) -> dict[str, Any]:
        return {
            "state": self.elements.Space(self.np.float32, (AGENT_STATE_SIZE,)),
            "action_mask": self.elements.Space(
                self.np.float32, (DREAMERV3_ACTION_SLOT_COUNT,), 0.0, 1.0
            ),
            "reward": self.elements.Space(self.np.float32),
            "is_first": self.elements.Space(bool),
            "is_last": self.elements.Space(bool),
            "is_terminal": self.elements.Space(bool),
        }

    @property
    def act_space(self) -> dict[str, Any]:
        return {
            "reset": self.elements.Space(bool),
            "action": self.elements.Space(
                self.np.float32,
                (ACTION_FEATURE_SIZE,),
                -1.0,
                1.0,
            ),
        }

    def close(self) -> None:
        return None

    def _snapshot(self) -> Any:
        if self.world is None:
            raise RuntimeError("Dreamer episode has not been reset")
        return self.world.snapshot()

    def _observation(
        self,
        *,
        reward: float,
        is_first: bool,
        is_last: bool,
        is_terminal: bool,
    ) -> dict[str, Any]:
        state = self._snapshot()
        return {
            "state": self.np.asarray(
                dreamer_observation_vector(state), dtype=self.np.float32
            ),
            "action_mask": self.np.asarray(
                dreamer_action_surface_mask(state), dtype=self.np.float32
            ),
            "reward": self.np.float32(reward),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
        }

    def _status(self, stalled: bool) -> str | None:
        assert self.world is not None
        if self.world.success and self.world.proof_acquired:
            return "success"
        if self.world.failed and self.world.locked:
            return "failure"
        if stalled:
            return "stalled"
        if self.world.rate_limited or self.transitions >= self.transition_cap:
            return "truncation"
        return None

    @staticmethod
    def _reward(status: str) -> float:
        if status == "success":
            return 1.0
        if status == "failure":
            return -1.0
        return 0.0

    def _finish(self, status: str) -> dict[str, Any]:
        self.done = True
        reward = self._reward(status)
        distances = self.projection_distances
        self.result = DreamerEpisodeResult(
            phase=self.phase,
            research_seed=self.research_seed,
            stage_index=self.stage_index,
            stage=self.stage.name,
            scenario_seed=self.scenario_seed,
            status=status,
            success=int(status == "success"),
            failure=int(status == "failure"),
            stalled=int(status == "stalled"),
            truncation=int(status == "truncation"),
            primitive_transitions=self.transitions,
            reward=reward,
            projection_mean_squared_distance=(fmean(distances) if distances else 0.0),
            projection_max_squared_distance=(max(distances) if distances else 0.0),
            projection_tie_events=self.projection_tie_events,
        )
        # Current DQN/AASSR explicitly cut bootstrap at every reset boundary,
        # including stall/rate-limit/budget truncation. Dreamer receives the same
        # boundary through is_terminal=True rather than silently bootstrapping
        # across a new scenario.
        return self._observation(
            reward=reward,
            is_first=False,
            is_last=True,
            is_terminal=True,
        )

    def step(self, action: Mapping[str, Any]) -> dict[str, Any]:
        if bool(action.get("reset", False)) or self.world is None or self.done:
            self.world = TransferDiagnosticWorld(
                self.scenario_seed,
                stage=self.stage,
            )
            self.done = False
            self.transitions = 0
            self.unchanged = 0
            self.recent_pairs.clear()
            self.projection_distances.clear()
            self.projection_tie_events = 0
            self.result = None
            return self._observation(
                reward=0.0,
                is_first=True,
                is_last=False,
                is_terminal=False,
            )

        before = self.world.snapshot()
        proposal = action.get("action")
        if proposal is None:
            raise ValueError("DreamerV3 action mapping is missing 'action'")
        projection = project_dreamer_action(before, proposal)
        self.projection_distances.append(projection.squared_distance)
        self.projection_tie_events += int(projection.tied_candidates > 1)
        self.world.step(projection.action)
        after = self.world.snapshot()
        self.transitions += 1

        semantic_before = semantic_fingerprint(before)
        semantic_after = semantic_fingerprint(after)
        self.unchanged = self.unchanged + 1 if semantic_after == semantic_before else 0
        self.recent_pairs.append((semantic_before, projection.action.signature))
        stalled = False
        if self.unchanged >= STALL_PATIENCE:
            counts = Counter(self.recent_pairs)
            stalled = len(counts) <= 3 or max(counts.values(), default=0) >= 4

        status = self._status(stalled)
        if status is not None:
            return self._finish(status)
        return self._observation(
            reward=0.0,
            is_first=False,
            is_last=False,
            is_terminal=False,
        )


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_official_dreamer(
    dreamer_root: str | Path,
    *,
    allow_upstream_mismatch: bool,
) -> dict[str, Any]:
    root = Path(dreamer_root).expanduser().resolve()
    if not (root / "dreamerv3" / "agent.py").exists():
        raise FileNotFoundError(
            f"{root} is not an official danijar/dreamerv3 checkout"
        )
    head = _git_head(root)
    if head != DREAMERV3_UPSTREAM_COMMIT and not allow_upstream_mismatch:
        raise RuntimeError(
            "DreamerV3 checkout SHA mismatch: "
            f"expected {DREAMERV3_UPSTREAM_COMMIT}, got {head}. "
            "Checkout the pinned commit or pass --allow-upstream-mismatch for a "
            "non-canonical diagnostic run."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import numpy as np
        import elements
        import embodied
        import ruamel.yaml as yaml
        from dreamerv3.agent import Agent
        dreamer_main = importlib.import_module("dreamerv3.main")
    except ImportError as exc:
        raise RuntimeError(
            "Official DreamerV3 dependencies are unavailable. Install the "
            "requirements of the pinned danijar/dreamerv3 checkout in this "
            "Python environment before running the baseline."
        ) from exc
    return {
        "root": root,
        "head": head,
        "np": np,
        "elements": elements,
        "embodied": embodied,
        "yaml": yaml,
        "Agent": Agent,
        "main": dreamer_main,
    }


def _official_config(
    upstream: Mapping[str, Any],
    *,
    output_dir: Path,
    research_seed: int,
    jax_platform: str,
    train_ratio: float | None,
    prealloc: bool,
) -> Any:
    elements = upstream["elements"]
    yaml = upstream["yaml"]
    root: Path = upstream["root"]
    configs = yaml.YAML(typ="safe").load(
        (root / "dreamerv3" / "configs.yaml").read_text(encoding="utf-8")
    )
    config = elements.Config(configs["defaults"])
    # Use the upstream vector/proprioceptive preset and smallest published model
    # size. This is predeclared before results and avoids tuning Dreamer on AASSR.
    config = config.update(configs["size1m"])
    config = config.update(configs["dmc_proprio"])
    config = config.update(
        logdir=str((output_dir / "official_dreamer_log").resolve()),
        seed=int(research_seed),
    )
    config = config.update(
        {
            "jax": {
                "platform": str(jax_platform),
                "prealloc": bool(prealloc),
            }
        }
    )
    if train_ratio is not None:
        config = config.update({"run": {"train_ratio": float(train_ratio)}})
    return config


def _make_agent(
    upstream: Mapping[str, Any],
    config: Any,
    probe_env: _DreamerPentestEnv,
) -> Any:
    elements = upstream["elements"]
    Agent = upstream["Agent"]
    obs_space = dict(probe_env.obs_space)
    act_space = {
        key: value for key, value in probe_env.act_space.items() if key != "reset"
    }
    return Agent(
        obs_space,
        act_space,
        elements.Config(
            **config.agent,
            logdir=config.logdir,
            seed=config.seed,
            jax=config.jax,
            batch_size=config.batch_size,
            batch_length=config.batch_length,
            replay_context=config.replay_context,
            report_length=config.report_length,
            replica=config.replica,
            replicas=config.replicas,
        ),
    )


def _write_episode_csv(path: Path, rows: Sequence[DreamerEpisodeResult]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _aggregate(rows: Sequence[DreamerEpisodeResult]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for stage in TRANSFER_STAGES:
        items = [row for row in rows if row.stage_index == stage.level]
        output.append(
            {
                "level": stage.level,
                "stage": stage.name,
                "episodes": len(items),
                "successes": sum(item.success for item in items),
                "success_rate": fmean(item.success for item in items) if items else 0.0,
                "stalled": sum(item.stalled for item in items),
                "truncations": sum(item.truncation for item in items),
                "mean_primitive_steps": (
                    fmean(item.primitive_transitions for item in items) if items else 0.0
                ),
                "projection_mean_squared_distance": (
                    fmean(item.projection_mean_squared_distance for item in items)
                    if items
                    else 0.0
                ),
            }
        )
    return output


def _run_episode(
    *,
    upstream: Mapping[str, Any],
    agent: Any,
    research_seed: int,
    stage_index: int,
    scenario_seed: int,
    transition_cap: int,
    phase: str,
    mode: str,
    replay: Any | None,
    train_stream: Any | None,
    train_carry: list[Any] | None,
    train_state: _DreamerTrainState | None,
    batch_size: int,
    batch_length: int,
    train_ratio: float,
) -> DreamerEpisodeResult:
    embodied = upstream["embodied"]
    env = _DreamerPentestEnv(
        elements=upstream["elements"],
        np=upstream["np"],
        research_seed=research_seed,
        stage_index=stage_index,
        scenario_seed=scenario_seed,
        transition_cap=transition_cap,
        phase=phase,
    )
    driver = embodied.Driver([lambda: env], parallel=False)
    if mode == "train":
        if replay is None or train_stream is None or train_carry is None or train_state is None:
            raise ValueError("Dreamer training episode is missing replay/train state")
        driver.on_step(replay.add)

        batch_steps = int(batch_size) * int(batch_length)

        def trainfn(tran: Mapping[str, Any], worker: int) -> None:
            del worker
            if bool(tran["is_first"]):
                return
            train_state.real_transitions += 1
            if len(replay) < batch_steps:
                return
            desired = int(
                train_state.real_transitions * float(train_ratio) / float(batch_steps)
            )
            while train_state.gradient_updates < desired:
                batch = next(train_stream)
                train_carry[0], outs, _ = agent.train(train_carry[0], batch)
                if "replay" in outs:
                    replay.update(outs["replay"])
                train_state.gradient_updates += 1

        driver.on_step(trainfn)

    policy = lambda *args: agent.policy(*args, mode=mode)
    driver.reset(agent.init_policy)
    try:
        driver(policy, episodes=1)
    finally:
        driver.close()
    if env.result is None:
        raise RuntimeError("DreamerV3 driver ended without an episode result")
    return env.result


def run_official_dreamerv3_current_baseline(
    output_dir: str | Path,
    *,
    dreamer_root: str | Path,
    research_seed: int,
    transition_budget: int = 10_000,
    block_target: int = 512,
    train_seeds: Sequence[int] = TRANSFER_TRAIN_SEEDS,
    validation_seeds: Sequence[int] = DREAMERV3_VALIDATION_SEEDS,
    diagnostic_seeds: Sequence[int] = TRANSFER_DIAGNOSTIC_SEEDS,
    jax_platform: str = "cuda",
    train_ratio: float | None = None,
    prealloc: bool = True,
    allow_upstream_mismatch: bool = False,
) -> dict[str, Any]:
    """Train unmodified official DreamerV3 under the current AASSR protocol.

    The algorithm implementation comes from the pinned upstream checkout. This
    function supplies only the relational environment adapter, exact real-step
    accounting, and the same independent adaptive curriculum used by the current
    DQN/AASSR conditions.
    """

    if transition_budget <= 0 or block_target <= 0:
        raise ValueError("Dreamer transition budget and block target must be positive")
    train_set = set(map(int, train_seeds))
    validation_set = set(map(int, validation_seeds))
    diagnostic_set = set(map(int, diagnostic_seeds))
    if not train_set or not validation_set or not diagnostic_set:
        raise ValueError("Dreamer seed pools must be non-empty")
    if train_set & validation_set or train_set & diagnostic_set or validation_set & diagnostic_set:
        raise ValueError("Dreamer train/validation/diagnostic seed pools overlap")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    upstream = _load_official_dreamer(
        dreamer_root,
        allow_upstream_mismatch=allow_upstream_mismatch,
    )
    config = _official_config(
        upstream,
        output_dir=output,
        research_seed=research_seed,
        jax_platform=jax_platform,
        train_ratio=train_ratio,
        prealloc=prealloc,
    )
    effective_train_ratio = float(config.run.train_ratio)

    probe = _DreamerPentestEnv(
        elements=upstream["elements"],
        np=upstream["np"],
        research_seed=research_seed,
        stage_index=0,
        scenario_seed=int(train_seeds[0]),
        transition_cap=1,
        phase="probe",
    )
    agent = _make_agent(upstream, config, probe)
    dreamer_main = upstream["main"]
    replay = dreamer_main.make_replay(config, "replay")
    train_stream = iter(agent.stream(dreamer_main.make_stream(config, replay, "train")))
    train_carry = [agent.init_train(int(config.batch_size))]
    train_state = _DreamerTrainState()

    curriculum = TransferAdaptiveCurriculum()
    training_rows: list[DreamerEpisodeResult] = []
    validation_rows: list[DreamerEpisodeResult] = []
    curriculum_trace: list[dict[str, Any]] = []
    transition_total = 0
    block = 0

    while transition_total < transition_budget:
        block_used = 0
        episode = 0
        weights = curriculum.weights()
        rng = random.Random(int(research_seed) ^ (block * 0x9E3779B1))
        focus_before = curriculum.focus_level
        while block_used < block_target and transition_total < transition_budget:
            level = schedule.weighted_level(rng, weights)
            stage = TRANSFER_STAGES[level]
            scenario_seed = int(
                train_seeds[(block * 97 + episode) % len(train_seeds)]
            )
            natural_cap = max(24, stage.rate_limit + STALL_PATIENCE)
            hard_left = transition_budget - transition_total
            cap = min(natural_cap, hard_left)
            row = _run_episode(
                upstream=upstream,
                agent=agent,
                research_seed=research_seed,
                stage_index=level,
                scenario_seed=scenario_seed,
                transition_cap=cap,
                phase="train",
                mode="train",
                replay=replay,
                train_stream=train_stream,
                train_carry=train_carry,
                train_state=train_state,
                batch_size=int(config.batch_size),
                batch_length=int(config.batch_length),
                train_ratio=effective_train_ratio,
            )
            if row.primitive_transitions <= 0:
                raise RuntimeError("DreamerV3 training consumed no real transitions")
            training_rows.append(row)
            transition_total += row.primitive_transitions
            block_used += row.primitive_transitions
            episode += 1

        block_validation: list[DreamerEpisodeResult] = []
        focus_stage = TRANSFER_STAGES[curriculum.focus_level]
        eval_cap = max(24, focus_stage.rate_limit + STALL_PATIENCE)
        updates_before = train_state.gradient_updates
        for scenario_seed in validation_seeds:
            block_validation.append(
                _run_episode(
                    upstream=upstream,
                    agent=agent,
                    research_seed=research_seed,
                    stage_index=curriculum.focus_level,
                    scenario_seed=int(scenario_seed),
                    transition_cap=eval_cap,
                    phase="curriculum_validation",
                    mode="eval",
                    replay=None,
                    train_stream=None,
                    train_carry=None,
                    train_state=None,
                    batch_size=int(config.batch_size),
                    batch_length=int(config.batch_length),
                    train_ratio=effective_train_ratio,
                )
            )
        if train_state.gradient_updates != updates_before:
            raise AssertionError("DreamerV3 validation mutated training updates")
        validation_rows.extend(block_validation)
        validation_success = fmean(row.success for row in block_validation)
        movement = curriculum.observe_block(validation_success)
        curriculum_trace.append(
            {
                "block": block,
                "transition_total": transition_total,
                "block_transitions": block_used,
                "focus_before": focus_before,
                "validation_success_rate": validation_success,
                "movement": movement,
                "focus_after": curriculum.focus_level,
                "train_weights": weights,
                "gradient_updates": train_state.gradient_updates,
            }
        )
        block += 1

    if transition_total != transition_budget:
        raise AssertionError(
            f"DreamerV3 real transition budget drift: {transition_total} != {transition_budget}"
        )
    if train_state.real_transitions != transition_budget:
        raise AssertionError(
            "DreamerV3 callback real-step accounting disagrees with episode accounting"
        )

    diagnostic_rows: list[DreamerEpisodeResult] = []
    updates_before_diagnostic = train_state.gradient_updates
    for stage_index, stage in enumerate(TRANSFER_STAGES):
        cap = max(24, stage.rate_limit + STALL_PATIENCE)
        for scenario_seed in diagnostic_seeds:
            diagnostic_rows.append(
                _run_episode(
                    upstream=upstream,
                    agent=agent,
                    research_seed=research_seed,
                    stage_index=stage_index,
                    scenario_seed=int(scenario_seed),
                    transition_cap=cap,
                    phase="diagnostic",
                    mode="eval",
                    replay=None,
                    train_stream=None,
                    train_carry=None,
                    train_state=None,
                    batch_size=int(config.batch_size),
                    batch_length=int(config.batch_length),
                    train_ratio=effective_train_ratio,
                )
            )
    if train_state.gradient_updates != updates_before_diagnostic:
        raise AssertionError("DreamerV3 diagnostic mutated training updates")

    diagnostic = _aggregate(diagnostic_rows)
    _write_episode_csv(output / "training_dreamerv3_relational.csv", training_rows)
    _write_episode_csv(
        output / "curriculum_validation_dreamerv3_relational.csv",
        validation_rows,
    )
    _write_episode_csv(output / "diagnostic_dreamerv3_relational.csv", diagnostic_rows)
    (output / "curriculum_trace_dreamerv3_relational.json").write_text(
        json.dumps(curriculum_trace, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "diagnostic_dreamerv3_relational.json").write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    projection_rows = training_rows + validation_rows + diagnostic_rows
    result = {
        "condition": DREAMERV3_CONDITION,
        "baseline_version": DREAMERV3_BASELINE_VERSION,
        "protocol_version": DREAMERV3_CURRENT_PROTOCOL_VERSION,
        "official_upstream": {
            **dreamer_adapter_manifest(),
            "actual_commit": upstream["head"],
            "commit_matches_pin": upstream["head"] == DREAMERV3_UPSTREAM_COMMIT,
        },
        "official_config": {
            "preset": DREAMERV3_OFFICIAL_CONFIG,
            "model_size": "size1m",
            "batch_size": int(config.batch_size),
            "batch_length": int(config.batch_length),
            "train_ratio": effective_train_ratio,
            "imag_length": int(config.agent.imag_length),
            "jax_platform": str(jax_platform),
            "compute_dtype": str(config.jax.compute_dtype),
            "prealloc": bool(prealloc),
        },
        "research_seed": int(research_seed),
        "transitions_used": transition_total,
        "exact_budget": transition_total == transition_budget,
        "gradient_updates": train_state.gradient_updates,
        "training_successes": sum(row.success for row in training_rows),
        "training_failures": sum(row.failure for row in training_rows),
        "training_stalls": sum(row.stalled for row in training_rows),
        "training_truncations": sum(row.truncation for row in training_rows),
        "final_focus_level": curriculum.focus_level,
        "diagnostic": diagnostic,
        "diagnostic_successes": sum(item["successes"] for item in diagnostic),
        "frontier": current_frontier(diagnostic),
        "validation_learning_frozen": True,
        "diagnostic_learning_frozen": True,
        "sparse_reward": {"success": 1.0, "failure": -1.0, "otherwise": 0.0},
        "bootstrap_cut_on_episode_boundary": True,
        "projection": {
            "episodes": len(projection_rows),
            "mean_squared_distance": (
                fmean(row.projection_mean_squared_distance for row in projection_rows)
                if projection_rows
                else 0.0
            ),
            "max_squared_distance": max(
                (row.projection_max_squared_distance for row in projection_rows),
                default=0.0,
            ),
            "tie_events": sum(row.projection_tie_events for row in projection_rows),
        },
        "train_seeds": list(map(int, train_seeds)),
        "validation_seeds": list(map(int, validation_seeds)),
        "diagnostic_seeds": list(map(int, diagnostic_seeds)),
        "final_blind_consumed": False,
        "stage_manifest": stage_manifest(),
    }
    (output / "summary_dreamerv3_relational.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result