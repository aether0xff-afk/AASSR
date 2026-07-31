from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .escape_reporting import serialize_agent_checkpoint
from .model_io import restore_agent_checkpoint
from .paper_types import AgentCheckpointParts, ExperimentPhase, PaperManifest


PAPER_RUNNER = "paper_suite"
PAPER_SCHEMA_VERSION = 1
PAPER_SUITES = {
    "autonomy",
    "ablation",
    "transfer",
    "creativity",
    "safe_application",
}
PAPER_CONDITIONS = {
    "random",
    "contextual_policy",
    "q_learning",
    "dqn",
    "prophecy_no_imagination",
    "full_aassr",
    "oracle_upper_bound",
    "novelty_search",
    "aassr_no_novelty",
    "aassr_no_imagination",
}
DEFAULT_ADAPTATION_BUDGETS = (0, 1, 4, 16, 64)
DEFAULT_STATISTICS = {
    "unit": "research_seed",
    "confidence": 0.95,
    "bootstrap_samples": 5000,
    "permutation_samples": 20000,
    "test": "paired_permutation",
    "multiple_comparisons": "holm",
}


@dataclass(frozen=True, slots=True)
class PaperPaths:
    root: Path
    raw: Path
    seed_level: Path
    statistics: Path
    tables: Path
    figures: Path
    manifests: Path

    @classmethod
    def create(cls, root: str | Path) -> PaperPaths:
        base = Path(root)
        paths = cls(
            base,
            base / "raw",
            base / "seed_level",
            base / "statistics",
            base / "tables",
            base / "figures",
            base / "manifests",
        )
        for path in (
            paths.root,
            paths.raw,
            paths.seed_level,
            paths.statistics,
            paths.tables,
            paths.figures,
            paths.manifests,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return paths


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_unique_ints(value: Any, name: str, *, minimum: int = 1) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) < minimum
        or any(type(item) is not int for item in value)
    ):
        raise ValueError(f"{name} must contain at least {minimum} integer seed(s)")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} contains duplicate seeds")
    return value


def _world_seed_sets(config: Mapping[str, Any]) -> dict[str, list[int]]:
    raw = config.get("world_seeds")
    if not isinstance(raw, Mapping):
        raise ValueError("world_seeds must be an object")
    result = {
        name: _require_unique_ints(raw.get(name), f"world_seeds.{name}")
        for name in ("train", "seen", "unseen")
    }
    for left, right in itertools.combinations(("train", "seen", "unseen"), 2):
        if set(result[left]) & set(result[right]):
            raise ValueError(
                f"{left} and {right} world seeds must be disjoint"
            )
    return result


def expand_suite_conditions(
    suite: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expand explicit conditions and an optional declarative matrix."""
    explicit = suite.get("conditions", ())
    if explicit and not isinstance(explicit, list):
        raise ValueError("suite conditions must be a list")
    result = [
        dict(item)
        for item in explicit
        if isinstance(item, Mapping)
    ]
    matrix = suite.get("condition_matrix")
    if matrix is None:
        return result
    if not isinstance(matrix, Mapping):
        raise ValueError("condition_matrix must be an object")
    base = matrix.get("base", {})
    axes = matrix.get("axes")
    template = str(matrix.get("name_template", "")).strip()
    if not isinstance(base, Mapping):
        raise ValueError("condition_matrix.base must be an object")
    if not isinstance(axes, Mapping) or not axes:
        raise ValueError("condition_matrix.axes must be a non-empty object")
    if not template:
        raise ValueError("condition_matrix.name_template is required")
    names = [str(name) for name in axes]
    values: list[list[Any]] = []
    for name in names:
        axis = axes[name]
        if not isinstance(axis, list) or not axis:
            raise ValueError(
                f"condition_matrix.axes.{name} must be a non-empty list"
            )
        values.append(list(axis))
    for combination in itertools.product(*values):
        parameters = dict(zip(names, combination, strict=True))
        try:
            condition_name = template.format(**parameters)
        except (KeyError, ValueError) as error:
            raise ValueError(
                "condition_matrix.name_template cannot be formatted"
            ) from error
        result.append({**dict(base), **parameters, "name": condition_name})
    return result


def _validate_phases(config: Mapping[str, Any]) -> None:
    requested = config.get(
        "phases",
        [
            phase.value
            for phase in (
                ExperimentPhase.TRAINING,
                ExperimentPhase.EVALUATION_SEEN,
                ExperimentPhase.EVALUATION_UNSEEN_ZERO_SHOT,
                ExperimentPhase.ADAPTATION,
                ExperimentPhase.EVALUATION_UNSEEN_ADAPTATION,
            )
        ],
    )
    if not isinstance(requested, list) or not requested:
        raise ValueError("phases must be a non-empty list")
    values = {phase.value for phase in ExperimentPhase}
    unknown = set(map(str, requested)) - values
    if unknown:
        raise ValueError(f"unknown experiment phases: {sorted(unknown)}")
    learning = config.get("phase_learning", {})
    if not isinstance(learning, Mapping):
        raise ValueError("phase_learning must be an object")
    for phase in ExperimentPhase:
        actual = bool(learning.get(phase.value, phase.permits_learning))
        if actual != phase.permits_learning:
            raise ValueError(
                f"phase_learning.{phase.value} must be {phase.permits_learning}"
            )


def _validate_budgets(config: Mapping[str, Any]) -> None:
    budgets = config.get("budgets")
    if not isinstance(budgets, Mapping):
        raise ValueError("budgets must be an object")
    for name in ("train_episodes", "eval_episodes", "real_transitions_per_episode"):
        if int(budgets.get(name, 0)) <= 0:
            raise ValueError(f"budgets.{name} must be positive")
    adaptation = budgets.get("adaptation_episodes", DEFAULT_ADAPTATION_BUDGETS)
    if (
        not isinstance(adaptation, list)
        or [int(item) for item in adaptation] != list(DEFAULT_ADAPTATION_BUDGETS)
    ):
        raise ValueError(
            "budgets.adaptation_episodes must be [0, 1, 4, 16, 64]"
        )


def _validate_statistics(config: Mapping[str, Any]) -> None:
    raw = config.get("statistics", DEFAULT_STATISTICS)
    if not isinstance(raw, Mapping):
        raise ValueError("statistics must be an object")
    expected = {
        "unit": "research_seed",
        "test": "paired_permutation",
        "multiple_comparisons": "holm",
    }
    for name, value in expected.items():
        if str(raw.get(name, value)) != value:
            raise ValueError(f"statistics.{name} must be {value}")
    confidence = float(raw.get("confidence", 0.95))
    if not 0.0 < confidence < 1.0:
        raise ValueError("statistics.confidence must be in (0, 1)")
    if int(raw.get("bootstrap_samples", 0)) <= 0:
        raise ValueError("statistics.bootstrap_samples must be positive")
    if int(raw.get("permutation_samples", 0)) <= 0:
        raise ValueError("statistics.permutation_samples must be positive")
    if (
        config.get("study_stage") == "final"
        and "statistics" not in config
    ):
        raise ValueError("final study requires frozen statistics settings")


def _validate_suites(config: Mapping[str, Any]) -> None:
    suites = config.get("suites")
    if not isinstance(suites, list) or not suites:
        raise ValueError("suites must be a non-empty list")
    names: set[str] = set()
    for suite in suites:
        if not isinstance(suite, Mapping):
            raise ValueError("each suite must be an object")
        kind = str(suite.get("kind", ""))
        if kind not in PAPER_SUITES:
            raise ValueError(f"unsupported paper suite: {kind}")
        if kind in names:
            raise ValueError(f"duplicate paper suite: {kind}")
        names.add(kind)
        conditions = expand_suite_conditions(suite)
        if conditions:
            condition_names = [
                str(item.get("name", "")) if isinstance(item, Mapping) else ""
                for item in conditions
            ]
            if any(not name for name in condition_names):
                raise ValueError(f"{kind} contains an unnamed condition")
            if len(set(condition_names)) != len(condition_names):
                raise ValueError(f"{kind} contains duplicate condition names")
            budgets = {
                int(item.get("real_transition_budget", -1))
                for item in conditions
                if isinstance(item, Mapping)
                and "real_transition_budget" in item
            }
            if len(budgets) > 1:
                raise ValueError(f"{kind} conditions use unequal interaction budgets")


def _validate_final(config: Mapping[str, Any]) -> None:
    if str(config["study_stage"]) != "final":
        return
    seeds = config["research_seeds"]
    if len(seeds) < 20:
        raise ValueError("final study requires at least 20 research seeds")
    gates = config.get("acceptance_gates")
    if not isinstance(gates, Mapping) or not all(
        bool(gates.get(name, False)) for name in ("p0", "p1", "p2", "p3")
    ):
        raise ValueError("final study requires accepted P0, P1, P2, and P3 gates")
    if not str(config.get("acceptance_gate_manifest", "")).strip():
        raise ValueError("final study requires acceptance_gate_manifest")
    suites = {str(item["kind"]) for item in config["suites"]}
    if "creativity" in suites and not str(
        config.get("frozen_creativity_rules", "")
    ).strip():
        raise ValueError("final creativity study needs frozen_creativity_rules")


def _validate_human_study(config: Mapping[str, Any]) -> None:
    human = config.get("human_study", {})
    if not isinstance(human, Mapping) or not human.get("merge_enabled", False):
        return
    for field in ("approval_id", "dataset_version", "dataset_dir"):
        if not str(human.get(field, "")).strip():
            raise ValueError(f"human merge requires human_study.{field}")
    if int(human.get("minimum_raters", 2)) < 2:
        raise ValueError("human merge requires at least two raters")


def _validate_safe_application(config: Mapping[str, Any]) -> None:
    suites = {str(item["kind"]) for item in config["suites"]}
    if "safe_application" not in suites:
        return
    safe = config.get("safe_application")
    if not isinstance(safe, Mapping):
        raise ValueError("safe_application settings are required")
    if not bool(safe.get("opt_in", False)):
        raise ValueError("safe_application must be explicitly opt-in")
    if not bool(safe.get("internal_network", False)):
        raise ValueError("safe_application requires an internal-only network")
    allowed = safe.get("allowed_hosts")
    if not isinstance(allowed, list) or not allowed:
        raise ValueError("safe_application.allowed_hosts must be non-empty")
    if any(
        host in {"0.0.0.0", "::", "*"} or "." in str(host).strip(".")
        for host in allowed
    ):
        raise ValueError("safe_application hosts must be Compose-internal names")


def validate_paper_config(config: Mapping[str, Any]) -> None:
    if config.get("runner") != PAPER_RUNNER:
        raise ValueError(f"paper config must set runner to {PAPER_RUNNER}")
    if int(config.get("schema_version", PAPER_SCHEMA_VERSION)) != PAPER_SCHEMA_VERSION:
        raise ValueError("unsupported paper config schema_version")
    if not str(config.get("name", "")).strip():
        raise ValueError("paper config needs a name")
    if not str(config.get("protocol_version", "")).strip():
        raise ValueError("paper config needs protocol_version")
    if config.get("study_stage") not in {"pilot", "final"}:
        raise ValueError("study_stage must be pilot or final")
    minimum = 5 if config.get("study_stage") == "pilot" else 20
    _require_unique_ints(config.get("research_seeds"), "research_seeds", minimum=minimum)
    _world_seed_sets(config)
    _validate_phases(config)
    _validate_budgets(config)
    _validate_statistics(config)
    _validate_suites(config)
    _validate_final(config)
    _validate_human_study(config)
    _validate_safe_application(config)


def load_paper_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_paper_config(payload)
    return payload


def capture_checkpoint_parts(
    agent: object,
    *,
    episode: int = 0,
    effect_representation: Mapping[str, Any] | None = None,
) -> AgentCheckpointParts:
    payload = serialize_agent_checkpoint(agent, episode=episode)
    return AgentCheckpointParts(
        policy=copy.deepcopy(payload.get("policy", {})),
        prophecy=copy.deepcopy(payload.get("prophecy", {})),
        holdout=copy.deepcopy(payload.get("holdout", {})),
        effect_representation=copy.deepcopy(effect_representation or {}),
        counters={
            "transition_index": payload.get("transition_index", 0),
            "decision_index": payload.get("decision_index", 0),
            "effect_novelty_motifs": copy.deepcopy(
                payload.get("effect_novelty_motifs", ())
            ),
        },
        random_state=copy.deepcopy(payload.get("random_state")),
    )


def restore_checkpoint_parts(
    agent: object,
    parts: AgentCheckpointParts,
    *,
    retain_policy: bool,
    retain_prophecy: bool,
    retain_holdout: bool,
    retain_effect_representation: bool = False,
) -> dict[str, Any]:
    fresh = serialize_agent_checkpoint(agent, episode=0)
    payload = {
        **fresh,
        "policy": copy.deepcopy(parts.policy if retain_policy else fresh["policy"]),
        "prophecy": copy.deepcopy(
            parts.prophecy if retain_prophecy else fresh["prophecy"]
        ),
        "holdout": copy.deepcopy(parts.holdout if retain_holdout else fresh["holdout"]),
        "transition_index": int(parts.counters.get("transition_index", 0)),
        "decision_index": int(parts.counters.get("decision_index", 0)),
        "random_state": copy.deepcopy(parts.random_state),
        "effect_novelty_motifs": copy.deepcopy(
            parts.counters.get("effect_novelty_motifs", ())
            if retain_effect_representation
            else fresh.get("effect_novelty_motifs", ())
        ),
    }
    restore_agent_checkpoint(agent, payload)
    return (
        copy.deepcopy(dict(parts.effect_representation))
        if parts.effect_representation
        else {}
    )


def checkpoint_fingerprint(
    agent: object, *, effect_representation: Mapping[str, Any] | None = None
) -> str:
    payload = serialize_agent_checkpoint(agent, episode=0)
    learned = {
        "policy": payload.get("policy", {}),
        "prophecy": payload.get("prophecy", {}),
        "holdout": payload.get("holdout", {}),
        "transition_index": payload.get("transition_index", 0),
        "decision_index": payload.get("decision_index", 0),
        "random_state": payload.get("random_state"),
        "effect_representation": effect_representation or {},
        "effect_novelty_motifs": payload.get(
            "effect_novelty_motifs", ()
        ),
    }
    return sha256_json(learned)


def assert_frozen(before: str, after: str, phase: ExperimentPhase) -> None:
    if phase.permits_learning:
        return
    if before != after:
        raise RuntimeError(f"learning state mutated during frozen phase {phase.value}")


def _git_sha(workdir: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def runtime_metadata() -> tuple[dict[str, str], dict[str, Any]]:
    software = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    for distribution, label in (
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
    ):
        try:
            software[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            software[label] = "not-installed"
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Failed to initialize NumPy.*"
            )
            import torch

        software["torch"] = str(torch.__version__)
        software["cuda"] = str(torch.version.cuda or "none")
        cuda_available = bool(torch.cuda.is_available())
        gpu = torch.cuda.get_device_name(0) if cuda_available else ""
    except (ImportError, RuntimeError):
        software["torch"] = "not-installed"
        software["cuda"] = "none"
        cuda_available = False
        gpu = ""
    total_ram_bytes: int | None = None
    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.length = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(
                ctypes.byref(status)
            ):
                total_ram_bytes = int(status.total_physical)
        except (AttributeError, OSError):
            pass
    else:
        try:
            total_ram_bytes = int(
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            )
        except (AttributeError, OSError, ValueError):
            pass
    hardware = {
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cuda_available": cuda_available,
        "gpu": gpu,
        "total_ram_bytes": total_ram_bytes,
    }
    return software, hardware


def build_manifest(
    config: Mapping[str, Any],
    *,
    started_at_utc: str,
    completed_at_utc: str,
    failed_runs: Sequence[Mapping[str, Any]] = (),
    excluded_runs: Sequence[Mapping[str, Any]] = (),
    workdir: str | Path = ".",
) -> PaperManifest:
    software, hardware = runtime_metadata()
    worlds = _world_seed_sets(config)
    human = config.get("human_study", {})
    execution = config.get("execution", {})
    lock_hashes: dict[str, str] = {}
    for field in ("acceptance_gate_manifest", "frozen_creativity_rules"):
        value = str(config.get(field, "")).strip()
        if not value:
            continue
        source = Path(value)
        if not source.is_absolute():
            source = Path(workdir) / source
        if source.exists():
            lock_hashes[field] = hashlib.sha256(source.read_bytes()).hexdigest()
    return PaperManifest(
        protocol_version=str(config["protocol_version"]),
        study_stage=str(config["study_stage"]),
        git_commit_sha=_git_sha(Path(workdir)),
        config_sha256=sha256_json(config),
        research_seeds=tuple(int(item) for item in config["research_seeds"]),
        world_seeds={
            key: tuple(int(item) for item in values)
            for key, values in worlds.items()
        },
        phase_definitions={
            phase.value: phase.permits_learning for phase in ExperimentPhase
        },
        started_at_utc=started_at_utc,
        completed_at_utc=completed_at_utc,
        software=software,
        hardware=hardware,
        execution=dict(execution) if isinstance(execution, Mapping) else {},
        failed_runs=tuple(dict(item) for item in failed_runs),
        excluded_runs=tuple(dict(item) for item in excluded_runs),
        human_dataset_version=(
            str(human.get("dataset_version", ""))
            if isinstance(human, Mapping)
            else ""
        ),
        human_approval_id=(
            str(human.get("approval_id", ""))
            if isinstance(human, Mapping)
            else ""
        ),
        protocol_locks=lock_hashes,
    )


def planned_paper_run_count(config: Mapping[str, Any]) -> int:
    validate_paper_config(config)
    budgets = config["budgets"]
    train = int(budgets["train_episodes"])
    evaluate = int(budgets["eval_episodes"])
    research = len(config["research_seeds"])
    worlds = _world_seed_sets(config)
    total = 0
    for suite in config["suites"]:
        kind = str(suite["kind"])
        conditions = max(1, len(expand_suite_conditions(suite)))
        if kind in {"autonomy", "ablation"}:
            environments = len(
                suite.get(
                    "environments",
                    suite.get("lengths", [4, 6, 8]),
                )
            )
            total += (
                research
                * environments
                * conditions
                * (train + 2 * evaluate)
            )
        elif kind == "transfer":
            adaptation = budgets.get(
                "adaptation_episodes", DEFAULT_ADAPTATION_BUDGETS
            )
            total += research * (
                train
                + conditions
                * len(worlds["unseen"])
                * sum(int(item) + evaluate for item in adaptation)
            )
        elif kind == "creativity":
            episodes = int(suite.get("episodes", train))
            total += research * conditions * (episodes + 3)
        else:
            episodes = int(suite.get("episodes", evaluate))
            total += research * conditions * episodes
    return total


def write_final_manifest(
    path: str | Path, manifest: PaperManifest
) -> tuple[Path, Path]:
    destination = Path(path)
    payload = manifest.to_dict()
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    digest_path = destination.with_suffix(destination.suffix + ".sha256")
    digest_path.write_text(
        hashlib.sha256(destination.read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )
    return destination, digest_path
