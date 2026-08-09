from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import time
from types import MethodType
from typing import Any, Iterator, Mapping


HOT_PATH_CATEGORIES: tuple[str, ...] = (
    "policy_selection",
    "aseq",
    "evaluator_pre_prediction",
    "evaluator_post_prediction",
    "holdout_before",
    "environment_step",
    "prophecy_learn",
    "holdout_after",
    "calibration",
    "dqn_observe_train",
    "critic_episode_learning",
    "information_value_feature_memory",
)


@dataclass(slots=True)
class _Timing:
    total_seconds: float = 0.0
    calls: int = 0


class CurrentHotPathProfiler:
    """Host-observed current-runtime timing with no accelerator synchronization.

    Timings are inclusive method-call durations. In particular, calibration is
    reported separately while also remaining part of the evaluator prediction
    call that triggered it. This keeps the profiler observational: it neither
    changes call order nor inserts a CUDA synchronization.
    """

    def __init__(self) -> None:
        self._phase_stack: list[str] = []
        self._totals: dict[str, _Timing] = {
            category: _Timing() for category in HOT_PATH_CATEGORIES
        }
        self._by_phase: dict[str, dict[str, _Timing]] = defaultdict(
            lambda: {
                category: _Timing() for category in HOT_PATH_CATEGORIES
            }
        )
        self._evaluator_frames: list[dict[str, int]] = []

    @property
    def phase_name(self) -> str:
        return self._phase_stack[-1] if self._phase_stack else "unscoped"

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        self._phase_stack.append(str(name))
        try:
            yield
        finally:
            self._phase_stack.pop()

    @contextmanager
    def evaluator_call(self) -> Iterator[None]:
        self._evaluator_frames.append({"predictions": 0, "holdouts": 0})
        try:
            yield
        finally:
            self._evaluator_frames.pop()

    def next_prediction_category(self) -> str:
        if not self._evaluator_frames:
            return "evaluator_pre_prediction"
        frame = self._evaluator_frames[-1]
        index = frame["predictions"]
        frame["predictions"] = index + 1
        # AdvancedTransitionEvaluator currently makes two pre-update predictions
        # (without and with Knowledge) followed by its post-update prediction.
        return (
            "evaluator_pre_prediction"
            if index < 2
            else "evaluator_post_prediction"
        )

    def next_holdout_category(self) -> str:
        if not self._evaluator_frames:
            return "holdout_before"
        frame = self._evaluator_frames[-1]
        index = frame["holdouts"]
        frame["holdouts"] = index + 1
        return "holdout_before" if index == 0 else "holdout_after"

    def record(self, category: str, elapsed: float) -> None:
        if category not in self._totals:
            raise KeyError(f"unknown current hot-path category: {category}")
        total = self._totals[category]
        total.total_seconds += float(elapsed)
        total.calls += 1
        phase = self._by_phase[self.phase_name][category]
        phase.total_seconds += float(elapsed)
        phase.calls += 1

    @staticmethod
    def _row(timing: _Timing) -> dict[str, float | int]:
        return {
            "total_seconds": timing.total_seconds,
            "calls": timing.calls,
            "seconds_per_call": (
                timing.total_seconds / timing.calls if timing.calls else 0.0
            ),
        }

    def snapshot(
        self,
        *,
        training_transitions: int,
        validator_runtime: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        transitions = int(training_transitions)
        training = self._by_phase.get("training", {})
        categories: dict[str, dict[str, float | int]] = {}
        for category in HOT_PATH_CATEGORIES:
            row = self._row(self._totals[category])
            training_timing = training.get(category, _Timing())
            row.update(
                {
                    "training_total_seconds": training_timing.total_seconds,
                    "training_calls": training_timing.calls,
                    "training_seconds_per_transition": (
                        training_timing.total_seconds / transitions
                        if transitions > 0
                        else 0.0
                    ),
                    "training_calls_per_transition": (
                        training_timing.calls / transitions
                        if transitions > 0
                        else 0.0
                    ),
                }
            )
            categories[category] = row

        phases = {
            phase: {
                category: self._row(rows[category])
                for category in HOT_PATH_CATEGORIES
            }
            for phase, rows in sorted(self._by_phase.items())
        }
        return {
            "enabled": True,
            "clock": "time.perf_counter",
            "host_observed": True,
            "cuda_synchronized": False,
            "timings_are_inclusive": True,
            "nested_timing_note": (
                "calibration is also included in the prediction call that "
                "triggered it; no category subtraction or CUDA synchronization"
            ),
            "training_transitions": transitions,
            "categories": categories,
            "phases": phases,
            "validator_runtime_diagnostics": dict(validator_runtime or {}),
        }


class _TimedEnvironmentProxy:
    __slots__ = ("_environment", "_profiler")

    def __init__(self, environment: object, profiler: CurrentHotPathProfiler) -> None:
        self._environment = environment
        self._profiler = profiler

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    def step(self, action: object) -> Any:
        started = time.perf_counter()
        try:
            return self._environment.step(action)
        finally:
            self._profiler.record(
                "environment_step",
                time.perf_counter() - started,
            )


def _install_timed_method(
    target: object,
    name: str,
    profiler: CurrentHotPathProfiler,
    category: str,
) -> None:
    original = getattr(target, name)

    def timed(_self: object, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            profiler.record(category, time.perf_counter() - started)

    setattr(target, name, MethodType(timed, target))


def install_current_hot_path_profiler(agent: object) -> CurrentHotPathProfiler:
    """Install opt-in wrappers on one current-generation agent instance."""

    existing = getattr(agent, "current_hot_path_profiler", None)
    if isinstance(existing, CurrentHotPathProfiler):
        return existing

    profiler = CurrentHotPathProfiler()
    agent.current_hot_path_profiler = profiler

    _install_timed_method(agent.policy, "select", profiler, "policy_selection")
    _install_timed_method(agent.aseq, "filter_state", profiler, "aseq")
    _install_timed_method(agent.aseq, "observe", profiler, "aseq")
    _install_timed_method(
        agent.calibrated_prophecy,
        "_calibration",
        profiler,
        "calibration",
    )
    _install_timed_method(agent.prophecy, "learn", profiler, "prophecy_learn")
    _install_timed_method(agent.dqn, "observe", profiler, "dqn_observe_train")
    _install_timed_method(
        agent.critic,
        "observe_episode",
        profiler,
        "critic_episode_learning",
    )

    information_methods = (
        (agent, "_observe_information_features"),
        (agent, "_learn_feature_use"),
        (agent.evaluator.predictor, "predict"),
        (agent.evaluator.predictor, "learn"),
        (agent.evaluator.unlock_estimator, "estimate"),
        (agent.evaluator.unlock_estimator, "observe_future_return"),
        (agent.policy, "observe_information_return"),
    )
    for target, name in information_methods:
        _install_timed_method(
            target,
            name,
            profiler,
            "information_value_feature_memory",
        )

    evaluator = agent.evaluator
    original_predict = evaluator._predict

    def timed_predict(_self: object, *args: Any, **kwargs: Any) -> Any:
        category = profiler.next_prediction_category()
        started = time.perf_counter()
        try:
            return original_predict(*args, **kwargs)
        finally:
            profiler.record(category, time.perf_counter() - started)

    evaluator._predict = MethodType(timed_predict, evaluator)

    validator = evaluator.validator
    original_validate = validator.evaluate

    def timed_validate(_self: object, *args: Any, **kwargs: Any) -> Any:
        category = profiler.next_holdout_category()
        started = time.perf_counter()
        try:
            return original_validate(*args, **kwargs)
        finally:
            profiler.record(category, time.perf_counter() - started)

    validator.evaluate = MethodType(timed_validate, validator)

    original_execute = evaluator.execute

    def timed_execute(
        _self: object,
        environment: object,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        with profiler.evaluator_call():
            return original_execute(
                _TimedEnvironmentProxy(environment, profiler),
                *args,
                **kwargs,
            )

    evaluator.execute = MethodType(timed_execute, evaluator)

    original_execute_frozen = agent._execute_frozen

    def timed_execute_frozen(
        _self: object,
        environment: object,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return original_execute_frozen(
            _TimedEnvironmentProxy(environment, profiler),
            *args,
            **kwargs,
        )

    agent._execute_frozen = MethodType(timed_execute_frozen, agent)
    agent.current_hot_path_profile_enabled = True
    return profiler


def current_hot_path_phase(agent: object, name: str) -> Any:
    profiler = getattr(agent, "current_hot_path_profiler", None)
    if isinstance(profiler, CurrentHotPathProfiler):
        return profiler.phase(name)
    return nullcontext()


def current_hot_path_snapshot(
    agent: object,
    *,
    training_transitions: int,
) -> dict[str, Any]:
    profiler = getattr(agent, "current_hot_path_profiler", None)
    if not isinstance(profiler, CurrentHotPathProfiler):
        return {"enabled": False}
    runtime_diagnostics = getattr(agent.evaluator.validator, "runtime_diagnostics", None)
    validator = runtime_diagnostics() if callable(runtime_diagnostics) else {}
    return profiler.snapshot(
        training_transitions=int(training_transitions),
        validator_runtime=validator,
    )
