from __future__ import annotations

import math
from dataclasses import asdict
from statistics import fmean
from typing import Any, Iterable, Sequence

from .benchmark_neural_prophecy import BenchmarkGridPushCodec
from .critic_prophecy_common import (
    BenchmarkGridPushVectorCodec,
    EpisodeRecord,
    NeuralDirectProphecy,
)
from .gru_prophecy import OnlineGRUProphecy
from .neural_delta_prophecy import NeuralDeltaConfig, NeuralDeltaProphecy
from .transition_prefix_prophecy import TransitionPrefixConfig, TransitionPrefixProphecy
from .types import Prediction, StateSnapshot


def _terminal_class(state: StateSnapshot) -> str:
    if state.available_actions:
        return "active"
    if state.goal_progress >= 1.0 or "success" in state.facts:
        return "success"
    return "failure"


def _representative(predictions: Iterable[Prediction]) -> StateSnapshot:
    items = tuple(predictions)
    if not items:
        raise ValueError("prediction list must not be empty")
    return max(items, key=lambda item: (item.probability, item.source)).next_state


def _transition_kind(before: StateSnapshot, after: StateSnapshot) -> str:
    terminal = _terminal_class(after)
    if terminal != "active":
        return terminal
    return "phase_change" if before.vector[12] != after.vector[12] else "ordinary"


def evaluate_one_step(
    model: Any,
    episodes: Sequence[EpisodeRecord],
    codec: BenchmarkGridPushCodec,
) -> dict[str, Any]:
    totals = {
        "count": 0,
        "mae": 0.0,
        "raw_mae": 0.0,
        "raw_count": 0,
        "core_mae": 0.0,
        "core_vector": 0,
        "vector": 0,
        "facts": 0,
        "actions": 0,
        "terminal": 0,
        "goal": 0,
        "full": 0,
    }
    by_kind: dict[str, list[int]] = {}
    for episode in episodes:
        reset = getattr(model, "reset_sequence", None)
        if callable(reset):
            reset()
        memory = getattr(model, "initial_memory", lambda: None)()
        for item in episode.transitions:
            raw_predict = getattr(model, "predict_vector", None)
            if callable(raw_predict):
                raw = raw_predict(item.before, item.action, memory=memory)
                totals["raw_mae"] += fmean(
                    abs(a - b)
                    for a, b in zip(raw, item.after.vector, strict=True)
                )
                totals["raw_count"] += 1
            predict_step = getattr(model, "predict_step", None)
            if callable(predict_step):
                step = predict_step(
                    item.before, item.action, memory=memory, samples=1
                )
                predictions, memory = step.predictions, step.memory
            else:
                predictions = model.predict(item.before, item.action, samples=1)
            predicted = _representative(predictions)
            core_left = tuple(float(value) for value in predicted.vector)
            core_right = tuple(float(value) for value in item.after.vector)
            core_mae = fmean(
                abs(a - b)
                for a, b in zip(core_left, core_right, strict=True)
            )
            core_vector = all(
                math.isclose(a, b, abs_tol=1e-6)
                for a, b in zip(core_left, core_right, strict=True)
            )
            left, right = codec.encode(predicted), codec.encode(item.after)
            mae = fmean(abs(a - b) for a, b in zip(left, right, strict=True))
            vector = all(
                math.isclose(a, b, abs_tol=1e-6)
                for a, b in zip(left, right, strict=True)
            )
            facts = predicted.facts == item.after.facts
            actions = tuple(a.signature for a in predicted.available_actions) == tuple(
                a.signature for a in item.after.available_actions
            )
            terminal = _terminal_class(predicted) == _terminal_class(item.after)
            goal = math.isclose(
                predicted.goal_progress, item.after.goal_progress, abs_tol=1e-6
            )
            full = vector and facts and actions and terminal and goal
            totals["count"] += 1
            totals["mae"] += mae
            totals["core_mae"] += core_mae
            totals["core_vector"] += int(core_vector)
            totals["vector"] += int(vector)
            totals["facts"] += int(facts)
            totals["actions"] += int(actions)
            totals["terminal"] += int(terminal)
            totals["goal"] += int(goal)
            totals["full"] += int(full)
            bucket = by_kind.setdefault(_transition_kind(item.before, item.after), [0, 0])
            bucket[0] += int(full)
            bucket[1] += 1
    count = max(1, totals["count"])
    return {
        "count": totals["count"],
        "vector_mae": totals["mae"] / count,
        "core_vector_mae": totals["core_mae"] / count,
        "core_vector_exact_rate": totals["core_vector"] / count,
        "raw_numeric_output_mae": (
            totals["raw_mae"] / totals["raw_count"]
            if totals["raw_count"] else None
        ),
        "vector_exact_rate": totals["vector"] / count,
        "facts_exact_rate": totals["facts"] / count,
        "actions_exact_rate": totals["actions"] / count,
        "terminal_accuracy": totals["terminal"] / count,
        "goal_exact_rate": totals["goal"] / count,
        "full_state_exact_rate": totals["full"] / count,
        "transition_counts": {
            name: total
            for name, (_, total) in sorted(by_kind.items())
        },
        "full_state_exact_by_transition": {
            name: matches / max(1, total)
            for name, (matches, total) in sorted(by_kind.items())
        },
    }


def train_prophecy(model: Any, episodes: Sequence[EpisodeRecord], *, epochs: int) -> None:
    for _ in range(epochs):
        for episode in episodes:
            reset = getattr(model, "reset_sequence", None)
            if callable(reset):
                reset()
            for item in episode.transitions:
                model.learn(item.before, item.action, item.after)


def build_prophecy_models(seed: int) -> dict[str, Any]:
    codec = BenchmarkGridPushCodec()
    common = dict(
        hidden_units=128,
        replay_capacity=50_000,
        batch_size=64,
        warmup_steps=128,
        learning_rate=1e-3,
        gradient_steps_per_observation=1,
        confidence_prior=256.0,
    )
    return {
        "legacy_gru": OnlineGRUProphecy(
            state_size=16, hidden_size=64, learning_rate=0.02,
            replay_limit=2048, seed=seed,
        ),
        "neural_delta": NeuralDeltaProphecy(
            codec, config=NeuralDeltaConfig(ensemble_size=3, **common), seed=seed
        ),
        "transition_prefix": TransitionPrefixProphecy(
            codec,
            config=TransitionPrefixConfig(
                model_dim=64, attention_heads=4, layers=2,
                feedforward_dim=128, replay_capacity=50_000,
                batch_size=64, warmup_steps=128, learning_rate=1e-3,
                gradient_steps_per_observation=1,
            ),
            seed=seed,
        ),
        "ablation_direct_target": NeuralDirectProphecy(
            codec, config=NeuralDeltaConfig(ensemble_size=3, **common), seed=seed
        ),
        "ablation_single_model": NeuralDeltaProphecy(
            codec, config=NeuralDeltaConfig(ensemble_size=1, **common), seed=seed
        ),
        "ablation_no_replay": NeuralDeltaProphecy(
            codec,
            config=NeuralDeltaConfig(
                ensemble_size=3, hidden_units=128, replay_capacity=1,
                batch_size=1, warmup_steps=1, learning_rate=1e-3,
                gradient_steps_per_observation=1, confidence_prior=256.0,
            ),
            seed=seed,
        ),
        "ablation_16_value_state": NeuralDeltaProphecy(
            BenchmarkGridPushVectorCodec(),
            config=NeuralDeltaConfig(ensemble_size=3, **common),
            seed=seed,
        ),
    }


def run_prophecy_audit(
    train: Sequence[EpisodeRecord],
    seen: Sequence[EpisodeRecord],
    unseen: Sequence[EpisodeRecord],
    *,
    seed: int,
    epochs: int,
) -> dict[str, Any]:
    codec = BenchmarkGridPushCodec()
    results = {}
    for name, model in build_prophecy_models(seed).items():
        train_prophecy(model, train, epochs=epochs)
        stats = (
            asdict(model.stats())
            if callable(getattr(model, "stats", None))
            else asdict(model.training_stats)
        )
        results[name] = {
            "seen": evaluate_one_step(model, seen, codec),
            "unseen": evaluate_one_step(model, unseen, codec),
            "stats": stats,
        }
    return results
