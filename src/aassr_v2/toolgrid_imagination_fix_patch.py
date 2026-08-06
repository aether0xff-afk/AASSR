from __future__ import annotations

from dataclasses import replace
from statistics import fmean
from typing import Any, Sequence

from . import toolgrid_factorial_masked as env


base = env.base
_ORIGINAL_CALIBRATED_PROPHECY = base.ToolGridCalibratedProphecy
_ORIGINAL_HYBRID_AGENT = base.ToolGridHybridAgent
_ORIGINAL_NEURAL_DELTA_PROPHECY = base.NeuralDeltaProphecy
_ORIGINAL_TOOLGRID_CODEC = base.ToolGridCodec


def _terminal_class(state: Any) -> int:
    if state.available_actions:
        return 0
    if state.goal_progress >= 1.0 or "success" in state.facts:
        return 1
    return 2


def _is_tool_decision(state: Any) -> bool:
    return bool(state.available_actions) and all(
        action.signature.startswith("tool_")
        for action in state.available_actions
    )


class CategoricalToolGridCodec(_ORIGINAL_TOOLGRID_CODEC):
    """Encode the required tool identity categorically for Prophecy only.

    The environment and critic retain the original raw observation. This codec
    replaces the single normalized tool-ID scalar with a one-hot category in the
    neural world model, then decodes predictions back to the frozen raw schema.
    It exposes no transition rule or correct-action relation.
    """

    @property
    def dimension(self) -> int:
        return env.TOOLGRID_STATE_SIZE - 1 + (self.action_count - 4)

    def encode(self, state: Any) -> tuple[float, ...]:
        raw = list(env.encode_toolgrid_state(state))
        tool_count = self.action_count - 4
        tool = min(
            tool_count - 1,
            max(0, int(round(raw[6] * (base.MAX_TOOL_COUNT - 1)))),
        )
        category = [0.0] * tool_count
        category[tool] = 1.0
        return tuple(raw[:6] + category + raw[7:])

    def decode(
        self,
        encoded: Sequence[float],
        *,
        scaffold: Any,
        terminal_class: int,
        source: str,
    ) -> Any:
        if len(encoded) != self.dimension:
            raise ValueError("categorical ToolGrid neural state has an unexpected size")
        tool_count = self.action_count - 4
        category = tuple(float(value) for value in encoded[6 : 6 + tool_count])
        tool = max(range(tool_count), key=lambda index: category[index])
        raw = (
            list(encoded[:6])
            + [tool / float(base.MAX_TOOL_COUNT - 1)]
            + list(encoded[6 + tool_count :])
        )
        return _ORIGINAL_TOOLGRID_CODEC(self.action_count).decode(
            raw,
            scaffold=scaffold,
            terminal_class=terminal_class,
            source=source,
        )


class EnumeratedActionNeuralDeltaProphecy(_ORIGINAL_NEURAL_DELTA_PROPHECY):
    """Represent the fixed action vocabulary with one-hot identity features.

    The base model hashes opaque action signatures into signed feature buckets.
    That is schema-free, but the eight-tool cell showed an action-identity
    bottleneck. One-hot identity remains rule-free: it says only which action was
    selected, not what that action does or which tool is correct.
    """

    def __init__(self, codec: Any, *args: Any, **kwargs: Any) -> None:
        actions = base.build_actions(codec.action_count)
        self._action_index = {
            action.signature: index for index, action in enumerate(actions)
        }
        super().__init__(codec, *args, **kwargs)
        if len(self._action_index) > self.config.action_feature_size:
            raise ValueError("action feature size is smaller than ToolGrid vocabulary")

    def _action_features(self, action: Any) -> tuple[float, ...]:
        values = [0.0] * self.config.action_feature_size
        values[self._action_index[action.signature]] = 1.0
        return tuple(values)


class OutcomeAwareCalibratedProphecy(_ORIGINAL_CALIBRATED_PROPHECY):
    """Fix sparse-action calibration and distinguish success from failure.

    The original cache stored a zero calibration before an action reached
    ``minimum_count`` and did not refresh that entry until 32 action-specific
    holdout samples existed. Sparse tool actions therefore remained at zero
    confidence even after becoming calibratable. This implementation does not
    cache the pre-ready state and starts a fresh bucket exactly at readiness.

    Terminal validation also compares nonterminal/success/failure classes rather
    than only checking whether any next actions exist. In ToolGrid both a correct
    tool and a wrong tool terminate, but only the correct one succeeds.
    """

    def _calibration(self, action: Any) -> float:
        items = [
            item
            for item in getattr(self.holdout, "_items", ())
            if item.action.signature == action.signature
        ]
        if len(items) < self.minimum_count:
            return 0.0

        bucket = (len(items) - self.minimum_count) // self.refresh_stride
        key = (bucket, action.signature)
        if key in self._cache:
            return self._cache[key]

        selected = items[-self.evaluation_limit :]
        scores: list[float] = []
        for item in selected:
            prediction = self.base.predict(item.before, item.action, samples=1)[0]
            predicted = prediction.next_state
            vector_error = fmean(
                abs(left - right)
                for left, right in zip(predicted.vector, item.after.vector, strict=True)
            )
            terminal_match = _terminal_class(predicted) == _terminal_class(item.after)
            available_match = {
                candidate.signature for candidate in predicted.available_actions
            } == {candidate.signature for candidate in item.after.available_actions}
            structural = 1.0 if available_match else 0.75
            scores.append(
                max(0.0, 1.0 - vector_error)
                * (1.0 if terminal_match else 0.0)
                * structural
            )

        value = (fmean(scores) if scores else 0.0) ** self.calibration_power
        value = max(0.0, min(1.0, value))
        self._cache[key] = value
        return value


class BalancedToolReplayHybridAgent(_ORIGINAL_HYBRID_AGENT):
    """Balance sparse tool transitions without contaminating the holdout split."""

    def observe(self, before: Any, action: Any, outcome: Any) -> None:
        observations_before = self.base_prophecy.observations
        super().observe(before, action, outcome)
        trained = self.base_prophecy.observations > observations_before
        if not trained or not action.signature.startswith("tool_"):
            return

        tool_count = max(1, len(self.actions) - 4)
        # One real observation was already inserted by AutonomousLearningAgent.
        # Repeating only train-split observations leaves frozen holdout examples
        # untouched while preventing navigation from dominating neural replay.
        for _ in range(tool_count - 1):
            self.base_prophecy.learn(before, action, outcome.snapshot)


class ToolDecisionGateHybridAgent(BalancedToolReplayHybridAgent):
    """Diagnostic gate: allow interventions only at semantic tool decisions.

    This is intentionally a benchmark diagnostic, not the proposed general
    architecture. It tests whether fixing Prophecy makes Imagination useful at
    the branching point while removing the known navigation confounder.
    """

    diagnostic_tool_coverage_floor = 0.35

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        original_select = self.agent.select_action

        def gated_internal_select(
            state: Any,
            *,
            episode: int,
            explore: bool,
        ) -> Any:
            original = self.agent.config
            if _is_tool_decision(state):
                if not original.use_imagination:
                    return original_select(state, episode=episode, explore=explore)
                self.agent.config = replace(
                    original,
                    imagination_minimum_coverage=self.diagnostic_tool_coverage_floor,
                )
                try:
                    return original_select(state, episode=episode, explore=explore)
                finally:
                    self.agent.config = original
            if not original.use_imagination:
                return original_select(state, episode=episode, explore=explore)
            self.agent.config = replace(original, use_imagination=False)
            try:
                return original_select(state, episode=episode, explore=explore)
            finally:
                self.agent.config = original

        # The paired diagnostic deliberately calls the internal autonomous agent
        # directly, so install the gate there as well as on this adapter.
        self.agent.select_action = gated_internal_select

    def select_action(
        self,
        state: Any,
        *,
        episode: int,
        training: bool,
    ) -> Any:
        return super().select_action(state, episode=episode, training=training)


def install_toolgrid_imagination_fix(strategy: str) -> None:
    if strategy not in {
        "baseline",
        "calibration_fix",
        "balanced_tool_replay",
        "enumerated_action_replay",
        "categorical_tool_replay",
        "tool_decision_gate",
    }:
        raise ValueError(f"unknown ToolGrid debug strategy: {strategy}")

    base.ToolGridCalibratedProphecy = _ORIGINAL_CALIBRATED_PROPHECY
    base.ToolGridHybridAgent = _ORIGINAL_HYBRID_AGENT
    base.NeuralDeltaProphecy = _ORIGINAL_NEURAL_DELTA_PROPHECY
    base.ToolGridCodec = _ORIGINAL_TOOLGRID_CODEC
    if strategy == "baseline":
        return

    base.ToolGridCalibratedProphecy = OutcomeAwareCalibratedProphecy
    if strategy == "balanced_tool_replay":
        base.ToolGridHybridAgent = BalancedToolReplayHybridAgent
    elif strategy == "enumerated_action_replay":
        base.NeuralDeltaProphecy = EnumeratedActionNeuralDeltaProphecy
        base.ToolGridHybridAgent = BalancedToolReplayHybridAgent
    elif strategy == "categorical_tool_replay":
        base.ToolGridCodec = CategoricalToolGridCodec
        base.NeuralDeltaProphecy = EnumeratedActionNeuralDeltaProphecy
        base.ToolGridHybridAgent = BalancedToolReplayHybridAgent
    elif strategy == "tool_decision_gate":
        base.ToolGridCodec = CategoricalToolGridCodec
        base.NeuralDeltaProphecy = EnumeratedActionNeuralDeltaProphecy
        base.ToolGridHybridAgent = ToolDecisionGateHybridAgent
