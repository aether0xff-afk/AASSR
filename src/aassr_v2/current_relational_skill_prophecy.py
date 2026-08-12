from __future__ import annotations

from statistics import fmean
from typing import Any

from .current_agent import CurrentSkillProphecy
from .current_relational_model import RelationalPrediction
from .knowledge import KnowledgeStore
from .skills import SKILL_VERB
from .types import Action, Prediction, StateSnapshot


SKILL_COMPLETE_OUTCOME_LIMIT = 64
_UNRESOLVED_TAIL_SOURCE = "unresolved-tail"


class RelationalStochasticSkillProphecy(CurrentSkillProphecy):
    """Relational Skill rollout with explicit stochastic outcome mass.

    When the primitive world model declares a complete outcome distribution, a
    Skill may not silently turn a top-k beam into a conditional probability
    distribution. Up to ``SKILL_COMPLETE_OUTCOME_LIMIT`` macro branches are kept
    with their original mass. If combinatorics exceed that bound, discarded mass
    is represented explicitly as a zero-reliability unresolved tail. Any Skill
    with unresolved tail therefore fails closed at the reliability gate instead
    of becoming an overconfident Imagination intervention.
    """

    name = "current-relational-stochastic-skill-prophecy-v3-tail-safe"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.skill_complete_rollouts = 0
        self.skill_unresolved_rollouts = 0
        self.skill_unresolved_mass_total = 0.0

    def _base_context_predictions(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        knowledge: KnowledgeStore,
        samples: int,
    ) -> tuple[Prediction, ...]:
        contextual = getattr(self.base, "predict_with_context", None)
        if callable(contextual):
            return tuple(
                contextual(
                    state,
                    action,
                    knowledge=knowledge,
                    samples=samples,
                )
            )
        return tuple(self.base.predict(state, action, samples=samples))

    def _skill_predictions(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        knowledge: KnowledgeStore | None,
        samples: int,
    ) -> tuple[Prediction, ...]:
        skill_id = str(action.target)
        complete_base = bool(
            getattr(self.base, "complete_outcome_distribution", False)
        )
        beam_width = (
            SKILL_COMPLETE_OUTCOME_LIMIT
            if complete_base
            else max(1, int(samples))
        )
        branches: list[tuple[StateSnapshot, float, float]] = [
            (state, 1.0, 1.0)
        ]
        unresolved_mass = 0.0

        for index in range(self.library.template_length(skill_id)):
            candidates: list[tuple[StateSnapshot, float, float]] = []
            for current, reliability, mass in branches:
                primitive = self.library.resolve_primitive(
                    skill_id,
                    index,
                    current,
                )
                if primitive is None:
                    candidates.append((current, 0.0, mass))
                    continue

                if knowledge is not None:
                    predictions = self._base_context_predictions(
                        current,
                        primitive,
                        knowledge=knowledge,
                        samples=beam_width,
                    )
                else:
                    predictions = tuple(
                        self.base.predict(
                            current,
                            primitive,
                            samples=beam_width,
                        )
                    )
                if not predictions:
                    candidates.append((current, 0.0, mass))
                    continue

                fallback_mass = 1.0 / len(predictions)
                for prediction in predictions:
                    candidates.append(
                        (
                            prediction.next_state,
                            reliability * float(prediction.probability),
                            mass
                            * max(
                                0.0,
                                float(
                                    getattr(
                                        prediction,
                                        "outcome_probability",
                                        fallback_mass,
                                    )
                                ),
                            ),
                        )
                    )

            if not candidates:
                return (
                    RelationalPrediction(
                        self.library.augment_state(state),
                        0.0,
                        source=f"{self.name}:unavailable",
                        outcome_probability=1.0,
                    ),
                )
            candidates.sort(
                key=lambda item: (item[2], item[1]),
                reverse=True,
            )

            if complete_base:
                retained = candidates[:beam_width]
                dropped = candidates[beam_width:]
                unresolved_mass += sum(max(0.0, item[2]) for item in dropped)
                # Preserve original probability mass. The retained branch masses
                # intentionally sum to <=1 when unresolved tail exists.
                branches = retained
            else:
                branches = candidates[:beam_width]
                retained_mass = sum(item[2] for item in branches)
                if retained_mass > 0.0:
                    branches = [
                        (current, reliability, mass / retained_mass)
                        for current, reliability, mass in branches
                    ]
                else:
                    uniform = 1.0 / len(branches)
                    branches = [
                        (current, reliability, uniform)
                        for current, reliability, _ in branches
                    ]

        rows = [
            RelationalPrediction(
                self.library.augment_state(current),
                max(0.0, min(1.0, reliability)),
                source=f"{self.name}:outcome-{index}",
                outcome_probability=max(0.0, min(1.0, mass)),
            )
            for index, (current, reliability, mass) in enumerate(branches)
        ]
        if complete_base:
            resolved_mass = sum(item.outcome_probability for item in rows)
            # Numerical drift should not invent or delete probability mass.
            residual = max(0.0, 1.0 - resolved_mass)
            unresolved_mass = max(unresolved_mass, residual)
            if unresolved_mass > 1e-12:
                rows.append(
                    RelationalPrediction(
                        self.library.augment_state(state),
                        0.0,
                        source=f"{self.name}:{_UNRESOLVED_TAIL_SOURCE}",
                        outcome_probability=min(1.0, unresolved_mass),
                    )
                )
                total = sum(item.outcome_probability for item in rows)
                if abs(total - 1.0) > 1e-8:
                    # Only numerical correction is allowed; never condition on the
                    # retained branches by dropping the explicit unresolved tail.
                    correction = max(0.0, 1.0 - (total - rows[-1].outcome_probability))
                    tail = rows[-1]
                    rows[-1] = RelationalPrediction(
                        tail.next_state,
                        0.0,
                        source=tail.source,
                        outcome_probability=min(1.0, correction),
                    )
                self.skill_unresolved_rollouts += 1
                self.skill_unresolved_mass_total += float(
                    rows[-1].outcome_probability
                )
            else:
                self.skill_complete_rollouts += 1
        return tuple(rows)

    @staticmethod
    def _has_unresolved_tail(predictions: tuple[Prediction, ...]) -> bool:
        return any(
            str(prediction.source).endswith(_UNRESOLVED_TAIL_SOURCE)
            for prediction in predictions
        )

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        if action.verb_name != SKILL_VERB:
            return super().confidence(state, action)
        predictions = self._skill_predictions(
            state,
            action,
            knowledge=self.knowledge,
            samples=3,
        )
        if not predictions or self._has_unresolved_tail(predictions):
            return 0.0
        weighted = []
        for prediction in predictions:
            mass = float(getattr(prediction, "outcome_probability", 0.0))
            weighted.append(mass * float(prediction.probability))
        total_mass = sum(
            float(getattr(prediction, "outcome_probability", 0.0))
            for prediction in predictions
        )
        if total_mass <= 0.0:
            return max(
                0.0,
                min(1.0, fmean(float(item.probability) for item in predictions)),
            )
        return max(0.0, min(1.0, sum(weighted) / total_mass))

    def diagnostics(self) -> dict[str, Any]:
        output = super().diagnostics()
        return {
            **output,
            "stochastic_skill_outcomes": 1,
            "skill_outcome_beam": "complete-up-to-bounded-tail-safe-limit",
            "skill_complete_outcome_limit": SKILL_COMPLETE_OUTCOME_LIMIT,
            "skill_complete_rollouts": self.skill_complete_rollouts,
            "skill_unresolved_rollouts": self.skill_unresolved_rollouts,
            "skill_unresolved_mass_total": self.skill_unresolved_mass_total,
            "skill_unresolved_tail_fail_closed": 1,
            "skill_reliability_outcome_mass_separate": 1,
            "skill_confidence_stochastic": 1,
            "skill_context_path_defined": 1,
        }
