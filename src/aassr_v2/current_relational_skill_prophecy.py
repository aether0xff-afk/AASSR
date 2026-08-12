from __future__ import annotations

from statistics import fmean
from typing import Any

from .current_agent import CurrentSkillProphecy
from .current_relational_model import RelationalPrediction
from .knowledge import KnowledgeStore
from .skills import SKILL_VERB
from .types import Action, Prediction, StateSnapshot


class RelationalStochasticSkillProphecy(CurrentSkillProphecy):
    """Relational Skill rollout that preserves several stochastic futures.

    Historical Skill Prophecy selected one best predicted outcome after every
    primitive, silently collapsing the repaired multi-outcome world model. This
    wrapper keeps a small outcome beam across the whole macro and multiplies
    reliability separately from stochastic outcome mass.
    """

    name = "current-relational-stochastic-skill-prophecy-v2"

    def _base_context_predictions(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        knowledge: KnowledgeStore,
        samples: int,
    ) -> tuple[Prediction, ...]:
        """Use the base model's explicit context API when it exists.

        The repaired semantic calibrator intentionally ignores concrete Knowledge
        re-injection because public response facts already live in the relational
        state. Keeping this helper on the Skill wrapper still matters: promoted
        Skills may be evaluated with episode Knowledge, and the previous v1 code
        called this method without defining it at all.
        """
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
        beam_width = max(1, int(samples))
        branches: list[tuple[StateSnapshot, float, float]] = [
            (state, 1.0, 1.0)
        ]

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
                            * float(
                                getattr(
                                    prediction,
                                    "outcome_probability",
                                    fallback_mass,
                                )
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

        return tuple(
            RelationalPrediction(
                self.library.augment_state(current),
                max(0.0, min(1.0, reliability)),
                source=f"{self.name}:outcome-{index}",
                outcome_probability=max(0.0, min(1.0, mass)),
            )
            for index, (current, reliability, mass) in enumerate(branches)
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
        if not predictions:
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
            "skill_outcome_beam": "planner_samples",
            "skill_reliability_outcome_mass_separate": 1,
            "skill_confidence_stochastic": 1,
            "skill_context_path_defined": 1,
        }
