import csv
import math
import tempfile
import unittest
from pathlib import Path

from aassr import DMPConfig, GridWorld, GridWorldDMP, KK
from aassr.analysis import analyze_results
from aassr.gridworld import ActionCandidate, ActionName, _candidate_match_payload, _prediction_alignment
from aassr.imagination import (
    ImaginationConfig,
    PredictedStateImaginationCycle,
    candidate_canonical_signature,
    trace_diagnostics,
    unique_candidates_by_signature,
)
from aassr.knowledge import ValueType
from aassr.prophecy import ProphecyPrediction


class SetupProphecy:
    def predict(self, state_signature, candidate):
        probs = {kk: 0.0 for kk in KK}
        if candidate.name == ActionName.INSPECT_CELL:
            probs[KK.KEY_OBJECT] = 1.0
            return ProphecyPrediction(probs, error_prob=0.0, flag_prob=0.0)
        if candidate.name == ActionName.USE_OBJECT:
            return ProphecyPrediction(probs, error_prob=0.0, flag_prob=1.0)
        return ProphecyPrediction(probs, error_prob=0.0, flag_prob=0.0)


class KeyUnlockProphecy:
    def predict(self, state_signature, candidate):
        probs = {kk: 0.0 for kk in KK}
        if candidate.name == ActionName.INSPECT_CELL:
            probs[KK.KEY_OBJECT] = 1.0
            return ProphecyPrediction(probs, error_prob=0.0, flag_prob=0.0)
        if candidate.name == ActionName.USE_OBJECT:
            return ProphecyPrediction(probs, error_prob=0.0, flag_prob=1.0)
        return ProphecyPrediction(probs, error_prob=0.0, flag_prob=0.0)


class APASSRDiagnosticTests(unittest.TestCase):
    def _dmp_with_door(self) -> GridWorldDMP:
        dmp = GridWorldDMP(
            GridWorld(width=3, height=2, start=(0, 0)),
            config=DMPConfig(independent_policy_axes=True),
        )
        dmp.store.add(KK.DOOR_CELL, (1, 0), ValueType.CELL_COORD)
        return dmp

    def test_newly_unlocked_action_is_counted_with_causal_kk(self) -> None:
        dmp = self._dmp_with_door()
        trace = PredictedStateImaginationCycle(
            SetupProphecy(),
            ImaginationConfig(rollout_depth=2, policy_prior_weight=0.0),
        ).choose("ignored", dmp.generate_candidates(), dmp=dmp)

        diagnostics = trace_diagnostics(trace)

        self.assertGreater(diagnostics.newly_unlocked_action_count, 0)
        self.assertTrue(any(kk == "KK_KEY_OBJECT" and count > 0 for kk, count in diagnostics.unlocked_by_kk_counts))

    def test_existing_candidate_is_not_counted_as_newly_unlocked(self) -> None:
        dmp = self._dmp_with_door()
        dmp.store.add(KK.KEY_OBJECT, "key@(0,0)", ValueType.OBJECT_INSTANCE)
        trace = PredictedStateImaginationCycle(
            SetupProphecy(),
            ImaginationConfig(rollout_depth=2, policy_prior_weight=0.0),
        ).choose("ignored", dmp.generate_candidates(), dmp=dmp)

        use_unlocks = [
            candidate
            for trajectory in trace.trajectories
            for step in trajectory.steps
            for candidate in step.newly_unlocked_candidates
            if candidate.name == ActionName.USE_OBJECT
            and candidate.bindings.get(KK.KEY_OBJECT) == "key@(0,0)"
        ]

        self.assertEqual(use_unlocks, [])

    def test_rollout_records_state_signatures_before_and_after(self) -> None:
        dmp = self._dmp_with_door()
        trace = PredictedStateImaginationCycle(SetupProphecy(), ImaginationConfig(rollout_depth=2)).choose(
            "ignored",
            dmp.generate_candidates(),
            dmp=dmp,
        )
        step = next(step for trajectory in trace.trajectories for step in trajectory.steps)

        self.assertIsNotNone(step.state_signature_before)
        self.assertIsNotNone(step.state_signature_after)
        self.assertNotEqual(step.state_signature_before, step.state_signature_after)

    def test_setup_action_selection_is_reported(self) -> None:
        dmp = self._dmp_with_door()
        trace = PredictedStateImaginationCycle(
            SetupProphecy(),
            ImaginationConfig(rollout_depth=2, flag_weight=10.0, policy_prior_weight=0.0),
        ).choose("ignored", dmp.generate_candidates(), dmp=dmp)

        diagnostics = trace_diagnostics(trace)

        self.assertTrue(diagnostics.selected_action_has_future_dependency)
        self.assertGreaterEqual(diagnostics.selected_action_future_value, 0.0)

    def test_prediction_alignment_precision_recall_f1(self) -> None:
        prediction = ProphecyPrediction(
            kk_probs={kk: 0.0 for kk in KK} | {KK.KEY_CELL: 0.9, KK.DOOR_CELL: 0.8},
            error_prob=0.1,
            flag_prob=0.1,
        )

        metrics = _prediction_alignment(
            prediction,
            {KK.KEY_CELL, KK.FLAG_CELL},
            actual_error=False,
            actual_flag=False,
            thresholds=(0.5, 0.5, 0.5),
        )

        self.assertEqual(metrics["predicted_kk_precision"], 0.5)
        self.assertEqual(metrics["predicted_kk_recall"], 0.5)
        self.assertEqual(metrics["predicted_kk_f1"], 0.5)

    def test_imagined_next_action_match_payload(self) -> None:
        dmp = self._dmp_with_door()
        first = dmp.generate_candidates()[0]
        same = type(first)(
            name=first.name,
            template=first.template,
            required_kk_slots=first.required_kk_slots,
            bindings=dict(first.bindings),
            strategy=first.strategy,
        )
        different = dmp.generate_candidates()[-1]

        self.assertTrue(_candidate_match_payload(first, same)["exact"])
        self.assertFalse(_candidate_match_payload(first, different)["exact"])

    def test_placeholder_does_not_leak_into_executed_first_action(self) -> None:
        dmp = self._dmp_with_door()
        trace = PredictedStateImaginationCycle(SetupProphecy(), ImaginationConfig(rollout_depth=2)).choose(
            "ignored",
            dmp.generate_candidates(),
            dmp=dmp,
        )
        diagnostics = trace_diagnostics(trace)

        self.assertGreater(diagnostics.placeholder_generated_candidate_count, 0)
        self.assertEqual(diagnostics.placeholder_selected_candidate_count, 0)

    def test_calibrated_candidate_signature_deduplicates_placeholders_only(self) -> None:
        first = ActionCandidate(
            name=ActionName.USE_OBJECT,
            template="USE_OBJECT {KK_KEY_OBJECT} AT {KK_DOOR_CELL}",
            required_kk_slots=(KK.KEY_OBJECT, KK.DOOR_CELL),
            bindings={KK.KEY_OBJECT: "imagined-key#1", KK.DOOR_CELL: (1, 0)},
        )
        second = ActionCandidate(
            name=ActionName.USE_OBJECT,
            template=first.template,
            required_kk_slots=first.required_kk_slots,
            bindings={KK.KEY_OBJECT: "imagined-key#2", KK.DOOR_CELL: (1, 0)},
        )
        concrete = ActionCandidate(
            name=ActionName.USE_OBJECT,
            template=first.template,
            required_kk_slots=first.required_kk_slots,
            bindings={KK.KEY_OBJECT: "key@(0,0)", KK.DOOR_CELL: (1, 0)},
        )
        different_where = ActionCandidate(
            name=ActionName.USE_OBJECT,
            template=first.template,
            required_kk_slots=first.required_kk_slots,
            bindings={KK.KEY_OBJECT: "imagined-key#3", KK.DOOR_CELL: (2, 0)},
        )

        self.assertEqual(candidate_canonical_signature(first), candidate_canonical_signature(second))
        unique = unique_candidates_by_signature([first, second, concrete, different_where])

        self.assertEqual(unique, [first, concrete, different_where])

    def test_calibrated_rollout_records_raw_and_unique_future_expansions(self) -> None:
        dmp = self._dmp_with_door()
        trace = PredictedStateImaginationCycle(
            KeyUnlockProphecy(),
            ImaginationConfig(
                rollout_depth=2,
                policy_prior_weight=0.0,
                calibrated_imagination_enabled=True,
                candidate_dedup_enabled=True,
                placeholder_confidence_scale=1.0,
                mixed_grounding_confidence_scale=1.0,
            ),
        ).choose("ignored", dmp.generate_candidates(), dmp=dmp)

        diagnostics = trace_diagnostics(trace)

        self.assertGreater(diagnostics.raw_future_candidate_count, 0)
        self.assertGreaterEqual(diagnostics.raw_future_candidate_count, diagnostics.unique_future_candidate_count)
        self.assertGreaterEqual(diagnostics.raw_newly_unlocked_action_count, diagnostics.unique_newly_unlocked_action_count)
        self.assertGreaterEqual(diagnostics.future_candidate_dedup_ratio, 0.0)
        self.assertLessEqual(diagnostics.future_candidate_dedup_ratio, 1.0)

    def test_calibrated_q_one_matches_full_and_q_zero_removes_future_value(self) -> None:
        dmp = self._dmp_with_door()
        candidates = dmp.generate_candidates()
        full = PredictedStateImaginationCycle(
            KeyUnlockProphecy(),
            ImaginationConfig(rollout_depth=2, policy_prior_weight=0.0, seed=3),
        ).choose("ignored", candidates, dmp=dmp)
        calibrated_one = PredictedStateImaginationCycle(
            KeyUnlockProphecy(),
            ImaginationConfig(
                rollout_depth=2,
                policy_prior_weight=0.0,
                seed=3,
                calibrated_imagination_enabled=True,
                candidate_dedup_enabled=True,
                placeholder_confidence_scale=1.0,
                mixed_grounding_confidence_scale=1.0,
            ),
        ).choose("ignored", candidates, dmp=dmp)
        calibrated_zero = PredictedStateImaginationCycle(
            KeyUnlockProphecy(),
            ImaginationConfig(
                rollout_depth=2,
                policy_prior_weight=0.0,
                seed=3,
                calibrated_imagination_enabled=True,
                candidate_dedup_enabled=True,
                placeholder_confidence_scale=0.0,
                mixed_grounding_confidence_scale=0.0,
            ),
        ).choose("ignored", candidates, dmp=dmp)

        full_diag = trace_diagnostics(full)
        one_diag = trace_diagnostics(calibrated_one)
        zero_diag = trace_diagnostics(calibrated_zero)

        self.assertAlmostEqual(one_diag.selected_action_future_value, full_diag.selected_action_future_value)
        self.assertAlmostEqual(zero_diag.calibrated_selected_future_value, 0.0)
        self.assertGreaterEqual(zero_diag.uncalibrated_selected_future_value, zero_diag.calibrated_selected_future_value)
        for value in (
            one_diag.mean_selected_path_confidence,
            zero_diag.mean_selected_path_confidence,
            zero_diag.future_value_discount_ratio,
        ):
            self.assertTrue(math.isfinite(value))

    def test_placeholder_grounding_discount_is_separate_from_concrete_and_mixed(self) -> None:
        dmp = self._dmp_with_door()
        imagination = PredictedStateImaginationCycle(
            KeyUnlockProphecy(),
            ImaginationConfig(placeholder_confidence_scale=0.25, mixed_grounding_confidence_scale=0.6),
        )
        placeholder = ActionCandidate(
            name=ActionName.USE_OBJECT,
            template="USE_OBJECT {KK_KEY_OBJECT}",
            required_kk_slots=(KK.KEY_OBJECT,),
            bindings={KK.KEY_OBJECT: "imagined-key#1"},
        )
        concrete = ActionCandidate(
            name=ActionName.USE_OBJECT,
            template="USE_OBJECT {KK_KEY_OBJECT}",
            required_kk_slots=(KK.KEY_OBJECT,),
            bindings={KK.KEY_OBJECT: "key@(0,0)"},
        )
        mixed = ActionCandidate(
            name=ActionName.USE_OBJECT,
            template="USE_OBJECT {KK_KEY_OBJECT} AT {KK_DOOR_CELL}",
            required_kk_slots=(KK.KEY_OBJECT, KK.DOOR_CELL),
            bindings={KK.KEY_OBJECT: "imagined-key#1", KK.DOOR_CELL: (1, 0)},
        )
        empty_delta = imagination._predicted_delta(ProphecyPrediction({kk: 0.0 for kk in KK}, 0.0, 0.0), 1)

        self.assertEqual(imagination._grounding_factor(concrete, empty_delta, ()), 1.0)
        self.assertEqual(imagination._grounding_factor(placeholder, empty_delta, ()), 0.25)
        self.assertEqual(imagination._grounding_factor(mixed, empty_delta, ()), 0.6)

    def test_old_episode_csv_analysis_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "old"
            condition_dir = root / "C3"
            condition_dir.mkdir(parents=True)
            with (condition_dir / "gridworld_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "condition",
                        "seed",
                        "episode",
                        "success",
                        "steps_to_flag",
                        "total_reward",
                        "external_reward",
                        "semantic_gain_total",
                        "prophecy_error_mean",
                        "repeat_count",
                        "error_count",
                        "knowledge_reuse_count",
                        "unique_action_count",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "condition": "C3",
                        "seed": 0,
                        "episode": 0,
                        "success": True,
                        "steps_to_flag": 5,
                        "total_reward": 1.0,
                        "external_reward": 1.0,
                        "semantic_gain_total": 2,
                        "prophecy_error_mean": 0.0,
                        "repeat_count": 0,
                        "error_count": 0,
                        "knowledge_reuse_count": 0,
                        "unique_action_count": 1,
                    }
                )

            analyze_results(input_dir=root, output_dir=root / "analysis", bootstrap_samples=5, learning_window=1)

            self.assertTrue((root / "analysis" / "diagnostic_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
