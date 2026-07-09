import unittest

from aassr import DMPConfig, GridWorld, GridWorldDMP, ImaginationCycle, KK
from aassr.imagination import ImaginationConfig
from aassr.policy import PolicyABC
from aassr.prophecy import ProphecyPrediction, TableProphecyModel


class FakeProphecy(TableProphecyModel):
    def __init__(self, predictions):
        super().__init__()
        self.predictions = {
            candidate_signature(candidate): value for candidate, value in predictions
        }
        self.predict_calls = []

    def predict(self, state_signature, candidate):
        self.predict_calls.append((state_signature, candidate))
        return self.predictions[candidate_signature(candidate)]


def prediction(*, kk_gain=0.0, error=0.0, flag=0.0):
    per_kk = kk_gain / len(KK)
    return ProphecyPrediction(
        kk_probs={kk: per_kk for kk in KK},
        error_prob=error,
        flag_prob=flag,
    )


def slot_prediction(*, kk: KK, probability: float, error=0.0, flag=0.0):
    return ProphecyPrediction(
        kk_probs={item: probability if item == kk else 0.0 for item in KK},
        error_prob=error,
        flag_prob=flag,
    )


def candidate_signature(candidate):
    return (
        candidate.template,
        tuple(sorted((kk.value, repr(value)) for kk, value in candidate.bindings.items())),
    )


class ImaginationCycleTests(unittest.TestCase):
    def test_flag_probability_increases_score(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))
        low, high = dmp.generate_candidates()[:2]
        prophecy = FakeProphecy([
            (low, prediction(flag=0.1)),
            (high, prediction(flag=0.8)),
        ])
        imagination = ImaginationCycle(prophecy)

        low_score = imagination.score_candidate("s", low).score
        high_score = imagination.score_candidate("s", high).score

        self.assertGreater(high_score, low_score)

    def test_error_probability_decreases_score(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))
        safe, risky = dmp.generate_candidates()[:2]
        prophecy = FakeProphecy([
            (safe, prediction(error=0.1)),
            (risky, prediction(error=0.9)),
        ])
        imagination = ImaginationCycle(prophecy)

        safe_score = imagination.score_candidate("s", safe).score
        risky_score = imagination.score_candidate("s", risky).score

        self.assertGreater(safe_score, risky_score)

    def test_expected_kk_gain_increases_score(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))
        low, high = dmp.generate_candidates()[:2]
        prophecy = FakeProphecy([
            (low, prediction(kk_gain=0.1)),
            (high, prediction(kk_gain=3.0)),
        ])
        imagination = ImaginationCycle(prophecy)

        low_score = imagination.score_candidate("s", low).score
        high_score = imagination.score_candidate("s", high).score

        self.assertGreater(high_score, low_score)

    def test_choose_selects_highest_score_candidate(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))
        first, second = dmp.generate_candidates()[:2]
        prophecy = FakeProphecy([
            (first, prediction(flag=0.1)),
            (second, prediction(flag=0.9)),
        ])
        imagination = ImaginationCycle(prophecy)

        trace = imagination.choose("s", [first, second])

        self.assertEqual(trace.selected, second)
        self.assertEqual(len(trace.scores), 2)

    def test_imagination_does_not_execute_candidates_or_read_world(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))
        first, second = dmp.generate_candidates()[:2]
        prophecy = FakeProphecy([
            (first, prediction(flag=0.1)),
            (second, prediction(flag=0.9)),
        ])
        imagination = ImaginationCycle(prophecy)
        before_step = dmp.step_index
        before_position = dmp.position

        imagination.choose("s", [first, second], dmp=dmp)

        self.assertEqual(dmp.step_index, before_step)
        self.assertEqual(dmp.position, before_position)
        self.assertGreaterEqual(len(prophecy.predict_calls), 2)

    def test_depth_limited_rollout_can_select_setup_action(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))
        setup, payoff = dmp.generate_candidates()[:2]
        payoff_slot = next(kk for kk in payoff.required_kk_slots if kk != KK.CURRENT_POS)
        prophecy = FakeProphecy([
            (setup, slot_prediction(kk=payoff_slot, probability=1.0)),
            (payoff, prediction(flag=0.4)),
        ])

        one_step = ImaginationCycle(
            prophecy,
            ImaginationConfig(
                rollout_depth=1,
                policy_prior_weight=0.0,
                dependency_weight=1.5,
            ),
        )
        rollout = ImaginationCycle(
            prophecy,
            ImaginationConfig(
                rollout_depth=2,
                rollout_discount=0.65,
                policy_prior_weight=0.0,
                dependency_weight=1.5,
            ),
        )

        one_step_trace = one_step.choose("s", [setup, payoff])
        rollout_trace = rollout.choose("s", [setup, payoff])

        self.assertEqual(one_step_trace.selected, payoff)
        self.assertEqual(rollout_trace.selected, setup)
        self.assertGreater(rollout_trace.selected_score.rollout_value, 0.0)

    def test_dmp_c3_records_imagination_trace_on_step_result(self) -> None:
        c3_candidates_source = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))
        candidates = c3_candidates_source.generate_candidates()
        prophecy = FakeProphecy([
            (candidate, prediction(flag=0.1))
            for candidate in candidates
        ])
        prophecy.predictions[candidate_signature(candidates[1])] = prediction(flag=0.9)
        policy = PolicyABC.uniform_gridworld(seed=0)
        c3 = GridWorldDMP(
            GridWorld(width=3, height=3, start=(1, 1)),
            scorer=policy,
            prophecy=prophecy,
            imagination=ImaginationCycle(prophecy, ImaginationConfig(policy_prior_weight=0.0)),
            config=DMPConfig(use_prophecy=True, use_imagination=True),
        )

        selected = c3.choose_candidate("scorer")
        result = c3.execute(selected)

        self.assertIsNotNone(result.imagination_trace)
        self.assertEqual(result.imagination_trace.selected, selected)
        self.assertGreater(result.to_dict()["imagination_candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()
