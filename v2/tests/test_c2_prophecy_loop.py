import unittest

from aassr import DMPConfig, GridWorld, GridWorldDMP
from aassr.knowledge import KK, KnowledgeDelta, KV, ValueType
from aassr.policy import PolicyABC, candidate_axes
from aassr.prophecy import ProphecyPrediction, ProphecyUpdate, TableProphecyModel


class SpyProphecy(TableProphecyModel):
    def __init__(self, prediction_error: float = 0.5) -> None:
        super().__init__()
        self.prediction_error = prediction_error
        self.predict_calls = []
        self.update_calls = []

    def predict(self, state_signature, candidate):
        self.predict_calls.append((state_signature, candidate))
        return ProphecyPrediction(
            kk_probs={kk: 0.0 for kk in KK},
            error_prob=0.25,
            flag_prob=0.1,
        )

    def update(self, state_signature, candidate, actual_delta, actual_error, actual_flag):
        self.update_calls.append(
            (state_signature, candidate, actual_delta, actual_error, actual_flag)
        )
        return ProphecyUpdate(prediction_error=self.prediction_error, loss=self.prediction_error)


class C2ProphecyLoopTests(unittest.TestCase):
    def test_c2_predict_is_called_before_execution(self) -> None:
        prophecy = SpyProphecy()
        dmp = GridWorldDMP(
            GridWorld(width=3, height=3, start=(1, 1)),
            prophecy=prophecy,
            config=DMPConfig(use_prophecy=True),
        )
        candidate = dmp.generate_candidates()[0]

        dmp.execute(candidate)

        self.assertEqual(len(prophecy.predict_calls), 1)
        self.assertEqual(prophecy.predict_calls[0][1], candidate)

    def test_c2_update_receives_semantic_delta_after_execution(self) -> None:
        prophecy = SpyProphecy()
        dmp = GridWorldDMP(
            GridWorld(width=3, height=3, start=(1, 1)),
            prophecy=prophecy,
            config=DMPConfig(use_prophecy=True),
        )
        candidate = dmp.generate_candidates()[0]

        result = dmp.execute(candidate)

        self.assertEqual(len(prophecy.update_calls), 1)
        update_delta = prophecy.update_calls[0][2]
        self.assertEqual(update_delta.semantic_changed_kk(), result.delta_k.semantic_changed_kk())

    def test_prophecy_error_is_recorded_on_step_result(self) -> None:
        dmp = GridWorldDMP(
            GridWorld(width=3, height=3, start=(1, 1)),
            prophecy=SpyProphecy(prediction_error=0.42),
            config=DMPConfig(use_prophecy=True),
        )

        result = dmp.execute(dmp.generate_candidates()[0])

        self.assertIsNotNone(result.prophecy_prediction)
        self.assertEqual(result.prophecy_error, 0.42)
        self.assertEqual(result.prophecy_loss, 0.42)
        self.assertEqual(result.to_dict()["predicted_error_prob"], 0.25)

    def test_total_reward_includes_beta_times_prophecy_error(self) -> None:
        dmp = GridWorldDMP(
            GridWorld(width=3, height=3, start=(1, 1)),
            prophecy=SpyProphecy(prediction_error=0.5),
            config=DMPConfig(use_prophecy=True, prophecy_beta=0.3),
        )

        result = dmp.execute(dmp.generate_candidates()[0])

        self.assertAlmostEqual(
            result.total_reward,
            result.external_reward + result.intrinsic_reward + 0.15,
        )

    def test_usage_only_delta_is_not_prophecy_target(self) -> None:
        model = TableProphecyModel(prior=0.01)
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))
        candidate = dmp.generate_candidates()[0]
        signature = ("same",)
        before = model.predict(signature, candidate).kk_probs[KK.KNOWN_CELL]
        delta = KnowledgeDelta(
            usage_updated=((KK.KNOWN_CELL, KV((1, 1), ValueType.CELL_COORD)),)
        )

        model.update(signature, candidate, delta, actual_error=False, actual_flag=False)

        after = model.predict(signature, candidate).kk_probs[KK.KNOWN_CELL]
        self.assertLessEqual(after, before)

    def test_policyabc_updates_with_prophecy_adjusted_total_reward(self) -> None:
        policy = PolicyABC.uniform_gridworld(learning_rate=1.0, seed=0)
        dmp = GridWorldDMP(
            GridWorld(width=3, height=3, start=(1, 1)),
            scorer=policy,
            prophecy=SpyProphecy(prediction_error=0.5),
            config=DMPConfig(use_prophecy=True, prophecy_beta=0.3),
        )
        candidate = dmp.generate_candidates()[0]
        what, _, _ = candidate_axes(candidate)
        before = policy.policy_a[what]

        result = dmp.execute(candidate)

        self.assertGreater(result.total_reward, result.external_reward + result.intrinsic_reward)
        self.assertGreater(policy.policy_a[what], before)


if __name__ == "__main__":
    unittest.main()
