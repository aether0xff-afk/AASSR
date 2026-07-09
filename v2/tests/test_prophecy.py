import unittest

from aassr import CellKind, GridWorld, GridWorldDMP, KK
from aassr.knowledge import KnowledgeDelta, KV, ValueType
from aassr.prophecy import (
    ProphecyModule,
    SequenceProphecyModel,
    TableProphecyModel,
    TransformerProphecyModel,
    gridworld_state_signature,
)


class ProphecyModelTests(unittest.TestCase):
    def test_unseen_action_returns_default_probabilities(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))
        candidate = dmp.generate_candidates()[0]
        model = TableProphecyModel(prior=0.1)

        prediction = model.predict(gridworld_state_signature(dmp), candidate)

        self.assertGreater(prediction.error_prob, 0.0)
        self.assertGreater(prediction.flag_prob, 0.0)
        self.assertIn(KK.KNOWN_CELL, prediction.kk_probs)

    def test_update_increases_delta_k_probability_for_action(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))
        candidate = dmp.generate_candidates()[0]
        signature = gridworld_state_signature(dmp)
        model = TableProphecyModel(prior=0.01)
        before = model.predict(signature, candidate).kk_probs[KK.KNOWN_CELL]
        delta = KnowledgeDelta(
            added=((KK.KNOWN_CELL, KV((1, 0), ValueType.CELL_COORD)),)
        )

        model.update(signature, candidate, delta, actual_error=False, actual_flag=False)

        after = model.predict(signature, candidate).kk_probs[KK.KNOWN_CELL]
        self.assertGreater(after, before)

    def test_error_update_increases_error_probability(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))
        candidate = dmp.generate_candidates()[0]
        signature = gridworld_state_signature(dmp)
        model = TableProphecyModel(prior=0.01)
        before = model.predict(signature, candidate).error_prob

        model.update(signature, candidate, KnowledgeDelta(), actual_error=True, actual_flag=False)

        after = model.predict(signature, candidate).error_prob
        self.assertGreater(after, before)

    def test_prediction_error_reflects_observation_difference(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1), cells={(1, 0): CellKind.WALL}))
        candidate = dmp.generate_candidates()[0]
        signature = gridworld_state_signature(dmp)
        model = TableProphecyModel(prior=0.01)
        delta = KnowledgeDelta(
            added=((KK.WALL_CELL, KV((1, 0), ValueType.CELL_COORD)),)
        )

        update = model.update(signature, candidate, delta, actual_error=True, actual_flag=False)

        self.assertGreater(update.prediction_error, 0.0)
        self.assertEqual(update.loss, update.prediction_error)

    def test_usage_delta_does_not_train_kk_prediction(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))
        candidate = dmp.generate_candidates()[0]
        signature = gridworld_state_signature(dmp)
        model = TableProphecyModel(prior=0.01)
        before = model.predict(signature, candidate).kk_probs[KK.KNOWN_CELL]
        delta = KnowledgeDelta(
            usage_updated=((KK.KNOWN_CELL, KV((1, 1), ValueType.CELL_COORD)),)
        )

        model.update(signature, candidate, delta, actual_error=False, actual_flag=False)

        after = model.predict(signature, candidate).kk_probs[KK.KNOWN_CELL]
        self.assertLessEqual(after, before)

    def test_sequence_prophecy_variant_updates_prediction_and_context(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))
        candidate = dmp.generate_candidates()[0]
        signature = gridworld_state_signature(dmp)
        model = SequenceProphecyModel(seed=3, input_dim=48, hidden_dim=12, learning_rate=0.05)
        before = model.predict(signature, candidate).kk_probs[KK.KNOWN_CELL]
        delta = KnowledgeDelta(
            added=((KK.KNOWN_CELL, KV((1, 0), ValueType.CELL_COORD)),)
        )

        update = model.update(signature, candidate, delta, actual_error=False, actual_flag=False)
        after = model.predict(signature, candidate).kk_probs[KK.KNOWN_CELL]

        self.assertGreater(update.loss, 0.0)
        self.assertNotEqual(after, before)

    def test_prophecy_implementations_share_common_interface(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))
        candidate = dmp.generate_candidates()[0]
        signature = gridworld_state_signature(dmp)

        for model in (
            TableProphecyModel(prior=0.01),
            SequenceProphecyModel(seed=4, input_dim=48, hidden_dim=12),
            TransformerProphecyModel(seed=5, input_dim=64, model_dim=12),
        ):
            prediction = model.predict(signature, candidate)
            update = model.update(signature, candidate, KnowledgeDelta(), False, False)

            self.assertIsInstance(prediction.kk_probs, dict)
            self.assertGreaterEqual(update.prediction_error, 0.0)
            _: ProphecyModule = model

    def test_transformer_prophecy_variant_updates_prediction_head(self) -> None:
        dmp = GridWorldDMP(GridWorld(width=3, height=3, start=(1, 1)))
        candidate = dmp.generate_candidates()[0]
        signature = gridworld_state_signature(dmp)
        model = TransformerProphecyModel(seed=6, input_dim=64, model_dim=12, learning_rate=0.05)
        before = model.predict(signature, candidate).kk_probs[KK.KNOWN_CELL]
        delta = KnowledgeDelta(
            added=((KK.KNOWN_CELL, KV((1, 0), ValueType.CELL_COORD)),)
        )

        update = model.update(signature, candidate, delta, actual_error=False, actual_flag=False)
        after = model.predict(signature, candidate).kk_probs[KK.KNOWN_CELL]

        self.assertGreater(update.loss, 0.0)
        self.assertNotEqual(after, before)


if __name__ == "__main__":
    unittest.main()
