from aassr_v2.information_value import information_value_from_measurements
from aassr_v2.knowledge import KnowledgeEntry, KnowledgeStore
from aassr_v2.metrics import imagination_uncertainty, prediction_similarity


def test_identical_imaginations_have_zero_uncertainty() -> None:
    uncertainty = imagination_uncertainty(((1.0, 0.0), (1.0, 0.0), (1.0, 0.0)))
    assert uncertainty == 0.0


def test_disagreeing_imaginations_have_more_uncertainty() -> None:
    low = imagination_uncertainty(((1.0, 0.0), (0.9, 0.1)))
    high = imagination_uncertainty(((1.0, 0.0), (0.0, 1.0)))
    assert high > low


def test_prediction_similarity_matches_reality() -> None:
    exact = prediction_similarity((1.0, 2.0), (1.0, 2.0))
    different = prediction_similarity((1.0, 0.0), (0.0, 1.0))
    assert exact == 1.0
    assert exact > different


def test_information_value_uses_improvement_not_raw_novelty() -> None:
    value = information_value_from_measurements(
        uncertainty_before=0.8,
        uncertainty_after=0.3,
        prediction_score_before=0.2,
        prediction_score_after=0.7,
        unlocked_action_value=0.4,
        goal_progress_before=0.0,
        goal_progress_after=0.1,
    )
    assert value.uncertainty_reduction == 0.5
    assert value.prediction_gain == 0.5
    assert value.total() == 1.5


def test_knowledge_store_tracks_provenance() -> None:
    store = KnowledgeStore()
    entry = KnowledgeEntry(
        key="door.required_color",
        value="blue",
        source_trace_id="aseq-001",
        enabled_action_signatures=("use|door|blue_key|_",),
    )

    delta = store.apply((entry,))

    assert delta.added == (entry,)
    assert store.get("door.required_color") == entry
