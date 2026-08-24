import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from similarity_estimator import estimate_impact
from models import ImpactEstimate


def test_shape_matches_schema_with_mock_data():
    """Day 1, 17:30 demo case: fake chunks -> correctly-shaped ImpactEstimate."""
    changed = np.array([1.0, 0.0, 0.0])
    candidates = [
        ("chunk_similar", np.array([0.99, 0.01, 0.0])),
        ("chunk_orthogonal", np.array([0.0, 1.0, 0.0])),
    ]

    results = estimate_impact("evt_1", changed, candidates)

    assert len(results) == 2
    for r in results:
        assert isinstance(r, ImpactEstimate)
        assert r.change_event_id == "evt_1"
        assert r.artifact_type == "CHUNK"
        assert r.estimator == "SIMILARITY"
        assert 0.0 <= r.impact_score <= 1.0
        assert 0.0 <= r.confidence <= 1.0


def test_similar_vector_scores_higher_than_orthogonal():
    changed = np.array([1.0, 0.0, 0.0])
    candidates = [
        ("chunk_similar", np.array([0.99, 0.01, 0.0])),
        ("chunk_orthogonal", np.array([0.0, 1.0, 0.0])),
    ]
    results = {r.artifact_id: r.impact_score for r in estimate_impact("evt_1", changed, candidates)}
    assert results["chunk_similar"] > results["chunk_orthogonal"]


def test_missing_candidate_embedding_does_not_crash():
    """Day 2, 16:00-17:00: race condition with P1's ingestion pipeline."""
    changed = np.array([1.0, 0.0, 0.0])
    candidates = [
        ("chunk_ready", np.array([1.0, 0.0, 0.0])),
        ("chunk_not_yet_embedded", None),
    ]

    results = {r.artifact_id: r for r in estimate_impact("evt_2", changed, candidates)}

    assert results["chunk_not_yet_embedded"].impact_score == 0.0
    assert results["chunk_not_yet_embedded"].confidence == 0.0
    assert results["chunk_ready"].impact_score > 0.0


def test_missing_changed_embedding_does_not_crash():
    candidates = [("chunk_a", np.array([1.0, 0.0, 0.0]))]
    results = estimate_impact("evt_3", None, candidates)
    assert results[0].impact_score == 0.0
    assert results[0].confidence == 0.0


def test_empty_candidate_list_returns_empty_result():
    changed = np.array([1.0, 0.0, 0.0])
    assert estimate_impact("evt_4", changed, []) == []


def test_invalid_impact_estimate_field_raises():
    with pytest.raises(ValueError):
        ImpactEstimate(
            change_event_id="e",
            artifact_id="a",
            artifact_type="NOT_A_TYPE",
            impact_score=0.5,
            estimator="SIMILARITY",
            confidence=0.5,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
