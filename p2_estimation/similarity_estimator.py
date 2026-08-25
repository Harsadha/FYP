"""
P2 - Semantic similarity estimator.

Day 1 (14:00-17:00): pure function tested against a hand-written mock
candidate list -- does not wait on P1's real graph.

Day 2 (16:00-17:00): error handling added for the race condition where
a candidate chunk has no embedding yet (P1's ingestion pipeline hasn't
caught up). Returns impact_score=0, confidence=0 rather than crashing,
per the sprint plan.

estimate_impact(...) -> List[ImpactEstimate], matching the frozen schema
exactly (see models.py / /schemas/impact_estimate.json).
"""
from typing import List, Tuple, Optional
import numpy as np

from p2_estimation.models import ImpactEstimate

# Fixed confidence for this deterministic estimator during the sprint.
# Not learned/calibrated -- that's Month 3+ (LightGBM) work.
DETERMINISTIC_CONFIDENCE = 0.9


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    sim = float(np.dot(a, b) / denom)  # in [-1, 1]
    return max(0.0, min(1.0, sim) ) # rescaled into [0, 1]


def estimate_impact(
    change_event_id: str,
    changed_embedding: Optional[np.ndarray],
    candidates: List[Tuple[str, Optional[np.ndarray]]],
) -> List[ImpactEstimate]:
    """
    Args:
        change_event_id: id of the ChangeEvent that triggered this run.
        changed_embedding: embedding of the chunk that changed, or None
            if it isn't available yet (race with ingestion).
        candidates: list of (artifact_id, embedding_or_None) pairs to
            score against the changed chunk.

    Returns:
        One ImpactEstimate per candidate, schema-correct, never raises
        on missing embeddings.
    """
    results: List[ImpactEstimate] = []

    for artifact_id, cand_embedding in candidates:
        if changed_embedding is None or cand_embedding is None:
            results.append(
                ImpactEstimate(
                    change_event_id=change_event_id,
                    artifact_id=artifact_id,
                    artifact_type="CHUNK",
                    impact_score=0.0,
                    estimator="SIMILARITY",
                    confidence=0.0,
                )
            )
            continue

        score = _cosine_similarity(changed_embedding, cand_embedding)
        results.append(
            ImpactEstimate(
                change_event_id=change_event_id,
                artifact_id=artifact_id,
                artifact_type="CHUNK",
                impact_score=score,
                estimator="SIMILARITY",
                confidence=DETERMINISTIC_CONFIDENCE,
            )
        )

    return results
