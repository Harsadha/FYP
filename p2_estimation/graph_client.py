"""
P2 - Day 2 integration point.

Day 1 built the similarity estimator against a hand-written mock
candidate list. Day 2's job is to swap that mock for real calls into
P1's graph via get_candidates(changed_artifact_id) -> List[artifact_id],
coordinated at the 9:00 standup.

This module defines the GraphClient interface P2 depends on. Swap
MockGraphClient for P1's real client once it's live -- nothing else
in this file (or in similarity_estimator.py) needs to change, because
both were built against the frozen schema from Day 1.
"""
from typing import List, Optional, Dict
import numpy as np


class GraphClient:
    """Interface P1's real graph client implements."""

    def get_candidates(self, changed_artifact_id: str) -> List[str]:
        raise NotImplementedError

    def get_embedding(self, artifact_id: str) -> Optional[np.ndarray]:
        raise NotImplementedError


class MockGraphClient(GraphClient):
    """Day 1 stand-in. Lets P2 test end-to-end without waiting on P1."""

    def __init__(
        self,
        neighbors: Dict[str, List[str]],
        embeddings: Dict[str, np.ndarray],
    ):
        self._neighbors = neighbors
        self._embeddings = embeddings

    def get_candidates(self, changed_artifact_id: str) -> List[str]:
        return self._neighbors.get(changed_artifact_id, [])

    def get_embedding(self, artifact_id: str) -> Optional[np.ndarray]:
        return self._embeddings.get(artifact_id)


def run_pipeline_for_change(
    client: GraphClient,
    change_event: dict,
) -> list:
    """
    Wires: get_candidates() -> get_embedding() -> estimate_impact().
    Works identically whether `client` is the Day-1 mock or P1's real
    Neo4j-backed client -- that's the point of freezing the interface.
    """
    from similarity_estimator import estimate_impact
    
    changed_artifact_id = change_event["source_artifact_id"]
    
    changed_embedding = client.get_embedding(changed_artifact_id)
    candidate_ids = client.get_candidates(changed_artifact_id)
    candidates = [(cid, client.get_embedding(cid)) for cid in candidate_ids]

    return estimate_impact(change_event_id=change_event["event_id"], changed_embedding=changed_embedding, candidates=candidates)
