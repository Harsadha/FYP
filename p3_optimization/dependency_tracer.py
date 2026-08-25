"""
P3 - Deterministic dependency-tracing (TRACE) estimator.

Given a changed artifact, performs a bounded k-hop breadth-first
traversal reachable via GraphClient.get_candidates(), scoring impact
as decay ** hop_count. Produces ImpactEstimate records with
estimator="TRACE", matching /schemas/impact_estimate.json exactly
(reuses p2_estimation.models.ImpactEstimate -- see p3_optimization/
models.py for why this is not redefined).

DOCUMENTED LIMITATION: the current GraphClient interface
(p2_estimation/graph_client.py) exposes get_candidates(artifact_id) ->
List[str] only. It does not expose edge type (e.g. DUPLICATES vs.
DERIVED_FROM) or edge weight. This tracer therefore CANNOT distinguish
relationship types or weight individual hops -- score is purely
hop-count decay. If/when P1's real graph client exposes typed, weighted
edges, this should be extended to multiply by edge weight per hop.
That is a TODO, not something invented here.

NO CHANGES to p2_estimation/graph_client.py were required: k-hop
traversal is built entirely by calling the existing get_candidates()
repeatedly, one hop at a time. This is intentional -- see project rule
"prefer zero changes to P2."

Also matches P2's own conventions for fields the graph client can't
supply: artifact_type is hard-coded "CHUNK" (get_candidates doesn't
expose type either, same simplification similarity_estimator.py makes)
and confidence is a fixed constant, since this is a deterministic,
unlearned estimator (same rationale as P2's DETERMINISTIC_CONFIDENCE).
"""
from typing import List, Set, Dict
from collections import deque

from p2_estimation.models import ImpactEstimate
from p2_estimation.graph_client import GraphClient

DEFAULT_DECAY = 0.7
DEFAULT_MAX_HOPS = 3
# Deterministic, unlearned estimator -> fixed confidence, same rationale
# as similarity_estimator.DETERMINISTIC_CONFIDENCE. TRACE is graph-
# structural rather than measured similarity, so it is given full
# confidence when it fires at all -- open to recalibration once real
# outcomes are available via the feedback loop.
DETERMINISTIC_CONFIDENCE = 1.0


def trace_impact(
    change_event_id: str,
    changed_artifact_id: str,
    client: GraphClient,
    decay: float = DEFAULT_DECAY,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> List[ImpactEstimate]:
    """
    Forward BFS from changed_artifact_id, bounded to max_hops.

    Note: get_candidates() is directed (from-artifact outward) with no
    reverse-lookup exposed, so this is a forward-only traversal, not a
    true bidirectional propagation. An artifact reachable via multiple
    paths is scored once, at its shortest-hop distance (BFS guarantees
    this), rather than being emitted as duplicate ImpactEstimates.

    Args:
        change_event_id: id of the ChangeEvent that triggered this run.
        changed_artifact_id: the artifact that changed.
        client: any GraphClient implementation (MockGraphClient today,
            P1's real client later -- interface is unchanged either way).
        decay: per-hop score multiplier, in (0, 1]. Lower = impact
            drops off faster with graph distance.
        max_hops: traversal depth bound.

    Returns:
        One ImpactEstimate per reachable artifact (excluding the
        changed artifact itself). Empty list if there are no
        candidates, the graph is empty, or changed_artifact_id is
        unknown to the client -- never raises for these cases.
    """
    if not (0.0 < decay <= 1.0):
        raise ValueError(f"decay must be in (0, 1], got {decay}")
    if max_hops < 1:
        raise ValueError(f"max_hops must be >= 1, got {max_hops}")

    visited_hop: Dict[str, int] = {}
    seen: Set[str] = {changed_artifact_id}
    frontier: deque = deque([(changed_artifact_id, 0)])

    while frontier:
        node_id, hop = frontier.popleft()
        if hop >= max_hops:
            continue
        for neighbor_id in client.get_candidates(node_id):
            if neighbor_id in seen:
                continue
            seen.add(neighbor_id)
            visited_hop[neighbor_id] = hop + 1
            frontier.append((neighbor_id, hop + 1))

    results: List[ImpactEstimate] = []
    for artifact_id, hop in visited_hop.items():
        score = max(0.0, min(1.0, decay ** hop))
        results.append(
            ImpactEstimate(
                change_event_id=change_event_id,
                artifact_id=artifact_id,
                artifact_type="CHUNK",
                impact_score=score,
                estimator="TRACE",
                confidence=DETERMINISTIC_CONFIDENCE,
            )
        )
    return results