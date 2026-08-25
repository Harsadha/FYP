"""
P3 - tests for dependency_tracer.py. Uses P2's real MockGraphClient
(not a P3-invented stand-in) so this test exercises the actual
interface TRACE will run against.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from p2_estimation.graph_client import MockGraphClient
from p3_optimization.dependency_tracer import trace_impact


def _client(neighbors):
    return MockGraphClient(neighbors=neighbors, embeddings={})


def test_direct_dependency():
    client = _client({"chunk_exp_02": ["chunk_faq_17"]})
    results = trace_impact("evt1", "chunk_exp_02", client)
    assert len(results) == 1
    assert results[0].artifact_id == "chunk_faq_17"
    assert results[0].estimator == "TRACE"
    assert abs(results[0].impact_score - 0.7) < 1e-9  # decay^1
    assert results[0].confidence == 1.0


def test_no_dependency():
    client = _client({"chunk_exp_02": []})
    results = trace_impact("evt1", "chunk_exp_02", client)
    assert results == []


def test_missing_graph_node():
    client = _client({})  # "chunk_exp_02" not present as a key at all
    results = trace_impact("evt1", "chunk_exp_02", client)
    assert results == []


def test_empty_graph():
    client = _client({})
    results = trace_impact("evt1", "anything", client)
    assert results == []


def test_multiple_dependencies_and_hop_decay():
    client = _client(
        {
            "root": ["a", "b"],
            "a": ["c"],
        }
    )
    results = {r.artifact_id: r for r in trace_impact("evt1", "root", client, decay=0.7, max_hops=3)}
    assert set(results.keys()) == {"a", "b", "c"}
    assert abs(results["a"].impact_score - 0.7) < 1e-9      # hop 1
    assert abs(results["b"].impact_score - 0.7) < 1e-9      # hop 1
    assert abs(results["c"].impact_score - 0.7 ** 2) < 1e-9  # hop 2


def test_shortest_path_wins_no_duplicate_estimate_per_artifact():
    # "target" is reachable at hop 1 (root->target) AND hop 2 (root->mid->target).
    # It must appear exactly once, scored at hop 1.
    client = _client(
        {
            "root": ["target", "mid"],
            "mid": ["target"],
        }
    )
    results = trace_impact("evt1", "root", client, decay=0.7, max_hops=3)
    target_results = [r for r in results if r.artifact_id == "target"]
    assert len(target_results) == 1
    assert abs(target_results[0].impact_score - 0.7) < 1e-9


def test_max_hops_bound_respected():
    client = _client({"root": ["a"], "a": ["b"], "b": ["c"]})
    results = {r.artifact_id for r in trace_impact("evt1", "root", client, max_hops=2)}
    assert results == {"a", "b"}  # "c" is 3 hops away, excluded


def test_changed_artifact_itself_never_in_results():
    client = _client({"root": ["root"]})  # self-loop, defensive case
    results = trace_impact("evt1", "root", client)
    assert all(r.artifact_id != "root" for r in results)


def test_invalid_decay_raises():
    client = _client({})
    try:
        trace_impact("evt1", "x", client, decay=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_invalid_max_hops_raises():
    client = _client({})
    try:
        trace_impact("evt1", "x", client, max_hops=0)
        assert False, "expected ValueError"
    except ValueError:
        pass