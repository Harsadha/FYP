import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from p2_estimation.models import ImpactEstimate
from p3_optimization.cost_table import CostTable
from p3_optimization.optimizer import (
    aggregate_estimates,
    threshold_based_plan,
    optimize_maintenance_plan,
)


def _est(artifact_id, score, estimator="TRACE", confidence=1.0, artifact_type="CHUNK"):
    return ImpactEstimate(
        change_event_id="evt1",
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        impact_score=score,
        estimator=estimator,
        confidence=confidence,
    )


# --- threshold baseline --------------------------------------------------

def test_high_impact_updates():
    plan = threshold_based_plan([_est("a", 0.95)])
    assert plan[0].action == "update"


def test_medium_impact_invalidates():
    plan = threshold_based_plan([_est("a", 0.5)])
    assert plan[0].action == "invalidate"


def test_low_impact_retains():
    plan = threshold_based_plan([_est("a", 0.1)])
    assert plan[0].action == "retain"


def test_empty_input_threshold():
    assert threshold_based_plan([]) == []


def test_deterministic_threshold_output():
    ests = [_est("a", 0.9), _est("b", 0.5), _est("c", 0.1)]
    plan1 = threshold_based_plan(ests)
    plan2 = threshold_based_plan(ests)
    assert [(p.artifact_id, p.action) for p in plan1] == [(p.artifact_id, p.action) for p in plan2]


def test_invalid_threshold_ordering_raises():
    try:
        threshold_based_plan([], update_threshold=0.3, invalidate_threshold=0.5)
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- aggregation of multiple estimators for the same artifact -----------

def test_multiple_estimators_same_artifact_aggregated_not_duplicated():
    ests = [
        _est("chunk_faq_17", 1.0, estimator="TRACE", confidence=1.0),
        _est("chunk_faq_17", 0.94, estimator="SIMILARITY", confidence=0.9),
    ]
    aggregated = aggregate_estimates(ests)
    assert len(aggregated) == 1
    agg = aggregated[0]
    assert agg.artifact_id == "chunk_faq_17"
    assert set(agg.source_estimators) == {"TRACE", "SIMILARITY"}
    # confidence-weighted average: (1.0*1.0 + 0.94*0.9) / (1.0+0.9)
    expected = (1.0 * 1.0 + 0.94 * 0.9) / (1.0 + 0.9)
    assert abs(agg.combined_score - expected) < 1e-9


def test_aggregate_estimates_order_independent():
    a = [_est("x", 0.8, "TRACE"), _est("x", 0.6, "SIMILARITY")]
    b = list(reversed(a))
    assert aggregate_estimates(a)[0].combined_score == aggregate_estimates(b)[0].combined_score


def test_duplicate_same_estimator_estimates_still_aggregate_cleanly():
    ests = [_est("x", 0.9, "TRACE"), _est("x", 0.3, "TRACE")]
    agg = aggregate_estimates(ests)
    assert len(agg) == 1
    assert agg[0].source_estimators == ["TRACE"]


# --- CP-SAT optimizer -----------------------------------------------------

def test_optimizer_empty_input():
    result = optimize_maintenance_plan([])
    assert result.plan == []
    assert result.solution_quality == "empty"
    assert result.total_cost == 0.0


def test_optimizer_high_impact_gets_updated():
    ests = [_est("chunk_faq_17", 1.0, "TRACE", 1.0), _est("chunk_faq_17", 0.94, "SIMILARITY", 0.9)]
    result = optimize_maintenance_plan(ests, quality_threshold=0.8)
    assert result.solution_quality in ("optimal", "feasible")
    assert len(result.plan) == 1
    assert result.plan[0].artifact_id == "chunk_faq_17"
    assert result.plan[0].action == "update"


def test_optimizer_retains_when_quality_threshold_is_zero():
    # At quality_threshold=0.0 the constraint sum(...) >= 0 is trivially
    # satisfied by retain-everything (LHS=0, RHS=0), so the cheapest
    # action (retain, cost 0) wins. Note quality_threshold=0.1 does NOT
    # give this result even for a low-impact artifact: the constraint
    # is proportional to total impact, so any positive threshold still
    # requires some non-retain action for the only artifact present.
    ests = [_est("unrelated_chunk", 0.02, "SIMILARITY", 0.9)]
    result = optimize_maintenance_plan(ests, quality_threshold=0.0)
    assert result.plan[0].action == "retain"


def test_optimizer_respects_configured_cost_table():
    ests = [_est("a", 0.9, "TRACE", 1.0)]
    ct = CostTable()
    ct.set_cost("CHUNK", "update", 100.0)  # make update artificially expensive
    ct.set_cost("CHUNK", "invalidate", 0.01)  # make invalidate artificially cheap
    result = optimize_maintenance_plan(ests, cost_table=ct, quality_threshold=0.5,
                                        efficacy={"update": 1.0, "invalidate": 1.0, "retain": 0.0})
    # with invalidate just as effective at satisfying quality and far cheaper,
    # the optimizer should prefer it over update.
    assert result.plan[0].action == "invalidate"


def test_optimizer_deterministic_output():
    ests = [_est("a", 0.9, "TRACE"), _est("b", 0.5, "SIMILARITY"), _est("c", 0.1, "TRACE")]
    r1 = optimize_maintenance_plan(ests)
    r2 = optimize_maintenance_plan(ests)
    assert [(p.artifact_id, p.action) for p in r1.plan] == [(p.artifact_id, p.action) for p in r2.plan]


def test_optimizer_total_cost_matches_plan():
    ests = [_est("a", 0.9, "TRACE")]
    ct = CostTable()
    result = optimize_maintenance_plan(ests, cost_table=ct)
    expected_cost = sum(ct.cost("CHUNK", p.action) for p in result.plan)
    assert abs(result.total_cost - expected_cost) < 1e-9


def test_optimizer_never_double_assigns_an_action():
    ests = [_est("a", 0.9, "TRACE"), _est("a", 0.2, "SIMILARITY")]
    result = optimize_maintenance_plan(ests)
    assert len(result.plan) == 1  # aggregated, not duplicated