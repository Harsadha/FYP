import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from p3_optimization.evaluation_harness import (
    build_expense_demo_scenario,
    run_strategy,
    compare_strategies,
    STRATEGIES,
)


def test_all_strategies_run_without_error():
    scenario = build_expense_demo_scenario()
    results = compare_strategies(scenario)
    assert set(results.keys()) == set(STRATEGIES)


def test_no_maintenance_leaves_faq_stale_and_inconsistent():
    scenario = build_expense_demo_scenario()
    result = run_strategy(scenario, "no_maintenance")
    consistency_results = [qr.consistency for qr in result.question_results if qr.consistency is not None]
    assert consistency_results[0].consistent is False
    assert result.total_cost == 0.0


def test_combined_threshold_baseline_invalidates_the_stale_duplicate():
    # Documented, real behavior with the default 0.8/0.4 thresholds:
    # chunk_faq_17's aggregated score (~0.59 -- a 1-hop TRACE match at
    # decay=0.7 blended with a moderate SIMILARITY score) lands in the
    # invalidate band, not the update band. The threshold baseline
    # suppresses the stale source rather than correcting it; with only
    # one surviving answer, consistency is trivially satisfied (nothing
    # left to disagree). This is a real, useful finding for the report,
    # not a bug -- see optimizer.py's comparison against "optimized" below.
    scenario = build_expense_demo_scenario()
    result = run_strategy(scenario, "combined")
    actions = {p.artifact_id: p.action for p in result.plan}
    assert actions["chunk_faq_17"] == "invalidate"
    consistency_results = [qr.consistency for qr in result.question_results if qr.consistency is not None]
    assert consistency_results[0].consistent is True
    assert consistency_results[0].method == "normalized_text_match"  # trivial: <2 answers to compare


def test_optimized_strategy_actually_corrects_the_duplicate_not_just_suppresses_it():
    # This is the key comparison against the threshold baseline above:
    # invalidate's quality-recovery efficacy (0.6) is not enough to
    # satisfy quality_threshold=0.8 at this artifact's combined score,
    # so the CP-SAT solver is forced to choose the costlier "update"
    # instead -- and the two duplicated chunks now genuinely agree
    # (key_value_match), not just "nothing left to compare."
    scenario = build_expense_demo_scenario()
    result = run_strategy(scenario, "optimized")
    actions = {p.artifact_id: p.action for p in result.plan}
    assert actions["chunk_faq_17"] == "update"
    consistency_results = [qr.consistency for qr in result.question_results if qr.consistency is not None]
    assert consistency_results[0].consistent is True
    assert consistency_results[0].method == "key_value_match"  # genuinely reconciled, not just suppressed


def test_unrelated_chunk_never_flagged():
    scenario = build_expense_demo_scenario()
    result = run_strategy(scenario, "combined")
    touched_ids = {p.artifact_id for p in result.plan}
    assert "chunk_exp_01" not in touched_ids
    assert "chunk_exp_03" not in touched_ids


def test_maintenance_cost_calculation_naive_vs_selective():
    scenario = build_expense_demo_scenario()
    naive = run_strategy(scenario, "update_everything")
    selective = run_strategy(scenario, "combined")
    # selective touches a subset of what naive touches -> cost should not exceed naive's
    assert selective.total_cost <= naive.total_cost


def test_quality_score_reflects_the_new_threshold_after_fix():
    # The mocked "RAG answer" is the raw chunk sentence, not a terse
    # generated answer -- so exact_match against a short gold value
    # like "$750" is not a meaningful check here (it will essentially
    # never be True against a full sentence). F1 (token overlap) is the
    # honest signal the mock can provide; a real RAG pipeline producing
    # short answers would make exact_match meaningful too.
    scenario = build_expense_demo_scenario()
    result = run_strategy(scenario, "combined")
    quality_results = [qr.quality for qr in result.question_results if qr.quality is not None]
    assert quality_results[0].f1 > 0.0
    assert "750" in quality_results[0].prediction


def test_unknown_strategy_raises():
    scenario = build_expense_demo_scenario()
    try:
        run_strategy(scenario, "not_a_real_strategy")
        assert False, "expected ValueError"
    except ValueError:
        pass