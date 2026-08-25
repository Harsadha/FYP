import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from p3_optimization.oracle import exact_match, f1_score, score_quality, consistency_check


def test_exact_match_true():
    assert exact_match("$750", "$750") is True


def test_exact_match_false():
    assert exact_match("$500", "$750") is False


def test_case_normalization():
    assert exact_match("Seven Hundred Fifty", "seven hundred fifty") is True


def test_whitespace_normalization():
    assert exact_match("  $750   dollars ", "$750 dollars") is True


def test_f1_partial_overlap():
    score = f1_score("manager approval required over $750", "$750 approval threshold")
    assert 0.0 < score < 1.0


def test_f1_identical_is_one():
    assert f1_score("$750", "$750") == 1.0


def test_f1_no_overlap_is_zero():
    assert f1_score("completely unrelated text", "totally different words") == 0.0


def test_f1_both_empty_is_one():
    assert f1_score("", "") == 1.0


def test_score_quality_structure():
    result = score_quality("What is the threshold?", "$750", "$750")
    assert result.exact_match is True
    assert result.f1 == 1.0
    assert result.question == "What is the threshold?"


def test_consistency_identical_answers():
    result = consistency_check({"a": "$750 required", "b": "$750 required"})
    assert result.consistent is True
    assert result.method == "normalized_text_match"


def test_consistency_key_value_match_despite_phrasing_difference():
    result = consistency_check(
        {
            "policy": "Manager approval is required for any single line item exceeding $750.",
            "faq": "Yes -- any single expense line item over $750 requires manager sign-off.",
        }
    )
    assert result.consistent is True
    assert result.method == "key_value_match"


def test_consistency_detects_stale_mismatch():
    result = consistency_check(
        {
            "policy": "Manager approval is required for any single line item exceeding $750.",
            "faq": "Yes -- any single expense line item over $500 requires manager sign-off.",
        }
    )
    assert result.consistent is False
    assert result.method == "mismatch"


def test_consistency_single_answer_trivially_consistent():
    result = consistency_check({"a": "$750"})
    assert result.consistent is True


def test_consistency_not_hardcoded_to_specific_values():
    # Same mechanism, totally different domain/numbers -- must still work.
    result = consistency_check({"a": "PTO accrues at 1.5 days per month", "b": "PTO accrues at 2.5 days per month"})
    assert result.consistent is False