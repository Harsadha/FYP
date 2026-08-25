"""
P3 - Quality & Consistency Oracle.

Fully offline. No LLM API required or called anywhere in this module.
Designed so NLI-based consistency and LLM-judge scoring can be added
later as additional, optional scoring paths without changing this
module's public interface (see TODOs at the bottom).

Two responsibilities:
  1. Quality: exact match + token-level F1 of a RAG prediction against
     a gold answer (SQuAD-style).
  2. Consistency: given two or more answers to the same question
     (e.g. drawn from duplicate source chunks), decide whether they
     agree -- via normalized-text equality first, then a generic
     key-value (numeric/currency/percentage token) comparison as a
     fallback. NOT hard-coded to any specific value like "$500"/"$750"
     -- the extraction regex is generic over any numeric token.
"""
import re
import string
from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class QualityResult:
    question: str
    gold: str
    prediction: str
    exact_match: bool
    f1: float


@dataclass
class ConsistencyResult:
    consistent: bool
    method: str  # "normalized_text_match" | "key_value_match" | "mismatch"
    answers: Dict[str, str]  # source_id -> raw answer text
    details: str


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace -- standard
    SQuAD-style normalization."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, gold: str) -> bool:
    return _normalize(prediction) == _normalize(gold)


def f1_score(prediction: str, gold: str) -> float:
    """Token-level F1 over normalized, whitespace-split tokens."""
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common: Dict[str, int] = {}
    for tok in pred_tokens:
        if tok in gold_tokens:
            common[tok] = common.get(tok, 0) + 1
    # num_same = count of overlapping tokens, respecting multiplicity
    gold_counts: Dict[str, int] = {}
    for tok in gold_tokens:
        gold_counts[tok] = gold_counts.get(tok, 0) + 1
    pred_counts: Dict[str, int] = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    num_same = sum(min(c, gold_counts.get(tok, 0)) for tok, c in pred_counts.items())

    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_quality(question: str, gold: str, prediction: str) -> QualityResult:
    return QualityResult(
        question=question,
        gold=gold,
        prediction=prediction,
        exact_match=exact_match(prediction, gold),
        f1=f1_score(prediction, gold),
    )


# Generic numeric/currency/percentage token extractor -- deliberately
# NOT specific to any single example value. Matches things like "$500",
# "750", "30 days" (captures "30"), "12.5%".
_KEY_VALUE_PATTERN = re.compile(r"\$?\d+(?:\.\d+)?%?")


def _extract_key_values(text: str) -> Set[str]:
    raw_matches = _KEY_VALUE_PATTERN.findall(text)
    return {m.replace("$", "").replace("%", "") for m in raw_matches}


def consistency_check(answers: Dict[str, str]) -> ConsistencyResult:
    """
    Args:
        answers: mapping of source_id (e.g. artifact_id or source
            document name) to the answer text obtained from that
            source, for the same question.

    Returns:
        ConsistencyResult with consistent=True if all normalized
        answers match, or (fallback) if all sources' extracted
        numeric/currency/percentage tokens agree. False otherwise.

    With fewer than 2 answers, trivially consistent (nothing to compare).
    """
    if len(answers) < 2:
        return ConsistencyResult(
            consistent=True,
            method="normalized_text_match",
            answers=dict(answers),
            details="Fewer than 2 answers supplied -- nothing to compare.",
        )

    normalized = {src: _normalize(text) for src, text in answers.items()}
    if len(set(normalized.values())) == 1:
        return ConsistencyResult(
            consistent=True,
            method="normalized_text_match",
            answers=dict(answers),
            details="All normalized answers are identical.",
        )

    key_values = {src: _extract_key_values(text) for src, text in answers.items()}
    non_empty = {src: kv for src, kv in key_values.items() if kv}
    if len(non_empty) == len(answers) and len({frozenset(kv) for kv in non_empty.values()}) == 1:
        shared = next(iter(non_empty.values()))
        return ConsistencyResult(
            consistent=True,
            method="key_value_match",
            answers=dict(answers),
            details=f"All sources share the same extracted key value(s): {sorted(shared)}",
        )

    return ConsistencyResult(
        consistent=False,
        method="mismatch",
        answers=dict(answers),
        details=f"Normalized text differs and extracted key values differ: {key_values}",
    )


# TODO(P3, post-30%): add an optional NLI-based consistency path
# (e.g. DeBERTa-v3-MNLI) as an alternative to the key-value heuristic
# above, selectable via a `method="nli"` parameter -- NOT implemented
# here since it would require a model download / API, and the current
# scope requires the Oracle to work fully offline.
# TODO(P3, post-30%): add an optional LLM-judge quality path for
# production queries without a gold answer -- also intentionally
# deferred, same offline-first reasoning.