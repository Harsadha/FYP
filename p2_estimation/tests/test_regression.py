"""
Day 3, 11:30-13:00: concrete regression test cases.

Each case is a real document edit against the real corpus, with an
*expected* impacted-chunk set verified by hand (per the sprint plan --
this becomes the seed for the Month 6 evaluation harness, not a
disposable sprint artifact).

Uses the real embedder (sentence-transformers), so this is slower than
the mock-based Day 1 tests and is meant to be run deliberately, not on
every save.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import embedder
from chunker import chunk
from embedder import embed
from similarity_estimator import estimate_impact

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus")

# Threshold matches the greedy optimizer's default (see /p2-estimation/README.md)
SIMILARITY_THRESHOLD = 0.5


def _load_corpus_chunks():
    """Returns {artifact_id: chunk_text} across the whole corpus."""
    chunks_by_id = {}
    for fname in sorted(os.listdir(CORPUS_DIR)):
        if not fname.endswith(".txt"):
            continue
        doc_id = fname.replace(".txt", "")
        with open(os.path.join(CORPUS_DIR, fname), encoding="utf-8") as f:
            text = f.read()
        for i, c in enumerate(chunk(text)):
            chunks_by_id[f"{doc_id}::chunk_{i}"] = c
    return chunks_by_id


def _impacted_above_threshold(changed_id, chunks_by_id, threshold=SIMILARITY_THRESHOLD):
    ids = list(chunks_by_id.keys())
    texts = [chunks_by_id[i] for i in ids]
    vectors = embed(texts)
    embeddings = dict(zip(ids, vectors))

    changed_embedding = embeddings[changed_id]
    candidates = [(cid, embeddings[cid]) for cid in ids if cid != changed_id]

    results = estimate_impact("evt_regression", changed_embedding, candidates)
    return {r.artifact_id for r in results if r.impact_score >= threshold}


# --- Regression cases -------------------------------------------------
# Each expected set was verified by hand: read both chunks, judge
# whether a human would call them "about the same thing."

# Chunk indices: each doc's chunk_0 is its title line (chunker splits on
# blank-line paragraphs and the title is its own paragraph); chunk_1+ are
# body paragraphs in source order.

REGRESSION_CASES = [
    {
        "name": "vpn_otp_auth_chunk_impacts_password_reset_otp_auth_chunk",
        "changed_id": "doc_vpn_setup::chunk_2",  # gateway address / OTP auth para
        "expect_impacted": {"doc_password_reset::chunk_2"},  # also OTP-based auth
        "expect_not_impacted": {"doc_pto_policy::chunk_1", "doc_expense_reports::chunk_1"},
    },
    {
        "name": "pto_accrual_chunk_does_not_impact_unrelated_docs",
        "changed_id": "doc_pto_policy::chunk_1",  # accrual rates para
        "expect_impacted": set(),  # nothing else in corpus is PTO-related
        "expect_not_impacted": {"doc_vpn_setup::chunk_1", "doc_expense_reports::chunk_1"},
    },
    {
        "name": "expense_approval_chunk_does_not_impact_vpn_docs",
        "changed_id": "doc_expense_reports::chunk_2",  # manager approval threshold para
        "expect_impacted": set(),
        "expect_not_impacted": {"doc_vpn_setup::chunk_1", "doc_vpn_setup::chunk_2"},
    },
]


def test_regression_cases():
    # These expected sets were hand-verified against real semantic
    # (sentence-transformer) embeddings. The bag-of-words offline
    # fallback in embedder.py is deliberately lower quality and will
    # not reproduce these judgments -- skip rather than false-fail.
    embedder.embed(["warm up model load"])
    if embedder._model is None:
        pytest.skip(
            "sentence-transformers model unavailable (no network to "
            "huggingface.co) -- regression assertions require the real "
            "semantic embedder, not the bag-of-words fallback."
        )

    chunks_by_id = _load_corpus_chunks()

    for case in REGRESSION_CASES:
        impacted = _impacted_above_threshold(case["changed_id"], chunks_by_id)

        missing = case["expect_impacted"] - impacted
        assert not missing, f"{case['name']}: expected impacted but missing: {missing}"

        unexpected = case["expect_not_impacted"] & impacted
        assert not unexpected, f"{case['name']}: expected NOT impacted but got: {unexpected}"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
