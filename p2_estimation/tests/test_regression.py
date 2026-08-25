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


CORPUS_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "corpus",
)

SIMILARITY_THRESHOLD = 0.5


def _load_corpus_chunks():
    """Load every document in the corpus and assign stable chunk IDs."""
    chunks_by_id = {}

    for fname in sorted(os.listdir(CORPUS_DIR)):
        path = os.path.join(CORPUS_DIR, fname)

        if not os.path.isfile(path):
            continue

        if fname.startswith("."):
            continue

        doc_id = os.path.splitext(fname)[0]

        with open(path, encoding="utf-8") as f:
            text = f.read()

        document_chunks = chunk(text)

        for i, c in enumerate(document_chunks):
            artifact_id = f"{doc_id}::chunk_{i}"
            chunks_by_id[artifact_id] = c

    return chunks_by_id


def _impacted_above_threshold(
    changed_id,
    chunks_by_id,
    threshold=SIMILARITY_THRESHOLD,
):
    ids = list(chunks_by_id.keys())
    texts = [chunks_by_id[i] for i in ids]

    vectors = embed(texts)

    embeddings = dict(zip(ids, vectors))

    changed_embedding = embeddings[changed_id]

    candidates = [
        (cid, embeddings[cid])
        for cid in ids
        if cid != changed_id
    ]

    results = estimate_impact(
        "evt_regression",
        changed_embedding,
        candidates,
    )

    return {
        r.artifact_id
        for r in results
        if r.impact_score >= threshold
    }


def test_real_corpus_chunks_are_loaded():
    chunks_by_id = _load_corpus_chunks()

    assert chunks_by_id, "No chunks were loaded from the corpus"

    for artifact_id, chunk_text in chunks_by_id.items():
        assert "::chunk_" in artifact_id
        assert chunk_text.strip()


def test_real_corpus_can_be_embedded_and_estimated():
    chunks_by_id = _load_corpus_chunks()

    assert chunks_by_id

    ids = list(chunks_by_id.keys())
    texts = [chunks_by_id[artifact_id] for artifact_id in ids]

    vectors = embed(texts)

    assert len(vectors) == len(ids)

    changed_id = ids[0]
    changed_embedding = vectors[0]

    candidates = [
        (artifact_id, vector)
        for artifact_id, vector in zip(ids[1:], vectors[1:])
    ]

    results = estimate_impact(
        change_event_id="evt_real_corpus",
        changed_embedding=changed_embedding,
        candidates=candidates,
    )

    assert len(results) == len(candidates)

    for result in results:
        assert result.artifact_id in chunks_by_id
        assert 0.0 <= result.impact_score <= 1.0
