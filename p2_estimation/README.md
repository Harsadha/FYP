# P2 — Estimation

Deterministic chunking + embedding + semantic-similarity impact estimation,
per the 3-day sprint plan (Month 1 + first half of Month 2 of the 7-month
roadmap).

## Files

| File | Role |
|---|---|
| `models.py` | `ImpactEstimate` dataclass, matches `/schemas/impact_estimate.json` exactly |
| `chunker.py` | `chunk(text) -> List[str]` — paragraph split, token-window fallback |
| `embedder.py` | `embed(chunks) -> List[vector]` — sentence-transformers, CPU |
| `similarity_estimator.py` | `estimate_impact(...) -> List[ImpactEstimate]` — cosine similarity |
| `graph_client.py` | Interface P2 expects from P1's graph; `MockGraphClient` for Day 1 |
| `corpus/` | Placeholder sample docs (swap for the real 20-30 doc TechQA subset) |
| `tests/test_chunker.py` | Day 1 unit tests |
| `tests/test_similarity_estimator.py` | Day 1 mock-data tests + Day 2 error-handling tests |
| `tests/test_regression.py` | Day 3 hand-verified regression cases on real corpus |

## Running

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -v
```

## Threshold choice

`SIMILARITY_THRESHOLD = 0.5` (in `tests/test_regression.py`, matches the
greedy optimizer's default from the Day 1 schema freeze). This was picked
by eyeballing 3-4 example pairs on Day 2 (see Day 2 sync notes) — a
same-topic pair (VPN OTP auth vs. password-reset OTP auth) scored ~0.7-0.9
after rescaling, unrelated pairs scored ~0.4-0.5. It has **not** been
tuned against a real precision/recall target; that's Month 6 evaluation
work. Treat 0.5 as a reasonable starting point, not a validated cutoff.

## Known limitations (explicitly out of scope for this milestone)

- **Similarity-only.** No ML-learned estimator (LightGBM/GraphSAGE) —
  that's Month 3-4 work per the roadmap.
- **Small corpus.** 4 placeholder docs here; real milestone uses a
  20-30 doc IBM TechQA subset. Similarity scores on a larger, more
  diverse corpus will behave differently — thresholds may need
  re-tuning.
- **Naive chunker.** Paragraph-boundary split with a word-count proxy
  for tokens. No sentence-boundary detection, no overlap between
  chunks, no semantic chunking.
- **Confidence is a fixed constant (0.9)** for this estimator, not
  calibrated. It exists so downstream consumers (the optimizer) have
  a field to combine across estimators, per the frozen schema — the
  number itself carries no statistical meaning yet.
- **Cosine similarity is rescaled from [-1,1] to [0,1]** to fit the
  `impact_score` schema range. This means a `0.5` score is "orthogonal
  in embedding space," not "50% likely to matter" — worth remembering
  when reading raw scores.
- **Offline fallback embedder.** If `sentence-transformers` can't
  reach huggingface.co to download model weights, `embedder.py` falls
  back to a deterministic bag-of-words vector so the rest of the
  pipeline doesn't hard-crash. This fallback is degraded quality and
  is not a substitute for the real model — `tests/test_regression.py`
  detects this case and skips its semantic assertions rather than
  false-failing. Don't demo on the fallback; confirm the real model
  loaded first.

## Seed for the Month 6 evaluation harness

`tests/test_regression.py::REGRESSION_CASES` are 3 hand-verified document
edits with expected impacted-chunk sets. This is intentionally reusable —
extend this list rather than replacing it when building the real
evaluation harness later in the roadmap.
