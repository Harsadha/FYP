"""
P2 - Deterministic chunker (Day 1, 9:30-11:00).

Pure function: chunk(document_text) -> List[str]

Deliberately simple for the sprint: split on blank-line paragraph
boundaries, with a fixed-token-window fallback for any paragraph that's
too long. No sentence-boundary detection, no overlap, no sophistication --
that's explicitly out of scope for the 30% milestone.
"""
from typing import List

DEFAULT_MAX_TOKENS = 200  # word-count used as a rough token-count proxy


def chunk(document_text: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> List[str]:
    if not document_text or not document_text.strip():
        return []

    paragraphs = [p.strip() for p in document_text.split("\n\n") if p.strip()]

    chunks: List[str] = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= max_tokens:
            chunks.append(para)
        else:
            for i in range(0, len(words), max_tokens):
                window = words[i : i + max_tokens]
                chunks.append(" ".join(window))

    return chunks
