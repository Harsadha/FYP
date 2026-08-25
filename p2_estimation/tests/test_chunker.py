import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chunker import chunk

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus")


def _read(name):
    with open(os.path.join(CORPUS_DIR, name), encoding="utf-8") as f:
        return f.read()


def test_empty_input_returns_empty_list():
    assert chunk("") == []
    assert chunk("   \n\n  ") == []


def test_chunks_split_on_paragraph_boundaries():
    text = "Para one.\n\nPara two.\n\nPara three."
    result = chunk(text)

    assert result == [
        "Para one.",
        "Para two.",
        "Para three.",
    ]


def test_long_paragraph_falls_back_to_token_window():
    long_para = " ".join(f"word{i}" for i in range(450))

    result = chunk(long_para, max_tokens=200)

    assert len(result) == 3
    assert len(result[0].split()) == 200
    assert len(result[1].split()) == 200
    assert len(result[2].split()) == 50


def test_all_corpus_documents_chunk_correctly():
    filenames = sorted(
        name
        for name in os.listdir(CORPUS_DIR)
        if os.path.isfile(os.path.join(CORPUS_DIR, name))
    )

    assert filenames, "Corpus directory is empty"

    for filename in filenames:
        text = _read(filename)
        result = chunk(text)

        assert len(result) > 0, f"{filename} produced no chunks"

        for c in result:
            assert isinstance(c, str), f"{filename} produced a non-string chunk"
            assert c.strip(), f"{filename} produced an empty chunk"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
