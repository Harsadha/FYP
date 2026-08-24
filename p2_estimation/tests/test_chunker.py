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
    assert result == ["Para one.", "Para two.", "Para three."]


def test_long_paragraph_falls_back_to_token_window():
    long_para = " ".join(f"word{i}" for i in range(450))
    result = chunk(long_para, max_tokens=200)
    # 450 words / 200-word window -> 3 chunks
    assert len(result) == 3
    assert len(result[0].split()) == 200
    assert len(result[1].split()) == 200
    assert len(result[2].split()) == 50


def test_sample_doc_vpn_setup_chunks_correctly():
    text = _read("doc_vpn_setup.txt")
    result = chunk(text)
    # title line + 3 body paragraphs = 4 chunks
    assert len(result) == 4
    assert result[0] == "Setting Up Company VPN Access"
    assert "VPN client" in result[1]
    assert "gateway address" in result[2]
    assert "split-tunneling" in result[3]


def test_sample_doc_password_reset_chunks_correctly():
    text = _read("doc_password_reset.txt")
    result = chunk(text)
    assert len(result) == 4  # title + 3 body paragraphs
    assert all(isinstance(c, str) and c for c in result)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
