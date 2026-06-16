"""Tests for the tokenizer backends."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from data.tokenizer import ByteTokenizer, build_tokenizer, train_sentencepiece


def test_byte_tokenizer_roundtrip_ascii() -> None:
    tok = ByteTokenizer()
    text = "Hello, world!"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_byte_tokenizer_roundtrip_unicode() -> None:
    tok = ByteTokenizer()
    text = "café — 日本語 — 🌟"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_byte_tokenizer_vocab_and_specials() -> None:
    tok = ByteTokenizer()
    assert tok.vocab_size == 259
    specials = {tok.bos_id, tok.eos_id, tok.pad_id}
    assert len(specials) == 3
    assert all(s is not None and s >= 256 for s in specials)


def test_byte_tokenizer_add_bos_eos() -> None:
    tok = ByteTokenizer()
    ids = tok.encode("hi", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    # The interior bytes still decode to the original text.
    assert tok.decode(ids) == "hi"


def test_encode_batch() -> None:
    tok = ByteTokenizer()
    batch = tok.encode_batch(["a", "bb", "ccc"])
    assert [len(x) for x in batch] == [1, 2, 3]


def test_build_tokenizer_unknown_backend() -> None:
    with pytest.raises(ValueError):
        build_tokenizer(backend="does-not-exist")


@pytest.mark.skipif(
    importlib.util.find_spec("sentencepiece") is None,
    reason="sentencepiece not installed",
)
def test_sentencepiece_train_and_roundtrip(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    lines = ["the quick brown fox jumps over the lazy dog"] * 200
    corpus.write_text("\n".join(lines), encoding="utf-8")

    model_prefix = tmp_path / "spm"
    model_file = train_sentencepiece(
        input_path=corpus,
        model_prefix=model_prefix,
        vocab_size=64,
        model_type="bpe",
        character_coverage=1.0,
    )
    tok = build_tokenizer(backend="sentencepiece", model_path=model_file)
    assert tok.vocab_size == 64
    ids = tok.encode("the quick brown fox", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert isinstance(tok.decode(ids), str)
