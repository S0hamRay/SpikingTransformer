"""Tests for autoregressive generation."""

from __future__ import annotations

import torch

from data.tokenizer import ByteTokenizer
from eval.generation import generate, generate_text
from model.model import LanguageModel, ModelConfig


def _tiny_model() -> LanguageModel:
    config = ModelConfig(
        vocab_size=ByteTokenizer().vocab_size,
        d_model=16,
        n_layers=2,
        n_heads=2,
        max_seq_len=32,
        n_timesteps=1,
        causal=True,
        dropout=0.0,
    )
    torch.manual_seed(0)
    return LanguageModel(config).eval()


def test_generate_length_and_prefix() -> None:
    model = _tiny_model()
    prompt = torch.randint(0, 256, (1, 4))
    out = generate(model, prompt, max_new_tokens=10, greedy=True)
    assert out.shape == (1, 14)
    # The prompt is preserved as a prefix.
    assert torch.equal(out[:, :4], prompt)


def test_greedy_is_deterministic() -> None:
    model = _tiny_model()
    prompt = torch.randint(0, 256, (1, 4))
    out_a = generate(model, prompt, max_new_tokens=10, greedy=True)
    out_b = generate(model, prompt, max_new_tokens=10, greedy=True)
    assert torch.equal(out_a, out_b)


def test_sampling_stays_in_vocab() -> None:
    model = _tiny_model()
    vocab_size = model.config.vocab_size
    prompt = torch.randint(0, 256, (2, 4))
    torch.manual_seed(123)
    out = generate(
        model, prompt, max_new_tokens=10, temperature=1.0, top_k=20, top_p=0.9
    )
    assert int(out.max()) < vocab_size
    assert int(out.min()) >= 0


def test_context_cropping_beyond_max_seq_len() -> None:
    model = _tiny_model()  # max_seq_len=32
    prompt = torch.randint(0, 256, (1, 30))
    out = generate(model, prompt, max_new_tokens=10, greedy=True)
    # Generation should not crash even though total length exceeds max_seq_len.
    assert out.shape[1] == 40


def test_generate_text_returns_string() -> None:
    model = _tiny_model()
    tok = ByteTokenizer()
    text = generate_text(
        model, tok, prompt="Once upon a time", max_new_tokens=20, greedy=True
    )
    assert isinstance(text, str)
    assert text.startswith("Once upon a time")
