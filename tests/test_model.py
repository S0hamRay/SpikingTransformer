"""Tests for the language model forward pass and architecture."""

from __future__ import annotations

import pytest
import torch

from model.attention import SpikingAttention, StandardAttention, build_attention
from model.model import LanguageModel, ModelConfig


def _tiny_config(**overrides: object) -> ModelConfig:
    base = dict(
        vocab_size=37,
        d_model=16,
        n_layers=2,
        n_heads=2,
        max_seq_len=16,
        mlp_ratio=4,
        n_timesteps=1,
        causal=True,
        dropout=0.0,
    )
    base.update(overrides)
    return ModelConfig(**base)  # type: ignore[arg-type]


def test_forward_output_shape() -> None:
    torch.manual_seed(0)
    model = LanguageModel(_tiny_config()).eval()
    input_ids = torch.randint(0, 37, (3, 8))
    logits = model(input_ids)
    assert logits.shape == (3, 8, 37)
    assert torch.isfinite(logits).all()


def test_backward_produces_gradients() -> None:
    torch.manual_seed(0)
    model = LanguageModel(_tiny_config()).train()
    input_ids = torch.randint(0, 37, (2, 8))
    logits = model(input_ids)
    loss = logits.float().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_attention_wrapper_preserves_shape() -> None:
    torch.manual_seed(0)
    attn = SpikingAttention(d_model=16, n_heads=2, n_timesteps=1, causal=True).eval()
    x = torch.randn(2, 8, 16)
    out = attn(x)
    assert out.shape == x.shape


def test_causal_masking_blocks_future_tokens() -> None:
    """In eval mode, logits at position i must not depend on tokens j > i."""
    torch.manual_seed(0)
    model = LanguageModel(_tiny_config()).eval()
    input_ids = torch.randint(0, 37, (1, 10))

    with torch.no_grad():
        logits_a = model(input_ids)
        modified = input_ids.clone()
        modified[0, -1] = (modified[0, -1] + 1) % 37  # change only the last token
        logits_b = model(modified)

    # All positions before the last must be unchanged.
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)


def test_sequence_length_guard() -> None:
    model = LanguageModel(_tiny_config(max_seq_len=8)).eval()
    too_long = torch.randint(0, 37, (1, 9))
    with pytest.raises(ValueError):
        model(too_long)


def test_weight_tying() -> None:
    model = LanguageModel(_tiny_config(tie_embeddings=True))
    assert model.lm_head.weight is model.token_embedding.weight


def test_num_parameters_positive() -> None:
    model = LanguageModel(_tiny_config())
    assert model.num_parameters() > 0


def test_standard_attention_preserves_shape() -> None:
    torch.manual_seed(0)
    attn = StandardAttention(d_model=16, n_heads=2, causal=True).eval()
    x = torch.randn(2, 8, 16)
    out = attn(x)
    assert out.shape == x.shape


def test_build_attention_factory() -> None:
    spiking = build_attention("spiking", d_model=16, n_heads=2)
    standard = build_attention("standard", d_model=16, n_heads=2)
    assert isinstance(spiking, SpikingAttention)
    assert isinstance(standard, StandardAttention)
    with pytest.raises(ValueError):
        build_attention("nope", d_model=16, n_heads=2)


def test_model_with_standard_attention_forward() -> None:
    torch.manual_seed(0)
    model = LanguageModel(_tiny_config(attn_type="standard")).eval()
    input_ids = torch.randint(0, 37, (3, 8))
    logits = model(input_ids)
    assert logits.shape == (3, 8, 37)
    assert torch.isfinite(logits).all()


def test_standard_attention_is_causal() -> None:
    torch.manual_seed(0)
    model = LanguageModel(_tiny_config(attn_type="standard")).eval()
    input_ids = torch.randint(0, 37, (1, 10))
    with torch.no_grad():
        logits_a = model(input_ids)
        modified = input_ids.clone()
        modified[0, -1] = (modified[0, -1] + 1) % 37
        logits_b = model(modified)
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)
