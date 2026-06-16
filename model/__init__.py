"""Model package: GPT-style decoder LM built around spiking attention."""

from __future__ import annotations

from model.attention import (
    SpikingAttention,
    StandardAttention,
    build_attention,
)
from model.mlp import MLP
from model.model import LanguageModel, ModelConfig
from model.norms import RMSNorm
from model.transformer_block import TransformerBlock

__all__ = [
    "SpikingAttention",
    "StandardAttention",
    "build_attention",
    "MLP",
    "LanguageModel",
    "ModelConfig",
    "RMSNorm",
    "TransformerBlock",
]
