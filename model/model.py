"""GPT-style decoder-only language model built around spiking attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from model.norms import RMSNorm
from model.transformer_block import TransformerBlock


@dataclass
class ModelConfig:
    """Hyperparameters describing the language model architecture."""

    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    max_seq_len: int = 1024
    mlp_ratio: int = 4
    mlp_hidden_dim: int | None = None
    attn_type: str = "spiking"
    n_timesteps: int = 1
    causal: bool = True
    dropout: float = 0.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Build a config from a dictionary, ignoring unknown keys.

        Args:
            data: Mapping of field names to values.

        Returns:
            A populated :class:`ModelConfig`.
        """
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in fields})


class LanguageModel(nn.Module):
    """Decoder-only transformer for autoregressive language modeling."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the language model.

        Args:
            config: Architecture hyperparameters.
        """
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    mlp_ratio=config.mlp_ratio,
                    mlp_hidden_dim=config.mlp_hidden_dim,
                    attn_type=config.attn_type,
                    n_timesteps=config.n_timesteps,
                    causal=config.causal,
                    dropout=config.dropout,
                    norm_eps=config.norm_eps,
                )
                for _ in range(config.n_layers)
            ]
        )

        self.norm_final = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Apply GPT-style weight initialization.

        Args:
            module: Module to initialize.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters.

        Args:
            trainable_only: Whether to count only parameters that require grad.

        Returns:
            The number of parameters.
        """
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Compute next-token logits for a batch of token sequences.

        Args:
            input_ids: Long tensor of token ids with shape ``[B, S]``.

        Returns:
            Logits tensor with shape ``[B, S, vocab_size]``.
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"Expected input_ids of shape [B, S], got {tuple(input_ids.shape)}."
            )
        _, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len "
                f"{self.config.max_seq_len}."
            )

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits
