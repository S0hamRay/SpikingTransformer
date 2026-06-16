"""A single GPT-style decoder block."""

from __future__ import annotations

from torch import Tensor, nn

from model.attention import build_attention
from model.mlp import MLP
from model.norms import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-norm decoder block with a spiking attention sublayer.

    The block applies two residual sublayers following the standard pre-norm
    layout::

        x = x + Attn(RMSNorm(x))
        x = x + MLP(RMSNorm(x))

    The attention module is injected so alternative mechanisms can be swapped in
    without touching the block.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: int = 4,
        mlp_hidden_dim: int | None = None,
        attn_type: str = "spiking",
        n_timesteps: int = 1,
        causal: bool = True,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ) -> None:
        """Initialize the decoder block.

        Args:
            d_model: Token embedding dimension.
            n_heads: Number of attention heads.
            mlp_ratio: Expansion ratio for the MLP hidden dimension.
            mlp_hidden_dim: Explicit MLP hidden dimension (overrides ``mlp_ratio``).
            attn_type: Which attention mechanism to use (``"spiking"`` or
                ``"standard"``).
            n_timesteps: Number of spiking timesteps for the attention module.
            causal: Whether attention is causally masked.
            dropout: Residual dropout probability.
            norm_eps: Epsilon for the RMSNorm layers.
        """
        super().__init__()
        self.norm_attn = RMSNorm(d_model, eps=norm_eps)
        self.attn = build_attention(
            attn_type=attn_type,
            d_model=d_model,
            n_heads=n_heads,
            n_timesteps=n_timesteps,
            causal=causal,
            dropout=dropout,
        )
        self.norm_mlp = RMSNorm(d_model, eps=norm_eps)
        self.mlp = MLP(
            d_model=d_model,
            hidden_dim=mlp_hidden_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Apply attention and feed-forward residual sublayers.

        Args:
            x: Input tensor with shape ``[B, S, D]``.

        Returns:
            Output tensor with shape ``[B, S, D]``.
        """
        x = x + self.dropout(self.attn(self.norm_attn(x)))
        x = x + self.dropout(self.mlp(self.norm_mlp(x)))
        return x
