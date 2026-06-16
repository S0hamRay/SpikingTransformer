"""Attention modules for the language model.

The novel spiking attention mechanism (``A2OS2A``) lives at the repository root
in ``attention.py`` and is treated as a black-box building block. This module
provides a thin, swappable wrapper, :class:`SpikingAttention`, that adapts it to
the standard ``[batch, seq, dim]`` interface used by the rest of the transformer
and adds the causal masking required for autoregressive language modeling.

The original ``A2OS2A`` class is never modified. The causal variant below is a
*subclass* that reuses the parent's exact projections and spiking neurons and
only inserts a causal mask between the score and value products. Swapping in a
different attention mechanism is as simple as providing another ``nn.Module``
with a matching ``forward(x: Tensor) -> Tensor`` signature, where ``x`` has shape
``[batch, seq, dim]``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from attention import A2OS2A
from spiking_neuron import LIFNeuron, TernaryLIFNeuron


class _CausalA2OS2A(A2OS2A):
    """``A2OS2A`` extended with an optional causal mask.

    This subclass leaves the parent attention algorithm untouched: it reuses the
    same query/key/value projections, batch norms, spiking neurons, and output
    projection. The only addition is a lower-triangular mask applied to the
    raw attention scores so that a token at position ``i`` cannot attend to
    positions ``j > i``. Because the underlying attention is additive/linear
    (no softmax), masking is equivalent to zeroing future score contributions.
    """

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        """Apply (optionally causal) spiking attention.

        Args:
            x: Input tensor with shape ``[T, B, N, D]`` where ``T`` is the number
                of spiking timesteps, ``B`` the batch size, ``N`` the sequence
                length, and ``D`` the embedding dimension.
            causal: When ``True`` a causal mask is applied to the scores.

        Returns:
            Output tensor with shape ``[T, B, N, D]``.
        """
        timesteps, batch_size, num_tokens, embed_dim = x.shape
        outputs: list[Tensor] = []

        if causal:
            # [1, 1, N, N] lower-triangular keep-mask shared across heads/batch.
            causal_mask = torch.tril(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=x.dtype)
            ).view(1, 1, num_tokens, num_tokens)
        else:
            causal_mask = None

        for t in range(timesteps):
            xt = x[t]

            q = self.W_Q(xt)
            q = self.bn_Q(q.reshape(batch_size * num_tokens, embed_dim)).reshape(
                batch_size, num_tokens, embed_dim
            )
            q = self.sn_Q(q)

            k = self.W_K(xt)
            k = self.bn_K(k.reshape(batch_size * num_tokens, embed_dim)).reshape(
                batch_size, num_tokens, embed_dim
            )
            k = self.relu_K(k)

            v = self.W_V(xt)
            v = self.bn_V(v.reshape(batch_size * num_tokens, embed_dim)).reshape(
                batch_size, num_tokens, embed_dim
            )
            v = self.sn_V(v)

            q = q.reshape(batch_size, num_tokens, self.num_heads, self.head_dim).permute(
                0, 2, 1, 3
            )
            k = k.reshape(batch_size, num_tokens, self.num_heads, self.head_dim).permute(
                0, 2, 1, 3
            )
            v = v.reshape(batch_size, num_tokens, self.num_heads, self.head_dim).permute(
                0, 2, 1, 3
            )

            attn = q @ k.transpose(-2, -1)
            if causal_mask is not None:
                attn = attn * causal_mask
            out = attn @ v
            out = out.permute(0, 2, 1, 3).reshape(batch_size, num_tokens, embed_dim)
            outputs.append(out)

        out = torch.stack(outputs, dim=0)

        # Conv2d expects channels-first tensors, so move D into the channel axis.
        out = out.permute(0, 1, 3, 2).reshape(
            timesteps * batch_size, embed_dim, num_tokens, 1
        )
        out = self.bn_proj(self.proj(out))
        out = out.reshape(timesteps, batch_size, embed_dim, num_tokens).permute(0, 1, 3, 2)
        return out.contiguous()


class SpikingAttention(nn.Module):
    """Swappable adapter around the spiking ``A2OS2A`` attention.

    Exposes the standard transformer attention interface
    ``forward(x: [B, S, D]) -> [B, S, D]`` while internally managing the spiking
    timestep dimension and resetting the stateful neurons on every call so that
    sequences in a batch do not leak membrane state into one another.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_timesteps: int = 1,
        causal: bool = True,
    ) -> None:
        """Initialize the attention adapter.

        Args:
            d_model: Token embedding dimension.
            n_heads: Number of attention heads.
            n_timesteps: Number of spiking timesteps to unroll. The static input
                is repeated across timesteps and the outputs are averaged.
            causal: Whether to apply a causal mask (required for autoregressive
                language modeling).
        """
        super().__init__()
        if n_timesteps < 1:
            raise ValueError("n_timesteps must be >= 1.")
        self.n_timesteps = n_timesteps
        self.causal = causal
        self.core = _CausalA2OS2A(embed_dim=d_model, num_heads=n_heads)

    def reset_state(self) -> None:
        """Reset the persistent membrane state of all spiking neurons."""
        for module in self.core.modules():
            if isinstance(module, (LIFNeuron, TernaryLIFNeuron)):
                module.reset()

    def forward(self, x: Tensor) -> Tensor:
        """Apply spiking attention to a standard token sequence.

        Args:
            x: Input tensor with shape ``[B, S, D]``.

        Returns:
            Output tensor with shape ``[B, S, D]``.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected a 3D [B, S, D] tensor, got shape {tuple(x.shape)}.")

        # Fresh membrane state for every forward pass.
        self.reset_state()

        # Lift to the spiking time-major layout [T, B, S, D].
        x_t = x.unsqueeze(0).expand(self.n_timesteps, *x.shape)
        out_t = self.core(x_t, causal=self.causal)

        # Collapse the spiking timestep dimension by averaging.
        return out_t.mean(dim=0)


class StandardAttention(nn.Module):
    """Conventional softmax multi-head self-attention.

    A standard scaled dot-product attention with an optional causal mask, used as
    a non-spiking baseline. Exposes the same ``forward(x: [B, S, D]) -> [B, S, D]``
    interface as :class:`SpikingAttention` so the two are fully interchangeable.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        causal: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the standard attention module.

        Args:
            d_model: Token embedding dimension.
            n_heads: Number of attention heads.
            causal: Whether to apply a causal mask.
            dropout: Dropout probability applied to attention weights.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.attn_dropout = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Apply causal multi-head self-attention.

        Args:
            x: Input tensor with shape ``[B, S, D]``.

        Returns:
            Output tensor with shape ``[B, S, D]``.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected a 3D [B, S, D] tensor, got shape {tuple(x.shape)}.")
        batch_size, seq_len, embed_dim = x.shape

        q, k, v = self.qkv(x).split(embed_dim, dim=-1)
        # [B, S, D] -> [B, n_heads, S, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=self.causal,
        )

        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        return self.resid_dropout(self.proj(out))


def build_attention(
    attn_type: str,
    d_model: int,
    n_heads: int,
    n_timesteps: int = 1,
    causal: bool = True,
    dropout: float = 0.0,
) -> nn.Module:
    """Construct an attention module by name.

    Args:
        attn_type: ``"spiking"`` for :class:`SpikingAttention`, or one of
            ``"standard"``/``"vanilla"``/``"regular"`` for :class:`StandardAttention`.
        d_model: Token embedding dimension.
        n_heads: Number of attention heads.
        n_timesteps: Spiking timesteps (only used by the spiking variant).
        causal: Whether attention is causally masked.
        dropout: Dropout probability (only used by the standard variant).

    Returns:
        An attention module exposing ``forward(x: [B, S, D]) -> [B, S, D]``.

    Raises:
        ValueError: If ``attn_type`` is unknown.
    """
    key = attn_type.lower()
    if key == "spiking":
        return SpikingAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_timesteps=n_timesteps,
            causal=causal,
        )
    if key in {"standard", "vanilla", "regular"}:
        return StandardAttention(
            d_model=d_model,
            n_heads=n_heads,
            causal=causal,
            dropout=dropout,
        )
    raise ValueError(
        f"Unknown attn_type {attn_type!r}. Expected 'spiking' or 'standard'."
    )
