from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from attention import A2OS2A
from spiking_neuron import LIFNeuron, TernaryLIFNeuron


class _CausalA2OS2A(A2OS2A):
    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        timesteps, batch_size, num_tokens, embed_dim = x.shape
        outputs: list[Tensor] = []

        if causal:
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

        out = out.permute(0, 1, 3, 2).reshape(
            timesteps * batch_size, embed_dim, num_tokens, 1
        )
        out = self.bn_proj(self.proj(out))
        out = out.reshape(timesteps, batch_size, embed_dim, num_tokens).permute(0, 1, 3, 2)
        return out.contiguous()


class SpikingAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_timesteps: int = 1,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if n_timesteps < 1:
            raise ValueError("n_timesteps must be >= 1.")
        self.n_timesteps = n_timesteps
        self.causal = causal
        self.core = _CausalA2OS2A(embed_dim=d_model, num_heads=n_heads)

    def reset_state(self) -> None:
        for module in self.core.modules():
            if isinstance(module, (LIFNeuron, TernaryLIFNeuron)):
                module.reset()

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"Dimension should be 3, got {x.dim()}")

        self.reset_state()

        x_t = x.unsqueeze(0).expand(self.n_timesteps, *x.shape)
        out_t = self.core(x_t, causal=self.causal)

        return out_t.mean(dim=0)


class StandardAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        causal: bool = True,
        dropout: float = 0.0,
    ) -> None:
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
        batch_size, seq_len, embed_dim = x.shape

        q, k, v = self.qkv(x).split(embed_dim, dim=-1)
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
