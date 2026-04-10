from __future__ import annotations

import torch
from torch import Tensor, nn

from spiking_neuron import LIFNeuron, TernaryLIFNeuron


class A2OS2A(nn.Module):
    """Accurate addition-only spiking self-attention."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """Initialize the attention module.

        Args:
            embed_dim: Token embedding dimension.
            num_heads: Number of attention heads.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads.')

        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)

        self.bn_Q = nn.BatchNorm1d(embed_dim)
        self.bn_K = nn.BatchNorm1d(embed_dim)
        self.bn_V = nn.BatchNorm1d(embed_dim)

        self.sn_Q = LIFNeuron()
        self.relu_K = nn.ReLU()
        self.sn_V = TernaryLIFNeuron()

        self.proj = nn.Conv2d(embed_dim, embed_dim, 1, 1, 0, bias=False)
        self.bn_proj = nn.BatchNorm2d(embed_dim)

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    def forward(self, x: Tensor) -> Tensor:
        """Apply spiking attention to a time-major token sequence.

        Args:
            x: Input tensor with shape [T, B, N, D].

        Returns:
            Output tensor with shape [T, B, N, D].
        """
        timesteps, batch_size, num_tokens, embed_dim = x.shape
        outputs: list[Tensor] = []

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
            out = attn @ v
            out = out.permute(0, 2, 1, 3).reshape(batch_size, num_tokens, embed_dim)
            outputs.append(out)

        out = torch.stack(outputs, dim=0)

        # Conv2d expects channels-first tensors, so move D into the channel axis.
        out = out.permute(0, 1, 3, 2).reshape(timesteps * batch_size, embed_dim, num_tokens, 1)
        out = self.bn_proj(self.proj(out))
        out = out.reshape(timesteps, batch_size, embed_dim, num_tokens).permute(0, 1, 3, 2)
        return out.contiguous()
