from __future__ import annotations

import torch
from torch import Tensor, nn

from attention import A2OS2A
from mlp import SpikingMLP
from spiking_neuron import LIFNeuron


class SpikingTransformerBlock(nn.Module):
    """Single spiking Transformer encoder block."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = 4) -> None:
        """Initialize the encoder block.

        Args:
            embed_dim: Token embedding dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Expansion ratio for the MLP hidden dimension.
        """
        super().__init__()
        self.norm_attn = LIFNeuron()
        self.attn = A2OS2A(embed_dim, num_heads)
        self.norm_mlp = LIFNeuron()
        self.mlp = SpikingMLP(embed_dim, mlp_ratio)

    def forward(self, U: Tensor) -> Tensor:
        """Apply the attention and MLP branches with membrane residuals.

        Args:
            U: Membrane potential tensor with shape [T, B, N, D].

        Returns:
            Updated membrane potential tensor with shape [T, B, N, D].
        """
        timesteps = U.shape[0]

        attn_steps: list[Tensor] = []
        for t in range(timesteps):
            attn_steps.append(self.norm_attn(U[t]))
        S = torch.stack(attn_steps, dim=0)

        U_prime = self.attn(S) + U

        mlp_steps: list[Tensor] = []
        for t in range(timesteps):
            mlp_steps.append(self.norm_mlp(U_prime[t]))
        S_prime = torch.stack(mlp_steps, dim=0)

        S_out = self.mlp(S_prime) + U_prime
        return S_out
