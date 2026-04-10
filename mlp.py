from __future__ import annotations

import torch
from torch import Tensor, nn

from spiking_neuron import LIFNeuron


class SpikingMLP(nn.Module):
    """Spiking feed-forward block used inside the encoder."""

    def __init__(self, embed_dim: int, mlp_ratio: int = 4) -> None:
        """Initialize the spiking MLP.

        Args:
            embed_dim: Token embedding dimension.
            mlp_ratio: Expansion ratio for the hidden layer.
        """
        super().__init__()
        hidden_dim = embed_dim * mlp_ratio
        self.fc1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.sn1 = LIFNeuron()
        self.fc2 = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.sn2 = LIFNeuron()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the spiking MLP to a time-major token sequence.

        Args:
            x: Input tensor with shape [T, B, N, D].

        Returns:
            Output tensor with shape [T, B, N, D].
        """
        timesteps, batch_size, num_tokens, embed_dim = x.shape

        x = x.reshape(timesteps * batch_size * num_tokens, embed_dim)
        x = self.fc1(x)
        x = self.bn1(x)
        x = x.reshape(timesteps, batch_size, num_tokens, -1)

        hidden_steps: list[Tensor] = []
        for t in range(timesteps):
            hidden_steps.append(self.sn1(x[t]))
        x = torch.stack(hidden_steps, dim=0)

        x = x.reshape(timesteps * batch_size * num_tokens, -1)
        x = self.fc2(x)
        x = self.bn2(x)
        x = x.reshape(timesteps, batch_size, num_tokens, embed_dim)

        output_steps: list[Tensor] = []
        for t in range(timesteps):
            output_steps.append(self.sn2(x[t]))
        return torch.stack(output_steps, dim=0)
