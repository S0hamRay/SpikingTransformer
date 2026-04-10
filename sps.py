from __future__ import annotations

import torch
from torch import Tensor, nn

from spiking_neuron import LIFNeuron


class SPS(nn.Module):
    """Spiking patch splitting stem for static images."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 256,
        T: int = 4,
    ) -> None:
        """Initialize the SPS module.

        Args:
            img_size: Input image resolution.
            patch_size: Requested patch size for API compatibility.
            in_channels: Number of image channels.
            embed_dim: Token embedding dimension.
            T: Number of timesteps.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.T = T

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(embed_dim // 8),
            LIFNeuron(),
            nn.MaxPool2d(2),
            nn.Conv2d(embed_dim // 8, embed_dim // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(embed_dim // 4),
            LIFNeuron(),
            nn.MaxPool2d(2),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            LIFNeuron(),
            nn.MaxPool2d(2),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            LIFNeuron(),
        )
        self.rpe_conv = nn.Conv2d(
            embed_dim, embed_dim, 3, 1, 1, groups=embed_dim, bias=False
        )
        self.rpe_bn = nn.BatchNorm2d(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Repeat a static image over time and convert it to tokens.

        Args:
            x: Input image tensor with shape [B, C, H, W].

        Returns:
            Token tensor with shape [T, B, N, D].
        """
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        outputs: list[Tensor] = []

        for t in range(self.T):
            proj_out = self.proj(x[t])
            rpe_out = self.rpe_bn(self.rpe_conv(proj_out))
            u = proj_out + rpe_out
            tokens = u.flatten(2).transpose(1, 2)
            outputs.append(tokens)

        return torch.stack(outputs, dim=0)
