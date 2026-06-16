"""Normalization layers for the language model."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization.

    A lightweight alternative to LayerNorm that normalizes by the RMS of the
    activations and applies a learnable per-channel gain. There is no mean
    subtraction and no bias term.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize the RMSNorm layer.

        Args:
            dim: Size of the trailing feature dimension to normalize.
            eps: Numerical stabilizer added to the variance.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize the trailing dimension of the input.

        Args:
            x: Input tensor with feature dimension last, shape [..., dim].

        Returns:
            Normalized tensor with the same shape as the input.
        """
        # Compute in float32 for numerical stability, then cast back.
        dtype = x.dtype
        x_fp32 = x.float()
        norm = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (norm.to(dtype)) * self.weight
