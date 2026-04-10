from __future__ import annotations

from torch import Tensor, nn

from encoder_block import SpikingTransformerBlock
from sps import SPS


class SpikingTransformer(nn.Module):
    """Spiking Transformer with A2OS2A attention."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 256,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        T: int = 4,
    ) -> None:
        """Initialize the full spiking Transformer.

        Args:
            img_size: Input image resolution.
            patch_size: Requested patch size for API compatibility.
            in_channels: Number of image channels.
            num_classes: Number of classification targets.
            embed_dim: Token embedding dimension.
            depth: Number of encoder blocks.
            num_heads: Number of attention heads.
            mlp_ratio: Expansion ratio for the MLP hidden layer.
            T: Number of timesteps.
        """
        super().__init__()
        self.T = T
        self.sps = SPS(img_size, patch_size, in_channels, embed_dim, T)
        self.blocks = nn.ModuleList(
            [
                SpikingTransformerBlock(embed_dim, num_heads, mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Encode an image and predict class logits.

        Args:
            x: Input image tensor with shape [B, C, H, W].

        Returns:
            Logits tensor with shape [B, num_classes].
        """
        U = self.sps(x)
        for block in self.blocks:
            U = block(U)
        out = U.mean(dim=(0, 2))
        return self.head(out)
