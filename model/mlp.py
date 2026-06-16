"""Feed-forward network for the transformer block."""

from __future__ import annotations

from torch import Tensor, nn


class MLP(nn.Module):
    """Position-wise feed-forward network with a SiLU activation.

    The hidden dimension defaults to ``4 * d_model`` following the standard
    transformer convention, but it is fully configurable.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        """Initialize the MLP.

        Args:
            d_model: Input and output feature dimension.
            hidden_dim: Explicit hidden dimension. When ``None`` it is derived
                as ``mlp_ratio * d_model``.
            mlp_ratio: Expansion ratio used when ``hidden_dim`` is ``None``.
            dropout: Dropout probability applied after the activation and the
                output projection.
            bias: Whether the linear layers use a bias term.
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = mlp_ratio * d_model
        self.fc1 = nn.Linear(d_model, hidden_dim, bias=bias)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the feed-forward transformation.

        Args:
            x: Input tensor with shape [..., d_model].

        Returns:
            Output tensor with shape [..., d_model].
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
