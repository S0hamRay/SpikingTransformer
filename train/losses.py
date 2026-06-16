"""Loss functions for language model training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def cross_entropy_loss(
    logits: Tensor,
    targets: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """Compute the next-token cross-entropy loss.

    Args:
        logits: Model output of shape ``[B, S, vocab_size]``.
        targets: Target token ids of shape ``[B, S]``.
        ignore_index: Target value to ignore (e.g. padding positions).

    Returns:
        A scalar loss tensor (mean over non-ignored tokens).
    """
    vocab_size = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


@torch.no_grad()
def perplexity_from_loss(loss: Tensor | float) -> float:
    """Convert a cross-entropy loss into perplexity.

    Args:
        loss: Mean cross-entropy loss (nats per token).

    Returns:
        The perplexity ``exp(loss)``.
    """
    value = float(loss)
    return float(torch.exp(torch.tensor(value)))
