"""Perplexity evaluation on held-out data."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from train.losses import cross_entropy_loss


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    max_batches: int | None = None,
    ignore_index: int = -100,
) -> dict[str, float]:
    """Compute validation loss and perplexity over a data loader.

    The loss is aggregated per token so that batches with differing token counts
    are weighted correctly.

    Args:
        model: A causal language model returning logits ``[B, S, vocab_size]``.
        loader: Iterable of batches with ``input_ids`` and ``target_ids``.
        device: Device to run evaluation on.
        max_batches: Optional cap on the number of batches.
        ignore_index: Target id to ignore in the loss.

    Returns:
        Dict with ``loss``, ``perplexity``, and ``num_tokens``.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        logits = model(input_ids)
        loss = cross_entropy_loss(logits, target_ids, ignore_index=ignore_index)

        n_tokens = int((target_ids != ignore_index).sum().item())
        n_tokens = n_tokens if n_tokens > 0 else int(target_ids.numel())
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    mean_loss = total_loss / max(1, total_tokens)
    perplexity = float(torch.exp(torch.tensor(mean_loss)))
    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "num_tokens": float(total_tokens),
    }
