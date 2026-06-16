"""Autoregressive text generation utilities.

Supports greedy decoding, temperature scaling, top-k, and top-p (nucleus)
sampling. Sampling strategies compose: ``temperature`` is applied first, then
``top_k`` and ``top_p`` filtering.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from data.tokenizer import BaseTokenizer


def _filter_top_k(logits: Tensor, top_k: int) -> Tensor:
    """Mask out all but the ``top_k`` highest-probability logits.

    Args:
        logits: Logit tensor of shape ``[B, vocab_size]``.
        top_k: Number of top logits to keep.

    Returns:
        Filtered logits with removed entries set to ``-inf``.
    """
    if top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _filter_top_p(logits: Tensor, top_p: float) -> Tensor:
    """Apply nucleus (top-p) filtering to the logits.

    Args:
        logits: Logit tensor of shape ``[B, vocab_size]``.
        top_p: Cumulative probability mass to retain (0 < top_p <= 1).

    Returns:
        Filtered logits with removed entries set to ``-inf``.
    """
    if top_p >= 1.0 or top_p <= 0.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens beyond the nucleus, always keeping at least the top token.
    remove = cum_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    remove_scattered = remove.scatter(-1, sorted_idx, remove)
    return logits.masked_fill(remove_scattered, float("-inf"))


def _sample_next_token(
    logits: Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    greedy: bool,
) -> Tensor:
    """Select the next token id from the final-position logits.

    Args:
        logits: Logits of shape ``[B, vocab_size]`` for the last position.
        temperature: Softmax temperature; lower is greedier.
        top_k: Top-k cutoff (0 disables).
        top_p: Top-p cutoff (>=1 disables).
        greedy: When ``True``, take the argmax and ignore sampling.

    Returns:
        Long tensor of shape ``[B, 1]`` with the chosen token ids.
    """
    if greedy or temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    logits = _filter_top_k(logits, top_k)
    logits = _filter_top_p(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: nn.Module,
    input_ids: Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    greedy: bool = False,
    eos_id: int | None = None,
) -> Tensor:
    """Autoregressively generate token ids from a model.

    Args:
        model: A causal language model returning logits ``[B, S, vocab_size]``.
        input_ids: Prompt token ids of shape ``[B, S]``.
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k cutoff (0 disables).
        top_p: Top-p / nucleus cutoff (>=1 disables).
        greedy: Force greedy decoding regardless of sampling parameters.
        eos_id: Optional end-of-sequence id; generation of a row continues but
            stops early for all rows once every row has emitted EOS.

    Returns:
        Token ids of shape ``[B, S + generated]`` including the prompt.
    """
    model.eval()
    device = next(model.parameters()).device
    generated = input_ids.to(device)

    max_seq_len = getattr(getattr(model, "config", None), "max_seq_len", None)
    finished = torch.zeros(generated.size(0), dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        # Crop context to the model's maximum sequence length.
        context = generated
        if max_seq_len is not None and context.size(1) > max_seq_len:
            context = context[:, -max_seq_len:]

        logits = model(context)[:, -1, :]
        next_token = _sample_next_token(logits, temperature, top_k, top_p, greedy)
        generated = torch.cat([generated, next_token], dim=1)

        if eos_id is not None:
            finished = finished | (next_token.squeeze(-1) == eos_id)
            if bool(finished.all()):
                break

    return generated


@torch.no_grad()
def generate_text(
    model: nn.Module,
    tokenizer: BaseTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    greedy: bool = False,
    add_bos: bool = False,
) -> str:
    """Generate a continuation for a text prompt.

    Args:
        model: A causal language model.
        tokenizer: Tokenizer used to encode the prompt and decode the output.
        prompt: The text prompt.
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k cutoff (0 disables).
        top_p: Top-p / nucleus cutoff (>=1 disables).
        greedy: Force greedy decoding.
        add_bos: Whether to prepend the BOS token to the prompt.

    Returns:
        The decoded generated text (prompt included).
    """
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt, add_bos=add_bos)
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_id if tokenizer.bos_id is not None else 0]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        greedy=greedy,
        eos_id=tokenizer.eos_id,
    )
    return tokenizer.decode(output_ids[0].tolist())
