"""Evaluation package: perplexity and text generation."""

from __future__ import annotations

from eval.generation import generate, generate_text
from eval.perplexity import evaluate_perplexity

__all__ = [
    "generate",
    "generate_text",
    "evaluate_perplexity",
]
