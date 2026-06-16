"""Evaluate a trained spiking language model.

Computes validation loss and perplexity on a held-out split. Optionally
generates a sample continuation for a prompt.

Usage:
    python evaluate.py --checkpoint checkpoints/latest.pt
    python evaluate.py --checkpoint checkpoints/latest.pt --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
from typing import Any

from eval.generation import generate_text
from eval.perplexity import evaluate_perplexity
from train.builders import (
    build_dataloaders,
    build_model_from_config,
    build_tokenizer_from_config,
)
from utils.checkpointing import load_checkpoint
from utils.config import load_config, select_device
from utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a spiking language model.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config override. Defaults to the config saved in the checkpoint.",
    )
    parser.add_argument(
        "--split", default=None, help="Override the validation split name."
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=200,
        help="Maximum number of batches to evaluate.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="If set, generate a continuation for this prompt.",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluation entrypoint."""
    args = parse_args()
    logger = get_logger()

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    config: dict[str, Any] | None = checkpoint.get("config")
    if args.config is not None:
        config = load_config(args.config)
    if config is None:
        raise ValueError(
            "No config found in checkpoint; pass --config explicitly."
        )

    device = select_device(config.get("device", "auto"))
    logger.info("Using device: %s", device)

    tokenizer = build_tokenizer_from_config(config)
    model = build_model_from_config(config, tokenizer)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Loaded checkpoint from step %s.", checkpoint.get("step", "?"))

    if args.split is not None:
        config.setdefault("data", {})["val_split"] = args.split

    _, val_loader = build_dataloaders(config, tokenizer, device)
    if val_loader is None:
        raise ValueError(
            "No validation split configured. Set data.val_split or pass --split."
        )

    metrics = evaluate_perplexity(
        model=model,
        loader=val_loader,
        device=device,
        max_batches=args.max_batches,
        ignore_index=config.get("training", {}).get("ignore_index", -100),
    )
    logger.info("=" * 40)
    logger.info("Validation Loss : %.4f", metrics["loss"])
    logger.info("Perplexity      : %.4f", metrics["perplexity"])
    logger.info("Tokens evaluated: %d", int(metrics["num_tokens"]))
    logger.info("=" * 40)

    if args.prompt is not None:
        gen_cfg = config.get("generation", {})
        text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=gen_cfg.get("max_new_tokens", 100),
            temperature=gen_cfg.get("temperature", 0.8),
            top_k=gen_cfg.get("top_k", 0),
            top_p=gen_cfg.get("top_p", 1.0),
            greedy=gen_cfg.get("greedy", False),
        )
        logger.info("Prompt: %s", args.prompt)
        logger.info("Generation:\n%s", text)


if __name__ == "__main__":
    main()
