"""Train an autoregressive spiking language model.

Usage:
    python train.py --config configs/tiny.yaml
    python train.py --config configs/tiny.yaml --resume checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
from typing import Any

from train.builders import (
    build_dataloaders,
    build_model_from_config,
    build_tokenizer_from_config,
)
from train.trainer import TrainConfig, Trainer
from utils.checkpointing import load_checkpoint, restore_training_state
from utils.config import load_config, select_device
from utils.logging import MetricLogger, get_logger
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a spiking language model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--resume", default=None, help="Optional checkpoint path to resume from."
    )
    parser.add_argument(
        "--attn-type",
        default=None,
        choices=["spiking", "standard"],
        help="Override model.attn_type from the config (spiking | standard).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Override training.checkpoint_dir from the config.",
    )
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Apply command-line overrides onto the loaded config in place.

    Args:
        config: The loaded experiment config.
        args: Parsed command-line arguments.
    """
    if args.attn_type is not None:
        config.setdefault("model", {})["attn_type"] = args.attn_type
    if args.checkpoint_dir is not None:
        config.setdefault("training", {})["checkpoint_dir"] = args.checkpoint_dir


def main() -> None:
    """Train entrypoint."""
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)

    logger = get_logger()
    set_seed(config.get("seed", 42))
    logger.info("Attention type: %s", config.get("model", {}).get("attn_type", "spiking"))
    device = select_device(config.get("device", "auto"))
    logger.info("Using device: %s", device)

    tokenizer = build_tokenizer_from_config(config)
    model = build_model_from_config(config, tokenizer)
    logger.info("Model parameters: %.2fM", model.num_parameters() / 1e6)

    train_loader, val_loader = build_dataloaders(config, tokenizer, device)

    train_config = TrainConfig.from_dict(config.get("training", {}))
    metric_logger = MetricLogger(train_config.checkpoint_dir)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
        full_config=config,
        logger=logger,
        metric_logger=metric_logger,
    )

    if args.resume:
        checkpoint = load_checkpoint(args.resume, map_location=device)
        state = restore_training_state(
            checkpoint,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler if trainer.scaler.is_enabled() else None,
        )
        trainer.global_step = state["step"]
        trainer.epoch = state["epoch"]
        logger.info("Resumed from %s at step %d.", args.resume, trainer.global_step)

    trainer.train()


if __name__ == "__main__":
    main()
