"""Checkpoint saving and loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    step: int = 0,
    epoch: int = 0,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist training state to disk.

    Args:
        path: Destination file path.
        model: Model whose ``state_dict`` is saved.
        optimizer: Optional optimizer state.
        scheduler: Optional LR scheduler state.
        scaler: Optional AMP grad scaler state.
        step: Global step counter.
        epoch: Epoch counter.
        config: The full experiment config, stored for reproducibility.
        extra: Any additional metadata to persist (e.g. best metric).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "epoch": epoch,
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint payload from disk.

    Args:
        path: Path to the checkpoint file.
        map_location: Device mapping passed to :func:`torch.load`.

    Returns:
        The raw checkpoint dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # weights_only=False because we persist Python objects (config dict).
    return torch.load(path, map_location=map_location, weights_only=False)


def restore_training_state(
    checkpoint: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Restore model and optimizer state from a checkpoint payload.

    Args:
        checkpoint: Payload produced by :func:`save_checkpoint`.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        scaler: Optional AMP scaler to restore.
        strict: Whether to strictly enforce matching state-dict keys.

    Returns:
        A dict with ``step`` and ``epoch`` resume counters.
    """
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return {
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
    }
