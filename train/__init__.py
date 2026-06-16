"""Training package: losses and the plain-PyTorch trainer."""

from __future__ import annotations

from train.losses import cross_entropy_loss, perplexity_from_loss
from train.trainer import (
    TrainConfig,
    Trainer,
    build_cosine_schedule,
    build_param_groups,
)

__all__ = [
    "cross_entropy_loss",
    "perplexity_from_loss",
    "TrainConfig",
    "Trainer",
    "build_cosine_schedule",
    "build_param_groups",
]
