"""Plain-PyTorch training loop for the language model."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from train.losses import cross_entropy_loss, perplexity_from_loss
from utils.checkpointing import save_checkpoint
from utils.logging import MetricLogger, get_logger


@dataclass
class TrainConfig:
    """Training hyperparameters.

    All values are expected to come from a YAML config; defaults exist only so
    the dataclass is constructible and to document the schema.
    """

    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    batch_size: int = 16

    epochs: int = 1
    max_steps: int | None = None
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1

    amp: bool = False
    amp_dtype: str = "bfloat16"

    log_interval: int = 10
    eval_interval: int = 0
    eval_max_batches: int = 50
    save_interval: int = 0
    checkpoint_dir: str = "checkpoints"
    ignore_index: int = -100

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        """Build a config from a dictionary, ignoring unknown keys.

        Args:
            data: Mapping of field names to values.

        Returns:
            A populated :class:`TrainConfig`.
        """
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in fields}
        if "betas" in kwargs and not isinstance(kwargs["betas"], tuple):
            kwargs["betas"] = tuple(kwargs["betas"])
        return cls(**kwargs)


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Split parameters into decay / no-decay groups.

    Biases and 1D parameters (norm gains, etc.) are excluded from weight decay,
    following common transformer training practice.

    Args:
        model: The model whose parameters to group.
        weight_decay: Weight decay applied to the decay group.

    Returns:
        A list of optimizer parameter-group dicts.
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create a linear-warmup + cosine-decay learning-rate schedule.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total number of optimizer steps.
        min_lr_ratio: Floor for the LR as a fraction of the base LR.

    Returns:
        A :class:`LambdaLR` scheduler.
    """

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Owns the optimizer, scheduler, AMP, and the training/eval loops."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: TrainConfig,
        device: torch.device,
        val_loader: DataLoader | None = None,
        full_config: dict[str, Any] | None = None,
        logger: Any | None = None,
        metric_logger: MetricLogger | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: The language model to train.
            train_loader: DataLoader yielding training batches.
            config: Training hyperparameters.
            device: Device to train on.
            val_loader: Optional DataLoader for validation.
            full_config: The complete experiment config (saved in checkpoints).
            logger: Optional console logger.
            metric_logger: Optional CSV metric logger.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.full_config = full_config
        self.logger = logger or get_logger()
        self.metric_logger = metric_logger

        self.optimizer = AdamW(
            build_param_groups(model, config.weight_decay),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
        )
        self.total_steps = self._resolve_total_steps()
        self.scheduler = build_cosine_schedule(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=self.total_steps,
            min_lr_ratio=config.min_lr_ratio,
        )

        self.amp_dtype = (
            torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
        )
        self.use_amp = config.amp and device.type == "cuda"
        # GradScaler is only needed for fp16; bf16 has enough range without it.
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype == torch.float16
        )

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.checkpoint_dir = Path(config.checkpoint_dir)

    def _resolve_total_steps(self) -> int:
        """Determine the total number of optimizer steps for scheduling."""
        if self.config.max_steps is not None:
            return self.config.max_steps
        try:
            steps_per_epoch = math.ceil(
                len(self.train_loader) / self.config.grad_accum_steps
            )
        except TypeError as exc:  # IterableDataset has no length
            raise ValueError(
                "max_steps must be set in the config when using a streaming "
                "(IterableDataset) train loader."
            ) from exc
        return max(1, steps_per_epoch * self.config.epochs)

    def _autocast(self) -> Any:
        """Return an autocast context manager appropriate for the device."""
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        )

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Move a batch to the training device."""
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def train(self) -> dict[str, float]:
        """Run the full training loop.

        Returns:
            A dict with the final ``global_step`` and ``best_val_loss``.
        """
        self.logger.info(
            "Starting training: total_steps=%d, grad_accum=%d, device=%s",
            self.total_steps,
            self.config.grad_accum_steps,
            self.device,
        )
        self.model.train()
        accum = self.config.grad_accum_steps
        micro_step = 0
        step_loss_sum = 0.0
        window_loss = 0.0
        window_count = 0
        t0 = time.time()
        done = False

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            for batch in self.train_loader:
                batch = self._move_batch(batch)

                with self._autocast():
                    logits = self.model(batch["input_ids"])
                    loss = cross_entropy_loss(
                        logits,
                        batch["target_ids"],
                        ignore_index=self.config.ignore_index,
                    )
                    loss_to_backward = loss / accum

                self.scaler.scale(loss_to_backward).backward()
                step_loss_sum += loss.item()
                micro_step += 1

                if micro_step % accum != 0:
                    continue

                # Optimizer step (one global step per `accum` micro-batches).
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                window_loss += step_loss_sum / accum
                window_count += 1
                step_loss_sum = 0.0

                if self.global_step % max(1, self.config.log_interval) == 0:
                    self._log_train(window_loss / window_count, t0)
                    window_loss = 0.0
                    window_count = 0
                    t0 = time.time()

                if (
                    self.config.eval_interval > 0
                    and self.val_loader is not None
                    and self.global_step % self.config.eval_interval == 0
                ):
                    self._run_eval()
                    self.model.train()

                if (
                    self.config.save_interval > 0
                    and self.global_step % self.config.save_interval == 0
                ):
                    self.save("step_%d.pt" % self.global_step)

                if self.global_step >= self.total_steps:
                    done = True
                    break
            if done:
                break

        # Final evaluation and checkpoint.
        if self.val_loader is not None:
            self._run_eval()
        self.save("latest.pt")
        self.logger.info("Training complete at step %d.", self.global_step)
        return {"global_step": self.global_step, "best_val_loss": self.best_val_loss}

    def _log_train(self, avg_loss: float, t0: float) -> None:
        """Log a training metric line and append to the metric file."""
        lr = self.scheduler.get_last_lr()[0]
        ppl = perplexity_from_loss(avg_loss)
        elapsed = time.time() - t0
        steps_per_sec = self.config.log_interval / elapsed if elapsed > 0 else 0.0
        self.logger.info(
            "step %d/%d | loss %.4f | ppl %.2f | lr %.2e | %.2f it/s",
            self.global_step,
            self.total_steps,
            avg_loss,
            ppl,
            lr,
            steps_per_sec,
        )
        if self.metric_logger is not None:
            self.metric_logger.log(
                {
                    "step": self.global_step,
                    "split": "train",
                    "loss": round(avg_loss, 6),
                    "perplexity": round(ppl, 6),
                    "lr": lr,
                }
            )

    def _run_eval(self) -> None:
        """Evaluate on the validation loader and checkpoint on improvement."""
        assert self.val_loader is not None
        val_loss, val_ppl = self.evaluate(
            self.val_loader, max_batches=self.config.eval_max_batches
        )
        self.logger.info(
            "[eval] step %d | val_loss %.4f | val_ppl %.2f",
            self.global_step,
            val_loss,
            val_ppl,
        )
        if self.metric_logger is not None:
            self.metric_logger.log(
                {
                    "step": self.global_step,
                    "split": "val",
                    "loss": round(val_loss, 6),
                    "perplexity": round(val_ppl, 6),
                    "lr": self.scheduler.get_last_lr()[0],
                }
            )
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save("best.pt", extra={"val_loss": val_loss})

    @torch.no_grad()
    def evaluate(
        self,
        loader: Iterable[dict[str, torch.Tensor]],
        max_batches: int | None = None,
    ) -> tuple[float, float]:
        """Compute average loss and perplexity over a loader.

        Args:
            loader: DataLoader or iterable of batches.
            max_batches: Optional cap on the number of evaluation batches.

        Returns:
            Tuple of ``(mean_loss, perplexity)`` aggregated over tokens.
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = self._move_batch(batch)
            with self._autocast():
                logits = self.model(batch["input_ids"])
                loss = cross_entropy_loss(
                    logits,
                    batch["target_ids"],
                    ignore_index=self.config.ignore_index,
                )
            n_tokens = int(batch["target_ids"].numel())
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

        mean_loss = total_loss / max(1, total_tokens)
        return mean_loss, perplexity_from_loss(mean_loss)

    def save(self, filename: str, extra: dict[str, Any] | None = None) -> None:
        """Save a training checkpoint.

        Args:
            filename: Name of the checkpoint file within the checkpoint dir.
            extra: Optional extra metadata to persist.
        """
        path = self.checkpoint_dir / filename
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.scaler.is_enabled() else None,
            step=self.global_step,
            epoch=self.epoch,
            config=self.full_config,
            extra=extra,
        )
        self.logger.info("Saved checkpoint: %s", path)
