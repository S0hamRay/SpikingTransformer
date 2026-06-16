"""Factory helpers that construct training components from a config dict.

These live in the ``train`` package (rather than in ``train.py``) so that both
the training and evaluation entrypoints can import them. The top-level name
``train`` resolves to this package, so ``train.py`` itself is not importable.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset

from data.dataset import build_dataset, build_text_file_datasets, collate_batch
from data.tokenizer import BaseTokenizer, build_tokenizer
from model.model import LanguageModel, ModelConfig
from utils.logging import get_logger


def build_tokenizer_from_config(config: dict[str, Any]) -> BaseTokenizer:
    """Construct the tokenizer described by the config.

    Args:
        config: The full experiment config.

    Returns:
        An initialized tokenizer.
    """
    tok_cfg = config.get("tokenizer", {})
    return build_tokenizer(
        backend=tok_cfg.get("backend", "byte"),
        model_path=tok_cfg.get("model_path"),
        encoding_name=tok_cfg.get("encoding_name", "gpt2"),
    )


def build_model_from_config(
    config: dict[str, Any],
    tokenizer: BaseTokenizer | None = None,
) -> LanguageModel:
    """Construct the language model, reconciling vocab size with the tokenizer.

    Args:
        config: The full experiment config.
        tokenizer: Optional tokenizer used as the source of truth for vocab size.

    Returns:
        An initialized :class:`LanguageModel`.
    """
    model_cfg_dict = dict(config.get("model", {}))
    if tokenizer is not None:
        vocab = tokenizer.vocab_size
        if model_cfg_dict.get("vocab_size") != vocab:
            get_logger().info(
                "Overriding model.vocab_size %s -> %d to match tokenizer.",
                model_cfg_dict.get("vocab_size"),
                vocab,
            )
        model_cfg_dict["vocab_size"] = vocab
        config["model"] = model_cfg_dict
    model_config = ModelConfig.from_dict(model_cfg_dict)
    return LanguageModel(model_config)


def build_dataloaders(
    config: dict[str, Any],
    tokenizer: BaseTokenizer,
    device: torch.device,
) -> tuple[DataLoader, DataLoader | None]:
    """Build train and validation data loaders from the config.

    Args:
        config: The full experiment config.
        tokenizer: Tokenizer used to encode text.
        device: Active device (controls ``pin_memory``).

    Returns:
        Tuple ``(train_loader, val_loader)`` where ``val_loader`` may be ``None``.
    """
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    seq_len = data_cfg.get("seq_len", config.get("model", {}).get("max_seq_len", 128))
    num_workers = data_cfg.get("num_workers", 0)
    batch_size = train_cfg.get("batch_size", 16)

    local_path = data_cfg.get("local_path")
    if local_path:
        # Small local corpus (e.g. tiny Shakespeare): read into memory and split.
        train_dataset, val_dataset = build_text_file_datasets(
            path=local_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            max_chars=data_cfg.get("max_chars"),
            val_fraction=data_cfg.get("val_fraction", 0.1),
            add_eos=data_cfg.get("add_eos", False),
        )
        get_logger().info(
            "Loaded local corpus %s | train blocks=%d | val blocks=%d",
            local_path,
            len(train_dataset),
            len(val_dataset) if val_dataset is not None else 0,
        )
    else:
        streaming = data_cfg.get("streaming", True)
        common = dict(
            name=data_cfg.get("name", "tinystories"),
            tokenizer=tokenizer,
            seq_len=seq_len,
            streaming=streaming,
            add_eos=data_cfg.get("add_eos", True),
            text_field=data_cfg.get("text_field"),
            hf_path=data_cfg.get("hf_path"),
            hf_config=data_cfg.get("hf_config"),
        )

        train_dataset = build_dataset(
            split=data_cfg.get("train_split", "train"), **common
        )

        val_dataset = None
        val_split = data_cfg.get("val_split")
        if val_split:
            try:
                val_dataset = build_dataset(split=val_split, **common)
            except Exception as exc:  # noqa: BLE001 - validation is optional
                get_logger().warning("Could not build validation dataset: %s", exc)
                val_dataset = None

    pin_memory = device.type == "cuda"
    is_iterable = isinstance(train_dataset, IterableDataset)
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_batch,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=not is_iterable,
        **loader_kwargs,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
