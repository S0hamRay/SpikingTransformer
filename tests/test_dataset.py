"""Tests for the dataset packing pipeline."""

from __future__ import annotations

import torch

from data.dataset import (
    PackedIterableDataset,
    PackedMapDataset,
    collate_batch,
)
from data.tokenizer import ByteTokenizer

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "spiking neurons integrate and fire over time",
    "attention is all you need for sequence modeling",
] * 20


def test_map_dataset_shapes_and_shift() -> None:
    tok = ByteTokenizer()
    ds = PackedMapDataset(CORPUS, tokenizer=tok, seq_len=8, add_eos=True)
    assert len(ds) > 0
    example = ds[0]
    assert example["input_ids"].shape == (8,)
    assert example["target_ids"].shape == (8,)
    assert example["attention_mask"].shape == (8,)
    # target is the next-token shift of input within the block.
    assert torch.equal(example["input_ids"][1:], example["target_ids"][:-1])


def test_iterable_dataset_matches_map() -> None:
    tok = ByteTokenizer()
    seq_len = 8
    map_ds = PackedMapDataset(CORPUS, tokenizer=tok, seq_len=seq_len, add_eos=True)
    iter_ds = PackedIterableDataset(CORPUS, tokenizer=tok, seq_len=seq_len, add_eos=True)
    iter_items = list(iter_ds)
    assert len(iter_items) == len(map_ds)
    assert torch.equal(iter_items[0]["input_ids"], map_ds[0]["input_ids"])


def test_dict_records_are_supported() -> None:
    tok = ByteTokenizer()
    records = [{"text": t} for t in CORPUS]
    ds = PackedMapDataset(records, tokenizer=tok, seq_len=8, text_field="text")
    assert len(ds) > 0


def test_collate_batch_stacks() -> None:
    tok = ByteTokenizer()
    ds = PackedMapDataset(CORPUS, tokenizer=tok, seq_len=8)
    batch = collate_batch([ds[0], ds[1], ds[2]])
    assert batch["input_ids"].shape == (3, 8)
    assert batch["target_ids"].shape == (3, 8)
    assert batch["attention_mask"].shape == (3, 8)


def test_overlap_preserves_continuity() -> None:
    """Consecutive blocks overlap by one token so targets are continuous."""
    tok = ByteTokenizer()
    ds = PackedMapDataset(CORPUS, tokenizer=tok, seq_len=8, add_eos=False)
    if len(ds) >= 2:
        first_last_target = ds[0]["target_ids"][-1].item()
        second_first_input = ds[1]["input_ids"][0].item()
        assert first_last_target == second_first_input
