"""Dataset pipeline for autoregressive language modeling.

Streams text from HuggingFace datasets (TinyStories, OpenWebText, FineWeb),
tokenizes it, packs the token stream into fixed-length blocks, and emits
``input_ids`` / ``target_ids`` pairs where ``target_ids`` is the next-token
shift of ``input_ids``.

Two dataset flavours are provided:

* :class:`PackedIterableDataset` -- a streaming ``IterableDataset`` suitable for
  large corpora that do not fit in memory. It shards correctly across
  ``DataLoader`` workers.
* :class:`PackedMapDataset` -- an in-memory, index-addressable ``Dataset`` that
  supports shuffling and is convenient for small corpora and unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from data.tokenizer import BaseTokenizer


@dataclass(frozen=True)
class DatasetSpec:
    """Describes how to load a named HuggingFace dataset."""

    path: str
    name: str | None
    text_field: str


# Friendly names mapped to their HuggingFace coordinates.
DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "tinystories": DatasetSpec(path="roneneldan/TinyStories", name=None, text_field="text"),
    "openwebtext": DatasetSpec(path="Skylion007/openwebtext", name=None, text_field="text"),
    "fineweb": DatasetSpec(
        path="HuggingFaceFW/fineweb", name="sample-10BT", text_field="text"
    ),
}


def load_dataset(
    name: str,
    split: str = "train",
    streaming: bool = True,
    text_field: str | None = None,
    hf_path: str | None = None,
    hf_config: str | None = None,
) -> tuple[Iterable, str]:
    """Load a text dataset from the HuggingFace hub.

    Args:
        name: Friendly dataset name (key of :data:`DATASET_REGISTRY`) or a raw
            HuggingFace dataset path when ``hf_path`` is provided.
        split: Dataset split to load (e.g. ``"train"`` or ``"validation"``).
        streaming: Whether to stream the dataset instead of downloading it fully.
        text_field: Override for the text column name.
        hf_path: Explicit HuggingFace dataset path (bypasses the registry).
        hf_config: Optional HuggingFace dataset configuration name.

    Returns:
        A tuple ``(dataset, text_field)``.
    """
    from datasets import load_dataset as hf_load_dataset

    if hf_path is not None:
        path = hf_path
        config_name = hf_config
        field = text_field or "text"
    else:
        key = name.lower()
        if key not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset {name!r}. Known: {sorted(DATASET_REGISTRY)} "
                "(or pass hf_path for an arbitrary dataset)."
            )
        spec = DATASET_REGISTRY[key]
        path = spec.path
        config_name = spec.name
        field = text_field or spec.text_field

    dataset = hf_load_dataset(path, name=config_name, split=split, streaming=streaming)
    return dataset, field


def _iter_texts(source: Iterable, text_field: str) -> Iterator[str]:
    """Yield text strings from a source of strings or dict records.

    Args:
        source: Iterable of strings or mapping records.
        text_field: Key to extract when records are dictionaries.

    Yields:
        Text strings.
    """
    for item in source:
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            text = item.get(text_field)
            if text:
                yield text
        else:
            raise TypeError(f"Unsupported sample type: {type(item)!r}")


def _pack_tokens(
    token_stream: Iterator[int],
    block_len: int,
) -> Iterator[list[int]]:
    """Pack a flat token stream into overlapping blocks of ``block_len``.

    Consecutive blocks overlap by one token so that the next-token target of the
    final position is preserved across block boundaries.

    Args:
        token_stream: Iterator of token ids.
        block_len: Length of each block (``seq_len + 1``).

    Yields:
        Lists of exactly ``block_len`` token ids.
    """
    buffer: list[int] = []
    for token in token_stream:
        buffer.append(token)
        if len(buffer) >= block_len:
            yield buffer[:block_len]
            # Keep the last token as the start of the next block (1-token overlap).
            buffer = buffer[block_len - 1 :]


def _to_example(block: list[int]) -> dict[str, torch.Tensor]:
    """Convert a packed block into an LM training example.

    Args:
        block: A list of ``seq_len + 1`` token ids.

    Returns:
        Dict with ``input_ids``, ``target_ids``, and ``attention_mask`` tensors.
    """
    input_ids = torch.tensor(block[:-1], dtype=torch.long)
    target_ids = torch.tensor(block[1:], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "attention_mask": attention_mask,
    }


class PackedIterableDataset(IterableDataset):
    """Streaming dataset that packs tokenized text into LM examples."""

    def __init__(
        self,
        source: Iterable,
        tokenizer: BaseTokenizer,
        seq_len: int,
        text_field: str = "text",
        add_eos: bool = True,
    ) -> None:
        """Initialize the streaming packed dataset.

        Args:
            source: Iterable of text strings or HuggingFace records.
            tokenizer: Tokenizer used to encode text.
            seq_len: Output sequence length (block length is ``seq_len + 1``).
            text_field: Field to read text from when records are dictionaries.
            add_eos: Whether to append an EOS token between documents.
        """
        super().__init__()
        self.source = source
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_field = text_field
        self.add_eos = add_eos

    def _token_stream(self) -> Iterator[int]:
        """Yield a flat stream of token ids across all documents."""
        worker = get_worker_info()
        num_workers = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0

        for index, text in enumerate(_iter_texts(self.source, self.text_field)):
            # Shard documents across workers to avoid duplicated samples.
            if index % num_workers != worker_id:
                continue
            ids = self.tokenizer.encode(text, add_eos=self.add_eos)
            yield from ids

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        block_len = self.seq_len + 1
        for block in _pack_tokens(self._token_stream(), block_len):
            yield _to_example(block)


class PackedMapDataset(Dataset):
    """In-memory, index-addressable packed dataset.

    Tokenizes the entire source up front and stores fixed-length blocks. Useful
    for small corpora, deterministic shuffling, and unit tests.
    """

    def __init__(
        self,
        source: Iterable,
        tokenizer: BaseTokenizer,
        seq_len: int,
        text_field: str = "text",
        add_eos: bool = True,
    ) -> None:
        """Initialize and materialize the packed dataset.

        Args:
            source: Iterable of text strings or HuggingFace records.
            tokenizer: Tokenizer used to encode text.
            seq_len: Output sequence length (block length is ``seq_len + 1``).
            text_field: Field to read text from when records are dictionaries.
            add_eos: Whether to append an EOS token between documents.
        """
        super().__init__()
        self.seq_len = seq_len

        def token_stream() -> Iterator[int]:
            for text in _iter_texts(source, text_field):
                yield from tokenizer.encode(text, add_eos=add_eos)

        self.blocks: list[list[int]] = list(_pack_tokens(token_stream(), seq_len + 1))

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return _to_example(self.blocks[index])


def build_dataset(
    name: str,
    tokenizer: BaseTokenizer,
    seq_len: int,
    split: str = "train",
    streaming: bool = True,
    add_eos: bool = True,
    text_field: str | None = None,
    hf_path: str | None = None,
    hf_config: str | None = None,
) -> Dataset | IterableDataset:
    """Build a packed LM dataset from a HuggingFace source.

    Args:
        name: Friendly dataset name or arbitrary name when ``hf_path`` is set.
        tokenizer: Tokenizer used to encode text.
        seq_len: Output sequence length.
        split: Dataset split to load.
        streaming: Whether to stream (yields :class:`PackedIterableDataset`) or
            materialize in memory (yields :class:`PackedMapDataset`).
        add_eos: Whether to append an EOS token between documents.
        text_field: Override for the text column name.
        hf_path: Explicit HuggingFace dataset path (bypasses the registry).
        hf_config: Optional HuggingFace dataset configuration name.

    Returns:
        A packed dataset ready to be wrapped in a ``DataLoader``.
    """
    source, field = load_dataset(
        name=name,
        split=split,
        streaming=streaming,
        text_field=text_field,
        hf_path=hf_path,
        hf_config=hf_config,
    )
    dataset_cls = PackedIterableDataset if streaming else PackedMapDataset
    return dataset_cls(
        source=source,
        tokenizer=tokenizer,
        seq_len=seq_len,
        text_field=field,
        add_eos=add_eos,
    )


def load_text_file(
    path: str | Path,
    max_chars: int | None = None,
    encoding: str = "utf-8",
) -> str:
    """Read a local UTF-8 text file, optionally truncated to a subset.

    Args:
        path: Path to the text file.
        max_chars: If set, keep only the first ``max_chars`` characters. Use this
            to train on a small subset of a large corpus.
        encoding: File encoding.

    Returns:
        The (possibly truncated) file contents as a single string.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    text = path.read_text(encoding=encoding)
    if max_chars is not None:
        text = text[:max_chars]
    return text


def build_text_file_datasets(
    path: str | Path,
    tokenizer: BaseTokenizer,
    seq_len: int,
    max_chars: int | None = None,
    val_fraction: float = 0.1,
    add_eos: bool = False,
    encoding: str = "utf-8",
) -> tuple["PackedMapDataset", "PackedMapDataset | None"]:
    """Build packed train/val datasets from a single local text file.

    The text is read into memory, optionally truncated to ``max_chars``, and
    split chronologically into a training and a validation portion. This is the
    convenient path for small corpora such as tiny Shakespeare.

    Args:
        path: Path to the text file.
        tokenizer: Tokenizer used to encode text.
        seq_len: Output sequence length.
        max_chars: Optional cap on the number of characters to use.
        val_fraction: Fraction of the (truncated) text held out for validation.
        add_eos: Whether to append an EOS token to each portion.
        encoding: File encoding.

    Returns:
        Tuple ``(train_dataset, val_dataset)`` where ``val_dataset`` may be
        ``None`` if ``val_fraction`` is 0 or the holdout is too small to pack.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")

    text = load_text_file(path, max_chars=max_chars, encoding=encoding)
    split_idx = int(len(text) * (1.0 - val_fraction))
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_dataset = PackedMapDataset(
        [train_text], tokenizer=tokenizer, seq_len=seq_len, add_eos=add_eos
    )

    val_dataset: PackedMapDataset | None = None
    if val_text:
        candidate = PackedMapDataset(
            [val_text], tokenizer=tokenizer, seq_len=seq_len, add_eos=add_eos
        )
        if len(candidate) > 0:
            val_dataset = candidate

    return train_dataset, val_dataset


def collate_batch(
    examples: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Stack a list of examples into a batch.

    All packed examples share the same length, so a simple stack suffices.

    Args:
        examples: List of example dicts.

    Returns:
        Batched dict of tensors with a leading batch dimension.
    """
    return {
        "input_ids": torch.stack([e["input_ids"] for e in examples]),
        "target_ids": torch.stack([e["target_ids"] for e in examples]),
        "attention_mask": torch.stack([e["attention_mask"] for e in examples]),
    }
