"""Tokenizer utilities.

Supports three interchangeable backends behind a common interface:

* ``sentencepiece`` -- train a new model or load an existing ``.model`` file.
* ``tiktoken`` -- byte-pair encodings shipped with OpenAI's ``tiktoken``.
* ``byte`` -- a dependency-free UTF-8 byte tokenizer, handy for tests and quick
  experiments where training a vocabulary is overkill.

A small command-line interface is provided for training a SentencePiece model
and for ad-hoc encode/decode round-trips::

    python -m data.tokenizer train --input corpus.txt --model-prefix tok --vocab-size 8000
    python -m data.tokenizer encode --backend sentencepiece --model-path tok.model --text "hello"
    python -m data.tokenizer decode --backend byte --ids 104 105
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sequence


class BaseTokenizer(ABC):
    """Common interface shared by all tokenizer backends."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Total number of token ids the tokenizer can produce."""

    @property
    def bos_id(self) -> int | None:
        """Beginning-of-sequence token id, if any."""
        return None

    @property
    def eos_id(self) -> int | None:
        """End-of-sequence token id, if any."""
        return None

    @property
    def pad_id(self) -> int | None:
        """Padding token id, if any."""
        return None

    @abstractmethod
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode a string into a list of token ids."""

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """Decode a list of token ids back into a string."""

    def encode_batch(
        self,
        texts: Iterable[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[list[int]]:
        """Encode multiple strings.

        Args:
            texts: An iterable of strings.
            add_bos: Whether to prepend the BOS token to each sequence.
            add_eos: Whether to append the EOS token to each sequence.

        Returns:
            A list of token-id lists.
        """
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]


class ByteTokenizer(BaseTokenizer):
    """UTF-8 byte-level tokenizer with reserved special tokens.

    Bytes ``0..255`` map to ids ``0..255``. Special tokens occupy the ids
    immediately above the byte range. This backend requires no external assets,
    so it always works offline and is ideal for unit tests.
    """

    def __init__(self) -> None:
        """Initialize the byte tokenizer with BOS/EOS/PAD specials."""
        self._byte_count = 256
        self._pad_id = self._byte_count + 0
        self._bos_id = self._byte_count + 1
        self._eos_id = self._byte_count + 2

    @property
    def vocab_size(self) -> int:
        return self._byte_count + 3

    @property
    def bos_id(self) -> int | None:
        return self._bos_id

    @property
    def eos_id(self) -> int | None:
        return self._eos_id

    @property
    def pad_id(self) -> int | None:
        return self._pad_id

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids = [self._bos_id, *ids]
        if add_eos:
            ids = [*ids, self._eos_id]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        byte_values = [i for i in ids if 0 <= i < self._byte_count]
        return bytes(byte_values).decode("utf-8", errors="replace")


class SentencePieceTokenizer(BaseTokenizer):
    """Tokenizer backed by a trained SentencePiece model."""

    def __init__(self, model_path: str | Path) -> None:
        """Load a SentencePiece model from disk.

        Args:
            model_path: Path to a ``.model`` file produced by training.
        """
        import sentencepiece as spm  # local import keeps the dep optional

        self.model_path = str(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def bos_id(self) -> int | None:
        bos = self.sp.bos_id()
        return bos if bos >= 0 else None

    @property
    def eos_id(self) -> int | None:
        eos = self.sp.eos_id()
        return eos if eos >= 0 else None

    @property
    def pad_id(self) -> int | None:
        pad = self.sp.pad_id()
        return pad if pad >= 0 else None

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        return self.sp.encode(text, out_type=int, add_bos=add_bos, add_eos=add_eos)

    def decode(self, ids: Sequence[int]) -> str:
        return self.sp.decode(list(ids))


class TiktokenTokenizer(BaseTokenizer):
    """Tokenizer backed by an OpenAI ``tiktoken`` encoding."""

    def __init__(self, encoding_name: str = "gpt2") -> None:
        """Load a tiktoken encoding.

        Args:
            encoding_name: Name of the tiktoken encoding (e.g. ``"gpt2"`` or
                ``"cl100k_base"``). The first use may download the vocabulary.
        """
        import tiktoken  # local import keeps the dep optional

        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)
        # tiktoken has no native BOS/PAD; reserve EOS at the top of the range.
        self._eos_id = self.enc.n_vocab

    @property
    def vocab_size(self) -> int:
        # +1 to account for the reserved EOS id.
        return self.enc.n_vocab + 1

    @property
    def eos_id(self) -> int | None:
        return self._eos_id

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = self.enc.encode(text)
        if add_eos:
            ids = [*ids, self._eos_id]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        # Drop reserved special ids that tiktoken cannot decode.
        decodable = [i for i in ids if i < self.enc.n_vocab]
        return self.enc.decode(decodable)


def build_tokenizer(
    backend: str = "byte",
    model_path: str | Path | None = None,
    encoding_name: str = "gpt2",
) -> BaseTokenizer:
    """Construct a tokenizer for the requested backend.

    Args:
        backend: One of ``"byte"``, ``"sentencepiece"``, or ``"tiktoken"``.
        model_path: Path to a SentencePiece ``.model`` file (required for the
            ``sentencepiece`` backend).
        encoding_name: tiktoken encoding name (used by the ``tiktoken`` backend).

    Returns:
        An initialized tokenizer.

    Raises:
        ValueError: If the backend is unknown or required arguments are missing.
    """
    backend = backend.lower()
    if backend == "byte":
        return ByteTokenizer()
    if backend == "sentencepiece":
        if model_path is None:
            raise ValueError("model_path is required for the sentencepiece backend.")
        return SentencePieceTokenizer(model_path)
    if backend == "tiktoken":
        return TiktokenTokenizer(encoding_name)
    raise ValueError(f"Unknown tokenizer backend: {backend!r}")


def train_sentencepiece(
    input_path: str | Path,
    model_prefix: str | Path,
    vocab_size: int = 8000,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
    **kwargs: object,
) -> str:
    """Train a SentencePiece model from a text corpus.

    Args:
        input_path: Path to a UTF-8 text file (one document or sentence per line).
        model_prefix: Output prefix; produces ``<prefix>.model`` and ``.vocab``.
        vocab_size: Target vocabulary size.
        model_type: SentencePiece algorithm (``"bpe"``, ``"unigram"``, ...).
        character_coverage: Fraction of characters covered by the model.
        **kwargs: Additional keyword arguments forwarded to the trainer.

    Returns:
        The path to the trained ``.model`` file.
    """
    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        unk_id=3,
        **kwargs,
    )
    return f"{model_prefix}.model"


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the tokenizer command-line parser."""
    parser = argparse.ArgumentParser(description="Tokenizer utilities.")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a SentencePiece model.")
    train_p.add_argument("--input", required=True, help="Path to a text corpus.")
    train_p.add_argument("--model-prefix", required=True, help="Output model prefix.")
    train_p.add_argument("--vocab-size", type=int, default=8000)
    train_p.add_argument("--model-type", default="bpe")
    train_p.add_argument("--character-coverage", type=float, default=1.0)

    encode_p = sub.add_parser("encode", help="Encode text to token ids.")
    encode_p.add_argument("--backend", default="byte")
    encode_p.add_argument("--model-path", default=None)
    encode_p.add_argument("--encoding-name", default="gpt2")
    encode_p.add_argument("--text", required=True)
    encode_p.add_argument("--add-bos", action="store_true")
    encode_p.add_argument("--add-eos", action="store_true")

    decode_p = sub.add_parser("decode", help="Decode token ids to text.")
    decode_p.add_argument("--backend", default="byte")
    decode_p.add_argument("--model-path", default=None)
    decode_p.add_argument("--encoding-name", default="gpt2")
    decode_p.add_argument("--ids", type=int, nargs="+", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the tokenizer CLI.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).
    """
    args = _build_arg_parser().parse_args(argv)

    if args.command == "train":
        model_file = train_sentencepiece(
            input_path=args.input,
            model_prefix=args.model_prefix,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=args.character_coverage,
        )
        print(f"Trained SentencePiece model: {model_file}")
        return

    tokenizer = build_tokenizer(
        backend=args.backend,
        model_path=args.model_path,
        encoding_name=args.encoding_name,
    )

    if args.command == "encode":
        ids = tokenizer.encode(args.text, add_bos=args.add_bos, add_eos=args.add_eos)
        print(" ".join(str(i) for i in ids))
    elif args.command == "decode":
        print(tokenizer.decode(args.ids))


if __name__ == "__main__":
    main()
