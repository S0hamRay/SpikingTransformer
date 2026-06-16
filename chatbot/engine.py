"""Chat generation engine.

Wraps a trained language model behind a simple conversational API. The engine
can load either the spiking or the standard-attention checkpoint (the attention
variant is recorded in the checkpoint's config), maintains a rolling
conversation transcript, and produces a reply for each user turn.

The models in this repo are small character-level models trained on tiny
Shakespeare, so replies are stylistic continuations rather than instruction
following -- the value here is the end-to-end interface, with the attention
mechanism fully swappable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from data.tokenizer import BaseTokenizer
from eval.generation import generate
from train.builders import build_model_from_config, build_tokenizer_from_config
from utils.checkpointing import load_checkpoint
from utils.config import select_device


@dataclass
class GenerationSettings:
    """Sampling settings for reply generation."""

    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    greedy: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GenerationSettings":
        """Build settings from a config's ``generation`` section."""
        gen = config.get("generation", {})
        return cls(
            max_new_tokens=gen.get("max_new_tokens", 200),
            temperature=gen.get("temperature", 0.8),
            top_k=gen.get("top_k", 40),
            top_p=gen.get("top_p", 0.95),
            greedy=gen.get("greedy", False),
        )


def default_checkpoint_for(attn_type: str, root: str | Path = "checkpoints") -> Path:
    """Return the conventional checkpoint path for an attention variant.

    Args:
        attn_type: ``"spiking"`` or ``"standard"``.
        root: Root checkpoint directory.

    Returns:
        Path ``<root>/<attn_type>/best.pt``.
    """
    return Path(root) / attn_type / "best.pt"


def default_history_for(attn_type: str, root: str | Path = "checkpoints") -> Path:
    """Return the conventional conversation-history path for a variant.

    Args:
        attn_type: ``"spiking"`` or ``"standard"``.
        root: Root checkpoint directory.

    Returns:
        Path ``<root>/<attn_type>/history.json``.
    """
    return Path(root) / attn_type / "history.json"


class ChatEngine:
    """Conversational wrapper around a trained language model."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: BaseTokenizer,
        config: dict[str, Any],
        device: torch.device,
        settings: GenerationSettings | None = None,
        max_context_chars: int = 1000,
        history_path: str | Path | None = None,
    ) -> None:
        """Initialize the chat engine.

        Args:
            model: The (loaded) language model.
            tokenizer: Tokenizer matching the model.
            config: The experiment config the model was built from.
            device: Device the model lives on.
            settings: Sampling settings; defaults derived from ``config``.
            max_context_chars: Maximum number of transcript characters fed back
                into the model as context for the next turn.
            history_path: Optional JSON file used to persist the conversation
                history across sessions. When set, prior history is loaded on
                construction and saved after every turn.
        """
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.settings = settings or GenerationSettings.from_config(config)
        self.max_context_chars = max_context_chars
        self.attn_type = config.get("model", {}).get("attn_type", "spiking")
        self.history: list[tuple[str, str]] = []

        self.history_path = Path(history_path) if history_path else None
        if self.history_path is not None and self.history_path.exists():
            self.load_history(self.history_path)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device | None = None,
        settings: GenerationSettings | None = None,
        history_path: str | Path | None | bool = None,
    ) -> "ChatEngine":
        """Build a chat engine from a saved checkpoint.

        Args:
            checkpoint_path: Path to a ``.pt`` checkpoint produced by training.
            device: Optional device override.
            settings: Optional sampling settings override.
            history_path: Conversation persistence file. ``None`` (default)
                stores history next to the checkpoint as ``history.json``; pass
                ``False`` to disable persistence, or an explicit path to override.

        Returns:
            A ready-to-use :class:`ChatEngine`.
        """
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config")
        if config is None:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has no embedded config; cannot "
                "rebuild the model."
            )
        device = device or select_device(config.get("device", "auto"))
        tokenizer = build_tokenizer_from_config(config)
        model = build_model_from_config(config, tokenizer)
        model.load_state_dict(checkpoint["model_state_dict"])

        resolved_history = cls._resolve_history_path(
            history_path, Path(checkpoint_path).parent / "history.json"
        )
        return cls(
            model,
            tokenizer,
            config,
            device,
            settings=settings,
            history_path=resolved_history,
        )

    @classmethod
    def from_attn_type(
        cls,
        attn_type: str,
        checkpoint_root: str | Path = "checkpoints",
        device: torch.device | None = None,
        settings: GenerationSettings | None = None,
        history_path: str | Path | None | bool = None,
    ) -> "ChatEngine":
        """Build a chat engine for a named attention variant.

        Args:
            attn_type: ``"spiking"`` or ``"standard"``.
            checkpoint_root: Root directory containing per-variant checkpoints.
            device: Optional device override.
            settings: Optional sampling settings override.
            history_path: Conversation persistence file. ``None`` (default) uses
                the per-variant ``history.json``; ``False`` disables persistence.

        Returns:
            A ready-to-use :class:`ChatEngine`.
        """
        path = default_checkpoint_for(attn_type, checkpoint_root)
        resolved_history = cls._resolve_history_path(
            history_path, default_history_for(attn_type, checkpoint_root)
        )
        return cls.from_checkpoint(
            path, device=device, settings=settings, history_path=resolved_history
        )

    @staticmethod
    def _resolve_history_path(
        history_path: str | Path | None | bool,
        default: Path,
    ) -> Path | None:
        """Resolve the persistence path: ``None`` -> default, ``False`` -> off."""
        if history_path is False:
            return None
        if history_path is None:
            return default
        return Path(history_path)

    def load_history(self, path: str | Path | None = None) -> None:
        """Load conversation history from a JSON file.

        A missing or corrupt file is treated as an empty history rather than an
        error, so a fresh conversation simply starts clean.

        Args:
            path: File to read; defaults to ``self.history_path``.
        """
        target = Path(path) if path is not None else self.history_path
        if target is None:
            return
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            self.history = [(str(user), str(bot)) for user, bot in data]
        except (OSError, ValueError, TypeError):
            self.history = []

    def save_history(self, path: str | Path | None = None) -> None:
        """Persist the conversation history to a JSON file atomically.

        Args:
            path: File to write; defaults to ``self.history_path``. No-op when
                neither is set.
        """
        target = Path(path) if path is not None else self.history_path
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(
            json.dumps([list(turn) for turn in self.history]), encoding="utf-8"
        )
        os.replace(tmp, target)

    def reset(self) -> None:
        """Clear the conversation history (and the persisted file, if any)."""
        self.history = []
        self.save_history()

    def _build_prompt(self, user_text: str) -> str:
        """Assemble the model prompt from history and the new user turn."""
        lines: list[str] = []
        for user, bot in self.history:
            lines.append(f"{user}\n{bot}")
        lines.append(user_text)
        transcript = "\n".join(lines) + "\n"
        # Keep only the most recent characters as context.
        return transcript[-self.max_context_chars :]

    def _clean_reply(self, continuation: str) -> str:
        """Trim a raw continuation into a single conversational turn."""
        reply = continuation.strip("\n")
        # Take up to the first blank line so a turn does not run on forever.
        for sep in ("\n\n", "\n"):
            if sep in reply:
                candidate = reply.split(sep, 1)[0].strip()
                if len(candidate) >= 2:
                    return candidate
        return reply.strip()

    @torch.no_grad()
    def generate_reply(self, user_text: str) -> str:
        """Generate a reply to a user message and update the history.

        Args:
            user_text: The user's message.

        Returns:
            The model's reply text.
        """
        user_text = user_text.strip()
        if not user_text:
            return ""

        prompt = self._build_prompt(user_text)
        prompt_ids = self.tokenizer.encode(prompt) or [
            self.tokenizer.bos_id if self.tokenizer.bos_id is not None else 0
        ]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        output_ids = generate(
            self.model,
            input_ids,
            max_new_tokens=self.settings.max_new_tokens,
            temperature=self.settings.temperature,
            top_k=self.settings.top_k,
            top_p=self.settings.top_p,
            greedy=self.settings.greedy,
            eos_id=self.tokenizer.eos_id,
        )

        new_ids = output_ids[0, input_ids.shape[1] :].tolist()
        reply = self._clean_reply(self.tokenizer.decode(new_ids))
        self.history.append((user_text, reply))
        self.save_history()
        return reply
