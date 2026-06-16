"""Chatbot package: generation engine and voice-to-text."""

from __future__ import annotations

from chatbot.engine import (
    ChatEngine,
    GenerationSettings,
    default_checkpoint_for,
    default_history_for,
)
from chatbot.speech import SpeechRecognitionError, SpeechTranscriber

__all__ = [
    "ChatEngine",
    "GenerationSettings",
    "default_checkpoint_for",
    "default_history_for",
    "SpeechTranscriber",
    "SpeechRecognitionError",
]
