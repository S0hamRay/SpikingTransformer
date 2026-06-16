"""Tests for the chatbot engine and speech-to-text wiring."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from chatbot.engine import (
    ChatEngine,
    GenerationSettings,
    default_checkpoint_for,
    default_history_for,
)
from data.tokenizer import ByteTokenizer
from train.builders import build_model_from_config
from utils.checkpointing import save_checkpoint


def _make_checkpoint(tmp_path: Path, attn_type: str = "standard") -> Path:
    """Create a tiny trained-style checkpoint for testing."""
    config = {
        "device": "cpu",
        "tokenizer": {"backend": "byte"},
        "model": {
            "vocab_size": ByteTokenizer().vocab_size,
            "d_model": 16,
            "n_layers": 2,
            "n_heads": 2,
            "max_seq_len": 64,
            "attn_type": attn_type,
            "dropout": 0.0,
        },
        "generation": {
            "max_new_tokens": 16,
            "temperature": 0.8,
            "top_k": 10,
            "top_p": 0.95,
            "greedy": True,
        },
    }
    tokenizer = ByteTokenizer()
    model = build_model_from_config(config, tokenizer)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, config=config, step=10)
    return path


def test_default_checkpoint_for() -> None:
    assert default_checkpoint_for("spiking") == Path("checkpoints/spiking/best.pt")
    assert default_checkpoint_for("standard", "ckpt") == Path("ckpt/standard/best.pt")


def test_engine_from_checkpoint_generates_reply(tmp_path: Path) -> None:
    path = _make_checkpoint(tmp_path)
    engine = ChatEngine.from_checkpoint(path)
    reply = engine.generate_reply("Hello there")
    assert isinstance(reply, str)
    assert len(engine.history) == 1
    assert engine.history[0][0] == "Hello there"


def test_engine_reset_clears_history(tmp_path: Path) -> None:
    engine = ChatEngine.from_checkpoint(_make_checkpoint(tmp_path))
    engine.generate_reply("first")
    engine.generate_reply("second")
    assert len(engine.history) == 2
    engine.reset()
    assert engine.history == []


def test_engine_respects_attn_type(tmp_path: Path) -> None:
    engine = ChatEngine.from_checkpoint(_make_checkpoint(tmp_path, attn_type="spiking"))
    assert engine.attn_type == "spiking"


def test_empty_message_returns_empty(tmp_path: Path) -> None:
    engine = ChatEngine.from_checkpoint(_make_checkpoint(tmp_path))
    assert engine.generate_reply("   ") == ""
    assert engine.history == []


def test_generation_settings_from_config() -> None:
    settings = GenerationSettings.from_config(
        {"generation": {"max_new_tokens": 5, "temperature": 0.5}}
    )
    assert settings.max_new_tokens == 5
    assert settings.temperature == 0.5


def test_default_history_for() -> None:
    assert default_history_for("spiking") == Path("checkpoints/spiking/history.json")
    assert default_history_for("standard", "ckpt") == Path("ckpt/standard/history.json")


def test_history_save_load_round_trip(tmp_path: Path) -> None:
    engine = ChatEngine.from_checkpoint(_make_checkpoint(tmp_path))
    engine.history = [("hi", "ho"), ("foo", "bar")]
    history_file = tmp_path / "hist.json"
    engine.save_history(history_file)

    restored = ChatEngine.from_checkpoint(
        _make_checkpoint(tmp_path), history_path=history_file
    )
    assert restored.history == [("hi", "ho"), ("foo", "bar")]


def test_generate_reply_persists_to_default_path(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path)
    engine = ChatEngine.from_checkpoint(checkpoint)
    # Default history path sits next to the checkpoint.
    assert engine.history_path == checkpoint.parent / "history.json"

    engine.generate_reply("hello")
    assert engine.history_path.exists()

    # A fresh engine on the same checkpoint reloads the saved turn.
    reloaded = ChatEngine.from_checkpoint(checkpoint)
    assert len(reloaded.history) == 1
    assert reloaded.history[0][0] == "hello"


def test_reset_empties_persisted_file(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path)
    engine = ChatEngine.from_checkpoint(checkpoint)
    engine.generate_reply("hello")
    engine.reset()

    reloaded = ChatEngine.from_checkpoint(checkpoint)
    assert reloaded.history == []


def test_history_path_false_disables_persistence(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path)
    engine = ChatEngine.from_checkpoint(checkpoint, history_path=False)
    assert engine.history_path is None
    engine.generate_reply("hello")
    assert not (checkpoint.parent / "history.json").exists()


def test_corrupt_history_file_starts_clean(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    engine = ChatEngine.from_checkpoint(
        _make_checkpoint(tmp_path), history_path=bad
    )
    assert engine.history == []


@pytest.mark.skipif(
    importlib.util.find_spec("speech_recognition") is None,
    reason="SpeechRecognition not installed",
)
def test_transcribe_array_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The array path should build AudioData and call the chosen backend."""
    from chatbot.speech import SpeechTranscriber

    transcriber = SpeechTranscriber(backend="google")
    captured = {}

    def fake_recognize_google(audio, language=None):  # noqa: ANN001
        captured["sample_rate"] = audio.sample_rate
        captured["width"] = audio.sample_width
        return "transcribed text"

    monkeypatch.setattr(
        transcriber.recognizer, "recognize_google", fake_recognize_google
    )
    samples = (np.sin(np.linspace(0, 6.28, 16000)) * 0.2).astype(np.float32)
    text = transcriber.transcribe_array(samples, sample_rate=16000)
    assert text == "transcribed text"
    assert captured["sample_rate"] == 16000
    assert captured["width"] == 2
