"""Voice-to-text using the ``SpeechRecognition`` library.

This wraps `SpeechRecognition <https://pypi.org/project/SpeechRecognition/>`_ to
turn spoken audio into text. It supports three input sources:

* microphone capture (CLI use; requires ``PyAudio``),
* a raw audio array such as the one Gradio yields from a browser microphone,
* an audio file.

Several recognizer backends are selectable:

* ``"google"``  -- Google Web Speech API (free, needs an internet connection),
* ``"whisper"`` -- local OpenAI Whisper (offline; requires ``openai-whisper``),
* ``"sphinx"``  -- CMU PocketSphinx (offline; requires ``pocketsphinx``).

All third-party imports are lazy so the rest of the project works without these
optional dependencies installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class SpeechRecognitionError(RuntimeError):
    """Raised when transcription fails or a backend is unavailable."""


class SpeechTranscriber:
    """Transcribe speech to text via the ``SpeechRecognition`` library."""

    def __init__(
        self,
        backend: str = "google",
        language: str = "en-US",
        whisper_model: str = "base",
        energy_threshold: int | None = None,
    ) -> None:
        """Initialize the transcriber.

        Args:
            backend: Recognizer backend (``"google"``, ``"whisper"``, ``"sphinx"``).
            language: Language/locale code for backends that accept one.
            whisper_model: Whisper model size when ``backend="whisper"``.
            energy_threshold: Optional fixed mic energy threshold; when ``None``
                the recognizer auto-calibrates to ambient noise.
        """
        try:
            import speech_recognition as sr
        except ImportError as exc:  # pragma: no cover - exercised only w/o dep
            raise SpeechRecognitionError(
                "SpeechRecognition is not installed. Install it with "
                "`pip install SpeechRecognition`."
            ) from exc

        self._sr = sr
        self.backend = backend.lower()
        self.language = language
        self.whisper_model = whisper_model
        self.recognizer = sr.Recognizer()
        if energy_threshold is not None:
            self.recognizer.energy_threshold = energy_threshold
            self.recognizer.dynamic_energy_threshold = False

    def _recognize(self, audio: Any) -> str:
        """Run the configured backend on a ``SpeechRecognition`` AudioData."""
        sr = self._sr
        try:
            if self.backend == "google":
                return self.recognizer.recognize_google(audio, language=self.language)
            if self.backend == "whisper":
                return self.recognizer.recognize_whisper(
                    audio, model=self.whisper_model
                )
            if self.backend == "sphinx":
                return self.recognizer.recognize_sphinx(audio)
            raise SpeechRecognitionError(f"Unknown backend: {self.backend!r}")
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as exc:
            raise SpeechRecognitionError(
                f"Speech recognition request failed: {exc}"
            ) from exc

    def transcribe_microphone(
        self,
        timeout: float | None = None,
        phrase_time_limit: float | None = None,
        calibrate_seconds: float = 0.5,
    ) -> str:
        """Capture a phrase from the default microphone and transcribe it.

        Args:
            timeout: Seconds to wait for speech to start before giving up.
            phrase_time_limit: Maximum seconds to record once speech starts.
            calibrate_seconds: Seconds spent calibrating to ambient noise.

        Returns:
            The recognized text (empty string if nothing was understood).
        """
        sr = self._sr
        try:
            microphone = sr.Microphone()
        except (OSError, AttributeError) as exc:
            raise SpeechRecognitionError(
                "No microphone available. Microphone capture requires PyAudio "
                "(`pip install pyaudio`; on macOS `brew install portaudio` first)."
            ) from exc

        with microphone as source:
            if calibrate_seconds > 0:
                self.recognizer.adjust_for_ambient_noise(
                    source, duration=calibrate_seconds
                )
            audio = self.recognizer.listen(
                source, timeout=timeout, phrase_time_limit=phrase_time_limit
            )
        return self._recognize(audio)

    def transcribe_array(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe a raw audio waveform array.

        Args:
            audio: Audio samples. Accepts float arrays in ``[-1, 1]`` or integer
                PCM arrays, mono or multi-channel (channels are averaged).
            sample_rate: Sample rate of the audio in Hz.

        Returns:
            The recognized text.
        """
        samples = np.asarray(audio)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        # Normalize to 16-bit PCM bytes expected by SpeechRecognition.
        if np.issubdtype(samples.dtype, np.floating):
            samples = np.clip(samples, -1.0, 1.0)
            samples = (samples * 32767.0).astype(np.int16)
        else:
            samples = samples.astype(np.int16)

        audio_data = self._sr.AudioData(
            samples.tobytes(), sample_rate=int(sample_rate), sample_width=2
        )
        return self._recognize(audio_data)

    def transcribe_file(self, path: str | Path) -> str:
        """Transcribe an audio file (e.g. WAV/AIFF/FLAC).

        Args:
            path: Path to the audio file.

        Returns:
            The recognized text.
        """
        with self._sr.AudioFile(str(path)) as source:
            audio = self.recognizer.record(source)
        return self._recognize(audio)
