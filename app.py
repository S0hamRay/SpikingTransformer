"""Gradio web chatbot with selectable attention and voice input.

Launches a browser chat UI where you can:

* pick the attention variant (spiking or standard) from a dropdown,
* type a message, or
* record a message with your microphone and transcribe it to text.

Run with:
    python app.py

The microphone is captured by the browser (via Gradio), so no system audio
libraries are required for the web UI; transcription uses the SpeechRecognition
backend configured below.
"""

from __future__ import annotations

import argparse
from functools import lru_cache

from chatbot.engine import ChatEngine, default_checkpoint_for

DEFAULT_STT_BACKEND = "google"
CHECKPOINT_ROOT = "checkpoints"


@lru_cache(maxsize=4)
def get_engine(attn_type: str) -> ChatEngine:
    """Load (and cache) the chat engine for an attention variant.

    Args:
        attn_type: ``"spiking"`` or ``"standard"``.

    Returns:
        The cached :class:`ChatEngine`.

    Raises:
        FileNotFoundError: If the corresponding checkpoint does not exist.
    """
    path = default_checkpoint_for(attn_type, CHECKPOINT_ROOT)
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Train it first:\n"
            f"  python train.py --config configs/shakespeare.yaml "
            f"--attn-type {attn_type} --checkpoint-dir {path.parent}"
        )
    return ChatEngine.from_checkpoint(path)


@lru_cache(maxsize=1)
def get_transcriber():
    """Lazily build and cache the speech transcriber."""
    from chatbot.speech import SpeechTranscriber

    return SpeechTranscriber(backend=DEFAULT_STT_BACKEND)


def transcribe(audio) -> str:
    """Transcribe a Gradio microphone recording into text.

    Args:
        audio: A ``(sample_rate, numpy_array)`` tuple from a Gradio Audio
            component, or ``None``.

    Returns:
        The recognized text, or an empty/diagnostic string.
    """
    if audio is None:
        return ""
    sample_rate, samples = audio
    try:
        return get_transcriber().transcribe_array(samples, sample_rate)
    except Exception as exc:  # noqa: BLE001 - surface any STT issue in the UI
        return f"[speech-to-text error: {exc}]"


def respond(message: str, history: list[dict], attn_type: str) -> tuple[list[dict], str]:
    """Generate a reply and append the turn to the chat history.

    Args:
        message: The user's message.
        history: The chat history (list of role/content dicts).
        attn_type: The selected attention variant.

    Returns:
        The updated history and a cleared input box value.
    """
    message = (message or "").strip()
    if not message:
        return history, ""
    try:
        engine = get_engine(attn_type)
    except FileNotFoundError as exc:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": str(exc)},
        ]
        return history, ""

    reply = engine.generate_reply(message)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply or "..."},
    ]
    return history, ""


def reset_engine(attn_type: str) -> list[dict]:
    """Clear the engine's conversation memory and the visible chat."""
    try:
        get_engine(attn_type).reset()
    except FileNotFoundError:
        pass
    return []


def history_to_messages(engine: ChatEngine) -> list[dict]:
    """Convert an engine's stored history into Gradio chat messages."""
    messages: list[dict] = []
    for user, bot in engine.history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": bot})
    return messages


def load_messages(attn_type: str) -> list[dict]:
    """Load a variant's persisted conversation for display in the UI."""
    try:
        return history_to_messages(get_engine(attn_type))
    except FileNotFoundError:
        return []


def build_demo():
    """Construct the Gradio Blocks demo."""
    import gradio as gr

    with gr.Blocks(title="Spiking vs Standard Attention Chatbot") as demo:
        gr.Markdown(
            "# Spiking Transformer Chatbot\n"
            "Chat with a character-level model trained on tiny Shakespeare. "
            "Pick the **attention variant** and type or **speak** your message."
        )
        with gr.Row():
            attn_type = gr.Dropdown(
                choices=["spiking", "standard"],
                value="spiking",
                label="Attention variant",
            )
        chatbot = gr.Chatbot(height=380, label="Conversation")
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Type a message and press Enter...",
                label="Message",
                scale=4,
            )
            send = gr.Button("Send", variant="primary", scale=1)
        with gr.Row():
            mic = gr.Audio(
                sources=["microphone"], type="numpy", label="Or speak a message"
            )
            transcribe_btn = gr.Button("Transcribe to message box")
        clear = gr.Button("Clear conversation")

        send.click(respond, [msg, chatbot, attn_type], [chatbot, msg])
        msg.submit(respond, [msg, chatbot, attn_type], [chatbot, msg])
        transcribe_btn.click(transcribe, [mic], [msg])
        clear.click(reset_engine, [attn_type], [chatbot])
        # Show the selected variant's saved conversation on switch and on open.
        attn_type.change(load_messages, [attn_type], [chatbot])
        demo.load(load_messages, [attn_type], [chatbot])

    return demo


def main() -> None:
    """Launch the web UI."""
    parser = argparse.ArgumentParser(description="Launch the chatbot web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public link.")
    args = parser.parse_args()

    demo = build_demo()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
