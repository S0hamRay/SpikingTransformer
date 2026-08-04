"""Gradio web chatbot with selectable attention and voice input.

The UI is built by :func:`build_demo` and served as a FastAPI sub-app at
``/`` (see :mod:`api.server`). Prefer:

    python -m api.server
    # Gradio UI → http://127.0.0.1:8000/
    # OpenAPI   → http://127.0.0.1:8000/docs
    # MCP       → http://127.0.0.1:8000/mcp

``python app.py`` remains a compatibility entry point that starts the same
FastAPI process (engines + Gradio + MCP).
"""

from __future__ import annotations

import uuid
from functools import lru_cache

from chatbot.engine import ChatEngine, default_checkpoint_for
from rag.engine import RAGEngine

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


@lru_cache(maxsize=8)
def get_rag_engine(session_id: str) -> RAGEngine:
    """Load (and cache) the RAG engine for a browser session."""
    return RAGEngine.from_session(session_id)


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


def respond(
    message: str,
    history: list[dict],
    chat_mode: str,
    session_id: str,
) -> tuple[list[dict], str]:
    """Generate a reply and append the turn to the chat history.

    Args:
        message: The user's message.
        history: The chat history (list of role/content dicts).
        chat_mode: ``"spiking"``, ``"standard"``, or ``"rag"``.
        session_id: Browser session id for RAG vector storage.

    Returns:
        The updated history and a cleared input box value.
    """
    message = (message or "").strip()
    if not message:
        return history, ""

    if chat_mode == "rag":
        reply = get_rag_engine(session_id).generate_reply(message)
    else:
        try:
            engine = get_engine(chat_mode)
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


def reset_chat(chat_mode: str, session_id: str) -> list[dict]:
    """Clear the engine's conversation memory and the visible chat."""
    if chat_mode == "rag":
        get_rag_engine(session_id).reset()
    else:
        try:
            get_engine(chat_mode).reset()
        except FileNotFoundError:
            pass
    return []


def ingest_documents(files, session_id: str) -> str:
    """Index uploaded plaintext/PDF files for the RAG session."""
    if not files:
        return "Upload one or more .txt or .pdf files to index."

    paths = [f.name for f in files]
    return get_rag_engine(session_id).ingest_files(paths)


def history_to_messages(engine: ChatEngine) -> list[dict]:
    """Convert an engine's stored history into Gradio chat messages."""
    messages: list[dict] = []
    for user, bot in engine.history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": bot})
    return messages


def load_messages(chat_mode: str, session_id: str) -> list[dict]:
    """Load a mode's persisted conversation for display in the UI."""
    if chat_mode == "rag":
        engine = get_rag_engine(session_id)
        return history_to_messages_rag(engine)
    try:
        return history_to_messages(get_engine(chat_mode))
    except FileNotFoundError:
        return []


def history_to_messages_rag(engine: RAGEngine) -> list[dict]:
    """Convert RAG history tuples into Gradio chat messages."""
    messages: list[dict] = []
    for user, bot in engine.history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": bot})
    return messages


def toggle_rag_ui(chat_mode: str, gr):
    """Show RAG upload controls only in RAG mode."""
    is_rag = chat_mode == "rag"
    return (
        gr.update(visible=is_rag),
        gr.update(visible=is_rag),
        gr.update(visible=is_rag),
        gr.update(visible=not is_rag),
        gr.update(visible=not is_rag),
    )


def build_demo():
    """Construct the Gradio Blocks demo (mounted by FastAPI at ``/``)."""
    import gradio as gr

    with gr.Blocks(title="Spiking vs Standard Attention Chatbot") as demo:
        gr.Markdown(
            "# Spiking Transformer Chatbot\n"
            "Chat with a character-level model trained on tiny Shakespeare, "
            "or use **RAG** mode to upload `.txt` / `.pdf` documents and ask "
            "questions with a local Ollama model (synced to Postgres/Neo4j when available)."
        )
        session_id = gr.State(value=lambda: str(uuid.uuid4()))

        with gr.Row():
            chat_mode = gr.Dropdown(
                choices=["spiking", "standard", "rag"],
                value="spiking",
                label="Chat mode",
            )

        with gr.Row(visible=False) as rag_row:
            doc_upload = gr.File(
                file_count="multiple",
                file_types=[".txt", ".pdf"],
                label="Upload documents (.txt / .pdf)",
            )
            index_btn = gr.Button("Index documents", variant="secondary")
            ingest_status = gr.Markdown("")

        chatbot = gr.Chatbot(height=380, label="Conversation")
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Type a message and press Enter...",
                label="Message",
                scale=4,
            )
            send = gr.Button("Send", variant="primary", scale=1)
        with gr.Row(visible=True) as voice_row:
            mic = gr.Audio(
                sources=["microphone"], type="numpy", label="Or speak a message"
            )
            transcribe_btn = gr.Button("Transcribe to message box")
        clear = gr.Button("Clear conversation")

        send.click(
            respond,
            [msg, chatbot, chat_mode, session_id],
            [chatbot, msg],
        )
        msg.submit(
            respond,
            [msg, chatbot, chat_mode, session_id],
            [chatbot, msg],
        )
        index_btn.click(ingest_documents, [doc_upload, session_id], [ingest_status])
        transcribe_btn.click(transcribe, [mic], [msg])
        clear.click(reset_chat, [chat_mode, session_id], [chatbot])
        chat_mode.change(
            lambda mode: toggle_rag_ui(mode, gr),
            [chat_mode],
            [rag_row, doc_upload, ingest_status, voice_row, transcribe_btn],
        )
        chat_mode.change(load_messages, [chat_mode, session_id], [chatbot])
        demo.load(load_messages, [chat_mode, session_id], [chatbot])

    return demo


def main() -> None:
    """Launch FastAPI with Gradio mounted at ``/`` and MCP at ``/mcp``."""
    from api.server import main as serve

    serve()


if __name__ == "__main__":
    main()
