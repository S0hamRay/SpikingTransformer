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

from functools import lru_cache

from chatbot.engine import ChatEngine, default_checkpoint_for
from rag.engine import RAGEngine

DEFAULT_STT_BACKEND = "google"
CHECKPOINT_ROOT = "checkpoints"
# Stable session so Index + chat share one Chroma store (Gradio State UUIDs
# were regenerating and querying empty indexes).
GRADIO_RAG_SESSION = "gradio"


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
        reply = get_rag_engine(session_id or GRADIO_RAG_SESSION).generate_reply(
            message
        )
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
    """Clear the engine's conversation memory and the visible chat.

    Does not delete the RAG vector index — re-index only when uploading new docs.
    """
    if chat_mode == "rag":
        get_rag_engine(session_id or GRADIO_RAG_SESSION).reset(clear_index=False)
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
    return get_rag_engine(session_id or GRADIO_RAG_SESSION).ingest_files(paths)


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
        engine = get_rag_engine(session_id or GRADIO_RAG_SESSION)
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


def refresh_graph_view(
    view: str,
    max_nodes: int,
    include_chunks: bool,
):
    """Load Neo4j data into a Plotly figure for the Graph tab."""
    from db.viz import render_graph_view

    return render_graph_view(
        view=view,
        max_nodes=int(max_nodes),
        include_chunks=bool(include_chunks),
    )


def run_leiden_and_refresh(
    view: str,
    max_nodes: int,
    include_chunks: bool,
    gamma: float,
):
    """Run Leiden community detection, then refresh the graph plot."""
    from db.leiden import LeidenError, run_leiden

    try:
        stats = run_leiden(gamma=float(gamma) if gamma is not None else None)
        status = stats.format_summary()
    except LeidenError as exc:
        status = f"Leiden failed: {exc}"
    except Exception as exc:  # noqa: BLE001
        status = f"Leiden failed: {exc}"

    # Prefer the communities view after a successful clustering run.
    view_after = (
        "Communities"
        if status.startswith("Leiden communities")
        else view
    )
    fig, stats_md = refresh_graph_view(view_after, max_nodes, include_chunks)
    return fig, stats_md, status, view_after


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
        session_id = gr.State(GRADIO_RAG_SESSION)

        with gr.Tabs():
            with gr.Tab("Chat"):
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
                        sources=["microphone"],
                        type="numpy",
                        label="Or speak a message",
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
                index_btn.click(
                    ingest_documents, [doc_upload, session_id], [ingest_status]
                )
                transcribe_btn.click(transcribe, [mic], [msg])
                clear.click(reset_chat, [chat_mode, session_id], [chatbot])
                chat_mode.change(
                    lambda mode: toggle_rag_ui(mode, gr),
                    [chat_mode],
                    [rag_row, doc_upload, ingest_status, voice_row, transcribe_btn],
                )
                chat_mode.change(load_messages, [chat_mode, session_id], [chatbot])
                demo.load(load_messages, [chat_mode, session_id], [chatbot])

            with gr.Tab("Knowledge graph"):
                gr.Markdown(
                    "Interactive view of the Neo4j academic graph. "
                    "**Knowledge graph** shows Paper / Author / Concept links. "
                    "**Communities** colors nodes by Leiden clusters "
                    "(run Leiden after ingesting papers)."
                )
                with gr.Row():
                    graph_view = gr.Dropdown(
                        choices=["Knowledge graph", "Communities"],
                        value="Knowledge graph",
                        label="View",
                        scale=2,
                    )
                    max_nodes = gr.Slider(
                        minimum=20,
                        maximum=200,
                        value=80,
                        step=10,
                        label="Max nodes",
                        scale=2,
                    )
                    include_chunks = gr.Checkbox(
                        value=False,
                        label="Include Chunk nodes",
                        scale=1,
                    )
                with gr.Row():
                    refresh_btn = gr.Button("Refresh graph", variant="primary")
                    gamma = gr.Number(
                        value=1.0,
                        label="Leiden gamma",
                        precision=2,
                    )
                    leiden_btn = gr.Button("Run Leiden", variant="secondary")
                graph_plot = gr.Plot(label="Graph")
                with gr.Row():
                    graph_stats = gr.Markdown("### Graph stats\n_Click Refresh graph._")
                    leiden_status = gr.Markdown("")

                refresh_btn.click(
                    refresh_graph_view,
                    [graph_view, max_nodes, include_chunks],
                    [graph_plot, graph_stats],
                )
                graph_view.change(
                    refresh_graph_view,
                    [graph_view, max_nodes, include_chunks],
                    [graph_plot, graph_stats],
                )
                leiden_btn.click(
                    run_leiden_and_refresh,
                    [graph_view, max_nodes, include_chunks, gamma],
                    [graph_plot, graph_stats, leiden_status, graph_view],
                )
                demo.load(
                    refresh_graph_view,
                    [graph_view, max_nodes, include_chunks],
                    [graph_plot, graph_stats],
                )

    return demo


def main() -> None:
    """Launch FastAPI with Gradio mounted at ``/`` and MCP at ``/mcp``."""
    from api.server import main as serve

    serve()


if __name__ == "__main__":
    main()
