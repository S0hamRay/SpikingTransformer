"""Command-line chatbot.

Talk to a trained model from the terminal using either the spiking or the
standard attention variant. Optionally speak instead of typing via the
``--voice`` flag (requires a microphone and the SpeechRecognition library).

Examples:
    python chat.py --attn-type spiking
    python chat.py --attn-type standard --voice
    python chat.py --checkpoint checkpoints/spiking/best.pt

In-chat commands:
    /reset   clear the conversation history
    /voice   capture a single spoken message (works without --voice)
    /quit    exit
"""

from __future__ import annotations

import argparse

from chatbot.engine import ChatEngine, GenerationSettings, default_checkpoint_for
from utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Chat with a trained model.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--attn-type",
        choices=["spiking", "standard"],
        default="spiking",
        help="Attention variant to load (uses checkpoints/<attn-type>/best.pt).",
    )
    source.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit checkpoint path (overrides --attn-type).",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints",
        help="Root directory holding per-variant checkpoints.",
    )
    parser.add_argument("--voice", action="store_true", help="Use the microphone.")
    parser.add_argument("--stt-backend", default="google", help="STT backend.")
    parser.add_argument(
        "--history-file",
        default=None,
        help="Override the conversation-history file (defaults to a per-variant "
        "history.json next to the checkpoint).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start a new conversation, ignoring and overwriting any saved history.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--greedy", action="store_true")
    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> GenerationSettings | None:
    """Build optional generation settings from CLI overrides."""
    overrides = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if args.greedy:
        overrides["greedy"] = True
    if not overrides:
        return None
    base = GenerationSettings()
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def main() -> None:
    """Run the interactive chat loop."""
    args = parse_args()
    logger = get_logger()

    settings = build_settings(args)
    history_path = args.history_file if args.history_file is not None else None
    if args.checkpoint:
        engine = ChatEngine.from_checkpoint(
            args.checkpoint, settings=settings, history_path=history_path
        )
    else:
        path = default_checkpoint_for(args.attn_type, args.checkpoint_root)
        if not path.exists():
            logger.error(
                "Checkpoint not found: %s. Train it first, e.g.:\n  python train.py "
                "--config configs/shakespeare.yaml --attn-type %s --checkpoint-dir %s",
                path,
                args.attn_type,
                path.parent,
            )
            return
        engine = ChatEngine.from_checkpoint(
            path, settings=settings, history_path=history_path
        )

    if args.fresh:
        # Discard any loaded history and overwrite the persisted file.
        engine.reset()
    elif engine.history:
        print(f"[loaded {len(engine.history)} previous turn(s)]")

    transcriber = None
    if args.voice:
        from chatbot.speech import SpeechRecognitionError, SpeechTranscriber

        try:
            transcriber = SpeechTranscriber(backend=args.stt_backend)
        except SpeechRecognitionError as exc:
            logger.error("Voice mode unavailable: %s", exc)
            return

    print(f"\nChatbot ready (attention: {engine.attn_type}, device: {engine.device}).")
    print("Type a message, or use /voice, /reset, /quit.\n")

    while True:
        try:
            if args.voice:
                print("[listening... speak now]")
                user_text = _capture_voice(transcriber, args, logger)
                print(f"You (voice): {user_text}")
            else:
                user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_text:
            continue
        command = user_text.lower()
        if command in {"/quit", "/exit"}:
            print("Goodbye.")
            break
        if command == "/reset":
            engine.reset()
            print("[history cleared]")
            continue
        if command == "/voice":
            if transcriber is None:
                from chatbot.speech import SpeechRecognitionError, SpeechTranscriber

                try:
                    transcriber = SpeechTranscriber(backend=args.stt_backend)
                except SpeechRecognitionError as exc:
                    print(f"[voice unavailable: {exc}]")
                    continue
            print("[listening... speak now]")
            user_text = _capture_voice(transcriber, args, logger)
            print(f"You (voice): {user_text}")
            if not user_text:
                continue

        reply = engine.generate_reply(user_text)
        print(f"Bot: {reply}\n")


def _capture_voice(transcriber, args: argparse.Namespace, logger) -> str:
    """Capture one spoken phrase, returning '' on failure."""
    from chatbot.speech import SpeechRecognitionError

    try:
        return transcriber.transcribe_microphone(phrase_time_limit=15)
    except SpeechRecognitionError as exc:
        logger.error("Voice capture failed: %s", exc)
        return ""


if __name__ == "__main__":
    main()
