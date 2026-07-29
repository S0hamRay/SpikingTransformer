"""Tavily web search for corrective RAG fallback."""

from __future__ import annotations

import os
from typing import Any


def tavily_configured() -> bool:
    """Return True when ``TAVILY_API_KEY`` is set."""
    return bool(os.getenv("TAVILY_API_KEY", "").strip())


def format_tavily_results(payload: Any, max_chars: int = 4000) -> str:
    """Normalize a Tavily tool response into plain text context."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload.strip()[:max_chars]

    results = payload
    if isinstance(payload, dict):
        results = payload.get("results", payload)

    if not isinstance(results, list):
        return str(payload).strip()[:max_chars]

    parts: list[str] = []
    for item in results:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            parts.append(str(item))
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        content = str(item.get("content") or item.get("raw_content") or "").strip()
        header = " — ".join(p for p in (title, url) if p)
        parts.append(f"{header}\n{content}".strip() if header else content)

    text = "\n\n".join(p for p in parts if p)
    return text[:max_chars]


def tavily_search(query: str, *, max_results: int = 3) -> str:
    """Search the web with Tavily and return concatenated snippets.

    Requires ``TAVILY_API_KEY`` in the environment (e.g. via ``.env``).
    """
    if not tavily_configured():
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Add it to your .env file "
            "(see .env.example)."
        )

    try:
        from langchain_tavily import TavilySearch
    except ImportError as exc:  # pragma: no cover - dependency wiring
        raise RuntimeError(
            "langchain-tavily is not installed. "
            "Run: pip install langchain-tavily"
        ) from exc

    tool = TavilySearch(max_results=max_results, topic="general")
    payload = tool.invoke({"query": query})
    return format_tavily_results(payload)
