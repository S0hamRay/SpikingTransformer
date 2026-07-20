"""PDF text extraction for RAG ingestion."""

from __future__ import annotations

from pathlib import Path


def read_pdf(path: str | Path) -> str:
    """Extract text from a PDF file using pypdf.

    Args:
        path: Path to a ``.pdf`` file.

    Returns:
        Concatenated page text with form-feed separators between pages.

    Raises:
        ImportError: If ``pypdf`` is not installed.
        ValueError: If the PDF has no extractable text.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "PDF support requires pypdf. Install with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())

    if not pages:
        raise ValueError(f"No extractable text in PDF: {path}")

    return "\n\n".join(pages)


def pdf_metadata(path: str | Path) -> dict[str, str | None]:
    """Pull basic PDF document-info metadata when present."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return {}

    reader = PdfReader(str(path))
    info = reader.metadata
    if info is None:
        return {"title": None, "author": None}

    title = getattr(info, "title", None)
    author = getattr(info, "author", None)
    return {
        "title": str(title).strip() if title else None,
        "author": str(author).strip() if author else None,
    }
