"""Utility functions shared across agents."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_pdf_text(path: str | Path) -> str:
    """Extract plain text from a PDF file."""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        raise RuntimeError("pypdf is required: pip install pypdf")


def load_docx_text(path: str | Path) -> str:
    """Extract plain text from a .docx file."""
    try:
        import docx
        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    except ImportError:
        raise RuntimeError("python-docx is required: pip install python-docx")


def load_resume(path: str | Path) -> str:
    """Auto-detect format and return resume text."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return load_pdf_text(p)
    if ext in (".docx", ".doc"):
        return load_docx_text(p)
    return p.read_text(encoding="utf-8")


def extract_json_block(text: str) -> dict[str, Any] | list[Any]:
    """
    Pull the first JSON object or array out of an LLM response that may
    contain markdown fences or prose around the JSON.
    """
    # Strip ```json ... ``` fences
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first { or [ and attempt to parse from there
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"No valid JSON found in LLM response:\n{text[:500]}")


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def truncate(text: str, max_chars: int = 8000) -> str:
    """Prevent context overflow by truncating very long inputs."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated ...]"
