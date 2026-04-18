"""
chunker.py — splits a Document into indexable Chunks.

Strategy: paragraph-first, then sentence-aware trimming.
No deps — pure stdlib re + dataclasses.

Rules:
- Split on blank lines (paragraph boundaries) first.
- If a paragraph exceeds max_chars, split further on sentence endings.
- Overlap: carry last `overlap_chars` of previous chunk into next.
  This preserves context across chunk boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from relevancedb.ingest.loader import Document

# sentence boundary: ends with . ! ? followed by whitespace or end-of-string
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


@dataclass
class Chunk:
    """A text chunk ready for embedding and storage."""

    text: str
    doc_path: Path
    doc_format: str
    chunk_index: int             
    char_start: int              
    char_end: int
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"Chunk(index={self.chunk_index}, "
            f"chars={len(self.text)}, "
            f"preview={preview!r})"
        )


def chunk(
    doc: Document,
    max_chars: int = 512,
    overlap_chars: int = 64,
) -> list[Chunk]:
    """
    Split a Document into Chunks.

    Args:
        doc:           The document to chunk.
        max_chars:     Target maximum characters per chunk.
        overlap_chars: How many characters of the previous chunk
                       to prepend to the next for context continuity.

    Returns:
        Ordered list of Chunks.
    """
    paragraphs = _split_paragraphs(doc.text)
    raw_chunks = _merge_paragraphs(paragraphs, max_chars)

    chunks = []
    carry = ""       # overlap from previous chunk
    char_cursor = 0

    for i, text in enumerate(raw_chunks):
        full_text = (carry + " " + text).strip() if carry else text
        char_start = max(0, char_cursor - len(carry))
        char_end = char_start + len(full_text)

        chunks.append(Chunk(
            text=full_text,
            doc_path=doc.path,
            doc_format=doc.format,
            chunk_index=i,
            char_start=char_start,
            char_end=char_end,
            metadata=doc.metadata.copy(),
        ))

        carry = full_text[-overlap_chars:] if overlap_chars else ""
        char_cursor += len(text)

    return chunks


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _split_paragraphs(text: str) -> list[str]:
    """Split on one or more blank lines."""
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]


def _merge_paragraphs(paragraphs: list[str], max_chars: int) -> list[str]:
    """
    Greedily merge short paragraphs up to max_chars.
    Split paragraphs that are already too long on sentence boundaries.
    """
    result = []
    current = ""

    for para in paragraphs:
        # paragraph itself is too long — split on sentences first
        if len(para) > max_chars:
            if current:
                result.append(current)
                current = ""
            result.extend(_split_on_sentences(para, max_chars))
            continue

        # fits alongside current accumulation
        if len(current) + len(para) + 1 <= max_chars:
            current = (current + " " + para).strip() if current else para
        else:
            if current:
                result.append(current)
            current = para

    if current:
        result.append(current)

    return result


def _split_on_sentences(text: str, max_chars: int) -> list[str]:
    """Split a long paragraph on sentence boundaries."""
    sentences = _SENTENCE_END.split(text)
    result = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip() if current else sent
        else:
            if current:
                result.append(current)
            if len(sent) > max_chars:
                for i in range(0, len(sent), max_chars):
                    result.append(sent[i:i + max_chars])
                current = ""
            else:
                current = sent

    if current:
        result.append(current)

    return result