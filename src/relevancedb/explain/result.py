from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from relevancedb.retrieve.fusion_ranker import RankedResult
    from relevancedb.retrieve.intent_classifier import Intent


@dataclass
class ResultChunk:
    """A single piece of evidence backing the answer."""

    text: str
    doc_path: str
    final_score: float
    semantic_score: float
    graph_score: float
    timeline_score: float
    source_heads: list[str]
    namespace: str
    metadata: dict = field(default_factory=dict)


class RelevanceResult:
    """
    The object returned by db.query().

    Attributes:
        question:  The original query.
        intent:    Detected intent (what/why/when/who/how).
        chunks:    Ranked list of ResultChunk objects.
        answer:    Best single chunk text (highest scored). For a full
                   synthesis you'd pass chunks into an LLM — that's a
                   future feature. For now, answer = top chunk.
    """

    def __init__(
        self,
        question: str,
        intent: Intent,
        ranked: list[RankedResult],
    ) -> None:
        self.question = question
        self.intent = intent
        self.chunks: list[ResultChunk] = [
            ResultChunk(
                text=r.text,
                doc_path=r.doc_path,
                final_score=r.final_score,
                semantic_score=r.semantic_score,
                graph_score=r.graph_score,
                timeline_score=r.timeline_score,
                source_heads=r.source_heads,
                namespace=r.namespace,
                metadata=r.metadata,
            )
            for r in ranked
        ]

    @property
    def answer(self) -> str:
        """Top-ranked chunk text. Empty string if no results."""
        if not self.chunks:
            return ""
        return self.chunks[0].text

    def explain(self) -> str:
        """
        Human-readable provenance trace for every result.

        Shows:
          - detected intent and head weights used
          - per-result: score breakdown, which heads contributed,
            namespace, source document
        """
        lines: list[str] = []
        lines.append(f"Query:  {self.question}")
        lines.append(f"Intent: {self.intent.value.upper()}")
        lines.append(f"Results: {len(self.chunks)}")
        lines.append("")

        if not self.chunks:
            lines.append("  No results found.")
            return "\n".join(lines)

        for i, chunk in enumerate(self.chunks, 1):
            lines.append(f"[{i}] score={chunk.final_score:.3f}  doc={chunk.doc_path}")
            lines.append(
                f"    heads : {' + '.join(chunk.source_heads)}"
            )
            lines.append(
                f"    scores: semantic={chunk.semantic_score:.2f}  "
                f"graph={chunk.graph_score:.2f}  "
                f"timeline={chunk.timeline_score:.2f}"
            )
            lines.append(f"    ns    : {chunk.namespace}")
            preview = chunk.text[:120].replace("\n", " ")
            lines.append(f"    text  : {preview!r}")
            lines.append("")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"RelevanceResult("
            f"intent={self.intent.value!r}, "
            f"chunks={len(self.chunks)}, "
            f"top_score={self.chunks[0].final_score if self.chunks else 0:.3f})"
        )