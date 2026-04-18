"""
pipeline.py — ingest orchestrator.

Ties together: loader → chunker → entity_extractor → all three stores.

Stages:
  1. Load    — parse file into Document
  2. Chunk   — split into indexable Chunks
  3. Extract — entities + relations via LLM
  4. Store   — write to all three heads
     a. timeline_store: record version + decay weight
     b. semantic_store: embed + store chunks (namespace="default" for now)
     c. graph_store:    entities + relations from extractor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from relevancedb.ingest.chunker import chunk
from relevancedb.ingest.entity_extractor import EntityExtractor
from relevancedb.ingest.loader import Document, load, load_dir
from relevancedb.store.graph_store import GraphStore
from relevancedb.store.semantic_store import SemanticStore
from relevancedb.store.timeline_store import TimelineStore


@dataclass
class IngestSummary:
    documents: int = 0
    chunks: int = 0
    entities: int = 0
    relations: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"IngestSummary("
            f"documents={self.documents}, "
            f"chunks={self.chunks}, "
            f"entities={self.entities}, "
            f"relations={self.relations}, "
            f"errors={len(self.errors)})"
        )


class IngestPipeline:
    """
    Orchestrates the full ingest flow.

    Args:
        semantic:   SemanticStore instance.
        graph:      GraphStore instance.
        timeline:   TimelineStore instance.
        llm_model:  litellm model string for entity extraction.
        chunk_size: Max chars per chunk.
        overlap:    Overlap chars between chunks.
        verbose:    Print progress per document.
    """

    def __init__(
        self,
        semantic: SemanticStore,
        graph: GraphStore,
        timeline: TimelineStore,
        llm_model: str = "gpt-4o-mini",
        chunk_size: int = 512,
        overlap: int = 64,
        verbose: bool = False,
    ) -> None:
        self.semantic = semantic
        self.graph = graph
        self.timeline = timeline
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.verbose = verbose
        self.extractor = EntityExtractor(llm_model=llm_model)

    def run(self, sources: list[Path]) -> IngestSummary:
        summary = IngestSummary()
        docs = self._collect_docs(sources, summary)

        for doc in docs:
            try:
                self._ingest_doc(doc, summary)
            except Exception as exc:
                msg = f"{doc.path.name}: {exc}"
                summary.errors.append(msg)
                if self.verbose:
                    print(f"[relevancedb] error: {msg}")

        if self.verbose:
            print(f"[relevancedb] done — {summary}")

        return summary

    def _collect_docs(
        self, sources: list[Path], summary: IngestSummary
    ) -> list[Document]:
        docs = []
        for source in sources:
            if source.is_dir():
                found = load_dir(source)
                docs.extend(found)
                if self.verbose:
                    print(f"[relevancedb] dir {source.name}: {len(found)} files")
            elif source.is_file():
                try:
                    docs.append(load(source))
                except ValueError:
                    summary.skipped += 1
            else:
                summary.errors.append(f"path not found: {source}")
        return docs

    def _ingest_doc(self, doc: Document, summary: IngestSummary) -> None:
        if self.verbose:
            print(f"[relevancedb] ingesting {doc.path.name} ({doc.format})")

        # 1. timeline
        self.timeline.add(
            doc_path=str(doc.path),
            char_count=len(doc.text),
            fmt=doc.format,
        )

        # 2. chunk
        chunks = chunk(doc, max_chars=self.chunk_size, overlap_chars=self.overlap)

        # 3. semantic — namespace="default" until disambiguator is wired
        self.semantic.add(chunks, namespace="default")

        # 4. entity extraction → graph
        extraction = self.extractor.extract(doc)
        self.graph.add_entities(extraction.entities)
        self.graph.add_relations(extraction.relations)

        summary.documents += 1
        summary.chunks += len(chunks)
        summary.entities += len(extraction.entities)
        summary.relations += len(extraction.relations)