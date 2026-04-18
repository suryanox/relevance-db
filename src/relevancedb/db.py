"""
db.py — public API. The only file users need to know about.

    from relevancedb import RelevanceDB

    db = RelevanceDB()
    db.ingest(["policy.txt", "notes/"])
    result = db.query("who approved the retention policy?")
    print(result.answer)
    print(result.explain())
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

from relevancedb.explain.result import RelevanceResult
from relevancedb.ingest.pipeline import IngestPipeline, IngestSummary
from relevancedb.retrieve.fusion_ranker import FusionRanker
from relevancedb.retrieve.query_planner import QueryPlanner
from relevancedb.store.graph_store import GraphStore
from relevancedb.store.semantic_store import SemanticStore
from relevancedb.store.timeline_store import TimelineStore


class RelevanceDB:
    """
    Zero-config embedded retrieval database.

    Args:
        data_dir:    Where the three DBs live on disk.
                     Default: ./relevancedb_data
        llm_model:   Any litellm-compatible string.
                     Default: RELEVANCEDB_LLM_MODEL env var → "gpt-4o-mini"
        embed_model: sentence-transformers model name, runs locally.
                     Default: RELEVANCEDB_EMBED_MODEL env var → "BAAI/bge-small-en-v1.5"
        top_k:       Default number of results. Default: 5
        verbose:     Print progress during ingest.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "./relevancedb_data",
        llm_model: str | None = None,
        embed_model: str | None = None,
        top_k: int = 5,
        verbose: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.llm_model = (
            llm_model or os.getenv("RELEVANCEDB_LLM_MODEL", "gpt-4o-mini")
        )
        self.embed_model = (
            embed_model
            or os.getenv("RELEVANCEDB_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
        )
        self.top_k = top_k
        self.verbose = verbose

        self._semantic = SemanticStore(
            data_dir=self.data_dir / "semantic",
            embed_model=self.embed_model,
        )
        self._graph = GraphStore(
            data_dir=self.data_dir / "graph",
        )
        self._timeline = TimelineStore(
            data_dir=self.data_dir / "timeline",
        )

        # --- ingest pipeline ---
        self._pipeline = IngestPipeline(
            semantic=self._semantic,
            graph=self._graph,
            timeline=self._timeline,
            llm_model=self.llm_model,
            verbose=self.verbose,
        )

        # --- retrieval ---
        self._planner = QueryPlanner(
            semantic=self._semantic,
            graph=self._graph,
            timeline=self._timeline,
            llm_model=self.llm_model,
            top_k=self.top_k,
        )
        self._ranker = FusionRanker()

 
    def ingest(
        self,
        sources: Union[str, Path, list[Union[str, Path]]],
    ) -> IngestSummary:
        """
        Ingest documents into RelevanceDB.

        Accepts a file, directory, or list of either.
        Supported formats: .txt, .md  (PDF/DOCX coming soon)

        Under the hood:
          load → chunk → extract entities → disambiguate → store in all 3 heads

        Args:
            sources: Path(s) to files or directories.

        Returns:
            IngestSummary with document/chunk/entity counts.

        Example:
            db.ingest("report.txt")
            db.ingest(["folder/", "extra.md"])
        """
        if not isinstance(sources, list):
            sources = [sources]
        return self._pipeline.run([Path(s) for s in sources])

    def query(
        self,
        question: str,
        top_k: int | None = None,
        as_of: str | None = None,
    ) -> RelevanceResult:
        """
        Query RelevanceDB with a natural-language question.

        Under the hood:
          classify intent → plan which heads to hit → retrieve → fuse + rank

        Args:
            question: Natural-language question.
            top_k:    Number of results. Overrides instance default.
            as_of:    ISO date string for point-in-time queries e.g. "2023-06-01"

        Returns:
            RelevanceResult with .answer, .chunks, .explain()

        Example:
            result = db.query("why was the policy changed?")
            print(result.answer)
            print(result.explain())
        """
        k = top_k or self.top_k
        raw = self._planner.run(question=question, top_k=k, as_of=as_of)
        ranked = self._ranker.rank(raw, top_k=k)
        return RelevanceResult(
            question=question,
            intent=raw.intent,
            ranked=ranked,
        )

    def __repr__(self) -> str:
        return (
            f"RelevanceDB("
            f"data_dir={str(self.data_dir)!r}, "
            f"llm={self.llm_model!r}, "
            f"embed={self.embed_model!r})"
        )