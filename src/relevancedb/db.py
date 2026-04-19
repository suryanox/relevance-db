from __future__ import annotations
from pathlib import Path
from typing import Union
from relevancedb.config import ModelConfig, default_data_dir
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
        llm_model: litellm-compatible model string. Required unless
                   RELEVANCEDB_LLM_MODEL env var is set.

                   OpenAI:    "gpt-4o-mini", "gpt-4o"
                   Anthropic: "claude-3-haiku-20240307"
                   Ollama:    "ollama/mistral"
                   Groq:      "groq/llama3-8b-8192"

        top_k:    Default number of results per query. Default: 5
        verbose:  Print progress during ingest.

    Data is stored in:
        Linux/Mac : ~/.local/share/relevancedb
        Windows   : %APPDATA%/relevancedb

    Raises:
        ValueError: If llm_model is not provided and
                    RELEVANCEDB_LLM_MODEL env var is not set.
    """

    def __init__(
        self,
        llm_model: str | None = None,
        top_k: int = 5,
        verbose: bool = False,
    ) -> None:
        self.models = ModelConfig(llm_model=llm_model)
        self.data_dir = default_data_dir()
        self.top_k = top_k
        self.verbose = verbose

        self._semantic = SemanticStore(
            data_dir=self.data_dir / "semantic",
            embed_model=self.models.embed_model,
        )
        self._graph = GraphStore(data_dir=self.data_dir / "graph")
        self._timeline = TimelineStore(data_dir=self.data_dir / "timeline")

        self._pipeline = IngestPipeline(
            semantic=self._semantic,
            graph=self._graph,
            timeline=self._timeline,
            llm_model=self.models.llm_model,
            verbose=self.verbose,
        )
        self._planner = QueryPlanner(
            semantic=self._semantic,
            graph=self._graph,
            timeline=self._timeline,
            llm_model=self.models.llm_model,
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
        Supported formats: .txt, .md

        Args:
            sources: Path(s) to files or directories.

        Returns:
            IngestSummary with document/chunk/entity counts.
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

        Args:
            question: Natural-language question.
            top_k:    Number of results. Overrides instance default.
            as_of:    ISO date for point-in-time queries e.g. "2023-06-01"

        Returns:
            RelevanceResult with .answer, .chunks, .explain()
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
            f"llm={self.models.llm_model!r}, "
            f"data_dir={str(self.data_dir)!r})"
        )