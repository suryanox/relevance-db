"""
query_planner.py — decides which heads to query and collects raw results.

Takes a ClassifiedQuery from intent_classifier.py and hits the right
storage heads in the right order. Returns raw results from each head
— fusion_ranker.py then combines and ranks them.

This is the orchestrator for the retrieval side, just as pipeline.py
is the orchestrator for the ingest side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from relevancedb.retrieve.intent_classifier import ClassifiedQuery, Intent, IntentClassifier
from relevancedb.store.graph_store import GraphSearchResult, GraphStore
from relevancedb.store.semantic_store import SearchResult, SemanticStore
from relevancedb.store.timeline_store import DocVersion, TimelineStore


@dataclass
class RawResults:
    """Collected raw results from all heads before fusion."""
    question: str
    intent: Intent
    semantic: list[SearchResult] = field(default_factory=list)
    graph: list[GraphSearchResult] = field(default_factory=list)
    timeline: list[DocVersion] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"RawResults("
            f"intent={self.intent.value!r}, "
            f"semantic={len(self.semantic)}, "
            f"graph={len(self.graph)}, "
            f"timeline={len(self.timeline)})"
        )


class QueryPlanner:
    """
    Orchestrates retrieval across the three storage heads.

    Args:
        semantic:   SemanticStore instance.
        graph:      GraphStore instance.
        timeline:   TimelineStore instance.
        llm_model:  litellm model for intent classification.
        top_k:      Max results per head.
    """

    def __init__(
        self,
        semantic: SemanticStore,
        graph: GraphStore,
        timeline: TimelineStore,
        llm_model: str = "gpt-4o-mini",
        top_k: int = 5,
    ) -> None:
        self.semantic = semantic
        self.graph = graph
        self.timeline = timeline
        self.top_k = top_k
        self.classifier = IntentClassifier(llm_model=llm_model)

    def run(
        self,
        question: str,
        top_k: int | None = None,
        as_of: str | None = None,
    ) -> RawResults:
        """
        Classify intent then query the appropriate heads.

        Args:
            question: Natural-language query.
            top_k:    Override default top_k.
            as_of:    ISO date for point-in-time timeline queries.

        Returns:
            RawResults with results from each head that was queried.
        """
        k = top_k or self.top_k
        classified = self.classifier.classify(question)

        raw = RawResults(question=question, intent=classified.intent)

        for head in classified.heads:
            if head == "semantic":
                raw.semantic = self._query_semantic(question, k)
            elif head == "graph":
                raw.graph = self._query_graph(question, k)
            elif head == "timeline":
                raw.timeline = self._query_timeline(question, k, as_of)

        return raw

   
    def _query_semantic(self, question: str, k: int) -> list[SearchResult]:
        """
        Search all known namespaces and merge results.
        We search every namespace because the query itself may not
        tell us which sense was intended — the ranker handles that.
        """
        namespaces = self.semantic.namespaces()
        if not namespaces:
            namespaces = ["default"]

        all_results: list[SearchResult] = []
        for ns in namespaces:
            results = self.semantic.search(question, namespace=ns, top_k=k)
            all_results.extend(results)

        # sort by score descending, keep top k overall
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:k]

    def _query_graph(self, question: str, k: int) -> list[GraphSearchResult]:
        """
        Extract entity mentions from the question and traverse their
        neighbourhood in the graph.
        """
        entities = self.graph.all_entities()
        if not entities:
            return []

        q_lower = question.lower()
        mentioned = [e for e in entities if e.lower() in q_lower]

        if not mentioned:
            return []

        results: list[GraphSearchResult] = []
        seen: set[tuple] = set()

        for entity in mentioned[:3]:   # cap at 3 to avoid explosion
            neighbours = self.graph.neighbours(entity, max_hops=2)
            for n in neighbours:
                key = (n.entity, n.relation, n.neighbour)
                if key not in seen:
                    seen.add(key)
                    results.append(n)

        return results[:k]

    def _query_timeline(
        self, question: str, k: int, as_of: str | None
    ) -> list[DocVersion]:
        """
        Return recent document versions, optionally filtered to as_of date.
        """
        docs = self.graph.all_entities() 
        all_docs = self.timeline.all_docs()

        if not all_docs:
            return []

        versions: list[DocVersion] = []
        for doc_path in all_docs:
            if as_of:
                v = self.timeline.as_of(doc_path, as_of)
            else:
                v = self.timeline.latest(doc_path)
            if v:
                versions.append(v)

        versions.sort(key=lambda v: v.decay_weight, reverse=True)
        return versions[:k]