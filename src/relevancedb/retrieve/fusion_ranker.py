"""
fusion_ranker.py — combines results from all three heads into one ranked list.

This is the second novel piece on the retrieval side.

The problem with naive fusion:
  Most hybrid search just averages scores or does reciprocal rank fusion (RRF).
  RRF treats every head equally regardless of query intent.
  A "why" query should weight graph results higher.
  A "when" query should weight timeline decay heavily.
  A "what" query should trust semantic score most.

Our approach:
  Intent-weighted scoring. Each head's contribution is scaled by a
  weight matrix derived from the query intent. The final score per
  result is a weighted sum of normalised scores from each head.

  final_score = (w_sem * semantic_score)
              + (w_graph * graph_score)
              + (w_time * timeline_score)

  Weights come from INTENT_WEIGHTS below — tunable without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from relevancedb.retrieve.intent_classifier import Intent
from relevancedb.retrieve.query_planner import RawResults
from relevancedb.store.graph_store import GraphSearchResult
from relevancedb.store.semantic_store import SearchResult


# Intent → (semantic_weight, graph_weight, timeline_weight)
# Must sum to 1.0 per row.
INTENT_WEIGHTS: dict[Intent, tuple[float, float, float]] = {
    Intent.WHAT:    (0.70, 0.20, 0.10),
    Intent.WHY:     (0.25, 0.65, 0.10),
    Intent.WHEN:    (0.20, 0.10, 0.70),
    Intent.WHO:     (0.20, 0.70, 0.10),
    Intent.HOW:     (0.55, 0.35, 0.10),
    Intent.UNKNOWN: (0.45, 0.35, 0.20),
}


@dataclass
class RankedResult:
    """A single fused result with a final relevance score and provenance."""

    text: str
    doc_path: str
    final_score: float
    semantic_score: float
    graph_score: float
    timeline_score: float
    namespace: str
    source_heads: list[str]    # which heads contributed to this result
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ") if self.text else ""
        return (
            f"RankedResult("
            f"score={self.final_score:.3f}, "
            f"heads={self.source_heads}, "
            f"preview={preview!r})"
        )


class FusionRanker:
    """
    Combines raw results from all three heads into a ranked list.

    Weights are determined by query intent — not a fixed average.
    """

    def rank(self, raw: RawResults, top_k: int = 5) -> list[RankedResult]:
        """
        Fuse and rank results from all heads.

        Args:
            raw:   RawResults from query_planner.py.
            top_k: How many results to return.

        Returns:
            List of RankedResult sorted by final_score descending.
        """
        w_sem, w_graph, w_time = INTENT_WEIGHTS.get(
            raw.intent, INTENT_WEIGHTS[Intent.UNKNOWN]
        )

        # normalise scores within each head to [0, 1]
        sem_scores   = self._normalise([r.score for r in raw.semantic])
        graph_scores = self._graph_to_scores(raw.graph, len(raw.graph))
        time_scores  = self._normalise(
            [v.decay_weight for v in raw.timeline]
        )

        # build a candidate pool keyed by doc_path
        # each doc accumulates contributions from all heads
        pool: dict[str, RankedResult] = {}

        # --- semantic contributions ---
        for result, norm_score in zip(raw.semantic, sem_scores):
            key = result.doc_path
            if key not in pool:
                pool[key] = self._empty_result(result.doc_path, result.namespace)
                pool[key].text = result.text
                pool[key].metadata = result.metadata

            pool[key].semantic_score = max(pool[key].semantic_score, norm_score)
            if "semantic" not in pool[key].source_heads:
                pool[key].source_heads.append("semantic")

        # --- graph contributions ---
        for result, norm_score in zip(raw.graph, graph_scores):
            # graph results reference doc_path — find or create pool entry
            key = result.doc_path
            if key not in pool:
                pool[key] = self._empty_result(result.doc_path, "graph")
                pool[key].text = (
                    f"{result.entity} {result.relation} {result.neighbour}"
                )

            pool[key].graph_score = max(pool[key].graph_score, norm_score)
            if "graph" not in pool[key].source_heads:
                pool[key].source_heads.append("graph")

        # --- timeline contributions ---
        for version, norm_score in zip(raw.timeline, time_scores):
            key = version.doc_path
            if key not in pool:
                pool[key] = self._empty_result(version.doc_path, "timeline")
                pool[key].text = f"Document version {version.version} ingested {version.ingested_at[:10]}"

            pool[key].timeline_score = max(pool[key].timeline_score, norm_score)
            if "timeline" not in pool[key].source_heads:
                pool[key].source_heads.append("timeline")

        # --- compute final weighted score ---
        for result in pool.values():
            result.final_score = round(
                w_sem   * result.semantic_score
                + w_graph * result.graph_score
                + w_time  * result.timeline_score,
                4,
            )

        ranked = sorted(pool.values(), key=lambda r: r.final_score, reverse=True)
        return ranked[:top_k]


    @staticmethod
    def _empty_result(doc_path: str, namespace: str) -> RankedResult:
        return RankedResult(
            text="",
            doc_path=doc_path,
            final_score=0.0,
            semantic_score=0.0,
            graph_score=0.0,
            timeline_score=0.0,
            namespace=namespace,
            source_heads=[],
        )

    @staticmethod
    def _normalise(scores: list[float]) -> list[float]:
        """Min-max normalise a list of scores to [0, 1]."""
        if not scores:
            return []
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return [1.0] * len(scores)
        return [(s - lo) / (hi - lo) for s in scores]

    @staticmethod
    def _graph_to_scores(
        results: list[GraphSearchResult], count: int
    ) -> list[float]:
        """
        Graph results have no inherent score — assign by position.
        First result (closest neighbour) gets score 1.0, last gets ~0.
        """
        if not results:
            return []
        return [1.0 - (i / max(count, 1)) for i in range(len(results))]