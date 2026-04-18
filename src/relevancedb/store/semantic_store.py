"""
semantic_store.py — vector storage head.

Uses LanceDB (embedded, zero server) with sentence-transformers for
local embeddings. No API key needed.

Key design: each entity sense gets its own LanceDB table (namespace).
"strawberry" the project and "strawberry" the fruit never share a table,
so cosine search never crosses context boundaries.

Deps: lancedb, sentence-transformers
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from relevancedb.ingest.chunker import Chunk


@dataclass
class SearchResult:
    text: str
    doc_path: str
    chunk_index: int
    score: float          
    namespace: str       
    metadata: dict


class SemanticStore:
    """
    Wraps LanceDB. One table per namespace (entity sense).

    Args:
        data_dir:    Where LanceDB stores its files on disk.
        embed_model: sentence-transformers model name.
                     Runs fully locally, no API key.
    """

    def __init__(
        self,
        data_dir: Path,
        embed_model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.embed_model_name = embed_model

        self._db = None        
        self._embedder = None 


    def add(self, chunks: list[Chunk], namespace: str = "default") -> None:
        """
        Embed and store chunks under a given namespace (sense-table).

        Args:
            chunks:    Chunks from chunker.py.
            namespace: Sense identifier. Defaults to "default".
                       Set by auto_disambiguator.py later.
        """
        if not chunks:
            return

        embedder = self._get_embedder()
        db = self._get_db()

        texts = [c.text for c in chunks]
        vectors = embedder.encode(texts, show_progress_bar=False).tolist()

        rows = [
            {
                "text": c.text,
                "vector": v,
                "doc_path": str(c.doc_path),
                "chunk_index": c.chunk_index,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "doc_format": c.doc_format,
                "namespace": namespace,
            }
            for c, v in zip(chunks, vectors)
        ]

        table_name = self._table_name(namespace)

        if table_name in db.table_names():
            tbl = db.open_table(table_name)
            tbl.add(rows)
        else:
            db.create_table(table_name, data=rows)

    def search(
        self,
        query: str,
        namespace: str = "default",
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Search within a specific namespace (sense-table).

        Args:
            query:     Natural language query.
            namespace: Which sense-table to search. Must match ingest namespace.
            top_k:     Number of results.

        Returns:
            List of SearchResult ordered by relevance score descending.
        """
        db = self._get_db()
        table_name = self._table_name(namespace)

        if table_name not in db.table_names():
            return []

        embedder = self._get_embedder()
        query_vector = embedder.encode(query, show_progress_bar=False).tolist()

        tbl = db.open_table(table_name)
        raw = (
            tbl.search(query_vector)
            .limit(top_k)
            .to_list()
        )

        return [
            SearchResult(
                text=r["text"],
                doc_path=r["doc_path"],
                chunk_index=r["chunk_index"],
                score=float(1 - r.get("_distance", 0)),  
                namespace=namespace,
                metadata={
                    "char_start": r["char_start"],
                    "char_end": r["char_end"],
                    "doc_format": r["doc_format"],
                },
            )
            for r in raw
        ]

    def namespaces(self) -> list[str]:
        """Return all namespaces (sense-tables) currently stored."""
        db = self._get_db()
        prefix = "sense__"
        return [
            t[len(prefix):]
            for t in db.table_names()
            if t.startswith(prefix)
        ]


    def _get_db(self):
        if self._db is None:
            import lancedb
            self._db = lancedb.connect(str(self.data_dir))
        return self._db

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder

    @staticmethod
    def _table_name(namespace: str) -> str:
        safe = namespace.lower().replace(" ", "_").replace("-", "_")
        return f"sense__{safe}"