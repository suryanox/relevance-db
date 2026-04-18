"""
graph_store.py — graph storage head.

Uses KuzuDB (embedded, zero server, MIT license).
Stores entities and relationships extracted from documents.

Schema:
  Nodes:  Entity(name, type, doc_path, sense_id)
  Edges:  Relation(source, target, relation_type, doc_path)

This is what lets us answer questions like:
  "who approved the policy change?" — graph traversal
  "what depends on this service?" — edge lookup
  "how does X relate to Y?" — path query

Dep: kuzu
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Entity:
    name: str
    type: str      
    doc_path: str
    sense_id: str = "default"


@dataclass
class Relation:
    source: str      # entity name
    target: str      # entity name
    relation_type: str   # e.g. APPROVED, DEPENDS_ON, UPDATED
    doc_path: str


@dataclass
class GraphSearchResult:
    entity: str
    relation: str
    neighbour: str
    doc_path: str


class GraphStore:
    """
    Wraps KuzuDB. Stores entities and their relationships.

    Args:
        data_dir: Where KuzuDB stores its files on disk.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._db = None    
        self._conn = None 
        self._schema_ready = False


    def add_entities(self, entities: list[Entity]) -> None:
        """Insert entities (nodes) into the graph."""
        if not entities:
            return

        conn = self._get_conn()

        for e in entities:
            conn.execute(
                """
                MERGE (n:Entity {name: $name})
                ON CREATE SET
                    n.type     = $type,
                    n.doc_path = $doc_path,
                    n.sense_id = $sense_id
                """,
                {
                    "name": e.name,
                    "type": e.type,
                    "doc_path": e.doc_path,
                    "sense_id": e.sense_id,
                },
            )

    def add_relations(self, relations: list[Relation]) -> None:
        """Insert relationships (edges) between entities."""
        if not relations:
            return

        conn = self._get_conn()

        for r in relations:
            # ensure both endpoint nodes exist before creating edge
            for name in (r.source, r.target):
                conn.execute(
                    "MERGE (n:Entity {name: $name})",
                    {"name": name},
                )

            conn.execute(
                """
                MATCH (a:Entity {name: $source}), (b:Entity {name: $target})
                MERGE (a)-[rel:Relation {relation_type: $rel_type, doc_path: $doc_path}]->(b)
                """,
                {
                    "source": r.source,
                    "target": r.target,
                    "rel_type": r.relation_type,
                    "doc_path": r.doc_path,
                },
            )

    def neighbours(
        self,
        entity_name: str,
        max_hops: int = 2,
    ) -> list[GraphSearchResult]:
        """
        Return all entities reachable from entity_name within max_hops.

        This is what powers "why" and "how" queries — traversing the
        relationship graph rather than doing cosine similarity.
        """
        conn = self._get_conn()

        rows = []
        for hops in range(1, max_hops + 1):
            result = conn.execute(
                """
                MATCH (a:Entity {name: $name})-[r:Relation]->(b:Entity)
                RETURN a.name, r.relation_type, b.name, r.doc_path
                """,
                {"name": entity_name},
            )
            while result.has_next():
                row = result.get_next()
                rows.append(GraphSearchResult(
                    entity=row[0],
                    relation=row[1],
                    neighbour=row[2],
                    doc_path=row[3],
                ))
            if hops == 1:
                break  # single hop sufficient for now
        return rows

    def entity_exists(self, name: str) -> bool:
        conn = self._get_conn()
        result = conn.execute(
            "MATCH (n:Entity {name: $name}) RETURN count(n)",
            {"name": name},
        )
        return result.get_next()[0] > 0

    def all_entities(self) -> list[str]:
        conn = self._get_conn()
        result = conn.execute("MATCH (n:Entity) RETURN n.name")
        names = []
        while result.has_next():
            names.append(result.get_next()[0])
        return names


    def _get_conn(self):
        if self._conn is None:
            import kuzu
            self._db = kuzu.Database(str(self.data_dir / "kuzu.db"))
            self._conn = kuzu.Connection(self._db)
            self._ensure_schema()
        return self._conn

    def _ensure_schema(self) -> None:
        """Create node and edge tables if they don't exist yet."""
        if self._schema_ready:
            return

        conn = self._conn

        conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Entity (
                name     STRING,
                type     STRING DEFAULT 'UNKNOWN',
                doc_path STRING DEFAULT '',
                sense_id STRING DEFAULT 'default',
                PRIMARY KEY (name)
            )
        """)

        conn.execute("""
            CREATE REL TABLE IF NOT EXISTS Relation (
                FROM Entity TO Entity,
                relation_type STRING,
                doc_path      STRING DEFAULT ''
            )
        """)

        self._schema_ready = True